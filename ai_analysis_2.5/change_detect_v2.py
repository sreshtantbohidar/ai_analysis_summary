"""
High-performance change-detection script.
Requires: elasticsearch>=8.0, requests, orjson, tqdm (optional)
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional
import orjson  # pip install orjson
import requests
from elasticsearch import Elasticsearch
from requests.exceptions import RequestException
from tqdm import tqdm   # pip install tqdm
import html

###############################################################################
# CONFIG
###############################################################################
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "192.168.1.125")
ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", 9200))
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "30oIsFcjJa8Zao+iq5*e")
INDEX_NAME = os.getenv("INDEX_NAME", "fatboy_data")
LLM_IP = os.getenv("LLM_IP", "192.168.1.125")
LLM_PORT = int(os.getenv("LLM_PORT", 11434))
IFC_LLM_PORT = int(os.getenv("IFC_LLM_PORT", 3005))
IFC_LLM_TOKEN = os.getenv("IFC_LLM_TOKEN", "E281H6R-AZC4EX4-GSPC3SX-1CRYYF2")
ANYTHINGLLM_WORKSPACE_SLUG = os.getenv("ANYTHINGLLM_WORKSPACE_SLUG", "ai-summary")

TARGET_FIELDS = [
    "equipment_name",
    "infra_name",
    "enemy_formation_name",
    "equipment_type",
]
TIMESTAMP_FIELD = "@timestamp"
HISTORY_LOOKBACK_DAYS = 365
TIME_NEAR_THRESHOLD_DAYS = 30

es = Elasticsearch(
    [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": "https"}],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
    request_timeout=60,
)

# Import constants from parent directory (overrides env-based config)
import importlib.util

_constants_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'constants.py'))
if os.path.exists(_constants_path):
    _spec = importlib.util.spec_from_file_location("constants", _constants_path)
    _constants_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_constants_mod)
    # Override env-based config with constants module values
    if hasattr(_constants_mod, 'IFC_LLM_PORT'):
        IFC_LLM_PORT = _constants_mod.IFC_LLM_PORT
    if hasattr(_constants_mod, 'IFC_LLM_TOKEN'):
        IFC_LLM_TOKEN = _constants_mod.IFC_LLM_TOKEN
    if hasattr(_constants_mod, 'LLM_IP'):
        LLM_IP = _constants_mod.LLM_IP
    if hasattr(_constants_mod, 'ANYTHINGLLM_WORKSPACE_SLUG'):
        ANYTHINGLLM_WORKSPACE_SLUG = _constants_mod.ANYTHINGLLM_WORKSPACE_SLUG

# AnythingLLM workspace model cache & defaults
_current_anythingllm_model = None
DEFAULT_CHAT_PROVIDER = "ollama"

###############################################################################
# UTILS
###############################################################################
_SOURCE_FILTER = [
    TIMESTAMP_FIELD,
    "location_name",
    "opposite_sector",
    "deployment_status",
    "description",
] + TARGET_FIELDS


def parse_timestamp(ts: str) -> datetime:
    """Return *naive* UTC datetime from ISO-8601 string."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    # Strip tzinfo → naive
    return dt.replace(tzinfo=None)


def hash_group(rec: Dict) -> str:
    key = (
        rec.get("target_field", ""),
        rec.get("target_value", ""),
        rec.get("location_name", ""),
        rec.get("opposite_sector", ""),
        rec.get("deployment_status", ""),
    )
    return hashlib.blake2b("|".join(str(v) for v in key).encode(), digest_size=8).hexdigest()


###############################################################################
# ELASTIC HELPERS
###############################################################################
def fetch_pit(query: Dict, index: str = INDEX_NAME) -> List[Dict]:
    """PIT + search_after (no scroll)."""
    pit = es.open_point_in_time(index=index, keep_alive="2m")["id"]
    try:
        body = {
            "size": 10000,
            "query": query["query"],
            "sort": [{TIMESTAMP_FIELD: "asc"}],
            "_source": _SOURCE_FILTER,
        }
        hits = []
        search_after = None
        while True:
            body["pit"] = {"id": pit, "keep_alive": "2m"}
            if search_after:
                body["search_after"] = search_after
            resp = es.search(body=body)
            batch = resp["hits"]["hits"]
            if not batch:
                break
            hits.extend(batch)
            search_after = batch[-1]["sort"]
        return [h["_source"] for h in hits]
    finally:
        es.close_point_in_time(body={"id": pit})


def extract_targets(records: Iterable[Dict]) -> Iterable[Dict]:
    """Yield one dict per target field occurrence (generator)."""
    for r in records:
        for field in TARGET_FIELDS:
            if field in r:
                yield {
                    "target_field": field,
                    "target_value": r[field],
                    "location_name": r.get("location_name"),
                    "opposite_sector": r.get("opposite_sector"),
                    "deployment_status": r.get("deployment_status"),
                    "description": r.get("description", ""),
                    "timestamp": r[TIMESTAMP_FIELD],
                }


def fetch_latest_historical(start_date: str, names: Dict[str, List[str]]) -> List[Dict]:
    """
    Pull the last 365 days, return RAW documents so the caller can explode
    them with extract_targets() later.
    """
    must = [
        {
            "range": {
                TIMESTAMP_FIELD: {
                    "gte": (
                        datetime.fromisoformat(start_date) - timedelta(days=HISTORY_LOOKBACK_DAYS)
                    ).strftime("%Y-%m-%d"),
                    "lt": start_date,
                }
            }
        }
    ]
    should = []
    if names["equipment"]:
        should.append({"terms": {"equipment_name.keyword": names["equipment"]}})
    if names["enemy"]:
        should.append({"terms": {"enemy_formation_name.keyword": names["enemy"]}})
    if names["infra"]:
        should.append({"terms": {"infra_name.keyword": names["infra"]}})

    if should:
        must.append({"bool": {"should": should, "minimum_should_match": 1}})

    # Return raw documents – caller will explode them
    return fetch_pit({"query": {"bool": {"must": must}}})
###############################################################################
# CHANGE DETECTION LOGIC
###############################################################################
def assign_priority(cur: Dict, hist: Dict) -> str:
    c_date = parse_timestamp(cur["timestamp"])
    h_date = parse_timestamp(hist["timestamp"])
    same_loc = cur.get("location_name") == hist.get("location_name") and cur.get("location_name")
    same_sector = cur.get("opposite_sector") == hist.get("opposite_sector") and cur.get("opposite_sector")
    same_deploy = cur.get("deployment_status") == hist.get("deployment_status") and cur.get("deployment_status")

    if same_loc:
        return "P1" if abs((c_date - h_date).days) <= TIME_NEAR_THRESHOLD_DAYS else "P2"
    if same_sector:
        return "P1" if abs((c_date - h_date).days) <= TIME_NEAR_THRESHOLD_DAYS else "P2"
    if same_deploy:
        return "P3"
    return "P4"

def changes_to_html(changes: list, new_activities: list) -> str:
    """
    Build one HTML doc with:
      1. A 3-column table per change (attributes | history | current)
         – mismatched values get red background.
      2. A separate table for new activities.
    """
    # ---------- changes ----------
    change_tables = []
    for idx, change in enumerate(changes, 1):
        h = change.get("history", {})
        c = change.get("current", {})
        attrs = sorted({*h.keys(), *c.keys()})

        rows = []
        for k in attrs:
            h_val, c_val = h.get(k, ""), c.get(k, "")
            same = str(h_val) == str(c_val)
            h_bg = "" if same else ' style="background-color:#ffcccc"'
            c_bg = "" if same else ' style="background-color:#ffcccc"'
            rows.append(
                f"<tr>"
                f"<td>{html.escape(str(k))}</td>"
                f"<td{h_bg}>{html.escape(str(h_val))}</td>"
                f"<td{c_bg}>{html.escape(str(c_val))}</td>"
                f"</tr>"
            )

        bottom = (
            f'<tr class="summary"><td colspan="3"><strong>CHANGE:</strong><br>'
            f'{html.escape(str(change.get("CHANGE","")))}</td></tr>'
            f'<tr class="summary"><td colspan="3"><strong>Priority:</strong> '
            f'{html.escape(str(change.get("priority","")))}</td></tr>'
        )

        change_tables.append(
            f'<h3>Change #{idx}</h3>'
            f'<table border="1" cellpadding="6" cellspacing="0">'
            f'<thead><tr><th>Attribute</th><th>History</th><th>Current</th></tr></thead>'
            f'<tbody>{"".join(rows)}{bottom}</tbody></table><br>'
        )

    # ---------- new activities ----------
    new_rows = []
    for rec in new_activities:
        new_rows.append(
            f"<tr>"
            f"<td>{html.escape(str(rec.get('target_field','')))}</td>"
            f"<td>{html.escape(str(rec.get('target_value','')))}</td>"
            f"<td>{html.escape(str(rec.get('location_name','')))}</td>"
            f"<td>{html.escape(str(rec.get('description','')))}</td>"
            f"<td>{html.escape(str(rec.get('timestamp','')))}</td>"
            f"</tr>"
        )

    new_table = (
        f'<h3>New Activities</h3>'
        f'<table border="1" cellpadding="6" cellspacing="0">'
        f'<thead><tr>'
        f'<th>target_field</th><th>target_value</th>'
        f'<th>location_name</th><th>description</th><th>timestamp</th>'
        f'</tr></thead><tbody>{"".join(new_rows)}</tbody></table>'
    ) if new_activities else '<h3>New Activities</h3><p>None.</p>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Change Detection Report</title>
<style>
  body{{font-family:Arial,sans-serif;margin:20px}}
  table{{width:100%;border-collapse:collapse;margin-bottom:30px}}
  th,td{{text-align:left;vertical-align:top;padding:4px 6px}}
  tr.summary{{background-color:#f2f2f2}}
</style>
</head>
<body>
{"".join(change_tables)}
{new_table}
</body>
</html>"""

def update_anythingllm_workspace_model(model_name: str, chat_provider: str = DEFAULT_CHAT_PROVIDER) -> bool:
    """
    Update the AnythingLLM workspace to use the specified model before making a chat request.
    Uses a module-level cache to avoid redundant API calls when the model hasn't changed.
    Returns True if the workspace was updated (or already using the correct model).
    """
    global _current_anythingllm_model

    # Skip if already set to this model
    if _current_anythingllm_model == model_name:
        return True

    update_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/update"

    payload = {
        "chatProvider": chat_provider,
        "chatModel": model_name
    }

    headers = {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                update_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if r.status_code in (200, 201, 204):
                print(f"[INFO] AnythingLLM workspace model updated to: {model_name}")
                _current_anythingllm_model = model_name
                return True
            else:
                print(f"[ERROR] AnythingLLM model update status {r.status_code}: {r.text}")

        except requests.exceptions.Timeout:
            print(f"[ERROR] Model update timeout (attempt {attempt+1}/{MAX_RETRIES})")

        except Exception as e:
            print(f"[ERROR] Model update failed: {str(e)}")

        time.sleep(2 ** attempt)

    print(f"[WARN] Could not update AnythingLLM model to {model_name}, proceeding with current model")
    return False


def get_description_change_summary(entity_name, entity_value, prev_desc, curr_desc, llm_model_name=None, max_retries=5, timeout=30):
    # Update AnythingLLM workspace model if specified
    if llm_model_name:
        update_anythingllm_workspace_model(llm_model_name)

    prompt = f"""
    You are a military analyst reviewing updates on a particular asset or entity named "{entity_name} for asset or entity value {entity_value}".
    Below are two descriptions of the same entity at different points in time.

    Previous Description:
    {prev_desc}

    Current Description:
    {curr_desc}

    Analyze and summarize the key changes or differences between the two. Be concise, highlight only meaningful updates, removals, or additions. Ignore minor wording changes unless they imply a significant change.
    """
    anythingllm_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
    payload = {
        "message": prompt,
        "mode": "chat"
    }
    headers = {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                anythingllm_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("textResponse")
        except RequestException as e:
            if attempt == max_retries:
                return None          # <-- caller will skip the key
            time.sleep(1.5 ** attempt)  # exponential back-off
    return None


###############################################################################
# MAIN PIPELINE
###############################################################################
def run_change_detection(user_query: Dict, start_date: str, llm_model_name: Optional[str] = None) -> str:
    print("===== Fast Change Detection =====")
    t0 = time.time()

    # 1. Current data
    current_raw = fetch_pit(user_query)
    current_targets = list(extract_targets(current_raw))
    print(f"[INFO] Current records: {len(current_targets)}")

    # 2. De-duplicate current on the fly
    seen = set()
    current_unique = []
    for rec in current_targets:
        h = hash_group(rec)
        if h not in seen:
            seen.add(h)
            current_unique.append(rec)
    print(f"[INFO] Unique current: {len(current_unique)}")

    # 3. Collect names for historical filter
    names = {
        "equipment": list({r["target_value"] for r in current_unique if r["target_field"] == "equipment_name"}),
        "enemy": list({r["target_value"] for r in current_unique if r["target_field"] == "enemy_formation_name"}),
        "infra": list({r["target_value"] for r in current_unique if r["target_field"] == "infra_name"}),
    }

    # 4. Historical – raw docs
    historical_raw = fetch_latest_historical(start_date, names)
    historical_targets = list(extract_targets(historical_raw))   # <— explode here
    print(f"[INFO] Historical unique: {len(historical_targets)}")

    # 5. Build lookup (field, value) -> newest exploded record
    latest_hist = {}
    for h in historical_targets:
        key = (h["target_field"], h["target_value"])
        ts = parse_timestamp(h["timestamp"])
        if key not in latest_hist or parse_timestamp(latest_hist[key]["timestamp"]) < ts:
            latest_hist[key] = h

    # 6. Detect changes & new activities
    changes, new_activities = [], []
    for cur in current_unique:
        key = (cur["target_field"], cur["target_value"])
        if key in latest_hist:
            changes.append({"current": cur, "history": latest_hist[key], "priority": assign_priority(cur, latest_hist[key])})
        else:
            new_activities.append(cur)


    for change in tqdm(changes, desc="Summarising changes"):
        summary = get_description_change_summary(
            change["current"]["target_field"],
            change["current"]["target_value"],
            change["history"].get("description", ""),
            change["current"].get("description", ""),
            llm_model_name=llm_model_name,
        )
        try:
            change["CHANGE"] = orjson.loads(summary) if summary else ""
        except Exception:
            change["CHANGE"] = summary or ""

    print(f"[DONE] {len(changes)} changes, {len(new_activities)} new activities in {time.time() - t0:.1f}s")
    changes_html = changes_to_html(changes, new_activities)
    return changes_html

def run_change_detection_previous_period(user_query: Dict, start_date: str, end_date: str, years_back: int = 1, llm_model_name: Optional[str] = None) -> str:

    print("===== Previous-Year Period Change Detection =====")
    t0 = time.time()

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    prev_start = start_dt.replace(year=start_dt.year - years_back)
    prev_end = end_dt.replace(year=end_dt.year - years_back)

    print(f"[INFO] Current window: {start_dt} → {end_dt}")
    print(f"[INFO] Previous window ({years_back} year(s) ago): {prev_start} → {prev_end}")

    # ---------- Current Data ----------
    current_raw = fetch_pit(user_query)
    current_targets = list(extract_targets(current_raw))
    print(f"[INFO] Current records: {len(current_targets)}")

    # ---------- Build Previous-Year Query ----------
    prev_query = json.loads(json.dumps(user_query))  # deep copy

    def update_range(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "range":
                    for field, val in v.items():
                        if "gte" in val:
                            val["gte"] = prev_start.isoformat()
                        if "lt" in val:
                            val["lt"] = prev_end.isoformat()
                else:
                    update_range(v)
        elif isinstance(obj, list):
            for item in obj:
                update_range(item)

    update_range(prev_query)

    # ---------- Previous-Year Data ----------
    previous_raw = fetch_pit(prev_query)
    previous_targets = list(extract_targets(previous_raw))
    print(f"[INFO] Previous records: {len(previous_targets)}")

    # ---------- Build Lookup ----------
    prev_lookup = {}
    for rec in previous_targets:
        key = (rec["target_field"], rec["target_value"])
        ts = parse_timestamp(rec["timestamp"])

        if key not in prev_lookup or parse_timestamp(prev_lookup[key]["timestamp"]) < ts:
            prev_lookup[key] = rec

    # ---------- Compare ----------
    changes = []
    new_activities = []

    for cur in current_targets:

        key = (cur["target_field"], cur["target_value"])

        if key in prev_lookup:

            hist = prev_lookup[key]

            changes.append({
                "current": cur,
                "history": hist,
                "priority": assign_priority(cur, hist)
            })

        else:
            new_activities.append(cur)

    # ---------- LLM Change Explanation ----------
    for change in tqdm(changes, desc="Summarising changes"):

        summary = get_description_change_summary(
            change["current"]["target_field"],
            change["current"]["target_value"],
            change["history"].get("description", ""),
            change["current"].get("description", ""),
            llm_model_name=llm_model_name,
        )

        try:
            change["CHANGE"] = orjson.loads(summary) if summary else ""
        except Exception:
            change["CHANGE"] = summary or ""

    print(f"[DONE] {len(changes)} changes, {len(new_activities)} new activities in {time.time() - t0:.1f}s")

    return changes_to_html(changes, new_activities)


###############################################################################
# MULTI-YEAR COMPARISON (predictive)
###############################################################################

def get_llm_prediction(multi_year_summary: str, model_name: str = "gemma2:9b-instruct-q8_0", max_retries=3, timeout=120) -> str:
    """Send multi-year entity trend data to LLM and return a future-outcome prediction."""
    # Update AnythingLLM workspace model before making the request
    update_anythingllm_workspace_model(model_name)

    prompt = f"""You are a military intelligence analyst reviewing multi-year trend data to predict future outcomes.

Below is a detailed summary of tracked military entities (equipment, formations, infrastructure) across multiple years, showing how each entity has appeared, changed location, changed status, or disappeared over time.

{multi_year_summary}

Based on this multi-year trend data, provide a forward-looking prediction. Structure your response with these sections:

1. **Key Trends** — What patterns are emerging? Which entity types are increasing or decreasing in activity?
2. **Significant Changes** — Highlight the most notable entity relocations or status shifts.
3. **Emerging Threats** — What new entities or reappearing entities are concerning?
4. **Future Outlook** — Predict what is likely to happen in the next period. Consider entities that may reappear, new deployments, or activity shifts.

Be specific, cite entity names, and keep your analysis concise but thorough. Format your response as HTML paragraphs inside <div> tags."""
    anythingllm_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
    payload = {
        "message": prompt,
        "mode": "chat"
    }
    headers = {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                anythingllm_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("textResponse", "")
        except RequestException as e:
            if attempt == max_retries:
                return "<p>Prediction unavailable.</p>"
            time.sleep(1.5 ** attempt)
    return "<p>Prediction unavailable.</p>"


def multi_year_to_html(windows: list, categories: dict,
                       entity_timeline: dict,
                       prediction: str) -> str:
    """Generate a comprehensive HTML report for multi-year comparison."""

    sorted_windows = sorted(windows, key=lambda w: w["offset"])

    # ---------- summary stat cards ----------
    stat_cards = ""
    stat_config = [
        ("new", "#e3f2fd", "\U0001f195 New", len(categories["new"])),
        ("evolved", "#fff3e0", "\U0001f504 Evolved", len(categories["evolved"])),
        ("recurring", "#fce4ec", "\U0001f501 Recurring", len(categories["recurring"])),
        ("persistent", "#e8f5e9", "\u2705 Persistent", len(categories["persistent"])),
        ("disappeared", "#f5f5f5", "\u274c Disappeared", len(categories["disappeared"])),
    ]
    for cls, bg, label, count in stat_config:
        stat_cards += f"""
        <div class="stat-card {cls}" style="background:{bg}">
            <span class="stat-num">{count}</span>
            <span class="stat-label">{label}</span>
        </div>"""

    # ---------- entity tables per category ----------
    cat_config = [
        ("new", "\U0001f195 New Entities",
         "First appeared in the current year \u2014 no history in any previous year window."),
        ("evolved", "\U0001f504 Evolved Entities",
         "Present across all tracked years but attributes (location, status) changed over time."),
        ("recurring", "\U0001f501 Recurring Entities",
         "Were present in some years, disappeared for one or more years, then reappeared."),
        ("persistent", "\u2705 Persistent Entities",
         "Present across all tracked years with stable attributes."),
        ("disappeared", "\u274c Disappeared Entities",
         "Were present in previous years but not found in the current year window."),
    ]

    all_sections = []
    for cat_key, cat_title, cat_desc in cat_config:
        entities = categories[cat_key]
        if not entities:
            continue

        rows = []
        for key in entities:
            field, value = key
            timeline = entity_timeline.get(key, {})

            cells = []
            for w in sorted_windows:
                off = w["offset"]
                if off in timeline:
                    rec = timeline[off]
                    loc = html.escape(str(rec.get("location_name", "")))
                    status = html.escape(str(rec.get("deployment_status", "")))
                    desc_snippet = html.escape(str(rec.get("description", "")[:60]))
                    cell_content = f"<b>{loc}</b>"
                    if status:
                        cell_content += f"<br><small>{status}</small>"
                    if desc_snippet:
                        cell_content += f"<br><small class='desc-snip'>{desc_snippet}</small>"
                    cells.append(f'<td>{cell_content}</td>')
                else:
                    cells.append('<td class="absent">\u2014</td>')

            rows.append(
                f"<tr>"
                f"<td><strong>{html.escape(field)}</strong></td>"
                f"<td><code>{html.escape(str(value))}</code></td>"
                f"{''.join(cells)}"
                f"</tr>"
            )

        if rows:
            header_cells = "".join(
                f'<th>{w["label"].split("(")[0].strip()}</th>' for w in sorted_windows
            )
            all_sections.append(f"""
            <div class="category-section">
                <div class="category-header">
                    <h3>{cat_title}</h3>
                    <span class="badge">{len(entities)}</span>
                </div>
                <p class="cat-desc">{cat_desc}</p>
                <div class="table-wrap">
                    <table class="entity-table">
                        <thead>
                            <tr>
                                <th>Field</th>
                                <th>Value</th>
                                {header_cells}
                            </tr>
                        </thead>
                        <tbody>{"".join(rows)}</tbody>
                    </table>
                </div>
            </div>""")

    # ---------- prediction section ----------
    pred_html = ""
    if prediction:
        pred_html = f"""
        <div class="prediction-section">
            <h2>\U0001f916 LLM Future Prediction</h2>
            <div class="prediction-content">{prediction}</div>
        </div>"""

    # ---------- assemble ----------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    period_label = html.escape(windows[0]["label"]) if windows else "N/A"
    total_entities = len(entity_timeline)
    num_windows = len(windows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Year Comparison Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
          background:#f5f7fa; color:#222; padding:30px; }}
  h1 {{ font-size:1.8em; color:#1a237e; margin-bottom:4px; }}
  .subtitle {{ color:#666; font-size:0.95em; margin-bottom:24px; }}
  .stats-grid {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:30px; }}
  .stat-card {{ flex:1; min-width:120px; padding:18px 14px; border-radius:10px;
                text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .stat-card .stat-num {{ display:block; font-size:2em; font-weight:700; color:#1a237e; }}
  .stat-card .stat-label {{ font-size:0.85em; color:#555; }}
  .category-section {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.08);
                       margin-bottom:24px; overflow:hidden; }}
  .category-header {{ display:flex; align-items:center; gap:10px; padding:16px 20px;
                       border-bottom:1px solid #eee; }}
  .category-header h3 {{ font-size:1.15em; color:#1a237e; }}
  .badge {{ background:#1a237e; color:#fff; font-size:0.8em; font-weight:600;
             padding:2px 10px; border-radius:12px; }}
  .cat-desc {{ padding:8px 20px 0; font-size:0.88em; color:#777; }}
  .table-wrap {{ overflow-x:auto; padding:0 0 4px 0; }}
  .entity-table {{ width:100%; border-collapse:collapse; font-size:0.88em; }}
  .entity-table th {{ background:#f5f7fa; color:#333; font-weight:600; text-align:left;
                       padding:10px 12px; border-bottom:2px solid #dee2e6; white-space:nowrap; }}
  .entity-table td {{ padding:8px 12px; border-bottom:1px solid #eee; vertical-align:top; }}
  .entity-table tr:hover {{ background:#f8f9ff; }}
  .entity-table td.absent {{ text-align:center; color:#ccc; }}
  .desc-snip {{ color:#999; font-size:0.82em; display:block; max-width:160px;
                 overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .prediction-section {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,0.08);
                          padding:20px 24px; margin-top:8px; }}
  .prediction-section h2 {{ font-size:1.3em; color:#1a237e; margin-bottom:12px; }}
  .prediction-content {{ line-height:1.7; color:#333; }}
  .prediction-content p {{ margin:8px 0; }}
  footer {{ margin-top:30px; font-size:0.82em; color:#999; text-align:center; }}
</style>
</head>
<body>
  <h1>Multi-Year Comparison Report</h1>
  <p class="subtitle">
    Period: {period_label} &nbsp;|&nbsp;
    Windows: {num_windows} &nbsp;|&nbsp;
    Total Entities: {total_entities} &nbsp;|&nbsp;
    Generated: {now}
  </p>
  <div class="stats-grid">{stat_cards}</div>
  {"".join(all_sections)}
  {pred_html}
  <footer>Report generated by AI Analysis System \u2014 Multi-Year Comparison Engine</footer>
</body>
</html>"""


def run_multi_year_comparison(user_query: Dict, start_date: str, end_date: str, years_back: int = 1, llm_model_name: str = "gemma2:9b-instruct-q8_0") -> str:
    """
    Collect data from current + N previous yearly windows (same calendar period).
    Track each entity across *all* years, categorise patterns (new / persistent /
    evolved / recurring / disappeared), and use an LLM to predict future outcomes.

    Args:
        user_query: Elasticsearch query dict
        start_date: Start date of the current window (YYYY-MM-DD)
        end_date: End date of the current window (YYYY-MM-DD)
        years_back: Number of years to look back
        llm_model_name: LLM model name to use for prediction (fetched from DB)
    """
    print(f"===== Multi-Year Comparison ({years_back} year(s) back, predictive) =====")
    t0 = time.time()

    start_dt = datetime.fromisoformat(start_date)
    end_dt   = datetime.fromisoformat(end_date)

    # ---------------------------------------------------------------
    # 1. Fetch data from all yearly windows (oldest to newest)
    # ---------------------------------------------------------------
    windows = []
    for i in range(years_back, -1, -1):
        if i == 0:
            raw = fetch_pit(user_query)
            label = f"Current ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})"
        else:
            shifted_start = start_dt.replace(year=start_dt.year - i)
            shifted_end   = end_dt.replace(year=end_dt.year - i)

            shifted_query = json.loads(json.dumps(user_query))

            def _update_range(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "range":
                            for field, val in v.items():
                                if "gte" in val:
                                    val["gte"] = shifted_start.isoformat()
                                if "lt" in val:
                                    val["lt"] = shifted_end.isoformat()
                        else:
                            _update_range(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _update_range(item)

            _update_range(shifted_query)
            raw = fetch_pit(shifted_query)
            label = f"{i} yr ago ({shifted_start.strftime('%Y-%m-%d')} to {shifted_end.strftime('%Y-%m-%d')})"

        targets = list(extract_targets(raw))
        # Deduplicate within this window
        seen = set()
        unique = []
        for rec in targets:
            h = hash_group(rec)
            if h not in seen:
                seen.add(h)
                unique.append(rec)

        windows.append({"offset": -i, "label": label, "records": unique})
        print(f"[INFO] Window {i}: {label} \u2192 {len(unique)} unique records")

    # ---------------------------------------------------------------
    # 2. Build entity timeline: (target_field, target_value) -> { offset -> latest_record }
    # ---------------------------------------------------------------
    entity_timeline: dict = {}
    for w in windows:
        for rec in w["records"]:
            key = (rec["target_field"], rec["target_value"])
            if key not in entity_timeline:
                entity_timeline[key] = {}
            ts = parse_timestamp(rec["timestamp"])
            if w["offset"] not in entity_timeline[key] or \
               parse_timestamp(entity_timeline[key][w["offset"]]["timestamp"]) < ts:
                entity_timeline[key][w["offset"]] = rec

    # ---------------------------------------------------------------
    # 3. Categorise every entity
    # ---------------------------------------------------------------
    expected_offsets = set(range(-years_back, 1))

    categories: dict = {
        "new": [], "persistent": [], "evolved": [],
        "recurring": [], "disappeared": [],
    }

    for key, timeline in entity_timeline.items():
        present       = set(timeline.keys())
        in_current    = 0 in present
        has_history   = any(o < 0 for o in present)

        if in_current and not has_history:
            categories["new"].append(key)
            continue

        if in_current:
            missing  = expected_offsets - present - {0}
            has_gap  = bool(missing)

            if has_gap:
                categories["recurring"].append(key)
                continue

            # Check for attribute changes across years
            cur_rec = timeline[0]
            changed = False
            for prev_o in sorted(timeline.keys(), reverse=True):
                if prev_o >= 0:
                    continue
                prev_rec = timeline[prev_o]
                if (cur_rec.get("location_name") != prev_rec.get("location_name") or
                    cur_rec.get("deployment_status") != prev_rec.get("deployment_status")):
                    changed = True
                    break

            categories["evolved" if changed else "persistent"].append(key)
        else:
            categories["disappeared"].append(key)

    # ---------------------------------------------------------------
    # 4. Build a concise summary for LLM prediction
    # ---------------------------------------------------------------
    prediction = ""
    if entity_timeline:
        summary_parts = [
            f"Multi-year analysis over {years_back + 1} windows (current through {years_back} year(s) back).",
            f"Period: {start_date} to {end_date}",
            f"Total entities tracked: {len(entity_timeline)}",
            f"  - New (first appeared this year): {len(categories['new'])}",
            f"  - Persistent (unchanged across years): {len(categories['persistent'])}",
            f"  - Evolved (location/status changed): {len(categories['evolved'])}",
            f"  - Recurring (gap in presence): {len(categories['recurring'])}",
            f"  - Disappeared (not found this year): {len(categories['disappeared'])}",
            "",
            "--- Year-by-year entity details (sample of notable entities) ---",
        ]

        for cat_name in ("evolved", "recurring", "new", "disappeared", "persistent"):
            entities = categories[cat_name]
            if not entities:
                continue
            summary_parts.append(f"\n[{cat_name.upper()}] ({len(entities)} entities):")
            for key in entities[:15]:
                field, value = key
                timeline_recs = entity_timeline[key]
                history_parts = []
                for off in sorted(timeline_recs.keys(), reverse=True):
                    rec = timeline_recs[off]
                    desc = rec.get("description", "")
                    snippet = (desc[:80] + "...") if len(desc) > 80 else desc
                    history_parts.append(
                        f"    Year offset {off}: "
                        f"loc={rec.get('location_name','')} "
                        f"status={rec.get('deployment_status','')} "
                        f"desc=\"{snippet}\""
                    )
                summary_parts.append(f"  Entity: {field} = \"{value}\"")
                summary_parts.extend(history_parts)

        llm_summary_text = "\n".join(summary_parts)
        prediction = get_llm_prediction(llm_summary_text, model_name=llm_model_name)

    # ---------------------------------------------------------------
    # 5. Generate HTML report
    # ---------------------------------------------------------------
    html = multi_year_to_html(windows, categories, entity_timeline, prediction)

    print(f"[DONE] Multi-year comparison: {len(entity_timeline)} entities, "
          f"{len(categories['new'])} new, {len(categories['disappeared'])} disappeared, "
          f"{len(categories['evolved'])} changed in {time.time() - t0:.1f}s")

    return html


###############################################################################
# CLI
###############################################################################
# if __name__ == "__main__":
#     payload = {
#         "size": 10000,
#         "query": {
#             "bool": {
#                 "must": [
#                     {"exists": {"field": "training_type"}},
#                     {"terms": {"daily_activity_type_id": [383]}},
#                     {"term": {"form_type.keyword": "training"}},
#                     {"exists": {"field": "location_name"}},
#                     {
#                         "range": {
#                             "activity_date": {
#                                 "gte": "2024-05-07T00:00:00",
#                                 "lt": "2025-05-15T00:00:00",
#                             }
#                         }
#                     },
#                 ],
#                 "must_not": [{"term": {"form_status": 5}}],
#             }
#         },
#     }
#     html_doc = run_change_detection(payload, start_date="2025-05-07")
#     debug_file = "debug_report.html"
#     with open(debug_file, "w", encoding="utf-8") as f:
#         f.write(html_doc)



    # print(f"[DONE] Results saved to {out_file}")
    # print(json.dumps({"changes": changes, "new": new_activities}, indent=2, default=str))