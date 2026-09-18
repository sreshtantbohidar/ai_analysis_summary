import argparse
import copy
import json
# import multiprocessing
import os
import re
import sys
import time
import tempfile
import base64
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import psycopg2
import requests
from elasticsearch import Elasticsearch
from tqdm import tqdm

from test_sitrep_trends import sitrep_main
from test_infra_trends import infra_main
from test_sam_trends import trends_sam_main
from test_airinspect_trends import trends_airfield_main
from test_training_trends import training_main
from test_force_disposition_trends import force_disposition_main
from change_detect_v2 import run_change_detection, run_multi_year_comparison
import importlib.util

import html as _html

# ──────────────────────────────────────────────
# Constants from parent directory constants.py
# ──────────────────────────────────────────────
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'constants.py'))
spec = importlib.util.spec_from_file_location("constants", file_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

PG_DB_NAME = constants.PG_DB_NAME
DJANGO_HOST = constants.DJANGO_HOST
CUREENT_INDEX_NAME = constants.CUREENT_INDEX_NAME
ELASTIC_HOST = constants.ELASTIC_HOST
ELASTIC_PORT = constants.ELASTIC_PORT
PG_USER = constants.PG_USER
PG_PORT = constants.PG_PORT
PG_PASSWORD = constants.PG_PASSWORD
LLM_IP = constants.LLM_IP
LLM_PORT = constants.LLM_PORT
LLM_MODEL = constants.LLM_MODEL
IFC_LLM_PORT = constants.IFC_LLM_PORT
IFC_LLM_TOKEN = constants.IFC_LLM_TOKEN
ANYTHINGLLM_WORKSPACE_SLUG = constants.ANYTHINGLLM_WORKSPACE_SLUG
ELASTIC_CLIENT_SCHEME = constants.ELASTIC_CLIENT_SCHEME
ELASTICSEARCH_USERNAME = constants.ELASTICSEARCH_USERNAME
ELASTICSEARCH_PASSWORD = constants.ELASTICSEARCH_PASSWORD

OLLAMA_TIMEOUT = 900
MAX_RETRIES = 3

# Fallback model if the configured model returns empty responses
FALLBACK_LLM_MODEL = "gemma2:9b-instruct-q8_0"

_current_anythingllm_model = None
DEFAULT_CHAT_PROVIDER = "ollama"

# Large-dataset scaling defaults. These are deliberately conservative because
# multiple simultaneous LLM generations can reduce throughput or increase VRAM
# pressure on smaller GPUs. Operators with larger GPUs can raise the worker
# count with --chunk_workers or AI_SUMMARY_CHUNK_WORKERS.
DEFAULT_CHUNK_WORKERS = 2
MAX_CHUNK_WORKERS = 8
DEFAULT_RAG_TOP_N = 48
DEFAULT_RAG_TOP_N_MAX = 64
DEFAULT_CHUNK_TARGET_FRACTION = 0.50  # 50% of the 75% threshold (~37.5% of declared context)
MAX_DIRECT_OLLAMA_OUTPUT_TOKENS = 6144  # bounded per-chunk generation ceiling
MAX_OVERALL_OLLAMA_OUTPUT_TOKENS = 10000  # larger ceiling for detailed top-level analysis when context permits
MIN_CHUNK_DOCUMENT_TOKENS = 1800
MAX_CHUNK_DOCUMENT_TOKENS = 5000

AI_ANALYSIS_QUALITY_GUARDRAILS = """
ANALYSIS QUALITY RULES:
- Treat the user's analysis request as the primary instruction. Do not replace it with a different task.
- Use ONLY the supplied records/analyses as factual evidence. Do not add outside facts or unsupported assumptions.
- Consider all supplied records/analyses before deciding what is important; do not let the first or longest item dominate.
- Prioritize high-information findings: important events, activities, entities, locations, dates, quantities, changes, relationships, risks and exceptions.
- Do not spend report space on routine record-by-record commentary unless the user explicitly asks for record-level detail.
- Distinguish repeated patterns from one-off findings and preserve material unusual, contradictory or negative findings.
- Combine overlapping observations into stronger judgments instead of repeating the same conclusion in different wording.
- Be explicit about uncertainty when evidence is insufficient; do not fill gaps with plausible-sounding claims.
- ONLY for forward-looking requests, distinguish OBSERVED evidence, INFERRED interpretation and FORECAST. OBSERVED means directly supported by the supplied records; INFERRED means a reasoned interpretation; FORECAST means a statement about a period after the latest supplied evidence.
- For material forecasts, give evidence/basis, time horizon only when supported, and confidence (High/Moderate/Low) when meaningful.
- Never invent a forecast time horizon, specific future date, escalation scenario, new capability, actor, or causal relationship. If the time horizon cannot be supported, state that it is not determinable from the supplied records.
- Never label a future statement as OBSERVED. Reconcile historical dates in the evidence with the latest evidence date before forecasting. Do not treat a historical plan, target, or expected completion date as a future event if that date has already passed.
- Consolidate overlapping forecasts into fewer, stronger judgments rather than repeating the same trend in different wording.
- For dataset-scale synthesis, preserve meaningful coverage of every supplied chunk/group. Do not allow a dominant theme in one chunk to erase distinct findings from other chunks.
- During coverage-ledger reduction, preserve provenance and material findings from every source chunk; never collapse distinct chunks into one generic theme when they contain materially different evidence.
- Do not forecast escalation, new capabilities or specific events unless the supplied evidence supports that possibility.
- For non-forward-looking requests, do not add forecast-specific sections or language.
"""

# ──────────────────────────────────────────────
# Elasticsearch client
# ──────────────────────────────────────────────
# The elasticsearch Python client's auth handling differs between major
# versions:
#   - 7.x: basic_auth= kwarg is fine on its own, BUT it has a bug (combined
#     with urllib3 >=2) where the Authorization header is silently dropped
#     whenever the password contains characters urllib3.make_headers treats
#     as reserved URL chars ('+', '*', etc.). Our password has both '+' and
#     '*', so we ALSO pass an explicit Authorization header.
#   - 8.x: if BOTH basic_auth= AND a custom Authorization header are set,
#     the constructor raises ValueError ("Can't set 'Authorization' HTTP
#     header with other authentication options"). So we pass ONLY the
#     explicit header in 8.x.
#
# Strategy: try the 7.x construction first (basic_auth + header); if the
# client refuses with ValueError, fall back to the 8.x form (header only).
# This keeps the file working on both client majors without runtime version
# inspection.
import base64 as _es_b64
_es_auth_header = "Basic " + _es_b64.b64encode(
    f"{ELASTICSEARCH_USERNAME}:{ELASTICSEARCH_PASSWORD}".encode("utf-8")
).decode("ascii")
_es_client_attempts = [
    # (kwargs, log_label)
    (
        {
            "basic_auth": (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
            "headers": {"Authorization": _es_auth_header},
            "verify_certs": False,
            "ssl_show_warn": False,
        },
        "basic_auth + explicit Authorization header (elasticsearch 7.x)",
    ),
    (
        {
            "headers": {"Authorization": _es_auth_header},
            "verify_certs": False,
            "ssl_show_warn": False,
        },
        "explicit Authorization header only (elasticsearch 8.x)",
    ),
]
es = None
for _es_kwargs, _es_label in _es_client_attempts:
    try:
        es = Elasticsearch(
            [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": ELASTIC_CLIENT_SCHEME}],
            **_es_kwargs,
        )
        print(f"[INFO] ES client built using {_es_label}")
        break
    except ValueError as _es_err:
        print(f"[INFO] ES client construction rejected ({_es_err}); trying next fallback")
        continue
if es is None:
    raise RuntimeError("Failed to construct Elasticsearch client with any known auth pattern")

# ──────────────────────────────────────────────
# PostgreSQL helper
# ──────────────────────────────────────────────
def postgres_connection():
    return psycopg2.connect(
        database=PG_DB_NAME,
        host=DJANGO_HOST,
        user=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
    )

# ──────────────────────────────────────────────
# HTML / Report helpers (reused from v3.0.0)
# ──────────────────────────────────────────────
def _safe_int(value) -> Optional[int]:
    """Parse an env-var or CLI value as int; return None if missing/invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strip_wrappers(html: str) -> str:
    html = re.sub(r'</?html[^>]*>', '', html, flags=re.I)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.I | re.S)
    html = re.sub(r'</?body[^>]*>', '', html, flags=re.I)
    return html.strip()

def html_to_plain_text(html: str) -> str:
    """Extract readable text from an HTML LLM response."""
    if not html:
        return ""

    # Convert common block-level HTML tags to line breaks
    text = re.sub(
        r'</?(?:h[1-6]|p|div|br|li|ul|ol|tr|table|thead|tbody|tfoot|section|article|blockquote|pre)[^>]*>',
        '\n',
        html,
        flags=re.I
    )

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode HTML entities
    
    text = _html.unescape(text)

    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


def _extract_citation_numbers(text: str, valid_min: int = 1, valid_max: Optional[int] = None) -> list:
    """Extract explicit parenthesized citation-number lists from LLM HTML/text.

    Only parenthesized comma-separated integer lists are treated as citations,
    e.g. ``(3, 45, 50)``. Numbers outside the supplied record range are ignored.
    """
    if not text:
        return []
    found = []
    pattern = re.compile(r"\((\s*\d+(?:\s*,\s*\d+)+\s*)\)")
    for match in pattern.finditer(html_to_plain_text(text)):
        for part in match.group(1).split(','):
            try:
                value = int(part.strip())
            except ValueError:
                continue
            if value < valid_min:
                continue
            if valid_max is not None and value > valid_max:
                continue
            found.append(value)
    return sorted(set(found))


def build_tabbed_html(html_strings: List[str], doc_table_html: Optional[str] = None) -> str:
    labels = ["AI Analysis", "AI Trends", "AI Change Detection"]
    sections = []
    for i, (label, content) in enumerate(zip(labels, html_strings)):
        if content:
            section_content = _strip_wrappers(content) if content else "<p>No data available</p>"

            sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-{i}">
                    <h2>{label}</h2>
                </div>
                <div class="section-content">
                    {section_content}
                </div>
            </section>''')

    # ⭐ NEW (v3.1.1.1): Trailing "Record Source Details" citation table.
    # Only appended when the caller actually supplies a non-empty table, so
    # back-compat is preserved for any caller that omits the argument.
    if doc_table_html:
        sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-citations">
                    <h2>Record Source Details</h2>
                </div>
                <div class="section-content">
                    {doc_table_html}
                </div>
            </section>''')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Reports – Analysis | Trends | Change</title>
  <style>
    body {{
        margin: 0;
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
        background: #f3f5f7;
        color: #212529;
        line-height: 1.6;
    }}
    header {{
        background: #004085;
        color: #fff;
        padding: 20px 40px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .report-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }}
    .section-header {{
        background: #fff;
        padding: 15px 25px;
        border-left: 4px solid #004085;
        margin: 20px 0 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .section-header h2 {{
        margin: 0;
        color: #004085;
        font-size: 1.5em;
    }}
    .section-content {{
        background: #fff;
        padding: 25px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }}
    .section-content h2 {{
        font-size: 1.3em;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-top: 0;
    }}
    .section-content p {{
        margin: 10px 0;
    }}
    footer {{
        padding: 15px 40px;
        background: #e9ecef;
        font-size: .8em;
        color: #6c757d;
        text-align: center;
        margin-top: 20px;
    }}
    @media (max-width: 768px) {{
        .report-container {{ padding: 10px; }}
        .section-content {{ padding: 15px; }}
        header {{ padding: 15px 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>AI Intelligence Report</h1>
    <p>Generated on: {now}</p>
  </header>
  <div class="report-container">
    {''.join(sections)}
  </div>
  <footer>
    Report generated by AI Analysis System
  </footer>
</body>
</html>"""


# ──────────────────────────────────────────────
# ⭐ NEW (v3.1.1.1): Record Source Details helper
# ──────────────────────────────────────────────
def build_doc_id_activity_date_table(hits, query_total=None, date_range=None, query_note=None) -> str:
    """
    Build an HTML table summarising each Elasticsearch hit by its citation
    metadata: ES doc _id, source_type, file_name (basename of file_http_path),
    ingester_name, and activity_date.

    Empty / missing fields render as em-dash placeholders; missing _source is
    tolerated (the row still shows the _id). When hits is empty, a friendly
    "No records available." message is returned.

    Optional context shown beneath the table:
      - query_total: the unfiltered ES `total.value` (i.e. how many docs the
        query matched before dedup). When supplied alongside len(hits), a
        "Showing N deduplicated of M raw matches" line is added so analysts
        see when the table is a sampled view.
      - date_range: tuple (start, end) of the query's @timestamp filter. When
        supplied, a "Date range: start → end" line is added.

    Used as the trailing "Record Source Details" section of the AI report so
    analysts can trace every claim back to its originating document.

    NOTE: this function is intentionally self-contained (does not call any
    private helper) so it can be unit-tested in isolation by extracting the
    function body with a regex like `^def NAME(...)`.
    """
    if not hits:
        return "<p>No records available.</p>"

    def _esc(value) -> str:
        """Inline HTML escape. Returns '—' for None/empty to make blank cells
        easier to scan in the rendered table."""
        if value is None:
            return "—"
        s = str(value).strip()
        if not s:
            return "—"
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    def _basename(path_like) -> str:
        """Inline basename extractor that handles both / and \\ separators."""
        if not path_like:
            return ""
        s = str(path_like).strip().rstrip("/").rstrip("\\")
        if not s:
            return ""
        parts = [p for p in re.split(r"[\\/]", s) if p]
        return parts[-1] if parts else ""

    headers = ["Citation No.", "Source Type", "File Name", "Activity Date"]
    rows = []
    for hit in hits:
        es_id = hit.get("_id", "") if isinstance(hit, dict) else ""
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict):
            source = {}

        source_type = source.get("source_type", "")
        file_name = _basename(source.get("file_http_path", ""))
        # Some deployments store this under the misspelled "injester_name"
        # (matches the typo used elsewhere in ai_analysis_report_generator.py);
        # accept either spelling.
        ingester_name = source.get("ingester_name") or source.get("injester_name") or ""
        activity_date = source.get("activity_date", "")

        rows.append([
            _esc(len(rows) + 1),
            _esc(source_type),
            _esc(file_name),
            _esc(activity_date),
        ])

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "\n".join(body_rows)

    # ⭐ NEW (v3.1.1.2): build a context footer line(s) describing what the
    # table covers (dedup ratio + date range). These help analysts spot when
    # the cited set is a deduplicated subset of a larger query.
    footer_lines = [f"Total records cited: {len(rows)}"]
    if query_note:
        footer_lines.append(str(query_note))
    elif query_total is not None and query_total != len(rows):
        footer_lines.append(
            f"Showing {len(rows)} deduplicated of {query_total} raw ES matches"
        )
    if date_range and (date_range[0] or date_range[1]):
        start = date_range[0] or "?"
        end = date_range[1] or "?"
        footer_lines.append(f"Query date range: {start} → {end}")
    footer_html = "".join(
        f"<p style=\"margin:4px 0 0 0;color:#6c757d;font-size:12px;\">{line}</p>"
        for line in footer_lines
    )

    return (
        "<table class=\"citation-table\" "
        "style=\"border-collapse:collapse;width:100%;font-size:14px;\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        + footer_html
    )


# ──────────────────────────────────────────────
# Profile Analysis helpers (reused from v3.0.0)
# ──────────────────────────────────────────────
def create_llm_prompt_for_profile(complete_summary_dict_list, search_form_type, person_names, organization_names):
    summary_question = {
        'profile analysis': """
        "%s"

        "%s"
        """
    }
    activity_question = summary_question[search_form_type]
    query_string = ""
    if person_names and organization_names:
        query_string = "Create the summary of person" + ", ".join(person_names) + " and organization " + ", ".join(
            organization_names) + " in mentioned list of dictionary and create relation and summary of each other with refrence of date and file paths."
    elif person_names:
        query_string = 'Deep dive and create a analysis of " "' + ", ".join(
            person_names) + '" from below json data description, the analysis should have all the personal details, involvement in any activity and his relations with other people, mention the relationship names with other people and create html table also mention the reference of file name along with your analysis. The output should be in html format. Title should be h2 with font-size:20px and paragraph should be in p tag with font-size:15px'
    elif organization_names:
        query_string = "Create the summary of organization in above list of dictionary and create a summary of " + ", ".join(
            organization_names)
    if query_string:
        question_temp = activity_question % (str(query_string), str(complete_summary_dict_list))
        return question_temp

# ──────────────────────────────────────────────
# AnythingLLM Workspace Model Update (reused from v3.0.0)
# ──────────────────────────────────────────────
def update_anythingllm_workspace_model(model_name: str, chat_provider: str = DEFAULT_CHAT_PROVIDER,
                                       workspace_slug: Optional[str] = None,
                                       force: bool = False) -> bool:
    global _current_anythingllm_model
    slug = workspace_slug or ANYTHINGLLM_WORKSPACE_SLUG
    # The global cache only applies to the main workspace
    # (ANYTHINGLLM_WORKSPACE_SLUG). Per-chunk workspaces are created fresh with
    # no chat model assigned, so they always need their own model set (force).
    if not force and workspace_slug is None and _current_anythingllm_model == model_name:
        return True

    update_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/update"
    payload = {"chatProvider": chat_provider, "chatModel": model_name}
    headers = {"Authorization": f"Bearer {IFC_LLM_TOKEN}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(update_url, json=payload, headers=headers, timeout=30)
            if r.status_code in (200, 201, 204):
                if workspace_slug is None:
                    _current_anythingllm_model = model_name
                    print(f"[INFO] AnythingLLM workspace model updated to: {model_name}")
                else:
                    print(f"[INFO] AnythingLLM workspace '{workspace_slug}' model set to: {model_name}")
                return True
            else:
                print(f"[ERROR] AnythingLLM model update status {r.status_code} for '{slug}': {r.text}")
        except requests.exceptions.Timeout:
            print(f"[ERROR] Model update timeout (attempt {attempt+1}/{MAX_RETRIES}) for '{slug}'")
        except Exception as e:
            print(f"[ERROR] Model update failed for '{slug}': {str(e)}")
        time.sleep(2 ** attempt)

    print(f"[WARN] Could not update AnythingLLM model to {model_name} on '{slug}', "
          f"proceeding with current model")
    return False


# ──────────────────────────────────────────────
# AnythingLLM helpers (Document Upload + RAG Query Mode)
# ──────────────────────────────────────────────

def get_anythingllm_headers() -> dict:
    """Return standard JSON headers for AnythingLLM API calls."""
    return {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }


def get_anythingllm_upload_headers() -> dict:
    """Return headers for multipart upload (no Content-Type — requests sets it)."""
    return {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}"
    }


def upload_document_to_anythingllm(markdown_content: str, filename: str) -> Optional[str]:
    """
    Upload a markdown document to AnythingLLM.

    POST /api/v1/document/upload (multipart/form-data)

    Returns the document location path (for use with update-embeddings adds)
    or None on failure.
    """
    upload_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/document/upload"
    headers = get_anythingllm_upload_headers()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            tmp.write(markdown_content)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            files = {'file': (filename, f, 'text/markdown')}
            for attempt in range(MAX_RETRIES):
                try:
                    r = requests.post(upload_url, headers=headers, files=files, timeout=60)
                    if r.status_code in (200, 201):
                        result = r.json()
                        # Extract the location path from the response
                        # Response: {"success": true, "documents": [{"id": "...", "location": "custom-documents/...json", ...}]}
                        if isinstance(result, dict) and 'documents' in result and result['documents']:
                            doc = result['documents'][0]
                            doc_id = doc.get('id')
                            location = doc.get('location', '')
                            print(f"[INFO] Document uploaded: {filename} (ID: {doc_id}, location: {location})")
                            return location
                        print(f"[WARN] Document uploaded but could not extract location from response")
                        return None
                    else:
                        print(f"[ERROR] Document upload status {r.status_code}: {r.text}")
                except requests.exceptions.Timeout:
                    print(f"[ERROR] Upload timeout (attempt {attempt+1}/{MAX_RETRIES})")
                except Exception as e:
                    print(f"[ERROR] Upload failed: {str(e)}")
                time.sleep(2 ** attempt)

        print(f"[WARN] Failed to upload document: {filename}")
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def add_document_to_workspace(document_location: str) -> bool:
    """
    Link an uploaded document to the workspace by adding it to the workspace's
    vector embeddings.

    POST /api/v1/workspace/{slug}/update-embeddings
    with payload: {"adds": ["custom-documents/xxx.json"], "deletes": []}

    The 'adds' array takes file paths (from the upload response's 'location' field),
    not document IDs. These are paths relative to the storage directory.

    Args:
        document_location: The 'location' field from the upload response.

    Returns:
        True if successful, False otherwise.
    """
    embed_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/update-embeddings"
    headers = get_anythingllm_headers()
    payload = {"adds": [document_location], "deletes": []}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(embed_url, json=payload, headers=headers, timeout=300)
            if r.status_code in (200, 201, 204):
                print(f"[INFO] Document '{document_location}' added to workspace embeddings")
                return True
            else:
                print(f"[ERROR] Update embeddings status {r.status_code}: {r.text}")
        except requests.exceptions.Timeout:
            print(f"[ERROR] Embedding update timeout (attempt {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            print(f"[ERROR] Embedding update failed: {str(e)}")
        time.sleep(2 ** attempt)

    print(f"[WARN] Could not add document '{document_location}' to workspace")
    return False


def query_anythingllm_rag(query: str, llm_model: str) -> str:
    """
    Send a query in RAG mode to AnythingLLM.

    POST /api/v1/workspace/{slug}/chat with mode='query'

    The response is generated based on documents embedded in the workspace.
    First updates the workspace model, then sends the query.
    """
    update_anythingllm_workspace_model(llm_model)

    chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
    headers = get_anythingllm_headers()

    payload = {
        "message": query.replace("\n", " ").strip(),
        "mode": "query",
        "reset": True,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(chat_url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
            if r.status_code == 200:
                result = r.json()
                text = result.get("textResponse", "")
                if text:
                    return text
                print(f"[WARN] RAG query returned empty response")
                return ""
            else:
                error_text = r.text
                print(f"[ERROR] RAG query status {r.status_code}: {error_text[:200]}")
                # If the model failed, try the fallback model on the next attempt
                if "failed to communicate" in error_text.lower() and attempt == 0:
                    print(f"[INFO] Retrying with fallback model: {FALLBACK_LLM_MODEL}")
                    update_anythingllm_workspace_model(FALLBACK_LLM_MODEL)
        except requests.exceptions.Timeout:
            print(f"[ERROR] RAG query timeout (attempt {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            print(f"[ERROR] RAG query failed: {str(e)}")
        time.sleep(2 ** attempt)

    return ""


def get_llm_response_with_data_context(markdown_content: str, query: str, llm_model: str,
                                       doc_label: str = "analysis_data",
                                       skip_rag: bool = False) -> str:
    """
    Send data to the LLM by uploading it as a document and querying in RAG mode.

    Workflow:
      1. Convert data to markdown and upload as a document to AnythingLLM
      2. Link the document to the workspace via update-embeddings adds
      3. Query in RAG mode against the embedded document
      4. Fall back to chat mode with inline data if RAG fails

    For large datasets the caller chunks the data and calls this function
    multiple times, then consolidates the chunk summaries.

    Args:
        markdown_content: The markdown-formatted data to upload.
        query: The analysis prompt to send.
        llm_model: LLM model name.
        doc_label: Label used for the uploaded document filename.
        skip_rag: If True, skip the upload+RAG path and go directly to inline
            chat mode. Use this when the content is structured tabular data
            (e.g. the markdown produced by convert_hits_to_markdown), where
            semantic RAG retrieval drops most rows that don't closely match
            the query embedding. For prose content (e.g. consolidation of
            multiple LLM summaries), keep RAG enabled.

    Returns:
        LLM response string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{doc_label}_{timestamp}.md"

    # ⭐ NEW (v3.1.1.2): tabular-data callers (single ES-result passes) opt out
    # of RAG because semantic retrieval drops rows that don't match the query
    # embedding, producing summaries that cover only a tiny fraction of the
    # dataset. Inline chat puts the full markdown table in the prompt.
    if skip_rag:
        print(f"[INFO] skip_rag=True for {filename} ({len(markdown_content)} chars); using inline chat mode")
        return _fallback_chat_with_inline(markdown_content, query, llm_model, filename)

    print(f"[INFO] Uploading markdown document: {filename} ({len(markdown_content)} chars)")

    # Step 1: Upload document → get location path
    doc_location = upload_document_to_anythingllm(markdown_content, filename)
    if not doc_location:
        print("[WARN] Document upload failed, falling back to chat mode with inline data")
        return _fallback_chat_with_inline(markdown_content, query, llm_model, filename)

    # Step 2: Link document to workspace
    print(f"[INFO] Adding document to workspace embeddings...")
    if not add_document_to_workspace(doc_location):
        print("[WARN] Failed to add document to workspace, trying RAG query anyway")

    # Step 3: Query in RAG mode
    # Brief pause to let embeddings settle
    time.sleep(2)

    print(f"[INFO] Sending query in RAG mode (model: {llm_model})...")
    rag_response = query_anythingllm_rag(query, llm_model)

    # ⭐ NEW (v3.1.1.2): Treat degenerate RAG outputs (e.g. "..." or single-word
    # "I don't know") as failure so we fall through to the inline path that has
    # the full data in the prompt.
    RAG_MIN_USEFUL_LEN = 20
    if rag_response and len(rag_response.strip()) >= RAG_MIN_USEFUL_LEN:
        print(f"[INFO] RAG response received ({len(rag_response)} chars)")
        return rag_response
    if rag_response:
        print(f"[WARN] RAG response too short ({len(rag_response.strip())} chars, min {RAG_MIN_USEFUL_LEN}); falling back to inline mode")

    # Step 4: Fallback to chat mode with inline data
    print("[WARN] RAG query returned no/short response, falling back to chat mode with inline data")
    return _fallback_chat_with_inline(markdown_content, query, llm_model, filename)


# ──────────────────────────────────────────────
# ⭐ NEW (v3.1.1.4): Per-chunk upload+RAG in fresh workspaces
# ──────────────────────────────────────────────
def create_anythingllm_workspace(slug: str, name: str) -> str:
    """
    POST /api/v1/workspace/new — create a fresh AnythingLLM workspace for a
    single chunk's analysis.

    ⭐ v3.1.1.4: this AnythingLLM build IGNORES the requested ``slug`` and
    generates the real slug from the workspace ``name`` (slugified). Every
    later call (document upload, settings update, RAG chat, delete) must use
    the server-generated slug, otherwise the API answers 400 "Bad Request".
    We therefore return the actual slug reported by the server, or "" on
    failure. Callers must use the returned value.

    Each chunk gets its own workspace so:
      - RAG retrieval cannot leak rows from other chunks (no cross-contamination)
      - topN can be tuned to exactly the chunk's row count
      - Workspace cleanup is one DELETE call (no per-document bookkeeping)
    """
    url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/new"
    headers = get_anythingllm_headers()
    try:
        r = requests.post(
            url, headers=headers,
            json={"name": name, "slug": slug},
            timeout=15,
        )
        if r.status_code in (200, 201):
            try:
                data = r.json()
                ws = data.get("workspace")
                candidates = []
                if isinstance(ws, dict):
                    candidates = [ws]
                elif isinstance(ws, list):
                    candidates = [w for w in ws if isinstance(w, dict)]
                for w in candidates:
                    actual = w.get("slug")
                    if actual:
                        print(f"[INFO] Created workspace '{slug}' (server slug: '{actual}')")
                        return actual
            except Exception as e:
                print(f"[WARN] Could not parse workspace-create response: {e}")
            print(f"[INFO] Created workspace '{slug}' (no slug in response; assuming '{slug}')")
            return slug
        print(f"[WARN] Workspace create status {r.status_code} for '{slug}': {r.text[:200]}")
        return ""
    except Exception as e:
        print(f"[WARN] Workspace create failed for '{slug}': {e}")
        return ""


def delete_anythingllm_workspace(slug: str) -> bool:
    """
    DELETE /api/v1/workspace/{slug} — remove the per-chunk workspace and all
    its uploaded documents. Idempotent: returns True if already gone.
    """
    if not slug:
        return True
    url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}"
    headers = get_anythingllm_headers()
    try:
        r = requests.delete(url, headers=headers, timeout=15)
        if r.status_code in (200, 201, 204):
            print(f"[INFO] Deleted workspace '{slug}'")
            return True
        if r.status_code == 404:
            return True  # already gone
        print(f"[WARN] Workspace delete status {r.status_code} for '{slug}': {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[WARN] Workspace delete failed for '{slug}': {e}")
        return False


def set_workspace_topn(slug: str, top_n: int, similarity_threshold: float = 0.0) -> bool:
    """
    POST /api/v1/workspace/{slug}/update — bump topN and drop similarity
    threshold for a per-chunk workspace. Returns True on success.
    """
    url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/update"
    headers = get_anythingllm_headers()
    try:
        r = requests.post(
            url, headers=headers,
            json={"topN": top_n, "similarityThreshold": similarity_threshold},
            timeout=15,
        )
        if r.status_code == 200:
            print(f"[INFO] Workspace '{slug}' tuned: topN={top_n}, sim={similarity_threshold}")
            return True
        print(f"[WARN] Workspace update status {r.status_code} for '{slug}': {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[WARN] Workspace update failed for '{slug}': {e}")
        return False


def upload_row_to_workspace(row_text: str, title: str, workspace_slug: str) -> Optional[str]:
    """
    POST /api/v1/document/raw-text — upload one row's natural-language
    paragraph to the chunk's workspace. The document is auto-embedded into
    that workspace (via the addToWorkspaces param).

    Returns the storage filename for later cleanup, or None on failure.
    """
    url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/document/raw-text"
    headers = get_anythingllm_headers()
    payload = {
        "textContent": row_text,
        "metadata": {"title": title, "docSource": "ai_analysis_pipeline_chunk"},
        "addToWorkspaces": workspace_slug,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code in (200, 201):
                result = r.json()
                if isinstance(result, dict) and result.get("documents"):
                    doc = result["documents"][0]
                    location = doc.get("location", "")
                    filename = location.split("/")[-1] if location else None
                    if filename:
                        return filename
                return None
            print(f"[ERROR] Raw-text upload status {r.status_code} for '{title}': {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[ERROR] Raw-text upload timeout for '{title}'")
        except Exception as e:
            print(f"[ERROR] Raw-text upload failed for '{title}': {e}")
        time.sleep(2 ** attempt)
    return None


def format_chunk_as_natural_paragraph(chunk_rows: list, start_idx: int) -> str:
    """
    Build a single natural-language paragraph summarizing a chunk of ES hits.
    Each row becomes a sentence with location, date, type, source, and (when
    present) a description. The paragraph is what gets uploaded per row.
    """
    sentences = []
    for offset, hit in enumerate(chunk_rows):
        idx = start_idx + offset
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict):
            source = {}
        es_id = hit.get("_id", "") if isinstance(hit, dict) else ""
        location = source.get("location_name") or source.get("pass_name") or source.get("visit_name") or ""
        infra_type = source.get("infra_type") or source.get("equipment_type") or source.get("enemy_formation_name") or ""
        activity_date = source.get("activity_date") or source.get("@timestamp") or ""
        source_type = source.get("source_type") or ""
        file_path = source.get("file_http_path") or ""
        file_name = file_path.split("/")[-1] if file_path else ""
        injester = source.get("ingester_name") or source.get("injester_name") or ""
        description = source.get("description") or ""

        parts = [
            f"Record {idx}.",
            f"Elastic Doc ID: {es_id}.",
        ]
        if activity_date:
            parts.append(f"Activity date: {activity_date}.")
        if source_type:
            parts.append(f"Source type: {source_type}.")
        if file_name:
            parts.append(f"File: {file_name}.")
        if injester:
            parts.append(f"Ingester: {injester}.")
        if location:
            parts.append(f"Location: {location}.")
        if infra_type:
            parts.append(f"Category: {infra_type}.")
        sentence = " ".join(parts)
        if description:
            sentence += f" Description: {str(description)[:400]}"
        sentences.append(sentence)
    return " ".join(sentences)


def format_chunk_as_document(chunk_rows: list, start_idx: int) -> str:
    """Build one Markdown document containing all records in a chunk.

    Keeping explicit record boundaries makes a single uploaded document easier
    for AnythingLLM's text splitter/retriever to preserve and for the LLM to
    cite internally. The document contains the same fields used by the legacy
    per-row uploader, but is embedded/uploaded exactly once per chunk.
    """
    if not chunk_rows:
        return ""

    lines = [
        f"# AI Analysis Chunk (Records {start_idx} to {start_idx + len(chunk_rows) - 1})",
        "",
    ]

    for offset, hit in enumerate(chunk_rows):
        idx = start_idx + offset
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict):
            source = {}

        es_id = hit.get("_id", "") if isinstance(hit, dict) else ""
        location = (
            source.get("location_name")
            or source.get("pass_name")
            or source.get("visit_name")
            or ""
        )
        infra_type = (
            source.get("infra_type")
            or source.get("equipment_type")
            or source.get("enemy_formation_name")
            or ""
        )
        activity_date = source.get("activity_date") or source.get("@timestamp") or ""
        source_type = source.get("source_type") or ""
        file_path = source.get("file_http_path") or ""
        file_name = file_path.split("/")[-1] if file_path else ""
        ingester = source.get("ingester_name") or source.get("injester_name") or ""
        description = source.get("description") or ""

        lines.append(f"## Record {idx}")
        lines.append(f"- Citation No.: {idx}")
        if activity_date:
            lines.append(f"- Activity date: {activity_date}")
        if source_type:
            lines.append(f"- Source type: {source_type}")
        if file_name:
            lines.append(f"- File: {file_name}")
        if ingester:
            lines.append(f"- Ingester: {ingester}")
        if location:
            lines.append(f"- Location: {location}")
        if infra_type:
            lines.append(f"- Category: {infra_type}")
        if es_id:
            lines.append(f"- Elastic Doc ID: {es_id}")
        if description:
            # Preserve the existing per-record safety cap so a single verbose
            # record cannot consume the whole chunk context.
            lines.append(f"- Description: {str(description)[:400]}")
        lines.append("")

    return "\n".join(lines).strip()


def upload_chunk_document_to_workspace(chunk_document: str, title: str,
                                       workspace_slug: str) -> Optional[str]:
    """Upload ONE combined chunk document to a specific AnythingLLM workspace.

    Uses /document/raw-text with addToWorkspaces so the document is directly
    associated with the isolated per-chunk workspace and auto-embedded there.
    This replaces the old one-API-call-per-row upload pattern.
    """
    if not chunk_document or not workspace_slug:
        return None

    url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/document/raw-text"
    headers = get_anythingllm_headers()
    payload = {
        "textContent": chunk_document,
        "metadata": {
            "title": title,
            "docSource": "ai_analysis_pipeline_chunk",
        },
        "addToWorkspaces": workspace_slug,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (200, 201):
                result = r.json()
                if isinstance(result, dict) and result.get("documents"):
                    doc = result["documents"][0]
                    location = doc.get("location", "")
                    filename = location.split("/")[-1] if location else None
                    print(
                        f"[INFO] Uploaded one chunk document '{title}' "
                        f"({len(chunk_document)} chars) to '{workspace_slug}'"
                    )
                    return filename or title
                print(f"[WARN] Chunk document uploaded but response had no document metadata")
                return title

            print(
                f"[ERROR] Chunk document upload status {r.status_code} "
                f"for '{title}': {r.text[:300]}"
            )
        except requests.exceptions.Timeout:
            print(
                f"[ERROR] Chunk document upload timeout for '{title}' "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
        except Exception as e:
            print(f"[ERROR] Chunk document upload failed for '{title}': {e}")
        time.sleep(2 ** attempt)

    return None


def _resolve_chunk_rag_topn(chunk_document_tokens: int) -> int:
    """Return a coverage-oriented, bounded retrieval count for one chunk document.

    The chunk document is deliberately small enough to fit well inside the LLM
    context, so RAG is used primarily for AnythingLLM compatibility/isolation.
    Retrieval therefore favors broad coverage rather than aggressive semantic
    filtering. The operator can override the value with AI_SUMMARY_RAG_TOPN.
    """
    override = _safe_int(os.getenv("AI_SUMMARY_RAG_TOPN"))
    if override is not None and override > 0:
        return max(8, min(DEFAULT_RAG_TOP_N_MAX, override))

    # AnythingLLM's internal splitter is deployment-specific. Use a generous
    # estimate and a high bounded ceiling so minority/one-off material is less
    # likely to be crowded out by dominant themes.
    estimated_fragments = max(8, (int(chunk_document_tokens) + 79) // 80)
    return max(24, min(DEFAULT_RAG_TOP_N, estimated_fragments + 12))


def upload_chunk_and_query_rag(chunk_rows: list, chunk_idx: int, total_chunks: int,
                                start_row_idx: int, query: str, llm_model: str,
                                doc_label: str, latest_evidence_date: Optional[str] = None) -> str:
    """Analyze one chunk using exactly ONE document upload.

    Lifecycle:
      1. Create an isolated AnythingLLM workspace.
      2. Build ONE document containing all rows in the chunk.
      3. Upload/embed that ONE document into the workspace.
      4. Set model + retrieval topN and query in RAG mode.
      5. Fall back to inline chat if RAG fails.
      6. ALWAYS delete the workspace in finally.

    This sharply reduces HTTP/embedding overhead versus uploading one document
    for every record and makes the pipeline much more practical for hundreds or
    thousands of Elasticsearch records.
    """
    if not chunk_rows:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_row_idx = start_row_idx + len(chunk_rows) - 1
    name = (
        f"Chunk {chunk_idx}/{total_chunks} (rows {start_row_idx}-{end_row_idx}) "
        f"{doc_label} {timestamp}"
    )
    requested_slug = (
        f"{doc_label}_chunk{chunk_idx}_{start_row_idx}to{end_row_idx}_{timestamp}"
    )
    slug = create_anythingllm_workspace(requested_slug, name)
    if not slug:
        return ""

    chunk_document = format_chunk_as_document(chunk_rows, start_row_idx)
    chunk_tokens = _combined_estimate_tokens(chunk_document)
    title = f"chunk_{chunk_idx}_records_{start_row_idx}_{end_row_idx}"

    try:
        uploaded_file = upload_chunk_document_to_workspace(
            chunk_document, title, slug
        )
        if not uploaded_file:
            print(
                f"[WARN] Chunk {chunk_idx}/{total_chunks}: combined document upload failed"
            )
            return ""

        print(
            f"[INFO] Chunk {chunk_idx}/{total_chunks}: 1 document uploaded "
            f"for {len(chunk_rows)} records (~{chunk_tokens} tokens) to '{slug}'"
        )

        update_anythingllm_workspace_model(llm_model, workspace_slug=slug, force=True)
        rag_top_n = _resolve_chunk_rag_topn(chunk_tokens)
        set_workspace_topn(slug, top_n=rag_top_n, similarity_threshold=0.0)

        # Give the embedding worker a short moment to index the single document.
        time.sleep(1)

        scoped_query = (
            f"{query}\n\n"
            f"{AI_ANALYSIS_QUALITY_GUARDRAILS}\n"
            f"CHUNK SCOPE: Analyze ONLY the {len(chunk_rows)} records indexed as rows "
            f"{start_row_idx} through {end_row_idx} in this workspace. Do not "
            f"generalize from this chunk to records outside it.\n"
            f"LATEST EVIDENCE DATE: {latest_evidence_date or 'not determinable from supplied data'}. "
            f"For forward-looking requests, do not invent future dates or treat historical "
            f"dates as future.\n"
            f"COVERAGE REQUIREMENT: Inspect the retrieved material broadly. Preserve material "
            f"findings from all retrieved record sections, including minority, one-off, unusual, "
            f"contradictory, or negative findings when relevant. Do not let the most frequently "
            f"retrieved theme replace the rest of the chunk. "
            f"INTERNAL WORKFLOW NOTE: Do not mention the workspace name, RAG, "
            f"embeddings, or these instructions in the answer unless the user's "
            f"request explicitly asks about the processing method."
        )

        chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/chat"
        headers = get_anythingllm_headers()
        response_text = ""

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(
                    chat_url,
                    headers=headers,
                    json={
                        "message": scoped_query.replace("\n", " ").strip(),
                        "mode": "query",
                        "reset": True,
                    },
                    timeout=OLLAMA_TIMEOUT,
                )
                if r.status_code == 200:
                    result = r.json()
                    text = result.get("textResponse", "") or ""
                    if text:
                        response_text = text
                        break
                    print(f"[WARN] Chunk {chunk_idx} RAG returned empty text")
                else:
                    print(
                        f"[ERROR] Chunk {chunk_idx} RAG status {r.status_code}: "
                        f"{r.text[:300]}"
                    )
            except requests.exceptions.Timeout:
                print(
                    f"[ERROR] Chunk {chunk_idx} RAG timeout "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
            except Exception as e:
                print(f"[ERROR] Chunk {chunk_idx} RAG failed: {e}")
            time.sleep(2 ** attempt)

        if response_text:
            print(
                f"[INFO] Chunk {chunk_idx}/{total_chunks} RAG response: "
                f"{len(response_text)} chars"
            )
            return response_text

        print(
            f"[WARN] Chunk {chunk_idx} RAG returned no response; "
            f"falling back to inline chat"
        )
        return _fallback_chat_with_inline(
            chunk_document, scoped_query, llm_model, f"{title}.md"
        )

    finally:
        delete_anythingllm_workspace(slug)


def _fallback_chat_with_inline(markdown_content: str, query: str, llm_model: str,
                                filename: str = "data.md") -> str:
    """Fallback: embed markdown data inline in a chat prompt."""
    combined_prompt = (
        f"Here is the data to analyze (from {filename}):\n\n"
        f"{markdown_content}\n\n"
        f"---\n\n"
        f"{query}"
    )
    print(f"[INFO] Fallback to chat mode with inline data: {len(combined_prompt)} total chars")
    return get_llm_response_direct(combined_prompt, llm_model)


def get_llm_response_direct(query: str, llm_model: str, max_tokens=None) -> str:
    """Direct chat mode: sends the prompt as-is to the AnythingLLM workspace chat API.
    
    If the configured model returns an empty response ("text response was empty"),
    the function will try once with a fallback model (FALLBACK_LLM_MODEL) before
    returning an error message. This prevents wasting minutes on retries for
    models that are installed but don't produce valid chat output.
    """
    chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
    headers = get_anythingllm_headers()

    def _attempt(model: str, timeout: int) -> str:
        """Single attempt with a given model and timeout."""
        update_anythingllm_workspace_model(model)
        payload = {
            "message": query.replace("\n", " ").strip(),
            "mode": "chat",
            "reset": True,
        }
        try:
            r = requests.post(chat_url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                result = r.json()
                text = result.get("textResponse", "")
                if text:
                    return text
                # Empty text response — model returned no output
                print(f"[WARN] Model '{model}' returned empty text response")
                return None
            else:
                error_text = r.text
                print(f"[ERROR] Chat mode status {r.status_code} for model '{model}': {error_text[:200]}")
                # Check if the model itself is the problem
                if "failed to communicate" in error_text.lower() or "text response was empty" in error_text.lower():
                    return None  # Model problem, not a transient error
                return None
        except requests.exceptions.Timeout:
            print(f"[ERROR] Timeout for model '{model}' (attempt with {timeout}s timeout)")
            return None
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return None

    # --- Primary attempt with configured model ---
    print(f"[INFO] Trying model: {llm_model}")
    result = _attempt(llm_model, OLLAMA_TIMEOUT)
    if result is not None:
        return result

    # --- Fallback attempt with known-good model ---
    if llm_model != FALLBACK_LLM_MODEL:
        print(f"[WARN] Model '{llm_model}' failed. Falling back to '{FALLBACK_LLM_MODEL}'...")
        result = _attempt(FALLBACK_LLM_MODEL, OLLAMA_TIMEOUT)
        if result is not None:
            return result

    print(f"[ERROR] All model attempts failed. Returning fallback message.")
    return "<p>AI analysis unavailable — LLM model could not generate a response.</p>"


# ──────────────────────────────────────────────
# ⭐ NEW: Convert ES hits to Markdown format
# ──────────────────────────────────────────────

# Mapping of analysis types to field configurations for markdown generation
TYPE_MAPPING = {
    "infra": {
        "types": ["infra development analysis", "event infra development analysis"],
        "fields": ["infra_type", "location_name", "activity_date", "coordinates", "description"],
        "field_labels": ["Infra Type", "Location Name", "Activity Date", "Coordinates", "Description"],
    },
    "training": {
        "types": ["training areas analysis", "event training areas analysis"],
        "fields": ["enemy_formation_name", "location_name", "description"],
        "field_labels": ["Enemy Formation Name", "Location Name", "Description"],
    },
    "general": {
        "types": ["general area analysis", "event general area analysis"],
        "fields": ["location_name", "coordinates", "description"],
        "field_labels": ["Location Name", "Coordinates", "Description"],
    },
    "force": {
        "types": ["force disposition analysis", "event force disposition analysis"],
        "fields": ["location_name", "coordinates", "base_location_name", "base_coordinates",
                   "enemy_formation_name", "orbate_title", "description"],
        "field_labels": ["Location Name", "Coordinates", "Base Location Name", "Base Coordinates",
                         "Enemy Formation Name", "ORBAT Title", "Description"],
    },
    "sitrep": {
        "types": ["pla sitrep analysis", "event pla sitrep analysis"],
        "fields": ["pass_name", "transgression_sighting_type", "sub_activity_type", "description"],
        "field_labels": ["Pass Name", "Transgression/Sighting Type", "Sub Activity Type", "Description"],
    },
    "air_aspects": {
        "types": ["air aspects analysis", "event air aspects analysis"],
        "fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name",
                   "equipement_type", "count", "airfield_type"],
        "field_labels": ["Location Name", "Coordinates", "Infra Name", "Infra Type", "Equipment Name",
                         "Equipment Type", "Count", "Airfield Type"],
    },
    "sam_deployment_analysis": {
        "types": ["sam deployment analysis", "event sam deployment analysis"],
        "fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name",
                   "equipment_type", "count"],
        "field_labels": ["Location Name", "Coordinates", "Infra Name", "Infra Type", "Equipment Name",
                         "Equipment Type", "Count"],
    },
    "mobile_interception": {
        "types": ["mobile interception analysis", "Mobile Interception Analysis"],
        "fields": ["start_location_name", "end_location_name", "opposite_to", "mobile_no", "description"],
        "field_labels": ["Start Location Name", "End Location Name", "Opposite To", "Mobile No", "Description"],
    },
    "internal_security": {
        "types": ["internal security analysis", "Internal Security Analysis"],
        "fields": ["coordinates", "terrorist_casualties_", "security_forces_casualties", "civilian_casualties",
                   "description", "army_name", "force_type_name", "formation_type", "enemy_formation_name",
                   "command_name", "command_coordinates", "comd_tps_loc_name", "comd_tps_coordinates",
                   "terrorists_casualties"],
        "field_labels": ["Coordinates", "Terrorist Casualties", "Security Forces Casualties", "Civilian Casualties",
                         "Description", "Army Name", "Force Type Name", "Formation Type", "Enemy Formation Name",
                         "Command Name", "Command Coordinates", "COMd TPS Loc Name", "COMd TPS Coordinates",
                         "Terrorists Casualties"],
    },
    "elint": {
        "types": ["elint analysis", "Elint Analysis"],
        "fields": ["description", "location_name", "coordinates", "category", "radar_type", "radar_name"],
        "field_labels": ["Description", "Location Name", "Coordinates", "Category", "Radar Type", "Radar Name"],
    },
    "visit": {
        "types": ["visit analysis", "Visit Analysis"],
        "fields": ["description", "visit_name", "purpose", "location_name", "coordinates"],
        "field_labels": ["Description", "Visit Name", "Purpose", "Location Name", "Coordinates"],
    }
}


def convert_hits_to_markdown(e_hits: list, search_form_type: str) -> str:
    """
    Convert Elasticsearch hits to a well-structured markdown document.
    
    For profile analysis: uses a description-based format grouped by file.
    For other types: uses the TYPE_MAPPING to create a table.
    
    Args:
        e_hits: Elasticsearch hit records
        search_form_type: Type of analysis (e.g., "infra development analysis")
    
    Returns:
        Markdown string
    """
    search_form_type_lower = search_form_type.lower()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # ── Header ──
    lines.append(f"# Analysis Report: {search_form_type}")
    lines.append(f"")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **Total Records:** {len(e_hits)}")
    lines.append(f"")

    if not e_hits:
        lines.append("_No records found._")
        return "\n".join(lines)

    # ── Profile Analysis: descriptive format ──
    if search_form_type_lower == 'profile analysis':
        lines.append("## Data Records\n")
        for idx, hit in enumerate(e_hits, 1):
            source = hit.get("_source", {})
            lines.append(f"### Record {idx}")
            lines.append("")
            if source.get("file_http_path"):
                lines.append(f"- **File Path:** {source['file_http_path']}")
            if source.get("description"):
                lines.append(f"- **Description:** {source['description']}")
            if source.get("@timestamp"):
                lines.append(f"- **Date:** {source['@timestamp']}")
            if source.get("person_name"):
                lines.append(f"- **Person Name:** {source['person_name']}")
            if source.get("civil_organization") or source.get("civil_organization_name"):
                org = source.get("civil_organization", {})
                if isinstance(org, dict):
                    org_name = org.get("civil_organization_name", "")
                else:
                    org_name = source.get("civil_organization_name", "")
                if org_name:
                    lines.append(f"- **Organization:** {org_name}")
            lines.append("")
        return "\n".join(lines)

    # ── Standard Analysis Types: table format ──
    type_config = None
    for key, config in TYPE_MAPPING.items():
        if search_form_type_lower in [t.lower() for t in config["types"]]:
            type_config = config
            break

    if type_config is None:
        # Fallback: generic format with all available fields
        lines.append("## Data Records\n")
        for idx, hit in enumerate(e_hits, 1):
            source = hit.get("_source", {})
            lines.append(f"### Record {idx}")
            lines.append("")
            for field, value in source.items():
                if field.startswith("@"):
                    continue
                lines.append(f"- **{field}:** {value}")
            lines.append("")
        return "\n".join(lines)

    fields = type_config["fields"]
    field_labels = type_config["field_labels"]

    # Section: Analysis Statistics
    lines.append("## Analysis Statistics")
    lines.append("")
    lines.append(f"- **Analysis Category:** {search_form_type}")
    lines.append(f"- **Fields Tracked:** {', '.join(field_labels)}")
    lines.append("")

    # Section: Detail Table
    lines.append("## Detail Records")
    lines.append("")

    # Build table header
    header = "| # | " + " | ".join(field_labels) + " |"
    separator = "|---|" + "|".join(["---"] * len(field_labels)) + "|"
    lines.append(header)
    lines.append(separator)

    # Build table rows
    for idx, hit in enumerate(e_hits, 1):
        source = hit.get("_source", {})
        row_values = []
        for field in fields:
            value = source.get(field, "")
            # Ensure value is string-safe
            if value is None:
                value = ""
            value_str = str(value)
            # Escape markdown table pipe characters
            value_str = value_str.replace("|", "\\|")
            # Truncate long values for readability
            if len(value_str) > 120:
                value_str = value_str[:117] + "..."
            row_values.append(value_str)

        row = f"| {idx} | " + " | ".join(row_values) + " |"
        lines.append(row)

    lines.append("")
    lines.append(f"---")
    lines.append(f"*End of report — {len(e_hits)} records*")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Existing helpers from v3.0.0
# ──────────────────────────────────────────────
def correct_json_format(json_str):
    open_curly_count = 0
    open_square_count = 0
    result = []
    prev_b = []
    prev_inverted = 0
    prev_inverted_array = []
    json_str = (
        json_str.replace(", ", ",")
        .replace("\n", "")
        .replace("\r", "")
        .replace(" }", "}")
        .replace("} ", "}")
        .replace("{ ", "{")
        .replace(" {", "{")
        .replace(" [", "[")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("] ", "]")
        .strip()
    )
    for char in json_str:
        if char == '"':
            if prev_inverted == 0:
                prev_inverted = 1
                prev_inverted_array.append('"')
            else:
                prev_inverted = 0
                prev_inverted_array.pop()
        if char == "{":
            open_curly_count += 1
            prev_b.append(char)
            result.append(char)
        elif char == "[":
            open_square_count += 1
            prev_b.append(char)
            result.append(char)
        elif char == "}":
            if prev_inverted == 1:
                result.append('"')
                prev_inverted = 0
                prev_inverted_array.pop()
            if prev_b and prev_b[-1] == "[":
                result.append("]")
            if prev_b:
                prev_b.pop()
            result.append(char)
        elif char == "]":
            if prev_inverted == 1:
                result.append('"')
                prev_inverted = 0
                prev_inverted_array.pop()
            if prev_b and prev_b[-1] == "{":
                result.append("}")
            if prev_b:
                prev_b.pop()
            result.append(char)
        else:
            result.append(char)
    for p_b in prev_b[::-1]:
        if p_b == "{":
            result.append("}")
        if p_b == "[":
            result.append("]")
    result = "".join(result)
    result = result.replace(",}", "}").replace(",]", "]").strip()
    try:
        result = json.loads(result)
    except Exception as e:
        traceback.print_exc()
        print("[!]json load failed:", e)
        result = {"summary": result}
    return result


def update_imint_ai_analysis(cursor, imint_ai_analysis_id, ai_analysis_text):
    q = """UPDATE public.imint_ai_analysis SET status=%d,ai_analysis_text='%s' WHERE imint_ai_analysis_id=%d;"""
    cursor.execute(q % (1, ai_analysis_text, imint_ai_analysis_id,))


def update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id, query_text):
    q = """UPDATE public.ai_analysis_summary SET query_status=%d,query_text='%s' WHERE ai_analysis_summary_id=%d;"""
    cursor.execute(q % (1, query_text, ai_analysis_summary_id,))


def _extract_field(source: dict, field_path: str):
    """Walk a dotted field path into a hit ``_source`` dict.

    Supports nested keys like ``description_hash.keyword``.
    Returns ``None`` if any segment is missing.
    """
    val = source
    for part in field_path.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def _build_dedup_key(source: dict, dedup_fields: list) -> str:
    """Build a composite dedup key string from the given fields.

    ``dedup_fields`` is a list of dotted field paths (e.g.
    ``["description_hash.keyword", "description", "comments"]``).
    Returns a hashable string; missing fields are treated as empty.
    """
    parts = []
    for field in dedup_fields:
        val = _extract_field(source, field)
        parts.append(str(val).strip() if val is not None else "")
    return "\x00".join(parts)


def generic_es_scroll(client: Elasticsearch, index_name: str, base_query_json: dict,
                      batch_size: int = 1000, keep_alive: str = "2m",
                      dedup_stats: dict = None, perform_dedup: bool = True):
    """
    Universal Elasticsearch scroll using the classic scroll API.

    Pages through all matching results in batches of ``batch_size``, yielding
    individual hit dicts.  The scroll context is always cleaned up in ``finally``.

    For ``collapse`` queries: Elasticsearch rejects both scroll and PIT when
    ``collapse`` is active, so the ``collapse`` parameter is stripped and
    deduplication is applied in Python after scrolling.
    """
    query = copy.deepcopy(base_query_json)

    # ── Collapse workaround: strip collapse, scroll, deduplicate in Python ──
    collapse_field = query.pop("collapse", None)
    collapse_key = None
    if collapse_field and isinstance(collapse_field, dict):
        collapse_key = collapse_field.get("field", None)

    # Always dedup by the composite fields below.
    if perform_dedup:
        dedup_fields = [
            "description_hash",
            "comments",
            "location_name",
            "activity_type",
            "activity_date",
            "enemy_formation_name",
        ]

        if collapse_key and collapse_key not in dedup_fields:
            dedup_fields.append(collapse_key)

        # Remove duplicates while preserving order
        seen_fields = set()
        unique_dedup_fields = []
        for f in dedup_fields:
            if f not in seen_fields:
                seen_fields.add(f)
                unique_dedup_fields.append(f)
    else:
        unique_dedup_fields = []

    if perform_dedup:
        if collapse_key:
            print(f"[INFO] Collapse on '{collapse_key}' — scrolling without collapse, "
                f"deduplicating by {unique_dedup_fields}")
        else:
            print(f"[INFO] Deduplicating hits by {unique_dedup_fields}")
    else:
        print("[INFO] Deduplication skipped — preserving all Elasticsearch hits")

    query["size"] = batch_size
    seen = set()
    dedup_count = 0
    total_count = 0

    scroll_id = None
    try:
        response = client.search(index=index_name, body=query, scroll=keep_alive)
        scroll_id = response.get("_scroll_id")

        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                total_count += 1

                if not perform_dedup:
                    yield hit
                    continue

                source = hit.get("_source", {})
                dedup_key = _build_dedup_key(source, unique_dedup_fields)

                if dedup_key not in seen:
                    seen.add(dedup_key)
                    yield hit
                else:
                    dedup_count += 1

            response = client.scroll(scroll_id=scroll_id, scroll=keep_alive)
            scroll_id = response.get("_scroll_id")

    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass  # best-effort cleanup

    if perform_dedup:
        print(f"[INFO] Scroll dedup: {total_count} total → {len(seen)} unique, "
            f"dropped {dedup_count} duplicates")
    else:
        print(f"[INFO] Scroll complete: {total_count} hits preserved "
            f"(deduplication skipped)")
    if dedup_stats is not None:
        dedup_stats.update({
            "total": total_count,
            "unique": len(seen),
            "dropped": dedup_count,
        })


def _dedup_hits(hits: list, dedup_fields: list) -> list:
    """Remove duplicate hits based on a composite key of ``dedup_fields``.

    ``dedup_fields`` is a list of dotted field paths to include in the key.
    Returns a de-duplicated list; the first occurrence is kept.
    """
    seen = set()
    result = []
    dropped = 0
    for hit in hits:
        source = hit.get("_source", {})
        key = _build_dedup_key(source, dedup_fields)
        if key not in seen:
            seen.add(key)
            result.append(hit)
        else:
            dropped += 1
    if dropped:
        print(f"[INFO] Post-search dedup by {dedup_fields}: "
              f"{len(hits)} → {len(result)} (dropped {dropped})")
    return result


def get_data_from_elastic(elastic_query, perform_dedup=True):
    """
    Fetch all matching documents from Elasticsearch.

    For small result sets (≤ batch_size), returns a single-page response.
    For ``collapse`` queries, strips ``collapse``, scrolls everything, then
    deduplicates in Python (ES rejects scroll/PIT with collapse).
    For larger non-collapse sets, uses the classic scroll API.

    All results are deduplicated using a composite key of
    ``description_hash``, ``comments``, ``location_name``, ``activity_type``,
    ``activity_date``, and ``enemy_formation_name`` (+ the collapse field
    when present) before being returned.

    When the query contains ``aggs``/``aggregation``, the full ES response
    (including the ``aggregations`` key) is returned so callers can access
    aggregation buckets.

    Returns the same dict shape as ``es.search()``:
        {"hits": {"total": {"value": N}, "hits": [...]}, "aggregations": {...}}
    """
    index_name = CUREENT_INDEX_NAME
    try:
        elastic_query = copy.deepcopy(elastic_query)
        size = elastic_query.get("size", None)
        has_aggs = "aggs" in elastic_query or "aggregation" in elastic_query
        has_collapse = "collapse" in elastic_query
        batch_size = 1000

        # Build dedup field list using the composite fields below.
        collapse_key = None
        if has_collapse:
            collapse_key = elastic_query.get("collapse", {}).get("field", None)
        if perform_dedup:
            dedup_fields = [
                "description_hash",
                "comments",
                "location_name",
                "activity_type",
                "activity_date",
                "enemy_formation_name",
            ]

            if collapse_key and collapse_key not in dedup_fields:
                dedup_fields.append(collapse_key)

            # Deduplicate while preserving order
            _seen = set()
            unique_dedup_fields = []

            for f in dedup_fields:
                if f not in _seen:
                    _seen.add(f)
                    unique_dedup_fields.append(f)
        else:
            unique_dedup_fields = []
        
        dedup_stats = {}
        # ── Collapse queries: strip collapse, scroll, dedup in Python ──
        # ── Collapse queries: strip collapse, scroll, dedup in Python ──
        if has_collapse:
            all_hits = list(
                generic_es_scroll(
                    es,
                    index_name,
                    elastic_query,
                    batch_size=batch_size,
                    dedup_stats=dedup_stats,
                    perform_dedup=perform_dedup
                )
            )

            if perform_dedup:
                all_hits = _dedup_hits(all_hits, unique_dedup_fields)

            return {
                "hits": {"total": {"value": len(all_hits)}, "hits": all_hits},
                "dedup_stats": dedup_stats
            }

        # ── Small result set: single page ──
        if size is not None and size <= batch_size:
            response = es.search(index=index_name, body=elastic_query)
            if response.get("hits", {}).get("hits"):
                raw_hits = response["hits"]["hits"]

                if perform_dedup:
                    deduped_hits = _dedup_hits(
                        raw_hits, unique_dedup_fields
                    )
                else:
                    deduped_hits = raw_hits

                dedup_stats.update({
                    "total": len(raw_hits),
                    "unique": len(deduped_hits),
                    "dropped": len(raw_hits) - len(deduped_hits),
                })

                response["hits"]["hits"] = deduped_hits
                response["hits"]["total"]["value"] = len(deduped_hits)

            response["dedup_stats"] = dedup_stats
            return response

        # ── Large or unbounded: paginate with classic scroll API ──
        # all_hits = list(
        #     generic_es_scroll(es, index_name, elastic_query, batch_size=batch_size)
        # )
        all_hits = list(
            generic_es_scroll(
                es,
                index_name,
                elastic_query,
                batch_size=batch_size,
                dedup_stats=dedup_stats,
                perform_dedup=perform_dedup
            )
        )
        result = {
            "hits": {"total": {"value": len(all_hits)}, "hits": all_hits},
            "dedup_stats": dedup_stats
        }

        # For aggregation queries, we also need the aggregation results.
        # Since generic_es_scroll only yields hits, we run a separate
        # aggregation-only query (size=0) to get the buckets.
        if has_aggs:
            agg_query = copy.deepcopy(elastic_query)
            agg_query["size"] = 0  # Only return aggs, not hits
            try:
                agg_response = es.search(index=index_name, body=agg_query)
                if "aggregations" in agg_response:
                    result["aggregations"] = agg_response["aggregations"]
                    if "hits" in agg_response and "total" in agg_response["hits"]:
                        result["hits"]["total"] = agg_response["hits"]["total"]
            except Exception as agg_err:
                print(f"[WARN] Aggregation query failed: {agg_err}")

        return result

    except Exception as e:
        print("[error] Failed to get data from elastic:", elastic_query)
        print(f"Exception {e}")
        return {"hits": {"total": {"value": 0}, "hits": []}}


def count_words_and_tokens(text: str):
    word_count = 0
    prev_space = True
    for ch in text:
        is_space = ch.isspace()
        if prev_space and not is_space:
            word_count += 1
        prev_space = is_space
    token_estimate = int(word_count * 1.3)
    return word_count, token_estimate


def build_chunks_from_lines(lines, max_tokens=1400):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for line in lines:
        line_tokens = int(len(line.split()) * 1.3)
        if current_tokens + line_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(line)
        current_tokens += line_tokens
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def analyze_chunk_direct_chat(chunk_rows: list, chunk_idx: int, total_chunks: int,
                              start_row_idx: int, query: str, llm_model: str,
                              latest_evidence_date: Optional[str] = None,
                              context_tokens: Optional[int] = None,
                              output_tokens: Optional[int] = None) -> str:
    """Analyze a complete bounded chunk with stateless direct Ollama chat.

    The full chunk document is included inline in the request. This is
    deliberately NOT semantic RAG: once a chunk is token-bounded, every record
    is considered relevant to the user's task, so retrieval ranking should not
    be allowed to hide less-similar records.

    The function is safe for parallel workers because each request is
    independent and does not share AnythingLLM workspace conversation history.
    """
    if not chunk_rows or not llm_model:
        return ""

    if not context_tokens or context_tokens <= 0:
        context_tokens = 8192

    chunk_document = format_chunk_as_document(chunk_rows, start_row_idx)
    chunk_tokens = _combined_estimate_tokens(chunk_document)
    end_row_idx = start_row_idx + len(chunk_rows) - 1

    prompt = (
        f"{query}\n\n"
        f"{AI_ANALYSIS_QUALITY_GUARDRAILS}\n"
        "FULL-CHUNK ANALYSIS MODE:\n"
        "The complete bounded chunk document is supplied below. Every record in "
        "the document is part of the evidence set for this chunk. Do NOT use "
        "semantic retrieval, relevance ranking, or topic similarity to omit "
        "records. Consider every record before deciding what is material.\n"
        f"CHUNK: {chunk_idx}/{total_chunks}; records {start_row_idx}-{end_row_idx}; "
        f"record count: {len(chunk_rows)}; estimated document tokens: {chunk_tokens}.\n"
        f"LATEST EVIDENCE DATE: {latest_evidence_date or 'not determinable from supplied data'}.\n"
        "For forward-looking requests, forecasts must concern only periods after "
        "the latest evidence date. Do not invent forecast dates or convert a past "
        "plan/target into a future event.\n"
        "COVERAGE REQUIREMENT: First inspect every numbered record. Build the answer from the full "
        "record set rather than sampling or selecting only the dominant theme. Preserve every "
        "material, task-relevant finding from every record, including unique, isolated, unusual, "
        "negative, contradictory, and one-off observations. When several records support the same "
        "finding, combine the repetition but retain material details contributed by individual records. "
        "Do not compress multiple distinct findings into one vague statement. Before finalizing, perform "
        "an internal record-by-record coverage check and ensure each supplied record has been considered "
        "for task relevance. STRICT PROMPT SCOPE: answer the user's request exactly as written. Do not "
        "introduce a different task, fixed domain template, or analysis category unless the user requested it.\n"
        "CITATION REQUIREMENT: Each supplied record has a Citation No. Use those numbers to cite the "
        "source of each analytical point. At the end of each distinct analytical point or paragraph, append "
        "the supporting citation numbers in parentheses, for example (3, 45, 50). Use only citation numbers "
        "that actually support that point. If a point is supported by several records, include all relevant "
        "citation numbers. Do not invent citation numbers.\n"
        "OUTPUT REQUIREMENT: Return a comprehensive, evidence-based answer to the user's request. "
        "Do not mention this workflow, workspace, RAG, embeddings, or chunking unless explicitly requested "
        "by the user. Do not use the words 'summary' or 'combined' in the generated report text; refer to "
        "the work as analysis or aggregation. Output HTML using only "
        "--- COMPLETE CHUNK DOCUMENT ---\n"
        f"{chunk_document}\n"
        "--- END COMPLETE CHUNK DOCUMENT ---"
    )

    prompt_tokens = _combined_estimate_tokens(prompt)
    print(
        f"[INFO] Chunk {chunk_idx}/{total_chunks}: direct full-chunk chat "
        f"({len(chunk_rows)} records, ~{chunk_tokens} data tokens, ~{prompt_tokens} total prompt tokens)"
    )

    # Prefer direct Ollama for stateless, parallel-safe execution.
    response = _get_direct_ollama_response(
        prompt,
        llm_model,
        context_tokens=context_tokens,
        stage_label=f"chunk {chunk_idx}/{total_chunks}",
        max_output_tokens=output_tokens,
    )
    if response:
        return sanitize_ai_report_text(response)

    # Last-resort AnythingLLM inline chat. This fallback is used only when the
    # direct Ollama request fails; the normal path never relies on shared chat
    # history or RAG retrieval.
    print(
        f"[WARN] Chunk {chunk_idx}/{total_chunks}: direct Ollama failed; "
        "trying AnythingLLM inline chat fallback"
    )
    fallback = get_llm_response_direct(prompt, llm_model, max_tokens=output_tokens)
    if fallback and len(fallback.strip()) > 20:
        return sanitize_ai_report_text(fallback.strip())
    return ""


def build_record_chunks(hits: list, start_row_idx: int, max_tokens: int,
                        max_records: Optional[int] = None) -> list:
    """Split ES hits by estimated token size, then rebalance a tiny tail chunk.

    The primary split is token-aware so records with long descriptions do not
    overflow the intended per-chunk document budget. A small post-pass moves
    records from the preceding chunk into an undersized final chunk whenever
    both chunks can still remain within the token budget. This avoids producing
    a disproportionately small final branch in the combined hierarchy.

    Returns dictionaries containing ``rows``, ``start``, ``end`` and ``tokens``.
    """
    if not hits:
        return []

    max_tokens = max(1, int(max_tokens))
    if max_records is not None:
        max_records = max(1, int(max_records))

    # Precompute the actual estimated document size of each record.
    row_entries = []
    for offset, hit in enumerate(hits):
        row_idx = start_row_idx + offset
        row_document = format_chunk_as_document([hit], row_idx)
        row_tokens = max(1, _combined_estimate_tokens(row_document))
        row_entries.append((hit, row_idx, row_tokens))

    chunks = []
    current_rows = []
    current_tokens = 0
    current_start = start_row_idx

    for hit, row_idx, row_tokens in row_entries:
        hit_limit_reached = (
            bool(current_rows)
            and max_records is not None
            and len(current_rows) >= max_records
        )
        token_limit_reached = (
            bool(current_rows)
            and current_tokens + row_tokens > max_tokens
        )

        if hit_limit_reached or token_limit_reached:
            chunks.append({
                "rows": current_rows,
                "start": current_start,
                "end": row_idx - 1,
                "tokens": current_tokens,
            })
            current_rows = []
            current_tokens = 0
            current_start = row_idx

        # A single oversized record remains a one-record chunk rather than
        # being dropped.
        current_rows.append(hit)
        current_tokens += row_tokens

    if current_rows:
        chunks.append({
            "rows": current_rows,
            "start": current_start,
            "end": current_start + len(current_rows) - 1,
            "tokens": current_tokens,
        })

    # Rebalance only the final chunk. The last chunk is the branch most likely
    # to become a tiny singleton/small branch after greedy token packing. Move
    # records from the end of the previous chunk to the beginning of the final
    # chunk until the final chunk reaches at least ~40% of the target or the
    # previous chunk cannot safely donate another record. Do not rebalance when
    # a max-record override would be violated.
    if len(chunks) >= 2:
        prev = chunks[-2]
        tail = chunks[-1]
        min_tail_tokens = max(1, int(max_tokens * 0.40))

        while (
            tail["tokens"] < min_tail_tokens
            and len(prev["rows"]) > 1
        ):
            if max_records is not None and len(tail["rows"]) >= max_records:
                break

            donor_hit = prev["rows"][-1]
            donor_index = prev["end"]
            donor_document = format_chunk_as_document([donor_hit], donor_index)
            donor_tokens = max(1, _combined_estimate_tokens(donor_document))

            if tail["tokens"] + donor_tokens > max_tokens:
                break
            if prev["tokens"] - donor_tokens <= 0:
                break

            prev["rows"].pop()
            prev["end"] -= 1
            prev["tokens"] -= donor_tokens

            tail["rows"].insert(0, donor_hit)
            tail["start"] -= 1
            tail["tokens"] += donor_tokens

        # If rows were moved, the final chunk's record ordering must still match
        # the global sorted ES order. Because we only donate from the end of the
        # preceding chunk and prepend to the tail, order is preserved.
        if tail["start"] != prev["end"] + 1:
            tail["start"] = prev["end"] + 1
            tail["end"] = tail["start"] + len(tail["rows"]) - 1

        if tail["tokens"] < min_tail_tokens and len(chunks) >= 3:
            # No safe donation was possible; leave the tail intact rather than
            # violating the token budget. This is expected for unusually long
            # final records.
            pass

    return chunks


def _resolve_chunk_workers(cli_value: Optional[int] = None) -> int:
    """Resolve a conservative worker count for independent chunk analyses."""
    if cli_value is not None and cli_value > 0:
        return max(1, min(MAX_CHUNK_WORKERS, cli_value))

    env_value = _safe_int(os.getenv("AI_SUMMARY_CHUNK_WORKERS"))
    if env_value is not None and env_value > 0:
        return max(1, min(MAX_CHUNK_WORKERS, env_value))

    return DEFAULT_CHUNK_WORKERS


def _process_chunk_job(job) -> tuple:
    """Worker wrapper for one complete, token-bounded chunk analysis.

    The normal path is direct Ollama with the COMPLETE chunk document inline.
    This avoids semantic-RAG retrieval hiding minority/one-off records.
    AnythingLLM inline chat is retained only as a last-resort fallback.
    """
    chunk_idx, total_chunks, chunk_info, query, model_name, doc_label, latest_evidence_date = job
    start_time = time.time()

    try:
        response = analyze_chunk_direct_chat(
            chunk_info["rows"],
            chunk_idx,
            total_chunks,
            start_row_idx=chunk_info["start"],
            query=query,
            llm_model=model_name,
            latest_evidence_date=latest_evidence_date,
            context_tokens=chunk_info.get("context_tokens"),
            output_tokens=chunk_info.get("output_tokens"),
        )
    except Exception as exc:
        print(f"[ERROR] Direct chunk {chunk_idx}/{total_chunks} failed: {exc}")
        traceback.print_exc()
        response = ""

    elapsed = time.time() - start_time
    return chunk_idx, chunk_info, response, elapsed

def _combined_estimate_tokens(*parts) -> int:
    """Rough token estimate (same heuristic as count_words_and_tokens) over all
    the given text parts joined together."""
    joined = "\n".join(part for part in parts if part)
    return count_words_and_tokens(joined)[1]


def _get_direct_ollama_response(prompt: str, llm_model: str,
                                context_tokens: int, stage_label: str = "analysis",
                                max_output_tokens: Optional[int] = None) -> str:
    """Stateless direct Ollama /api/chat request used for scalable analysis.

    The call uses one user message and never reuses prior conversation context.
    Ollama's keep_alive keeps the selected model resident between successive
    calls while bounded prompts ensure the declared context window is respected.
    Output is capped at MAX_DIRECT_OLLAMA_OUTPUT_TOKENS so chunk analysis has
    more room to preserve findings without allowing unbounded generation.
    """
    if not prompt or not llm_model:
        return ""

    ollama_url = f"http://{LLM_IP}:{LLM_PORT}/api/chat"
    prompt_tokens = _combined_estimate_tokens(prompt)
    available_output = context_tokens - prompt_tokens - 256
    # Avoid forcing a negative/oversized generation budget when the prompt is
    # close to the model context. A small minimum gives the model room to answer.
    if available_output < 256:
        print(
            f"[WARN] Direct Ollama {stage_label}: prompt too large "
            f"(~{prompt_tokens} tokens for context {context_tokens})"
        )
        return ""
    num_predict = min(MAX_DIRECT_OLLAMA_OUTPUT_TOKENS, max(256, available_output))
    if max_output_tokens is not None and max_output_tokens > 0:
        num_predict = min(num_predict, int(max_output_tokens))

    def _attempt(model: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": context_tokens,
                "num_predict": num_predict,
            },
        }
        try:
            r = requests.post(ollama_url, json=payload, timeout=OLLAMA_TIMEOUT)
            if r.status_code != 200:
                print(
                    f"[ERROR] Direct Ollama {stage_label} status {r.status_code} "
                    f"for model '{model}': {r.text[:300]}"
                )
                return ""
            result = r.json()
            message = result.get("message") if isinstance(result, dict) else None
            text = message.get("content", "") if isinstance(message, dict) else ""
            if not text and isinstance(result, dict):
                text = result.get("response", "") or ""
            text = str(text or "").strip()
            if text:
                return text
            print(f"[WARN] Direct Ollama {stage_label} returned empty response")
            return ""
        except requests.exceptions.Timeout:
            print(
                f"[ERROR] Direct Ollama {stage_label} timeout for model '{model}' "
                f"({OLLAMA_TIMEOUT}s)"
            )
            return ""
        except Exception as e:
            print(f"[ERROR] Direct Ollama {stage_label} failed: {e}")
            return ""

    print(
        f"[INFO] Direct Ollama {stage_label}: model={llm_model}, "
        f"prompt~{prompt_tokens} tokens, num_ctx={context_tokens}, "
        f"num_predict={num_predict}"
    )
    result = _attempt(llm_model)
    if result:
        return result
    if llm_model != FALLBACK_LLM_MODEL:
        print(
            f"[WARN] Direct Ollama {stage_label}: model '{llm_model}' failed; "
            f"trying fallback '{FALLBACK_LLM_MODEL}'"
        )
        result = _attempt(FALLBACK_LLM_MODEL)
        if result:
            return result
    return ""



def query_anythingllm_rag_workspace(query: str, llm_model: str, workspace_slug: str) -> str:
    """Run a stateless RAG query against a specific temporary AnythingLLM workspace."""
    if not query or not workspace_slug:
        return ""

    update_anythingllm_workspace_model(llm_model, workspace_slug=workspace_slug, force=True)
    chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{workspace_slug}/chat"
    headers = get_anythingllm_headers()
    payload = {
        "message": query.replace("\n", " ").strip(),
        "mode": "query",
        "reset": True,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(chat_url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
            if r.status_code == 200:
                result = r.json()
                text = result.get("textResponse", "") if isinstance(result, dict) else ""
                if text:
                    return str(text).strip()
                print("[WARN] Combined RAG query returned empty response")
                return ""
            error_text = r.text
            print(f"[ERROR] Combined RAG query status {r.status_code}: {error_text[:300]}")
            if "failed to communicate" in error_text.lower() and attempt == 0:
                print(f"[INFO] Combined RAG retry with fallback model: {FALLBACK_LLM_MODEL}")
                update_anythingllm_workspace_model(
                    FALLBACK_LLM_MODEL, workspace_slug=workspace_slug, force=True
                )
        except requests.exceptions.Timeout:
            print(
                f"[ERROR] Combined RAG query timeout "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
        except Exception as e:
            print(f"[ERROR] Combined RAG query failed: {e}")
        time.sleep(2 ** attempt)
    return ""


def build_combined_evidence_document(chunk_responses) -> str:
    """Build the single evidence document used by the optional combined RAG pass.

    The document contains the COMPLETE original chunk analyses with explicit chunk
    boundaries. RAG is used only to discover cross-chunk relationships; it is never
    the authoritative store for finding preservation.
    """
    parts = [
        "# Combined Chunk Evidence",
        "This document contains the complete analyses returned by every successful source chunk.",
        "Chunk labels and record ranges are part of the evidence and must be preserved.",
        "",
    ]
    for idx, (start, end, html_text) in enumerate(chunk_responses, 1):
        text = html_to_plain_text(html_text)
        parts.append(f"## Chunk {idx} — Records {start}-{end}")
        parts.append(text.strip())
        parts.append("")
    return "\n".join(parts).strip()


def run_combined_rag_enrichment(chunk_responses, full_query: str, model_name: str,
                                latest_evidence_date: Optional[str] = None) -> str:
    """Upload all chunk analyses as ONE AnythingLLM document and use RAG for enrichment.

    This is deliberately an enrichment pass, not a replacement for full-context
    synthesis. The final synthesizer still receives all source chunk analyses when
    they fit; the deterministic coverage appendix always preserves them in full.
    """
    if not chunk_responses or len(chunk_responses) < 2:
        return ""

    enabled = os.getenv("AI_SUMMARY_COMBINED_RAG", "1").strip().lower()
    if enabled in {"0", "false", "no", "off", "disabled"}:
        print("[INFO] Combined RAG enrichment disabled by AI_SUMMARY_COMBINED_RAG")
        return ""

    evidence_document = build_combined_evidence_document(chunk_responses)
    evidence_tokens = _combined_estimate_tokens(evidence_document)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    requested_slug = f"combined_rag_{timestamp}"
    workspace_name = f"Combined Evidence RAG {timestamp}"
    slug = create_anythingllm_workspace(requested_slug, workspace_name)
    if not slug:
        print("[WARN] Combined RAG workspace could not be created; continuing without RAG enrichment")
        return ""

    title = f"combined_chunk_evidence_{timestamp}"
    try:
        uploaded = upload_chunk_document_to_workspace(evidence_document, title, slug)
        if not uploaded:
            print("[WARN] Combined RAG evidence upload failed; continuing without RAG enrichment")
            return ""

        print(
            f"[INFO] Combined RAG: uploaded ONE evidence document (~{evidence_tokens} tokens) "
            f"covering {len(chunk_responses)} chunks to '{slug}'"
        )
        update_anythingllm_workspace_model(model_name, workspace_slug=slug, force=True)

        # Use the broadest supported retrieval setting. RAG is for relationship
        # discovery here, not for hard coverage; the original source analyses remain
        # separately available to the final synthesis and in the deterministic appendix.
        combined_rag_topn = _safe_int(os.getenv("AI_SUMMARY_COMBINED_RAG_TOPN"))
        if combined_rag_topn is None or combined_rag_topn <= 0:
            combined_rag_topn = DEFAULT_RAG_TOP_N_MAX
        combined_rag_topn = max(24, min(DEFAULT_RAG_TOP_N_MAX, combined_rag_topn))
        set_workspace_topn(slug, top_n=combined_rag_topn, similarity_threshold=0.0)
        time.sleep(1.5)

        # The user's prompt is the authoritative task.  RAG must remain prompt-agnostic:
        # do not hard-code domain-specific retrieval questions (for example, troop movements).
        # The uploaded evidence is used to retrieve the source material most relevant to whatever
        # task the user requested, while the complete chunk analyses remain available separately
        # to the final synthesizer and are preserved deterministically in the report.
        rag_query = (
            "USER REQUEST:\n"
            f"{full_query}\n\n"
            f"{AI_ANALYSIS_QUALITY_GUARDRAILS}\n"
            f"LATEST SUPPLIED EVIDENCE DATE: {latest_evidence_date or 'not determinable from supplied data'}\n\n"
            "You are performing a PROMPT-DRIVEN CROSS-CHUNK EVIDENCE RETRIEVAL pass. "
            "The uploaded document contains the complete analyses from every source chunk. "
            "Retrieve evidence that is relevant to the USER REQUEST above, especially evidence "
            "from different chunks that can help answer the request, corroborate or qualify an "
            "answer, reveal related observations, changes, patterns, exceptions, contradictions, "
            "or complementary details. Do not impose a domain-specific task that the user did not "
            "request. For example, if the user asks only for a summary, retrieve evidence useful "
            "for that summary; if the user asks about movements, retrieve movement-related evidence; "
            "if the user asks about equipment, retrieve equipment-related evidence. Preserve the "
            "meaning and scope of the user's request.\n\n"
            "This retrieval result is SUPPLEMENTARY evidence, not a complete inventory of findings. "
            "Do not intentionally summarize away source details merely because they are less prominent. "
            "When returning evidence, include the source chunk number and record range when available. "
            "Prefer concrete source-supported details over vague themes. Do not invent facts, infer "
            "unsupported information, or answer a different question. Do not mention embeddings, "
            "retrieval internals, workspaces, or implementation details.\n\n"
            "Return concise HTML using only h2, p, ul, ol, li, strong and b tags."
        )
        print(
            f"[INFO] Combined RAG enrichment query: evidence~{evidence_tokens} tokens, "
            f"topN={combined_rag_topn}"
        )
        enrichment = query_anythingllm_rag_workspace(rag_query, model_name, slug)
        if enrichment:
            print(f"[INFO] Combined RAG enrichment returned {len(enrichment)} chars")
            return enrichment
        print("[WARN] Combined RAG enrichment returned no usable response")
        return ""
    finally:
        delete_anythingllm_workspace(slug)




def sanitize_ai_report_text(text: str) -> str:
    """Normalize report-facing wording without changing source data.

    The report should consistently use 'analysis'/'aggregation' terminology.
    This is applied only to generated AI prose, never to the underlying ES
    records or citation table values.
    """
    if not text:
        return text
    replacements = [
        (r"(?i)\bSummaries\b", "Analyses"),
        (r"(?i)\bSummary\b", "Analysis"),
        (r"(?i)\bsummarize\b", "analyze"),
        (r"(?i)\bsummarises\b", "analyzes"),
        (r"(?i)\bsummarises\b", "analyzes"),
        (r"(?i)\bsummarizing\b", "analyzing"),
        (r"(?i)\bsummarisation\b", "analysis"),
        (r"(?i)\bsummarization\b", "analysis"),
        (r"(?i)\bCombined\b", "Aggregated"),
        (r"(?i)\bcombination\b", "aggregation"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def build_overall_source_document(e_hits: list) -> str:
    """Build a citation-numbered, full-source evidence document for Overall Analysis.

    The overall model should reason from the original ES records whenever the
    dataset fits in the declared context window. This avoids a second lossy
    compression step (chunk analysis -> overall analysis) and gives the overall
    analyst the richest available source representation.

    Technical/internal identifiers are intentionally excluded from the prompt;
    the stable Citation No. is the record-level reference used by the report.
    """
    if not e_hits:
        return ""

    def _render_value(value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)

    blocks = []
    for citation_no, hit in enumerate(e_hits, 1):
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        if not isinstance(source, dict):
            source = {}

        lines = [f"===== RECORD / CITATION NO. {citation_no} ====="]
        # Keep all substantive source fields. Hide only internal transport / UI
        # fields that are not useful for analysis and should not become report text.
        hidden = {
            "_id", "id", "es_id", "elastic_doc_id", "_index", "_score",
            "file_http_path", "ingester_name", "injester_name"
        }
        for field in sorted(source.keys()):
            if field in hidden:
                continue
            value = _render_value(source.get(field))
            if value == "":
                continue
            lines.append(f"{field}: {value}")
        lines.append(f"===== END RECORD / CITATION NO. {citation_no} =====")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)



def repair_overall_analysis_coverage(chunk_responses: list, overall_draft: str,
                                    full_query: str, model_name: str,
                                    llm_context_tokens: int) -> str:
    """Run a per-chunk coverage pass and return only source findings missing from the draft.

    Each chunk gets an independent comparison against the draft. This is a
    coverage-oriented verification step: it does not rewrite or summarize the
    draft. Any material findings absent from the draft are returned as detailed
    additions with record-level citations. The additions are appended to the
    Overall Analysis by the caller.
    """
    if not chunk_responses or not overall_draft:
        return ""

    additions = []
    for idx, (start_rec, end_rec, html_text) in enumerate(chunk_responses, 1):
        if not html_text or not html_text.strip():
            continue
        chunk_text = html_to_plain_text(_strip_wrappers(html_text)).strip()
        audit_prompt = f"""
You are performing a source-coverage verification for an Overall Analysis.

USER REQUEST:
{full_query}

CURRENT OVERALL ANALYSIS:
{html_to_plain_text(_strip_wrappers(overall_draft)).strip()}

SOURCE CHUNK {idx} (RECORDS {start_rec}-{end_rec}):
{chunk_text}

Task:
Compare the current Overall Analysis against EVERY material finding in this
source chunk. Identify any material, task-relevant information from this
chunk that is missing or materially underrepresented in the Overall Analysis.

Rules:
- Check the whole source chunk, not just its main theme.
- Preserve unique, one-off, negative, unusual, and less prominent findings.
- Do not report information already adequately represented.
- Do not invent or infer unsupported facts.
- If nothing material is missing, return exactly: NONE
- Otherwise return ONLY the missing findings as detailed HTML paragraphs or
  bullet points, with supporting Citation No. values in parentheses.
- Do not use the words "summary" or "combined".
- Do not mention this verification task, chunking, RAG, embeddings, or workflow.
""".strip()

        prompt_tokens = _combined_estimate_tokens(audit_prompt)
        available = int(llm_context_tokens - prompt_tokens - 512)
        if available < 512:
            print(f"[WARN] Overall coverage check {idx}/{len(chunk_responses)} skipped: prompt too large")
            continue
        max_output = min(3072, available)
        try:
            result = _get_direct_ollama_response(
                audit_prompt,
                model_name,
                context_tokens=llm_context_tokens,
                stage_label=f"overall-coverage-check-{idx}",
                max_output_tokens=max_output,
            ) or ""
        except Exception as exc:
            print(f"[WARN] Overall coverage check {idx}/{len(chunk_responses)} failed: {exc}")
            result = ""

        clean = sanitize_ai_report_text(_strip_wrappers(result).strip()) if result else ""
        if clean and clean.strip().upper() != "NONE" and len(clean) > 20:
            additions.append(f"<h3>Additional Detailed Findings — Source Chunk {idx}</h3>{clean}")
        print(
            f"[INFO] Overall coverage check {idx}/{len(chunk_responses)}: "
            f"{'missing findings identified' if clean and clean.strip().upper() != 'NONE' else 'no additional material findings'}"
        )

    if not additions:
        print("[INFO] Overall coverage repair: no additional material findings identified")
        return ""
    repaired = "<h2>Additional Detailed Findings</h2>" + "".join(additions)
    print(f"[INFO] Overall coverage repair: {len(additions)} source-chunk additions appended")
    return repaired


def _overall_dimension_value(source: dict, fields: tuple) -> list:
    """Return normalized non-empty values from a set of candidate fields."""
    values = []
    if not isinstance(source, dict):
        return values
    for field in fields:
        value = source.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        for item in items:
            if isinstance(item, dict):
                try:
                    item = json.dumps(item, ensure_ascii=False, sort_keys=True)
                except Exception:
                    item = str(item)
            text = str(item).strip()
            if text:
                values.append(text)
    return values


def _record_matches_dimension(source: dict, dimension: dict) -> bool:
    """Decide whether a source record belongs to an Overall-analysis dimension."""
    if not isinstance(source, dict):
        return False

    lower_keys = {str(k).lower(): k for k in source.keys()}
    exact_fields = dimension.get("fields", ())
    text_fields = dimension.get("text_fields", ())

    # Explicit fields are the primary match.
    for field in exact_fields:
        if field in source and source.get(field) not in (None, "", [], {}):
            return True

    # Flexible schema support: match populated field names by semantic tokens.
    tokens = dimension.get("key_tokens", ())
    for lk, original in lower_keys.items():
        if not any(token in lk for token in tokens):
            continue
        value = source.get(original)
        if value not in (None, "", [], {}):
            return True

    # For description-driven dimensions, only classify when relevant keywords
    # occur in the description or declared activity/sub-activity values.
    keywords = dimension.get("keywords", ())
    if keywords:
        search_parts = []
        for field in text_fields:
            if field in source and source.get(field) not in (None, "", [], {}):
                search_parts.append(str(source.get(field)))
        if not search_parts:
            # Common free-text fields as a fallback.
            for field in ("description", "comments", "activity_type", "sub_activity_type", "category", "purpose"):
                if field in source and source.get(field) not in (None, "", [], {}):
                    search_parts.append(str(source.get(field)))
        blob = " ".join(search_parts).lower()
        if any(keyword in blob for keyword in keywords):
            return True

    return False


def _build_dimension_groups(e_hits: list) -> list:
    """Build deterministic, record-backed analytical dimensions and value groups.

    Every group stores original Elasticsearch records plus stable Citation No.
    values. Records may belong to more than one dimension; this is intentional
    because a single observation can contain a location, equipment, training,
    movement and visit simultaneously.
    """
    dimensions = [
        {
            "name": "Location Analysis",
            "fields": ("location_name", "base_location_name", "start_location_name", "end_location_name", "pass_name", "visit_name", "comd_tps_loc_name"),
            "key_tokens": ("location_name", "base_location", "start_location", "end_location", "pass_name", "visit_name"),
            "group_fields": ("location_name", "base_location_name", "start_location_name", "end_location_name", "pass_name", "visit_name", "comd_tps_loc_name"),
        },
        {
            "name": "Force Movement and Deployment Analysis",
            "fields": ("activity_type", "movement_type", "deployment_type", "start_location_name", "end_location_name", "enemy_formation_name", "army_name", "force_type_name", "formation_type", "command_name", "orbate_title", "mobile_no"),
            "key_tokens": ("movement", "deployment", "formation", "force_type", "army_name", "command_name", "orbat", "mobile_no", "unit"),
            "group_fields": ("activity_type", "movement_type", "deployment_type", "enemy_formation_name", "army_name", "force_type_name", "formation_type", "command_name", "orbate_title"),
            "keywords": ("movement", "moved", "deployed", "deployment", "relocated", "transferred", "positioned", "convoy", "troop", "force"),
            "text_fields": ("description", "comments", "activity_type", "sub_activity_type"),
        },
        {
            "name": "Equipment and Vehicle Analysis",
            "fields": ("equipment_name", "equipment_type", "equipement_type", "vehicle_type", "vehicle_name", "mobile_no", "count", "weapon_system", "radar_name", "radar_type", "airfield_type"),
            "key_tokens": ("equipment", "equipement", "vehicle", "weapon", "radar", "mobile", "aircraft", "tank", "artillery", "drone", "missile", "system"),
            "group_fields": ("equipment_name", "equipment_type", "equipement_type", "vehicle_type", "vehicle_name", "weapon_system", "radar_name", "radar_type", "airfield_type"),
            "keywords": ("equipment", "vehicle", "drone", "aircraft", "tank", "artillery", "howitzer", "missile", "radar", "weapon", "system"),
            "text_fields": ("description", "comments", "activity_type", "category"),
        },
        {
            "name": "Infrastructure Analysis",
            "fields": ("infra_name", "infra_type", "airfield_type", "infrastructure_type", "project_type", "coordinates"),
            "key_tokens": ("infra", "infrastructure", "airfield", "facility", "construction", "building", "bridge", "road", "railway"),
            "group_fields": ("infra_name", "infra_type", "airfield_type", "infrastructure_type", "project_type"),
            "keywords": ("infrastructure", "construction", "bridge", "road", "building", "facility", "airfield", "railway", "solar", "camp"),
            "text_fields": ("description", "comments", "activity_type", "sub_activity_type"),
        },
        {
            "name": "Training and Readiness Analysis",
            "fields": ("training_type", "training_area", "exercise_type", "readiness_type", "activity_type", "sub_activity_type"),
            "key_tokens": ("training", "exercise", "readiness", "drill", "practice", "familiarisation", "familiarization"),
            "group_fields": ("training_type", "training_area", "exercise_type", "readiness_type", "activity_type", "sub_activity_type"),
            "keywords": ("training", "exercise", "readiness", "drill", "practice", "familiarisation", "familiarization", "training exercise"),
            "text_fields": ("description", "comments", "activity_type", "sub_activity_type"),
        },
        {
            "name": "Visits and Inspections Analysis",
            "fields": ("visit_name", "purpose", "visit_type", "inspection_type"),
            "key_tokens": ("visit", "inspection", "review", "assessment", "meeting"),
            "group_fields": ("visit_name", "purpose", "visit_type", "inspection_type"),
            "keywords": ("visit", "visited", "inspection", "review", "assessment", "meeting", "tour"),
            "text_fields": ("description", "comments", "activity_type", "purpose"),
        },
        {
            "name": "Force Composition, Strength and Capability Analysis",
            "fields": ("force_type_name", "formation_type", "army_name", "command_name", "orbate_title", "count", "strength", "personnel_count", "capability", "capabilities"),
            "key_tokens": ("force", "formation", "army", "command", "orbat", "strength", "personnel", "capabilit", "composition", "count"),
            "group_fields": ("force_type_name", "formation_type", "army_name", "command_name", "orbate_title", "strength", "personnel_count", "capability", "capabilities"),
            "keywords": ("force composition", "strength", "capability", "personnel", "formation", "force size", "troops"),
            "text_fields": ("description", "comments", "activity_type", "category"),
        },
        {
            "name": "Events, Security and Other Relevant Analysis",
            "fields": ("event_type", "category", "incident_type", "casualties", "civilian_casualties", "security_forces_casualties", "terrorists_casualties", "political_event", "security_event"),
            "key_tokens": ("event", "incident", "casualt", "security", "terror", "political", "conflict", "attack", "protest"),
            "group_fields": ("event_type", "category", "incident_type", "security_event", "political_event"),
            "keywords": ("event", "incident", "security", "attack", "conflict", "protest", "casualties", "terrorist", "political"),
            "text_fields": ("description", "comments", "activity_type", "category"),
        },
    ]

    for dimension in dimensions:
        groups = {}
        unmatched = []
        for citation_no, hit in enumerate(e_hits, 1):
            source = hit.get("_source", {}) if isinstance(hit, dict) else {}
            if not isinstance(source, dict):
                source = {}
            if not _record_matches_dimension(source, dimension):
                continue
            values = _overall_dimension_value(source, tuple(dimension.get("group_fields", ())))
            if not values:
                values = ["All relevant records"]
            normalized_values = []
            for value in values:
                key = re.sub(r"\s+", " ", value.strip()).lower()
                if key:
                    normalized_values.append((key, value.strip()))
            if not normalized_values:
                unmatched.append(citation_no)
                continue
            for key, display in normalized_values:
                if key not in groups:
                    groups[key] = {"label": display, "citations": [], "records": []}
                groups[key]["citations"].append(citation_no)
                groups[key]["records"].append((citation_no, hit))

        # De-duplicate a record's contribution within a value group.
        for group in groups.values():
            seen = set()
            records = []
            for citation_no, hit in group["records"]:
                if citation_no in seen:
                    continue
                seen.add(citation_no)
                records.append((citation_no, hit))
            group["records"] = records
            group["citations"] = sorted(seen)
        dimension["groups"] = sorted(groups.values(), key=lambda g: g["label"].lower())
    return dimensions


def _render_dimension_records(records: list, include_all_fields: bool = False) -> str:
    """Render source records for a dimension, preserving citation numbers."""
    if not records:
        return ""
    blocks = []
    hidden = {"_id", "id", "es_id", "elastic_doc_id", "_index", "_score", "file_http_path", "ingester_name", "injester_name"}
    preferred = (
        "activity_date", "location_name", "base_location_name", "start_location_name", "end_location_name",
        "enemy_formation_name", "army_name", "force_type_name", "formation_type", "command_name", "orbate_title",
        "activity_type", "sub_activity_type", "visit_name", "purpose", "equipment_name", "equipment_type",
        "equipement_type", "vehicle_type", "vehicle_name", "infra_name", "infra_type", "airfield_type",
        "training_type", "training_area", "exercise_type", "readiness_type", "strength", "personnel_count",
        "capability", "capabilities", "count", "description", "comments", "coordinates", "category"
    )
    for citation_no, hit in records:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        if not isinstance(source, dict):
            source = {}
        lines = [f"### Citation No. {citation_no}"]
        keys = list(source.keys()) if include_all_fields else [f for f in preferred if f in source]
        seen = set()
        for field in keys:
            if field in seen or field in hidden:
                continue
            seen.add(field)
            value = source.get(field)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list, tuple, set)):
                try:
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                except Exception:
                    value = str(value)
            lines.append(f"- {field}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _batch_dimension_groups(groups: list, context_limit_tokens: int, fixed_prompt_tokens: int,
                            per_group_min_tokens: int = 700) -> list:
    """Pack value-groups into LLM-sized batches without dropping any group."""
    batches = []
    current = []
    current_tokens = fixed_prompt_tokens
    for group in groups:
        group_text = _render_dimension_records(group["records"])
        group_tokens = max(1, _combined_estimate_tokens(group_text))
        if current and current_tokens + group_tokens > context_limit_tokens:
            batches.append(current)
            current = []
            current_tokens = fixed_prompt_tokens
        # A single large value-group remains intact and is handled by the
        # caller with a reduced document representation only if necessary.
        current.append(group)
        current_tokens += group_tokens
    if current:
        batches.append(current)
    return batches


def _analyze_overall_dimension_batch(dimension_name: str, groups: list, full_query: str,
                                     model_name: str, llm_context_tokens: int,
                                     latest_evidence_date: Optional[str] = None) -> str:
    """Analyze one field-wise group batch directly from original ES records."""
    if not groups:
        return ""

    group_blocks = []
    citations_all = []
    for group in groups:
        citations_all.extend(group["citations"])
        group_blocks.append(
            f"\n===== {dimension_name.upper()} VALUE: {group['label']} =====\n"
            f"{_render_dimension_records(group['records'])}\n"
            f"===== END VALUE: {group['label']} =====\n"
        )
    evidence = "\n".join(group_blocks).strip()

    prompt = f"""
You are producing one section of an Overall Analysis directly from the
original source records.

USER REQUEST:
{full_query}

ANALYTICAL DIMENSION:
{dimension_name}

TASK:
Analyze and aggregate the supplied source records for this dimension while
following the USER REQUEST exactly. Cover every supplied value/group and every
source record that contributes to this dimension. Describe concrete details,
relationships, changes, recurring observations, and other relevant findings.
Do not treat this as a short overview. Preserve all material source details.
If the same entity or activity occurs in several records, aggregate the
observations while retaining distinct dates, locations, units, equipment,
vehicles, quantities and context.

COVERAGE:
Every supplied Citation No. must be considered. Do not omit a record because
it is less prominent, unique, repeated, or appears in only one value/group.
Do not invent information or merge distinct facts into a vague statement.

CITATIONS:
End each distinct analytical point or paragraph with the supporting Citation
No. values in parentheses, for example (3, 45, 50). Use only citation numbers
actually present in the supplied evidence.

STYLE:
Produce detailed analysis, not a short overview. Use headings by value/group
when useful. Follow the user's requested subject and scope. Clearly separate
source-supported observations from analytical assessment when interpretation
is necessary. Do not mention workflow, chunking, retrieval, embeddings, or
implementation details. Do not use the words "summary" or "combined" in report
text. Return HTML using only h2, h3, p, ul, ol, li, strong and b tags.

LATEST EVIDENCE DATE:
{latest_evidence_date or 'not determinable from supplied data'}

SOURCE RECORDS:
{evidence}
""".strip()

    prompt_tokens = _combined_estimate_tokens(prompt)
    available = int(llm_context_tokens - prompt_tokens - 512)
    if available < 768:
        return ""
    max_output = min(MAX_OVERALL_OLLAMA_OUTPUT_TOKENS, max(1024, available))
    print(
        f"[INFO] Overall {dimension_name}: {len(groups)} grouped values, "
        f"{len(sorted(set(citations_all)))} source citations; prompt~{prompt_tokens}; "
        f"output_budget={max_output}"
    )

    response = ""
    try:
        response = _get_direct_ollama_response(
            prompt, model_name, context_tokens=llm_context_tokens,
            stage_label=f"overall-{re.sub(r'[^a-z0-9]+', '-', dimension_name.lower()).strip('-')}",
            max_output_tokens=max_output,
        ) or ""
    except Exception as exc:
        print(f"[WARN] Overall {dimension_name}: direct Ollama failed: {exc}")
    if not response:
        try:
            response = get_llm_response_direct(prompt, model_name, max_tokens=max_output) or ""
        except Exception as exc:
            print(f"[WARN] Overall {dimension_name}: AnythingLLM fallback failed: {exc}")
    return sanitize_ai_report_text(_strip_wrappers(response).strip()) if response else ""


def fieldwise_overall_analysis(e_hits: list, full_query: str, model_name: str,
                               llm_context_tokens: int,
                               latest_evidence_date: Optional[str] = None) -> str:
    """Build Overall Analysis as concatenated field-wise analyses from original ES records.

    This deliberately does not analyze chunk responses. The original records
    retained for the report are grouped deterministically by analytical
    dimensions, each dimension is analyzed independently, and the resulting
    sections are concatenated in a stable order. There is no final second-pass
    model call that can compress or discard earlier dimension analyses.
    """
    if not e_hits:
        return ""

    dimensions = _build_dimension_groups(e_hits)
    section_order = (
        "Location Analysis",
        "Force Movement and Deployment Analysis",
        "Equipment and Vehicle Analysis",
        "Infrastructure Analysis",
        "Training and Readiness Analysis",
        "Visits and Inspections Analysis",
        "Force Composition, Strength and Capability Analysis",
        "Events, Security and Other Relevant Analysis",
    )
    by_name = {d["name"]: d for d in dimensions}
    sections = []
    total_citations = set()

    for name in section_order:
        dimension = by_name.get(name)
        if not dimension or not dimension.get("groups"):
            print(f"[INFO] Overall {name}: no relevant source records; skipped")
            continue

        # Determine how many grouped values can fit per call. We batch by value,
        # not by chunks, so a single record can contribute to multiple dimensions.
        batches = _batch_dimension_groups(
            dimension["groups"],
            context_limit_tokens=max(2048, int(llm_context_tokens * 0.72)),
            fixed_prompt_tokens=950,
        )
        dimension_results = []
        for batch_idx, batch in enumerate(batches, 1):
            batch_name = name if len(batches) == 1 else f"{name} — Part {batch_idx} of {len(batches)}"
            result = _analyze_overall_dimension_batch(
                batch_name, batch, full_query, model_name, llm_context_tokens, latest_evidence_date
            )
            batch_citations = sorted({citation for group in batch for citation in group["citations"]})
            if result:
                citation_tail = "(" + ", ".join(str(n) for n in batch_citations) + ")" if batch_citations else ""
                if citation_tail and "<p><strong>Citations:</strong>" not in result:
                    result = result + f"<p><strong>Citations:</strong> {citation_tail}</p>"
                dimension_results.append(result)
                total_citations.update(batch_citations)

        if dimension_results:
            sections.append(f"<h2>{name}</h2>" + "".join(f"<div>{r}</div>" for r in dimension_results))

    # Ensure records which did not land in any semantic dimension still receive
    # analytical attention rather than disappearing silently.
    matched = set()
    for dimension in dimensions:
        for group in dimension.get("groups", []):
            matched.update(group.get("citations", []))
    unmatched_citations = sorted(set(range(1, len(e_hits) + 1)) - matched)
    if unmatched_citations:
        fallback_group = {"label": "Other Relevant Records", "citations": unmatched_citations,
                          "records": [(n, e_hits[n - 1]) for n in unmatched_citations]}
        result = _analyze_overall_dimension_batch(
            "Other Relevant Records", [fallback_group], full_query, model_name,
            llm_context_tokens, latest_evidence_date
        )
        if result:
            sections.append(f"<h2>Other Relevant Analysis</h2><div>{result}</div>")
            total_citations.update(unmatched_citations)
        else:
            print(f"[WARN] Overall Other Relevant Records analysis returned no usable response for {len(unmatched_citations)} records")

    print(
        f"[INFO] Overall field-wise analysis completed: {len(sections)} sections; "
        f"{len(total_citations)}/{len(e_hits)} source citations represented in field analyses"
    )
    if not sections:
        return ""

    return "<div>" + "".join(sections) + "</div>"


def synthesize_combined_analysis(chunk_responses, full_query, model_name,
                                 token_budget, total_rows, club_size,
                                 llm_context_tokens, latest_evidence_date=None,
                                 max_levels=None, output_tokens=None,
                                 raw_evidence_document: str = "",
                                 raw_evidence_hits: Optional[list] = None) -> str:
    """Backward-compatible entry point for the Overall Analysis.

    The Overall Analysis is now field-wise and record-driven. It analyzes the
    original Elasticsearch records retained for the report, not the chunk
    responses. Each analytical dimension is generated independently and the
    resulting sections are concatenated without a final compression pass.
    """
    if raw_evidence_hits:
        return fieldwise_overall_analysis(
            raw_evidence_hits, full_query, model_name, llm_context_tokens,
            latest_evidence_date=latest_evidence_date
        )

    # Compatibility fallback for callers that do not have the original hit
    # list: retain the previous direct-source behavior rather than failing.
    if raw_evidence_document:
        source_material = raw_evidence_document
    elif chunk_responses:
        source_blocks = []
        for idx, (start_rec, end_rec, html_text) in enumerate(chunk_responses, 1):
            if not html_text or not html_text.strip():
                continue
            source_blocks.append(
                f"\n===== SOURCE CHUNK {idx} | RECORDS {start_rec}-{end_rec} =====\n"
                f"{html_to_plain_text(_strip_wrappers(html_text)).strip()}\n"
                f"===== END SOURCE CHUNK {idx} =====\n"
            )
        source_material = "\n".join(source_blocks).strip()
    else:
        return ""

    if not source_material:
        return ""

    prompt = f"""
USER REQUEST:
{full_query}

Analyze all supplied information in detail according to the user's request.
Preserve material source details and cite each analytical point with the
provided Citation No. values. Return detailed analysis rather than a short
overview. Do not invent unsupported facts. Do not use the words "summary" or
"combined" in report text.

SOURCE MATERIAL:
{source_material}
""".strip()
    prompt_tokens = _combined_estimate_tokens(prompt)
    available = int(llm_context_tokens - prompt_tokens - 512)
    if available < 768:
        return ""
    max_output = min(MAX_OVERALL_OLLAMA_OUTPUT_TOKENS, max(1024, available))
    try:
        response = _get_direct_ollama_response(
            prompt, model_name, context_tokens=llm_context_tokens,
            stage_label="overall-analysis-fallback", max_output_tokens=max_output
        ) or ""
    except Exception:
        response = ""
    if not response:
        try:
            response = get_llm_response_direct(prompt, model_name, max_tokens=max_output) or ""
        except Exception:
            response = ""
    return sanitize_ai_report_text(_strip_wrappers(response).strip()) if response else ""


def _keep_latest_by_form_id(hits: list) -> list:
    """For General Analysis, keep only the latest record per form_id."""

    latest_by_form_id = {}
    records_without_form_id = []

    def _activity_date_key(hit):
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        value = source.get("activity_date") or source.get("@timestamp") or ""

        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

            # Treat timezone-less timestamps as UTC.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return (1, dt.timestamp())

        except (ValueError, TypeError, OSError):
            # Invalid/missing dates sort below valid dates.
            return (0, 0)

    # Collect latest record plus historical equipment/infra values per form_id.
    equipment_by_form_id = {}
    infra_by_form_id = {}

    for hit in hits:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        form_id = source.get("form_id")

        if form_id in (None, ""):
            records_without_form_id.append(hit)
            continue

        key = str(form_id)

        # Collect equipment_name from every record belonging to this form_id.
        eq_name = (
            source.get("equipment_name")
            or source.get("equipment.equipment_name")
        )

        # Collect infra_name from every record belonging to this form_id.
        infra_name = (
            source.get("infra_name")
            or source.get("infra.infra_name")
        )

        if eq_name not in (None, "", "None"):
            equipment_by_form_id.setdefault(key, set()).add(str(eq_name).strip())

        if infra_name not in (None, "", "None"):
            infra_by_form_id.setdefault(key, set()).add(str(infra_name).strip())

        # Keep only the latest record as the base document.
        current = latest_by_form_id.get(key)

        if current is None or _activity_date_key(hit) > _activity_date_key(current):
            latest_by_form_id[key] = hit

    # Build final result using the latest document for each form_id.
    result = []

    for key, latest_hit in latest_by_form_id.items():
        # Do not modify the original ES hit.
        final_hit = copy.deepcopy(latest_hit)
        source = final_hit.get("_source", {})

        # Merge historical + latest equipment values.
        equipment_values = equipment_by_form_id.get(key, set())
        if equipment_values:
            source["equipment_name"] = ", ".join(sorted(equipment_values))

        # Merge historical + latest infra values.
        infra_values = infra_by_form_id.get(key, set())
        if infra_values:
            source["infra_name"] = ", ".join(sorted(infra_values))

        result.append(final_hit)

    result.extend(records_without_form_id)

    print(
        f"[INFO] General form_id grouping: "
        f"{len(hits)} → {len(result)} records"
    )

    return result

# ──────────────────────────────────────────────
# ⭐ Modified: ai_analysis_summary_check with Query Mode
# ──────────────────────────────────────────────
def ai_analysis_summary_check(row, ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                              chunk_token_threshold=3000, chunk_size=2200,
                              chunk_output_tokens=800, consolidation_output_tokens=1500,
                              llm_context_tokens=None, chunk_workers_arg=None):
    elastic_query, ai_analysis_summary_id, search_form_type = row
    search_form_type = search_form_type.lower()
    model_prompt_dict = None
    if search_form_type != 'profile analysis':
        model_prompt_dict = get_prompt_and_model(cursor, ai_analysis_summary_id)

        is_general_analysis = (
            search_form_type in [t.lower() for t in TYPE_MAPPING["general"]["types"]]
        )

        e_response = get_data_from_elastic(
            elastic_query,
            perform_dedup=not is_general_analysis
        ) if elastic_query else None

        if e_response and e_response['hits']['total']['value'] > 0:
            e_hits = e_response['hits']['hits']
        else:
            e_hits = []

        dedup_stats = e_response.get("dedup_stats", {}) if e_response else {}

        # General Analysis:
        # 1. Group by form_id.
        # 2. Keep latest activity_date.
        # 3. Merge historical equipment/infra values.
        # 4. ONLY THEN apply the normal deduplication.
        if is_general_analysis:
            e_hits = _keep_latest_by_form_id(e_hits)

            e_hits = _dedup_hits(
                e_hits,
                [
                    "description_hash",
                    "comments",
                    "location_name",
                    "activity_type",
                    "activity_date",
                    "enemy_formation_name",
                ]
            )

            # General analysis intentionally skips ES-level deduplication and
            # performs form_id grouping followed by general deduplication here.
            # Refresh the report statistics from the final retained records so
            # the generated report reflects the actual General Analysis result.
            raw_general_total = (
                e_response.get("dedup_stats", {}).get(
                    "total",
                    e_response.get("hits", {}).get("total", {}).get("value", len(e_hits))
                )
                if e_response
                else len(e_hits)
            )
            dedup_stats["total"] = raw_general_total
            dedup_stats["unique"] = len(e_hits)
            dedup_stats["dropped"] = raw_general_total - len(e_hits)

    ai_change_repsonse = ''
    ai_trend_response = ''
    llm_response = ''

    print(f"-- {search_form_type} --")

    # ── Change Detection (unchanged from v3.0.0) ──
    if ai_change:
        if e_hits:
            trend_start_time = time.time()
            start_date = find_lt_value(elastic_query)
            if search_form_type.lower() in [x.lower() for x in
                ["SAM Deployment Analysis", "Training Areas Analysis", "Infra Development Analysis",
                 "AIR Aspects Analysis", "PLA Sitrep Analysis", "Force Disposition Analysis"]]:
                _change_llm_model = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
                print(f"[INFO] Running change detection for {search_form_type} [{ai_analysis_summary_id}] "
                      f"using model: {_change_llm_model}")
                ai_change_repsonse = run_change_detection(elastic_query, start_date=start_date,
                                                          llm_model_name=_change_llm_model)

    if ai_change_previous:
        start_date, end_date = extract_date_range(elastic_query)
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            _llm_model = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
            print(f"[INFO] Running multi-year comparison ({ai_change_previous} year(s)) for "
                  f"{search_form_type} [{ai_analysis_summary_id}] using model: {_llm_model}")
            raw_change_output = run_multi_year_comparison(
                elastic_query, start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"), years_back=ai_change_previous,
                llm_model_name=_llm_model)
            ai_change_repsonse = raw_change_output

    # ── Trends (unchanged from v3.0.0) ──
    if ai_trends:
        if e_hits:
            trend_start_time = time.time()
            start_date = find_lt_value(elastic_query)

            if search_form_type == "sam deployment analysis":
                ai_trend_response = trends_sam_main(start_date, elastic_query)
            elif search_form_type == "training areas analysis":
                ai_trend_response = training_main(start_date, elastic_query)
            elif search_form_type == "infra development analysis":
                ai_trend_response = infra_main(start_date, elastic_query)
            elif search_form_type == "air aspects analysis":
                ai_trend_response = trends_airfield_main(start_date, elastic_query)
            elif search_form_type == "pla sitrep analysis":
                ai_trend_response = sitrep_main(start_date, elastic_query)
            elif search_form_type == "force disposition analysis":
                ai_trend_response = force_disposition_main(start_date, elastic_query)
            else:
                print(f"Trends not created for {search_form_type}")
                ai_trend_response = ""

            print(f"[DONE] Trend Analysis Complete for {search_form_type} [{ai_analysis_summary_id}] "
                  f"[time={(time.time() - trend_start_time):.2f} s]")

    # ── ⭐ AI Summary with Query Mode ──
    if ai_summ:
        ai_summ_start_time = time.time()

        if search_form_type == 'profile analysis':
            print(f"[INFO] Profile analysis for {ai_analysis_summary_id}")
            if elastic_query:
                person_names = []
                organization_names = []
                if 'query' in elastic_query and 'bool' in elastic_query['query'] and 'must' in elastic_query["query"]["bool"]:
                    for condition in elastic_query["query"]["bool"]["must"]:
                        if "match_phrase_prefix" in condition:
                            if "person_name" in condition["match_phrase_prefix"]:
                                person_names.append(condition["match_phrase_prefix"]["person_name"])
                            elif "civil_organization.civil_organization_name" in condition["match_phrase_prefix"]:
                                organization_names.append(condition["match_phrase_prefix"]["civil_organization.civil_organization_name"])
                            elif "civil_organization_name" in condition["match_phrase_prefix"]:
                                organization_names.append(condition["match_phrase_prefix"]["civil_organization_name"])

                if person_names or organization_names:
                    all_descriptions = []
                    index = 0
                    for single_json in e_hits:
                        single_json = single_json['_source']
                        if single_json.get("description"):
                            file_http_path = single_json.get('file_http_path', f'file{index}')
                            index += 1 if 'file_http_path' not in single_json else 0
                            raw_timestamp = single_json.get("@timestamp", "")
                            try:
                                activity_date = datetime.fromisoformat(
                                    raw_timestamp.replace("Z", "+00:00")
                                ).strftime("%Y-%m-%d")
                            except (ValueError, TypeError):
                                activity_date = ""
                            file_name = file_http_path.split('/')[-1]
                            complete_single_profile = {
                                "file_path": file_http_path, "file_name": file_name,
                                "date": str(activity_date), "description": single_json['description']
                            }
                            all_descriptions.append(complete_single_profile)

                    if all_descriptions:
                        df = pd.DataFrame(all_descriptions)
                        grouped = df.groupby("file_name").agg({
                            "file_path": lambda x: list(set(x)),
                            "description": lambda x: list(set(x)),
                            "date": lambda x: list(set(x))
                        }).reset_index()
                        complete_summary_dict_list = grouped.to_dict(orient="records")

                        llm_prompt = create_llm_prompt_for_profile(
                            complete_summary_dict_list, search_form_type, person_names, organization_names)
                        extra_prompt = "\n\n(PLEASE NOTE: The response you will give should be in html. "
                        extra_prompt += "Any title in the response should be in h2 tag and paragraph should be in p tag.)"
                        llm_prompt += extra_prompt

                        # For profile analysis, use direct chat mode (complex data not suitable for table upload)
                        temp_llm_response = get_llm_response_direct(llm_prompt, LLM_MODEL)
                        temp_llm_response = f"<div>{temp_llm_response}</div><br>"
                        llm_response += temp_llm_response

        # ── Standard Analysis Types ──
        for key, value in TYPE_MAPPING.items():
            if search_form_type in value["types"]:
                print(f"[INFO] AI analysis for {key} [{ai_analysis_summary_id}]")

                if not e_hits:
                    print(f"[WARN] No ES hits for {key}")
                    continue

                # Step 1: Convert ES hits to markdown
                markdown_data = convert_hits_to_markdown(e_hits, search_form_type)
                print(f"[INFO] Generated {len(markdown_data)} chars of markdown for {key}")

                # Step 2: Build the analysis prompt
                base_prompt = model_prompt_dict.get("prompt", "") if model_prompt_dict else ""
                extra_prompt = "\n\n(PLEASE NOTE: The response you will give should be in html. "
                extra_prompt += "Any title in the response should be in h2 tag and paragraph should be in p tag.)"
                full_query = base_prompt + extra_prompt

                model_name = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
                words, total_tokens = count_words_and_tokens(markdown_data + full_query)
                print(f"[INFO] Words: {words}, Approx Tokens: {total_tokens}")

                # Step 3: Analyse the dataset with per-chunk upload + RAG in
                # fresh AnythingLLM workspaces and then synthesise a Combined
                # Analysis. This path is used for EVERY dataset size: larger
                # ranges previously used a hierarchical + RAG branch. All standard
                # datasets now use the same complete-chunk direct-analysis path. (⭐ v3.1.1.4)
                print(f"[DEBUG] Model: {model_name}")

                # Scalable per-chunk analysis:
                #   * token-aware chunk boundaries instead of rows ~= 400 tokens
                #   * ONE upload per chunk, not one upload per row
                #   * bounded parallel workers for independent chunks
                #   * result order is restored before report/combined synthesis
                print(f"[DEBUG] Model: {model_name}")

                _row_override = _safe_int(os.getenv("AI_SUMMARY_CLUB_SIZE"))
                max_records_per_chunk = _row_override if (_row_override and _row_override > 0) else None
                if max_records_per_chunk:
                    club_reason = f"max {max_records_per_chunk} records/chunk override + token-aware sizing"
                else:
                    club_reason = "token-aware sizing"

                # Sort by activity_date DESC so the latest records come first in
                # each chunk and in the report. Missing dates sort last.
                def _activity_date_sort_key(hit):
                    src = hit.get("_source") if isinstance(hit, dict) else None
                    if not isinstance(src, dict):
                        return (1, "")
                    ad = src.get("activity_date") or src.get("@timestamp") or ""
                    if not ad:
                        return (1, "")
                    return (0, ad)

                e_hits_sorted = sorted(e_hits, key=_activity_date_sort_key, reverse=True)
                e_hits = e_hits_sorted
                n_rows = len(e_hits)

                # Compute the latest evidence date once and pass it into the
                # synthesis prompt. Forecasts must always refer to a period after
                # the latest supplied evidence, never a historical date range.
                latest_evidence_date = None
                _evidence_dates = []
                for _h in e_hits_sorted:
                    _src = _h.get("_source") if isinstance(_h, dict) else None
                    if isinstance(_src, dict):
                        _ad = _src.get("activity_date") or _src.get("@timestamp")
                        if _ad:
                            _evidence_dates.append(str(_ad)[:10])
                if _evidence_dates:
                    latest_evidence_date = max(_evidence_dates)

                # Token-aware chunk target. Keep chunks materially smaller than the
                # full 75%-of-context threshold so the COMPLETE chunk can be supplied
                # inline to the LLM without semantic-RAG retrieval hiding records.
                # Use it as the upper data-document target, but calculate actual
                # chunk boundaries from the real estimated size of each record.
                record_chunks = build_record_chunks(
                    e_hits_sorted,
                    start_row_idx=1,
                    max_tokens=chunk_size,
                    max_records=max_records_per_chunk,
                )
                n_chunks = len(record_chunks)

                print(
                    f"[INFO] Token-aware chunking: {n_rows} rows → {n_chunks} chunks "
                    f"({club_reason}, target ~{chunk_size} document tokens)"
                )

                chunk_workers = _resolve_chunk_workers(chunk_workers_arg)
                print(
                    f"[INFO] Chunk execution: {chunk_workers} worker(s) "
                    f"(set --chunk_workers N or AI_SUMMARY_CHUNK_WORKERS=N)"
                )

                chunk_responses_by_idx = {}
                chunk_loop_start = time.time()

                jobs = [
                    (
                        idx,
                        n_chunks,
                        chunk_info,
                        full_query,
                        model_name,
                        f"{key}_{ai_analysis_summary_id}",
                        latest_evidence_date,
                    )
                    for idx, chunk_info in enumerate(record_chunks, 1)
                ]

                # Attach the declared model context to each chunk job without
                # changing the public chunk metadata returned to the report.
                jobs = [
                    (
                        job[0],
                        job[1],
                        dict(
                            job[2],
                            context_tokens=llm_context_tokens,
                            output_tokens=chunk_output_tokens,
                        ),
                        job[3],
                        job[4],
                        job[5],
                        job[6],
                    )
                    for job in jobs
                ]

                if chunk_workers == 1 or n_chunks <= 1:
                    completed = 0
                    for job in jobs:
                        chunk_idx, chunk_info, response, elapsed = _process_chunk_job(job)
                        completed += 1
                        if response:
                            chunk_responses_by_idx[chunk_idx] = (
                                chunk_info["start"], chunk_info["end"], response
                            )
                        elapsed_total = time.time() - chunk_loop_start
                        avg = elapsed_total / completed
                        eta = avg * (n_chunks - completed)
                        print(
                            f"[CHUNK {chunk_idx}/{n_chunks}] done in {elapsed:.1f}s "
                            f"| total {elapsed_total:.1f}s | avg {avg:.1f}s "
                            f"| ETA {eta/60:.1f}m"
                        )
                else:
                    with ThreadPoolExecutor(max_workers=chunk_workers, thread_name_prefix="ai-chunk") as executor:
                        future_map = {
                            executor.submit(_process_chunk_job, job): job[0]
                            for job in jobs
                        }
                        completed = 0
                        for future in as_completed(future_map):
                            chunk_idx = future_map[future]
                            completed += 1
                            try:
                                result_idx, chunk_info, response, elapsed = future.result()
                                if response:
                                    chunk_responses_by_idx[result_idx] = (
                                        chunk_info["start"], chunk_info["end"], response
                                    )
                                elapsed_total = time.time() - chunk_loop_start
                                avg = elapsed_total / completed
                                eta = avg * (n_chunks - completed)
                                print(
                                    f"[CHUNK {result_idx}/{n_chunks}] done in {elapsed:.1f}s "
                                    f"| completed {completed}/{n_chunks} "
                                    f"| elapsed {elapsed_total:.1f}s | ETA {eta/60:.1f}m"
                                )
                            except Exception as exc:
                                print(f"[ERROR] Chunk {chunk_idx}/{n_chunks} worker failed: {exc}")
                                traceback.print_exc()

                # Restore original chunk order for the human-readable report and
                # deterministic coverage-preserving synthesis.
                chunk_responses = [
                    chunk_responses_by_idx[idx]
                    for idx in sorted(chunk_responses_by_idx)
                ]

                print(
                    f"[INFO] Chunk processing complete: {len(chunk_responses)}/{n_chunks} "
                    f"chunks returned usable responses in {time.time() - chunk_loop_start:.1f}s"
                )

                # Build Overall Analysis separately, followed by the chunk analyses in source order.
                if chunk_responses:
                    def _chunk_dates(rows):
                        dates = []
                        for h in rows:
                            src = h.get("_source") if isinstance(h, dict) else None
                            if isinstance(src, dict):
                                ad = src.get("activity_date") or src.get("@timestamp")
                                if ad:
                                    dates.append(str(ad)[:10])
                        return (min(dates), max(dates)) if dates else ("?", "?")

                    chunk_sections = []
                    for chunk_num, (chunk_start, chunk_end, chunk_html_text) in enumerate(chunk_responses, 1):
                        chunk_rows = e_hits_sorted[chunk_start - 1:chunk_end]
                        d_min, d_max = _chunk_dates(chunk_rows)
                        chunk_row_count = chunk_end - chunk_start + 1
                        cited_numbers = _extract_citation_numbers(chunk_html_text, chunk_start, chunk_end)
                        if not cited_numbers:
                            cited_numbers = list(range(chunk_start, chunk_end + 1))
                        citation_tail = "(" + ", ".join(str(n) for n in cited_numbers) + ")"
                        chunk_sections.append(
                            "<h2>Chunk Analysis "
                            f"{chunk_num} — Records {chunk_start} to {chunk_end} "
                            f"({chunk_row_count} records, {d_min} to {d_max})</h2>"
                            f"<div>{chunk_html_text}</div>"
                            f"<p><strong>Citations:</strong> {citation_tail}</p>"
                        )
                    chunk_html = "".join(chunk_sections)
                    rows_total = sum(end - start + 1 for start, end, _ in chunk_responses)
                    n_chunks_done = len(chunk_responses)
                    max_rows_per_chunk_actual = max((len(info["rows"]) for info in record_chunks), default=0)

                    overall_html = ""
                    try:
                        overall_token_budget = int(llm_context_tokens * 0.85)
                        overall_response = synthesize_combined_analysis(
                            chunk_responses, full_query=full_query, model_name=model_name,
                            token_budget=overall_token_budget, total_rows=rows_total,
                            club_size=max_rows_per_chunk_actual, llm_context_tokens=llm_context_tokens,
                            latest_evidence_date=latest_evidence_date,
                            output_tokens=consolidation_output_tokens,
                            raw_evidence_document=build_overall_source_document(e_hits),
                            raw_evidence_hits=e_hits,
                        )
                        if overall_response:
                            if is_general_analysis:
                                stats_text = (
                                    f"Elasticsearch returned {dedup_stats.get('total', rows_total)} raw records. "
                                    f"After latest-record grouping by form_id and general deduplication, "
                                    f"{rows_total} records were retained for analysis."
                                )
                            else:
                                stats_text = (
                                    f"Elasticsearch returned {dedup_stats.get('total', rows_total)} records, "
                                    f"of which {dedup_stats.get('unique', rows_total)} were unique after analysis filtering "
                                    f"and {dedup_stats.get('dropped', 0)} duplicates were removed."
                                )
                            overall_cited_numbers = _extract_citation_numbers(overall_response, 1, rows_total)
                            if not overall_cited_numbers:
                                overall_cited_numbers = sorted({
                                    n
                                    for _start, _end, _html in chunk_responses
                                    for n in _extract_citation_numbers(_html, _start, _end)
                                })
                            if not overall_cited_numbers:
                                overall_cited_numbers = list(range(1, rows_total + 1))
                            overall_citations = "(" + ", ".join(str(n) for n in overall_cited_numbers) + ")"
                            overall_html = (
                                "<h2>Overall Analysis</h2>"
                                "<p style=\"color:#6c757d;font-size:13px;\">"
                                f"Detailed analysis across all {n_chunks_done} chunks. {stats_text}"
                                "</p>"
                                f"<div>{overall_response}</div>"
                                f"<p><strong>Citations:</strong> {overall_citations}</p>"
                            )
                    except Exception as e:
                        print(f"[WARN] Overall analysis failed: {e}")

                    master = (
                        "<div>"
                        f"<p style=\"color:#6c757d;font-size:13px;\">"
                        f"{rows_total} records were analyzed across {n_chunks_done} chunks, with up to "
                        f"{max_rows_per_chunk_actual} records per chunk.</p>"
                        + (overall_html if overall_html else "<h2>Overall Analysis</h2><p>Analysis could not be generated.</p>")
                        + chunk_html
                        + "</div><br>"
                    )
                    temp_llm_response = master
                else:
                    temp_llm_response = (
                        "<div><h2>Analysis Unavailable</h2>"
                        "<p>AI returned no responses for this group of records.</p>"
                        "</div><br>"
                    )

                temp_llm_response = f"<div>{temp_llm_response}</div><br>"
                llm_response += temp_llm_response

        print(f"[DONE] AI analysis complete [{ai_analysis_summary_id}] "
              f"[time={(time.time() - ai_summ_start_time):.2f} s]")

    # ── Build HTML report ──
    # ⭐ NEW (v3.1.1.1): trailing citation table built from the same ES hits
    # that drove the AI analysis, so the report carries auditable source
    # metadata for every record referenced.
    #
    # ⭐ NEW (v3.1.1.2): pass the unfiltered ES total.value + the query's
    # @timestamp range so the citation table footer can show "Showing N of M
    # raw matches" and the date range — makes dedup and time-scope visible.
    query_total = dedup_stats.get("total") if dedup_stats else None
    query_note = None
    if is_general_analysis:
        query_note = (
            f"Showing {len(e_hits)} records retained from {query_total or len(e_hits)} raw ES matches "
            f"after latest-record grouping by form_id and general deduplication"
        )
    date_range = None
    if elastic_query:
        date_range = extract_date_range(elastic_query)
    doc_table_html = build_doc_id_activity_date_table(
        e_hits, query_total=query_total, date_range=date_range, query_note=query_note
    ) if e_hits else ""
    tabbed_html = build_tabbed_html(
        [llm_response, ai_trend_response, ai_change_repsonse],
        doc_table_html=doc_table_html,
    )
    ai_analysis_text_base64 = base64.b64encode(tabbed_html.encode("utf-8")).decode("utf-8")

    update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id, ai_analysis_text_base64)


# ──────────────────────────────────────────────
# IMINT Analysis (unchanged from v3.0.0)
# ──────────────────────────────────────────────
def imint_ai_analysis_check(row, cursor):
    elastic_query, imint_ai_analysis_id = row
    model_prompt_dict = get_prompt_and_model(cursor, imint_ai_analysis_id, imint=True)
    print("[INFO] for imint ai analysis", imint_ai_analysis_id)
    if elastic_query:
        comments = []
        e_response = get_data_from_elastic(elastic_query)
        if e_response and e_response['hits']['total']['value'] > 0:
            imint_results = e_response['hits']['hits']
            for single_json in imint_results:
                single_json = single_json['_source']
                if single_json.get("comments"):
                    location_name = single_json.get("location_name")
                    activity_date = single_json.get("activity_date", "")
                    print("['comments']", single_json.get("comments")[-1], location_name, activity_date)
                    complete_single_comment = "location_name: " + location_name + ", date:" + str(
                        activity_date) + ', comments: ' + single_json['comments'][-1].strip().replace('\n', '. ')
                    if complete_single_comment not in comments:
                        comments.append(complete_single_comment)
        comments = '\n\n'.join(comments)
        ai_analysis_text_base64 = ''
        if comments:
            llm_prompt = comments + "\n\n" + model_prompt_dict.get("prompt") + \
                         "The response you will give should be in html(hyper text markup language). " \
                         "Any title in the response should be in h2 tag and paragraph should be in p tag."
            llm_response = get_llm_response_direct(llm_prompt, model_prompt_dict.get("model_info"))
            corrected_result = correct_json_format(str(llm_response))
            ai_analysis_text = corrected_result.get("summary", "")
            print("[DONE] imint_ai_analysis_id: ", imint_ai_analysis_id)
            ai_analysis_text_base64 = str(base64.b64encode(str(ai_analysis_text).encode('utf-8')).decode('utf-8'))
        update_imint_ai_analysis(cursor, imint_ai_analysis_id, ai_analysis_text_base64)


# ──────────────────────────────────────────────
# Prompt and Model helpers (unchanged from v3.0.0)
# ──────────────────────────────────────────────
def get_prompt_and_model(cursor, primary_key, imint=False):
    output_dict = {}
    table_name = "imint_ai_analysis" if imint else "ai_analysis_summary"
    query = f"SELECT prompt, prompt_id, model_id FROM {table_name} WHERE {table_name}_id"
    query = query + " = %s;"
    cursor.execute(query, (primary_key,))
    result = cursor.fetchone()
    if result:
        prompt, prompt_id, model_id = result
    else:
        print("No result found for the provided id {}".format(primary_key))
        raise Exception("No result found for the provided id {}".format(primary_key))

    if prompt:
        output_dict["prompt"] = prompt
    elif prompt_id:
        prompt_subquery = "SELECT prompt FROM prompt_master WHERE prompt_id = %s;"
        cursor.execute(prompt_subquery, (prompt_id,))
        result = cursor.fetchone()
        if result:
            output_dict["prompt"] = result[0]
        else:
            raise Exception("Prompt not found for prompt_id {}! Please fix in the database!".format(prompt_id))
    else:
        raise Exception("Prompt ID is None! Please fix in the database!")

    if model_id:
        model_subquery = "SELECT ai_model_info FROM ai_model_master WHERE ai_model_id = %s;"
        cursor.execute(model_subquery, (model_id,))
        result = cursor.fetchone()
        if result:
            output_dict["model_info"] = result[0]
        else:
            raise Exception("Model Info not found for pmodel_id {}! Please fix in the database!".format(model_id))
    else:
        raise Exception("Model ID is None! Please fix in the database!")
    return output_dict


def find_lt_value(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'lt':
                try:
                    return datetime.fromisoformat(value).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            else:
                result = find_lt_value(value)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_lt_value(item)
            if result:
                return result
    return None


def extract_date_range(data):
    start = None
    end = None
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "gte":
                start = value
            elif key == "lt":
                end = value
            else:
                s, e = extract_date_range(value)
                start = start or s
                end = end or e
    elif isinstance(data, list):
        for item in data:
            s, e = extract_date_range(item)
            start = start or s
            end = end or e
    return start, end


# ──────────────────────────────────────────────
# Thread-safe wrappers (unchanged from v3.0.0)
# ──────────────────────────────────────────────
def ai_analysis_summary_check_threadsafe(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                                         chunk_token_threshold=3000, chunk_size=2200,
                                         chunk_output_tokens=800, consolidation_output_tokens=1500,
                                         llm_context_tokens=None, chunk_workers_arg=None):
    try:
        conn = postgres_connection()
        with conn.cursor() as cursor:
            filter_json, ai_analysis_summary_id, search_form_type = row
            ai_analysis_summary_check((filter_json, ai_analysis_summary_id, search_form_type),
                           ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                           chunk_token_threshold, chunk_size,
                           chunk_output_tokens, consolidation_output_tokens,
                           llm_context_tokens, chunk_workers_arg)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Error processing row {row}: {e}")
        traceback.print_exc()


def imint_ai_analysis_summary_check_threadsafe(row):
    try:
        conn = postgres_connection()
        with conn.cursor() as cursor:
            filter_json, imint_ai_analysis_id = row
            imint_ai_analysis_check((filter_json, imint_ai_analysis_id), cursor)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Error processing row {row}: {e}")
        traceback.print_exc()


def lock_rows_for_processing_1(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE public.ai_analysis_summary
                SET query_status = 2
                WHERE ai_analysis_summary_id IN (
                    SELECT ai_analysis_summary_id
                    FROM public.ai_analysis_summary
                    WHERE query_status = 0
                    ORDER BY ai_analysis_summary_id DESC
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING filter_json, ai_analysis_summary_id, search_form_type;
            """)
            return cursor.fetchall()
    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] ProgrammingError in lock_rows_for_processing_1: {e}")
        return []


def lock_rows_for_processing_2(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE public.imint_ai_analysis
                SET status = 2
                WHERE imint_ai_analysis_id IN (
                    SELECT imint_ai_analysis_id
                    FROM public.imint_ai_analysis
                    WHERE status = 0
                    ORDER BY imint_ai_analysis_id DESC
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING filter_json, imint_ai_analysis_id;
            """)
            return cursor.fetchall()
    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] ProgrammingError in lock_rows_for_processing_2: {e}")
        return []


def ensure_connection(conn):
    try:
        if conn.closed != 0:
            print("[INFO] PostgreSQL connection closed. Reconnecting...")
            return postgres_connection()
        else:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
        return conn
    except Exception:
        print("[ERROR] Connection check failed. Reconnecting...")
        traceback.print_exc()
        return postgres_connection()


def run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                 chunk_token_threshold=3000, chunk_size=2200,
                                 chunk_output_tokens=800, consolidation_output_tokens=1500,
                                 llm_context_tokens=None, chunk_workers_arg=None):                                
    conn = postgres_connection()

    while True:
        try:
            conn = ensure_connection(conn)
            ai_rows = lock_rows_for_processing_1(conn)
            conn.commit()

            conn = ensure_connection(conn)
            imint_rows = lock_rows_for_processing_2(conn)
            conn.commit()

            if ai_rows:
                task_categories = []
                if ai_summ:
                    task_categories.append('ai_summary')
                if ai_trends:
                    task_categories.append('ai_trends')
                if ai_change:
                    task_categories.append('ai_change')
                if ai_change_previous:
                    task_categories.append(f'ai_change_previous({ai_change_previous}y)')
                print(f"[INFO] Found {len(ai_rows)} new rows to process for {', '.join(task_categories)}")
                for row in ai_rows:
                    ai_analysis_summary_check_threadsafe(
                        row,
                        ai_trends,
                        ai_summ,
                        ai_change,
                        ai_change_previous,
                        chunk_token_threshold,
                        chunk_size,
                        chunk_output_tokens,
                        consolidation_output_tokens,
                        llm_context_tokens,
                        chunk_workers_arg
                    )
                    # p = multiprocessing.Process(target=ai_analysis_summary_check_threadsafe,
                    #                             args=(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                    #                                   chunk_token_threshold, chunk_size,
                    #                                   chunk_output_tokens, consolidation_output_tokens))
                    # p.daemon = True
                    # p.start()

            if imint_rows:
                print(f"[INFO] Found {len(imint_rows)} new rows to process for imint_ai_analysis")
                for irow in imint_rows:
                    imint_ai_analysis_summary_check_threadsafe(irow)
                    # p = multiprocessing.Process(target=imint_ai_analysis_summary_check_threadsafe, args=(irow,))
                    # p.daemon = True
                    # p.start()

        except Exception as e:
            print("[ERROR] Exception in main loop:")
            traceback.print_exc()

        time.sleep(poll_interval)

    if conn:
        conn.close()


def call_main_func():
    parser = argparse.ArgumentParser(
        prog='ai_analysis_v3.1.1.22',
        description='AI Analysis Summary — polls PostgreSQL for pending queries, '
                    'fetches Elasticsearch data, and generates AI-powered summary, '
                    'trend analysis, and change-detection reports.',
        epilog='examples:\n'
               '  python3 %(prog)s --ai_summary\n'
               '  python3 %(prog)s --ai-summary --ai-trends\n'
               '  python3 %(prog)s --ai_summary --chunk-token-threshold 8000\n'
               '  python3 %(prog)s --ai-change --ai-change-previous 2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ai_summary', '--ai-summary', action='store_true',
                        help='Enable AI analysis mode — generates LLM-based analysis '
                             'of ES query results using document upload + RAG query')
    parser.add_argument('--ai_trends', '--ai-trends', action='store_true',
                        help='Enable AI trends mode — produces trend charts for SAM, '
                             'training, infra, airfield, sitrep, and force disposition')
    parser.add_argument('--ai_change', '--ai-change', action='store_true',
                        help='Enable AI change detection — compares current period '
                             'against previous period for drift analysis')
    parser.add_argument('--ai_change_previous', '--ai-change-previous', type=int,
                        metavar='N',
                        help='Enable multi-year comparison — compare the selected '
                             'period with the same period N years ago')
    parser.add_argument('--chunk_token_threshold', '--chunk-token-threshold',
                        type=int, default=None, metavar='N',
                        help='Token threshold for processing large ES results before sending to LLM. Defaults to 75%% of --llm_context_tokens. Inputs are split into token-aware chunks for per-chunk analysis; Overall Analysis uses original source evidence when it fits.')
    parser.add_argument('--llm_context_tokens', '--llm-context-tokens',
                        type=int, default=None, metavar='N',
                        help='REQUIRED: total context-window size of the LLM in tokens '
                             '(e.g. 8192 for an 8B model, 32768 for a 31B/32B model). '
                             'Drives the per-chunk token threshold (75%% of this value) '
                             'and the combined-analysis prompt size. If not given, '
                             'the env var LLM_CONTEXT_TOKENS is used. The script will '
                             'exit with an error if neither is set, because guessing '
                             'from the model name is unreliable across model families.')
    parser.add_argument('--chunk_workers', '--chunk-workers',
                        type=int, default=None, metavar='N',
                        help='Number of independent chunk-analysis workers. Defaults to 2; '
                             'can also be set with AI_SUMMARY_CHUNK_WORKERS. Use 1 for '
                             'fully sequential execution. Maximum supported value is 8.')
    parser.add_argument('--rag_topn', '--rag-topn', type=int, default=None, metavar='N',
                        help='Override AnythingLLM RAG topN for each uploaded chunk document. '
                             'Default is coverage-oriented and bounded; can also be set with AI_SUMMARY_RAG_TOPN.')
    parser.add_argument('--chunk_target_tokens', '--chunk-target-tokens', type=int, default=None, metavar='N',
                        help='Optional direct target size for each uploaded chunk document in estimated tokens. '
                             'Default scales from context using a quality-oriented fraction and remains bounded.')
    args = parser.parse_args()

    ai_summ = False
    ai_trends = False
    ai_change = False
    ai_change_previous = None
    poll_interval = 5

    if args.ai_summary:
        ai_summ = True
        print("AI analysis mode is enabled. (Chunks use full-context direct Ollama with bounded parallel workers; AnythingLLM inline chat is fallback; Overall Analysis uses detailed aggregation from source evidence)")
    else:
        print("AI summary mode is disabled.")

    if args.ai_trends:
        ai_trends = True
        print("AI trends mode is enabled.")
    else:
        print("AI trends mode is disabled.")

    if args.ai_change:
        ai_change = True
        print("AI change mode is enabled.")
    else:
        print("AI change mode is disabled.")

    if args.ai_change_previous is not None:
        ai_change_previous = args.ai_change_previous
        print(f"AI previous-year change detection mode enabled (comparing with {ai_change_previous} year(s) ago).")
    else:
        print("AI previous-year change detection mode disabled.")

    # Resolve context-window size. Strict precedence — the operator MUST declare it:
    #   1. --llm_context_tokens CLI flag (per-run override)
    #   2. LLM_CONTEXT_TOKENS env var (deploy-time override)
    #   3. ERROR — never guess from the model name. The script refuses to run
    #      so the operator is forced to declare the context window for whatever
    #      LLM is currently configured in constants.LLM_MODEL.
    #
    # Common values: 8192 for an 8B model, 16384 for 14B, 32768 for 31B/32B/70B.
    if args.llm_context_tokens is not None:
        llm_context_tokens = args.llm_context_tokens
    elif _env_int := _safe_int(os.getenv("LLM_CONTEXT_TOKENS")):
        llm_context_tokens = _env_int
    else:
        print("ERROR: --llm_context_tokens (or env LLM_CONTEXT_TOKENS) is required.")
        print("       The script refuses to guess from the model name because")
        print("       context windows vary across model families (8B model = 8K,")
        print("       31B model = 32K, llama3.1 = 128K, etc.).")
        print("       Examples: dev  = 8192,   prod = 32768")
        print("       Run with: --llm_context_tokens 8192  (dev)")
        print("                 --llm_context_tokens 32768 (prod with 31B/32B)")
        print("       or export LLM_CONTEXT_TOKENS=*** before launching.")
        sys.exit(2)

    if args.chunk_token_threshold is not None:
        chunk_token_threshold = args.chunk_token_threshold
    else:
        # Default to 75% of the model's context window, leaving room for the
        # system prompt, the user's query, and the response. So a 32K-context
        # model gets a ~19K threshold; an 8K model gets ~4.9K.
        chunk_token_threshold = int(llm_context_tokens * 0.75)

    # Use a smaller, quality-oriented chunk-document target than the full
    # chunk threshold. With 8192 context this yields ~3072 tokens, which is
    # materially safer for complete inline analysis while still scaling well.
    _chunk_target_override = args.chunk_target_tokens if args.chunk_target_tokens is not None else _safe_int(os.getenv("AI_SUMMARY_CHUNK_TARGET_TOKENS"))
    if _chunk_target_override is not None and _chunk_target_override > 0:
        chunk_size = max(MIN_CHUNK_DOCUMENT_TOKENS, min(MAX_CHUNK_DOCUMENT_TOKENS, _chunk_target_override))
        print(f"  Per-chunk target override: {chunk_size} estimated document tokens")
    else:
        chunk_size = min(
            MAX_CHUNK_DOCUMENT_TOKENS,
            max(MIN_CHUNK_DOCUMENT_TOKENS, int(chunk_token_threshold * DEFAULT_CHUNK_TARGET_FRACTION))
        )
    # Give each chunk materially more generation headroom so the model is less likely
    # to compress away distinct prompt-relevant findings. Keep it bounded for very large contexts.
    chunk_output_tokens = min(
        MAX_DIRECT_OLLAMA_OUTPUT_TOKENS,
        max(3072, int(chunk_token_threshold * 0.40)),
    )
    consolidation_output_tokens = min(
        MAX_DIRECT_OLLAMA_OUTPUT_TOKENS,
        max(4096, int(chunk_token_threshold * 0.50)),
    )

    print(f"LLM context window: {llm_context_tokens} tokens (configured default model: {LLM_MODEL})")
    print("[INFO] The actual analysis model is read from ai_model_master for each queued request.")
    print(f"Chunk token threshold set to: {chunk_token_threshold} (75% of context)")
    print(f"  Per-chunk document target: {chunk_size} tokens (50% of threshold, quality-oriented)")
    print(f"  Per-chunk LLM output budget: {chunk_output_tokens} tokens (loss-resistant bounded generation)")
    print(f"  Combined-analysis output budget: {consolidation_output_tokens} tokens (bounded final generation)")
    print(f"  Combined-analysis context budget: {int(llm_context_tokens * 0.85)} tokens (85% of context)")
    if args.rag_topn is not None and args.rag_topn > 0:
        os.environ["AI_SUMMARY_RAG_TOPN"] = str(args.rag_topn)
    resolved_chunk_workers = _resolve_chunk_workers(args.chunk_workers)
    print(f"  Chunk workers: {resolved_chunk_workers}")
    print(f"  RAG topN: {os.getenv('AI_SUMMARY_RAG_TOPN', f'dynamic up to {DEFAULT_RAG_TOP_N}')} (legacy per-chunk path)")
    print("  Overall Analysis mode: field-wise source-record analysis from original ES records; sections analyzed independently and concatenated without a second synthesis pass")

    if not ai_summ and not ai_trends and not ai_change and not ai_change_previous:
        sys.exit(1)
    else:
        run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                             chunk_token_threshold, chunk_size,
                             chunk_output_tokens, consolidation_output_tokens,
                             llm_context_tokens, args.chunk_workers)


if __name__ == '__main__':
    while True:
        call_main_func()
