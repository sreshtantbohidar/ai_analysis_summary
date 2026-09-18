import argparse
import copy
import json
import multiprocessing
import os
import re
import sys
import time
import tempfile
import base64
import traceback
from datetime import datetime, timedelta
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

# ⭐ v3.2.0: safety caps for the inline chat path. The CLI chunk_token_threshold
# may be set high (to cut the number of LLM calls), but an inline chat prompt is
# bounded by the model's context window — cap single calls and chunk sizes so
# small-context models (e.g. gemma2:9b, 8k) never overflow.
MAX_INLINE_CHAT_TOKENS = 4000
MAX_CHUNK_TOKENS = 2200

# Fallback model if the configured model returns empty responses
FALLBACK_LLM_MODEL = "gemma2:9b-instruct-q8_0"

_current_anythingllm_model = None
DEFAULT_CHAT_PROVIDER = "ollama"

# ──────────────────────────────────────────────
# Elasticsearch client
# ──────────────────────────────────────────────
es = Elasticsearch(
    [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": ELASTIC_CLIENT_SCHEME}],
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False
)

# ⭐ v3.2.0: All records fetched from ES are sorted by activity_date descending.
# Documents without activity_date are pushed to the end.
SORT_BY_ACTIVITY_DATE_DESC = [{"activity_date": {"order": "desc", "missing": "_last"}}]

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
# HTML / Report helpers (reused from v3.0.0 / v3.1.0)
# ──────────────────────────────────────────────
def _strip_wrappers(html: str) -> str:
    html = re.sub(r'</?html[^>]*>', '', html, flags=re.I)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.I | re.S)
    html = re.sub(r'</?body[^>]*>', '', html, flags=re.I)
    return html.strip()


def build_tabbed_html(html_strings: List[str]) -> str:
    labels = ["AI Summary", "AI Trends", "AI Change Detection"]
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

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Reports – Summary | Trends | Change</title>
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
    /* ⭐ v3.2.0 — Monthly grouped report styles */
    .monthly-report {{
        margin-top: 6px;
    }}
    .report-intro {{
        background: linear-gradient(135deg, #004085 0%, #0b5caf 100%);
        color: #fff;
        padding: 20px 25px;
        border-radius: 6px;
        margin-bottom: 16px;
    }}
    .report-intro h1 {{
        margin: 0 0 6px 0;
        font-size: 1.5em;
        color: #fff;
        border: none;
        padding: 0;
    }}
    .report-intro p {{
        margin: 0;
        opacity: .92;
        font-size: .9em;
    }}
    .report-toc {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 8px;
    }}
    .toc-item {{
        flex: 1 1 170px;
        display: flex;
        flex-direction: column;
        background: #fff;
        padding: 10px 14px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-decoration: none;
        color: #333;
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .toc-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.16);
    }}
    .toc-month {{
        font-weight: 600;
        color: #004085;
    }}
    .toc-count {{
        font-size: .8em;
        color: #6c757d;
        margin-top: 2px;
    }}
    .month-section {{
        margin-bottom: 24px;
    }}
    .month-header {{
        background: #fff;
        padding: 12px 20px;
        margin: 18px 0 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .month-header h2 {{
        margin: 0;
        color: #004085;
        font-size: 1.35em;
        border: none;
        padding: 0;
    }}
    .month-meta {{
        font-size: .82em;
        color: #495057;
        background: #eef2f7;
        padding: 4px 12px;
        border-radius: 12px;
        white-space: nowrap;
    }}
    .month-content {{
        background: #fff;
        padding: 22px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .location-block {{
        margin-bottom: 16px;
        border: 1px solid #e3e8ef;
        border-radius: 4px;
        overflow: hidden;
    }}
    .location-block h3 {{
        margin: 0;
        padding: 10px 16px;
        background: #f1f5f9;
        color: #0f4c81;
        font-size: 1.05em;
        border-bottom: 1px solid #e3e8ef;
    }}
    .location-content {{
        padding: 12px 16px;
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
    Report automatically generated by AI Analysis System
  </footer>
</body>
</html>"""


# ──────────────────────────────────────────────
# ⭐ v3.2.0: All-in-one monthly report builder
# ──────────────────────────────────────────────
def build_monthly_report_html(monthly_sections: List[dict], search_form_type: str) -> str:
    """
    Build the all-in-one HTML report with one section per month.

    Each element of monthly_sections is a dict with keys:
        - "month":   "YYYY-MM" label
        - "content": HTML string produced for that month
        - "records": (optional) number of records analysed in that month
    Months must already be ordered latest → oldest (the caller sorts them).

    The returned document is meant to be embedded inside the "AI Summary" tab
    of build_tabbed_html(), which strips the <html>/<head>/<body> wrappers.
    """
    month_colors = ["#004085", "#1e7e34", "#b45309", "#7c3aed", "#be185d", "#0f766e"]
    toc_items = []
    sections = []
    total_records = sum(sec.get("records", 0) for sec in monthly_sections)

    for i, sec in enumerate(monthly_sections):
        month_key = sec.get("month", f"Unknown-{i + 1}")
        # ⭐ v3.2.0: readable section titles ("August 2026") instead of the bare
        # "YYYY-MM" key, so month sections can never be mistaken for each other.
        month_label = format_month_label(month_key)
        records = sec.get("records", 0)
        color = month_colors[i % len(month_colors)]
        content = _strip_wrappers(sec.get("content", "")) if sec.get("content") else "<p>No analysis available for this month.</p>"
        meta_text = f"{month_key} · {records} records" if month_key != "unknown" else f"{records} records"

        toc_items.append(
            f'<a class="toc-item" href="#month-{i}" style="border-left:4px solid {color};">'
            f'<span class="toc-month">{month_label}</span>'
            f'<span class="toc-count">{meta_text}</span></a>'
        )
        sections.append(f'''
        <div class="month-section" id="month-{i}">
            <div class="month-header" style="border-left:4px solid {color};">
                <h2>&#128197; {month_label}</h2>
                <span class="month-meta">{meta_text}</span>
            </div>
            <div class="month-content">
                {content}
            </div>
        </div>''')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    toc_html = "\n".join(toc_items)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Monthly Analysis Report – {search_form_type}</title>
</head>
<body>
  <div class="monthly-report">
    <div class="report-intro">
      <h1>Monthly Analysis Report</h1>
      <p><strong>{search_form_type}</strong> &middot; {len(monthly_sections)} month(s) &middot; {total_records} records &middot; Generated on {now}</p>
    </div>
    <div class="report-toc">
      {toc_html}
    </div>
    {''.join(sections)}
  </div>
</body>
</html>"""


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
    workspace_slug = workspace_slug or None
    ws_slug = workspace_slug or ANYTHINGLLM_WORKSPACE_SLUG
    # The cache only tracks the shared workspace: ephemeral workspaces always
    # start with the system default model and must be updated explicitly.
    if workspace_slug is None and not force and _current_anythingllm_model == model_name:
        return True

    update_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ws_slug}/update"
    payload = {"chatProvider": chat_provider, "chatModel": model_name}
    headers = {"Authorization": f"Bearer {IFC_LLM_TOKEN}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(update_url, json=payload, headers=headers, timeout=30)
            if r.status_code in (200, 201, 204):
                print(f"[INFO] AnythingLLM workspace '{ws_slug}' model updated to: {model_name}")
                if workspace_slug is None:
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


# ──────────────────────────────────────────────
# ⭐ v3.2.0: Ephemeral workspace lifecycle
# ──────────────────────────────────────────────
def create_anythingllm_workspace(name: str) -> Optional[str]:
    """
    Create a fresh AnythingLLM workspace and return its slug.

    POST /api/v1/workspace/new  body: {"name": ...}

    The server generates the slug from the name; it is read back from the
    response (workspace.slug). Returns None on failure. A fresh workspace is
    what guarantees RAG queries never see stale documents from other runs.
    """
    create_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/new"
    headers = get_anythingllm_headers()
    payload = {"name": name}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(create_url, json=payload, headers=headers, timeout=30)
            if r.status_code in (200, 201):
                data = r.json()
                workspace = data.get("workspace", data) if isinstance(data, dict) else {}
                slug = workspace.get("slug") if isinstance(workspace, dict) else None
                if slug:
                    print(f"[INFO] Ephemeral workspace created: '{slug}'")
                    return slug
                print(f"[WARN] Workspace created but no slug in response: {r.text[:200]}")
                return None
            else:
                print(f"[ERROR] Workspace create status {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[ERROR] Workspace create timeout (attempt {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            print(f"[ERROR] Workspace create failed: {str(e)}")
        time.sleep(2 ** attempt)

    print(f"[WARN] Failed to create ephemeral workspace '{name}'")
    return None


def delete_anythingllm_workspace(workspace_slug: str) -> bool:
    """
    Delete an AnythingLLM workspace (removes its documents and vector namespace).

    DELETE /api/v1/workspace/{slug}
    """
    if not workspace_slug:
        return False
    delete_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{workspace_slug}"
    headers = get_anythingllm_headers()

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.delete(delete_url, headers=headers, timeout=30)
            if r.status_code in (200, 202, 204):
                print(f"[INFO] Ephemeral workspace deleted: '{workspace_slug}'")
                return True
            else:
                print(f"[ERROR] Workspace delete status {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[ERROR] Workspace delete timeout (attempt {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            print(f"[ERROR] Workspace delete failed: {str(e)}")
        time.sleep(2 ** attempt)

    print(f"[WARN] Could not delete ephemeral workspace '{workspace_slug}'")
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


def upload_document_to_anythingllm(markdown_content: str, filename: str,
                                   add_to_workspaces: Optional[str] = None) -> Optional[str]:
    """
    Upload a markdown document to AnythingLLM.

    POST /api/v1/document/upload (multipart/form-data)

    If add_to_workspaces is given, the document is embedded into that workspace
    during the upload itself (single step — no separate update-embeddings call).

    Returns the document location path or None on failure.
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
            data = {'addToWorkspaces': add_to_workspaces} if add_to_workspaces else None
            for attempt in range(MAX_RETRIES):
                try:
                    r = requests.post(upload_url, headers=headers, files=files, data=data, timeout=60)
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


def query_anythingllm_rag(query: str, llm_model: str,
                          workspace_slug: Optional[str] = None) -> str:
    """
    Send a query in RAG mode to AnythingLLM.

    POST /api/v1/workspace/{slug}/chat with mode='query'

    The response is generated based on documents embedded in the workspace.
    First updates the workspace model, then sends the query.

    ⭐ v3.2.0: when the document was just embedded (ephemeral workspace), the
    first query may hit the workspace's queryRefusalResponse ("no relevant
    information") before the vectors are queryable — retry a few times, waiting
    for embedding to settle, before giving up. Returns "" when no usable
    context is found so the caller can fall back to chat mode.
    """
    ws_slug = workspace_slug or ANYTHINGLLM_WORKSPACE_SLUG
    update_anythingllm_workspace_model(llm_model, workspace_slug=ws_slug, force=True)

    chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ws_slug}/chat"
    headers = get_anythingllm_headers()

    payload = {
        "message": query.replace("\n", " ").strip(),
        "mode": "query"
    }

    # AnythingLLM answers with its configured queryRefusalResponse when no
    # context is found; keep the marker narrow to avoid false positives.
    REFUSAL_MARKERS = ("no relevant information",)
    attempts = MAX_RETRIES  # initial + retries with growing waits

    for attempt in range(attempts):
        try:
            r = requests.post(chat_url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
            if r.status_code == 200:
                result = r.json()
                text = (result.get("textResponse") or "").strip()
                lowered = text.lower()
                if text and not any(m in lowered for m in REFUSAL_MARKERS):
                    return text
                print(f"[WARN] RAG query returned no usable context (attempt {attempt+1}/{attempts})")
                if attempt < attempts - 1:
                    time.sleep(3 * (attempt + 1))  # wait for embedding to settle
                    continue
                return ""
            else:
                error_text = r.text
                print(f"[ERROR] RAG query status {r.status_code}: {error_text[:200]}")
                # If the model failed, try the fallback model on the next attempt
                if "failed to communicate" in error_text.lower() and attempt == 0:
                    print(f"[INFO] Retrying with fallback model: {FALLBACK_LLM_MODEL}")
                    update_anythingllm_workspace_model(FALLBACK_LLM_MODEL, workspace_slug=ws_slug, force=True)
        except requests.exceptions.Timeout:
            print(f"[ERROR] RAG query timeout (attempt {attempt+1}/{attempts})")
        except Exception as e:
            print(f"[ERROR] RAG query failed: {str(e)}")
        if attempt < attempts - 1:
            time.sleep(3 * (attempt + 1))

    return ""


def get_llm_response_with_data_context(markdown_content: str, query: str, llm_model: str,
                                       doc_label: str = "analysis_data",
                                       workspace_slug: Optional[str] = None) -> str:
    """
    ⭐ v3.2.0: send data to the LLM through an (ephemeral) AnythingLLM workspace.

    The caller creates a fresh workspace per month (create_anythingllm_workspace)
    and passes its slug; each call uploads its document straight into that
    workspace (addToWorkspaces → one-step embed) and queries in RAG mode. Because
    the workspace is brand new per month, RAG can never return stale documents
    from other runs — the root cause of the repeated "July 2026" titles.

    Workflow:
      1. Upload the markdown document into the ephemeral workspace (embedded at upload)
      2. Query in RAG mode against that workspace (retries while embedding settles)
      3. Fall back to chat mode with inline data if RAG fails / no workspace available

    Args:
        markdown_content: The markdown-formatted data to upload.
        query: The analysis prompt to send.
        llm_model: LLM model name.
        doc_label: Label used for the uploaded document filename.
        workspace_slug: Slug of the ephemeral workspace; None → chat mode only.

    Returns:
        LLM response string.
    """
    if not workspace_slug:
        print("[WARN] No ephemeral workspace available — using chat mode with inline data")
        return get_llm_response_chat_with_data(markdown_content, query, llm_model, doc_label=doc_label)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{doc_label}_{timestamp}.md"

    print(f"[INFO] Uploading markdown document into ephemeral workspace '{workspace_slug}': "
          f"{filename} ({len(markdown_content)} chars)")

    # Step 1: Upload the document directly into the ephemeral workspace (one-step embed)
    doc_location = upload_document_to_anythingllm(markdown_content, filename, add_to_workspaces=workspace_slug)
    if not doc_location:
        print("[WARN] Document upload failed, falling back to chat mode with inline data")
        return get_llm_response_chat_with_data(markdown_content, query, llm_model, doc_label=doc_label)

    # Step 2: Query in RAG mode (query_anythingllm_rag retries while embedding settles)
    print(f"[INFO] Sending query in RAG mode to '{workspace_slug}' (model: {llm_model})...")
    rag_response = query_anythingllm_rag(query, llm_model, workspace_slug=workspace_slug)

    if rag_response:
        print(f"[INFO] RAG response received ({len(rag_response)} chars)")
        return rag_response

    # Step 3: Fallback to chat mode with inline data
    print("[WARN] RAG query returned no usable response, falling back to chat mode with inline data")
    return get_llm_response_chat_with_data(markdown_content, query, llm_model, doc_label=doc_label)


def get_llm_response_chat_with_data(markdown_content: str, query: str, llm_model: str,
                                    doc_label: str = "analysis_data") -> str:
    """
    ⭐ v3.2.0: send data to the LLM in chat mode with the data embedded directly
    in the prompt (no RAG / workspace upload). Used as the fallback when the
    ephemeral-workspace RAG flow fails or no workspace is available.

    Inline chat guarantees the model reads exactly the supplied records; the
    per-group token limit keeps every call safely inside the model's context.
    """
    combined_prompt = (
        f"Here is the data to analyze (from {doc_label}):\n\n"
        f"{markdown_content}\n\n"
        f"---\n\n"
        f"{query}"
    )
    print(f"[INFO] Chat-mode analysis for {doc_label}: {len(markdown_content)} chars of data")
    return get_llm_response_direct(combined_prompt, llm_model, preserve_newlines=True)


def get_llm_response_direct(query: str, llm_model: str,
                            preserve_newlines: bool = False) -> str:
    """Direct chat mode: sends the prompt as-is to the AnythingLLM workspace chat API.

    If the configured model returns an empty response ("text response was empty"),
    the function will try once with a fallback model (FALLBACK_LLM_MODEL) before
    returning an error message. This prevents wasting minutes on retries for
    models that are installed but don't produce valid chat output.

    By default newlines are collapsed to spaces; pass preserve_newlines=True to
    keep the prompt verbatim (used when embedding data inline, where markdown
    tables need their line structure).
    """
    chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
    headers = get_anythingllm_headers()

    def _attempt(model: str, timeout: int) -> str:
        """Single attempt with a given model and timeout."""
        update_anythingllm_workspace_model(model)
        message = query if preserve_newlines else query.replace("\n", " ").strip()
        payload = {
            "message": message,
            "mode": "chat"
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
# Convert ES hits to Markdown format
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


def convert_hits_to_markdown(e_hits: list, search_form_type: str, title: Optional[str] = None) -> str:
    """
    Convert Elasticsearch hits to a well-structured markdown document.

    For profile analysis: uses a description-based format grouped by file.
    For other types: uses the TYPE_MAPPING to create a table.

    Args:
        e_hits: Elasticsearch hit records
        search_form_type: Type of analysis (e.g., "infra development analysis")
        title: Optional custom document title (used for month / location groups)

    Returns:
        Markdown string
    """
    search_form_type_lower = search_form_type.lower()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_title = title if title else f"Analysis Report: {search_form_type}"
    lines = []

    # ── Header ──
    lines.append(f"# {report_title}")
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

    # Section: Summary Statistics
    lines.append("## Summary Statistics")
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
# ⭐ v3.2.0: Monthly / location grouping helpers
# ──────────────────────────────────────────────
def get_record_date(source: dict) -> str:
    """
    Return the date of an ES source document.

    Prefers the `activity_date` field (used for the ES sort); falls back to
    `@timestamp` when activity_date is absent (e.g. profile analysis records).
    """
    if not isinstance(source, dict):
        return ""
    date_val = source.get("activity_date") or source.get("@timestamp")
    if date_val is None:
        return ""
    return str(date_val)


def extract_month_key(date_str: str) -> str:
    """
    Extract a "YYYY-MM" month key from an arbitrary date string.

    Handles ISO-8601 strings (with/without timezone "Z"), plain "YYYY-MM-DD"
    dates and free-text values. Returns "unknown" when no month can be parsed.
    """
    date_str = (date_str or "").strip()
    if not date_str:
        return "unknown"

    # Quick regex for the leading YYYY-MM (works for ISO strings too)
    m = re.match(r"(\d{4})-(\d{1,2})", date_str)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"

    # Fallback: try to parse as ISO datetime (Python 3.10 needs explicit "Z" handling)
    try:
        normalized = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).strftime("%Y-%m")
    except ValueError:
        pass

    # Epoch timestamps (seconds or milliseconds) — e.g. "1723021800" or "1723021800000"
    if re.fullmatch(r"\d{10,13}", date_str):
        try:
            ts = int(date_str)
            if ts > 10 ** 12:  # milliseconds
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts).strftime("%Y-%m")
        except (ValueError, OSError, OverflowError):
            pass

    return "unknown"


def group_hits_by_month(e_hits: list) -> List[tuple]:
    """
    Group ES hits by calendar month (YYYY-MM) and order the months from
    latest to oldest. Records without a parseable date go into an "unknown"
    bucket which is always placed last.
    """
    monthly = {}
    for hit in e_hits:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        month = extract_month_key(get_record_date(source))
        monthly.setdefault(month, []).append(hit)

    known_months = sorted([m for m in monthly if m != "unknown"], reverse=True)
    ordered = [(m, monthly[m]) for m in known_months]
    if "unknown" in monthly:
        ordered.append(("unknown", monthly["unknown"]))
    return ordered


def group_hits_by_location(hits: list) -> List[tuple]:
    """
    Group a set of hits by their location field.

    Uses `location_name` when present, otherwise falls back to other
    location-ish fields (start_location_name, base_location_name). Groups are
    ordered by record count (descending), then by name.
    """
    LOCATION_FIELDS = ["location_name", "start_location_name", "base_location_name"]
    locations = {}
    for hit in hits:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        loc = "Unknown Location"
        for field in LOCATION_FIELDS:
            value = source.get(field)
            if value:
                loc = str(value).strip()
                break
        locations.setdefault(loc, []).append(hit)

    return sorted(locations.items(), key=lambda kv: (-len(kv[1]), str(kv[0]).lower()))


def sanitize_label(text: str) -> str:
    """Turn an arbitrary label into a safe short filename fragment."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")
    return (cleaned or "unknown")[:60]


# ⭐ v3.2.0: readable month-name helpers
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def format_month_label(month_key: str) -> str:
    """
    Turn a "YYYY-MM" month key into a readable "MonthName YYYY" label.

    e.g. "2026-08" → "August 2026". Returns "Undated records" for the
    unknown bucket and passes through anything that cannot be parsed.
    """
    if not month_key or month_key == "unknown":
        return "Undated records"
    m = re.match(r"(\d{4})-(\d{1,2})", str(month_key))
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{MONTH_NAMES[month - 1]} {year}"
    return str(month_key)


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


def get_data_from_elastic(elastic_query):
    index_name = CUREENT_INDEX_NAME
    try:
        elastic_query = copy.deepcopy(elastic_query)
        size = elastic_query.get("size", None)

        # ⭐ v3.2.0: enforce sort by activity_date DESC for every fetch.
        # The user-provided query keeps any existing sort, but our sort is
        # applied first so records always arrive newest-first.
        sort_clause = SORT_BY_ACTIVITY_DATE_DESC
        if "sort" not in elastic_query:
            elastic_query["sort"] = sort_clause
        elif isinstance(elastic_query.get("sort"), list):
            elastic_query["sort"] = sort_clause + elastic_query["sort"]

        if size is None or size <= 1000:
            all_hits = []
            batch_size = 1000
            elastic_query["size"] = batch_size

            response = es.search(index=index_name, body=elastic_query, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
            all_hits.extend(hits)

            while hits:
                response = es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]
                if not hits:
                    break
                all_hits.extend(hits)

            return {"hits": {"total": {"value": len(all_hits)}, "hits": all_hits}}

        return es.search(index=index_name, body=elastic_query)

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


# ──────────────────────────────────────────────
# ⭐ v3.2.0: Per-group LLM processing
# ──────────────────────────────────────────────
def llm_analyze_hits(hits: list, search_form_type: str, title: str, doc_label: str,
                     base_prompt: str, extra_prompt: str, model_name: str, key: str,
                     chunk_token_threshold: int, chunk_size: int,
                     workspace_slug: Optional[str] = None) -> str:
    """
    Send a group of hits to the AnythingLLM API.

    If the group's markdown + prompt fits inside the token safe limit it is
    sent in a single call. Otherwise hierarchical chunking (from v3.1.0) is
    used as a safety net: chunks are summarised individually and the summaries
    are consolidated in a final call. All calls go through the ephemeral
    workspace (workspace_slug) in RAG mode, falling back to inline chat.
    """
    effective_threshold = min(chunk_token_threshold, MAX_INLINE_CHAT_TOKENS)
    markdown_data = convert_hits_to_markdown(hits, search_form_type, title=title)
    full_query = base_prompt + extra_prompt
    words, total_tokens = count_words_and_tokens(markdown_data + full_query)
    print(f"[INFO] {doc_label}: {len(hits)} records | {words} words | ~{total_tokens} tokens | "
          f"threshold={effective_threshold}")

    if total_tokens < effective_threshold:
        print(f"[INFO] {doc_label}: within token limit → single RAG call (ephemeral workspace)")
        return get_llm_response_with_data_context(markdown_data, full_query, model_name,
                                                  doc_label=doc_label, workspace_slug=workspace_slug)

    print(f"[INFO] {doc_label}: exceeds token limit (~{total_tokens} tokens) → hierarchical chunking")
    fields = TYPE_MAPPING.get(key, {}).get("fields", [])
    data_lines = []
    for hit in hits:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        formatted = " | ".join(str(source.get(f, "")) for f in fields)
        if formatted not in data_lines:
            data_lines.append(formatted)

    if not data_lines:
        return ""

    effective_chunk_size = min(chunk_size, MAX_CHUNK_TOKENS)
    chunks = build_chunks_from_lines(data_lines, max_tokens=effective_chunk_size)
    print(f"[INFO] {doc_label}: processing {len(chunks)} chunk(s)...")
    chunk_summaries = []

    for idx, chunk in enumerate(tqdm(chunks, desc=f"LLM Chunks ({doc_label})", unit="chunk")):
        chunk_markdown = f"# Data Chunk {idx + 1}/{len(chunks)}\n\n"
        chunk_markdown += "```\n" + chunk + "\n```\n\n"
        chunk_markdown += f"---\n_Chunk {idx + 1} of {len(chunks)}_"

        summary = get_llm_response_with_data_context(
            chunk_markdown, base_prompt + extra_prompt, model_name,
            doc_label=f"{doc_label}_chunk{idx}", workspace_slug=workspace_slug
        )
        chunk_summaries.append(summary)

    combined_text = "\n\n".join(chunk_summaries)
    final_markdown = f"# Consolidated Summaries\n\n{combined_text}"
    return get_llm_response_with_data_context(
        final_markdown, base_prompt + extra_prompt, model_name,
        doc_label=f"{doc_label}_consolidated", workspace_slug=workspace_slug
    )


def process_month_group(month: str, month_hits: list, search_form_type: str, key: str,
                        ai_analysis_summary_id: int, model_prompt_dict: dict,
                        chunk_token_threshold: int, chunk_size: int) -> str:
    """
    Analyse one month of data with the LLM.

    If the month's data fits inside the token safe limit a single LLM call is
    made. Otherwise the month is split by location and every location group is
    analysed separately (each location may itself fall back to hierarchical
    chunking if it is still too large).
    """
    base_prompt = model_prompt_dict.get("prompt", "") if model_prompt_dict else ""
    extra_prompt = ("\n\n(PLEASE NOTE: The response you will give should be in html. "
                    "Any title in the response should be in h2 tag and paragraph should be in p tag.)")
    # ⭐ v3.2.0: make the month explicit so the LLM never labels this report
    # with a different month (previously every section repeated "July 2026").
    month_label = format_month_label(month)
    month_hint = (
        f"\n\n(IMPORTANT CONTEXT: The records provided above ALL belong to the calendar month "
        f"{month} ({month_label}). Analyse ONLY these records — do not use data from any other "
        f"period. The report title MUST clearly state the period: {month_label}.)"
    )
    model_name = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
    month_doc_label = f"{key}_{ai_analysis_summary_id}_{month}"

    # Cap the decision threshold: a CLI value larger than the model-safe inline
    # limit must still split by location instead of overflowing the model.
    effective_threshold = min(chunk_token_threshold, MAX_INLINE_CHAT_TOKENS)
    month_markdown = convert_hits_to_markdown(month_hits, search_form_type, title=f"{search_form_type} — {month_label}")
    full_query = base_prompt + month_hint + extra_prompt
    words, total_tokens = count_words_and_tokens(month_markdown + full_query)
    print(f"[MONTH {month}] {len(month_hits)} records | {words} words | ~{total_tokens} tokens | "
          f"threshold={effective_threshold}")

    # ⭐ v3.2.0: one fresh AnythingLLM workspace per month. It is reused by all
    # of the month's LLM calls (single call or every location group) and deleted
    # afterwards — so RAG can never see stale documents from other months/runs.
    # Note: the month's own location documents do accumulate inside this one
    # workspace; a location query most-similarity-matches its own document, and
    # any RAG failure falls back to exact inline chat.
    workspace_slug = create_anythingllm_workspace(
        f"ai-summary-{key}-{ai_analysis_summary_id}-{month}-{int(time.time())}-{os.getpid()}")
    if workspace_slug:
        update_anythingllm_workspace_model(model_name, workspace_slug=workspace_slug, force=True)

    try:
        if total_tokens < effective_threshold:
            print(f"[MONTH {month}] Within token limit → single RAG call (ephemeral workspace)")
            return get_llm_response_with_data_context(
                month_markdown, full_query, model_name,
                doc_label=month_doc_label, workspace_slug=workspace_slug)

        print(f"[MONTH {month}] Over token limit (~{total_tokens} tokens) → grouping by location")
        location_groups = group_hits_by_location(month_hits)
        print(f"[MONTH {month}] Found {len(location_groups)} location group(s)")

        location_parts = []
        for loc, loc_hits in location_groups:
            loc_doc_label = f"{month_doc_label}_{sanitize_label(loc)}"
            print(f"[MONTH {month}] 📍 {loc} → {len(loc_hits)} record(s)")
            loc_response = llm_analyze_hits(
                loc_hits, search_form_type,
                title=f"{search_form_type} — {month_label} — {loc}",
                doc_label=loc_doc_label, base_prompt=base_prompt + month_hint, extra_prompt=extra_prompt,
                model_name=model_name, key=key,
                chunk_token_threshold=effective_threshold, chunk_size=chunk_size,
                workspace_slug=workspace_slug,
            )  # llm_analyze_hits applies the MAX_CHUNK_TOKENS cap internally
            if loc_response:
                location_parts.append(
                    f'<div class="location-block"><h3>&#128205; {loc}</h3>'
                    f'<div class="location-content">{_strip_wrappers(loc_response)}</div></div>'
                )

        return "\n".join(location_parts)
    finally:
        # ⭐ v3.2.0: always remove the ephemeral workspace, even on error, so the
        # AnythingLLM server never accumulates per-month workspaces/documents.
        if workspace_slug:
            delete_anythingllm_workspace(workspace_slug)


def process_profile_month(month: str, month_hits: list, person_names: list,
                          organization_names: list, ai_analysis_summary_id: int) -> str:
    """
    Analyse one month of profile-analysis data.

    Profile analysis has its own prompt (grouped by file) and uses direct chat
    mode, so this keeps the v3.1.0 logic but scoped to a single month.
    """
    all_descriptions = []
    index = 0
    for single_json in month_hits:
        single_json = single_json.get("_source", {}) if isinstance(single_json, dict) else {}
        if single_json.get("description"):
            file_http_path = single_json.get('file_http_path', f'file{index}')
            index += 1 if 'file_http_path' not in single_json else 0
            activity_date = get_record_date(single_json)
            try:
                date_str = datetime.fromisoformat(str(activity_date).replace("Z", "+00:00")).strftime("%Y-%m-%d")
            except ValueError:
                date_str = str(activity_date)
            file_name = file_http_path.split('/')[-1]
            complete_single_profile = {
                "file_path": file_http_path, "file_name": file_name,
                "date": date_str, "description": single_json['description']
            }
            all_descriptions.append(complete_single_profile)

    if not all_descriptions:
        return ""

    df = pd.DataFrame(all_descriptions)
    grouped = df.groupby("file_name").agg({
        "file_path": lambda x: list(set(x)),
        "description": lambda x: list(set(x)),
        "date": lambda x: list(set(x))
    }).reset_index()
    complete_summary_dict_list = grouped.to_dict(orient="records")

    llm_prompt = create_llm_prompt_for_profile(
        complete_summary_dict_list, 'profile analysis', person_names, organization_names)
    extra_prompt = ("\n\n(PLEASE NOTE: The response you will give should be in html. "
                    "Any title in the response should be in h2 tag and paragraph should be in p tag.)")
    llm_prompt += extra_prompt
    # ⭐ v3.2.0: pin the month so profile sections are labelled with the right period.
    llm_prompt += (
        f"\n\n(IMPORTANT CONTEXT: The data above belongs to the calendar month "
        f"{month} ({format_month_label(month)}). The report title MUST clearly state "
        f"this period.)"
    )

    print(f"[MONTH {month}] Profile analysis → {len(all_descriptions)} description(s)")
    # preserve_newlines: keep the multi-line dict readable for the model.
    temp_llm_response = get_llm_response_direct(llm_prompt, LLM_MODEL, preserve_newlines=True)
    return f"<div>{temp_llm_response}</div><br>"


# ──────────────────────────────────────────────
# ⭐ Modified: ai_analysis_summary_check with Monthly Grouping
# ──────────────────────────────────────────────
def ai_analysis_summary_check(row, ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                               chunk_token_threshold=3000, chunk_size=2200):
    elastic_query, ai_analysis_summary_id, search_form_type = row
    search_form_type = search_form_type.lower()
    model_prompt_dict = None
    if search_form_type != 'profile analysis':
        model_prompt_dict = get_prompt_and_model(cursor, ai_analysis_summary_id)

    e_response = get_data_from_elastic(elastic_query) if elastic_query else None
    if e_response and e_response['hits']['total']['value'] > 0:
        e_hits = e_response['hits']['hits']
    else:
        e_hits = []

    ai_change_repsonse = ''
    ai_trend_response = ''
    llm_response = ''

    print(f"-- {search_form_type} --")

    # ── Change Detection (unchanged from v3.1.0) ──
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

    # ── Trends (unchanged from v3.1.0) ──
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

    # ── ⭐ AI Summary with Monthly Grouping (v3.2.0) ──
    if ai_summ:
        ai_summ_start_time = time.time()

        # All month results are collected here and assembled into one report
        # with a section per month (latest month first).
        monthly_sections = []

        if search_form_type == 'profile analysis':
            print(f"[INFO] Profile analysis for {ai_analysis_summary_id}")
            person_names = []
            organization_names = []
            if elastic_query and 'query' in elastic_query and 'bool' in elastic_query['query'] \
                    and 'must' in elastic_query["query"]["bool"]:
                for condition in elastic_query["query"]["bool"]["must"]:
                    if "match_phrase_prefix" in condition:
                        if "person_name" in condition["match_phrase_prefix"]:
                            person_names.append(condition["match_phrase_prefix"]["person_name"])
                        elif "civil_organization.civil_organization_name" in condition["match_phrase_prefix"]:
                            organization_names.append(condition["match_phrase_prefix"]["civil_organization.civil_organization_name"])
                        elif "civil_organization_name" in condition["match_phrase_prefix"]:
                            organization_names.append(condition["match_phrase_prefix"]["civil_organization_name"])

            if (person_names or organization_names) and e_hits:
                monthly_groups = group_hits_by_month(e_hits)
                print(f"[INFO] Profile: {len(e_hits)} records → {len(monthly_groups)} month(s), latest first")
                for month, month_hits in monthly_groups:
                    month_response = process_profile_month(
                        month, month_hits, person_names, organization_names, ai_analysis_summary_id)
                    if month_response:
                        monthly_sections.append({
                            "month": month, "content": month_response, "records": len(month_hits)})

        # ── Standard Analysis Types ──
        for key, value in TYPE_MAPPING.items():
            if search_form_type in value["types"]:
                print(f"[INFO] AI analysis for {key} [{ai_analysis_summary_id}]")

                if not e_hits:
                    print(f"[WARN] No ES hits for {key}")
                    continue

                monthly_groups = group_hits_by_month(e_hits)
                print(f"[INFO] {len(e_hits)} records → {len(monthly_groups)} month(s), latest first")

                for month, month_hits in monthly_groups:
                    month_response = process_month_group(
                        month, month_hits, search_form_type, key, ai_analysis_summary_id,
                        model_prompt_dict, chunk_token_threshold, chunk_size)
                    if month_response:
                        monthly_sections.append({
                            "month": month, "content": f"<div>{month_response}</div><br>",
                            "records": len(month_hits)})

        if monthly_sections:
            llm_response = build_monthly_report_html(monthly_sections, search_form_type)
            print(f"[INFO] Assembled monthly report with {len(monthly_sections)} month section(s)")

        print(f"[DONE] AI analysis complete [{ai_analysis_summary_id}] "
              f"[time={(time.time() - ai_summ_start_time):.2f} s]")

    # ── Build HTML report ──
    tabbed_html = build_tabbed_html([llm_response, ai_trend_response, ai_change_repsonse])
    ai_analysis_text_base64 = base64.b64encode(tabbed_html.encode("utf-8")).decode("utf-8")

    update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id, ai_analysis_text_base64)


# ──────────────────────────────────────────────
# IMINT Analysis (unchanged from v3.1.0)
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
# Prompt and Model helpers (unchanged from v3.1.0)
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
# Thread-safe wrappers (unchanged from v3.1.0)
# ──────────────────────────────────────────────
def ai_analysis_summary_check_threadsafe(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                                          chunk_token_threshold=3000, chunk_size=2200):
    try:
        conn = postgres_connection()
        with conn.cursor() as cursor:
            filter_json, ai_analysis_summary_id, search_form_type = row
            ai_analysis_summary_check((filter_json, ai_analysis_summary_id, search_form_type),
                                       ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                                       chunk_token_threshold, chunk_size)
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
                                  chunk_token_threshold=3000, chunk_size=2200):
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
                    p = multiprocessing.Process(target=ai_analysis_summary_check_threadsafe,
                                                args=(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                                                      chunk_token_threshold, chunk_size))
                    p.daemon = True
                    p.start()

            if imint_rows:
                print(f"[INFO] Found {len(imint_rows)} new rows to process for imint_ai_analysis")
                for irow in imint_rows:
                    p = multiprocessing.Process(target=imint_ai_analysis_summary_check_threadsafe, args=(irow,))
                    p.daemon = True
                    p.start()

        except Exception as e:
            print("[ERROR] Exception in main loop:")
            traceback.print_exc()

        time.sleep(poll_interval)

    if conn:
        conn.close()


def call_main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai_summary', action='store_true', help='Enable AI summary mode')
    parser.add_argument('--ai_trends', action='store_true', help='Enable AI trends mode')
    parser.add_argument('--ai_change', action='store_true', help='Enable AI change mode')
    parser.add_argument('--ai_change_previous', type=int, metavar='N',
                        help='Compare selected period with same period N years ago')
    parser.add_argument('--chunk_token_threshold', type=int, default=3000, metavar='N',
                        help='Token threshold for chunking / month size limit (default: 3000)')
    args = parser.parse_args()

    ai_summ = False
    ai_trends = False
    ai_change = False
    ai_change_previous = None
    poll_interval = 5

    if args.ai_summary:
        ai_summ = True
        print("AI summary mode is enabled. (Monthly grouped report, per-month LLM analysis)")
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

    chunk_token_threshold = args.chunk_token_threshold
    chunk_size = chunk_token_threshold * 3 // 4
    print(f"Chunk token threshold set to: {chunk_token_threshold}")

    if not ai_summ and not ai_trends and not ai_change and not ai_change_previous:
        sys.exit(1)
    else:
        run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                      chunk_token_threshold, chunk_size)


if __name__ == '__main__':
    while True:
        call_main_func()
