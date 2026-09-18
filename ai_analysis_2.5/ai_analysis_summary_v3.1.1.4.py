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


def build_tabbed_html(html_strings: List[str], doc_table_html: Optional[str] = None) -> str:
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
def build_doc_id_activity_date_table(hits, query_total=None, date_range=None) -> str:
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

    headers = ["Elastic Doc ID", "Source Type", "File Name", "Ingester Name", "Activity Date"]
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
            _esc(es_id),
            _esc(source_type),
            _esc(file_name),
            _esc(ingester_name),
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
    if query_total is not None and query_total != len(rows):
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
        "reset": True
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


def upload_chunk_and_query_rag(chunk_rows: list, chunk_idx: int, total_chunks: int,
                                start_row_idx: int, query: str, llm_model: str,
                                doc_label: str) -> str:
    """
    Upload one chunk's rows to a fresh AnythingLLM workspace, run a RAG query
    against it, then delete the workspace. Returns the LLM response string.

    Lifecycle:
      1. Create per-chunk workspace with unique slug.
      2. Upload each row as a separate document to that workspace.
      3. Configure topN = len(chunk_rows) so RAG retrieves every row.
      4. Send the user's prompt via /chat?mode=query.
      5. ALWAYS delete the workspace in finally (cleanup even on failure).

    Per-chunk isolation means no cross-chunk retrieval contamination — every
    response is grounded only in its own chunk's rows.
    """
    if not chunk_rows:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_row_idx = start_row_idx + len(chunk_rows) - 1
    # ⭐ v3.1.1.4: the AnythingLLM server derives the workspace slug from its
    # NAME (the slug we request is ignored), so make the name unique and use
    # the slug returned by create_anythingllm_workspace() everywhere below.
    name = (f"Chunk {chunk_idx}/{total_chunks} (rows {start_row_idx}-{end_row_idx}) "
            f"{doc_label} {timestamp}")
    requested_slug = f"{doc_label}_chunk{chunk_idx}_{start_row_idx}to{end_row_idx}_{timestamp}"
    slug = create_anythingllm_workspace(requested_slug, name)
    if not slug:
        return ""

    try:
        # Upload each row to the per-chunk workspace
        uploaded = 0
        for offset, hit in enumerate(chunk_rows):
            idx = start_row_idx + offset
            es_id = hit.get("_id", "") if isinstance(hit, dict) else f"row_{idx}"
            row_text = format_chunk_as_natural_paragraph([hit], idx)
            title = f"row{idx}_{es_id}"
            filename = upload_row_to_workspace(row_text, title, slug)
            if filename:
                uploaded += 1

        if uploaded == 0:
            print(f"[WARN] Chunk {chunk_idx}/{total_chunks}: no rows uploaded")
            return ""

        print(f"[INFO] Chunk {chunk_idx}/{total_chunks}: uploaded {uploaded}/{len(chunk_rows)} rows to '{slug}'")

        # Point this fresh workspace at the LLM model (a newly created workspace
        # has chatModel=None, so a RAG query without this returns nothing), then
        # tune topN so RAG retrieves every row of the chunk.
        update_anythingllm_workspace_model(llm_model, workspace_slug=slug, force=True)
        set_workspace_topn(slug, top_n=uploaded, similarity_threshold=0.0)

        # Brief pause for embeddings to settle
        time.sleep(2)

        # RAG query — scope it explicitly to the chunk
        scoped_query = (
            f"{query}\n\n"
            f"(Scope: this response must summarize ONLY the {uploaded} records "
            f"indexed as rows {start_row_idx} through {end_row_idx} in the "
            f"workspace '{slug}'. Do not generalize to other chunks.)"
        )

        chat_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/chat"
        headers = get_anythingllm_headers()

        response_text = ""
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(
                    chat_url, headers=headers,
                    json={"message": scoped_query.replace("\n", " ").strip(), "mode": "query", "reset": True},
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
                    print(f"[ERROR] Chunk {chunk_idx} RAG status {r.status_code}: {r.text[:200]}")
            except requests.exceptions.Timeout:
                print(f"[ERROR] Chunk {chunk_idx} RAG timeout (attempt {attempt+1})")
            except Exception as e:
                print(f"[ERROR] Chunk {chunk_idx} RAG failed: {e}")
            time.sleep(2 ** attempt)

        if response_text:
            print(f"[INFO] Chunk {chunk_idx}/{total_chunks} RAG response: {len(response_text)} chars")
            return response_text

        # Fallback: inline chat with the chunk data (covers RAG failures)
        print(f"[WARN] Chunk {chunk_idx} RAG returned no response; falling back to inline")
        inline_md = format_chunk_as_natural_paragraph(chunk_rows, start_row_idx)
        return _fallback_chat_with_inline(inline_md, scoped_query, llm_model, f"{slug}.md")

    finally:
        # ALWAYS delete the per-chunk workspace — no leakage between chunks
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


def get_llm_response_direct(query: str, llm_model: str, max_tokens: int = 1200) -> str:
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
            "maxTokens": max_tokens
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
                      batch_size: int = 1000, keep_alive: str = "2m", dedup_stats: dict = None):
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

    if collapse_key:
        print(f"[INFO] Collapse on '{collapse_key}' — scrolling without collapse, "
              f"deduplicating by {unique_dedup_fields}")
    else:
        print(f"[INFO] Deduplicating hits by {unique_dedup_fields}")

    query["size"] = batch_size
    seen = set()
    dedup_count = 0
    total_count = 0

    scroll_id = None
    try:
        response = client.search(index=index_name, body=query, scroll=keep_alive)
        scroll_id = response.get("_scroll_id")

        while True:
            hits = response["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                total_count += 1
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

    print(f"[INFO] Scroll dedup: {total_count} total → {len(seen)} unique, "
          f"dropped {dedup_count} duplicates")
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


def get_data_from_elastic(elastic_query):
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
        dedup_stats = {}
        # ── Collapse queries: strip collapse, scroll, dedup in Python ──
        if has_collapse:
            all_hits = list(
                generic_es_scroll(
                    es,
                    index_name,
                    elastic_query,
                    batch_size=batch_size,
                    dedup_stats=dedup_stats
                )
            )
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
                deduped_hits = _dedup_hits(
                    raw_hits, unique_dedup_fields
                )

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
                dedup_stats=dedup_stats
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


def _combined_estimate_tokens(*parts) -> int:
    """Rough token estimate (same heuristic as count_words_and_tokens) over all
    the given text parts joined together."""
    joined = "\n".join(part for part in parts if part)
    return count_words_and_tokens(joined)[1]

def synthesize_combined_analysis(
    chunk_responses,
    full_query,
    model_name,
    token_budget,
    total_rows,
    club_size,
    consolidation_output_tokens=None,
    max_levels=6
) -> str:
    """
    Synthesize ALL per-chunk analyses into one overall analysis.

    Important design rules:
      1. Every chunk is explicitly identified by group number and record range.
      2. Chunk 1 is NOT treated as the primary context.
      3. Large collections are reduced hierarchically.
      4. Every intermediate summary must preserve coverage information.
      5. The final synthesis must be based on ALL surviving groups.
      6. Each LLM call is stateless because get_llm_response_direct()
         uses reset=True.
    """

    if not chunk_responses or len(chunk_responses) < 2:
        return ""

    n_chunks = len(chunk_responses)

    # Use the caller's configured consolidation budget when available.
    # Fall back to half the combined-analysis threshold.
    output_budget = (
        consolidation_output_tokens
        if consolidation_output_tokens and consolidation_output_tokens > 0
        else max(token_budget // 2, 500)
    )

    print(
        f"[INFO] Combined synthesis starting: "
        f"{n_chunks} chunk analyses, "
        f"{total_rows} records, "
        f"output budget={output_budget} tokens"
    )

    def clean_summary(text):
        """Convert LLM HTML into readable text for the next synthesis pass."""
        if not text:
            return ""

        cleaned = html_to_plain_text(text)

        # Prevent pathological whitespace from consuming context.
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        return cleaned.strip()

    def make_chunk_block(group_num, start, end, html_text):
        """
        Create a strongly labelled block.

        The explicit GROUP marker is important because it prevents the LLM
        from interpreting the first chunk as the main/current context.
        """
        text = clean_summary(html_text)

        return (
            f"=== GROUP {group_num} ===\n"
            f"RECORD RANGE: {start}-{end}\n"
            f"GROUP COVERAGE: records {start} through {end}\n"
            f"GROUP ANALYSIS:\n"
            f"{text}\n"
            f"=== END GROUP {group_num} ==="
        )

    # ------------------------------------------------------------
    # Build the initial blocks.
    # ------------------------------------------------------------
    first_blocks = []

    for group_num, (start, end, html_text) in enumerate(
        chunk_responses, start=1
    ):
        block = make_chunk_block(
            group_num,
            start,
            end,
            html_text
        )

        if block:
            first_blocks.append(block)

    if len(first_blocks) < 2:
        print("[WARN] Fewer than two usable chunk analyses; skipping combined analysis")
        return ""

    # ------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------
    def build_prompt(blocks, level, role):
        all_groups = "\n\n".join(
            f"[INPUT {idx}]\n{block}"
            for idx, block in enumerate(blocks, start=1)
        )

        if role == "interim":
            task = """
Create an INTERMEDIATE combined analysis from ALL of the input groups.

IMPORTANT:
- Every input group must contribute to this analysis.
- Do NOT use the first group as the primary context.
- Do NOT ignore later groups.
- Preserve facts that occur only in one group.
- Preserve specific locations, formations, units, dates, counts,
  coordinates, activities and other concrete observations.
- Identify similarities and differences across the supplied groups.
- Do not invent information that is not present in the supplied groups.
- This is only an intermediate synthesis and will be merged with
  other intermediate syntheses later.
- State the record/group coverage represented by this synthesis.
"""

        else:
            task = f"""
Create ONE FINAL OVERALL ANALYSIS of the COMPLETE DATASET.

The dataset contains {total_rows} records divided into {n_chunks} groups.

IMPORTANT:
- You MUST synthesize ALL supplied groups.
- Do NOT reproduce Group 1 or any single group as the overall answer.
- Do NOT treat the first input as more important than later inputs.
- Look for patterns that emerge ACROSS groups.
- Aggregate counts and date ranges where the source analyses support it.
- Identify recurring locations, formations, activities and other patterns.
- Identify important differences or changes between groups.
- Preserve important group-specific observations when they materially
  affect the overall picture.
- Resolve conflicts cautiously; do not invent facts.
- The result must represent the COMPLETE dataset, not a single group.
"""

        return f"""
{full_query}

{task}

--- BEGIN SYNTHESIS INPUTS ---

{all_groups}

--- END SYNTHESIS INPUTS ---

Coverage requirement:
The output must be based on every supplied input group.

Output only HTML using:
<h2>, <p>, <ul>, <ol>, <li>, <strong>, <b>

Every HTML tag must be properly closed.
Do not output markdown.
"""

    def prompt_cost(blocks, level, role):
        return _combined_estimate_tokens(
            build_prompt(blocks, level, role)
        )

    # ------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------
    def call_llm(blocks, level, role):
        prompt = build_prompt(blocks, level, role)

        estimated = _combined_estimate_tokens(prompt)

        print(
            f"[INFO] Combined synthesis "
            f"level={level}, role={role}, "
            f"inputs={len(blocks)}, "
            f"estimated_tokens={estimated}, "
            f"output_budget={output_budget}"
        )

        response = get_llm_response_direct(
            prompt,
            model_name,
            max_tokens=output_budget
        )

        if response and len(response.strip()) > 20:
            return response.strip()

        print(
            f"[WARN] Combined synthesis returned empty/short response "
            f"at level {level}"
        )
        return ""

    # ------------------------------------------------------------
    # Partition input blocks.
    #
    # We deliberately leave more headroom than the old implementation.
    # This reduces the chance that the LLM effectively attends to only
    # the beginning of a large prompt.
    # ------------------------------------------------------------
    def partition(blocks, level):
        query_cost = _combined_estimate_tokens(full_query)

        # Reserve substantial space for:
        #   - instructions
        #   - group labels
        #   - separators
        #   - model response
        #
        # The previous implementation used only 500 tokens of overhead.
        # That was too aggressive for a context-constrained local model.
        overhead = 1000

        usable_budget = max(
            token_budget - query_cost - overhead,
            1000
        )

        groups = []
        current = []
        current_cost = 0

        for block in blocks:
            block_cost = _combined_estimate_tokens(block)

            # If adding this block would exceed the usable prompt budget,
            # close the current group first.
            if current and current_cost + block_cost > usable_budget:
                groups.append(current)
                current = []
                current_cost = 0

            current.append(block)
            current_cost += block_cost

        if current:
            groups.append(current)

        print(
            f"[INFO] Combined synthesis level {level}: "
            f"{len(blocks)} inputs → {len(groups)} synthesis groups "
            f"(usable budget ~{usable_budget} tokens)"
        )

        return groups

    # ------------------------------------------------------------
    # Recursive hierarchical synthesis
    # ------------------------------------------------------------
    def synthesize(blocks, level=0):
        if not blocks:
            return []

        if level > max_levels:
            print(
                f"[WARN] Maximum combined synthesis depth "
                f"({max_levels}) reached"
            )
            return blocks

        # If everything fits, perform ONE final synthesis.
        if prompt_cost(blocks, level, "final") <= token_budget:
            result = call_llm(blocks, level, "final")

            if result:
                return [result]

            return []

        # Otherwise divide the inputs into manageable groups.
        groups = partition(blocks, level)

        merged = []

        for group_idx, group in enumerate(groups, start=1):

            # A single oversized block cannot safely be sent to the model.
            if prompt_cost(group, level, "interim") > token_budget:
                print(
                    f"[WARN] Synthesis group {group_idx}/{len(groups)} "
                    f"is still too large "
                    f"(~{prompt_cost(group, level, 'interim')} tokens). "
                    f"Skipping this group."
                )
                continue

            partial = call_llm(
                group,
                level,
                "interim"
            )

            if partial:
                merged.append(
                    f"=== INTERMEDIATE GROUP {group_idx} ===\n"
                    f"{partial}\n"
                    f"=== END INTERMEDIATE GROUP {group_idx} ==="
                )

        if not merged:
            print(
                "[WARN] No intermediate synthesis results were produced"
            )
            return []

        # One result means we have only one surviving branch.
        # Do NOT pretend that it is a complete final analysis.
        if len(merged) == 1:
            print(
                "[WARN] Only one intermediate synthesis survived; "
                "combined analysis cannot be guaranteed to cover all groups"
            )
            return []

        # Merge the intermediate analyses recursively.
        return synthesize(
            merged,
            level + 1
        )

    # ------------------------------------------------------------
    # Start synthesis
    # ------------------------------------------------------------
    final_results = synthesize(first_blocks, 0)

    if not final_results:
        print("[WARN] Overall combined analysis could not be generated")
        return ""

    final = final_results[0].strip()

    print(
        f"[INFO] Overall combined analysis generated: "
        f"{len(final)} chars from {n_chunks} chunk analyses"
    )

    return final

# ──────────────────────────────────────────────
# ⭐ Modified: ai_analysis_summary_check with Query Mode
# ──────────────────────────────────────────────
def ai_analysis_summary_check(row, ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                               chunk_token_threshold=3000, chunk_size=2200,
                               chunk_output_tokens=800, consolidation_output_tokens=1500):
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
    dedup_stats = e_response.get("dedup_stats", {}) if e_response else {}

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
                            activity_date = datetime.fromisoformat(single_json['@timestamp']).strftime("%Y-%m-%d")
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
                # ranges previously fell into the old hierarchical + RAG branch,
                # which produced a small, partial summary with no per-chunk /
                # combined sections. (⭐ v3.1.1.4)
                print(f"[DEBUG] Model: {model_name}")

                # ⭐ NEW (v3.1.1.4): per-chunk upload+RAG in fresh workspaces.
                # The dataset is split into chunks sized to fit within the
                # LLM's effective per-chunk prompt budget (chunk_size tokens).
                # Each chunk gets:
                #   1. its own AnythingLLM workspace (no cross-contamination)
                #   2. one row-upload per ES row, then a single RAG query
                #   3. workspace deletion in finally (no persistent state)
                # Chunk summaries are concatenated in the AI Summary section
                # with explicit "Records A-B" headers so analysts can see each
                # chunk's output separately.
                #
                # Club size policy:
                #   - If AI_SUMMARY_CLUB_SIZE env var is set, use it (operator
                #     override — useful when an operator knows the row token
                #     density is unusually high or low, or wants fewer, larger
                #     chunks on very large ranges).
                #   - Otherwise derive from chunk_size (the per-chunk upload
                #     token budget). We assume ~400 tokens per structured-data
                #     row (location + date + category + description) and back
                #     off to keep headroom for the query + response.
                #     Minimum 1, no maximum.
                _override = _safe_int(os.getenv("AI_SUMMARY_CLUB_SIZE"))
                if _override is not None and _override > 0:
                    CLUB_SIZE = _override
                    club_reason = f"env override AI_SUMMARY_CLUB_SIZE={_override}"
                else:
                    CLUB_SIZE = max(1, chunk_size // 400)
                    club_reason = (
                        f"derived from chunk_size={chunk_size} tokens "
                        f"(~400 tokens/row)"
                    )

                n_rows = len(e_hits)
                n_chunks = max(1, (n_rows + CLUB_SIZE - 1) // CLUB_SIZE)
                print(f"[INFO] Per-chunk upload+RAG: {n_rows} rows → {n_chunks} chunks "
                      f"of up to {CLUB_SIZE} rows each ({club_reason})")

                # ⭐ NEW (v3.1.1.4): Sort by activity_date DESC so the latest
                # records come first in each chunk and in the report. Rows
                # without activity_date sort to the end. Within a tie we
                # preserve ES's natural order (stable sort).
                def _activity_date_sort_key(hit):
                    src = hit.get("_source") if isinstance(hit, dict) else None
                    if not isinstance(src, dict):
                        return (1, "")  # missing rows go last
                    ad = src.get("activity_date") or src.get("@timestamp") or ""
                    if not ad:
                        return (1, "")
                    # ISO date strings compare correctly; latest sorts first
                    # because of reverse=True.
                    return (0, ad)

                e_hits_sorted = sorted(e_hits, key=_activity_date_sort_key, reverse=True)
                e_hits = e_hits_sorted  # use the sorted order for chunking + downstream

                chunk_responses = []
                chunk_loop_start = time.time()
                for chunk_idx in range(n_chunks):
                    start = chunk_idx * CLUB_SIZE
                    end = min(start + CLUB_SIZE, n_rows)
                    chunk_rows = e_hits[start:end]
                    if not chunk_rows:
                        continue
                    chunk_response = upload_chunk_and_query_rag(
                        chunk_rows, chunk_idx + 1, n_chunks,
                        start_row_idx=start + 1,  # 1-based for human-friendly output
                        query=full_query,
                        llm_model=model_name,
                        doc_label=f"{key}_{ai_analysis_summary_id}",
                    )
                    if chunk_response:
                        chunk_responses.append((start + 1, end, chunk_response))
                    elapsed = time.time() - chunk_loop_start
                    avg = elapsed / (chunk_idx + 1)
                    eta = avg * (n_chunks - chunk_idx - 1)
                    print(f"[CHUNK {chunk_idx+1}/{n_chunks}] done in {elapsed:.1f}s "
                          f"| avg {avg:.1f}s | ETA {eta/60:.1f}m")

                # Concatenate chunk summaries with explicit boundaries so
                # the report shows each chunk's analysis as a separate
                # section. The LLM has been told to scope each response;
                # we wrap them in a master section so they're visually
                # grouped in the report.
                if chunk_responses:
                    # Build per-chunk sections, each with a clear header
                    # showing the records count and the activity_date range
                    # covered by that chunk. We pull activity_date from the
                    # ES hits we already have so the header reflects what
                    # actually went into the chunk.
                    def _chunk_dates(rows):
                        dates = []
                        for h in rows:
                            src = h.get("_source") if isinstance(h, dict) else None
                            if isinstance(src, dict):
                                ad = src.get("activity_date") or src.get("@timestamp")
                                if ad:
                                    dates.append(str(ad)[:10])
                        if not dates:
                            return ("?", "?")
                        return (min(dates), max(dates))

                    chunk_sections = []
                    for group_num, (chunk_start, chunk_end, chunk_html_text) in enumerate(chunk_responses, 1):
                        chunk_rows = e_hits_sorted[chunk_start - 1:chunk_end]
                        d_min, d_max = _chunk_dates(chunk_rows)
                        chunk_row_count = chunk_end - chunk_start + 1
                        section = (
                            "<h3>"
                            f"Group {group_num} (Record {chunk_start} to {chunk_end}) "
                            f"(Containing {chunk_row_count} records, from {d_min} to {d_max})"
                            "</h3>"
                            + chunk_html_text
                        )
                        chunk_sections.append(section)
                    chunk_html = "".join(chunk_sections)
                    rows_total = sum(end - start + 1 for start, end, _ in chunk_responses)
                    n_chunks_done = len(chunk_responses)

                    # ⭐ NEW (v3.1.1.4): Combined analysis pass. After each
                    # chunk has been analyzed in isolation, take all chunk
                    # summaries as input to chat calls that synthesise them
                    # with cross-chunk context. Sets whose summaries fit one
                    # prompt are merged in a single call; larger sets are
                    # merged in stages inside synthesize_combined_analysis.
                    combined_html = ""
                    if n_chunks_done > 1:
                        try:
                            combined_response = synthesize_combined_analysis(
                                chunk_responses,
                                full_query=full_query,
                                model_name=model_name,
                                token_budget=chunk_token_threshold,
                                total_rows=rows_total,
                                club_size=CLUB_SIZE,
                                consolidation_output_tokens=consolidation_output_tokens,
                            )
                            if combined_response:
                                combined_html = (
                                    "<h2>Overall Analysis</h2>"
                                    "<p style=\"color:#6c757d;font-size:13px;\">"
                                    f"Synthesis of all {n_chunks_done} groups. "
                                    f"Elasticsearch returned {dedup_stats.get('total', rows_total)} records, "
                                    f"of which {dedup_stats.get('unique', rows_total)} were unique after deduplication "
                                    f"and {dedup_stats.get('dropped', 0)} duplicates were removed."
                                    "</p>"
                                    + f"<div>{combined_response}</div>"
                                )
                        except Exception as e:
                            print(f"[WARN] Overall analysis failed: {e}")

                    master = (
                        "<div>"
                        f"<p style=\"color:#6c757d;font-size:13px;\">"
                        f"For hardware constraint {rows_total} records were analyzed in "
                        f"{n_chunks_done} groups containg {CLUB_SIZE} records each. "
                        f"</p>"
                        + (combined_html if combined_html else "")
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
    date_range = None
    if elastic_query:
        date_range = extract_date_range(elastic_query)
    doc_table_html = build_doc_id_activity_date_table(
        e_hits, query_total=query_total, date_range=date_range
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
                                          chunk_output_tokens=800, consolidation_output_tokens=1500):
    try:
        conn = postgres_connection()
        with conn.cursor() as cursor:
            filter_json, ai_analysis_summary_id, search_form_type = row
            ai_analysis_summary_check((filter_json, ai_analysis_summary_id, search_form_type),
                                       ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                                       chunk_token_threshold, chunk_size,
                                       chunk_output_tokens, consolidation_output_tokens)
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
                                  chunk_output_tokens=800, consolidation_output_tokens=1500):
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
                        consolidation_output_tokens
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
        prog='ai_analysis_summary_v3.1.1.4',
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
                        help='Enable AI summary mode — generates LLM-based analysis '
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
                        help='Token threshold for chunking large ES results before '
                             'sending to LLM. Defaults to 60%% of --llm_context_tokens. '
                             'Inputs exceeding this limit are split into chunks, summarized '
                             'individually, then consolidated.')
    parser.add_argument('--llm_context_tokens', '--llm-context-tokens',
                        type=int, default=None, metavar='N',
                        help='REQUIRED: total context-window size of the LLM in tokens '
                             '(e.g. 8192 for an 8B model, 32768 for a 31B/32B model). '
                             'Drives the per-chunk token threshold (60%% of this value) '
                             'and the combined-analysis prompt size. If not given, '
                             'the env var LLM_CONTEXT_TOKENS is used. The script will '
                             'exit with an error if neither is set, because guessing '
                             'from the model name is unreliable across model families.')
    args = parser.parse_args()

    ai_summ = False
    ai_trends = False
    ai_change = False
    ai_change_previous = None
    poll_interval = 5

    if args.ai_summary:
        ai_summ = True
        print("AI summary mode is enabled. (Using document upload + RAG query mode)")
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
        # Default to 60% of the model's context window, leaving room for the
        # system prompt, the user's query, and the response. So a 32K-context
        # model gets a ~19K threshold; an 8K model gets ~4.9K.
        chunk_token_threshold = int(llm_context_tokens * 0.6)

    chunk_size = chunk_token_threshold * 3 // 4
    chunk_output_tokens = chunk_token_threshold // 4
    consolidation_output_tokens = chunk_token_threshold // 2

    print(f"LLM context window: {llm_context_tokens} tokens (model: {LLM_MODEL})")
    print(f"Chunk token threshold set to: {chunk_token_threshold} (60% of context)")
    print(f"  Per-chunk upload size target: {chunk_size} tokens (75% of threshold)")
    print(f"  Per-chunk LLM output budget: {chunk_output_tokens} tokens (25% of threshold)")
    print(f"  Combined-analysis output budget: {consolidation_output_tokens} tokens (50% of threshold)")

    if not ai_summ and not ai_trends and not ai_change and not ai_change_previous:
        sys.exit(1)
    else:
        run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                      chunk_token_threshold, chunk_size,
                                      chunk_output_tokens, consolidation_output_tokens)


if __name__ == '__main__':
    while True:
        call_main_func()
