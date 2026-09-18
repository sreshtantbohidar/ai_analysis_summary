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
def update_anythingllm_workspace_model(model_name: str, chat_provider: str = DEFAULT_CHAT_PROVIDER) -> bool:
    global _current_anythingllm_model
    if _current_anythingllm_model == model_name:
        return True

    update_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/update"
    payload = {"chatProvider": chat_provider, "chatModel": model_name}
    headers = {"Authorization": f"Bearer {IFC_LLM_TOKEN}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(update_url, json=payload, headers=headers, timeout=30)
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
        "mode": "query"
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
                                       doc_label: str = "analysis_data") -> str:
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

    Returns:
        LLM response string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{doc_label}_{timestamp}.md"

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

    if rag_response:
        print(f"[INFO] RAG response received ({len(rag_response)} chars)")
        return rag_response

    # Step 4: Fallback to chat mode with inline data
    print("[WARN] RAG query returned no response, falling back to chat mode with inline data")
    return _fallback_chat_with_inline(markdown_content, query, llm_model, filename)


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


def get_data_from_elastic(elastic_query):
    index_name = CUREENT_INDEX_NAME
    try:
        elastic_query = copy.deepcopy(elastic_query)
        size = elastic_query.get("size", None)

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

                # Step 3: Check size — use inline chat mode; chunk if very large
                if total_tokens < chunk_token_threshold:
                    print(f"[DEBUG] Model: {model_name}")
                    temp_llm_response = get_llm_response_with_data_context(
                        markdown_data, full_query, model_name,
                        doc_label=f"{key}_{ai_analysis_summary_id}"
                    )
                else:
                    print(f"[INFO] Large input ({total_tokens} tokens) → Using hierarchical chunking")
                    # Extract data lines from ES hits then chunk them
                    fields = value["fields"]
                    data_lines = []
                    for hit in e_hits:
                        source = hit["_source"]
                        formatted = " | ".join(str(source.get(f, "")) for f in fields)
                        if formatted not in data_lines:
                            data_lines.append(formatted)

                    if not data_lines:
                        continue

                    chunks = build_chunks_from_lines(data_lines, max_tokens=chunk_size)
                    print(f"[INFO] Processing {len(chunks)} chunks...")

                    chunk_summaries = []
                    chunk_loop_start = time.time()

                    for idx, chunk in enumerate(tqdm(chunks, desc="LLM Chunk Processing", unit="chunk")):
                        single_chunk_start = time.time()

                        chunk_markdown = f"# Data Chunk {idx + 1}/{len(chunks)}\n\n"
                        chunk_markdown += "```\n" + chunk + "\n```\n\n"
                        chunk_markdown += f"---\n_Chunk {idx + 1} of {len(chunks)}_"

                        summary = get_llm_response_with_data_context(
                            chunk_markdown, base_prompt + extra_prompt, model_name,
                            doc_label=f"{key}_{ai_analysis_summary_id}_chunk{idx}"
                        )
                        chunk_summaries.append(summary)

                        chunk_time = time.time() - single_chunk_start
                        avg_time = (time.time() - chunk_loop_start) / (idx + 1)
                        remaining = len(chunks) - (idx + 1)
                        eta = avg_time * remaining
                        print(f"[CHUNK {idx+1}/{len(chunks)}] Time: {chunk_time:.2f}s | "
                              f"Avg: {avg_time:.2f}s | ETA: {eta/60:.2f}m")

                    total_chunk_time = time.time() - chunk_loop_start
                    print(f"[INFO] All chunks completed in {total_chunk_time/60:.2f} minutes")

                    # Final consolidation
                    combined_text = "\n\n".join(chunk_summaries)
                    final_markdown = f"# Consolidated Summaries\n\n{combined_text}"
                    temp_llm_response = get_llm_response_with_data_context(
                        final_markdown, base_prompt + extra_prompt, model_name,
                        doc_label=f"{key}_{ai_analysis_summary_id}_consolidated"
                    )

                temp_llm_response = f"<div>{temp_llm_response}</div><br>"
                llm_response += temp_llm_response

        print(f"[DONE] AI analysis complete [{ai_analysis_summary_id}] "
              f"[time={(time.time() - ai_summ_start_time):.2f} s]")

    # ── Build HTML report ──
    tabbed_html = build_tabbed_html([llm_response, ai_trend_response, ai_change_repsonse])
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
                    p = multiprocessing.Process(target=ai_analysis_summary_check_threadsafe,
                                                args=(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                                                      chunk_token_threshold, chunk_size,
                                                      chunk_output_tokens, consolidation_output_tokens))
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
                        help='Token threshold for chunking (default: 3000)')
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

    chunk_token_threshold = args.chunk_token_threshold
    chunk_size = chunk_token_threshold * 3 // 4
    chunk_output_tokens = chunk_token_threshold // 4
    consolidation_output_tokens = chunk_token_threshold // 2
    print(f"Chunk token threshold set to: {chunk_token_threshold}")

    if not ai_summ and not ai_trends and not ai_change and not ai_change_previous:
        sys.exit(1)
    else:
        run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                      chunk_token_threshold, chunk_size,
                                      chunk_output_tokens, consolidation_output_tokens)


if __name__ == '__main__':
    while True:
        call_main_func()
