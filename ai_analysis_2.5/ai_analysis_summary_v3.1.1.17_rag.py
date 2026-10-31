import argparse
import copy
import difflib
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
from publication_summary_client import build_publication_summary_section  # ⭐ v3.1.1.15: Publication Summary section
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

# ⭐ NEW (robustness): rows that crashed once and were requeued; a second crash
# marks them done with an error report instead of looping forever.
_requeued_failed_rows = set()
# ⭐ v3.1.1.15 fix: same retry bookkeeping for IMINT rows, which previously had
# no failure handling at all (a crash left them locked at status=2 forever).
_requeued_failed_imint_rows = set()
MAX_RETRIES = 3

# Fallback model if the configured model returns empty responses
FALLBACK_LLM_MODEL = "gemma2:9b-instruct-q8_0"

# ⭐ v3.1.1.15: prefer models that are actually installed on the LLM host.
# ai_model_master may name a model that is not pulled on the Ollama server
# (e.g. "gemma4:12b" while only "gemma4:12b-it-q4_K_M" exists). Resolving the
# name up front avoids a guaranteed 404 + wasted retry on every report.
_INSTALLED_MODELS_CACHE = None


def _list_installed_models():
    """Return the model names installed on the LLM host (cached per process)."""
    global _INSTALLED_MODELS_CACHE
    if _INSTALLED_MODELS_CACHE is not None:
        return _INSTALLED_MODELS_CACHE
    tags = []
    try:
        r = requests.get(f"http://{LLM_IP}:{LLM_PORT}/api/tags", timeout=10)
        if r.status_code == 200:
            tags = [m.get("name", "") for m in (r.json() or {}).get("models", [])]
            tags = [t for t in tags if t]
        else:
            print(f"[WARN] Could not list Ollama models (status {r.status_code})")
    except Exception as e:
        print(f"[WARN] Could not list Ollama models: {e}")
    _INSTALLED_MODELS_CACHE = tags
    return tags


def resolve_available_model(model_name):
    """Map a configured model name to one that is installed on the LLM host.

    Resolution order:
      1. exact match -> use as-is;
      2. same repository/tag base (e.g. "gemma4:12b" -> "gemma4:12b-it-q4_K_M");
      3. FALLBACK_LLM_MODEL if installed;
      4. first installed non-embedding model.

    When the host cannot be reached the name is returned unchanged so the
    existing runtime fallback still applies.
    """
    if not model_name:
        return model_name
    tags = _list_installed_models()
    if not tags:
        return model_name
    if model_name in tags:
        return model_name
    base = str(model_name).split(":")[0]
    same_base = [t for t in tags if t.split(":")[0] == base]
    if same_base:
        chosen = sorted(same_base, key=len)[0]
        print(f"[INFO] Model '{model_name}' not installed; using available '{chosen}'")
        return chosen
    if FALLBACK_LLM_MODEL in tags:
        print(f"[INFO] Model '{model_name}' not installed; using fallback '{FALLBACK_LLM_MODEL}'")
        return FALLBACK_LLM_MODEL
    for t in tags:
        if "embed" not in t.lower():
            print(f"[INFO] Model '{model_name}' not installed; using available '{t}'")
            return t
    return model_name

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
# ⭐ v3.1.1.20: an Overall Analysis batch whose answer budget falls below this
# floor is split instead of answered — a starved model re-emits the supplied
# source records rather than analysing them (seen in production).
MIN_OVERALL_BATCH_OUTPUT_TOKENS = 1536

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


def _strip_record_labels_from_report(html: str, include_citations: bool = True) -> str:
    if not html or include_citations:
        return html
    parts = re.split(r"(?is)(<style\b[^>]*>.*?</style>|<script\b[^>]*>.*?</script>|<[^>]+>|https?://[^\s<]+)", html)
    for idx in range(0, len(parts), 2):
        parts[idx] = _strip_record_labels_in_text(parts[idx], include_citations=False)
    return "".join(parts)


def _strip_record_labels_in_text(html: str, include_citations: bool = True) -> str:
    """Remove 'Record N' / 'Item N' labels from the final report HTML.

    ⭐ NEW (v3.1.1.15): the report must not display record-number labels such
    as 'Record 24', 'Item 24' or 'Records 5 to 10' unless the row was requested
    WITH citations (is_citation = TRUE). When include_citations is True the
    labels are kept because they map the report text to the citation table; when
    include_citations is False the labels are removed entirely (they are neither
    kept as 'Record' nor renamed to 'Item').

    This runs on the fully assembled report HTML as a final safety net, so
    neither LLM prose nor the code-generated chunk headings can surface the
    wording. It does NOT touch citation numbers such as (3, 45, 67), record
    counts such as '50 records', or the 'Record Source Details' heading -
    none of those has a number directly after the word 'Record' / 'Item'.
    """
    if not html or include_citations:
        return html
    text = html
    # Ranges first: drop "Records 5 to 10" / "records from 75 to 102" /
    # "Items 5 through 10" / "Record 5-10".
    text = re.sub(
        r"(?i)\b(?:Records?|Items?)\s+(?:from\s+)?\d+\s*(?:to|through|[-\u2013\u2014])\s*\d+",
        "",
        text,
    )
    # Labels (single or list): drop "Record 24" / "Record No. 7" / "Record-24"
    # and lists such as "records 21, 22 and 23" or
    # "records 101, 103, 109, 114, 120 and 123". The WHOLE list is consumed so
    # the removal cannot leave dangling numbers behind.
    text = re.sub(
        r"(?i)\b(?:Records?|Items?)\s*[,\-\u2013\u2014:.]?\s*(?:No\.?\s*)?\d+"
        r"(?:\s*,\s*\d+)*(?:\s*,?\s*and\s*\d+)*",
        "",
        text,
    )
    # ⭐ NEW (v3.1.1.18): explicit "Citation No. N" labels and "Citations:"
    # lines are also citation syntax and must not survive when citations are
    # disabled. Removing "Record N" above can leave an empty "()" behind (the
    # model wrote "Port Blair (Record 3)"), so the parens are cleaned next.
    text = re.sub(r"(?i)\bCitations?\s*No\.?\s*[-:]?\s*\d+", "", text)
    text = re.sub(r"(?i)\bCitations?\s*:\s*", "", text)
    text = re.sub(r"(?i)\bCitations?\s*\b", "", text)
    # Parenthesized citation number lists, e.g. "(23, 45, 67)", "(,23,45,67)"
    # or "(3, 4, and 5)".
    text = re.sub(
        r"\(\s*[,\s]*\d+(?:\s*,\s*\d+)*(?:\s*,?\s*and\s*\d+)*\s*\)",
        "",
        text,
    )
    # Empty "()" / "( )" placeholders left by the model or by label removal.
    text = re.sub(r"\(\s*\)", "", text)
    # Orphaned number lists left at the start of a paragraph (after a tag) or
    # right after a sentence period - the residue of an earlier "records N"
    # removal, e.g. ", 103, and 109 indicate ..." or "deployments., 22, and 23".
    _cite_list = r"\d+(?:\s*,\s*\d+)*(?:\s*,?\s*and\s*\d+)"
    text = re.sub(r"(?<=>)\s*,?\s*" + _cite_list + r"\s*", "", text)
    text = re.sub(r"(?<=[.!?])\s*,?\s*" + _cite_list + r"\s+", " ", text)
    # Tidy the separators / doubled spaces the removals leave behind.
    text = re.sub(r"\s*[\u2013\u2014]\s*(?=[(<])", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # A list whose label was removed can strand a ", 22, and 23" fragment after
    # a period; collapse those and any repeated separators.
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r"([(\[])\s*[,\s]+", r"\1", text)
    text = re.sub(r"\s*,\s*(?=[,.;:)\]<])", "", text)
    text = re.sub(r"\s*\b(?:and|or)\b\s*([.,;:)])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # A second pass in case a removal exposed a new empty pair.
    text = re.sub(r"\(\s*\)", "", text)
    return text


# ⭐ NEW (v3.1.1.19): analyst-facing dates use DD-MM-YYYY. Only the presentation
# copy is rewritten — Elasticsearch query bodies are built before this runs and
# keep their ISO (YYYY-MM-DD) form.
_DISPLAY_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

# ⭐ NEW (v3.1.1.20): the LLM writes narrative dates in long form ("October 1,
# 2026") even when the analyst asked for DD-MM-YYYY. The year-bearing long
# forms are converted too; month-only references ("October 2026") carry no day
# and are left as written rather than guessed.
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_MONTH_NUMBER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_FIRST_DATE_RE = re.compile(
    r"(?i)\b(" + _MONTH_NAMES + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+(\d{4})\b"
)
_DAY_FIRST_DATE_RE = re.compile(
    r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_NAMES + r")\.?\s*,?\s+(\d{4})\b"
)


def _display_date(day: int, month: int, year: int) -> str:
    return f"{day:02d}-{month:02d}-{year:04d}"


def _convert_long_form_dates(text: str) -> str:
    """Rewrite year-bearing long-form dates as DD-MM-YYYY."""
    def _month_first(m):
        month = _MONTH_NUMBER.get(m.group(1).lower()[:3])
        if not month:
            return m.group(0)
        return _display_date(int(m.group(2)), month, int(m.group(3)))

    def _day_first(m):
        month = _MONTH_NUMBER.get(m.group(2).lower()[:3])
        if not month:
            return m.group(0)
        return _display_date(int(m.group(1)), month, int(m.group(3)))

    text = _MONTH_FIRST_DATE_RE.sub(_month_first, text)
    return _DAY_FIRST_DATE_RE.sub(_day_first, text)


def format_dates_for_display(text: str) -> str:
    """Render dates as DD-MM-YYYY for analyst-facing output.

    Applies to bare ISO dates (``2026-09-12``), full ISO timestamps
    (``2026-09-12T08:30:00`` / ``2026-09-12 08:30:00``) and long-form prose
    dates (``October 1, 2026`` / ``1 October 2026``), leaving the time and
    separators untouched. Month-only references (``October 2026``) are left
    as written because they carry no day.
    """
    if not text:
        return text
    # Rewrite visible prose only; dates in href/src, CSS, scripts and URLs
    # are resource identifiers and must remain byte-for-byte unchanged.
    protected = re.compile(r"(?is)(<style\b[^>]*>.*?</style>|<script\b[^>]*>.*?</script>|<[^>]+>|https?://[^\s<]+)")
    parts = protected.split(text)
    for idx in range(0, len(parts), 2):
        visible = _DISPLAY_DATE_RE.sub(
            lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}", parts[idx]
        )
        parts[idx] = _convert_long_form_dates(visible)
    return "".join(parts)


def build_tabbed_html(html_strings: List[str], doc_table_html: Optional[str] = None,
                      publication_summary_html: Optional[str] = None,
                      equipment_specifications_html: Optional[str] = None,
                      formation_details_html: Optional[str] = None,
                      imagery_links_html: Optional[str] = None) -> str:
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

    # ⭐ NEW: "IMINT Imagery" section — the image paths stored in each IMINT
    # record's all_images field, rendered deterministically from the source
    # records (the LLM only receives them as evidence and does not reproduce
    # them). Appended only when the caller supplies a non-empty section.
    if imagery_links_html:
        sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-imint-imagery">
                    <h2>IMINT Imagery</h2>
                </div>
                <div class="section-content">
                    {imagery_links_html}
                </div>
            </section>''')

    # ⭐ NEW (v3.1.1.14): "Equipment Specifications" section — appended only for
    # equipment analyses whose retained records reference one or more equipments.
    if equipment_specifications_html:
        sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-equipment-specs">
                    <h2>Equipment Specifications</h2>
                </div>
                <div class="section-content">
                    {equipment_specifications_html}
                </div>
            </section>''')

    # ⭐ NEW (v3.1.1.23): "Enemy Formation Details" section — appended directly
    # after Equipment Specifications, only when the caller supplies a non-empty
    # section (i.e. the report's ORBAT formations were looked up successfully).
    if formation_details_html:
        sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-formation-details">
                    <h2>Enemy Formation Details</h2>
                </div>
                <div class="section-content">
                    {formation_details_html}
                </div>
            </section>''')

    # ⭐ NEW (v3.1.1.15): "Publication Summary" section — inserted BEFORE the
    # Record Source Details (citation) table, only when the caller supplies a
    # non-empty section (i.e. summary_type_publication was TRUE and the
    # publication API responded successfully).
    if publication_summary_html:
        sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-publication">
                    <h2>Publication Summary</h2>
                </div>
                <div class="section-content">
                    {publication_summary_html}
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

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
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
    .equipment-spec-table {{
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        margin: 0;
    }}
    .equipment-spec-table th,
    .equipment-spec-table td {{
        border: 1px solid #d9e0e7;
        padding: 12px 14px;
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
    }}
    .equipment-spec-table th {{
        background: #eef4fa;
        color: #004085;
    }}
    .equipment-spec-table th:first-child,
    .equipment-spec-table td:first-child {{
        width: 32%;
    }}
    .equipment-spec-table pre {{
        margin: 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-family: inherit;
    }}
    /* v3.1.1.17: clear horizontal separator line between equipment rows so
       each equipment's specification block is visually distinct. */
    .equipment-spec-table tbody tr + tr > td {{
        border-top: 2px solid #b9c6d3;
    }}
    /* ⭐ Publication evidence — every source is rendered in full: document,
       page/section, excerpt, source image, source table and links. */
    .publication-summary .muted {{
        color: #6c757d;
        font-weight: normal;
        font-size: .88em;
    }}
    .publication-sources {{
        margin: 12px 0 0 0;
        padding-left: 22px;
    }}
    .publication-source {{
        margin: 0 0 18px 0;
        padding: 12px 14px;
        background: #fafcfe;
        border: 1px solid #e3e8ee;
        border-radius: 4px;
    }}
    .publication-source-head {{
        font-size: .97em;
        margin-bottom: 4px;
    }}
    .publication-source-meta {{
        color: #6c757d;
        font-size: .8em;
        margin-bottom: 6px;
    }}
    .publication-source blockquote {{
        margin: 8px 0;
        padding: 8px 12px;
        background: #fff;
        border-left: 3px solid #b9c6d3;
        font-size: .9em;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}
    .publication-source-image {{
        margin: 10px 0;
    }}
    .publication-source-image img {{
        max-width: 100%;
        height: auto;
        border: 1px solid #d9e0e7;
        border-radius: 4px;
        background: #fff;
    }}
    .publication-source-image figcaption {{
        color: #6c757d;
        font-size: .82em;
        margin-top: 4px;
    }}
    .publication-source-table {{
        margin: 10px 0;
    }}
    .publication-source-table-label {{
        color: #004085;
        font-weight: 600;
        font-size: .82em;
        margin-bottom: 4px;
    }}
    .publication-source-table .table-scroll {{
        overflow-x: auto;
    }}
    .publication-source-table table {{
        width: 100%;
        border-collapse: collapse;
        font-size: .85em;
        background: #fff;
    }}
    .publication-source-table th,
    .publication-source-table td {{
        border: 1px solid #d9e0e7;
        padding: 6px 8px;
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
    }}
    .publication-source-table th {{
        background: #eef4fa;
        color: #004085;
    }}
    .publication-source-links {{
        margin-top: 6px;
        font-size: .85em;
    }}
    .publication-source-links a {{
        color: #004085;
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
        .equipment-spec-table,
        .equipment-spec-table thead,
        .equipment-spec-table tbody,
        .equipment-spec-table tr,
        .equipment-spec-table th,
        .equipment-spec-table td {{ display: block; width: auto; }}
        .equipment-spec-table thead {{ display: none; }}
        .equipment-spec-table tr {{ margin-bottom: 14px; }}
        .equipment-spec-table tr + tr {{ border-top: 2px solid #b9c6d3; margin-top: 0; }}
        .equipment-spec-table td:first-child {{
            width: auto;
            background: #eef4fa;
            font-weight: 600;
        }}
    }}
    /* ⭐ v3.1.1.16: PDF/print output — proper page margins and no clipped
       content. A4 page with safe margins; content blocks are not sliced
       mid-section and long tables/words wrap inside the page. */
    @page {{
        size: A4;
        margin: 18mm 14mm;
    }}
    @media print {{
        html, body {{
            width: auto;
            max-width: none;
            background: #fff;
            font-size: 12px;
            line-height: 1.45;
        }}
        .report-container {{
            max-width: none;
            margin: 0;
            padding: 0;
        }}
        header {{
            padding: 10px 0;
        }}
        .section-header {{
            box-shadow: none;
            page-break-after: avoid;
            break-after: avoid-page;
        }}
        .section-content {{
            box-shadow: none;
            border-radius: 0;
            padding: 10px 0;
            overflow: visible;
        }}
        h2, h3, h4 {{
            page-break-after: avoid;
            break-after: avoid-page;
        }}
        table {{
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            overflow: visible;
        }}
        th, td {{
            word-wrap: break-word;
            overflow-wrap: anywhere;
            white-space: normal;
        }}
        p, li, div {{
            overflow-wrap: anywhere;
        }}
        pre, code {{
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }}
        footer {{
            page-break-inside: avoid;
        }}
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
# ⭐ NEW (v3.1.1.14): Equipment Specifications helper
# ──────────────────────────────────────────────
EQUIPMENT_SPEC_INDEX = CUREENT_INDEX_NAME
EQUIPMENT_SPEC_MAX_LOOKUPS = 50
# ⭐ NEW (v3.1.1.18): the live index stores equipment specification sheets under
# TWO injester_name spellings. The original catalog query only matched
# "equipment_injester", so every equipment whose sheet was ingested under the
# "equipment_ingester" spelling (MQ-1, RQ-4, J-20, F-22, F-35, Mi-10, ...) was
# silently absent from the Equipment Specifications section. Both spellings
# carry valid sheets, so both are searched.
EQUIPMENT_SPEC_INJESTERS = ("equipment_injester", "equipment_ingester")
EQUIPMENT_SPEC_RESPONSE_FIELDS = (
    "equipment_name", "equipment_type", "crew", "additional_crew", "weight",
    "weight_unit", "effective_firing_range", "max_firing_range",
    "operational_range", "range_unit", "max_speed", "speed_unit", "gun_name",
    "gun_type", "gun_count", "gun_radius", "gun_radius_unit", "gun_relation",
    "aux_name", "aux_type", "aux_count", "aux_radius", "aux_relation",
    "field_craft_system", "calibre", "calibre_unit", "mag_cap", "rate_of_fire",
    "rate_of_fire_unit", "no_of_tube", "size_of_tube", "size_of_tube_unit",
    "missile_name", "missile_type", "missile_count", "missile_range",
    "missile_range_unit", "missile_speed", "missile_speed_unit", "warhead_weight",
    "warhead_weight_unit", "flt_ceiling", "flt_ceiling_unit", "guidence",
    "bomb_type", "mtow", "mtow_weight", "muzzle_velocity",
    "muzzle_velocity_unit", "max_payload", "max_payload_unit",
    "equipment_name_keywords", "equipment_type_keywords",
    # ⭐ v3.1.1.15: the live index stores the specification sheet as a single
    # pre-formatted text block in `specifications` (the flat crew/weight/...
    # fields above exist in the Additional-Equipments workbook export but are
    # not populated in Elasticsearch). `equipment_key` is the canonical
    # "Name(type)" key and is useful when a name has several variants.
    "equipment_key", "specifications",
)


def _collect_equipment_names(e_hits):
    """
    Return an ordered list of unique {"name", "type"} equipment entries found
    in the given ES hits.

    Handles the shapes seen in the live index:
      * explicit list fields (equipment_names / imagery_data.equipment_names),
      * the nested ``equipment`` specification object,
      * flat / dotted keys (equipment_name, equipment.equipment_name, ...),
        including the comma-joined equipment_name produced by the General
        Analysis form_id merge.
    """
    # Unique by equipment name (case-insensitive); keep the first non-empty
    # type seen for that name so empty type-variants don't duplicate entries.
    # ``query_name`` preserves the raw value (some index values carry trailing
    # spaces) so the exact-term lookup still matches; ``name`` is display-only.
    order = []
    types_by_name = {}
    query_names = {}

    def _add(name, etype=""):
        if name in (None, "", "None", "nan"):
            return
        raw_name = str(name)
        display = raw_name.strip()
        if not display:
            return
        etype = "" if etype in (None, "None", "nan") else str(etype).strip()
        key = display.lower()
        if key not in types_by_name:
            types_by_name[key] = etype
            query_names[key] = raw_name
            order.append((display, key))
        elif not types_by_name[key] and etype:
            types_by_name[key] = etype

    for hit in e_hits or []:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source", {})
        if not isinstance(source, dict):
            continue

        # 1) Explicit list-of-equipment fields.
        for list_key in ("equipment_names", "imagery_data.equipment_names"):
            entries = source.get(list_key)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        _add(entry.get("equipment_name") or entry.get("name"),
                             entry.get("equipment_type") or entry.get("equipement_type")
                             or entry.get("type", ""))
                    else:
                        _add(entry)

        # 2) Nested equipment object (carries equipment.specifications).
        nested = source.get("equipment")
        for node in (nested if isinstance(nested, list) else [nested]):
            if isinstance(node, dict):
                _add(node.get("equipment_name") or node.get("name"),
                     node.get("equipment_type") or node.get("equipement_type")
                     or node.get("type", ""))

        # 3) Flat / dotted keys (comma-separated values allowed).
        for name_key, type_key in (
            ("equipment_name", "equipment_type"),
            ("equipment_name", "equipement_type"),
            ("equipment.equipment_name", "equipment.equipment_type"),
            ("deployment.equipment_name", "deployment.equipment_type"),
        ):
            raw_name = source.get(name_key)
            if raw_name in (None, "", "None"):
                continue
            raw_type = source.get(type_key, "")
            names = raw_name if isinstance(raw_name, list) else str(raw_name).split(",")
            types = raw_type if isinstance(raw_type, list) else str(raw_type).split(",")
            for idx, nm in enumerate(names):
                if len(types) == len(names):
                    etype = types[idx]
                else:
                    etype = types[0] if len(types) == 1 else ""
                _add(nm, etype)

    return [
        {"name": display, "query_name": query_names[key], "type": types_by_name[key]}
        for display, key in order
    ]


# ⭐ NEW (v3.1.1.15): single-query, process-cached equipment specification catalog.
EQUIPMENT_SPEC_CATALOG_SIZE = 5000
_EQUIPMENT_SPEC_CATALOG_CACHE = None
_EQUIPMENT_SPEC_INDEX_CACHE = None


def _search_equipment_spec_catalog():
    """Fetch the ENTIRE equipment specification catalog with ONE ES query.

    NOTE 1: no `sort` is applied. The previous implementation sorted on
    `timestamp`, which is NOT mapped in this index; Elasticsearch then rejects
    the search with 400 ('No mapping found for [timestamp] in order to sort on')
    whenever it matches a document, and the section rendered empty.

    NOTE 2: the query selects every equipment specification document that
    actually carries a `specifications` sheet (~590 docs, cached once per
    process). The previous `must_not form_status=5` filter excluded the very
    documents that hold the sheets, which is why the section rendered empty;
    equipment sheets are stored under form_status 5, not filtered out.

    NOTE 3 (⭐ v3.1.1.18): the sheets are stored under TWO `injester_name`
    spellings — "equipment_injester" and "equipment_ingester". The query must
    match both (terms) or the equipment ingested under the second spelling is
    missing from the report.

    Set AI_SUMMARY_EQUIP_SPEC_EXCLUDE_FORM_STATUS_5=1 to restore the legacy
    behaviour of excluding form_status=5 rows (the section will then only see
    the ~682 non-spec rows and render very little).
    """
    query_body = {
        "size": EQUIPMENT_SPEC_CATALOG_SIZE,
        "query": {
            "bool": {
                "must": [
                    {"terms": {"injester_name.keyword": list(EQUIPMENT_SPEC_INJESTERS)}},
                    {"exists": {"field": "specifications"}},
                ],
            }
        },
        "_source": list(EQUIPMENT_SPEC_RESPONSE_FIELDS),
    }
    if os.getenv("AI_SUMMARY_EQUIP_SPEC_EXCLUDE_FORM_STATUS_5", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        query_body["query"]["bool"]["must_not"] = [{"term": {"form_status": 5}}]
    _t0 = time.time()
    response = es.search(index=EQUIPMENT_SPEC_INDEX, body=query_body)
    _ms = (time.time() - _t0) * 1000
    hits = response.get("hits", {}).get("hits", [])
    print(
        f"[INFO] Equipment spec catalog: {len(hits)} doc(s) in ONE query "
        f"(ES took={response.get('took', '?')}ms, wall={_ms:.0f}ms)"
    )
    return hits


def _search_equipment_spec_catalog_for(requests):
    """Fetch ONLY the specification docs for the equipment named in this report.

    ⭐ NEW (v3.1.1.18): instead of loading the whole equipment catalog, this
    issues ONE filtered ES query selecting just the sheets for the equipment
    named in the current report. Recall is preserved because the query asks for:
      * an exact ``equipment_name`` match,
      * a ``match_phrase_prefix`` on ``equipment_name`` (covers "C-130" ->
        "C-130B", "J-10" -> "J-10CE"), and
      * a keyword-phrase match (covers alias/keyword hits).
    The in-memory matcher then picks the best variant from this small candidate
    set. Validated against the full-catalog matcher over 736 truncated catalog
    names with zero lost matches.
    """
    should = []
    seen = set()
    for request_name, request_type in requests or []:
        if not request_name:
            continue
        if request_name not in seen:
            seen.add(request_name)
            should.append({"term": {"equipment_name.keyword": request_name}})
            if len(request_name) >= 3:
                should.append({"match_phrase_prefix": {
                    "equipment_name": {"query": request_name, "max_expansions": 100}}})
            should.append({"match_phrase": {"equipment_name_keywords": request_name}})
        if request_type:
            key = f"{request_name}({request_type})"
            if key not in seen:
                seen.add(key)
                should.append({"term": {"equipment_key.keyword": key}})
    if not should:
        return []
    query_body = {
        "size": EQUIPMENT_SPEC_CATALOG_SIZE,
        "query": {
            "bool": {
                "must": [
                    {"terms": {"injester_name.keyword": list(EQUIPMENT_SPEC_INJESTERS)}},
                    {"exists": {"field": "specifications"}},
                ],
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "_source": list(EQUIPMENT_SPEC_RESPONSE_FIELDS),
    }
    if os.getenv("AI_SUMMARY_EQUIP_SPEC_EXCLUDE_FORM_STATUS_5", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        query_body["query"]["bool"]["must_not"] = [{"term": {"form_status": 5}}]
    _t0 = time.time()
    response = es.search(index=EQUIPMENT_SPEC_INDEX, body=query_body)
    _ms = (time.time() - _t0) * 1000
    hits = response.get("hits", {}).get("hits", [])
    print(
        f"[INFO] Equipment spec filtered lookup: {len(hits)} doc(s) for "
        f"{len(requests)} requested name(s) "
        f"(ES took={response.get('took', '?')}ms, wall={_ms:.0f}ms)"
    )
    return hits


def _load_equipment_spec_catalog():
    """Return the equipment specification catalog, fetched once and cached.

    ⭐ v3.1.1.15: the whole catalog (~700 docs / ~180 KB) is small enough to
    load in a single query and reuse for every report in the process, instead
    of issuing a filtered query per report. Set AI_SUMMARY_EQUIP_SPEC_NO_CACHE=1
    to disable caching (forces a fresh single query each call).
    """
    global _EQUIPMENT_SPEC_CATALOG_CACHE
    if _EQUIPMENT_SPEC_CATALOG_CACHE is not None:
        return _EQUIPMENT_SPEC_CATALOG_CACHE
    hits = _search_equipment_spec_catalog()
    if os.getenv("AI_SUMMARY_EQUIP_SPEC_NO_CACHE", "").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        _EQUIPMENT_SPEC_CATALOG_CACHE = hits
    return hits


def _norm_equipment_token(value):
    """Lower-case and collapse every non-alphanumeric run to a single space.

    e.g. "F-16A/B" -> "f 16a b", "Mi-17" -> "mi 17", "007" -> "007".
    """
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _compact_equipment_token(value):
    """Lower-case and strip all non-alphanumeric characters entirely.

    e.g. "F-16A/B" -> "f16ab", "Mi-17" -> "mi17".
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _equipment_record_richness(rec):
    """Score how complete a catalog record's specification sheet is.

    ⭐ v3.1.1.15b: used to keep the most detailed document when the catalog
    holds several documents for the same equipment (one per form submission /
    re-ingestion). Richer = longer specification text, then more populated
    specification fields.
    """
    source = rec.get("source", {}) or {}
    specs = source.get("specifications")
    specs_len = len(str(specs)) if specs else 0
    populated = 0
    for field in EQUIPMENT_SPEC_RESPONSE_FIELDS:
        if source.get(field) not in (None, "", [], {}):
            populated += 1
    return (specs_len, populated)


def _build_equipment_catalog_index(catalog_hits):
    """Index the catalog docs for fast in-memory name/keyword matching.

    Each record keeps its normalised & compacted equipment name, the set of
    normalised/compacted keywords shipped with the doc (``equipment_name_keywords``
    / ``equipment_type_keywords``), and the raw ``_source``.

    ⭐ v3.1.1.15b: the ES catalog stores MANY documents per equipment (one per
    form submission / ingest), so the raw hit list is heavily duplicated. Docs
    are collapsed by compacted equipment name (so "Mi-17", "Mi17" and "mi 17"
    are treated as one equipment) and the richest record is retained, while
    keyword sets from every duplicate are merged so matching still sees all
    aliases. This is what stops the same equipment appearing several times in
    the report.
    """
    by_compact = {}
    order = []
    for hit in catalog_hits or []:
        source = hit.get("_source", {}) or {}
        name = source.get("equipment_name") or source.get("equipment") or ""
        if not name:
            continue
        compact = _compact_equipment_token(name)
        if not compact:
            continue
        kw_norms = set()
        kw_compacts = set()
        for kw_field in ("equipment_name_keywords", "equipment_type_keywords"):
            raw = source.get(kw_field)
            if not raw:
                continue
            for token in str(raw).split(","):
                token = token.strip()
                if not token:
                    continue
                kw_norms.add(_norm_equipment_token(token))
                kw_compacts.add(_compact_equipment_token(token))
        rec = {
            "source": source,
            "name": name,
            "norm": _norm_equipment_token(name),
            "compact": compact,
            "kw_norms": kw_norms,
            "kw_compacts": kw_compacts,
            "type_norm": _norm_equipment_token(
                source.get("equipment_type") or source.get("eq_type") or ""
            ),
        }
        existing = by_compact.get(compact)
        if existing is None:
            by_compact[compact] = rec
            order.append(compact)
            continue
        # Duplicate equipment document: merge aliases and keep the richer sheet.
        merged_norms = existing["kw_norms"] | kw_norms
        merged_compacts = existing["kw_compacts"] | kw_compacts
        if _equipment_record_richness(rec) > _equipment_record_richness(existing):
            rec["kw_norms"] = merged_norms
            rec["kw_compacts"] = merged_compacts
            by_compact[compact] = rec
        else:
            existing["kw_norms"] = merged_norms
            existing["kw_compacts"] = merged_compacts
    return [by_compact[key] for key in order]


def _match_equipment_record(request_name, request_type, catalog_index):
    """Return the best-matching catalog record for one requested equipment.

    The report rows carry short equipment codes / free-text names (e.g. "mi-17",
    "F-16", "c-130") while the catalog stores full designations ("Mi-17",
    "F-16A/B", "C-130B Hercules"). Matching therefore degrades gracefully:

      score 4  exact normalised or compacted name match
      score 3  requested name matches a catalog keyword
      score 2  catalog name starts with the requested compacted name
      score 1  requested compacted name appears inside the catalog name

    When several catalog rows match by name, rows whose type contains the
    requested type are preferred (the type is a soft tie-breaker, not a hard
    filter, because report types are codes while catalog types are descriptive).
    """
    q_name = _norm_equipment_token(request_name)
    q_compact = _compact_equipment_token(request_name)
    if len(q_compact) < 3:
        return None
    q_type = _norm_equipment_token(request_type or "")

    best = None
    best_key = (0, 0)
    for rec in catalog_index:
        if q_compact and q_compact == rec["compact"]:
            score = 4
        elif q_name and q_name == rec["norm"]:
            score = 4
        elif (q_compact and q_compact in rec["kw_compacts"]) or \
                (q_name and q_name in rec["kw_norms"]):
            score = 3
        elif q_compact and rec["compact"].startswith(q_compact):
            score = 2
        elif q_compact and q_compact in rec["compact"]:
            score = 1
        else:
            continue
        type_hit = 1 if (q_type and q_type in rec["type_norm"]) else 0
        key = (score, type_hit)
        if key > best_key:
            best_key = key
            best = rec
    return best


def fetch_equipment_specifications(equipment_pairs):
    """
    Look up specification details for each unique equipment and return the
    inner HTML for the "Equipment Specifications" report section (empty string
    when nothing is found or on error).

    ⭐ v3.1.1.15: equipment sheets are located with ONE Elasticsearch query and
    the requested equipment names are then matched in memory. The returned
    source is restricted to the equipment specification fields defined by the
    Additional Equipments workbook.

    ⭐ v3.1.1.18: by default the single query is FILTERED to the equipment named
    in this report (see `_search_equipment_spec_catalog_for`) so unrelated
    equipment is never fetched. Set AI_SUMMARY_EQUIP_SPEC_FULL_CATALOG=1 to
    fetch the whole (process-cached) catalog instead.
    """
    if not equipment_pairs:
        return ""
    rows = []
    try:
        # Collect the requested names/types into flat, de-duplicated lists.
        names = []
        types = []
        requests = []
        seen_request_keys = set()
        for item in equipment_pairs[:EQUIPMENT_SPEC_MAX_LOOKUPS]:
            equipment_name = item.get("query_name") or item.get("name")
            equipment_type = item.get("type") or ""
            if not equipment_name:
                continue
            # ⭐ v3.1.1.15b: collapse spelling variants ("Mi-17" / "Mi17" /
            # "mi 17") so the same equipment is only looked up once.
            request_key = _compact_equipment_token(equipment_name) or _norm_equipment_token(equipment_name)
            if request_key in seen_request_keys:
                continue
            seen_request_keys.add(request_key)
            if equipment_name not in names:
                names.append(equipment_name)
            if equipment_type and equipment_type not in types:
                types.append(equipment_type)
            requests.append((equipment_name, equipment_type))

        if not requests:
            return ""

        # ⭐ NEW (v3.1.1.18): fetch ONLY the equipment named in this report via
        # one filtered query, then match in memory. Set
        # AI_SUMMARY_EQUIP_SPEC_FULL_CATALOG=1 to restore the previous behavior
        # of loading the whole (process-cached) catalog.
        if os.getenv("AI_SUMMARY_EQUIP_SPEC_FULL_CATALOG", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            global _EQUIPMENT_SPEC_INDEX_CACHE
            catalog_hits = _load_equipment_spec_catalog()
            if _EQUIPMENT_SPEC_INDEX_CACHE is None:
                _EQUIPMENT_SPEC_INDEX_CACHE = _build_equipment_catalog_index(catalog_hits)
            catalog_index = _EQUIPMENT_SPEC_INDEX_CACHE
            print(
                f"[INFO] Equipment specifications: FULL catalog={len(catalog_hits)} "
                f"doc(s) (process-cached), {len(names)} requested name(s)"
            )
        else:
            catalog_hits = _search_equipment_spec_catalog_for(requests)
            catalog_index = _build_equipment_catalog_index(catalog_hits)
            print(
                f"[INFO] Equipment specifications: filtered query for {len(names)} "
                f"requested name(s) returned {len(catalog_hits)} doc(s)"
            )

        seen_rows = set()
        matched = 0
        for request_name, request_type in requests:
            rec = _match_equipment_record(request_name, request_type, catalog_index)
            if rec is None:
                continue
            source = rec["source"]
            name = rec["name"]
            etype = source.get("equipment_type") or source.get("eq_type") or ""
            # ⭐ v3.1.1.15b: de-duplicate by equipment identity (compacted name)
            # rather than the (name, type) pair. The catalog holds several
            # variants of the same equipment with different type text, which
            # previously produced two identical-looking rows in the report.
            row_key = rec["compact"] or rec["norm"]
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            matched += 1

            detail_items = []
            specs_text = source.get("specifications")
            if specs_text:
                # The sheet is a pre-formatted text block whose lines look like
                # "Flt Ceiling          : \t15,240 m" separated by dashed rules.
                # Render each "Label : value" pair as its own row.
                for line in str(specs_text).splitlines():
                    line = line.strip().lstrip("-").strip()
                    if not line:
                        continue
                    label, sep, val = line.partition(":")
                    if not sep:
                        continue
                    label = label.strip()
                    val = val.strip()
                    if not label or not val:
                        continue
                    detail_items.append(
                        f"<div><strong>{_html.escape(label)}:</strong> "
                        f"{_html.escape(val)}</div>"
                    )
            for field in EQUIPMENT_SPEC_RESPONSE_FIELDS:
                if field in {
                    "equipment_name", "equipment_type", "equipment_key",
                    "equipment_name_keywords", "equipment_type_keywords",
                    "specifications",
                }:
                    continue
                value = source.get(field)
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                label = field.replace("_", " ").title()
                detail_items.append(
                    f"<div><strong>{_html.escape(label)}:</strong> "
                    f"{_html.escape(str(value))}</div>"
                )

            equipment_html = f"<strong>{_html.escape(str(name))}</strong>"
            if etype:
                equipment_html += f"<br><span>Type: {_html.escape(str(etype))}</span>"
            details_html = (
                '<div class="equipment-spec-fields">' + "".join(detail_items) + "</div>"
                if detail_items else "<span>No specification details available</span>"
            )
            rows.append(
                "<tr>"
                f"<td>{equipment_html}</td>"
                f"<td>{details_html}</td>"
                "</tr>"
            )

        print(
            f"[INFO] Equipment specifications: {matched} of {len(names)} "
            f"requested name(s) matched the catalog"
        )
        if not rows:
            return ""
        return (
            '<table class="equipment-spec-table">'
            '<thead><tr><th>Equipment</th><th>Specification Details</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table>'
        )
    except Exception as exc:
        print(f"[WARN] Equipment specifications section skipped: {exc}")
        return ""


# ──────────────────────────────────────────────
# ⭐ NEW (v3.1.1.23): Enemy Formation Details helper
# ──────────────────────────────────────────────
# Mirrors the Equipment Specifications section for enemy formations: the enemy
# formations named in the retained records are looked up in the ORBAT documents
# (``injester_name = "orbat_injester"``) and their details rendered in a table
# under the "Enemy Formation Details" heading.
FORMATION_DETAIL_INDEX = CUREENT_INDEX_NAME
FORMATION_DETAIL_INJESTER = "orbat_injester"
FORMATION_DETAIL_MAX_LOOKUPS = 50
FORMATION_DETAIL_SIZE = 1000
# Placeholder values that recur across almost every record and carry no ORBAT
# detail worth a dedicated row.
FORMATION_NAME_PLACEHOLDERS = frozenset({
    "unidentified", "unknown", "n/a", "na", "none", "nil", "-",
    "not identified",
})
# Internal / metadata fields that carry no ORBAT detail worth showing. The
# section renders EVERY other populated field of the matched ORBAT document, so
# these are hidden from the report (and skipped in the Elasticsearch response so
# bulky values such as embeddings/images are never transferred).
FORMATION_DETAIL_EXCLUDED_FIELDS = frozenset({
    "file_hash",
    "file_format",
    "file_local_path",
    "file_http_path",
    "para_id",
    "form_id",
    "form_status",
    "form_upload_type",
    "is_para_verified",
    "is_updated_verify",
    "verify_status",
    "created_date",
    "updated_date",
    "created_user_id",
    "updated_user_id",
    "status",
    "icon",
    "@timestamp",
    "updated_user_id_id",
    "form_unique_id",
    "form_type",
    "description_hash",
    "cluster_uuid",
    "is_emotion_sentiment_updated",
    "form_entity_join",
    "int_task_status",
    "relevance",
    "activity_type",
    "form_data_search",
    "form_title",
    "location_profile_id",
    "created_user_id_id",
    "daily_activity_type_id",
    "file_id",
    "file_name",
    "form_activity_type",
    "injester_name",
    "main_form_id",
    "co_ordinates",
    "test_orbat_version",
    "http_path",
    "file_path",
    "test_version",
    "date",
    "activity_date",
    "day",
    "week_number",
    "month",
    "year",
    "orbat_title",
    "file_type",
    "location_list",
    "images",
    "pdf_file_path",
    "main_image_path",
    "all_images",
    "equipment_list",
    "person",
    "equipment",
    "child_id",
    "parent_id",
    "parent_type",
    "child_type",
    "hierarchy_relationship_id",
    "herarchy_relationship_id",
    "hash",
    "count",
    "relationship_name",
    "person_image",
    "created_at",
    "updated_at",
    "enemy_orbat",
    "desc_embedding",
})

def _collect_formation_names(e_hits):
    """
    Return an ordered list of unique enemy-formation names found in the given
    ES hits.

    Handles the shapes seen in the live index:
      * the flat ``enemy_formation_name`` field (string, list, or a comma-joined
        value),
      * dotted variants (``deployment.enemy_formation_name``,
        ``force.enemy_formation_name``).

    ``query_name`` preserves the raw value (some index values carry trailing
    spaces) so the exact-term lookup still matches; ``name`` is display-only.
    Placeholder values ("unidentified", "unknown", ...) are skipped because
    they carry no ORBAT detail and would add a row to almost every report.
    """
    order = []
    query_names = {}

    def _add(name):
        if name in (None, "", "None", "nan"):
            return
        raw_name = str(name)
        display = raw_name.strip()
        if not display:
            return
        if display.lower() in FORMATION_NAME_PLACEHOLDERS:
            return
        key = display.lower()
        if key in query_names:
            return
        query_names[key] = raw_name
        order.append((display, key))

    for hit in e_hits or []:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source", {})
        if not isinstance(source, dict):
            continue
        for name_key in (
            "enemy_formation_name",
            "deployment.enemy_formation_name",
            "force.enemy_formation_name",
        ):
            raw = source.get(name_key)
            if raw in (None, "", "None"):
                continue
            values = raw if isinstance(raw, list) else str(raw).split(",")
            for value in values:
                _add(value)

    return [
        {"name": display, "query_name": query_names[key]}
        for display, key in order
    ]


def _formation_record_richness(rec):
    """Score how complete an ORBAT document is.

    The index holds many documents per formation (one per form submission /
    re-ingestion), so the richer record is kept when several match the same
    formation (richer = more populated ORBAT fields).
    """
    source = rec.get("source", {}) or {}
    populated = 0
    for field, value in source.items():
        if field in FORMATION_DETAIL_EXCLUDED_FIELDS:
            continue
        if value not in (None, "", [], {}):
            populated += 1
    return populated


# Process-cached sort field. The reference query sorts on ``timestamp``, but
# that field is NOT mapped in this index (Elasticsearch answers 400 "No mapping
# found for [timestamp] in order to sort on"). The first successful attempt is
# cached so later reports don't pay for a failing round-trip.
_FORMATION_SORT_FIELD_CACHE = "__unset__"


def _formation_query_body(names):
    """Build the ORBAT lookup query for the given formation names.

    Mirrors the reference query — injester_name = "orbat_injester",
    enemy_formation_name IN formation_list, form_status != 5 — and adds
    ``match_phrase`` clauses so casing variants ("pak army" vs "Pak Army") are
    still matched, since the ``keyword`` field is case-sensitive.

    Every field of the ORBAT document is returned except the internal /
    metadata fields in ``FORMATION_DETAIL_EXCLUDED_FIELDS`` (these are never
    shown in the report, so they are also kept out of the response).
    """
    return {
        "size": FORMATION_DETAIL_SIZE,
        "query": {
            "bool": {
                "must": [
                    {"term": {"injester_name.keyword": FORMATION_DETAIL_INJESTER}},
                ],
                "should": [
                    {"terms": {"enemy_formation_name.keyword": list(names)}},
                ] + [
                    {"match_phrase": {"enemy_formation_name": name}}
                    for name in names
                ],
                "minimum_should_match": 1,
                "must_not": [{"term": {"form_status": 5}}],
            }
        },
        "_source": {"excludes": sorted(FORMATION_DETAIL_EXCLUDED_FIELDS)},
    }


def _search_formation_details(names):
    """Fetch ORBAT documents for the given formation names in ONE query.

    The sort field is probed once (``timestamp`` per the reference query, then
    ``@timestamp``, then no sort) and the first field that works is cached for
    the process, so an unmapped ``timestamp`` mapping never sinks the section.
    """
    global _FORMATION_SORT_FIELD_CACHE
    query_body = _formation_query_body(names)
    if _FORMATION_SORT_FIELD_CACHE == "__unset__":
        candidates = ["timestamp", "@timestamp", None]
    else:
        candidates = [_FORMATION_SORT_FIELD_CACHE]
    last_exc = None
    for field in candidates:
        body = copy.deepcopy(query_body)
        if field:
            body["sort"] = [{field: {"order": "desc"}}]
        try:
            _t0 = time.time()
            response = es.search(index=FORMATION_DETAIL_INDEX, body=body)
            _ms = (time.time() - _t0) * 1000
            if _FORMATION_SORT_FIELD_CACHE == "__unset__":
                _FORMATION_SORT_FIELD_CACHE = field
            hits = response.get("hits", {}).get("hits", [])
            print(
                f"[INFO] Enemy formation details: {len(hits)} doc(s) for "
                f"{len(names)} requested name(s) "
                f"(sort={field or 'none'}, ES took={response.get('took', '?')}ms, "
                f"wall={_ms:.0f}ms)"
            )
            return hits
        except Exception as exc:  # fall through to the next sort candidate
            last_exc = exc
            print(
                f"[WARN] Enemy formation details: sort={field!r} failed "
                f"({exc}); trying next"
            )
    if last_exc is not None:
        raise last_exc
    return []


def _build_formation_index(formation_hits):
    """Index the ORBAT hits for fast in-memory name matching.

    Each record keeps its normalised & compacted formation name and the raw
    ``_source``. Duplicate documents for the same formation are collapsed by
    compacted name, keeping the richest sheet so a formation never appears twice
    in the report.
    """
    by_compact = {}
    order = []
    for hit in formation_hits or []:
        source = hit.get("_source", {}) or {}
        name = str(source.get("enemy_formation_name") or "").strip()
        if not name:
            continue
        compact = _compact_equipment_token(name)
        if not compact:
            continue
        rec = {
            "source": source,
            "name": name,
            "norm": _norm_equipment_token(name),
            "compact": compact,
        }
        existing = by_compact.get(compact)
        if existing is None:
            by_compact[compact] = rec
            order.append(compact)
        elif _formation_record_richness(rec) > _formation_record_richness(existing):
            by_compact[compact] = rec
    return {key: by_compact[key] for key in order}


def _match_formation_record(request_name, formation_index):
    """Return the best-matching ORBAT record for one requested formation name.

    Matching is exact on the normalised / compacted name first (so "Pak Army"
    matches "pak army" and "Mi-17"/"Mi17" are treated as one), then falls back
    to containment so a short report code still finds the full designations.
    """
    q_compact = _compact_equipment_token(request_name)
    q_norm = _norm_equipment_token(request_name)
    if not q_compact:
        return None
    for rec in formation_index.values():
        if q_compact == rec["compact"] or (q_norm and q_norm == rec["norm"]):
            return rec
    for rec in formation_index.values():
        if q_compact in rec["compact"] or rec["compact"] in q_compact:
            return rec
    return None


def fetch_formation_details(formation_pairs):
    """
    Look up ORBAT details for each unique enemy formation and return the inner
    HTML for the "Enemy Formation Details" report section (empty string when
    nothing is found or on error).

    Mirrors `fetch_equipment_specifications`: the ORBAT sheets are located with
    ONE Elasticsearch query and the requested formation names are then matched
    in memory, rendering every populated ORBAT field for each match except the
    internal / metadata fields listed in ``FORMATION_DETAIL_EXCLUDED_FIELDS``.
    """
    if not formation_pairs:
        return ""
    rows = []
    try:
        names = []
        query_names = []
        seen = set()
        for item in formation_pairs[:FORMATION_DETAIL_MAX_LOOKUPS]:
            name = item.get("query_name") or item.get("name")
            if not name:
                continue
            compact = _compact_equipment_token(name) or _norm_equipment_token(name)
            if compact in seen:
                continue
            seen.add(compact)
            names.append(item.get("name") or name)
            query_names.append(name)
        if not query_names:
            return ""

        hits = _search_formation_details(query_names)
        formation_index = _build_formation_index(hits)

        matched = 0
        seen_rows = set()
        for display_name, raw_name in zip(names, query_names):
            rec = (_match_formation_record(display_name, formation_index)
                   or _match_formation_record(raw_name, formation_index))
            if rec is None:
                continue
            if rec["compact"] in seen_rows:
                continue
            seen_rows.add(rec["compact"])
            matched += 1

            source = rec["source"]
            detail_items = []
            for field, value in source.items():
                if field in FORMATION_DETAIL_EXCLUDED_FIELDS:
                    continue
                # The formation name is already shown as the row's label, so it
                # is not repeated among the detail fields.
                if field == "enemy_formation_name":
                    continue
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                label = field.replace("_", " ").replace("@", "").title()
                detail_items.append(
                    f"<div><strong>{_html.escape(label)}:</strong> "
                    f"{_html.escape(str(value))}</div>"
                )

            formation_html = f"<strong>{_html.escape(str(rec['name']))}</strong>"
            details_html = (
                '<div class="equipment-spec-fields">' + "".join(detail_items) + "</div>"
                if detail_items else "<span>No formation details available</span>"
            )
            rows.append(
                "<tr>"
                f"<td>{formation_html}</td>"
                f"<td>{details_html}</td>"
                "</tr>"
            )

        print(
            f"[INFO] Enemy formation details: {matched} of {len(query_names)} "
            f"requested name(s) matched the ORBAT records"
        )
        if not rows:
            return ""
        return (
            '<table class="equipment-spec-table formation-detail-table">'
            '<thead><tr><th>Formation</th><th>Formation Details</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table>'
        )
    except Exception as exc:
        print(f"[WARN] Enemy formation details section skipped: {exc}")
        return ""


# ──────────────────────────────────────────────
# ⭐ NEW (v3.1.1.1): Record Source Details helper
# ──────────────────────────────────────────────
def build_doc_id_activity_date_table(hits, query_total=None, date_range=None, query_note=None) -> str:
    """
    Build an HTML table summarising each Elasticsearch hit by its citation
    metadata: ES doc _id, source_type, file_name (the explicit file_name field,
    else the basename of any file path field), ingester_name, and activity_date.

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
        # ⭐ v3.1.1.19: prefer the explicit file_name field, then fall back to
        # the basename of any available path field. Real ES docs carry a
        # file_name but frequently lack file_http_path, which previously left
        # the File Name column blank (em-dash) for those records.
        file_name = source.get("file_name") or ""
        if not file_name:
            _file_path = (
                source.get("file_http_path")
                or source.get("pdf_file_path")
                or source.get("file_local_path")
                or ""
            )
            file_name = _basename(_file_path)
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

    # ⭐ v3.1.1.15: swap a configured-but-missing model for an installed one.
    llm_model = resolve_available_model(llm_model)
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


# ──────────────────────────────────────────────
# ⭐ NEW: imagery + infra records
# ──────────────────────────────────────────────
# Imagery (IMINT) records carry a different column set from the standard
# analysis records. The columns below are added to the prompt evidence for
# those records ONLY, so every other record keeps exactly the representation it
# had before. A record counts as IMINT when it is an imagery record reporting
# infra activity, or when it identifies itself via source_type/imint_number/
# aoi_details. Every IMINT value is passed to the model in full: the flattened
# per-AOI columns are re-attributed to their AOI names from aoi_details and all
# values are kept on a single line so the markdown table stays intact.
IMAGERY_INJESTER_NAME = "imagery"
IMAGERY_INFRA_FIELDS = (
    "activity_type",
    "previous_imagery",
    "present_imagery",
    "changes_discerned",
    "gen",
    "comments",
    "imint_number",
    "source_type",
    "all_images",
    "aoi_name",
    "injester_name",
    "activity_date",
)
IMAGERY_FIELD_LABELS = {
    "activity_type": "Activity Type",
    "previous_imagery": "Previous Imagery",
    "present_imagery": "Present Imagery",
    "changes_discerned": "Changes Discerned",
    "gen": "Gen",
    "comments": "Comments",
    "imint_number": "IMINT Number",
    "source_type": "Source Type",
    "all_images": "All Images",
    "aoi_name": "AOI Name",
    "injester_name": "Injester Name",
    "activity_date": "Activity Date",
}


def _is_imagery_infra_record(source: dict) -> bool:
    """True when a record is an imagery (IMINT) record reporting infra activity."""
    if not isinstance(source, dict):
        return False
    injester_name = str(source.get("injester_name") or "").strip().lower()
    if injester_name != IMAGERY_INJESTER_NAME:
        return False
    activity_types = source.get("activity_types")
    if activity_types in (None, "", [], {}):
        activity_types = source.get("activity_type")
    if activity_types in (None, "", [], {}):
        return False
    if isinstance(activity_types, (list, tuple, set)):
        values = list(activity_types)
    else:
        values = str(activity_types).split(",")
    return any("infra" in str(value).strip().lower() for value in values)


def _render_imagery_field_value(value) -> str:
    """Stringify an imagery column value for prompt evidence."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(value)
    return str(value)


IMAGERY_AOI_DETAILS_FIELD = "aoi_details"
# The flattened IMINT columns are built by concatenating these per-AOI values in
# aoi_details order; the AOI name is lost in the flattened copy, so the prompt
# evidence re-aligns them from aoi_details whenever that list is present.
IMAGERY_AOI_COLUMN_FIELDS = ("previous_imagery", "present_imagery", "changes_discerned")
# IMINT values must reach the LLM intact — this cap only stops a pathological
# value from consuming the whole context (real IMINT fields are well under 1k).
IMAGERY_TABLE_CELL_MAX_CHARS = 4000


def _is_imint_record(source: dict) -> bool:
    """True when a record carries IMINT (imagery intelligence) values.

    Covers the original imagery+infra records and any other record that
    identifies itself as IMINT via source_type, imint_number or aoi_details, so
    the model always receives the imagery fields whenever IMINT data exists in
    the source record.
    """
    if not isinstance(source, dict):
        return False
    if _is_imagery_infra_record(source):
        return True
    if str(source.get("source_type") or "").strip().lower() == "imint":
        return True
    if str(source.get("imint_number") or "").strip():
        return True
    aoi_details = source.get(IMAGERY_AOI_DETAILS_FIELD)
    if isinstance(aoi_details, (list, tuple)) and len(aoi_details) > 0:
        return True
    injester_name = str(source.get("injester_name") or "").strip().lower()
    if injester_name == IMAGERY_INJESTER_NAME and (
        source.get("previous_imagery") or source.get("present_imagery")
        or source.get("changes_discerned") or source.get("all_images")
    ):
        return True
    return False


def _single_line_evidence(value) -> str:
    """Flatten a value to one line so it cannot break a markdown table row."""
    value = _clean_evidence_value(value)
    if value in (None, "", [], {}):
        return ""
    return re.sub(r"\s+", " ", _render_imagery_field_value(value)).strip()


def _render_aoi_aligned_column(source: dict, field: str) -> str:
    """Render an imagery column per AOI from aoi_details.

    The stored previous_imagery/present_imagery/changes_discerned values are
    newline-joined copies of the per-AOI values, so which AOI changed is lost.
    This restores that attribution, e.g.
    'AOI-I: 02 x UAVs seen; AOI-II: 15 x Tents; AOI-III: Vehicle Observation'.
    """
    details = source.get(IMAGERY_AOI_DETAILS_FIELD)
    if not isinstance(details, (list, tuple)):
        return ""
    aliases = {
        "changes_discerned": ("change_discerned", "changes_discerned"),
        "previous_imagery": ("previous_imagery",),
        "present_imagery": ("present_imagery",),
    }.get(field, (field,))
    parts = []
    for entry in details:
        if not isinstance(entry, dict):
            continue
        raw = None
        for key in aliases:
            if entry.get(key) not in (None, "", [], {}):
                raw = entry.get(key)
                break
        text = _single_line_evidence(raw)
        if not text:
            continue
        label = _single_line_evidence(entry.get("aoi_name"))
        parts.append(f"{label}: {text}" if label else text)
    return "; ".join(parts)


def _render_aoi_change_summary(source: dict) -> str:
    """One-line per-AOI change chain from aoi_details.

    Example: 'AOI-I: 02 x UAVs seen (unchanged); AOI-II: 15 x Tents -> 20 x
    Tents (05 x Tents added); AOI-III: Vehicle Observation (unchanged)'.
    """
    details = source.get(IMAGERY_AOI_DETAILS_FIELD)
    if not isinstance(details, (list, tuple)):
        return ""
    parts = []
    for entry in details:
        if not isinstance(entry, dict):
            continue
        label = _single_line_evidence(entry.get("aoi_name"))
        previous = _single_line_evidence(entry.get("previous_imagery"))
        present = _single_line_evidence(entry.get("present_imagery"))
        change = _single_line_evidence(
            entry.get("change_discerned") or entry.get("changes_discerned")
        )
        no_change = change.strip().lower() in ("", "no changes", "no change", "nil", "none")
        if previous and present and previous.strip().lower() != present.strip().lower():
            body = f"{previous} -> {present}"
        else:
            body = present or previous
            if body and no_change:
                body = f"{body} (unchanged)"
        if not no_change:
            body = f"{body} ({change})" if body else change
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    return "; ".join(parts)


def _imagery_field_evidence(source: dict, field: str) -> str:
    """Full IMINT value for one column, safe to embed in a markdown table.

    Every IMINT value is passed through intact (no 120-char truncation and no
    URL stripping); the only transformations are whitespace flattening so the
    value stays on one line, and re-attributing the flattened per-AOI columns
    to their AOI names.
    """
    if field in IMAGERY_AOI_COLUMN_FIELDS:
        aligned = _render_aoi_aligned_column(source, field)
        if aligned:
            return aligned
    return _single_line_evidence(source.get(field))


def _imint_image_urls(source: dict) -> list:
    """Return the image paths stored in a record's all_images field.

    all_images normally arrives as a list of URLs; a JSON-encoded string or a
    single URL string is also accepted so the image paths are never lost.
    """
    raw = source.get("all_images") if isinstance(source, dict) else None
    if raw in (None, "", [], {}):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                raw = json.loads(text)
            except Exception:
                raw = [text]
        else:
            raw = [text]
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    urls = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("url") or item.get("path") or item.get("image") or ""
        item = str(item).strip()
        if item and item not in urls:
            urls.append(item)
    return urls


def _imint_image_label(source: dict, fallback_idx: int) -> str:
    """Human label for one record's image set (IMINT number / AOI / date)."""
    parts = []
    imint_number = str(source.get("imint_number") or "").strip()
    if imint_number:
        parts.append(f"IMINT {imint_number}")
    aoi_name = str(source.get("aoi_name") or "").strip()
    if aoi_name:
        parts.append(aoi_name)
    activity_date = str(source.get("activity_date") or "").strip()
    if activity_date:
        parts.append(activity_date[:10])
    if not parts:
        return f"IMINT imagery set {fallback_idx}"
    if not imint_number and not aoi_name:
        # Date-only record: prefix it so the heading is not a bare date.
        return f"IMINT imagery — {parts[0]}"
    return " — ".join(parts)


def build_imint_imagery_html(e_hits: list) -> str:
    """Image paths (all_images) of every IMINT record, as a link list.

    The report body is written by the LLM, which receives the image paths as
    evidence but never reproduces them, so the paths are rendered here from the
    source records instead. Each path is both the link target and its visible
    text, so it can be copied even when the browser cannot load the image.
    """
    blocks = []
    seen = set()
    for hit in e_hits or []:
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict) or not _is_imint_record(source):
            continue
        urls = _imint_image_urls(source)
        if not urls:
            continue
        label = _imint_image_label(source, len(blocks) + 1)
        # Several IMINT records can share the same imagery (same IMINT number,
        # same pages); the identical set is listed once.
        key = (label, tuple(urls))
        if key in seen:
            continue
        seen.add(key)
        items = "".join(
            f'<li><a href="{_html.escape(url, quote=True)}" target="_blank" rel="noopener">'
            f'{_html.escape(url)}</a></li>'
            for url in urls
        )
        blocks.append(f"<h3>{_html.escape(label)}</h3><ul>{items}</ul>")
    return "".join(blocks)


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
        # ⭐ NEW: IMINT records also expose all of their imagery values.
        # Activity date and source type are already emitted above.
        if _is_imint_record(source):
            for field in IMAGERY_INFRA_FIELDS:
                if field in ("activity_date", "source_type"):
                    continue
                text = _imagery_field_evidence(source, field)
                if not text:
                    continue
                lines.append(f"- {IMAGERY_FIELD_LABELS[field]}: {text}")
            aoi_summary = _render_aoi_change_summary(source)
            if aoi_summary:
                lines.append(f"- AOI Changes: {aoi_summary}")
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
    # ⭐ v3.1.1.15: swap a configured-but-missing model for an installed one.
    llm_model = resolve_available_model(llm_model)
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
        "types": ["general area analysis", "event general area analysis", "search by query analysis"],
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


def convert_hits_to_markdown(e_hits: list, search_form_type: str, preserve_all_fields: bool = False,
                             start_row_idx: int = 1) -> str:
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
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
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
        for idx, hit in enumerate(e_hits, start_row_idx):
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
        for idx, hit in enumerate(e_hits, start_row_idx):
            source = hit.get("_source", {})
            lines.append(f"### Record {idx}")
            lines.append("")
            for field, value in source.items():
                if field.startswith("@"):
                    continue
                lines.append(f"- **{field}:** {value}")
            lines.append("")
        return "\n".join(lines)

    fields = list(type_config["fields"])
    field_labels = list(type_config["field_labels"])
    if preserve_all_fields:
        for hit in e_hits:
            for field in hit.get("_source", {}):
                # Retrieval vectors and duplicated index payloads are not
                # source facts. Supplying them bloats custom requests and can
                # cause the model to analyze embedding coordinates as data.
                internal = field in {"desc_embedding", "embedding", "embeddings", "vector",
                                     "form_data_search", "hash", "description_hash"} or field.endswith("_embedding")
                if field not in fields and not field.startswith("_") and not internal:
                    fields.append(field)
                    field_labels.append(field.replace("_", " ").title())

    # ⭐ NEW: when the record set contains IMINT records, add the imagery
    # columns those records carry. Analysis types whose records do not qualify
    # keep exactly their existing table columns.
    if any(_is_imint_record(hit.get("_source", {})) for hit in e_hits):
        fields = list(fields)
        field_labels = list(field_labels)
        for field in IMAGERY_INFRA_FIELDS:
            if field in fields:
                continue
            fields.append(field)
            field_labels.append(IMAGERY_FIELD_LABELS[field])

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
    for idx, hit in enumerate(e_hits, start_row_idx):
        source = hit.get("_source", {})
        imint_record = _is_imint_record(source)
        row_values = []
        for field in fields:
            if imint_record:
                # ⭐ IMINT values are passed through in full and flattened onto a
                # single line: raw newlines inside a cell used to split a single
                # record across 13 markdown lines and destroy the table.
                value_str = _imagery_field_evidence(source, field)
                cell_cap = None
            else:
                value = _clean_evidence_value(source.get(field, ""))
                # Ensure value is string-safe
                if value is None:
                    value = ""
                value_str = _single_line_evidence(value)
                cell_cap = None
            # Escape markdown table pipe characters
            value_str = value_str.replace("|", "\\|")
            # Truncate long values for readability
            if cell_cap and len(value_str) > cell_cap:
                value_str = value_str[:cell_cap - 3] + "..."
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
    # Word counts alone severely underestimate JSON, identifiers and text
    # without spaces. Use a UTF-8 size floor as a conservative approximation.
    token_estimate = max(int(word_count * 1.3), (len(text.encode("utf-8")) + 3) // 4)
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
                              output_tokens: Optional[int] = None,
                              include_citations: bool = True,
                              document: Optional[str] = None) -> str:
    """Analyze a complete bounded chunk with stateless direct Ollama chat.

    The full chunk document is included inline in the request. This is
    deliberately NOT semantic RAG: once a chunk is token-bounded, every record
    is considered relevant to the user's task, so retrieval ranking should not
    be allowed to hide less-similar records.

    The function is safe for parallel workers because each request is
    independent and does not share AnythingLLM workspace conversation history.

    ⭐ NEW (v3.1.1.17): include_citations=False drops the CITATION REQUIREMENT
    from the prompt so the model does not add parenthesized citation numbers
    to the generated text. Evidence records keep their numbers because other
    pipeline stages rely on the stable Citation No. ordering.
    """
    if not chunk_rows or not llm_model:
        return ""

    if not context_tokens or context_tokens <= 0:
        context_tokens = 8192

    # ⭐ NEW: new_prompt chunks pass their TYPE_MAPPING markdown explicitly;
    # every other caller keeps the standard chunk document format.
    chunk_document = document or format_chunk_as_document(chunk_rows, start_row_idx)
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
        # ⭐ NEW (v3.1.1.17): the CITATION REQUIREMENT is included only when the
        # row's is_citation flag is TRUE. When citations are not requested, the
        # prompt explicitly forbids citation-number lists in the output.
        + (
            "CITATION REQUIREMENT: Each supplied record has a Citation No. Use those numbers to cite the "
            "source of each analytical point. At the end of each distinct analytical point or paragraph, append "
            "the supporting citation numbers in parentheses, for example (3, 45, 50). Use only citation numbers "
            "that actually support that point. If a point is supported by several records, include all relevant "
            "citation numbers. Do not invent citation numbers.\n"
            if include_citations else
            "CITATIONS: Do NOT include any citation numbers or parenthesized number lists such as (3, 45, 50) "
            "in the output. Do NOT mention the word 'Record' or 'Citation', do NOT reference record numbers, "
            "and do NOT write empty parentheses '()'. Write the analysis without referencing numbered records.\n"
        )
        + "OUTPUT REQUIREMENT: Return a concise, evidence-based answer to the user's request. "
        "Keep the analysis compact: aggregate repeated observations instead of restating them, "
        "avoid repeating the user's request or announcing section headings that merely restate the "
        "record grouping, and do not pad with routine per-record commentary when several records "
        "support the same point. "
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

    if prompt_tokens + int(output_tokens or 1500) + 512 > context_tokens:
        # A single record can itself exceed the chunk target. Use bounded
        # delivery parts instead of an oversized fallback that truncates it.
        return analyze_type_mapping_records(
            chunk_document, query, llm_model, len(chunk_rows), context_tokens,
            output_tokens=output_tokens, latest_evidence_date=latest_evidence_date,
            include_citations=include_citations,
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
        return sanitize_ai_report_text(response, include_citations=include_citations)

    # Last-resort AnythingLLM inline chat. This fallback is used only when the
    # direct Ollama request fails; the normal path never relies on shared chat
    # history or RAG retrieval.
    print(
        f"[WARN] Chunk {chunk_idx}/{total_chunks}: direct Ollama failed; "
        "trying AnythingLLM inline chat fallback"
    )
    fallback = get_llm_response_direct(prompt, llm_model, max_tokens=output_tokens)
    if fallback and len(fallback.strip()) > 20:
        return sanitize_ai_report_text(fallback.strip(), include_citations=include_citations)
    return ""


def analyze_type_mapping_records(markdown_data: str, query: str, model_name: str,
                                 record_count: int, context_tokens: int,
                                 output_tokens: Optional[int] = None,
                                 latest_evidence_date: Optional[str] = None,
                                 include_citations: bool = True) -> str:
    """Analyze the records exactly as rendered by the TYPE_MAPPING markdown view.

    ⭐ NEW: 'new_prompt' rows use this single-pass analysis instead of the
    field-wise Overall Analysis, which splits the records into multiple
    aggregated entity-type sections. There is NO dimension/value grouping, no
    record merging and no synthesis pass: the LLM receives only the per-record
    markdown produced by convert_hits_to_markdown for the row's analysis type.
    For very large record sets the markdown is split into token-bounded
    delivery parts; each part is analyzed independently and concatenated as-is
    (no cross-part aggregation).
    """
    if not markdown_data or not model_name:
        return ""

    # Send only the record rows to the model. The report title, statistics and
    # footer are presentation metadata that the model tends to echo back as
    # duplicated headings/statements in the final report.
    data_view = markdown_data
    for _marker in ("## Detail Records", "## Data Records"):
        if _marker in data_view:
            data_view = data_view[data_view.index(_marker) + len(_marker):].lstrip()
            break
    data_view = re.split(r"\n---\s*\n\*End of report", data_view, maxsplit=1)[0].rstrip()
    if not include_citations:
        # The '#' column exists only to supply citation numbers; without
        # citations it invites the model to mention record numbers/ranges.
        _rows = []
        for _line in data_view.splitlines():
            if _line.startswith("|"):
                _cells = re.split(r"(?<!\\)\|", _line)
                if len(_cells) > 3:
                    _line = "|".join([_cells[0]] + _cells[2:])
            _rows.append(_line)
        data_view = "\n".join(_rows)

    context_tokens = context_tokens or 8192
    prompt_header = (
        f"{query}\n\n"
        f"{AI_ANALYSIS_QUALITY_GUARDRAILS}\n"
        "TYPE-MAPPED RECORD ANALYSIS:\n"
        "The records below are supplied exactly as stored in the analysis type's "
        f"field mapping ({record_count} records in total). Analyze every record; do "
        "not use semantic retrieval, relevance ranking or topic similarity to omit "
        "records. Do not regroup, merge or aggregate the records into other views.\n"
        f"LATEST EVIDENCE DATE: {latest_evidence_date or 'not determinable from supplied data'}.\n"
        "For forward-looking requests, forecasts must concern only periods after "
        "the latest evidence date. Do not invent dates or unsupported facts.\n"
        + (
            "CITATION REQUIREMENT: Each supplied record has a Citation No. Use those numbers to cite "
            "the source of each analytical point, in parentheses, for example (3, 45, 50). Use only "
            "citation numbers that actually support that point. Do not invent citation numbers.\n"
            if include_citations else
            "CITATIONS: Do NOT include citation numbers or parenthesized number lists such as (3, 45, 50), "
            "and do NOT mention the word 'Record' or 'Citation' in the output.\n"
        )
        + "OUTPUT REQUIREMENT: Answer the user's request in HTML. Any title should be in an h2 tag and "
        "each paragraph in a p tag. Do not mention this workflow, the field mapping, or the record format."
    )

    answer_budget = min(int(output_tokens or 1500), 1500)
    data_budget = context_tokens - _combined_estimate_tokens(prompt_header) - answer_budget - 640
    data_chunks = _split_text_to_token_budget(data_view, data_budget)
    answers = []
    for part_idx, data_part in enumerate(data_chunks, 1):
        prompt = (
            f"{prompt_header}\n\n"
            f"--- TYPE-MAPPED RECORDS ({part_idx}/{len(data_chunks)}) ---\n"
            f"{data_part}\n"
            "--- END TYPE-MAPPED RECORDS ---"
        )
        response = _get_direct_ollama_response(
            prompt, model_name, context_tokens=context_tokens,
            stage_label=f"new_prompt-type-mapping {part_idx}/{len(data_chunks)}",
            max_output_tokens=output_tokens,
        )
        if not response:
            response = get_llm_response_direct(prompt, model_name, max_tokens=output_tokens) or ""
        if response and response.strip() and "AI analysis unavailable" not in response:
            answers.append(sanitize_ai_report_text(response.strip(), include_citations=include_citations))
        else:
            answers.append(_analysis_error_html(
                f"The model could not process source portion {part_idx} of {len(data_chunks)}. "
                "The analysis is incomplete; please retry or check the AI service."
            ))
    return "\n".join(part for part in answers if part)


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
    chunk_idx, total_chunks, chunk_info, query, model_name, doc_label, latest_evidence_date, include_citations = job
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
            include_citations=include_citations,
            document=chunk_info.get("document"),
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
    # ⭐ v3.1.1.15: swap a configured-but-missing model for an installed one.
    llm_model = resolve_available_model(llm_model)

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




# ⭐ NEW (v3.1.1.18): LLM prose quality cleanups. These remove filler the model
# wraps around the real analysis without touching source-derived content.
_LLM_META_LEADIN_PATTERNS = (
    r"(?is)based on\s+the\s+(?:provided|supplied|given|above)\s+"
    r"(?:text|records?|data|information|document|evidence)\s*,?\s*"
    r"(?:here(?:'s| is| are)|below\s+is|i\s+(?:will|have)|the\s+following\s+is)\b"
    r"[^.\n:]{0,140}?[.:]\s*",
    r"(?is)here(?:'s| is| are)\s+(?:a|an|the)\s+[^.\n:]{0,90}?"
    r"(?:analysis|summary|breakdown|overview|list)\b[^.\n:]{0,90}?[.:]\s*",
    r"(?is)the\s+following\s+is\s+(?:a|an|the)\s+[^.\n:]{0,90}?[.:]\s*",
)


def _strip_llm_meta_leadins(text: str) -> str:
    """Remove model filler such as "Based on the provided text, here is a
    analysis of ..." that announces the answer instead of stating findings.

    ⭐ NEW (v3.1.1.18). Conservative: only sentences that reference the
    *provided* input AND announce an analysis are removed; substantive sentences
    (e.g. "Based on the records, the unit deployed...") are left untouched.
    """
    if not text:
        return text
    for pattern in _LLM_META_LEADIN_PATTERNS:
        text = re.sub(pattern, "", text)
    return text


def _dedupe_repeated_sentences(html_text: str) -> str:
    """Drop exact duplicate sentences within one generated section.

    ⭐ NEW (v3.1.1.18). The model frequently repeats a sentence verbatim (once
    in the intro, again under a bullet). Only *exact* duplicates above a minimum
    length are removed, keeping the first occurrence; unique wording and all
    HTML structure are preserved. Style/script blocks are never touched.
    """
    if not html_text or "<" not in html_text:
        return html_text
    min_len = 60
    seen = set()
    out = []
    in_raw_block = False
    for part in re.split(r"(<[^>]+>)", html_text):
        if part.startswith("<"):
            tag = part.strip().lower()
            if tag.startswith("<style") or tag.startswith("<script"):
                in_raw_block = True
            elif tag.startswith("</style") or tag.startswith("</script"):
                in_raw_block = False
            out.append(part)
            continue
        if in_raw_block or not part.strip():
            out.append(part)
            continue
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", part):
            norm = re.sub(r"[^a-z0-9 ]", "",
                          re.sub(r"\s+", " ", sentence).strip().lower())
            if len(norm) >= min_len and norm in seen:
                continue
            if len(norm) >= min_len:
                seen.add(norm)
            kept.append(sentence)
        out.append(" ".join(s for s in kept if s != ""))
    return "".join(out)


_JUNK_HEADING_DEFAULT = "unknown,n/a,na,none,null,tpt"


def _unwrap_junk_headings(html: str) -> str:
    """Remove report headings whose whole title is a data junk token.

    ⭐ NEW (v3.1.1.18): the Overall Analysis sometimes renders a group label
    such as "unknown" or the equipment type code "tpt" as a heading, which is
    noise. The heading tag is dropped but the content beneath it is kept. The
    token list is configurable via AI_SUMMARY_JUNK_HEADINGS (comma-separated).
    """
    if not html:
        return html
    junk = {
        t.strip().lower()
        for t in os.getenv("AI_SUMMARY_JUNK_HEADINGS", _JUNK_HEADING_DEFAULT).split(",")
        if t.strip()
    }
    if not junk:
        return html

    def _repl(m):
        title = re.sub(r"<[^>]+>", "", m.group(2))
        norm = re.sub(r"\s+", " ", title).strip().lower().rstrip(":.")
        if norm in junk:
            return ""
        # Trim junk tokens embedded as list items inside a longer heading, e.g.
        # "Analysis of Cargo Trucks, J-20, unknown" -> "Analysis of Cargo Trucks, J-20".
        cleaned = title
        for token in junk:
            # Drop the token when it is its own list item / leading word, even
            # if more words follow ("..., J-20, Unknown Equipment Analysis" ->
            # "..., J-20 Equipment Analysis"; "Unknown Equipment Deployments"
            # -> "Equipment Deployments").
            cleaned = re.sub(
                r"(?i)(?:,\s*|^)" + re.escape(token) + r"(?=\s|,|$)",
                "", cleaned,
            )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*,\s*$", "", cleaned).strip()
        if cleaned and cleaned != title:
            return f"<h{m.group(1)}>{cleaned}</h{m.group(1)}>"
        return m.group(0)

    return re.sub(r"(?is)<h([1-6])[^>]*>(.*?)</h\1>", _repl, html)


def sanitize_ai_report_text(text: str, include_citations: bool = True) -> str:
    """Normalize report-facing wording without changing source data.

    The report should consistently use 'analysis'/'aggregation' terminology.
    This is applied only to generated AI prose, never to the underlying ES
    records or citation table values.

    ⭐ NEW (v3.1.1.17): include_citations=False skips the "Citations:" line
    normalization so a stray model-emitted citation line is not converted into
    a polished <p>Citations: ...</p> report paragraph when citations were not
    requested.
    """
    if not text:
        return text
    # ⭐ v3.1.1.14: repair LLM output defects (raw markdown headings/bullets,
    # broken </ul closes, bold "Citations:" lines) before any rewording.
    text = _sanitize_llm_section_html(text, include_citations=include_citations)
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
    # ⭐ NEW (v3.1.1.18): drop announcing filler and exact-duplicate sentences.
    text = _strip_llm_meta_leadins(text)
    text = _dedupe_repeated_sentences(text)
    text = _strip_record_labels_from_report(text, include_citations=include_citations)
    text = re.sub(r"[\[(]\s*[\])]", "", text)
    text = re.sub(r"(?is)<(p|li)\b[^>]*>\s*:?\s*</\1>", "", text)
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


# ──────────────────────────────────────────────
# ⭐ NEW (v3.1.1.14): entity_type_analysis scoping for Overall Analysis
# ──────────────────────────────────────────────
# Canonical entity types supported by the field-wise Overall Analysis, with the
# ES source fields each one is built from. This is the authoritative list of
# fields the script supports for Overall Analysis.
OVERALL_ENTITY_TYPE_LABELS = (
    "Location Analysis",
    "Force Movement and Deployment Analysis",
    "Equipment and Vehicle Analysis",
    "Infrastructure Analysis",
    "Training and Readiness Analysis",
    "Visits and Inspections Analysis",
    "Force Composition, Strength and Capability Analysis",
    "Events, Security and Other Relevant Analysis",
    "Other Relevant Records",
)

OVERALL_ENTITY_TYPE_FIELDS = {
    "Location Analysis": (
        "location_name", "base_location_name", "start_location_name", "end_location_name",
        "pass_name", "visit_name", "comd_tps_loc_name",
    ),
    "Force Movement and Deployment Analysis": (
        "activity_type", "movement_type", "deployment_type", "start_location_name",
        "end_location_name", "enemy_formation_name", "army_name", "force_type_name",
        "formation_type", "command_name", "orbate_title", "mobile_no",
    ),
    "Equipment and Vehicle Analysis": (
        "equipment_name", "equipment_type", "equipement_type", "vehicle_type", "vehicle_name",
        "mobile_no", "count", "weapon_system", "radar_name", "radar_type", "airfield_type",
    ),
    "Infrastructure Analysis": (
        "infra_name", "infra_type", "airfield_type", "infrastructure_type", "project_type", "coordinates",
    ),
    "Training and Readiness Analysis": (
        "training_type", "training_area", "exercise_type", "readiness_type",
        "activity_type", "sub_activity_type",
    ),
    "Visits and Inspections Analysis": (
        "visit_name", "purpose", "visit_type", "inspection_type",
    ),
    "Force Composition, Strength and Capability Analysis": (
        "force_type_name", "formation_type", "army_name", "command_name", "orbate_title",
        "count", "strength", "personnel_count", "capability", "capabilities",
    ),
    "Events, Security and Other Relevant Analysis": (
        "event_type", "category", "incident_type", "casualties", "civilian_casualties",
        "security_forces_casualties", "terrorists_casualties", "political_event", "security_event",
    ),
    "Other Relevant Records": (),
}

OVERALL_ENTITY_TYPE_KEY_TOKENS = {
    "Location Analysis": ("location_name", "base_location", "start_location", "end_location",
                          "pass_name", "visit_name"),
    "Force Movement and Deployment Analysis": ("movement", "deployment", "formation", "force_type",
                                               "army_name", "command_name", "orbat", "mobile_no", "unit"),
    "Equipment and Vehicle Analysis": ("equipment", "equipement", "vehicle", "weapon", "radar",
                                       "mobile", "aircraft", "tank", "artillery", "drone", "missile"),
    "Infrastructure Analysis": ("infra", "infrastructure", "airfield", "facility", "construction",
                                "building", "bridge", "road", "railway"),
    "Training and Readiness Analysis": ("training", "exercise", "readiness", "drill", "practice",
                                        "familiarisation", "familiarization"),
    "Visits and Inspections Analysis": ("visit", "inspection", "review", "assessment", "meeting"),
    "Force Composition, Strength and Capability Analysis": ("force", "formation", "army", "command",
                                                            "orbat", "strength", "personnel",
                                                            "capabilit", "composition", "count"),
    "Events, Security and Other Relevant Analysis": ("event", "incident", "casualt", "security",
                                                     "terror", "political", "conflict", "attack",
                                                     "protest"),
    "Other Relevant Records": (),
}

# Keywords inside entity_type_analysis that mean "no restriction" (all fields).
ENTITY_TYPE_ALL_KEYWORDS = {"all", "all fields", "overall", "overall analysis", "*"}


def _entity_type_words(text) -> list:
    """Lower-case word tokens for fuzzy entity-type comparisons."""
    return [w for w in re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split() if w]


def _entity_type_stem(word: str) -> str:
    """Crude singular/plural stem so 'visits' matches 'visit' and vice versa."""
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _normalize_entity_type_selection(raw) -> list:
    """Normalize the ai_analysis_summary.entity_type_analysis jsonb value.

    The column holds an array of entity/field names (jsonb array of text), e.g.
    ["location_name", "enemy_formation_name"]. Accepted shapes:
      - None / SQL NULL, [] / jsonb '[]'  -> []  (Overall Analysis over ALL fields)
      - list/tuple/set of strings        -> cleaned, de-duplicated, order preserved
      - JSON string of an array          -> parsed then normalized
      - SQL array literal '{a,b}'        -> split then normalized
      - single string                    -> one-element list
      - dict                             -> its values flattened

    Every entry is lower-cased and de-duplicated. Matching against the supported
    entity types happens in _match_entity_type_selection_to_dimensions.
    """
    def _finalize(entries) -> list:
        result = []
        seen = set()
        for entry in entries:
            # Flatten one level of nested lists/tuples/sets (jsonb allows them).
            if isinstance(entry, (list, tuple, set)):
                for sub in _finalize(entry):
                    if sub not in seen:
                        seen.add(sub)
                        result.append(sub)
                continue
            text = str(entry or "").strip()
            if not text:
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result

    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # psycopg2 returns jsonb already parsed; a string here means some caller
        # passed an unparsed JSON or SQL array literal.
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (list, tuple, set)):
                    return _finalize(parsed)
                return _finalize([parsed])
            except Exception:
                inner = text.strip()[1:-1] if text.endswith("]") else text.strip()
                return _finalize(str(inner).split(","))
        if text.startswith("{") and text.endswith("}"):
            return _finalize(text[1:-1].split(","))
        return _finalize([text])
    if isinstance(raw, dict):
        flattened = []
        for value in raw.values():
            if isinstance(value, (list, tuple, set)):
                flattened.extend(value)
            else:
                flattened.append(value)
        return _finalize(flattened)
    if isinstance(raw, (list, tuple, set)):
        return _finalize(raw)
    # Any other scalar (int, bool, ...) is not a usable selection.
    return []


# ⭐ NEW: entity_type_analysis marker for a user-supplied ("new prompt") run.
NEW_PROMPT_MARKER = "new_prompt"


def _clean_evidence_value(value):
    """Remove missing-value sentinels in the prompt copy, retaining zero/False."""
    if isinstance(value, str):
        return "" if value.strip().lower() in {
            "undefined", "null", "none", "n/a", "unknown", "location not known"
        } else value
    if isinstance(value, dict):
        return {k: cleaned for k, v in value.items()
                if (cleaned := _clean_evidence_value(v)) not in (None, "", [], {})}
    if isinstance(value, (list, tuple)):
        return [cleaned for v in value
                if (cleaned := _clean_evidence_value(v)) not in (None, "", [], {})]
    return value


def _extract_table_prompt(prompt):
    """Decode the UI's #from_table# envelope without sending data as instructions.

    None means an ordinary prompt; [] means an explicitly empty table. Never
    replace an empty/invalid submitted table with unrelated Elasticsearch data.
    """
    prompt = str(prompt or "")
    if "#from_table#" not in prompt:
        return prompt, None
    request, payload = prompt.split("#from_table#", 1)
    start = payload.find("[")
    if start < 0:
        raise ValueError("Table-search request has no JSON record array. Please submit it again.")
    try:
        records, end = json.JSONDecoder().raw_decode(payload[start:])
    except (ValueError, TypeError) as exc:
        raise ValueError("Table-search records are invalid JSON. Please submit the request again.") from exc
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Table-search records must be an array of objects.")
    suffix = payload[start + end:].strip()
    request = "\n".join(part for part in (request.strip(), suffix) if part)
    hits = [{"_source": r.get("_source", r)} for r in records]
    if any(not isinstance(h["_source"], dict) for h in hits):
        raise ValueError("Table-search record sources must be objects.")
    return request, hits


def _split_text_to_token_budget(text, budget):
    """Losslessly split even a single oversized record; no evidence truncation."""
    if budget < 128:
        raise ValueError("The request leaves insufficient model context for source records. Shorten the prompt.")
    pending = [text]
    result = []
    while pending:
        part = pending.pop()
        if _combined_estimate_tokens(part) <= budget:
            if part:
                result.append(part)
            continue
        mid = len(part) // 2
        if mid == 0:
            raise ValueError("Cannot fit source evidence into the model context.")
        # Prefer a record/line boundary, while ensuring both pieces shrink.
        boundary = part.rfind("\n", max(1, mid // 2), mid + 1)
        if boundary > 0:
            mid = boundary + 1
        pending.extend([part[mid:], part[:mid]])
    return result


def _analysis_error_html(message):
    return ("<h2>Analysis Unavailable</h2><p>" + _html.escape(str(message)) + "</p>")


def _is_new_prompt_selection(raw) -> bool:
    """True when entity_type_analysis carries the 'new_prompt' marker.

    In that mode the row's own prompt (ai_analysis_summary.prompt) defines what
    is analyzed, and the records must come from the row's filter_json query
    directly instead of the aggregated (form_id-grouped) records.
    """
    return any(
        str(sel or "").strip().lower().replace(" ", "_") in {NEW_PROMPT_MARKER, "add_new_prompt"}
        for sel in _normalize_entity_type_selection(raw)
    )


def _match_entity_type_selection_to_dimensions(selected_types) -> tuple:
    """Map a normalized entity_type_analysis selection onto Overall Analysis dimensions.

    Returns (selected_dimension_names, unmatched). A selection matches a
    dimension when it equals the dimension name, equals one of its ES source
    fields, is word/stem-compatible with the dimension name ("equipment" ->
    Equipment and Vehicle Analysis), fuzzy-matches one of its key tokens, or is
    a close typo of any known name/field (difflib, e.g. 'enemey_foramtion_name').
    Any "all" keyword (or an empty selection) means ALL entity types. Unknown
    entries are returned as `unmatched` so callers can log them instead of
    dropping them silently.
    """
    if not selected_types:
        return (), ()

    selections = [str(s or "").strip().lower() for s in selected_types]
    # ⭐ NEW: 'new_prompt' is a mode marker (user-supplied prompt), not an entity
    # type — drop it so it neither warns nor restricts the analyzed dimensions.
    selections = [
        s for s in selections
        if s.replace(" ", "_") != NEW_PROMPT_MARKER
    ]
    if not selections or any(s in ENTITY_TYPE_ALL_KEYWORDS for s in selections if s):
        return (), ()

    def _norm(text) -> str:
        return " ".join(_entity_type_words(text))

    def _field_key(field: str) -> str:
        return " ".join(_entity_type_words(str(field).replace("_", " ")))

    all_fields = {name: tuple(_field_key(f) for f in fields)
                  for name, fields in OVERALL_ENTITY_TYPE_FIELDS.items()}

    def _close_match(sel: str):
        """Typo-tolerant fallback: nearest known name/field above a strict cutoff."""
        candidates = []
        for name in OVERALL_ENTITY_TYPE_LABELS:
            candidates.append((name, name))
            candidates.append((name, _norm(name).replace(" ", "_")))
        for name, fields in OVERALL_ENTITY_TYPE_FIELDS.items():
            for field in fields:
                candidates.append((name, field))
        matches = difflib.get_close_matches(sel, [c[1] for c in candidates], n=1, cutoff=0.85)
        if matches:
            best = matches[0]
            for name, candidate in candidates:
                if candidate == best:
                    return name
        return None

    matched = []
    unmatched = []
    matched_set = set()

    def _add(name) -> None:
        if name and name not in matched_set:
            matched_set.add(name)
            matched.append(name)

    for sel in selections:
        if not sel:
            continue
        sel_words = _entity_type_words(sel)
        sel_stems = {_entity_type_stem(w) for w in sel_words}
        sel_norm = " ".join(sel_words)

        # 1) Exact dimension-name match (case/underscore/punctuation insensitive).
        #    A dimension name targets exactly that one dimension.
        exact_dim = None
        for name in OVERALL_ENTITY_TYPE_LABELS:
            if sel_norm == _norm(name):
                exact_dim = name
                break
        if exact_dim:
            _add(exact_dim)
            continue

        # 2) Exact ES source-field match. ⭐ A field may legitimately belong to
        #    several dimensions (e.g. activity_type -> Force Movement AND Training;
        #    mobile_no -> Force Movement AND Equipment). Match ALL of them so a
        #    new/never-seen combination list scopes correctly instead of silently
        #    dropping the later dimensions.
        field_hits = [name for name, fields in all_fields.items() if sel_norm in fields]
        if field_hits:
            for name in field_hits:
                _add(name)
            continue

        # 3) Word/stem subset of the dimension name ("equipment" -> Equipment and
        #    Vehicle Analysis; "visits" -> Visits and Inspections Analysis).
        hit = None
        if sel_stems:
            for name in OVERALL_ENTITY_TYPE_LABELS:
                name_stems = {_entity_type_stem(w) for w in _entity_type_words(name)}
                if sel_stems.issubset(name_stems):
                    hit = name
                    break
        # 4) Key-token containment ("formation" -> Force Movement and Deployment).
        if hit is None and sel_norm:
            for name, tokens in OVERALL_ENTITY_TYPE_KEY_TOKENS.items():
                for token in tokens:
                    token_norm = _norm(token)
                    if token_norm and (token_norm in sel_norm or sel_norm in token_norm):
                        hit = name
                        break
                if hit:
                    break
        # 5) Typo-tolerant fallback for values like 'enemey_foramtion_name'.
        if hit is None:
            hit = _close_match(sel)

        if hit is None:
            unmatched.append(sel)
        else:
            _add(hit)
    return tuple(matched), tuple(unmatched)


def _overall_dimension_value(source: dict, fields: tuple) -> list:
    """Return normalized non-empty values from a set of candidate fields."""
    values = []
    if not isinstance(source, dict):
        return values
    for field in fields:
        value = _clean_evidence_value(source.get(field))
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

    source = _clean_evidence_value(source)
    lower_keys = {str(k).lower(): k for k in source.keys()}
    exact_fields = dimension.get("fields", ())
    text_fields = dimension.get("text_fields", ())

    # Explicit fields are the primary match.
    for field in exact_fields:
        # These common fields are not evidence of a particular subject: a
        # coordinate does not establish construction, and an arbitrary activity
        # is not automatically training/deployment. Classify their values using
        # the dimension's keyword rules below instead.
        if field in {"activity_type", "sub_activity_type", "coordinates", "count"}:
            continue
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


def _render_dimension_records(records: list, include_all_fields: bool = False,
                              include_citations: bool = True) -> str:
    """Render source records for a dimension, preserving citation numbers.

    ⭐ NEW (v3.1.1.17): include_citations=False renames the per-record heading
    from "### Citation No. N" to "### Record N" so the evidence document does
    not use citation wording. Record numbering is preserved because downstream
    stages rely on the stable record order.
    """
    if not records:
        return ""
    blocks = []
    hidden = {"_id", "id", "es_id", "elastic_doc_id", "_index", "_score", "file_http_path", "ingester_name", "injester_name"}
    preferred = (
        "activity_date", "location_name", "base_location_name", "start_location_name", "end_location_name",
        "enemy_formation_name", "army_name", "force_type_name", "formation_type", "command_name", "orbate_title",
        "activity_type", "sub_activity_type", "visit_name", "purpose", "equipment_name", "equipment_type",
        "equipment", "equipment_names", "combined_equipment_name", "orbat_title", "comment",
        "equipement_type", "vehicle_type", "vehicle_name", "infra_name", "infra_type", "airfield_type",
        "training_type", "training_area", "exercise_type", "readiness_type", "strength", "personnel_count",
        "capability", "capabilities", "count", "description", "comments", "coordinates", "category"
    )
    for citation_no, hit in records:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        if not isinstance(source, dict):
            source = {}
        # ⭐ NEW (v3.1.1.17): heading wording depends on the is_citation flag.
        # ⭐ v3.1.1.18: when citations are off the evidence blocks carry NO
        # number, so the model is never given record numbers it could echo back
        # as "records 21, 22 and 23". The heading still separates the blocks.
        if include_citations:
            lines = [f"### Citation No. {citation_no}"]
        else:
            lines = ["### Record"]
        keys = list(source.keys()) if include_all_fields else [f for f in preferred if f in source]
        # ⭐ NEW: IMINT records keep all of their imagery columns in the Overall
        # Analysis evidence (they are not part of the standard preferred set and
        # injester_name is normally hidden).
        imagery_record = _is_imint_record(source)
        if imagery_record:
            for field in IMAGERY_INFRA_FIELDS:
                # Keep the column even when only aoi_details carries the value.
                if field not in keys:
                    keys.append(field)
        seen = set()
        for field in keys:
            if field in seen:
                continue
            if field in hidden and not (imagery_record and field in IMAGERY_INFRA_FIELDS):
                continue
            seen.add(field)
            if imagery_record and field == IMAGERY_AOI_DETAILS_FIELD:
                continue  # rendered once, per AOI, below
            if imagery_record and field in IMAGERY_INFRA_FIELDS:
                text = _imagery_field_evidence(source, field)
                if text:
                    lines.append(f"- {field}: {text}")
                continue
            value = _clean_evidence_value(source.get(field))
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list, tuple, set)):
                try:
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                except Exception:
                    value = str(value)
            lines.append(f"- {field}: {value}")
        if imagery_record:
            aoi_summary = _render_aoi_change_summary(source)
            if aoi_summary:
                lines.append(f"- {IMAGERY_AOI_DETAILS_FIELD}: {aoi_summary}")
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


def _sanitize_llm_section_html(response: str, include_citations: bool = True) -> str:
    """Repair common LLM output defects before the section reaches the report.

    ⭐ v3.1.1.14 fix — found in the all-fields large-volume report:
      1. Model returns raw markdown (## headings, * bullets, **bold**) even
         though the prompt demands HTML. Convert the obvious cases so the
         report shows clean headings/bullets instead of literal '##' text.
      2. A broken '<ul>' close renders as '</ul<p>' in the report because the
         model emitted '</ul' followed immediately by '<p>'. Repair the
         missing '>' so the browser does not render stray '</ul' text.
      3. The model sometimes emits its own bold "Citations:" line. Keep only
         one, rendered in plain weight (matches the appended tail style).
    """
    if not response:
        return response
    text = response

    # -- 2) Repair malformed </ul closes like "</ul<p>" or "</ul <p>" --------
    text = re.sub(r"</ul(?![>a-zA-Z])", "</ul>", text, flags=re.I)
    text = re.sub(r"</ol(?![>a-zA-Z])", "</ol>", text, flags=re.I)

    # -- 1) Convert raw markdown artifacts ----------------------------------
    # Drop lines that are only '#' markers with no heading text (degenerate
    # truncated model output like "##" alone) BEFORE heading conversion.
    text = re.sub(r"(?m)^\s*#{1,6}[ \t]*$", "", text)
    # Headings: lines made only of leading '#' characters. Separator is
    # horizontal whitespace only, so '##' on its own line cannot swallow the
    # following line into a heading.
    def _heading_repl(m):
        level = min(len(m.group(1)), 6)
        return f"<h{level}>{m.group(2).strip()}</h{level}>"

    text = re.sub(r"(?m)^\s*(#{1,6})[ \t]+([^#\n][^\n]*)$", _heading_repl, text)

    # Bold markers: **text** -> <strong>text</strong> (skip already-HTML lines).
    text = re.sub(r"(?<!\*)\*\*(?!\*)([^*\n]+?)\*\*(?!\*)", r"<strong>\1</strong>", text)

    # Bullet lists: consecutive markdown bullets (-, *, •) become <ul><li>.
    lines = text.split("\n")
    out_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(?:[-*•])\s+(.+)$", stripped)
        if m and not stripped.startswith("<"):
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            out_lines.append(f"<li>{m.group(1).strip()}</li>")
            continue
        if in_list:
            out_lines.append("</ul>")
            in_list = False
        out_lines.append(line)
    if in_list:
        out_lines.append("</ul>")
    text = "\n".join(out_lines)

    # -- 3) Normalize bold 'Citations:' lines to plain weight ----------------
    # ⭐ v3.1.1.17: skipped when include_citations is False — when citations
    # were not requested, a stray model-emitted "Citations:" line is not
    # normalized into a polished report paragraph.
    if include_citations:
        text = re.sub(
            r"(?i)<p>\s*<strong>\s*Citations\s*:\s*</strong>\s*(\([^)]*\))\s*</p>",
            r"<p>Citations: \1</p>",
            text,
        )
        text = re.sub(
            r"(?i)<strong>\s*Citations\s*:\s*</strong>\s*(\([^)]*\))",
            r"Citations: \1",
            text,
        )
    return text


def _strip_dimension_name_echo(dimension_name: str, response: str) -> str:
    """Remove an LLM-echoed copy of the dimension name from a section response.

    ⭐ v3.1.1.14 fix: the wrapper already emits "<h2>{dimension_name}</h2>", but
    the model often repeats the section title as its own heading (or as part of
    the prompt's topic line), so the report showed the name twice.
    Leading headings/paragraphs that consist solely of the dimension-name words
    are dropped here. Body content is never touched.
    """
    if not response or not dimension_name:
        return response

    name_words = {w for w in _entity_type_words(dimension_name)}
    if not name_words:
        return response

    # ⭐ v3.1.1.14: models echo the title in variants such as
    #   "Location Analysis - Part 14 of 22", "Force Movements and Deployments
    #   Analysis", "Equipment and Vehicle Analysis - Analysis".
    # Normalize before comparing: drop "Part N of M" fragments and trailing
    # separators, and fold simple plurals so "movements" matches "movement".
    part_fragment_re = re.compile(r"(?i)\bpart\s+\d+\s+of\s+\d+\b")

    def _singular(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") else word

    singular_name_words = {_singular(w) for w in name_words}

    def _is_name_echo(block: str) -> bool:
        text = html_to_plain_text(block).strip()
        if not text:
            return True
        # Drop trailing group labels like "LOCATION ANALYSIS VALUE: xyz" before
        # comparing against the dimension name words.
        text = re.sub(r"(?i)value\s*:\s*.*$", "", text, flags=re.S).strip()
        text = part_fragment_re.sub(" ", text)
        text = re.sub(r"(?i)[\-\u2014\u2013:]+", " ", text)
        block_words = {_singular(w) for w in _entity_type_words(text) if w}
        return bool(block_words) and block_words.issubset(singular_name_words)

    cleaned = response.strip()
    changed = True
    while changed:
        changed = False
        m = re.match(r"\s*<(h[1-6]|p|strong|b|div)\b[^>]*>(.*?)</\1\s*>", cleaned, flags=re.I | re.S)
        if m and _is_name_echo(m.group(0)):
            cleaned = cleaned[m.end():].lstrip()
            changed = True
            continue
        # Plain-text prefix without any tag (e.g. "Location Analysis\n...").
        first_tag = re.search(r"<", cleaned)
        prefix = cleaned[:first_tag.start()] if first_tag else cleaned
        if prefix.strip() and _is_name_echo(prefix.strip()):
            cleaned = cleaned[first_tag.start():].lstrip() if first_tag else ""
            changed = True
    return cleaned


# ⭐ NEW (v3.1.1.20) — found in production report row 1842 ("Equipment and
# Vehicle Analysis — Part 2 of 3"): the Overall Analysis prompt hands the model
# its source records inside "===== DIMENSION VALUE: <label> =====" blocks. When
# the batch's answer budget is tight, the model can answer with that
# scaffolding instead of an analysis — repeating the VALUE: delimiters,
# listing the rendered evidence fields, and describing how the values were
# extracted. None of it is report content, so it is stripped before display.
_EVIDENCE_ECHO_LINE_RES = (
    # Free-standing evidence lines: "+ Record: activity_date: ..." or
    # "- Citation No. 7: ...".
    re.compile(r"(?i)^[+\-*\u2022]\s*(?:record|citation\s*no\.?)\s*\d*\s*[:\-]"),
    # The truncated-evidence placeholder "+... (other records)".
    re.compile(r"(?i)^[+\-*\u2022]\s*\.\.\.\s*\(other records?\)\s*\.?$"),
    # The placeholder bullet "This value is associated with the following
    # record(s):".
    re.compile(r"(?i)^this value\b[^.]{0,80}?\bis associated with the following\b"),
)
# The "There are several records of X, each with a unique value. Here are the
# extracted values:" preamble the model copies from the evidence framing.
_ECHO_PREAMBLE_RE = re.compile(
    r"(?is)\bthere (?:are|is|were)\s+(?:several|multiple|many|\d+)\s+records?\s+of\b"
    r"[^<]{0,300}?here are the extracted values\s*:?"
)
# The duplicate-value meta note ("This value is a duplicate of the '...' value
# and is not included in the analysis.").
_ECHO_META_NOTE_RE = re.compile(
    r"(?is)\bthis value\b[^<]{0,120}?\bnot included in the analysis\.?"
)
# Leaf-ish elements that can hold a whole echoed evidence fragment.
_ECHO_ELEMENT_RE = re.compile(r"(?is)<(p|h[1-6]|li|strong|b|span)\b[^>]*>(.*?)</\1\s*>")


# Name words which carry no identifying value when matching an echoed
# delimiter ("Equipment and Vehicle Analysis — Part 2 of 3" must match the
# echoed "EQUIPMENT AND VEHICLE ANALYSIS" but not a stray "Equipment value:").
_ECHO_NAME_STOPWORDS = {"and", "of", "the", "for", "with", "part"}


def _singular_word(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _is_value_delimiter_echo(text: str, name_words: set) -> bool:
    """True when *text* is an echoed ``... VALUE: <label>`` evidence delimiter.

    The text before ``VALUE:`` must consist only of the section's own name
    words (with an optional "Part N of M" fragment) and must still carry every
    identifying name word, so a genuine analytical sentence such as "the
    estimated equipment value: 12" is never dropped.
    """
    parts = re.split(r"(?i)value\s*:", html_to_plain_text(text), maxsplit=1)
    if len(parts) != 2:
        return False
    prefix = parts[0]
    if not prefix.strip(" =-\u2014\u2013:\t"):
        return False
    prefix = re.sub(r"(?i)\bpart\s+\d+\s+of\s+\d+\b", " ", prefix)
    prefix = re.sub(r"[=\-\u2014\u2013:]+", " ", prefix)
    words = {_singular_word(w) for w in _entity_type_words(prefix)}
    significant = {w for w in name_words if w not in _ECHO_NAME_STOPWORDS and not w.isdigit()}
    return bool(words) and words.issubset(name_words) and significant.issubset(words)


def _strip_evidence_echo(dimension_name: str, response: str) -> str:
    """Drop prompt/evidence scaffolding the model echoed back as its answer.

    ⭐ v3.1.1.20 fix: an Overall Analysis response must contain only analysis.
    This removes the echoed ``VALUE:`` delimiters, the rendered evidence lines
    ("+ Record: field: value, ..."), the value-extraction preamble and the
    duplicate-value meta note. Elements that become empty after the removals
    are collapsed so the report shows no stray markup.
    """
    if not response:
        return response
    name_words = {_singular_word(w) for w in _entity_type_words(dimension_name)}
    if not name_words:
        return response

    text = _ECHO_PREAMBLE_RE.sub("", response)
    text = _ECHO_META_NOTE_RE.sub("", text)

    def _drop_echo_element(m):
        inner = m.group(2)
        if _is_value_delimiter_echo(inner, name_words):
            return ""
        plain = html_to_plain_text(inner).strip()
        if plain and any(rx.match(plain) for rx in _EVIDENCE_ECHO_LINE_RES):
            return ""
        return m.group(0)

    text = _ECHO_ELEMENT_RE.sub(_drop_echo_element, text)

    kept = []
    for line in text.split("\n"):
        plain = html_to_plain_text(line).strip()
        if plain:
            if "=====" in plain and re.search(r"(?i)\bvalue\s*:", plain):
                continue
            if any(rx.match(plain) for rx in _EVIDENCE_ECHO_LINE_RES):
                continue
            if _is_value_delimiter_echo(plain, name_words):
                continue
        kept.append(line)
    text = "\n".join(kept)

    # Collapse the structures the removals left behind: empty lists/paragraphs
    # and any opening tag left dangling by a removed note.
    text = re.sub(r"(?is)<(ul|ol)\b[^>]*>\s*</\1\s*>", "", text)
    text = re.sub(r"(?is)<p\b[^>]*>\s*</p\s*>", "", text)
    text = re.sub(r"(?is)<p\b[^>]*>\s*(?=<(?:/|h|ul|ol|p|div))", "", text)
    text = re.sub(r"(?is)<p\b[^>]*>\s*$", "", text)
    text = re.sub(r"(?is)^\s*</(?:p|h[1-6])\s*>", "", text)
    # A bare opening <div> that only wrapped the removed scaffolding is dropped.
    if text.lstrip().startswith("<div") and not re.search(r"(?i)</div\s*>", text):
        text = re.sub(r"(?is)^\s*<div\b[^>]*>", "", text, count=1)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _analyze_overall_dimension_batch(dimension_name: str, groups: list, full_query: str,
                                     model_name: str, llm_context_tokens: int,
                                     latest_evidence_date: Optional[str] = None,
                                     include_citations: bool = True,
                                     _evidence_part: Optional[str] = None) -> str:
    """Analyze bounded evidence; split oversized groups without losing records."""
    if not groups:
        return ""

    group_blocks = []
    citations_all = []
    seen_records = set()
    for group in groups:
        citations_all.extend(group["citations"])
        unique_records = [(n, h) for n, h in group["records"] if n not in seen_records]
        seen_records.update(n for n, _ in unique_records)
        group_blocks.append(
            f"\n===== {dimension_name.upper()} VALUE: {group['label']} =====\n"
            f"{_render_dimension_records(unique_records, include_citations=include_citations)}\n"
            f"===== END VALUE: {group['label']} =====\n"
        )
    evidence = _evidence_part if _evidence_part is not None else "\n".join(group_blocks).strip()

    # ⭐ NEW (v3.1.1.17): citation wording depends on the is_citation flag.
    # Extracted before the f-string so both the evidence framing and the
    # CITATIONS instruction switch cleanly.
    _record_word = "Citation No." if include_citations else "Record"
    _citation_rule = (
        "End each distinct analytical point or paragraph with the supporting Citation\n"
        "No. values in parentheses, for example (3, 45, 50). Use only citation numbers\n"
        "actually present in the supplied evidence."
        if include_citations else
        "Do NOT include any citation numbers or parenthesized number lists such as\n"
        "(3, 45, 50) in the output. Do NOT mention the word 'Record' or 'Citation',\n"
        "do NOT reference record numbers, and do NOT write empty parentheses '()'.\n"
        "Write the analysis without referencing numbered records."
    )

    # ⭐ NEW (v3.1.1.14): apply the same analysis-quality guardrails used by the
    # chunk prompts to each Overall Analysis dimension section.
    prompt = f"""
You are producing one section of an Overall Analysis directly from the
original source records.

USER REQUEST:
{full_query}

{AI_ANALYSIS_QUALITY_GUARDRAILS}

ANALYTICAL DIMENSION:
{dimension_name}

TASK:
Analyze and aggregate the supplied source records for this dimension while
following the USER REQUEST exactly. Discuss only facts relevant to this
analytical dimension. Do not repeat equipment specifications, AOI changes or
forecasts belonging to other dimensions. Cover every supplied value/group and every
source record that contributes to this dimension. Describe concrete details,
relationships, changes, recurring observations, and other relevant findings.
Do not treat this as a short overview. Preserve all material source details.
If the same entity or activity occurs in several records, aggregate the
observations while retaining distinct dates, locations, units, equipment,
vehicles, quantities and context.

COVERAGE:
Every supplied {_record_word} must be considered. Do not omit a record because
it is less prominent, unique, repeated, or appears in only one value/group.
Do not invent information or merge distinct facts into a vague statement.

HEADING RULE:
Start directly with your analysis. Do NOT begin with a heading or paragraph
that repeats the analytical dimension name "{dimension_name}" — it is already
shown as the section title.

SOURCE RECORD HANDLING:
The SOURCE RECORDS section below is private input, never report content. Do not
copy it into your answer: do not repeat the "===== ... VALUE: ... ====="
delimiters or any "VALUE:" heading, do not list the evidence fields (lines such
as "+ Record: activity_date: ..."), and do not describe how the records were
grouped or extracted — no phrases like "here are the extracted values", "this
value is associated with the following records", or "this value is a duplicate
... and is not included in the analysis". Write the finished analysis only.

CITATIONS:
{_citation_rule}

STYLE:
Produce a compact but detailed analysis; aggregate repeated observations and
avoid restating the same finding in different wording. Use headings by
value/group only when several distinct values are supplied. Follow the user's requested subject and scope. Clearly separate
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
    if available < MIN_OVERALL_BATCH_OUTPUT_TOKENS:
        overhead = prompt_tokens - _combined_estimate_tokens(evidence)
        data_budget = llm_context_tokens - overhead - MIN_OVERALL_BATCH_OUTPUT_TOKENS - 640
        parts = _split_text_to_token_budget(evidence, data_budget)
        if len(parts) < 2:
            raise ValueError("The prompt exceeds the model context. Shorten the request.")
        return "".join(_analyze_overall_dimension_batch(
            dimension_name, groups, full_query, model_name, llm_context_tokens,
            latest_evidence_date, include_citations=include_citations, _evidence_part=part
        ) for part in parts)
    max_output = min(1536, MAX_OVERALL_OLLAMA_OUTPUT_TOKENS, available)
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
    if not response or "AI analysis unavailable" in response:
        return ""
    # ⭐ v3.1.1.14 fix: drop an LLM-echoed copy of the section title so the
    # report does not show the dimension name twice.
    cleaned = _sanitize_llm_section_html(response, include_citations=include_citations)
    cleaned = sanitize_ai_report_text(_strip_wrappers(cleaned).strip(),
                                      include_citations=include_citations)
    cleaned = _strip_dimension_name_echo(dimension_name, cleaned)
    # ⭐ v3.1.1.20 fix: strip prompt/evidence scaffolding the model echoed back
    # instead of analysing the supplied records.
    cleaned = _strip_evidence_echo(dimension_name, cleaned)
    return cleaned


def _dedupe_report_sections(html: str) -> str:
    """Consolidate duplicate top-level sections in the generated report.

    ⭐ v3.1.1.16 fix (production feedback): chunk analyses and the field-wise
    Overall Analysis can both emit sections under the same analytical topic
    (e.g. "Training and Readiness Analysis" appearing again later in the
    summary). The same topic must appear as ONE section: later <h2> sections
    whose normalized title matches an earlier one are merged under the first
    occurrence instead of being repeated.
    """
    if not html:
        return html

    part_fragment_re = re.compile(r"(?i)\s*[-—–]\s*part\s+\d+\s*(?:of\s*\d+)?\s*")

    def _norm_title(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.lower())
        # Fold simple plurals so "movements" matches "movement".
        return " ".join(w if len(w) <= 3 or not w.endswith("s") else w[:-1] for w in words)

    h2_re = re.compile(r"(?i)<h2[^>]*>(.*?)</h2>", flags=re.S)
    out = []
    seen_titles = {}  # normalized title -> list of later bodies (or None once merged)
    for part in re.split(r"(?i)(?=<h2\b)", html):
        m = h2_re.match(part)
        if not m:
            out.append(part)
            continue
        title_text = html_to_plain_text(part_fragment_re.sub(" ", m.group(1))).strip()
        key = _norm_title(title_text)
        if key and key in seen_titles:
            seen_titles[key].append(part[m.end():])
            continue
        out.append(part)
        if key:
            seen_titles[key] = []
    merged = []
    for part in out:
        m = h2_re.match(part)
        if m:
            title_text = html_to_plain_text(part_fragment_re.sub(" ", m.group(1))).strip()
            key = _norm_title(title_text)
            extras = seen_titles.get(key)
            if extras is not None:
                merged.append(part.rstrip() + "".join(extras))
                seen_titles[key] = None  # mark consumed
                continue
        merged.append(part)
    return "".join(merged)


# ⭐ NEW (v3.1.1.18): headings the pipeline itself emits (NOT the LLM).
OVERALL_DIMENSION_NAMES = (
    "Location Analysis",
    "Force Movement and Deployment Analysis",
    "Equipment and Vehicle Analysis",
    "Infrastructure Analysis",
    "Training and Readiness Analysis",
    "Visits and Inspections Analysis",
    "Force Composition, Strength and Capability Analysis",
    "Events, Security and Other Relevant Analysis",
    "Other Relevant Analysis",
)

_ANALYSIS_CODE_H2_RE = re.compile(
    r"<h2[^>]*>\s*(?:Overall Analysis|Chunk Analysis\b[^<]*|"
    + "|".join(re.escape(n) for n in OVERALL_DIMENSION_NAMES)
    + r")\s*</h2>",
    re.I | re.S,
)


def _rebase_heading_levels(segment: str) -> str:
    """Shift the heading levels in one segment so the shallowest becomes h3."""
    if not segment or "<h" not in segment:
        return segment
    heads = list(re.finditer(r"(?is)<h([1-6])([^>]*)>(.*?)</h\1>", segment))
    if not heads:
        return segment
    shift = 3 - min(int(m.group(1)) for m in heads)
    if shift == 0:
        return segment

    def _repl(m):
        level = max(1, min(6, int(m.group(1)) + shift))
        return f"<h{level}{m.group(2)}>{m.group(3)}</h{level}>"

    return re.sub(r"(?is)<h([1-6])([^>]*)>(.*?)</h\1>", _repl, segment)


def _normalize_analysis_heading_levels(html: str) -> str:
    """Give the Overall Analysis a consistent heading hierarchy.

    ⭐ NEW (v3.1.1.18): the pipeline emits the top-level headings ("Overall
    Analysis", each dimension name, "Chunk Analysis N"); the LLM emits the
    group/sub-topic headings inside them at an unpredictable level — sometimes
    h2 (the same level as its parent dimension), sometimes h3. This rebases the
    LLM headings within each code-generated section so the shallowest becomes
    h3, while preserving the model's relative nesting (h2 -> h3, h3 -> h4, ...).

    Only heading tags are changed; text and document structure are untouched.
    """
    if not html or "<h" not in html:
        return html
    out = []
    last = 0
    seen_code_heading = False
    for m in _ANALYSIS_CODE_H2_RE.finditer(html):
        # Content before the first analysis heading (page title, tab header,
        # styles) is left untouched; only content INSIDE the analysis sections
        # is rebased.
        if seen_code_heading:
            out.append(_rebase_heading_levels(html[last:m.start()]))
        else:
            out.append(html[last:m.start()])
            seen_code_heading = True
        out.append(m.group(0))
        last = m.end()
    if seen_code_heading:
        out.append(_rebase_heading_levels(html[last:]))
    else:
        out.append(html[last:])
    return "".join(out)


def _render_master_analysis(rows_total: int, n_chunks_done: int,
                            max_rows_per_chunk_actual: int,
                            overall_html: str, chunk_html: str) -> str:
    """Assemble the master analysis block shown in the report.

    ⭐ v3.1.1.16 fix (production feedback): a single-chunk run no longer
    renders a "Chunk Analysis 1" heading — the chunk content is emitted
    directly after the Overall Analysis. Multi-chunk runs keep the numbered
    chunk headings.
    """
    overall_block = overall_html if overall_html else "<h2>Overall Analysis</h2><p>Analysis could not be generated.</p>"
    return "<div>" + overall_block + chunk_html + "</div><br>"


def fieldwise_overall_analysis(e_hits: list, full_query: str, model_name: str,
                               llm_context_tokens: int,
                               latest_evidence_date: Optional[str] = None,
                               selected_entity_types=None,
                               include_citations: bool = True) -> str:
    """Build Overall Analysis as concatenated field-wise analyses from original ES records.

    This deliberately does not analyze chunk responses. The original records
    retained for the report are grouped deterministically by analytical
    dimensions, each dimension is analyzed independently, and the resulting
    sections are concatenated in a stable order. There is no final second-pass
    model call that can compress or discard earlier dimension analyses.

    ⭐ NEW (v3.1.1.14) — entity_type_analysis scoping:
      - selected_entity_types is None / empty list -> analyze ALL entity types
        using the user's prompt (previous behavior, unchanged).
      - a non-empty selection -> analyze ONLY the selected entity types. The
        prompt supplied by the caller (the ai_analysis_summary/prompt_master
        prompt) is used as-is; the selection is matched case-insensitively
        against dimension names, their ES source fields and key tokens, with
        typo tolerance. Records that do not match any selected dimension are
        deliberately NOT analyzed.
    """
    if not e_hits:
        return ""

    dimensions = _build_dimension_groups(e_hits)

    # ⭐ v3.1.1.14: restrict to the entity types selected in
    # ai_analysis_summary.entity_type_analysis. Empty/None selection = all.
    selected_dimension_names, unmatched_selections = _match_entity_type_selection_to_dimensions(
        _normalize_entity_type_selection(selected_entity_types)
    )
    # ⭐ Robustness: if the selection contains ONLY unrecognized values, treat it
    # as no restriction (analyze ALL fields) instead of producing an empty
    # Overall Analysis — a missing section looks broken to end users.
    if not selected_dimension_names and unmatched_selections:
        print(
            f"[WARN] Overall Analysis: none of the entity_type_analysis values were recognized "
            f"({', '.join(unmatched_selections)}); analyzing ALL fields instead"
        )
        unmatched_selections = ()
    scope_is_limited = bool(selected_dimension_names)
    # ⭐ v3.1.1.15 change: the prompt from Postgres (ai_analysis_summary.prompt
    # / prompt_master via prompt_id) is used for EVERY Overall Analysis run,
    # scoped or not. entity_type_analysis only restricts WHICH entity-type
    # sections are analyzed; it never replaces the database prompt.
    effective_query = full_query
    if scope_is_limited:
        selected_set = set(selected_dimension_names)
        dimensions = [d for d in dimensions if d.get("name") in selected_set]
        print(
            f"[INFO] Overall Analysis entity-type scope: {len(selected_dimension_names)} selected "
            f"({', '.join(selected_dimension_names)})"
        )
        if unmatched_selections:
            print(
                f"[WARN] Overall Analysis: entity_type_analysis values not recognized and skipped: "
                f"{', '.join(unmatched_selections)}"
            )
    else:
        print("[INFO] Overall Analysis entity-type scope: ALL fields (no entity_type_analysis restriction)")

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
                batch_name, batch, effective_query, model_name, llm_context_tokens, latest_evidence_date,
                include_citations=include_citations
            )
            batch_citations = sorted({citation for group in batch for citation in group["citations"]})
            if result:
                # ⭐ v3.1.1.14 fix: append the Citations tail only when the model
                # did not already emit one — previously both appeared. Plain weight
                # (not bold) per report style; sanitizer normalizes model-bold ones.
                # ⭐ v3.1.1.17: the tail is appended ONLY when is_citation is TRUE.
                if include_citations and not re.search(r"(?i)Citations\s*:", result):
                    citation_tail = "(" + ", ".join(str(n) for n in batch_citations) + ")" if batch_citations else ""
                    if citation_tail:
                        result = result + f"<p>Citations: {citation_tail}</p>"
                dimension_results.append(result)
                total_citations.update(batch_citations)
            else:
                dimension_results.append(_analysis_error_html(
                    f"{batch_name} could not be generated. This part of the analysis is incomplete; please retry."
                ))

        if dimension_results:
            sections.append(f"<h2>{name}</h2>" + "".join(f"<div>{r}</div>" for r in dimension_results))

    # Ensure records which did not land in any semantic dimension still receive
    # analytical attention rather than disappearing silently. When the scope is
    # limited to selected entity types (⭐ v3.1.1.14), records outside the
    # selected dimensions are deliberately NOT analyzed.
    matched = set()
    for dimension in dimensions:
        for group in dimension.get("groups", []):
            matched.update(group.get("citations", []))
    unmatched_citations = sorted(set(range(1, len(e_hits) + 1)) - matched)
    if unmatched_citations and not scope_is_limited:
        fallback_group = {"label": "Other Relevant Records", "citations": unmatched_citations,
                          "records": [(n, e_hits[n - 1]) for n in unmatched_citations]}
        result = _analyze_overall_dimension_batch(
            "Other Relevant Records", [fallback_group], effective_query, model_name,
            llm_context_tokens, latest_evidence_date, include_citations=include_citations
        )
        if result:
            sections.append(f"<h2>Other Relevant Analysis</h2><div>{result}</div>")
            total_citations.update(unmatched_citations)
        else:
            print(f"[WARN] Overall Other Relevant Records analysis returned no usable response for {len(unmatched_citations)} records")
    elif unmatched_citations and scope_is_limited:
        print(
            f"[INFO] Overall Analysis scoped run: {len(unmatched_citations)} record(s) outside the "
            f"selected entity types were not analyzed"
        )

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
                                 raw_evidence_hits: Optional[list] = None,
                                 selected_entity_types=None,
                                 include_citations: bool = True) -> str:
    """Backward-compatible entry point for the Overall Analysis.

    The Overall Analysis is now field-wise and record-driven. It analyzes the
    original Elasticsearch records retained for the report, not the chunk
    responses. Each analytical dimension is generated independently and the
    resulting sections are concatenated without a final compression pass.

    ⭐ NEW (v3.1.1.14): selected_entity_types carries the normalized
    ai_analysis_summary.entity_type_analysis value. None/empty -> analyze ALL
    entity types (legacy behavior); a non-empty list restricts the Overall
    Analysis to the selected entity types only.

    ⭐ NEW (v3.1.1.17): include_citations carries the row's is_citation flag;
    when False the generated analysis does not mention citation numbers.
    """
    if raw_evidence_hits:
        return fieldwise_overall_analysis(
            raw_evidence_hits, full_query, model_name, llm_context_tokens,
            latest_evidence_date=latest_evidence_date,
            selected_entity_types=selected_entity_types,
            include_citations=include_citations
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

    # ⭐ NEW (v3.1.1.14): same analysis-quality guardrails as the chunk/field-wise
    # prompts, applied to the Overall Analysis fallback synthesis.
    prompt = f"""
USER REQUEST:
{full_query}

{AI_ANALYSIS_QUALITY_GUARDRAILS}

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
    # ⭐ NEW (v3.1.1.14/15): row carries entity_type_analysis and
    # summary_type_publication (5-tuple), or a legacy shorter shape.
    # ⭐ NEW (v3.1.1.17): 6-tuple rows also carry is_citation. Citations are
    # included in the report ONLY when is_citation is TRUE.
    # ⭐ NEW (v3.1.1.18): 7-tuple rows also carry is_chunk. Per-chunk analysis
    # runs ONLY when is_chunk = 1. When the column is absent (older row shape)
    # is_chunk defaults to True so legacy deployments are unaffected.
    if len(row) >= 7:
        (elastic_query, ai_analysis_summary_id, search_form_type,
         selected_entity_types, summary_type_publication, is_citation, is_chunk) = row[:7]
    elif len(row) >= 6:
        (elastic_query, ai_analysis_summary_id, search_form_type,
         selected_entity_types, summary_type_publication, is_citation) = row[:6]
        is_chunk = True
    elif len(row) >= 5:
        (elastic_query, ai_analysis_summary_id, search_form_type,
         selected_entity_types, summary_type_publication) = row[:5]
        is_citation = False
        is_chunk = True
    elif len(row) == 4:
        elastic_query, ai_analysis_summary_id, search_form_type, selected_entity_types = row[:4]
        summary_type_publication = False
        is_citation = False
        is_chunk = True
    else:
        elastic_query, ai_analysis_summary_id, search_form_type = row
        selected_entity_types = None
        summary_type_publication = False
        is_citation = False
        is_chunk = True
    include_citations = bool(is_citation)
    include_chunks = bool(is_chunk)
    # ⭐ NEW: entity_type_analysis = ['new_prompt'] means the user supplied their
    # own prompt for this run. Such a row must analyze the raw records matched by
    # filter_json — NOT the aggregated (form_id-grouped) records.
    new_prompt_mode = _is_new_prompt_selection(selected_entity_types)
    if new_prompt_mode:
        print(
            f"[INFO] Row {ai_analysis_summary_id}: entity_type_analysis contains "
            f"'new_prompt' — filter_json records will be used without aggregation"
        )
    if not include_citations:
        print(f"[INFO] Row {ai_analysis_summary_id}: is_citation is FALSE — report will not mention citations")
    if not include_chunks:
        print(
            f"[INFO] Row {ai_analysis_summary_id}: is_chunk is FALSE — per-chunk analyses will be skipped "
            f"({'TYPE_MAPPING-only analysis is used' if new_prompt_mode else 'Overall Analysis is still generated'})"
        )
    search_form_type = (search_form_type or "").strip().lower()
    if not search_form_type:
        # ⭐ NEW (robustness): a NULL/empty search_form_type used to crash here
        # (AttributeError) and wedge the row at query_status=2 forever. Write a
        # visible error report and finish the row cleanly instead.
        err_html = (
            "<html><body><h3>AI Analysis Error</h3>"
            "<p>This request could not be processed: the analysis type "
            "(search_form_type) is missing for this row. "
            "Please re-submit the request.</p></body></html>"
        )
        update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id,
                                             base64.b64encode(err_html.encode("utf-8")).decode("ascii"))
        print(f"[WARN] Row {ai_analysis_summary_id}: NULL/empty search_form_type — error report written, row marked done")
        return
    # ⭐ NEW (v3.1.1.14): defaults so the Profile Analysis path (which skips the
    # standard ES-fetch/report-stats block below) still has a consistent state.
    e_hits = []
    dedup_stats = {}
    is_general_analysis = False
    model_prompt_dict = None
    if search_form_type != 'profile analysis':
        model_prompt_dict = get_prompt_and_model(cursor, ai_analysis_summary_id)

        if model_prompt_dict.get("table_hits") is not None:
            new_prompt_mode = True

        is_general_analysis = (
            search_form_type in [t.lower() for t in TYPE_MAPPING["general"]["types"]]
        )

        table_hits = model_prompt_dict.get("table_hits")
        if table_hits is not None:
            e_response = {"hits": {"total": {"value": len(table_hits)}, "hits": table_hits}}
            print(f"[INFO] Using {len(table_hits)} submitted table records; ES query skipped")
        else:
            e_response = get_data_from_elastic(
                elastic_query,
                perform_dedup=not (is_general_analysis or new_prompt_mode)
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
        # ⭐ NEW: when entity_type_analysis carries 'new_prompt', skip step 1-4 and
        # use exactly the records the filter_json query returned.
        if is_general_analysis:
            if new_prompt_mode:
                print(
                    f"[INFO] new_prompt: using all {len(e_hits)} records matched by "
                    f"filter_json (form_id aggregation and general deduplication skipped)"
                )
            else:
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
            model_prompt_dict = get_prompt_and_model(cursor, ai_analysis_summary_id)
            table_hits = model_prompt_dict.get("table_hits")
            if table_hits is not None:
                e_hits = table_hits
            else:
                profile_response = get_data_from_elastic(elastic_query) if elastic_query else None
                e_hits = profile_response.get("hits", {}).get("hits", []) if profile_response else []
                dedup_stats = profile_response.get("dedup_stats", {}) if profile_response else {}
            if e_hits:
                # Profiles can be selected through bool/filter/terms as well as
                # match_phrase_prefix. The query already defines the selection;
                # do not require one particular query syntax to produce output.
                llm_response = analyze_type_mapping_records(
                    convert_hits_to_markdown(e_hits, search_form_type, preserve_all_fields=True),
                    model_prompt_dict.get("prompt", ""),
                    model_prompt_dict.get("model_info", LLM_MODEL), len(e_hits),
                    llm_context_tokens or 8192, output_tokens=consolidation_output_tokens,
                    include_citations=include_citations,
                )

        # ── Standard Analysis Types ──
        for key, value in TYPE_MAPPING.items():
            if search_form_type in value["types"]:
                print(f"[INFO] AI analysis for {key} [{ai_analysis_summary_id}]")

                if not e_hits:
                    print(f"[WARN] No ES hits for {key}")
                    continue

                # Step 1: Convert ES hits to markdown
                markdown_data = convert_hits_to_markdown(e_hits, search_form_type, preserve_all_fields=new_prompt_mode)
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
                # The model and source table must use exactly the same order.
                markdown_data = convert_hits_to_markdown(
                    e_hits, search_form_type, preserve_all_fields=new_prompt_mode
                )

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

                # ⭐ NEW: 'new_prompt' rows are analyzed only from the
                # TYPE_MAPPING record markdown. The field-wise Overall Analysis
                # (multiple aggregated entity-type sections) is NOT run for this
                # mode. When chunking is enabled (is_chunk=1) the normal chunk
                # stage runs on that same TYPE_MAPPING markdown; otherwise this
                # single analysis covers all records.
                if new_prompt_mode and not include_chunks:
                    print(
                        f"[INFO] new_prompt: TYPE_MAPPING-only analysis for {key} "
                        f"({n_rows} records) — Overall Analysis aggregation skipped"
                    )
                    temp_llm_response = analyze_type_mapping_records(
                        markdown_data, full_query, model_name, n_rows,
                        llm_context_tokens,
                        output_tokens=consolidation_output_tokens,
                        latest_evidence_date=latest_evidence_date,
                        include_citations=include_citations,
                    )
                    if temp_llm_response:
                        llm_response += f"<div>{temp_llm_response}</div><br>"
                    else:
                        llm_response += (
                            "<div><h2>Analysis Unavailable</h2>"
                            "<p>AI returned no response for this group of records.</p>"
                            "</div><br>"
                        )
                    continue

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

                # ⭐ NEW: for new_prompt rows every chunk document is the
                # TYPE_MAPPING markdown of that chunk's records — no other
                # record view and no aggregation are used.
                if new_prompt_mode:
                    for _chunk_info in record_chunks:
                        _chunk_info["document"] = convert_hits_to_markdown(
                            _chunk_info["rows"], search_form_type, preserve_all_fields=True,
                            start_row_idx=_chunk_info["start"]
                        )

                print(
                    f"[INFO] Token-aware chunking: {n_rows} rows → {n_chunks} chunks "
                    f"({club_reason}, target ~{chunk_size} document tokens)"
                )

                # ⭐ NEW (v3.1.1.18): is_chunk gates the per-chunk stage. When
                # the row's is_chunk flag is not 1 the chunk documents must NOT
                # be built or analysed; only the Overall Analysis is produced.
                if not include_chunks:
                    print(
                        f"[INFO] is_chunk is FALSE for row {ai_analysis_summary_id}: "
                        f"{n_rows} records -> per-chunk analysis skipped "
                        f"(Overall Analysis only)"
                    )
                    record_chunks = []
                    n_chunks = 0

                chunk_workers = _resolve_chunk_workers(chunk_workers_arg)
                print(
                    f"[INFO] Chunk execution: {chunk_workers} worker(s) "
                    f"(set --chunk_workers N or AI_SUMMARY_CHUNK_WORKERS=N)"
                )

                chunk_responses_by_idx = {}
                chunk_loop_start = time.time()

                # ⭐ NEW (v3.1.1.17): the 8-tuple job also carries the row's
                # is_citation flag so chunk prompts honor it.
                jobs = [
                    (
                        idx,
                        n_chunks,
                        chunk_info,
                        full_query,
                        model_name,
                        f"{key}_{ai_analysis_summary_id}",
                        latest_evidence_date,
                        include_citations,
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
                        job[7],
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
                # ⭐ NEW (v3.1.1.18): is_chunk gates ONLY the per-chunk analyses.
                # When is_chunk is FALSE the per-chunk stage is skipped, but the
                # Overall Analysis is still generated from the original records.
                rows_total = len(e_hits)
                n_chunks_done = len(chunk_responses)
                max_rows_per_chunk_actual = max((len(info["rows"]) for info in record_chunks), default=0)

                chunk_html = ""
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
                    # ⭐ v3.1.1.16: a single-chunk run renders its content
                    # directly (no "Chunk Analysis 1" heading). Multi-chunk
                    # runs keep the numbered, dated chunk headings.
                    single_chunk_mode = len(chunk_responses) == 1
                    for chunk_num, (chunk_start, chunk_end, chunk_html_text) in enumerate(chunk_responses, 1):
                        chunk_rows = e_hits_sorted[chunk_start - 1:chunk_end]
                        d_min, d_max = _chunk_dates(chunk_rows)
                        chunk_row_count = chunk_end - chunk_start + 1
                        cited_numbers = _extract_citation_numbers(chunk_html_text, chunk_start, chunk_end)
                        if not cited_numbers:
                            cited_numbers = list(range(chunk_start, chunk_end + 1))
                        citation_tail = "(" + ", ".join(str(n) for n in cited_numbers) + ")"
                        if single_chunk_mode:
                            heading = ""
                        elif include_citations:
                            heading = (
                                "<h2>Chunk Analysis "
                                f"{chunk_num} — Records {chunk_start} to {chunk_end} "
                                f"({chunk_row_count} records, {d_min} to {d_max})</h2>"
                            )
                        else:
                            # ⭐ v3.1.1.15: without citations the report must
                            # not label records (no "Records 5 to 10").
                            heading = (
                                "<h2>Chunk Analysis "
                                f"{chunk_num} ({chunk_row_count} records, {d_min} to {d_max})</h2>"
                            )
                        chunk_sections.append(
                            f"{heading}"
                            f"<div>{chunk_html_text}</div>"
                            + (f"<p>Citations: {citation_tail}</p>" if include_citations else "")
                        )
                    chunk_html = "".join(chunk_sections)

                if n_chunks and len(chunk_responses) < n_chunks:
                    chunk_html += _analysis_error_html(
                        f"{n_chunks - len(chunk_responses)} source portions could not be analyzed. "
                        "The chunk analysis is incomplete; check AI service availability and retry."
                    )
                overall_html = ""
                overall_response = ""
                if new_prompt_mode:
                    print(
                        "[INFO] new_prompt: Overall Analysis skipped — only the "
                        "TYPE_MAPPING record analyses are generated"
                    )
                try:
                    if not new_prompt_mode:
                        overall_token_budget = int(llm_context_tokens * 0.85)
                        overall_response = synthesize_combined_analysis(
                            chunk_responses, full_query=full_query, model_name=model_name,
                            token_budget=overall_token_budget, total_rows=rows_total,
                            club_size=max_rows_per_chunk_actual, llm_context_tokens=llm_context_tokens,
                            latest_evidence_date=latest_evidence_date,
                            output_tokens=consolidation_output_tokens,
                            raw_evidence_document=build_overall_source_document(e_hits),
                            raw_evidence_hits=e_hits,
                            selected_entity_types=selected_entity_types,
                            include_citations=include_citations,
                        )
                    if overall_response:
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
                            f"<div>{overall_response}</div>"
                            + (f"<p>Citations: {overall_citations}</p>" if include_citations else "")
                        )
                except Exception as e:
                    print(f"[WARN] Overall analysis failed: {e}")

                if overall_html or chunk_html:
                    # ⭐ v3.1.1.16: single-chunk runs no longer show a
                    # "Chunk Analysis 1" heading, and duplicate topic
                    # sections are consolidated into one across the whole
                    # master block (Overall Analysis + chunk bodies).
                    # ⭐ v3.1.1.18: normalize the heading hierarchy BEFORE the
                    # h2-level de-dup so LLM-emitted group headings are not
                    # mistaken for top-level sections.
                    temp_llm_response = _dedupe_report_sections(
                        _normalize_analysis_heading_levels(_render_master_analysis(
                            rows_total, n_chunks_done, max_rows_per_chunk_actual,
                            overall_html, chunk_html,
                        ))
                    )
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

    llm_response = _dedupe_repeated_sentences(llm_response)
    if ai_summ and not html_to_plain_text(llm_response).strip():
        llm_response = _analysis_error_html(
            "No matching source records were found for this request. Check the filters."
            if not e_hits else "No analysis was returned for the selected analysis type. Please retry or check the service logs."
        )

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
        if new_prompt_mode:
            query_note = (
                f"Showing {len(e_hits)} records matched by the filter query from "
                f"{query_total or len(e_hits)} raw ES matches (form_id aggregation skipped)"
            )
        else:
            query_note = (
                f"Showing {len(e_hits)} records retained from {query_total or len(e_hits)} raw ES matches "
                f"after latest-record grouping by form_id and general deduplication"
            )
    date_range = None
    if elastic_query:
        date_range = extract_date_range(elastic_query)
    doc_table_html = build_doc_id_activity_date_table(
        e_hits, query_total=query_total, date_range=date_range, query_note=query_note
    ) if (e_hits and include_citations) else ""
    if e_hits and not include_citations:
        print("[INFO] Record Source Details citation table omitted (is_citation is FALSE)")

    # ⭐ NEW (v3.1.1.15): Publication Summary — rendered BEFORE the citation
    # table ONLY when summary_type_publication is TRUE. Any API error, timeout
    # or missing token leaves the report unchanged (section skipped).
    #
    # ⭐ v3.1.1.26: the publication API performs its OWN unified search across
    # every Ready publication (the same all-publications search as the
    # Publication Assistant chat) and answers from the AI summary prompt. It is
    # NOT scoped by this row's filter_json or by the ES hits, so the section
    # must be attempted even when the record query returned nothing — gating on
    # e_hits silently dropped the only publication-derived content in that case.
    publication_summary_html = ""
    if summary_type_publication:
        _pub_prompt = (
            model_prompt_dict.get("prompt", "") if model_prompt_dict else ""
        ).strip()
        if not _pub_prompt:
            print("[WARN] Publication summary skipped: row has no prompt to send")
        else:
            try:
                _pub_model = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
                publication_summary_html = build_publication_summary_section(
                    e_hits or [],
                    user_prompt=_pub_prompt,
                    filter_json=elastic_query if isinstance(elastic_query, dict) else None,
                    model_name=_pub_model,
                    page_name=f"AI Analysis {search_form_type} [{ai_analysis_summary_id}]",
                )
                if not publication_summary_html:
                    publication_summary_html = (
                        "<p>Publication analysis is unavailable. The publication service returned no usable answer. "
                        "This report is incomplete for publications; check service availability and ingestion, then retry.</p>"
                    )
                    print(
                        "[WARN] Publication summary: no usable API answer for this prompt; "
                        "an availability warning is included in the report"
                    )
            except Exception as pub_exc:
                print(f"[WARN] Publication summary section skipped: {pub_exc}")
                publication_summary_html = "<p>Publication analysis failed. Check the publication service, then retry.</p>"

    # ⭐ NEW (v3.1.1.14): Equipment Specifications — added only when the row's
    # entity_type_analysis (jsonb array) selects 'equipment_name'. Collect the
    # unique equipments referenced by the retained records and look up their
    # specifications in Elasticsearch.
    equipment_specifications_html = ""
    if "equipment_name" in _normalize_entity_type_selection(selected_entity_types) and e_hits:
        try:
            equipment_pairs = _collect_equipment_names(e_hits)
            equipment_specifications_html = fetch_equipment_specifications(equipment_pairs)
        except Exception as eq_exc:
            print(f"[WARN] Equipment specifications section skipped: {eq_exc}")
            equipment_specifications_html = ""

    # ⭐ v3.1.1.24: Enemy Formation Details — rendered ONLY when the row's
    # entity_type_analysis actually selects Formation (the UI's "Formation"
    # option is stored as "enemy_formation_name"), mirroring how Equipment
    # Specifications is gated on "equipment_name". Previously the section
    # appeared for EVERY analysis whose records merely happened to carry
    # enemy_formation_name. An empty selection ("Overall Data" = all fields)
    # does NOT render the section. The formations are then looked up in the
    # ORBAT documents (injester_name='orbat_injester').
    formation_selected = any(
        ("formation" in sel) or ("orbat" in sel)
        for sel in _normalize_entity_type_selection(selected_entity_types)
    )
    formation_details_html = ""
    if formation_selected and e_hits:
        try:
            formation_pairs = _collect_formation_names(e_hits)
            print("[INFO] Enemy formation details: formation entity type selected")
            formation_details_html = fetch_formation_details(formation_pairs)
        except Exception as fm_exc:
            print(f"[WARN] Enemy formation details section skipped: {fm_exc}")
            formation_details_html = ""

    # ⭐ NEW: IMINT imagery — the all_images paths of every IMINT record in the
    # source hits, rendered deterministically (see build_imint_imagery_html).
    # Records without all_images, and non-IMINT analyses, are unaffected.
    imint_imagery_html = build_imint_imagery_html(e_hits) if e_hits else ""
    if imint_imagery_html:
        print("[INFO] IMINT imagery section: image paths rendered from all_images")

    tabbed_html = build_tabbed_html(
        [llm_response, ai_trend_response, ai_change_repsonse],
        doc_table_html=doc_table_html,
        publication_summary_html=publication_summary_html,
        equipment_specifications_html=equipment_specifications_html,
        formation_details_html=formation_details_html,
        imagery_links_html=imint_imagery_html,
    )
    # ⭐ NEW (v3.1.1.15): final safety net — when citations were NOT requested
    # (is_citation is FALSE) the assembled report must not contain 'Record N' or
    # 'Item N' labels, regardless of what the LLM emitted or which section they
    # appear in. Citation numbers are left intact.
    tabbed_html = _strip_record_labels_from_report(tabbed_html, include_citations=include_citations)
    # ⭐ NEW (v3.1.1.18): drop headings whose title is just a data junk token
    # (e.g. "unknown", "tpt"); the content beneath them is preserved.
    tabbed_html = _unwrap_junk_headings(tabbed_html)
    # ⭐ NEW (v3.1.1.19): present every date in the report as DD-MM-YYYY.
    tabbed_html = format_dates_for_display(tabbed_html)
    ai_analysis_text_base64 = base64.b64encode(tabbed_html.encode("utf-8")).decode("utf-8")

    update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id, ai_analysis_text_base64)


# ──────────────────────────────────────────────
# IMINT Analysis
# ──────────────────────────────────────────────
def _imint_extract_report_text(llm_response) -> str:
    """Return the IMINT report HTML from a direct-chat LLM response.

    ⭐ v3.1.1.15 fix: the response used to be pushed through
    ``correct_json_format()``, which collapses ", "/newlines and returns
    ``{"summary": ...}`` — so an HTML answer came back mangled and a JSON answer
    without a "summary" key came back as an empty string (blank report). This
    keeps the HTML intact and only unwraps a JSON envelope when the model
    actually returned one.
    """
    text = str(llm_response or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("summary", "analysis", "report", "response", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    # Drop a ```html ... ``` fence if the model wrapped the report in one.
    text = re.sub(r"^\s*```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def imint_ai_analysis_check(row, cursor):
    elastic_query, imint_ai_analysis_id = row
    model_prompt_dict = get_prompt_and_model(cursor, imint_ai_analysis_id, imint=True)
    print("[INFO] for imint ai analysis", imint_ai_analysis_id)
    # ⭐ v3.1.1.15 fix: the row must ALWAYS be finished with a report. Previously
    # an empty filter_json returned before the UPDATE (row stuck at status=2), and
    # a hit set without comments wrote an empty string (done, but no report).
    ai_analysis_text = ""
    had_comments = False
    if isinstance(elastic_query, str):
        try:
            elastic_query = json.loads(elastic_query)
        except Exception:
            print(f"[WARN] IMINT {imint_ai_analysis_id}: filter_json is not valid JSON")
            elastic_query = None
    if elastic_query:
        comments = []
        e_response = get_data_from_elastic(elastic_query)
        if e_response and e_response['hits']['total']['value'] > 0:
            imint_results = e_response['hits']['hits']
            for single_json in imint_results:
                single_json = single_json['_source']
                if single_json.get("comments"):
                    location_name = str(single_json.get("location_name") or "").strip()
                    activity_date = single_json.get("activity_date", "")
                    latest_comment = str(single_json["comments"][-1]).strip().replace('\n', '. ')
                    print("['comments']", latest_comment, location_name, activity_date)
                    complete_single_comment = "location_name: " + location_name + ", date:" + str(
                        activity_date) + ', comments: ' + latest_comment
                    if complete_single_comment not in comments:
                        comments.append(complete_single_comment)
        if comments:
            had_comments = True
            llm_prompt = '\n\n'.join(comments) + "\n\n" + model_prompt_dict.get("prompt") + \
                         "The response you will give should be in html(hyper text markup language). " \
                         "Any title in the response should be in h2 tag and paragraph should be in p tag."
            _imint_model = model_prompt_dict.get("model_info") or LLM_MODEL
            # ⭐ v3.1.1.15 fix: every other analysis in this version talks to
            # Ollama directly; this legacy path still used the AnythingLLM chat
            # API (IFC_LLM_PORT), which is not served in production — so the LLM
            # call always failed and the stored report was "AI analysis
            # unavailable". Use direct Ollama first, keep AnythingLLM as fallback.
            _imint_context_tokens = _safe_int(os.getenv("LLM_CONTEXT_TOKENS")) or 8192
            llm_response = _get_direct_ollama_response(
                llm_prompt, _imint_model, _imint_context_tokens,
                stage_label="imint-analysis")
            if not llm_response:
                llm_response = get_llm_response_direct(llm_prompt, _imint_model)
            # ⭐ v3.1.1.15 fix: keep the LLM HTML as-is (see helper above).
            ai_analysis_text = _imint_extract_report_text(llm_response)
            print("[DONE] imint_ai_analysis_id: ", imint_ai_analysis_id)
    if not ai_analysis_text:
        if had_comments:
            ai_analysis_text = ("<html><body><h2>IMINT Analysis</h2>"
                                "<p>The AI model did not return an analysis for the matched "
                                "records. Please re-submit the request.</p></body></html>")
        else:
            ai_analysis_text = ("<html><body><h2>IMINT Analysis</h2>"
                                "<p>No imagery-intelligence records with comments were matched "
                                "by this query, so no analysis could be generated.</p></body></html>")
        print(f"[WARN] imint_ai_analysis_id {imint_ai_analysis_id}: no LLM report produced; "
              f"writing fallback report so the row is not left without output")
    ai_analysis_text_base64 = str(base64.b64encode(ai_analysis_text.encode('utf-8')).decode('utf-8'))
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
        if result and result[0]:
            output_dict["prompt"] = result[0]
        else:
            output_dict["prompt"] = "Analyze the supplied source records. Describe supported activities, locations, entities and changes without inventing facts."
            print(f"[WARN] Row {primary_key}: historical prompt id {prompt_id} no longer exists; using the standard source-analysis request")
    else:
        # Legacy standard/profile rows may have neither an explicit prompt nor
        # a prompt_master id. They still need a useful deterministic default.
        output_dict["prompt"] = "Analyze the supplied source records. Describe supported activities, locations, entities and changes without inventing facts."
        print(f"[WARN] Row {primary_key}: no prompt configured; using the standard source-analysis request")

    if model_id:
        model_subquery = "SELECT ai_model_info FROM ai_model_master WHERE ai_model_id = %s;"
        cursor.execute(model_subquery, (model_id,))
        result = cursor.fetchone()
        if result and result[0]:
            output_dict["model_info"] = result[0]
        else:
            output_dict["model_info"] = LLM_MODEL
            print(f"[WARN] Row {primary_key}: historical model id {model_id} no longer exists; using the configured default")
    else:
        output_dict["model_info"] = LLM_MODEL
        print(f"[WARN] Row {primary_key}: no model selected; using the configured default")
    output_dict["prompt"], table_hits = _extract_table_prompt(output_dict.get("prompt", ""))
    if table_hits is not None:
        output_dict["table_hits"] = table_hits
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
            # ⭐ NEW (v3.1.1.14/15): rows carry entity_type_analysis and
            # summary_type_publication (5-tuple) or a legacy shorter shape.
            # ⭐ NEW (v3.1.1.17): 6-tuple rows also carry is_citation; only
            # rows with is_citation = TRUE include citations in the report.
            # ⭐ NEW (v3.1.1.18): 7-tuple rows also carry is_chunk.
            if len(row) >= 7:
                (filter_json, ai_analysis_summary_id, search_form_type,
                 selected_entity_types, summary_type_publication, is_citation, is_chunk) = row[:7]
            elif len(row) >= 6:
                (filter_json, ai_analysis_summary_id, search_form_type,
                 selected_entity_types, summary_type_publication, is_citation) = row[:6]
                is_chunk = True
            elif len(row) >= 5:
                (filter_json, ai_analysis_summary_id, search_form_type,
                 selected_entity_types, summary_type_publication) = row[:5]
                is_citation = False
                is_chunk = True
            elif len(row) == 4:
                filter_json, ai_analysis_summary_id, search_form_type, selected_entity_types = row[:4]
                summary_type_publication = False
                is_citation = False
                is_chunk = True
            else:
                filter_json, ai_analysis_summary_id, search_form_type = row
                selected_entity_types = None
                summary_type_publication = False
                is_citation = False
                is_chunk = True
            ai_analysis_summary_check((filter_json, ai_analysis_summary_id, search_form_type,
                                       selected_entity_types, bool(summary_type_publication),
                                       bool(is_citation), bool(is_chunk)),
                          ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                          chunk_token_threshold, chunk_size,
                          chunk_output_tokens, consolidation_output_tokens,
                          llm_context_tokens, chunk_workers_arg)
        conn.commit()
        conn.close()
    except Exception as e:
        if 'conn' in locals() and conn is not None:
            try:
                conn.rollback()
            except Exception as cleanup_exc:
                print(f"[WARN] Failed connection rollback: {cleanup_exc}")
            finally:
                conn.close()
        print(f"[ERROR] Error processing row {row[1] if len(row) > 1 else '?'}: {e}")
        traceback.print_exc()
        # ⭐ NEW (robustness): don't leave the row wedged at query_status=2 when
        # processing crashes (e.g. VPN flap, zombie LLM request). First failure →
        # requeue for retry; second failure → terminal error report in query_text.
        try:
            row_id = row[1] if isinstance(row, (list, tuple)) and len(row) >= 2 else None
            if row_id is not None:
                fix_conn = postgres_connection()
                try:
                    with fix_conn.cursor() as fix_cur:
                        if row_id in _requeued_failed_rows:
                            fix_cur.execute(
                                "UPDATE public.ai_analysis_summary SET query_status = 1, query_text = %s "
                                "WHERE ai_analysis_summary_id = %s",
                                (base64.b64encode(_analysis_error_html(
                                    "Processing failed repeatedly. Please re-submit the request; see service logs for details."
                                ).encode("utf-8")).decode("ascii"), row_id))
                            _requeued_failed_rows.discard(row_id)
                            print(f"[WARN] Row {row_id} failed twice — marked done with error report")
                        else:
                            fix_cur.execute(
                                "UPDATE public.ai_analysis_summary SET query_status = 0 "
                                "WHERE ai_analysis_summary_id = %s",
                                (row_id,))
                            _requeued_failed_rows.add(row_id)
                            print(f"[WARN] Row {row_id} crashed — requeued for retry")
                    fix_conn.commit()
                finally:
                    fix_conn.close()
        except Exception as e2:
            print(f"[ERROR] Failed-row recovery failed: {e2}")


def imint_ai_analysis_summary_check_threadsafe(row):
    try:
        conn = postgres_connection()
        try:
            with conn.cursor() as cursor:
                filter_json, imint_ai_analysis_id = row
                imint_ai_analysis_check((filter_json, imint_ai_analysis_id), cursor)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[ERROR] Error processing imint row {row}: {e}")
        traceback.print_exc()
        # ⭐ v3.1.1.15 fix: an exception used to leave the row locked at status=2
        # for good — no report, no retry. Requeue once; a second failure stores
        # an error report and marks the row done so the caller is never waiting.
        try:
            row_id = row[1] if isinstance(row, (list, tuple)) and len(row) >= 2 else None
            if row_id is not None:
                fix_conn = postgres_connection()
                try:
                    with fix_conn.cursor() as fix_cur:
                        if row_id in _requeued_failed_imint_rows:
                            err_html = ("<html><body><h2>IMINT Analysis</h2>"
                                        "<p>Processing failed repeatedly. Please re-submit the "
                                        "request; see service logs for details.</p></body></html>")
                            fix_cur.execute(
                                "UPDATE public.imint_ai_analysis SET status = 1, ai_analysis_text = %s "
                                "WHERE imint_ai_analysis_id = %s",
                                (base64.b64encode(err_html.encode('utf-8')).decode('utf-8'), row_id))
                            _requeued_failed_imint_rows.discard(row_id)
                            print(f"[WARN] IMINT row {row_id} failed twice — marked done with error report")
                        else:
                            fix_cur.execute(
                                "UPDATE public.imint_ai_analysis SET status = 0 "
                                "WHERE imint_ai_analysis_id = %s",
                                (row_id,))
                            _requeued_failed_imint_rows.add(row_id)
                            print(f"[WARN] IMINT row {row_id} crashed — requeued for retry")
                    fix_conn.commit()
                finally:
                    fix_conn.close()
        except Exception as e2:
            print(f"[ERROR] IMINT failed-row recovery failed: {e2}")


def lock_rows_for_processing_1(conn):
    # ⭐ NEW (v3.1.1.14/15): also RETURN entity_type_analysis (jsonb array that
    # scopes the Overall Analysis) and summary_type_publication (bool that adds
    # the Publication Summary section).
    # ⭐ NEW (v3.1.1.17): also RETURN is_citation (bool). Citations appear in
    # the report ONLY when is_citation is TRUE. Older deployments without the
    # is_citation column fall back to the 5-column shape, then to the legacy
    # 3-column RETURNING shape.
    # ⭐ NEW (v3.1.1.18): also RETURN is_chunk (int). Per-chunk analysis runs
    # ONLY when is_chunk = 1; anything else (0 or NULL) skips the chunk stage
    # while the Overall Analysis is still generated from the original records.
    # Deployments without the column fall back to the 6-column shape.
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
                RETURNING filter_json, ai_analysis_summary_id, search_form_type,
                          entity_type_analysis, summary_type_publication, is_citation,
                          is_chunk;
            """)
            return cursor.fetchall()
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        print("[WARN] is_chunk column not found; retrying without it")
        return _lock_rows_without_is_chunk(conn)
    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] ProgrammingError in lock_rows_for_processing_1: {e}")
        return _lock_rows_without_is_chunk(conn)


def _lock_rows_without_is_chunk(conn):
    """Fallback: RETURN the 6-column shape used before is_chunk was added.

    ⭐ v3.1.1.18: this path is only reached when the is_chunk column is
    unavailable. Rows locked through it default is_chunk to TRUE (legacy
    behavior: per-chunk analysis runs), so a deployment without the column
    keeps working exactly as before.
    """
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
                RETURNING filter_json, ai_analysis_summary_id, search_form_type,
                          entity_type_analysis, summary_type_publication, is_citation;
            """)
            return cursor.fetchall()
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        print("[WARN] is_citation column not found; retrying without it")
        return _lock_rows_without_is_citation(conn)
    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] ProgrammingError in _lock_rows_without_is_chunk: {e}")
        return _lock_rows_without_is_citation(conn)


def _lock_rows_without_is_citation(conn):
    """Fallback: RETURN the 5-column shape used before is_citation was added.

    ⭐ v3.1.1.17: rows locked through this path default is_citation to FALSE,
    so no citations are mentioned in their reports.
    """
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
                RETURNING filter_json, ai_analysis_summary_id, search_form_type,
                          entity_type_analysis, summary_type_publication;
            """)
            return cursor.fetchall()
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        print("[WARN] entity_type_analysis/summary_type_publication columns not found; using legacy row shape")
        return _lock_rows_legacy_shape(conn)
    except psycopg2.ProgrammingError as e2:
        print(f"[ERROR] ProgrammingError in lock_rows_for_processing_1 (5-column fallback): {e2}")
        return _lock_rows_legacy_shape(conn)


def _lock_rows_legacy_shape(conn):
    """Legacy fallback: RETURN the original 3 columns only."""
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
    except psycopg2.ProgrammingError as e2:
        print(f"[ERROR] ProgrammingError in lock_rows_for_processing_1 (fallback): {e2}")
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


def reset_stale_processing_rows() -> dict:
    """⭐ NEW (robustness): startup sweep for the hard-kill wedge bug.

    A hard kill (kill -9, power loss, VPN-flap OOM) bypasses the in-process
    auto-requeue, leaving rows locked at "processing" (query_status/status = 2)
    forever — a later restart finds nothing to do because the lock queries only
    select rows at 0, so the wedge persists indefinitely. This sweep runs ONCE
    at loop startup and resets any row still marked 2 back to 0 (pending).
    Safe at startup because no instance of ours is mid-row yet.

    Multi-instance deployments: set AI_SUMMARY_NO_RESET_STALE=1 to skip the
    sweep when another instance may be actively processing rows.

    Returns {"ai_analysis_summary": n, "imint_ai_analysis": n}, or {} on failure.
    """
    counts = {}
    conn = postgres_connection()
    try:
        with conn.cursor() as cursor:
            for table, id_col, status_col in (
                ("ai_analysis_summary", "ai_analysis_summary_id", "query_status"),
                ("imint_ai_analysis", "imint_ai_analysis_id", "status"),
            ):
                cursor.execute(
                    f"UPDATE public.{table} SET {status_col} = 0 "
                    f"WHERE {status_col} = 2"
                )
                counts[table] = cursor.rowcount
        conn.commit()
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[WARN] Startup sweep failed (leaving rows untouched): {exc}")
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if any(counts.values()):
        print(f"[INFO] Startup sweep: reset stale in-progress rows to pending: "
              f"ai_analysis_summary={counts['ai_analysis_summary']}, "
              f"imint_ai_analysis={counts['imint_ai_analysis']}")
    else:
        print("[INFO] Startup sweep: no stale in-progress rows found")
    return counts


def run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                 chunk_token_threshold=3000, chunk_size=2200,
                                 chunk_output_tokens=800, consolidation_output_tokens=1500,
                                 llm_context_tokens=None, chunk_workers_arg=None):
    conn = postgres_connection()

    # ⭐ NEW (robustness): one-time startup sweep — rows wedged at "processing"
    # by a previously hard-killed instance are reset to pending so they are not
    # silently lost. Skip with AI_SUMMARY_NO_RESET_STALE=1 for multi-instance.
    if os.getenv("AI_SUMMARY_NO_RESET_STALE", "") not in ("1", "true", "True"):
        try:
            reset_stale_processing_rows()
        except Exception as sweep_exc:
            print(f"[WARN] Startup sweep error: {sweep_exc}")

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
        prog='ai_analysis_v3.1.1.17',
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
    parser.add_argument('--no_reset_stale', action='store_true',
                        help='Skip the startup sweep that resets rows wedged at "processing" by a '
                             'previously hard-killed instance. Set this when another pipeline '
                             'instance is already running against the same database.')
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
    if args.no_reset_stale:
        os.environ["AI_SUMMARY_NO_RESET_STALE"] = "1"
        print("  Startup sweep: DISABLED (--no_reset_stale) — stale in-progress rows left untouched")
    resolved_chunk_workers = _resolve_chunk_workers(args.chunk_workers)
    print(f"  Chunk workers: {resolved_chunk_workers}")
    print(f"  RAG topN: {os.getenv('AI_SUMMARY_RAG_TOPN', f'dynamic up to {DEFAULT_RAG_TOP_N}')} (legacy per-chunk path)")
    print("  Overall Analysis mode: field-wise source-record analysis from original ES records; sections analyzed independently and concatenated without a second synthesis pass")
    print("  Overall Analysis entity_type_analysis: ai_analysis_summary.entity_type_analysis (jsonb array) restricts the analysis to the selected entity types; NULL/empty = all fields")
    print("  new_prompt mode: entity_type_analysis = ['new_prompt'] analyzes ALL records matched by filter_json (no form_id aggregation) with the user prompt from ai_analysis_summary.prompt, using only the TYPE_MAPPING markdown (no field-wise Overall Analysis)")

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
