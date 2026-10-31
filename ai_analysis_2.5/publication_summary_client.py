"""
Publication Summary section for the AI analysis report (⭐ v3.1.1.16/17).

When ai_analysis_summary.summary_type_publication is TRUE, the report gains a
"Publication Summary" section rendered BEFORE the Record Source Details
(citation) table.

⭐ v3.1.1.17: SELF-CONTAINED — the Publication Assistant v4.42 All-Publications
Q&A API client (formerly publication_api_client.py) is merged into this module,
so the whole publication integration lives in ONE file. API endpoints
(documented at http://127.0.0.1:8998/apidoc):

    POST /api/v1/publication-qa/questions            submit one question
    GET  /api/v1/publication-qa/questions/{id}       poll until complete/failed

Auth:  Authorization: Bearer <publication_token>   (constants.py, parent dir).
The token is read from constants.publication_token, falling back to the
SUMMARY_API_TOKEN environment variable.

The API searches all publications server-side, so the record data (e_hits,
filter_json, page_name) is NO longer submitted — the question alone (built
from the user's prompt) drives the answer.

Behavior contract (unchanged from v3.1.1.15):
  - summary_type_publication is FALSE/NULL  -> section is skipped entirely.
  - summary_type_publication is TRUE        -> ask one question and render the
    answer (bounded, default 240 s).
  - If the API is unreachable, the token is missing/invalid, submission fails,
    or the wait times out -> the section is skipped (empty string) and the
    pipeline continues. Report generation NEVER fails because of this section.

⭐ v3.1.1.26 — PUBLICATION READINESS:
The API snapshots every Ready publication at submission. A publication that is
still extracting/embedding therefore yields the sentinel answer "No publication
was Ready when this API question was submitted." — previously rendered into the
report as if it were a real summary (seen in production row 1918, where the
question was asked 80 s before the PDF became Ready). The client now:
  1. waits (SUMMARY_API_READY_WAIT, default 900 s) and re-submits while the
     snapshot scope is empty, so a publication that finishes ingesting during
     the wait is picked up;
  2. never renders a no-data sentinel — it logs a WARN and skips the section
     instead. Set SUMMARY_API_READY_WAIT=0 to restore the single-attempt
     behaviour;
  3. retries ONCE with a retrieval-friendly question when the API answers
     "No relevant passages…". The publication retrieval treats a capitalised
     leading word as a hard named-entity anchor that must literally occur in
     the passage, so an instruction-style prompt such as "List all army bases
     details Please show result in English language." (“List” and “English”)
     retrieves nothing while the same request phrased naturally works. The
     retry drops the instruction verb and the language clause
     (“army bases details”).

Token / base-URL resolution (first hit wins):
  1. explicit token=/base_url= arguments (the main script passes the values
     it loads from constants.py),
  2. constants.py (parent dir):  publication_token / publication_base_url
  3. environment:                SUMMARY_API_TOKEN / SUMMARY_API_BASE
  4. default base URL:           http://127.0.0.1:8998
"""

import html as _html
import importlib.util
import os
import re
import time
import uuid
from typing import Optional

import requests

DEFAULT_PUBLICATION_QA_BASE_URL = "http://127.0.0.1:8998"
DEFAULT_SUMMARY_API_TIMEOUT = 240          # bounded wait for the whole section
POLL_INTERVAL_SECONDS = 2                  # poll cadence for the answer
HTTP_TIMEOUT = 30                          # per-request timeout

# ⭐ v3.1.1.26: how long to keep re-submitting while no publication is Ready yet
# (a freshly uploaded PDF can take a couple of minutes to embed).
DEFAULT_READY_WAIT_SECONDS = 900
DEFAULT_READY_POLL_SECONDS = 30

# Answers the publication-qa API returns when it could not search anything.
# They are NOT report content and must never be rendered as a summary.
_NO_DATA_ANSWER_RES = (
    re.compile(r"(?i)\bno publication was ready\b"),
    re.compile(r"(?i)\bno relevant passages were found\b"),
    re.compile(r"(?i)\bno (?:ready )?publications? (?:are|were) (?:ready|available)\b"),
)

SUMMARY_SECTION_TITLE = "Publication Summary"

# ⭐ v3.1.1.16: optional server model override. By default the publication-qa
# API picks its own registered model (the endpoint rejects unknown names), so
# nothing is sent unless the operator sets SUMMARY_API_MODEL_NAME.
USER_AGENT = "ai-analysis-pipeline/1.0"


# ---------------------------------------------------------------------------
# constants.py (parent dir) — primary source for publication_token and
# publication_base_url. Loaded with the same importlib pattern the main script
# uses; any failure falls back to the environment variables.
# ---------------------------------------------------------------------------
def _load_constants_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "constants.py"))
    try:
        spec = importlib.util.spec_from_file_location("publication_summary_constants", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"[WARN] Publication summary: could not load constants.py ({exc}); using env fallbacks")
        return None


_constants_module = _load_constants_module()


def _api_base() -> str:
    base = str(getattr(_constants_module, "publication_base_url", "") or "").strip()
    if not base:
        base = os.getenv("SUMMARY_API_BASE", DEFAULT_PUBLICATION_QA_BASE_URL)
    return base.rstrip("/")


def _token() -> str:
    token = str(getattr(_constants_module, "publication_token", "") or "").strip()
    if not token:
        token = (os.getenv("SUMMARY_API_TOKEN") or "").strip()
    return token


def _model_env_override() -> Optional[str]:
    return (os.getenv("SUMMARY_API_MODEL_NAME") or "").strip() or None


def _env_seconds(name: str, default: int, minimum: int = 0) -> int:
    """Read an int-seconds env override; fall back to *default* when unset/invalid."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(float(raw)))
    except (TypeError, ValueError):
        print(f"[WARN] Publication summary: ignoring invalid {name}={raw!r}")
        return default


def _answer_is_no_data(answer: str) -> bool:
    """True when *answer* is the API's 'nothing to search / nothing found' sentinel."""
    text = (answer or "").strip()
    return bool(text) and any(regex.search(text) for regex in _NO_DATA_ANSWER_RES)


def _scope_publication_count(result: dict) -> Optional[int]:
    """Number of Ready publications the submission was scoped to (None when unknown)."""
    if not isinstance(result, dict):
        return None
    value = result.get("scope_publications")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ⭐ v3.1.1.26: instruction-style AI summary prompts retrieve nothing because the
# publication engine turns a capitalised leading word into a hard named-entity
# anchor ("List …" must literally appear in the passage) and does the same for a
# trailing language instruction ("… in English language"). Removing the
# instruction verb and the language clause restores the natural-language query.
_INSTRUCTION_VERBS = (
    "list", "show", "give", "display", "provide", "enumerate", "describe",
    "summarize", "summarise", "explain", "generate", "produce", "prepare",
    "find", "search", "tell", "fetch", "extract", "analyze", "analyse",
    "identify", "highlight", "outline", "report", "compile", "collect",
    "return", "write", "draft", "bring", "print", "present", "create",
)
_LANGUAGE_CLAUSE_RE = re.compile(
    r"(?i)\b(?:please\s+)?(?:show|provide|give|return|write|answer|respond|output)"
    r"(?:\s+(?:the\s+)?(?:result|results|answer|response|output|reply|details))*"
    r"\s+in\s+[A-Za-z ]{0,20}?language\b[.\s]*"
)
_BARE_LANGUAGE_CLAUSE_RE = re.compile(
    r"(?i)\b(?:please\s+)?in\s+[A-Za-z ]{0,20}?\s*language\b[.\s]*"
)
_LEADING_INSTRUCTION_RE = re.compile(
    r"^\s*(?:" + "|".join(_INSTRUCTION_VERBS) + r")\b[\s,:.\-]*"
    r"(?:me\s+)?(?:the\s+)?(?:all\s+)?(?:of\s+)?",
    re.I,
)


def _retrieval_friendly_question(text: str) -> str:
    """Strip instruction wording so the publication retriever gets a real query.

    "List all army bases details Please show result in English language." ->
    "army bases details". Returns '' when nothing meaningful is left.

    ⭐ v3.1.1.27: the publication retriever also treats ANY capitalised leading
    word as a hard named-entity anchor that must literally occur in a passage,
    so a prompt such as "Impact of self reliance on defense manufacturing
    economy" retrieves nothing while "impact …"/"self reliance …" both work.
    The instruction verbs above cannot cover such nouns, so the leading word is
    lowercased as a final step.
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _LANGUAGE_CLAUSE_RE.sub(" ", cleaned)
    cleaned = _BARE_LANGUAGE_CLAUSE_RE.sub(" ", cleaned)
    cleaned = _LEADING_INSTRUCTION_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")
    if cleaned and cleaned[0].isupper():
        cleaned = cleaned[0].lower() + cleaned[1:]
    return cleaned


# ---------------------------------------------------------------------------
# Publication Assistant v4.42 All-Publications Q&A API client
# (merged from publication_api_client.py — ⭐ v3.1.1.17)
# ---------------------------------------------------------------------------
def _submit_question(base_url, question, model_name=None, timeout=None, token=None):
    """Authenticate with the Bearer token and submit one question.

    Returns:
        question_id (str)
    """
    url = f"{base_url.rstrip('/')}/api/v1/publication-qa/questions"

    payload = {"question": question}
    if model_name:
        payload["model_name"] = model_name

    token = token if token is not None else _token()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        # Makes a network retry safer by preventing accidental duplicate jobs.
        "Idempotency-Key": str(uuid.uuid4()),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(url, json=payload, headers=headers, timeout=timeout or HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    question_id = data.get("question_id")
    if not question_id:
        raise RuntimeError(f"API did not return question_id: {data}")

    return question_id


def _wait_for_answer(base_url, question_id,
                     poll_interval=None, max_wait=600, timeout=None, token=None):
    """Poll the API until the answer is complete or failed.

    Returns:
        the complete response dict (``answer`` plus the ``evidence`` list).
    """
    url = (
        f"{base_url.rstrip('/')}"
        f"/api/v1/publication-qa/questions/{question_id}"
    )

    token = token if token is not None else _token()

    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    started = time.monotonic()

    while True:
        response = requests.get(url, headers=headers, timeout=timeout or HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        status = str(data.get("status", "")).lower()

        if status == "complete":
            return data

        if status == "failed":
            raise RuntimeError(
                data.get("error")
                or "Publication Assistant API question failed."
            )

        if time.monotonic() - started >= max_wait:
            raise TimeoutError(
                f"Question {question_id} did not finish "
                f"within {max_wait} seconds."
            )

        time.sleep(poll_interval or POLL_INTERVAL_SECONDS)


def _ask_question(base_url, question, model_name=None,
                  poll_interval=None, max_wait=600, token=None):
    """Submit the question, wait for processing, and return the response dict.

    The response carries ``answer`` and the ``evidence`` (citation / source)
    list. Authentication uses constants.publication_token (Bearer).

    Example:
        result = _ask_question(
            "http://127.0.0.1:8998",
            "Summarize the major developments."
        )
    """
    token = token if token is not None else _token()

    question_id = _submit_question(
        base_url=base_url,
        question=question,
        model_name=model_name,
        token=token,
    )

    return _wait_for_answer(
        base_url=base_url,
        question_id=question_id,
        poll_interval=poll_interval,
        max_wait=max_wait,
        token=token,
    )


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------
def _safe_markdown_to_html(md: str) -> str:
    """Very small markdown-to-HTML conversion for the answer text.

    Handles headings, bullet/numbered lists, bold, italics and inline code.
    Anything else is rendered as plain paragraphs. Escaping is deliberately
    light: the publication QA API output is trusted operator-side content.
    """
    if not md:
        return ""

    def _inline(text: str) -> str:
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return text

    lines = md.replace("\r\n", "\n").split("\n")
    out = []
    list_open = None  # 'ul' | 'ol' | None
    para = []

    def _close_list():
        nonlocal list_open
        if list_open:
            out.append(f"</{list_open}>")
            list_open = None

    def _close_para():
        nonlocal para
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _close_para()
            _close_list()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            _close_para()
            _close_list()
            level = min(len(m.group(1)) + 1, 6)  # demote: h1 -> h2 etc.
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue
        m = re.match(r"^[-*•]\s+(.*)$", stripped)
        if m:
            _close_para()
            if list_open != "ul":
                _close_list()
                out.append("<ul>")
                list_open = "ul"
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        m = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m:
            _close_para()
            if list_open != "ol":
                _close_list()
                out.append("<ol>")
                list_open = "ol"
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        _close_list()
        para.append(stripped)
    _close_para()
    _close_list()
    return "\n".join(out)


def _absolute_url(base_url: str, path: Optional[str]) -> str:
    """Return an absolute URL for a (possibly relative) API asset path."""
    if not path:
        return ""
    path = str(path)
    if path.startswith(("http://", "https://")):
        return path
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _source_meta_html(source: dict) -> str:
    """One-line evidence metadata: block type, section, retrieval, dates."""
    parts = []
    block_type = str(source.get("block_type") or "").strip()
    if block_type:
        parts.append(f"{block_type} block")
    section = str(source.get("section") or "").strip()
    if section:
        parts.append(f"section: {section}")
    retrieval = source.get("retrieval")
    if isinstance(retrieval, (list, tuple)) and retrieval:
        parts.append("retrieval: " + ", ".join(str(item) for item in retrieval))
    dates = source.get("dates")
    if isinstance(dates, (list, tuple)) and dates:
        parts.append("dates: " + ", ".join(str(item) for item in dates))
    source_pages = source.get("source_pages")
    if isinstance(source_pages, (list, tuple)) and len(source_pages) > 1:
        parts.append("source pages: " + ", ".join(str(item) for item in source_pages))
    if not parts:
        return ""
    meta_text = _html.escape(" \u00b7 ".join(parts))
    return f'<div class="publication-source-meta">{meta_text}</div>'


def _render_source_image(source: dict, base_url: str) -> str:
    """Render a retrieved source image (``asset_url``) as a figure, or ''."""
    asset = _absolute_url(base_url, source.get("asset_url"))
    if not asset or (source.get("block_type") != "image" and not source.get("inline_media")):
        return ""
    caption = str(source.get("asset_caption") or "").strip()
    alt = caption or f"Source image from {source.get('doc_name') or 'publication'}"
    caption_html = f"<figcaption>{_html.escape(caption)}</figcaption>" if caption else ""
    return (
        '<figure class="publication-source-image">'
        f'<a href="{_html.escape(asset)}" target="_blank" rel="noopener">'
        f'<img src="{_html.escape(asset)}" alt="{_html.escape(alt)}" loading="lazy">'
        "</a>"
        f"{caption_html}</figure>"
    )


def _render_source_table(source: dict) -> str:
    """Render the retrieved table rows (``table_data``) as an HTML table, or ''."""
    table_data = source.get("table_data")
    if not isinstance(table_data, dict):
        return ""
    headers = [str(header) for header in (table_data.get("headers") or [])]
    table_rows = [row for row in (table_data.get("rows") or []) if isinstance(row, (list, tuple))]
    if not headers or not table_rows:
        return ""
    label_parts = [
        f"Table {table_data['table_index']}" if table_data.get("table_index") else "Source table"
    ]
    if source.get("page"):
        label_parts.append(f"page {source['page']}")
    label = " \u00b7 ".join(label_parts)
    head_html = "".join(f"<th>{_html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(
            f"<td>{_html.escape(str(cell if cell is not None else ''))}</td>" for cell in row
        ) + "</tr>"
        for row in table_rows
    )
    return (
        '<div class="publication-source-table">'
        f'<div class="publication-source-table-label">{_html.escape(label)}</div>'
        '<div class="table-scroll"><table>'
        f"<thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody>"
        "</table></div></div>"
    )


def _render_publication_sources(evidence, base_url: str, max_items: Optional[int] = None) -> str:
    """Render the API ``evidence`` list as a numbered 'Publication Sources' block.

    Every returned source is rendered in full — document, page range, section,
    block type, retrieval channels, dates, the retrieved text excerpt, the
    source image (``asset_url``) and the source table (``table_data``) — plus
    Preview / Download links when the API exposes them. By default nothing is
    truncated; pass ``max_items`` to cap the list. Returns '' when there is no
    usable evidence.
    """
    items = [item for item in (evidence or []) if isinstance(item, dict)]
    if not items:
        return ""
    shown = items if not max_items else items[:max_items]
    rows = []
    # Numbered with <ol> rather than typed digits: the report's citation
    # sanitizer strips a bare digit that follows a tag, so "1." text would be
    # mangled when citations are disabled. The list order matches the API's
    # 1-based citation_no.
    for source in shown:
        name = _html.escape(str(source.get("doc_name") or "Publication"))
        page = source.get("page")
        page_end = source.get("page_end")
        if page and page_end and page_end != page:
            where = f"pages {page}\u2013{page_end}"
        elif page:
            where = f"page {page}"
        else:
            where = ""
        where_html = f' <span class="muted">\u00b7 {_html.escape(where)}</span>' if where else ""
        attributes = []
        if source.get("doc_id"):
            attributes.append(f'data-doc-id="{_html.escape(str(source["doc_id"]))}"')
        if source.get("chunk_id") is not None:
            attributes.append(f'data-chunk-id="{_html.escape(str(source["chunk_id"]))}"')
        attributes_html = (" " + " ".join(attributes)) if attributes else ""
        excerpt = str(source.get("excerpt") or "").strip()
        excerpt_html = f"<blockquote>{_html.escape(excerpt)}</blockquote>" if excerpt else ""
        image_html = _render_source_image(source, base_url)
        table_html = _render_source_table(source)
        links = []
        preview = _absolute_url(base_url, source.get("preview_url"))
        if preview:
            links.append(f'<a href="{_html.escape(preview)}" target="_blank" rel="noopener">Preview</a>')
        download = _absolute_url(base_url, source.get("download_url"))
        if download:
            links.append(f'<a href="{_html.escape(download)}">Download</a>')
        links_html = (
            f'<div class="publication-source-links">{" &middot; ".join(links)}</div>' if links else ""
        )
        rows.append(
            f'<li class="publication-source"{attributes_html}>'
            f'<div class="publication-source-head"><strong>{name}</strong>{where_html}</div>'
            f"{_source_meta_html(source)}"
            f"{image_html}{table_html}{excerpt_html}{links_html}"
            "</li>"
        )
    more = (
        f'<p class="muted">Showing {max_items} of {len(items)} sources.</p>'
        if max_items and len(items) > max_items else ""
    )
    return (
        f"<h3>Publication Sources <span class=\"muted\">({len(items)} returned)</span></h3>"
        f'<ol class="publication-sources">{"".join(rows)}</ol>{more}'
    )


def build_publication_summary_section(e_hits: list, user_prompt: str,
                                      filter_json: Optional[dict] = None,
                                      model_name: Optional[str] = None,
                                      timeout_seconds: Optional[int] = None,
                                      target_pages: int = 1,
                                      page_name: Optional[str] = None,
                                      base_url: Optional[str] = None) -> str:
    """Build the full 'Publication Summary' HTML section, or '' when skipped.

    ⭐ v3.1.1.17: implemented on top of the merged publication-qa client
    (_ask_question) — submit the question, poll until complete, render the
    answer. The record payload (e_hits / filter_json / page_name /
    target_pages) is accepted for call-site compatibility but no longer
    submitted: the v4.42 publication-qa API searches all publications
    server-side. model_name is likewise unused unless SUMMARY_API_MODEL_NAME
    is set.

    Contract (per v3.1.1.15): called only when summary_type_publication is TRUE
    by the caller. Any failure here yields '' so the report is unaffected.

    The request is authenticated with constants.publication_token as
    ``Authorization: Bearer <token>``. The answer is followed by a "Publication
    Sources" list rendered from the API's ``evidence`` field so the report
    shows its citations/sources in full — text excerpt, source image
    (``asset_url``) and source table (``table_data``) for every returned item.
    """
    question = (user_prompt or "").strip()
    if question:
        question = re.sub(r"\s*\(PLEASE NOTE.*$", "", question, flags=re.S | re.I).strip()
    if not question:
        print("[INFO] Publication summary: empty prompt; section skipped")
        return ""

    resolved_base = (base_url or _api_base()).rstrip("/")
    token = _token()
    print(f"[INFO] Publication summary: asking {resolved_base} (token={'set' if token else 'missing'})")
    if not token:
        print("[WARN] Publication summary: no publication_token in constants.py / SUMMARY_API_TOKEN; sending unauthenticated request")
    max_wait = int(timeout_seconds or os.getenv("SUMMARY_API_TIMEOUT", DEFAULT_SUMMARY_API_TIMEOUT))

    # ⭐ v3.1.1.26: a newly uploaded publication is still embedding when a row is
    # processed, so the submission-time scope can be empty. Keep re-submitting
    # (each submission snapshots the then-Ready publications) until something is
    # searchable or the readiness budget runs out.
    ready_wait = _env_seconds("SUMMARY_API_READY_WAIT", DEFAULT_READY_WAIT_SECONDS)
    ready_poll = _env_seconds("SUMMARY_API_READY_POLL", DEFAULT_READY_POLL_SECONDS, minimum=1)
    deadline = time.monotonic() + ready_wait
    attempt = 0
    result = None
    answer = ""
    # ⭐ v3.1.1.26: fallback prompt used once when the API finds no passages for
    # an instruction-style prompt (see _retrieval_friendly_question).
    fallback_question = _retrieval_friendly_question(question)
    if fallback_question == question:
        fallback_question = ""
    retry_done = False
    question_used = question

    while True:
        attempt += 1
        try:
            result = _ask_question(
                base_url=resolved_base,
                question=question_used,
                model_name=_model_env_override(),
                max_wait=max_wait,
                token=token,
            )
        except Exception as exc:
            print(f"[WARN] Publication summary section skipped due to error: {exc}")
            return ""

        if not isinstance(result, dict):
            result = {"answer": result}
        answer = str(result.get("answer") or "").strip()
        if answer and not _answer_is_no_data(answer):
            break

        scope_count = _scope_publication_count(result)
        # An empty submission scope (or the API's 'nothing was Ready' answer on
        # an older API that omits scope_publications) is transient: wait for
        # ingestion, then ask again so the new publication is included.
        waiting_for_ingestion = scope_count == 0 or (
            scope_count is None and "no publication was ready" in answer.lower()
        )
        if waiting_for_ingestion:
            remaining = deadline - time.monotonic()
            if remaining >= ready_poll:
                print(
                    f"[INFO] Publication summary: no Ready publication at submission "
                    f"(attempt {attempt}); waiting {ready_poll}s for ingestion "
                    f"({int(remaining)}s of readiness budget left)"
                )
                time.sleep(ready_poll)
                continue
            print(
                f"[WARN] Publication summary section skipped: no publication became "
                f"Ready within {ready_wait}s (after {attempt} attempt(s))"
            )
            return ""

        # Publications were searchable but nothing matched. Instruction-style
        # prompts ("List …", "… in English language") are turned into hard
        # anchors by the retriever, so retry once with the instruction words
        # removed before giving up on the section.
        if fallback_question and not retry_done:
            retry_done = True
            question_used = fallback_question
            print(
                "[INFO] Publication summary: no passages for the raw prompt; "
                f"retrying with {fallback_question!r}"
            )
            continue

        print(
            "[WARN] Publication summary section skipped: publication API returned "
            f"no usable answer: {answer or 'empty answer'}"
        )
        return ""

    scope_count = _scope_publication_count(result)
    if scope_count is not None:
        print(f"[INFO] Publication summary: {scope_count} Ready publication(s) in scope")

    answer_html = _safe_markdown_to_html(answer)
    if "<" not in answer_html:
        answer_html = f"<p>{_html.escape(answer)}</p>"

    sources_html = _render_publication_sources(result.get("evidence"), resolved_base)

    # ⭐ v3.1.1.27: no <h2> here — the report's build_tabbed_html already wraps
    # this section in a "Publication Summary" heading, and emitting a second
    # one printed the title twice in every generated report.
    section = f"<div class=\"publication-summary\">{answer_html}{sources_html}</div>"
    evidence_count = len(result.get("evidence") or []) if isinstance(result.get("evidence"), list) else 0
    print(
        f"[INFO] Publication summary section built ({len(section)} chars, "
        f"{evidence_count} source(s), attempt {attempt}, "
        f"query={'retry' if retry_done else 'raw'})"
    )
    return section


# Module-level exports guard for naive `from module import *` consumers.
__all__ = [
    "build_publication_summary_section",
    "SUMMARY_SECTION_TITLE",
]
