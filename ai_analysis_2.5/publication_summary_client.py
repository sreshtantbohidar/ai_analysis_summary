"""
Publication Summary section for the AI analysis report (⭐ v3.1.1.15).

When ai_analysis_summary.summary_type_publication is TRUE, the report gains a
"Publication Summary" section rendered BEFORE the Record Source Details
(citation) table. The section is produced by the isolated Publication/Summary
API documented at http://192.168.1.125:8998/apidoc:

    POST /api/v1/ai-summaries                          submit data, queue summary
    GET  /api/v1/ai-summaries/{summary_id}             poll until ready/failed
    POST /api/v1/ai-summaries/{summary_id}/questions   ask a chat question
    GET  /api/v1/ai-summaries/{summary_id}/questions/{question_id}

Auth:  Authorization: Bearer <SUMMARY_API_TOKEN>   (integration credential)

Behavior contract:
  - summary_type_publication is FALSE/NULL  -> section is skipped entirely.
  - summary_type_publication is TRUE        -> submit the record data and wait
    (bounded, default 240 s) for the summary; then ask one chat question built
    from the user's prompt and append its answer.
  - If the API is unreachable, the token is missing/invalid, submission fails,
    or the wait times out -> the section is skipped (empty string) and the
    pipeline continues. Report generation NEVER fails because of this section.

Environment:
  SUMMARY_API_TOKEN   integration bearer token (required for the section)
  SUMMARY_API_BASE    optional override of the API base URL
  SUMMARY_API_TIMEOUT overall bounded wait in seconds (default 240)
"""

import hashlib
import os
import re
import time
import uuid
from typing import Optional

import requests

DEFAULT_SUMMARY_API_BASE = "http://192.168.1.125:8998"
DEFAULT_SUMMARY_API_TIMEOUT = 240          # bounded wait for the whole section
POLL_INTERVAL_SECONDS = 5                  # poll cadence for summary/question
HTTP_TIMEOUT = 20                          # per-request timeout
RETRY_ATTEMPTS = 2

USER_AGENT = "ai-analysis-pipeline/1.0"

SUMMARY_SECTION_TITLE = "Publication Summary"

# ⭐ v3.1.1.15: preferred chat model for the follow-up question. gemma4 is used
# when available/registered; if the API rejects it the client retries once with
# the known-registered server model (8998 requires an exact registered name —
# omitting model_name is itself a 400 there).
PREFERRED_CHAT_MODEL = "gemma4:12b"
FALLBACK_REGISTERED_MODEL = os.getenv("SUMMARY_API_FALLBACK_MODEL", "llama3:8b-instruct-q8_0")


def _api_base() -> str:
    return os.getenv("SUMMARY_API_BASE", DEFAULT_SUMMARY_API_BASE).rstrip("/")


def _token() -> str:
    return (os.getenv("SUMMARY_API_TOKEN") or "").strip()


def summary_api_configured() -> bool:
    """True when a token is configured (cheap check used before any network I/O)."""
    return bool(_token())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def _safe_markdown_to_html(md: str) -> str:
    """Very small markdown-to-HTML conversion for summary/answer text.

    Handles headings, bullet/numbered lists, bold, italics and inline code.
    Anything else is rendered as plain paragraphs. Escaping is deliberately
    light: the summary API output is trusted operator-side content.
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


def _records_to_payload(e_hits: list, max_chars: int = 90000) -> list:
    """Convert ES hits into the array-of-objects 'data' payload for submission."""
    data = []
    total = 0
    for hit in e_hits:
        source = hit.get("_source") if isinstance(hit, dict) else None
        if not isinstance(source, dict):
            continue
        rec = {}
        for key in sorted(source.keys()):
            value = source.get(key)
            if value in (None, "", [], {}):
                continue
            rec[key] = value
        if not rec:
            continue
        data.append(rec)
        total += len(str(rec))
        if total >= max_chars:
            break
    return data


def _model_env_override() -> Optional[str]:
    return (os.getenv("SUMMARY_API_MODEL_NAME") or "").strip() or None


def submit_summary(e_hits: list, page_name: str, filter_json: Optional[dict],
                   model_name: Optional[str] = None,
                   target_pages: int = 1) -> Optional[str]:
    """POST the data and return summary_id, or None on failure."""
    url = f"{_api_base()}/api/v1/ai-summaries"
    data = _records_to_payload(e_hits)
    if not data:
        print("[WARN] Publication summary: no usable records to submit")
        return None
    payload = {
        "page_name": page_name,
        "filterjson": filter_json or {"note": "filter omitted"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S+0000"),
        "data": data,
        "target_pages": 1 if target_pages not in (1, 2) else target_pages,
    }
    model = model_name or _model_env_override()
    if model:
        payload["model_name"] = model
    idem = hashlib.sha256(
        (page_name + "|" + str(len(data)) + "|" + time.strftime("%Y%m%d%H%M%S")).encode()
    ).hexdigest()[:32]

    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = requests.post(url, headers=_headers(), json=payload, timeout=HTTP_TIMEOUT)
            if r.status_code in (200, 202):
                body = r.json()
                summary_id = body.get("summary_id") or body.get("summaryId")
                if summary_id:
                    print(f"[INFO] Publication summary submitted: summary_id={summary_id}")
                    return summary_id
                print(f"[WARN] Publication summary: submit response without summary_id: {body}")
                return None
            if r.status_code == 400 and "model" in r.text.lower() and payload.get("model_name"):
                # Registered model_name mismatch — retry with the known-registered
                # server model (the 8998 API requires an exact registered name).
                print(f"[WARN] Publication summary submit rejected model '{payload['model_name']}'; retrying with '{FALLBACK_REGISTERED_MODEL}'")
                payload["model_name"] = FALLBACK_REGISTERED_MODEL
                continue
            print(f"[WARN] Publication summary submit status {r.status_code}: {r.text[:200]}")
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            print(f"[WARN] Publication summary submit unreachable (attempt {attempt + 1}): {exc}")
            time.sleep(1)
        except Exception as exc:
            print(f"[WARN] Publication summary submit failed: {exc}")
            return None
    return None


def poll_summary(summary_id: str, deadline: float) -> Optional[str]:
    """Poll GET /ai-summaries/{id} until ready; return summary text or None."""
    url = f"{_api_base()}/api/v1/ai-summaries/{summary_id}"
    while time.time() < deadline:
        try:
            r = requests.get(url, headers=_headers(), timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                body = r.json()
                status = body.get("status")
                if status == "ready":
                    summary = body.get("summary") or ""
                    if summary.strip():
                        return summary
                    print("[WARN] Publication summary ready but empty")
                    return None
                if status == "failed":
                    print(f"[WARN] Publication summary failed: {body.get('error', '')[:200]}")
                    return None
                # still queued/running
            elif r.status_code == 401:
                print("[WARN] Publication summary: invalid token (401)")
                return None
            elif r.status_code == 404:
                print("[WARN] Publication summary: summary_id not found (404)")
                return None
            else:
                print(f"[WARN] Publication summary poll status {r.status_code}")
        except requests.exceptions.RequestException as exc:
            print(f"[WARN] Publication summary poll error: {exc}")
        time.sleep(POLL_INTERVAL_SECONDS)
    print("[WARN] Publication summary: bounded wait elapsed before the summary was ready")
    return None


def ask_question(summary_id: str, message: str, scope: str = "data",
                 deadline: Optional[float] = None) -> Optional[str]:
    """Ask one chat question and poll its answer. Returns the answer or None."""
    base = f"{_api_base()}/api/v1/ai-summaries/{summary_id}"
    request_id = f"ai-analysis-{uuid.uuid4().hex[:12]}"
    payload = {"message": message[:4000], "request_id": request_id, "scope": scope}
    if os.getenv("SUMMARY_API_CHAT_MODEL", PREFERRED_CHAT_MODEL):
        payload["model_name"] = os.getenv("SUMMARY_API_CHAT_MODEL", PREFERRED_CHAT_MODEL)
    question_id = None
    try:
        r = requests.post(f"{base}/questions", headers=_headers(), json=payload,
                          timeout=HTTP_TIMEOUT)
        if r.status_code in (200, 202):
            question_id = r.json().get("question_id")
        elif r.status_code == 400 and "model" in r.text.lower() and payload.get("model_name"):
            # gemma4 (or the configured chat model) is not registered — retry with
            # the known-registered server chat model.
            print(f"[WARN] Publication chat model '{payload['model_name']}' rejected; retrying with '{FALLBACK_REGISTERED_MODEL}'")
            payload["model_name"] = FALLBACK_REGISTERED_MODEL
            try:
                r = requests.post(f"{base}/questions", headers=_headers(), json=payload,
                                  timeout=HTTP_TIMEOUT)
                if r.status_code in (200, 202):
                    question_id = r.json().get("question_id")
            except requests.exceptions.RequestException as exc:
                print(f"[WARN] Publication summary question retry failed: {exc}")
        else:
            print(f"[WARN] Publication summary question submit status {r.status_code}: {r.text[:200]}")
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] Publication summary question submit failed: {exc}")
        return None
    if not question_id:
        return None

    poll_until = deadline if deadline is not None else time.time() + 120
    while time.time() < poll_until:
        try:
            r = requests.get(f"{base}/questions/{question_id}", headers=_headers(),
                             timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                body = r.json()
                status = body.get("status")
                if status == "complete":
                    answer = body.get("answer") or ""
                    return answer.strip() or None
                if status in ("failed", "cancelled"):
                    print(f"[WARN] Publication summary question {status}: {body.get('error', '')[:200]}")
                    return None
            elif r.status_code == 404:
                # Per the API doc a cleared question returns 404 — treat as terminal.
                print("[WARN] Publication summary question not found (404)")
                return None
        except requests.exceptions.RequestException as exc:
            print(f"[WARN] Publication summary question poll error: {exc}")
        time.sleep(POLL_INTERVAL_SECONDS)
    print("[WARN] Publication summary question: bounded wait elapsed")
    return None


def build_publication_summary_section(e_hits: list, user_prompt: str,
                                      filter_json: Optional[dict] = None,
                                      model_name: Optional[str] = None,
                                      timeout_seconds: Optional[int] = None,
                                      target_pages: int = 1,
                                      page_name: Optional[str] = None) -> str:
    """Build the full 'Publication Summary' HTML section, or '' when skipped.

    Contract (per v3.1.1.15): called only when summary_type_publication is TRUE
    by the caller. Any failure here yields '' so the report is unaffected.
    """
    if not summary_api_configured():
        print("[INFO] Publication summary: SUMMARY_API_TOKEN not configured; section skipped")
        return ""
    total_timeout = int(timeout_seconds or os.getenv("SUMMARY_API_TIMEOUT", DEFAULT_SUMMARY_API_TIMEOUT))
    deadline = time.time() + total_timeout

    try:
        summary_id = submit_summary(
            e_hits,
            page_name=page_name or "AI Analysis Report",
            filter_json=filter_json,
            model_name=model_name,
            target_pages=target_pages,
        )
        if not summary_id:
            return ""
        summary_md = poll_summary(summary_id, deadline)
        if not summary_md:
            return ""

        parts = [f"<h2>{SUMMARY_SECTION_TITLE}</h2>"]
        summary_html = _safe_markdown_to_html(summary_md)
        parts.append(f"<div class=\"publication-summary\">{summary_html}</div>")

        question = (user_prompt or "").strip()
        if question:
            question = re.sub(r"\s*\(PLEASE NOTE.*$", "", question, flags=re.S | re.I).strip()
        if question:
            answer = ask_question(summary_id, f"{question}\n\nAnswer in HTML.", scope="data",
                                  deadline=deadline)
            if answer:
                answer_html = _safe_markdown_to_html(answer)
                if "<" not in answer_html:
                    answer_html = f"<p>{answer}</p>"
                parts.append(
                    "<h3>Analysis Follow-up</h3>"
                    f"<div class=\"publication-followup\">{answer_html}</div>"
                )
        print(f"[INFO] Publication summary section built ({sum(len(p) for p in parts)} chars)")
        return "\n".join(parts)
    except Exception as exc:
        print(f"[WARN] Publication summary section skipped due to error: {exc}")
        return ""


# Module-level exports guard for naive `from module import *` consumers.
__all__ = [
    "build_publication_summary_section",
    "summary_api_configured",
    "SUMMARY_SECTION_TITLE",
]
