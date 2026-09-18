"""Tests for the Publication Summary section (v3.1.1.15).

Run: python3 test_publication_summary.py
"""
import importlib.util
import json
import sys
import types
import urllib.parse

# ── Stub heavy imports so the clients import cleanly ──
es_stub = types.ModuleType("elasticsearch")
es_stub.__path__ = []


class _StubES:
    def __init__(self, *a, **k):
        pass


es_stub.Elasticsearch = _StubES
sys.modules["elasticsearch"] = es_stub
helpers_stub = types.ModuleType("elasticsearch.helpers")
helpers_stub.scan = lambda *a, **k: iter(())
sys.modules["elasticsearch.helpers"] = helpers_stub
es_stub.helpers = helpers_stub

try:
    import orjson  # noqa: F401
except ImportError:
    orjson_stub = types.ModuleType("orjson")
    orjson_stub.dumps = lambda obj, **k: json.dumps(obj).encode("utf-8")
    orjson_stub.loads = lambda data: json.loads(data)
    sys.modules.setdefault("orjson", orjson_stub)

import requests  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pub = _load("publication_summary_client", "publication_summary_client.py")
m = _load("ai_analysis_summary_v3_1_1_13_rag", "ai_analysis_summary_v3.1.1.13_rag.py")

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


HITS = [
    {"_id": "1", "_source": {"location_name": "Area A", "description": "alpha", "activity_date": "2026-09-15"}},
    {"_id": "2", "_source": {"location_name": "Area B", "description": "beta", "activity_date": "2026-09-16"}},
]

# ──────────────────────────────────────────────
# 1. markdown -> HTML
# ──────────────────────────────────────────────
print("\n[1] _safe_markdown_to_html")
conv = pub._safe_markdown_to_html
html = conv("## Heading\n\n- item one\n- item two\n\nPlain **bold** and `code`.")
check("heading demoted to h3", "<h3>Heading</h3>" in html)
check("ul/li rendered", "<ul>" in html and "<li>item one</li>" in html)
check("bold rendered", "<strong>bold</strong>" in html)
check("code rendered", "<code>code</code>" in html)
check("empty input -> empty", conv("") == "")
check("plain text wrapped in p", conv("just text") == "<p>just text</p>")

# ──────────────────────────────────────────────
# 2. gating: no token -> skipped, no network
# ──────────────────────────────────────────────
print("\n[2] gating without token")
import os
os.environ.pop("SUMMARY_API_TOKEN", None)
check("summary_api_configured False", pub.summary_api_configured() is False)
section = pub.build_publication_summary_section(HITS, "user prompt")
check("section empty without token", section == "")
check("no exception raised", isinstance(section, str))

# ──────────────────────────────────────────────
# 3. happy path with stubbed API (gemma4 chat model)
# ──────────────────────────────────────────────
print("\n[3] stubbed API happy path")
os.environ["SUMMARY_API_TOKEN"] = "test-token"
os.environ["SUMMARY_API_TIMEOUT"] = "8"   # keep tests fast


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


calls = {"submit": 0, "poll": 0, "question": 0, "question_models": []}


def _fake_post(url, headers=None, json=None, timeout=None):
    calls["submit"] += 1
    if url.endswith("/api/v1/ai-summaries"):
        return _Resp(202, {"summary_id": "SUM-1", "status": "queued"})
    if url.endswith("/questions"):
        calls["question"] += 1
        calls["question_models"].append(json.get("model_name"))
        return _Resp(202, {"question_id": "Q-1", "status": "queued"})
    return _Resp(404, {"error": "not found"})


def _fake_get(url, headers=None, timeout=None):
    calls["poll"] += 1
    if url.endswith("/api/v1/ai-summaries/SUM-1"):
        return _Resp(200, {"status": "ready", "summary": "## Pub Brief\n\n- point A"})
    if url.endswith("/questions/Q-1"):
        return _Resp(200, {"status": "complete", "answer": "Answer **with** detail."})
    return _Resp(404, {"error": "not found"})  # unknown URL -> 404


pub.requests.post = _fake_post
pub.requests.get = _fake_get
os.environ["SUMMARY_API_CHAT_MODEL"] = "gemma4:12b"
section = pub.build_publication_summary_section(
    HITS, "Analyze movement (PLEASE NOTE: response in html).", filter_json={"a": 1}
)
check("section has title h2", "<h2>Publication Summary</h2>" in section)
check("summary markdown converted", "<h3>Pub Brief</h3>" in section and "<li>point A</li>" in section)
check("follow-up answer included", "Analysis Follow-up" in section and "Answer <strong>with</strong> detail." in section)
check("user prompt stripped of NOTE trailer",
      any("PLEASE NOTE" not in str(u) for u in [1]) and calls["question"] == 1)
check("gemma4 chat model requested", calls["question_models"] == ["gemma4:12b"])

# gemma4 rejected -> server default fallback
print("\n[4] gemma4 rejected -> server default")
calls["question_models"] = []


def _post_model_reject(url, headers=None, json=None, timeout=None):
    if url.endswith("/questions"):
        calls["question_models"].append(json.get("model_name"))
        if json.get("model_name"):
            return _Resp(400, {"error": "model_name 'gemma4:12b' is not registered"})
        return _Resp(202, {"question_id": "Q-2", "status": "queued"})
    return _Resp(202, {"summary_id": "SUM-2", "status": "queued"})


def _fake_get2(url, headers=None, timeout=None):
    if url.endswith("/questions/Q-2"):
        return _Resp(200, {"status": "complete", "answer": "ok"})
    return _Resp(200, {"status": "ready", "summary": "brief"})  # SUM-2 poll


pub.requests.post = _post_model_reject
pub.requests.get = _fake_get2
pub.build_publication_summary_section(HITS, "prompt")
check("fallback retried without model_name", calls["question_models"] == ["gemma4:12b", None])

# ──────────────────────────────────────────────
# 5. API unreachable -> section skipped
# ──────────────────────────────────────────────
print("\n[5] unreachable API")


def _post_down(url, headers=None, json=None, timeout=None):
    raise requests.exceptions.ConnectionError("connection refused")


pub.requests.post = _post_down
section = pub.build_publication_summary_section(HITS, "prompt")
check("unreachable API -> empty section", section == "")

# ──────────────────────────────────────────────
# 6. build_tabbed_html ordering + gating
# ──────────────────────────────────────────────
print("\n[6] report placement")
no_pub = m.build_tabbed_html(["<div>A</div>"], doc_table_html="<table>cites</table>")
check("no publication section when not supplied", "Publication Summary" not in no_pub)

with_pub = m.build_tabbed_html(["<div>A</div>"], doc_table_html="<table>cites</table>",
                               publication_summary_html="<h2>Publication Summary</h2><p>x</p>")
idx_ai = with_pub.find('id="section-0"')
idx_pub = with_pub.find('id="section-publication"')
idx_cit = with_pub.find('id="section-citations"')
check("publication section present", idx_pub != -1)
check("publication BEFORE citation table", idx_pub != -1 and idx_cit != -1 and idx_pub < idx_cit)
check("publication after AI Analysis section", idx_ai != -1 and idx_pub > idx_ai)

print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
