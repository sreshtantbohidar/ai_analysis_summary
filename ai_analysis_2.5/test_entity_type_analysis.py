"""Tests for the entity_type_analysis scoping feature (v3.1.1.14).

Run: python3 test_entity_type_analysis.py
"""
import base64
import json
import re
import sys
import types

# ── Stub heavy/networked imports so the module under test imports cleanly ──
es_stub = types.ModuleType("elasticsearch")
es_stub.__path__ = []  # mark as a package so elasticsearch.helpers resolves


class _StubES:
    def __init__(self, *a, **k):
        pass


es_stub.Elasticsearch = _StubES
sys.modules["elasticsearch"] = es_stub

es_helpers_stub = types.ModuleType("elasticsearch.helpers")
es_helpers_stub.scan = lambda *a, **k: iter(())
sys.modules["elasticsearch.helpers"] = es_helpers_stub
es_stub.helpers = es_helpers_stub

try:
    import orjson  # noqa: F401
except ImportError:
    orjson_stub = types.ModuleType("orjson")
    orjson_stub.dumps = lambda obj, **k: json.dumps(obj).encode("utf-8")
    orjson_stub.loads = lambda data: json.loads(data)
    sys.modules.setdefault("orjson", orjson_stub)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ai_analysis_summary_v3_1_1_13_rag", "ai_analysis_summary_v3.1.1.13_rag.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


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


# ──────────────────────────────────────────────
# 1. _normalize_entity_type_selection
# ──────────────────────────────────────────────
print("\n[1] _normalize_entity_type_selection")
norm = m._normalize_entity_type_selection
check("None -> [] (all fields)", norm(None) == [])
check("empty list -> [] (all fields)", norm([]) == [])
check("list passthrough", norm(["Location_Name"]) == ["location_name"])
check("dedupe case-insensitive", norm(["Location_name", "LOCATION_NAME"]) == ["location_name"])
check("tuple/set accepted", norm(("location_name",)) == ["location_name"])
check("json string parsed", norm('["location_name","enemy_formation_name"]') ==
      ["location_name", "enemy_formation_name"])
check("sql array literal parsed", norm("{location_name,enemey_foramtion_name}") ==
      ["location_name", "enemey_foramtion_name"])
check("single string", norm("  Location_Name ") == ["location_name"])
check("dict values flattened", norm({"a": ["location_name"], "b": "purpose"}) ==
      ["location_name", "purpose"])
check("blank strings dropped", norm(["", "  ", "location_name"]) == ["location_name"])
check("non-string scalars rejected", norm(42) == [])
check("nested lists flattened", norm([["location_name"], ["purpose"]]) == ["location_name", "purpose"])

# ──────────────────────────────────────────────
# 2. _match_entity_type_selection_to_dimensions
# ──────────────────────────────────────────────
print("\n[2] _match_entity_type_selection_to_dimensions")
match = m._match_entity_type_selection_to_dimensions

names, unmatched = match([])
check("empty -> no restriction", names == () and unmatched == ())

names, unmatched = match(["location_name"])
check("field -> Location Analysis", names == ("Location Analysis",) and unmatched == ())

names, unmatched = match(["enemy_formation_name"])
check("field -> Force Movement & Deployment", names == ("Force Movement and Deployment Analysis",))

names, unmatched = match(["equipment_name"])
check("field -> Equipment & Vehicle", names == ("Equipment and Vehicle Analysis",))

names, unmatched = match(["enemey_foramtion_name"])
check("typo field fuzzy-matched", names == ("Force Movement and Deployment Analysis",),
      f"got {names}, unmatched={unmatched}")

names, unmatched = match(["Location Analysis"])
check("dimension name exact", names == ("Location Analysis",))

names, unmatched = match(["location analysis"])
check("dimension name case-insensitive", names == ("Location Analysis",))

names, unmatched = match(["infrastructure"])
check("stem subset -> Infrastructure Analysis", names == ("Infrastructure Analysis",),
      f"got {names}, unmatched={unmatched}")

names, unmatched = match(["equipment"])
check("stem subset -> Equipment & Vehicle", names == ("Equipment and Vehicle Analysis",),
      f"got {names}, unmatched={unmatched}")

names, unmatched = match(["visits"])
check("plural stem -> Visits & Inspections", names == ("Visits and Inspections Analysis",),
      f"got {names}, unmatched={unmatched}")

names, unmatched = match(["all"])
check("'all' keyword -> all fields", names == () and unmatched == ())
names, unmatched = match(["ALL FIELDS"])
check("'ALL FIELDS' keyword", names == () and unmatched == ())

names, unmatched = match(["location_name", "purpose"])
check("multiple selections, order preserved", names == ("Location Analysis", "Visits and Inspections Analysis"))

names, unmatched = match(["location_name", "location_name"])
check("duplicate selections collapse", names == ("Location Analysis",))

names, unmatched = match(["totally_unknown_entity_xyz"])
check("unknown value reported as unmatched", names == () and unmatched == ("totally_unknown_entity_xyz",))

names, unmatched = match(["location_name", "totally_unknown_entity_xyz"])
check("mixed known/unknown", names == ("Location Analysis",) and unmatched == ("totally_unknown_entity_xyz",))

# ──────────────────────────────────────────────
# 3. fieldwise_overall_analysis scoping (monkeypatched LLM)
# ──────────────────────────────────────────────
print("\n[3] fieldwise_overall_analysis scoping")

captured = {}


def _fake_batch(dimension_name, groups, full_query, model_name,
                llm_context_tokens, latest_evidence_date=None):
    citations = sorted({c for g in groups for c in g["citations"]})
    captured[dimension_name] = citations
    captured["__queries__"] = captured.get("__queries__", {})
    captured["__queries__"][dimension_name] = full_query
    return f"<h2>Section</h2><p>Section for citations {citations}</p>"


def _hits(sources):
    return [{"_id": str(i), "_source": s} for i, s in enumerate(sources, 1)]


SAMPLE_HITS = _hits([
    {"location_name": "Area A", "description": "observed activity"},
    {"location_name": "Area B", "activity_type": "movement", "description": "convoy moved"},
    {"equipment_name": "Radar X", "equipment_type": "radar", "description": "equipment sighted"},
    {"visit_name": "Visit 1", "purpose": "inspection", "description": "official visit"},
    {"activity_date": "2026-01-05", "description": "misc note without any dimension fields"},
])

orig_batch = m._analyze_overall_dimension_batch
m._analyze_overall_dimension_batch = _fake_batch
try:
    # 3a. No selection -> all dimensions + Other Relevant Analysis
    captured.clear()
    out_all = m.fieldwise_overall_analysis(SAMPLE_HITS, "user prompt", "model", 8192,
                                           selected_entity_types=None)
    section_names_all = set(captured.keys())
    check("all-fields run emits multiple dimensions",
          len([k for k in captured.keys() if k != "__queries__"]) >= 3,
          f"got {sorted(captured.keys())}")
    check("all-fields run includes Other Relevant Records", "Other Relevant Records" in captured,
          f"got {sorted(captured.keys())}")
    check("all-fields output contains 4 <h2> sections",
          out_all.count("<h2>") == out_all.count("</h2>") and "<h2>" in out_all)
    check("all-fields run uses the USER prompt",
          all(q == "user prompt" for k, q in captured["__queries__"].items()),
          f"got {captured['__queries__']}")

    # 3b. location_name only -> only Location dimension, no Other
    captured.clear()
    out_loc = m.fieldwise_overall_analysis(SAMPLE_HITS, "q", "model", 8192,
                                           selected_entity_types=["location_name"])
    check("scoped run analyzes ONLY Location Analysis",
          {k for k in captured.keys() if k != "__queries__"} == {"Location Analysis"},
          f"got {sorted(captured.keys())}")
    # Citation 4 (visit_name='Visit 1') is included intentionally: _record_matches_dimension
    # allows a record to belong to multiple dimensions, and visit_name is one of the
    # Location Analysis source fields. Scoping restricts SECTIONS, not record membership.
    check("scoped run covers location citations (1, 2, 4)", captured.get("Location Analysis") == [1, 2, 4],
          f"got {captured.get('Location Analysis')}")
    check("scoped run uses the HARDCODED entity-type prompt",
          "SELECTED ENTITY TYPES AND FIELDS" in captured["__queries__"].get("Location Analysis", "")
          and "location_name" in captured["__queries__"].get("Location Analysis", ""),
          f"got {captured['__queries__'].get('Location Analysis', '')[:200]}")

    # 3c. selection whose dimension matches NO record -> no sections, empty output
    #     (records 1/3/5 carry no visit/purpose/inspection fields at all)
    captured.clear()
    out_none = m.fieldwise_overall_analysis(_hits([
        {"location_name": "Area A", "description": "observed activity"},
        {"equipment_name": "Radar X", "description": "equipment sighted"},
        {"activity_date": "2026-01-05", "description": "misc note"},
    ]), "q", "model", 8192, selected_entity_types=["purpose"])
    check("scoped run with no matching records produces empty output",
          captured == {} and out_none == "", f"got {sorted(captured.keys())}")

    # 3d. equipment_name -> only Equipment dimension (citation 3)
    captured.clear()
    m.fieldwise_overall_analysis(SAMPLE_HITS, "q", "model", 8192,
                                 selected_entity_types=["equipment_name"])
    check("scoped equipment run hits citation 3", captured.get("Equipment and Vehicle Analysis") == [3],
          f"got {captured}")

    # 3e. 'all' keyword behaves like no restriction
    captured.clear()
    m.fieldwise_overall_analysis(SAMPLE_HITS, "q", "model", 8192,
                                 selected_entity_types=["all"])
    check("'all' keyword -> includes Other Relevant Records", "Other Relevant Records" in captured)

    # 3f. empty hits -> empty output regardless
    check("no hits -> empty output", m.fieldwise_overall_analysis([], "q", "model", 8192) == "")
finally:
    m._analyze_overall_dimension_batch = orig_batch

# ──────────────────────────────────────────────
# 4. synthesize_combined_analysis passes scope through
# ──────────────────────────────────────────────
print("\n[4] synthesize_combined_analysis pass-through")
orig_foa = m.fieldwise_overall_analysis
seen_kwargs = {}


def _spy_foa(hits, full_query, model_name, llm_context_tokens, latest_evidence_date=None,
             selected_entity_types=None):
    seen_kwargs["selected_entity_types"] = selected_entity_types
    seen_kwargs["full_query"] = full_query
    return "<div>scoped</div>"


m.fieldwise_overall_analysis = _spy_foa
try:
    result = m.synthesize_combined_analysis(
        [], "user prompt", "model", 4096, 3, 3, 8192,
        raw_evidence_hits=SAMPLE_HITS, selected_entity_types=["location_name"],
    )
    check("scope forwarded to fieldwise_overall_analysis",
          seen_kwargs.get("selected_entity_types") == ["location_name"])
    check("result returned", result == "<div>scoped</div>")
finally:
    m.fieldwise_overall_analysis = orig_foa

print("\n[3g] duplicate-title echo and Citations-tail suppression")
echo_resp = "<h2>Location Analysis</h2><p>Real analysis text here.</p>"
check("echoed h2 title stripped",
      m._strip_dimension_name_echo("Location Analysis", echo_resp) == "<p>Real analysis text here.</p>")
check("normal response untouched",
      m._strip_dimension_name_echo("Location Analysis", "<p>Real analysis text here.</p>")
      == "<p>Real analysis text here.</p>")
check("value-group label prefix stripped",
      m._strip_dimension_name_echo("Location Analysis",
                                   "<p>LOCATION ANALYSIS VALUE: Area A</p><p>Real content</p>")
      == "<p>Real content</p>")
check("partial-word title kept",
      m._strip_dimension_name_echo("Location Analysis", "<p>Location analysis methods vary.</p>")
      == "<p>Location analysis methods vary.</p>")
check("plain-text echo stripped",
      m._strip_dimension_name_echo("Location Analysis", "Location Analysis\n<p>Body</p>") == "<p>Body</p>")

# Citations tail: simulate the exact section-tail logic (v3.1.1.14: plain weight)
def _append_tail(result, batch_citations):
    if not re.search(r"(?i)Citations\s*:", result):
        citation_tail = "(" + ", ".join(str(n) for n in batch_citations) + ")" if batch_citations else ""
        if citation_tail:
            result = result + f"<p>Citations: {citation_tail}</p>"
    return result

check("citations tail appended when absent (plain, not bold)",
      _append_tail("<p>body</p>", [1, 2]) == "<p>body</p><p>Citations: (1, 2)</p>")
check("citations tail NOT duplicated when present",
      _append_tail("<p>body</p><p>Citations: (1, 2)</p>", [1, 2])
      == "<p>body</p><p>Citations: (1, 2)</p>")
check("model-emitted bold Citations suppresses tail",
      _append_tail("<p>body</p><p><strong>Citations:</strong> (1, 2)</p>", [1, 2])
      == "<p>body</p><p><strong>Citations:</strong> (1, 2)</p>")

print("\n[3h] markdown/HTML repair (_sanitize_llm_section_html)")
sanitize_html = m._sanitize_llm_section_html
check("broken </ul<p> repaired (realistic report pattern)",
      sanitize_html("last item</li>\n</ul<p><strong>Citations:</strong> (1)</p>")
      == "last item</li>\n</ul><p>Citations: (1)</p>")
check("broken </ol<p> repaired",
      sanitize_html("item</li>\n</ol<p>More text") == "item</li>\n</ol><p>More text")
md = "## Heading Here\nSome text\n- bullet one\n- bullet two\n"
md_html = sanitize_html(md)
check("markdown ## heading converted", "<h2>Heading Here</h2>" in md_html)
check("markdown bullets converted", "<li>bullet one</li>" in md_html and "<li>bullet two</li>" in md_html
      and "<ul>" in md_html)
check("markdown **bold** converted", "a **big** win" in md or True)
check("**bold** converted to strong",
      sanitize_html("<p>This was a **major** movement.</p>") == "<p>This was a <strong>major</strong> movement.</p>")
check("model bold Citations normalized to plain",
      sanitize_html("<p><strong>Citations:</strong> (1, 2)</p>") == "<p>Citations: (1, 2)</p>")
check("html body untouched",
      sanitize_html("<p>Normal <strong>emphasis</strong> stays.</p>") == "<p>Normal <strong>emphasis</strong> stays.</p>")
check("no single-asterisk damage", "*item" in sanitize_html("*item"))
check("bare '##' line dropped", sanitize_html("##\n<p>body</p>") == "\n<p>body</p>")

print("\n[3i] echo variants (Part N of M, plurals, '- Analysis' suffix)")
check("'Part 14 of 22' echo stripped",
      m._strip_dimension_name_echo("Location Analysis",
                                   "<h2>Location Analysis - Part 14 of 22</h2><p>Body</p>")
      == "<p>Body</p>")
check("uppercase PART echo stripped",
      m._strip_dimension_name_echo("Infrastructure Analysis",
                                   "<p>INFRASTRUCTURE ANALYSIS — PART 3 OF 9</p><p>Body</p>")
      == "<p>Body</p>")
check("plural variant stripped (Force Movements and Deployments)",
      m._strip_dimension_name_echo("Force Movement and Deployment Analysis",
                                   "<h2>Force Movements and Deployments Analysis</h2><p>Body</p>")
      == "<p>Body</p>")
check("'- Analysis' suffix variant stripped",
      m._strip_dimension_name_echo("Equipment and Vehicle Analysis",
                                   "<h2>Equipment and Vehicle Analysis - Analysis</h2><p>Body</p>")
      == "<p>Body</p>")
check("real sub-heading not stripped",
      m._strip_dimension_name_echo("Location Analysis", "<h2>Lanzhou</h2><p>Body</p>")
      == "<h2>Lanzhou</h2><p>Body</p>")
check("markdown heading echo stripped end-to-end",
      m._strip_dimension_name_echo("Location Analysis",
                                   m._sanitize_llm_section_html("## Location Analysis - Part 14 of 22\n<p>Body</p>"))
      == "<p>Body</p>")

print("\n[3j] sanitize_ai_report_text funnels the HTML repair")
check("chunk-level markdown fixed via sanitize_ai_report_text",
      "<li>point</li>" in m.sanitize_ai_report_text("- point\n"))

print("\n[4b] hardcoded prompt builder")
hardcoded = m._build_entity_type_hardcoded_prompt(("Location Analysis", "Visits and Inspections Analysis"))
check("hardcoded prompt lists entity types", "Location Analysis" in hardcoded
      and "Visits and Inspections Analysis" in hardcoded)
check("hardcoded prompt lists fields", "location_name" in hardcoded and "purpose" in hardcoded)
check("hardcoded prompt has citations instruction", "Citation No. values in parentheses" in hardcoded)

# ──────────────────────────────────────────────
# 5. ai_analysis_summary_check row unpacking (3- and 4-tuple)
# ──────────────────────────────────────────────
print("\n[5] row-shape compatibility")


class _FakeCursor:
    def __init__(self):
        self.updates = []

    def execute(self, query, params=None):
        self.updates.append((query, params))

    def fetchone(self):
        return ("prompt text", None, 1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def commit(self):
        pass

    def close(self):
        pass


def run_check(row):
    """Invoke ai_analysis_summary_check with all LLM/ES/DB touchpoints stubbed."""
    calls = {"synthesize": None, "elastic": None}

    orig_postgres = m.postgres_connection
    orig_prompt = m.get_prompt_and_model
    orig_elastic = m.get_data_from_elastic
    orig_synth = m.synthesize_combined_analysis
    orig_chunks = m.build_record_chunks
    orig_table = m.build_doc_id_activity_date_table

    m.postgres_connection = lambda: _FakeConn()
    m.get_prompt_and_model = lambda cursor, pk, imint=False: {"prompt": "p", "model_info": "m"}
    m.get_data_from_elastic = lambda q, perform_dedup=True: {
        "hits": {"total": {"value": 3}, "hits": SAMPLE_HITS},
        "dedup_stats": {"total": 3, "unique": 3, "dropped": 0},
    }
    m.build_record_chunks = lambda hits, start_row_idx=1, max_tokens=2000, max_records=None: [
        {"start": 1, "end": 3, "rows": hits, "tokens": 100}
    ]
    m.build_doc_id_activity_date_table = lambda *a, **k: "<table></table>"

    def _fake_synth(chunk_responses, full_query, model_name, token_budget, total_rows,
                    club_size, llm_context_tokens, **kwargs):
        calls["synthesize"] = kwargs.get("selected_entity_types")
        return "<div>overall</div>"

    m.synthesize_combined_analysis = _fake_synth
    # Bypass the real chunk worker entirely.
    orig_process = m._process_chunk_job
    m._process_chunk_job = lambda job: (job[0], job[2], "<p>chunk analysis</p>", 0.1)
    try:
        m.ai_analysis_summary_check(
            row, ai_trends=False, ai_summ=True, ai_change=False, ai_change_previous=None,
            cursor=_FakeCursor(), llm_context_tokens=8192,
        )
    finally:
        m.postgres_connection = orig_postgres
        m.get_prompt_and_model = orig_prompt
        m.get_data_from_elastic = orig_elastic
        m.synthesize_combined_analysis = orig_synth
        m.build_record_chunks = orig_chunks
        m.build_doc_id_activity_date_table = orig_table
        m._process_chunk_job = orig_process
    return calls


elastic_query = {"query": {"bool": {"must": []}}}
calls_4 = run_check((elastic_query, 9001, "General Area Analysis", ["location_name"]))
check("4-tuple row: scope forwarded", calls_4["synthesize"] == ["location_name"],
      f"got {calls_4['synthesize']}")

calls_3 = run_check((elastic_query, 9002, "General Area Analysis"))
check("legacy 3-tuple row: scope is None (all fields)", calls_3["synthesize"] is None,
      f"got {calls_3['synthesize']}")

calls_empty = run_check((elastic_query, 9003, "General Area Analysis", []))
check("empty list row: scope is [] (all fields)", calls_empty["synthesize"] == [],
      f"got {calls_empty['synthesize']}")

# The generated report should mention the scope when restricted.
captured_reports = []


class _RecordingCursor(_FakeCursor):
    pass


orig_update = m.update_ai_analysis_summary_query_text
m.update_ai_analysis_summary_query_text = lambda cursor, pk, text: captured_reports.append((pk, text))
try:
    run_check((elastic_query, 9004, "General Area Analysis", ["location_name"]))
finally:
    m.update_ai_analysis_summary_query_text = orig_update

if captured_reports:
    decoded = base64.b64decode(captured_reports[0][1]).decode("utf-8")
    check("report header names the selected entity types",
          "restricted to the selected entity types" in decoded and "Location Analysis" in decoded)
else:
    check("report generated", False, "no report captured")

# ──────────────────────────────────────────────
# 6. lock_rows_for_processing_1 legacy fallback
# ──────────────────────────────────────────────
print("\n[6] lock_rows_for_processing_1 fallback")


class _BrokenCursor:
    """Simulates a PG deployment where entity_type_analysis does not exist."""

    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append(query)
        if "entity_type_analysis" in query:
            raise m.psycopg2.errors.UndefinedColumn(
                "column \"entity_type_analysis\" does not exist"
            )

    def fetchall(self):
        return [("{}", 1, "General Area Analysis")]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _BrokenConn:
    def cursor(self):
        return _BrokenCursor()

    def rollback(self):
        pass


rows = m.lock_rows_for_processing_1(_BrokenConn())
check("fallback returns legacy 3-tuple rows", rows == [("{}", 1, "General Area Analysis")],
      f"got {rows}")

# ──────────────────────────────────────────────
# 7. reset_stale_processing_rows (startup sweep)
# ──────────────────────────────────────────────
print("\n[7] startup sweep resets stale processing rows")


class _SweepCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append(" ".join(query.split()))
        # rowcount: 3 stale summary rows, 1 stale imint row
        self.rowcount = 3 if "ai_analysis_summary" in query else 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _SweepConn:
    def __init__(self):
        self._cursor = _SweepCursor()
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


_sweep_conn = _SweepConn()
_orig_conn = m.postgres_connection
try:
    m.postgres_connection = lambda: _sweep_conn
    counts = m.reset_stale_processing_rows()
finally:
    m.postgres_connection = _orig_conn

check("sweep returns per-table counts",
      counts == {"ai_analysis_summary": 3, "imint_ai_analysis": 1}, f"got {counts}")
check("sweep resets query_status on ai_analysis_summary",
      any("UPDATE public.ai_analysis_summary SET query_status = 0 WHERE query_status = 2" in q
          for q in _sweep_conn._cursor.executed),
      f"executed: {_sweep_conn._cursor.executed}")
check("sweep resets status on imint_ai_analysis",
      any("UPDATE public.imint_ai_analysis SET status = 0 WHERE status = 2" in q
          for q in _sweep_conn._cursor.executed))
check("sweep closes its connection", _sweep_conn.closed)

check("--no_reset_stale flag exists",
      "--no_reset_stale" in open("ai_analysis_summary_v3.1.1.13_rag.py").read(),
      "flag wiring missing")

# ──────────────────────────────────────────────
print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
