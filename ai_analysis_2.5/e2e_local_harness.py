"""Local e2e harness: run ai_analysis_summary_check_threadsafe directly on row 1753.

Bypasses lock_rows_for_processing_1 so remote pipeline instances sharing this DB
cannot steal the row. Uses real Elasticsearch data, the real Postgres prompt and
the real Ollama LLM. Writes the report to the test row (1753) like a normal run.

Usage: python3 -u e2e_local_harness.py
"""
import importlib.util
import os
import sys
import time

import psycopg2

spec = importlib.util.spec_from_file_location(
    "constants", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "constants.py"))
)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

_mod_spec = importlib.util.spec_from_file_location(
    "ai_analysis_summary_v3_1_1_13_rag", "ai_analysis_summary_v3.1.1.13_rag.py"
)
m = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(m)

TEST_ROW_ID = 1753


def connect(retries=8):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.connect(
                database=constants.PG_DB_NAME, host=constants.DJANGO_HOST,
                user=constants.PG_USER, password=constants.PG_PASSWORD,
                port=constants.PG_PORT, connect_timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[retry {attempt}/{retries}] PG connect failed: {type(exc).__name__}")
            time.sleep(5)
    raise last


def main():
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT filter_json, search_form_type, entity_type_analysis "
        "FROM public.ai_analysis_summary WHERE ai_analysis_summary_id = %s",
        (TEST_ROW_ID,),
    )
    fetched = cur.fetchone()
    conn.close()
    if not fetched:
        print(f"[FATAL] row {TEST_ROW_ID} not found")
        sys.exit(1)
    filter_json, search_form_type, entity_types = fetched
    print(f"[HARNESS] row={TEST_ROW_ID} form={search_form_type} scope={entity_types}")

    row = (filter_json, TEST_ROW_ID, search_form_type, entity_types, False)
    m.ai_analysis_summary_check_threadsafe(
        row,
        False,   # ai_trends
        True,    # ai_summ  (--ai_summary)
        False,   # ai_change
        None,    # ai_change_previous
        llm_context_tokens=8192,
    )
    print("[HARNESS] done")


if __name__ == "__main__":
    main()
