"""Set up a scoped e2e test row to verify title/citation fixes (v3.1.1.14).

Usage:
  python3 e2e_setup_fixverify.py setup   -> inserts test row, parks other pending rows
  python3 e2e_setup_fixverify.py restore <row_id> <comma-separated parked ids>
  python3 e2e_setup_fixverify.py status  -> prints row states
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


def connect(retries=5):
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
            print(f"[retry {attempt}/{retries}] connect failed: {exc}")
            time.sleep(4)
    raise last


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    conn = connect()
    cur = conn.cursor()

    if action == "status":
        cur.execute(
            "SELECT ai_analysis_summary_id, query_status, entity_type_analysis "
            "FROM public.ai_analysis_summary ORDER BY ai_analysis_summary_id DESC LIMIT 8"
        )
        for r in cur.fetchall():
            print(r)

    elif action == "setup":
        cur.execute(
            "SELECT ai_analysis_summary_id FROM public.ai_analysis_summary "
            "WHERE filter_json IS NOT NULL ORDER BY ai_analysis_summary_id DESC LIMIT 1"
        )
        src = cur.fetchone()
        if not src:
            print("no source row with filter_json found")
            sys.exit(1)
        src_id = src[0]
        cur.execute("SELECT ai_analysis_summary_id FROM public.ai_analysis_summary WHERE query_status=0")
        parked = [r[0] for r in cur.fetchall()]
        if parked:
            cur.execute(
                "UPDATE public.ai_analysis_summary SET query_status=1 "
                "WHERE ai_analysis_summary_id = ANY(%s)", (parked,)
            )
        cur.execute(
            """
            INSERT INTO public.ai_analysis_summary
              (status, filter_json, search_form_type, link_analysis_status, work_space_name,
               model_id, prompt, summary_type_publication, entity_type_analysis,
               query_status, last_insertion_time)
            SELECT status, filter_json, search_form_type, 1, work_space_name,
                   model_id, prompt, false, '["location_name"]'::jsonb, 0, now()
            FROM public.ai_analysis_summary WHERE ai_analysis_summary_id = %s
            RETURNING ai_analysis_summary_id
            """,
            (src_id,),
        )
        rid = cur.fetchone()[0]
        conn.commit()
        print(f"test_row={rid} cloned_from={src_id} parked={','.join(map(str, parked)) or 'none'}")

    elif action == "restore":
        rid = int(sys.argv[2])
        parked = [int(x) for x in sys.argv[3].split(",") if x] if len(sys.argv) > 3 else []
        cur.execute("DELETE FROM public.ai_analysis_summary WHERE ai_analysis_summary_id=%s", (rid,))
        if parked:
            cur.execute(
                "UPDATE public.ai_analysis_summary SET query_status=0 "
                "WHERE ai_analysis_summary_id = ANY(%s)", (parked,)
            )
        conn.commit()
        print(f"deleted test row {rid}; restored {len(parked)} rows")

    conn.close()


if __name__ == "__main__":
    main()
