"""
Test: Verify collapse + aggs + dedup logic works with the new scroll implementation.
"""
import copy
import json
import sys
import traceback
from collections import Counter

import psycopg2
from elasticsearch import Elasticsearch

PG_DB_NAME = 'isquare'
DJANGO_HOST = '192.168.1.125'
PG_USER = 'postgres'
PG_PORT = 5432
PG_PASSWORD = 'TeCtUmFaTbOy321'
ELASTIC_HOST = '192.168.1.125'
ELASTIC_PORT = 9200
ELASTIC_CLIENT_SCHEME = 'https'
ELASTICSEARCH_USERNAME = 'elastic'
ELASTICSEARCH_PASSWORD = '30oIsFcjJa8Zao+iq5*e'
CUREENT_INDEX_NAME = 'fatboy_data'

pg_conn = psycopg2.connect(
    database=PG_DB_NAME, host=DJANGO_HOST,
    user=PG_USER, password=PG_PASSWORD, port=PG_PORT,
)
es = Elasticsearch(
    [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": ELASTIC_CLIENT_SCHEME}],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False, ssl_show_warn=False,
)


# ── Helpers matching ai_analysis_summary_v3.1.1.py ──
def _extract_field(source, field_path):
    val = source
    for part in field_path.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def _build_dedup_key(source, dedup_fields):
    parts = []
    for field in dedup_fields:
        val = _extract_field(source, field)
        parts.append(str(val).strip() if val is not None else "")
    return "\x00".join(parts)


def _dedup_hits(hits, dedup_fields):
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
        print(f"     Dedup by {dedup_fields}: {len(hits)} -> {len(result)} (dropped {dropped})")
    else:
        print(f"     No duplicates found (all {len(hits)} unique)")
    return result


def generic_es_scroll_test(client, index_name, base_query_json, batch_size=1000, keep_alive="2m"):
    """Matches ai_analysis_summary_v3.1.1.py generic_es_scroll."""
    query = copy.deepcopy(base_query_json)
    collapse_field = query.pop("collapse", None)
    collapse_key = None
    if collapse_field and isinstance(collapse_field, dict):
        collapse_key = collapse_field.get("field", None)

    dedup_fields = []
    if collapse_key:
        dedup_fields.append(collapse_key)
    dedup_fields.extend(["description", "comments"])
    _seen = set()
    unique_dedup_fields = []
    for f in dedup_fields:
        if f not in _seen:
            _seen.add(f)
            unique_dedup_fields.append(f)

    if collapse_key:
        print(f"     Collapse on '{collapse_key}' — scrolling, dedup by {unique_dedup_fields}")
    else:
        print(f"     Dedup by {unique_dedup_fields}")

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
                pass

    print(f"     Scroll total: {total_count}, unique: {len(seen)}, dropped: {dedup_count}")


def get_data_from_elastic_test(elastic_query):
    """Matches ai_analysis_summary_v3.1.1.py get_data_from_elastic."""
    index_name = CUREENT_INDEX_NAME
    try:
        elastic_query = copy.deepcopy(elastic_query)
        size = elastic_query.get("size", None)
        has_aggs = "aggs" in elastic_query or "aggregation" in elastic_query
        has_collapse = "collapse" in elastic_query
        batch_size = 1000

        if has_collapse:
            all_hits = list(generic_es_scroll_test(es, index_name, elastic_query, batch_size=batch_size))
            result = {"hits": {"total": {"value": len(all_hits)}, "hits": all_hits}}
            return result

        if size is not None and size <= batch_size:
            return es.search(index=index_name, body=elastic_query)

        all_hits = list(generic_es_scroll_test(es, index_name, elastic_query, batch_size=batch_size))
        result = {"hits": {"total": {"value": len(all_hits)}, "hits": all_hits}}

        if has_aggs:
            agg_query = copy.deepcopy(elastic_query)
            agg_query["size"] = 0
            try:
                agg_response = es.search(index=index_name, body=agg_query)
                if "aggregations" in agg_response:
                    result["aggregations"] = agg_response["aggregations"]
                    if "hits" in agg_response and "total" in agg_response["hits"]:
                        result["hits"]["total"] = agg_response["hits"]["total"]
            except Exception as agg_err:
                print(f"     [WARN] Agg query failed: {agg_err}")

        return result
    except Exception as e:
        print(f"     [ERROR] {e}")
        traceback.print_exc()
        return {"hits": {"total": {"value": 0}, "hits": []}}


# ── Fetch queries from DB ──
print("=" * 70)
print("STEP 1: Fetching queries from DB")
print("=" * 70)

with pg_conn.cursor() as cur:
    cur.execute("""
        SELECT ai_analysis_summary_id, search_form_type, filter_json
        FROM public.ai_analysis_summary
        WHERE filter_json IS NOT NULL
        ORDER BY ai_analysis_summary_id DESC
        LIMIT 30
    """)
    rows = cur.fetchall()

collapse_q, agg_q, plain_q = [], [], []
for row_id, form_type, fjson in rows:
    if not fjson:
        continue
    try:
        q = json.loads(fjson) if isinstance(fjson, str) else fjson
    except Exception:
        continue
    if "collapse" in q:
        collapse_q.append((row_id, form_type, q))
    elif "aggs" in q or "aggregation" in q:
        agg_q.append((row_id, form_type, q))
    else:
        plain_q.append((row_id, form_type, q))

print(f"  Collapse: {len(collapse_q)}  |  Aggs: {len(agg_q)}  |  Plain: {len(plain_q)}\n")


# ── Test COLLAPSE ──
print("=" * 70)
print("STEP 2: COLLAPSE queries (scroll + Python dedup)")
print("=" * 70)

for row_id, form_type, q in collapse_q[:3]:
    print(f"\n>> [{row_id}] {form_type}")
    print(f"   Collapse field: {q.get('collapse', {})}")
    try:
        resp = get_data_from_elastic_test(q)
        hits = resp["hits"]["hits"]
        total = resp["hits"]["total"]["value"]
        print(f"   Returned: {total} hits")
        if hits:
            src0 = hits[0].get("_source", {})
            ck = q.get("collapse", {}).get("field", "")
            print(f"   Sample collapse key: {src0.get(ck.split('.')[0], 'N/A')}")
            print(f"   Sample description:  {str(src0.get('description', ''))[:100]}...")
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()


# ── Test AGGS ──
print("\n" + "=" * 70)
print("STEP 3: AGGS queries (scroll + aggs bucket fetch)")
print("=" * 70)

for row_id, form_type, q in agg_q[:3]:
    print(f"\n>> [{row_id}] {form_type}")
    agg_keys = list(q.get("aggs", q.get("aggregation", {})).keys())
    print(f"   Aggs keys: {agg_keys}")
    try:
        resp = get_data_from_elastic_test(q)
        hits = resp["hits"]["hits"]
        agg_data = resp.get("aggregations", {})
        print(f"   Hits: {len(hits)}  |  Aggregations: {'YES' if agg_data else 'NO'}")
        if agg_data:
            for key, val in agg_data.items():
                if isinstance(val, dict) and "buckets" in val:
                    buckets = val["buckets"]
                    print(f"   Agg '{key}': {len(buckets)} buckets")
                    for b in buckets[:3]:
                        print(f"     {b['key']}: {b['doc_count']} docs")
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()


# ── Test PLAIN ──
print("\n" + "=" * 70)
print("STEP 4: PLAIN queries (scroll + dedup regression)")
print("=" * 70)

for row_id, form_type, q in plain_q[:3]:
    print(f"\n>> [{row_id}] {form_type}")
    try:
        resp = get_data_from_elastic_test(q)
        hits = resp["hits"]["hits"]
        total = resp["hits"]["total"]["value"]
        print(f"   Returned: {total} hits")
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()


pg_conn.close()
print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
