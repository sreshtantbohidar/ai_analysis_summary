# AI Summary v3.1.1.17 — changes and validation

## Files and deployment

The original `ai_analysis_summary_v3.1.1.16_rag.py` is retained unchanged.
Fixes live in `ai_analysis_summary_v3.1.1.17_rag.py`. The existing
`start_ai_summary.sh` selects `.17`; no additional launcher is introduced.
The original launcher was backed up on the server before replacement.

The publication client continues importing `publication_base_url` and
`publication_token` from the same `/app/constants.py` as the worker. The host
file is `/home/demo/Documents/pyspark/constants.py`. Only those two settings
were updated with the endpoint and token supplied by the user. Credentials and
constants.py are excluded from this backup repository.

## Changes

- Recognize both `new_prompt` and the UI's `Add New Prompt` marker.
- Decode the `#from_table#` envelope into user instructions and submitted records.
  Analyze those submitted records without substituting Elasticsearch results or
  grouping away table rows. Explicitly empty/invalid tables do not fall back to
  unrelated records.
- Keep full source descriptions and business fields for custom prompts. Exclude
  embeddings and duplicated index payloads from model evidence.
- Budget prompts using both word count and UTF-8 size. Split even a single large
  record or value group without truncating its text. Budget custom requests
  after accounting for the actual instruction length.
- Preserve global record numbering across chunks and align custom-request
  markdown with the citation table's record order.
- Include nested equipment data and AOI change details in model evidence. An AOI
  field was previously marked handled before its content was emitted.
- Skip missing-value sentinels in prompt copies while preserving meaningful zero
  and false values. Do not edit Elasticsearch source data.
- Reduce repeated evidence within analytical dimensions; stop classifying every
  coordinate as infrastructure and every arbitrary activity as training.
- Keep the citation toggle effective in generated prose, remove empty brackets
  and empty colon-only paragraphs, and preserve URL/HTML attributes during
  citation and date cleanup.
- Format visible dates as DD-MM-YYYY without changing dates inside download links.
- Support stored `Search by Query Analysis` requests and profile queries using
  filter shapes beyond `match_phrase_prefix`.
- Use logged defaults for missing historical prompt/model configuration rather
  than crash on deleted master-table references.
- Show incomplete-analysis/failed-publication warnings rather than silently
  completing a blank report. Encode terminal error output consistently as base64
  and release a failed processing transaction before retry recovery.

## Existing behavior preserved

The Elasticsearch retrieval, general-analysis form_id grouping/deduplication,
trend/change-detection entry points, equipment catalog lookup, formation lookup,
source table construction, imagery links, queue selection and legacy row shapes
are retained. Regression tests compare grouping, deduplication, source-table and
imagery helpers with `.16`, and exercise all eight publication/citation/chunk
flag combinations plus equipment and formation section gates.

Historical missing configuration now falls back to the configured default;
these runs log that change rather than claiming the unavailable historical
model was used. Valid existing prompt/model selections continue taking priority.

## Database coverage

A read-only audit inspected all 674 stored requests: 10 analysis types, 38
table-search requests (1–1,467 submitted records), 40 custom-prompt requests,
and seven stored flag combinations. All 674 configurations resolve through
`.17` without configuration errors. No stored rows had chunking enabled, so
that mode is covered separately in regression and live tests.

The historical audit found 14 rows already marked processing and 51 completed
reports containing unavailable-analysis text. Those historical statuses/reports
were not mass-reset or overwritten.

## Validation

- 29 offline regression tests pass locally and are run in the actual PySpark
  container before release.
- Python compilation and existing launcher shell syntax checks pass.
- Live tests call real Elasticsearch, the configured model, and PostgreSQL.
  Standard cases are deliberately restricted to one real matching source
  document; the table-search case uses its original submitted table.
- Each live test temporarily sets `query_status=2`, generates output and checks
  completion/base64 HTML, then rolls back the transaction. Original status,
  filters, flags, prompt and report are compared after rollback.
- Full-dataset prompt preparation was checked for all 38 stored table-search
  cases. These are not full-dataset semantic/model-quality benchmarks.

The final live matrix has 17 distinct cases: 14 pass and 3 publication-enabled
cases are dependency-blocked by the publication server's CUDA failure. The
initial citation-scope sample was out of scope; an equipment-bearing sample
was selected and that case passed on rerun. All temporary database changes
were rolled back and the original rows verified unchanged.

The supplied publication token authenticates successfully. The API route is
present even though it is absent from the service's published OpenAPI list;
the remaining error is inside the publication service: PyTorch cannot access a
CUDA device (torch 2.14.1+cu130 / CUDA build 13.0). The client implementation
and constants import mechanism are retained.

Version `.17` is running in `pyspark` through the updated existing launcher.
The original `.16` source is unchanged and the 14 historical processing rows
remain untouched. Live results are recorded in `validation_v17.json`.

## Defect coverage and remaining integration work

| Defect IDs | Coverage / remaining verification |
|---|---|
| 775, 852, 1551, 1856 | Visible blank/model/publication failure handling; live generation tests. Model/service failure still requires service availability. |
| 1271, 1611, 1783 | Full evidence retention, table envelope decoding and bounded delivery. No claim that every generated fact has been manually adjudicated. |
| 1318, 1373, 1627 | Bounded output and fewer duplicate evidence blocks. Large-dataset latency remains model/hardware dependent; no SLA is claimed. |
| 1372, 1609, 1610, 1673 | Missing sentinels, empty brackets/paragraphs and citation toggle cleanup. |
| 1530, 1789 | Nested equipment and AOI evidence inclusion. |
| 1617, 1618 | Existing source filename fallback and separate equipment rendering retained and regression checked. |
| 1619 | Visible date formatting without breaking resource URLs. |
| 1278, 1790 | Existing report heading handling retained; repetition reduced through dimension scoping and evidence deduplication. Generative wording still needs QA. |
| 1781, 1815 | Worker-side custom prompt handling. Frontend/API rejection before insertion into the queue requires its owning application. |
| 1791, 1835, 1841 | Publication settings corrected; client still requests all-publication Q&A and renders every returned source. Retrieval breadth and relevance depend on the publication API. |
| 1602, 1672, 1780, 1792, 1834 | PDF/Word watermarks, PDF download performance, immediate UI loading state and HTTP-to-HTTPS publication download proxying belong to the web/export application. They are not fixed by this worker copy. |
| 1633 | Combined-report generation uses a separate report-generation path; this worker's tests do not establish an end-to-end pass for that flow. |

Already Released entries remain regression/QA cases; this work does not change
the defect workbook or mark any QA item closed.

## Commands

From `/app/ai_analysis_2.5` inside the `pyspark` container:

```sh
python3 test_ai_summary_v17.py
./start_ai_summary.sh status
./start_ai_summary.sh restart
```

`e2e_ai_summary_v17.py` is an explicit live-test utility. It uses real service
calls and temporary database updates, all rolled back. Its small representative
samples do not replace full-volume or browser/export acceptance testing.
