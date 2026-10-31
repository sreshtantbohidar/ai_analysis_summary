"""Opt-in live validation. Every DB mutation is rolled back, including output.
Run inside the existing pyspark container after reviewing the chosen case IDs.
"""
import base64,copy,importlib.util,json,re,time,os
from pathlib import Path
from psycopg2.extras import Json
spec=importlib.util.spec_from_file_location('v17',str(Path(__file__).with_name('ai_analysis_summary_v3.1.1.17_rag.py')))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OLLAMA_TIMEOUT=90
cases=[(2016,'general'),(1933,'event_training'),(1931,'training'),(1926,'event_general'),
       (1870,'force'),(1860,'event_force'),(1803,'search_by_query'),(1781,'infrastructure'),
       (1292,'sitrep'),(1207,'profile'),(1941,'table'),(2003,'custom'),
       (2017,'publication_unavailable'),(1919,'citations_publication'),(1846,'citations'),(2016,'chunks_enabled')]
requested=os.getenv('AI_SUMMARY_E2E_CASES','')
if requested:
    labels=set(requested.split(','));cases=[case for case in cases if case[1] in labels]
if os.getenv('AI_SUMMARY_E2E_CUSTOM_PUBLICATION')=='1':
    cases.append((2006,'custom_publication'))
results=[]
for rid,label in cases:
 c=m.postgres_connection();started=time.monotonic();original=None
 try:
  with c.cursor() as q:
   q.execute("SET LOCAL lock_timeout='2s'")
   q.execute('SELECT query_status,query_text,filter_json,search_form_type,entity_type_analysis,summary_type_publication,is_citation,is_chunk,prompt FROM ai_analysis_summary WHERE ai_analysis_summary_id=%s FOR UPDATE NOWAIT',(rid,))
   original=q.fetchone()
   if not original or original[0]!=1:raise AssertionError('Only completed historical requests may be used')
   status,text,filters,typ,selection,pub,cite,chunk,prompt=original
   # Keep each real ES test bounded to one matching real document. Table cases
   # use their original submitted table in full. Original filters and outputs
   # are restored automatically by rollback below.
   test_filter=copy.deepcopy(filters)
   if '#from_table#' not in (prompt or ''):
    response=m.es.search(index=m.CUREENT_INDEX_NAME,body={'query':filters.get('query',{'match_all':{}}),'size':100,'_source':{'excludes':['desc_embedding']}})
    hits=response.get('hits',{}).get('hits',[])
    if not hits:raise AssertionError('Stored filter has no current ES matches')
    selected_names,_=m._match_entity_type_selection_to_dimensions(m._normalize_entity_type_selection(selection))
    eligible=hits
    if selected_names and not m._is_new_prompt_selection(selection):
        eligible=[hit for hit in hits if any(d['name'] in selected_names and d.get('groups') for d in m._build_dimension_groups([hit]))]
        if not eligible:raise AssertionError('No sample document matches the requested entity scope')
    original_query=test_filter.get('query',{'match_all':{}})
    test_filter={'query':{'bool':{'must':[original_query], 'filter':[{'ids':{'values':[eligible[0]['_id']]}}]}}}
   if label=='chunks_enabled':chunk=1
   q.execute('UPDATE ai_analysis_summary SET query_status=2,filter_json=%s,is_chunk=%s WHERE ai_analysis_summary_id=%s',(Json(test_filter),chunk,rid))
   m.ai_analysis_summary_check((test_filter,rid,typ,selection,pub,cite,chunk),False,True,False,0,q,
        chunk_size=2200,chunk_output_tokens=600,consolidation_output_tokens=1200,llm_context_tokens=8192,chunk_workers_arg=1)
   q.execute('SELECT query_status,query_text FROM ai_analysis_summary WHERE ai_analysis_summary_id=%s',(rid,));new_status,new_text=q.fetchone()
   assert new_status==1,'Request did not finish'
   html=base64.b64decode(new_text,validate=True).decode('utf-8')
   assert len(html)>1000,'Empty report'
   assert 'Analysis Unavailable' not in html,'Report contains failed analysis portion'
   assert 'AI analysis unavailable' not in html,'Model failure placeholder returned'
   assert 'No matching source records' not in html,'Source records unexpectedly empty'
   if cite:assert 'Citation No.' in html,'Missing requested citation table'
   publication_warning='Publication analysis is unavailable' in html or 'Publication analysis failed' in html
   if pub:assert publication_warning or 'publication-summary' in html,'Publication request silently omitted'
   results.append({'id':rid,'case':label,'status':'DEPENDENCY_BLOCKED' if publication_warning else 'PASS','seconds':round(time.monotonic()-started,1),
                   'report_bytes':len(html.encode()),'publication_dependency_unavailable':publication_warning})
 except Exception as ex:
  results.append({'id':rid,'case':label,'status':'FAIL','seconds':round(time.monotonic()-started,1),'error':str(ex)})
 finally:
  c.rollback()
  if original:
   with c.cursor() as q:
    q.execute('SELECT query_status,query_text,filter_json,search_form_type,entity_type_analysis,summary_type_publication,is_citation,is_chunk,prompt FROM ai_analysis_summary WHERE ai_analysis_summary_id=%s',(rid,))
    assert q.fetchone()==original, 'Original row was not restored'
  c.close()
 Path('/tmp/ai_summary_v17_e2e_progress.json').write_text(json.dumps(results,indent=2))
 print('E2E_RESULT '+json.dumps(results[-1]),flush=True)
Path('/tmp/ai_summary_v17_e2e_results.json').write_text(json.dumps(results,indent=2))
print('E2E_TOTAL '+json.dumps({'cases':len(results),'passed':sum(r['status']=='PASS' for r in results),'failed':sum(r['status']=='FAIL' for r in results),'dependency_blocked':sum(r['status']=='DEPENDENCY_BLOCKED' for r in results),'all_original_rows_restored':True}),flush=True)
