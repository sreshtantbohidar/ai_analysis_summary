"""Offline regressions for the v3.1.1.17 worker. No DB/service access required.

Run: python3 test_ai_summary_v17.py
Functions/constants are loaded from AST so importing the worker cannot connect
services, claim queue rows, or require the production constants.py.
"""
import ast
import base64
import copy
import difflib
import html
import json
import os
from pathlib import Path
import re
import time
import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import Mock

SOURCE = Path(__file__).with_name('ai_analysis_summary_v3.1.1.17_rag.py')

def load_worker(source=SOURCE):
    env = dict(globals(), _html=html, Elasticsearch=object, LLM_MODEL="test")
    tree = ast.parse(source.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Only static constants and compiled regexes; omit live configuration
            # and Elasticsearch construction, including nested branches.
            calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            if any(not (isinstance(n.func, ast.Name) and n.func.id == 'frozenset' or
                        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == 're' and n.func.attr in ('compile', 'escape') or
                        isinstance(n.func, ast.Attribute) and n.func.attr == 'join') for n in calls):
                continue
            if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == 'constants' for n in ast.walk(node.value)):
                continue
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
            except NameError:
                pass
        elif isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env

class Regressions(unittest.TestCase):
    def setUp(self):
        self.m = load_worker()

    def test_ui_custom_prompt_aliases(self):
        for selection in (['new_prompt'], ['Add New Prompt'], '["Add New Prompt"]', ['NEW PROMPT']):
            self.assertTrue(self.m['_is_new_prompt_selection'](selection))
        self.assertFalse(self.m['_is_new_prompt_selection'](['equipment_name']))

    def test_table_envelope_keeps_all_records_and_request(self):
        records = [{'description': 'detail ' + str(i), 'activity_date': '2026-10-31'} for i in range(1350)]
        prompt, hits = self.m['_extract_table_prompt']('Show deployment#from_table# perform the analysis on below data ' + json.dumps(records) + ' Please answer in English.')
        self.assertEqual(len(hits), 1350)
        self.assertNotIn('detail 1349', prompt)
        self.assertEqual(hits[-1]['_source'], records[-1])
        self.assertIn('Please answer in English.', prompt)

    def test_empty_table_is_not_normal_prompt(self):
        self.assertEqual(self.m['_extract_table_prompt']('Request#from_table# []')[1], [])
        self.assertIsNone(self.m['_extract_table_prompt']('ordinary request')[1])

    def test_invalid_table_does_not_fall_back_to_es(self):
        for prompt in ('Q#from_table# invalid', 'Q#from_table# [{broken}]', 'Q#from_table# [1]'):
            with self.assertRaises(ValueError):
                self.m['_extract_table_prompt'](prompt)

    def test_lossless_split_of_long_single_record(self):
        original = 'one two three ' * 12000 + 'LAST_UNIQUE_FACT'
        parts = self.m['_split_text_to_token_budget'](original, 900)
        self.assertGreater(len(parts), 1)
        self.assertEqual(''.join(parts), original)
        self.assertTrue(all(self.m['_combined_estimate_tokens'](p) <= 900 for p in parts))

    def test_overlong_request_raises_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'Shorten the prompt'):
            self.m['_split_text_to_token_budget']('source', 50)

    def test_full_custom_record_fields_not_120_char_excerpt(self):
        desc = 'Full description with distinct observations. ' * 50 + 'FINAL_EQUIPMENT'
        source = {'description':desc,'activity_date':'2026-10-31','combined_equipment_name':'Test equipment', 'location_name':'Site A'}
        result = self.m['convert_hits_to_markdown']([{'_source':source}], 'general area analysis', preserve_all_fields=True)
        self.assertIn(desc, result)
        self.assertIn('Test equipment', result)
        self.assertIn('2026-10-31', result)

    def test_null_sentinels_removed_but_zero_is_kept(self):
        source = {'a':'Undefined','b':'Location not known','count':0,'flag':False,'description':'unknown unit at site'}
        cleaned = self.m['_clean_evidence_value'](source)
        self.assertNotIn('a', cleaned)
        self.assertNotIn('b', cleaned)
        self.assertEqual(cleaned['count'], 0)
        self.assertIs(cleaned['flag'], False)
        self.assertEqual(cleaned['description'], source['description'])

    def test_aoi_details_reach_overall_evidence(self):
        source = {'source_type':'imint','aoi_details':[{'aoi_name':'AOI-I','previous_imagery':'2 tents','present_imagery':'5 tents','change_discerned':'3 new tents'}]}
        result = self.m['_render_dimension_records']([(1,{'_source':source})],include_citations=False)
        self.assertIn('aoi_details:', result)
        self.assertIn('AOI-I', result)
        self.assertIn('3 new tents', result)
        self.assertNotIn('Citation No.', result)

    def test_citation_toggle_and_empty_brackets(self):
        original = '<p>Deployment at site (3, 4). [] ()</p><p>:</p>'
        hidden = self.m['sanitize_ai_report_text'](original, include_citations=False)
        self.assertNotIn('(3, 4)', hidden)
        self.assertNotIn('[]', hidden)
        self.assertNotIn('()', hidden)
        self.assertNotIn('<p>:</p>', hidden)
        self.assertIn('(3, 4)', self.m['sanitize_ai_report_text'](original, include_citations=True))

    def test_dates_preserve_urls(self):
        result = self.m['format_dates_for_display']('<p>2026-10-31 and October 1, 2026</p><a href="/2026-10-31/file">link</a>')
        self.assertIn('31-10-2026', result)
        self.assertIn('01-10-2026', result)
        self.assertIn('href="/2026-10-31/file"', result)

    def test_single_large_dimension_group_is_fully_delivered(self):
        source = {'description':'Observation with unique detail. ' * 6000 + 'LAST_FACT'}
        groups = [{'label':'Site A','citations':[1],'records':[(1,{'_source':source})]}]
        prompts=[]
        def call(prompt,*args,**kwargs):
            prompts.append(prompt)
            self.assertLess(self.m['_combined_estimate_tokens'](prompt) + kwargs['max_output_tokens'] + 256, 8193)
            return '<p>Distinct finding ' + str(len(prompts)) + '.</p>'
        self.m['_get_direct_ollama_response']=call
        result = self.m['_analyze_overall_dimension_batch']('Location Analysis', groups, 'Analyze source', 'test', 8192, include_citations=False)
        self.assertGreater(len(prompts),1)
        self.assertTrue(any('LAST_FACT' in p for p in prompts))
        self.assertIn('Distinct finding', result)

    def test_single_long_custom_table_row_fits_context(self):
        prompts=[]
        def call(prompt,*args,**kwargs):
            prompts.append(prompt)
            self.assertLess(self.m['_combined_estimate_tokens'](prompt) + 1500 + 256, 8193)
            return '<p>Finding.</p>'
        self.m['_get_direct_ollama_response']=call
        markdown = '| Description |\n|---|\n| ' + 'data item ' * 16000 + 'TAIL_FACT |'
        result=self.m['analyze_type_mapping_records'](markdown,'Analyze this', 'test',1,8192,include_citations=False)
        self.assertGreater(len(prompts),1)
        self.assertTrue(any('TAIL_FACT' in p for p in prompts))
        self.assertIn('Finding',result)

    def test_failed_custom_part_is_visible(self):
        self.m['_get_direct_ollama_response']=lambda *a,**k:''
        self.m['get_llm_response_direct']=lambda *a,**k:'<p>AI analysis unavailable — LLM model could not generate a response.</p>'
        result=self.m['analyze_type_mapping_records']('| data |','Analyze','test',1,8192)
        self.assertIn('analysis is incomplete',result)

    def test_submitted_table_never_fetches_elasticsearch(self):
        # Exercise the actual worker entry point without any service/DB writes.
        self.m['get_prompt_and_model']=lambda *a,**k:{'prompt':'Analyze', 'model_info':'test','table_hits':[{'_source':{'description':'table fact'}}]}
        self.m['get_data_from_elastic']=Mock(side_effect=AssertionError('ES must not be called'))
        self.m['analyze_type_mapping_records']=lambda *a,**k:'<p>table fact analyzed</p>'
        self.m['build_imint_imagery_html']=lambda *a,**k:''
        self.m['build_tabbed_html']=lambda parts,**k:''.join(parts)
        captured=[]
        self.m['update_ai_analysis_summary_query_text']=lambda c,i,t:captured.append(base64.b64decode(t).decode())
        self.m['ai_analysis_summary_check'](({},999,'General Area Analysis',[],False,False,False),False,True,False,0,Mock(),llm_context_tokens=8192)
        self.assertIn('table fact analyzed',captured[0])
        self.m['get_data_from_elastic'].assert_not_called()

    def test_publication_failure_is_visible_in_report(self):
        self.m['get_prompt_and_model']=lambda *a,**k:{'prompt':'Analyze','model_info':'test','table_hits':[]}
        self.m['build_publication_summary_section']=lambda *a,**k:''
        self.m['build_tabbed_html']=lambda parts,**k:''.join(parts)+k.get('publication_summary_html','')
        captured=[]
        self.m['update_ai_analysis_summary_query_text']=lambda c,i,t:captured.append(base64.b64decode(t).decode())
        self.m['ai_analysis_summary_check'](({},999,'General Area Analysis',[],True,False,False),False,True,False,0,Mock(),llm_context_tokens=8192)
        self.assertIn('report is incomplete for publications',captured[0])
        self.assertIn('No matching source records',captured[0])

    def test_token_estimate_handles_json_without_spaces(self):
        text = '{"description":"' + 'x' * 40000 + '"}'
        self.assertGreaterEqual(self.m['_combined_estimate_tokens'](text),10000)
        parts = self.m['_split_text_to_token_budget'](text,1000)
        self.assertEqual(''.join(parts),text)
        self.assertTrue(all(self.m['_combined_estimate_tokens'](x)<=1000 for x in parts))

    def test_citation_cleanup_preserves_resource_attributes(self):
        text='<a href="/citations/Record 3/file" class="citations">Site (3, 4)</a><style>.citations {color:red}</style>'
        result=self.m['_strip_record_labels_from_report'](text,False)
        self.assertIn('href="/citations/Record 3/file"',result)
        self.assertIn('class="citations"',result)
        self.assertIn('.citations {color:red}',result)
        self.assertNotIn('Site (3, 4)',result)

    def test_chunk_citations_keep_global_record_numbers(self):
        result=self.m['convert_hits_to_markdown']([{'_source':{'description':'example'}}], 'general area analysis',start_row_idx=41)
        self.assertIn('| 41 |',result)
        self.assertNotIn('| 1 |',result)

    def test_legacy_missing_prompt_and_model_use_defaults(self):
        cursor=Mock();cursor.fetchone.return_value=(None,None,None)
        result=self.m['get_prompt_and_model'](cursor,1932)
        self.assertIn('supplied source records',result['prompt'])
        self.assertEqual(result['model_info'],'test')

    def test_all_stored_analysis_types_are_supported(self):
        supported={t.lower() for config in self.m['TYPE_MAPPING'].values() for t in config['types']}
        stored={'general area analysis','event training areas analysis','training areas analysis',
                'event general area analysis','force disposition analysis','event force disposition analysis',
                'search by query analysis','infra development analysis','pla sitrep analysis'}
        self.assertFalse(stored-supported)

    def test_deleted_historical_model_uses_default(self):
        cursor=Mock();cursor.fetchone.side_effect=[('Analyze',None,29),None]
        self.assertEqual(self.m['get_prompt_and_model'](cursor,1849)['model_info'],'test')

    def test_generic_activity_is_not_training_or_construction(self):
        hit={'_source':{'activity_type':'routine observation','coordinates':'1,2','location_name':'Site A'}}
        dimensions=self.m['_build_dimension_groups']([hit])
        for dimension in dimensions:
            if dimension['name'] in ('Training and Readiness Analysis','Infrastructure Analysis'):
                self.assertFalse(dimension['groups'],dimension['name'])

    def test_nested_equipment_reaches_dimension_evidence(self):
        result=self.m['_render_dimension_records']([(1,{'_source':{'equipment':[{'equipment_name':'Test UAV','count':2}]}})])
        self.assertIn('Test UAV',result)
        self.assertIn('"count": 2',result)

    def test_existing_grouping_dedup_and_date_filters_match_v16(self):
        old=load_worker(SOURCE.with_name('ai_analysis_summary_v3.1.1.16_rag.py'))
        hits=[{'_id':str(i),'_source':{'form_id':i//2,'activity_date':f'2026-10-{i+1:02d}',
               'description_hash':str(i//2),'description':f'detail {i}','equipment_name':'Test UAV',
               'location_name':'Site A','comments':''}} for i in range(6)]
        for name,args in [('_keep_latest_by_form_id',(hits,)),
                          ('_dedup_hits',(hits,['description_hash','location_name'])),
                          ('_collect_equipment_names',(hits,)),
                          ('_collect_formation_names',(hits,)),
                          ('extract_date_range',({'query':{'bool':{'filter':[{'range':{'activity_date':{'gte':'2026-10-01','lte':'2026-10-31'}}}]}}},))]:
            self.assertEqual(old[name](*copy.deepcopy(args)),self.m[name](*copy.deepcopy(args)),name)

    def test_existing_source_table_and_imagery_output_match_v16(self):
        old=load_worker(SOURCE.with_name('ai_analysis_summary_v3.1.1.16_rag.py'))
        hits=[{'_id':'x','_source':{'source_type':'imint','file_name':'file.pdf','activity_date':'2026-10-31',
              'imint_number':'1','all_images':['https://example.test/image.png']}}]
        for name in ('build_doc_id_activity_date_table','build_imint_imagery_html'):
            self.assertEqual(old[name](hits),self.m[name](hits),name)

    def test_existing_section_toggles_and_chunk_gate(self):
        for pub in (False,True):
          for cite in (False,True):
           for chunk in (False,True):
            with self.subTest(publication=pub,citation=cite,chunk=chunk):
                m=load_worker()
                hit={'_source':{'description':'A substantive source observation.','equipment_name':'Test UAV',
                      'enemy_formation_name':'Test formation','activity_date':'2026-10-31'}}
                m['get_prompt_and_model']=lambda *a,**k:{'prompt':'Analyze','model_info':'test'}
                m['get_data_from_elastic']=lambda *a,**k:{'hits':{'total':{'value':1},'hits':[hit]}}
                m['synthesize_combined_analysis']=lambda *a,**k:'<p>Source analysis.</p>'
                m['_process_chunk_job']=Mock(side_effect=lambda job:(job[0],job[2],'<p>Chunk findings.</p>',0.01))
                m['build_publication_summary_section']=Mock(return_value='<p>Publication facts.</p>')
                m['fetch_equipment_specifications']=Mock(return_value='<p>Equipment specifications.</p>')
                m['fetch_formation_details']=Mock(return_value='<p>Formation details.</p>')
                m['build_imint_imagery_html']=lambda *a:''
                m['build_tabbed_html']=lambda parts,**k:''.join(parts)+''.join(str(v or '') for v in k.values())
                out=[];m['update_ai_analysis_summary_query_text']=lambda c,i,t:out.append(base64.b64decode(t).decode())
                m['ai_analysis_summary_check'](({'query':{'match_all':{}}},999,'general area analysis',
                     ['equipment_name','enemy_formation_name'],pub,cite,chunk),False,True,False,0,Mock(),llm_context_tokens=8192)
                self.assertIn('Source analysis.',out[0])
                self.assertIn('Equipment specifications.',out[0]);self.assertIn('Formation details.',out[0])
                self.assertEqual('Publication facts.' in out[0],pub)
                self.assertEqual('Citation No.' in out[0],cite)
                self.assertEqual(m['_process_chunk_job'].called,chunk)

    def test_profile_supports_filters_without_match_phrase_prefix(self):
        self.m['get_prompt_and_model']=lambda *a,**k:{'prompt':'Analyze profile','model_info':'test'}
        self.m['get_data_from_elastic']=lambda *a,**k:{'hits':{'hits':[{'_source':{'person_name':'Example person','description':'Profile facts'}}]}}
        self.m['analyze_type_mapping_records']=Mock(return_value='<p>Profile facts analyzed.</p>')
        self.m['build_imint_imagery_html']=lambda *a:''
        self.m['build_tabbed_html']=lambda parts,**k:''.join(parts)
        out=[];self.m['update_ai_analysis_summary_query_text']=lambda c,i,t:out.append(base64.b64decode(t).decode())
        self.m['ai_analysis_summary_check'](({'query':{'terms':{'person_name':['Example person']}}},999,
                  'profile analysis',[],False,False,False),False,True,False,0,Mock(),llm_context_tokens=8192)
        self.assertIn('Profile facts analyzed.',out[0])
        self.m['analyze_type_mapping_records'].assert_called_once()

    def test_custom_evidence_excludes_embeddings_but_keeps_source_facts(self):
        hit={'_source':{'description':'Source fact','equipment':[{'name':'Test UAV'}],
             'desc_embedding':[0.123]*1024,'form_data_search':'duplicated index internals'}}
        result=self.m['convert_hits_to_markdown']([hit],'general area analysis',preserve_all_fields=True)
        self.assertIn('Source fact',result);self.assertIn('Test UAV',result)
        self.assertNotIn('0.123',result);self.assertNotIn('duplicated index internals',result)

if __name__ == '__main__':
    unittest.main(verbosity=2)
