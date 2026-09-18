import argparse

""" AI Analysis Summary """
import json
import time
import base64
import traceback
from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor
# import threading
import multiprocessing
import pandas as pd
import psycopg2
import requests
import sys
from elasticsearch import Elasticsearch

import psycopg2
from test_sitrep_trends import sitrep_main
from test_infra_trends import infra_main
from test_sam_trends import trends_sam_main
from test_airinspect_trends import trends_airfield_main
from test_training_trends import training_main
from test_force_disposition_trends import force_disposition_main
from change_detect_v2 import run_change_detection, run_multi_year_comparison
from elasticsearch import Elasticsearch

import importlib.util
import os
import re
# Path to the file (1 folder up)
import importlib.util
import os
from tqdm import tqdm


from typing import List
import re

from datetime import timedelta


# -----------------new import-----------------
# Path to constants.py (one folder up)
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'constants.py'))

# Load the module
spec = importlib.util.spec_from_file_location("constants", file_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

# Now manually import specific variables
PG_DB_NAME = constants.PG_DB_NAME
DJANGO_HOST = constants.DJANGO_HOST
CUREENT_INDEX_NAME = constants.CUREENT_INDEX_NAME
ELASTIC_HOST = constants.ELASTIC_HOST
ELASTIC_PORT = constants.ELASTIC_PORT
PG_USER = constants.PG_USER
PG_PORT = constants.PG_PORT
PG_PASSWORD = constants.PG_PASSWORD
LLM_IP = constants.LLM_IP
LLM_PORT = constants.LLM_PORT
LLM_MODEL = constants.LLM_MODEL
IFC_LLM_PORT = constants.IFC_LLM_PORT
IFC_LLM_TOKEN = constants.IFC_LLM_TOKEN
ANYTHINGLLM_WORKSPACE_SLUG = constants.ANYTHINGLLM_WORKSPACE_SLUG
ELASTIC_CLIENT_SCHEME = constants.ELASTIC_CLIENT_SCHEME
ELASTICSEARCH_USERNAME = constants.ELASTICSEARCH_USERNAME
ELASTICSEARCH_PASSWORD = constants.ELASTICSEARCH_PASSWORD


OLLAMA_TIMEOUT = 900   # 15 min max
MAX_RETRIES = 3

# AnythingLLM workspace model cache & defaults
_current_anythingllm_model = None
DEFAULT_CHAT_PROVIDER = "ollama"

# -----------------old import-----------------
# from constants import (
#     PG_DB_NAME,
#     DJANGO_HOST,
#     CUREENT_INDEX_NAME,
#     ELASTIC_HOST,
#     ELASTIC_PORT,
#     PG_USER,
#     PG_PORT,
#     PG_PASSWORD,
#     LLM_IP,
#     LLM_PORT,
#     LLM_MODEL,
# )

es = Elasticsearch(
    [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": ELASTIC_CLIENT_SCHEME}],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False
)


def postgres_connection():
	connect = psycopg2.connect(
		database=PG_DB_NAME,
		host=DJANGO_HOST,
		user=PG_USER,
		password=PG_PASSWORD,
		port=PG_PORT,
	)
	return connect


def _strip_wrappers(html: str) -> str:
    """Remove outer <html>, <head>, <body> tags."""
    html = re.sub(r'</?html[^>]*>', '', html, flags=re.I)
    html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.I | re.S)
    html = re.sub(r'</?body[^>]*>', '', html, flags=re.I)
    return html.strip()


def build_tabbed_html(html_strings: List[str]) -> str:
    """
    Accept exactly 3 HTML strings in this order:
    [0] AI Summary
    [1] AI Trends
    [2] AI Change Detection
    """
    labels = ["AI Summary", "AI Trends", "AI Change Detection"]
    
    # Create sections with headers
    sections = []
    for i, (label, content) in enumerate(zip(labels, html_strings)):
        if content:
            section_content = _strip_wrappers(content) if content else "<p>No data available</p>"
            sections.append(f'''
            <section class="report-section">
                <div class="section-header" id="section-{i}">
                    <h2>{label}</h2>
                </div>
                <div class="section-content">
                    {section_content}
                </div>
            </section>''')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Reports – Summary | Trends | Change</title>
  <style>
    body {{
        margin: 0;
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
        background: #f3f5f7;
        color: #212529;
        line-height: 1.6;
    }}
    header {{
        background: #004085;
        color: #fff;
        padding: 20px 40px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .report-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }}
    .section-header {{
        background: #fff;
        padding: 15px 25px;
        border-left: 4px solid #004085;
        margin: 20px 0 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .section-header h2 {{
        margin: 0;
        color: #004085;
        font-size: 1.5em;
    }}
    .section-content {{
        background: #fff;
        padding: 25px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }}
    .section-content h2 {{
        font-size: 1.3em;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-top: 0;
    }}
    .section-content p {{
        margin: 10px 0;
    }}
    footer {{
        padding: 15px 40px;
        background: #e9ecef;
        font-size: .8em;
        color: #6c757d;
        text-align: center;
        margin-top: 20px;
    }}
    @media (max-width: 768px) {{
        .report-container {{
            padding: 10px;
        }}
        .section-content {{
            padding: 15px;
        }}
        header {{
            padding: 15px 20px;
        }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>AI Intelligence Report</h1>
    <p>Generated on: {now}</p>
  </header>
  
  <div class="report-container">
    {''.join(sections)}
  </div>
  
  <footer>
    Report automatically generated by AI Analysis System
  </footer>
</body>
</html>"""


def create_llm_prompt_for_profile(complete_summary_dict_list, search_form_type, person_names, organization_names):
	summary_question = {
		'profile analysis': """
        "%s"

        "%s"
        """
	}
	activity_question = summary_question[search_form_type]
	query_string = ""
	if person_names and organization_names:
		query_string = "Create the summary of person" + ", ".join(person_names) + " and organization " + ", ".join(
			organization_names) + " in mentioned list of dictionary and create relation and summary of each other with refrence of date and file paths."
	elif person_names:
		query_string = """Deep dive and create a analysis of " """ + ", ".join(
			person_names) + """" from below json data description, the analysis should have all the personal details, involvement in any activity and his relations with other people, mention the relationship names with other people and create html table also mention the reference of file name along with your analysis".   The output should be in html format. Title should be h2 with font-size:20px and paragraph should be in p tag with font-size:15px"""
	# query_string = "Create a analysis of "+  ", ".join(person_names)+" from above json data, the analysis should have all the personal details, invlovement in any activity and his relations with other people"
	elif organization_names:
		query_string = "Create the summary of organization in above list of dictionary and create a summary of " + ", ".join(
			organization_names)
	if query_string:
		# print(activity_question,str(complete_summary_dict_list),str(query_string))
		question_temp = activity_question % (str(query_string), str(complete_summary_dict_list))
		return question_temp
	# .print_exec()

def update_anythingllm_workspace_model(model_name: str, chat_provider: str = DEFAULT_CHAT_PROVIDER) -> bool:
    """
    Update the AnythingLLM workspace to use the specified model before making a chat request.
    Uses a module-level cache to avoid redundant API calls when the model hasn't changed.
    Returns True if the workspace was updated (or already using the correct model).
    """
    global _current_anythingllm_model

    # Skip if already set to this model
    if _current_anythingllm_model == model_name:
        return True

    update_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/update"

    payload = {
        "chatProvider": chat_provider,
        "chatModel": model_name
    }

    headers = {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                update_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if r.status_code in (200, 201, 204):
                print(f"[INFO] AnythingLLM workspace model updated to: {model_name}")
                _current_anythingllm_model = model_name
                return True
            else:
                print(f"[ERROR] AnythingLLM model update status {r.status_code}: {r.text}")

        except requests.exceptions.Timeout:
            print(f"[ERROR] Model update timeout (attempt {attempt+1}/{MAX_RETRIES})")

        except Exception as e:
            print(f"[ERROR] Model update failed: {str(e)}")

        time.sleep(2 ** attempt)

    print(f"[WARN] Could not update AnythingLLM model to {model_name}, proceeding with current model")
    return False


def get_llm_response(query, llm_model, max_tokens=1200):

    # Update AnythingLLM workspace model before making the request
    update_anythingllm_workspace_model(llm_model)

    anythingllm_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"

    prompt = query.replace("\n", " ").strip()

    payload = {
        "message": prompt,
        "mode": "chat"
    }

    headers = {
        "Authorization": f"Bearer {IFC_LLM_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                anythingllm_url,
                json=payload,
                headers=headers,
                timeout=OLLAMA_TIMEOUT
            )

            if r.status_code == 200:
                result = r.json()
                return result.get("textResponse", "")

            else:
                print(f"[ERROR] AnythingLLM status {r.status_code}: {r.text}")

        except requests.exceptions.Timeout:
            print(f"[ERROR] Timeout (attempt {attempt+1}/{MAX_RETRIES})")

        except Exception as e:
            print(f"[ERROR] {str(e)}")

        time.sleep(2 ** attempt)

    return "<p>AI unavailable / offline</p>"


def correct_json_format(json_str):
	# unable to optimise this - Aditya
	open_curly_count = 0
	open_square_count = 0
	result = []
	prev_b = []
	prev_inverted = 0
	prev_inverted_array = []
	json_str = (
		json_str.replace(", ", ",")
		.replace("\n", "")
		.replace("\r", "")
		.replace(" }", "}")
		.replace("} ", "}")
		.replace("{ ", "{")
		.replace(" {", "{")
		.replace(" [", "[")
		.replace("[ ", "[")
		.replace(" ]", "]")
		.replace("] ", "]")
		.strip()
	)
	for char in json_str:
		if char == '"':
			if prev_inverted == 0:
				prev_inverted = 1
				prev_inverted_array.append('"')
			else:
				prev_inverted = 0
				prev_inverted_array.pop()
		if char == "{":
			open_curly_count += 1
			prev_b.append(char)
			result.append(char)
		elif char == "[":
			open_square_count += 1
			prev_b.append(char)
			result.append(char)
		elif char == "}":
			if prev_inverted == 1:
				result.append('"')
				prev_inverted = 0
				prev_inverted_array.pop()
			if prev_b[-1] == "[":
				result.append("]")
			prev_b.pop()
			result.append(char)
		elif char == "]":
			if prev_inverted == 1:
				result.append('"')
				prev_inverted = 0
				prev_inverted_array.pop()
			if prev_b[-1] == "{":
				result.append("}")
			prev_b.pop()
			result.append(char)
		else:
			result.append(char)
	for p_b in prev_b[::-1]:
		if p_b == "{":
			result.append("}")
		if p_b == "[":
			result.append("]")
	result = "".join(result)
	result = result.replace(",}", "}").replace(",]", "]").strip()
	# print("~~~~~~~~~~",result)
	try:
		result = json.loads(result)
	except Exception as e:
		traceback.print_exc()
		print("[!]json load failed:", e)
		result = {"summary": result}
	return result


def update_imint_ai_analysis(cursor, imint_ai_analysis_id, ai_analysis_text):
	q = """UPDATE public.imint_ai_analysis SET status=%d,ai_analysis_text='%s' WHERE imint_ai_analysis_id=%d;"""
	cursor.execute(q % (1, ai_analysis_text, imint_ai_analysis_id,))


# conn.commit()

def update_ai_analysis_summary_query_text(cursor, ai_analysis_summary_id, query_text):
	q = """UPDATE public.ai_analysis_summary SET query_status=%d,query_text='%s' WHERE ai_analysis_summary_id=%d;"""
	cursor.execute(q % (1, query_text, ai_analysis_summary_id,))


# conn.commit()


# def get_data_from_elastic(elastic_query):
# 	index_name = CUREENT_INDEX_NAME
# 	try:
# 		e_response = es.search(index=index_name, body=elastic_query)
# 		return e_response
# 	except Exception as e:
# 		print("[error] Failed to get data from elastic:", elastic_query)
# 		print(f"Exception {e}")
# 		return

def get_data_from_elastic(elastic_query):
    index_name = CUREENT_INDEX_NAME
    try:
        import copy
        elastic_query = copy.deepcopy(elastic_query)

        size = elastic_query.get("size", None)

        # =============================
        # AUTO FULL FETCH LOGIC
        # =============================
        if size is None or size <= 1000:
            all_hits = []
            batch_size = 1000

            elastic_query["size"] = batch_size

            response = es.search(
                index=index_name,
                body=elastic_query,
                scroll="2m"
            )

            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

            all_hits.extend(hits)

            while hits:
                response = es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]

                if not hits:
                    break

                all_hits.extend(hits)

            return {
                "hits": {
                    "total": {"value": len(all_hits)},
                    "hits": all_hits
                }
            }

        # =============================
        # NORMAL FLOW (explicit size)
        # =============================
        return es.search(index=index_name, body=elastic_query)

    except Exception as e:
        print("[error] Failed to get data from elastic:", elastic_query)
        print(f"Exception {e}")
        return {"hits": {"total": {"value": 0}, "hits": []}}

def count_words_and_tokens(text: str):
    word_count = 0
    prev_space = True

    for ch in text:
        is_space = ch.isspace()
        if prev_space and not is_space:
            word_count += 1
        prev_space = is_space

    token_estimate = int(word_count * 1.3)

    return word_count, token_estimate


def count_tokens_approx(text):
    word_count = 0
    prev_space = True

    for ch in text:
        is_space = ch.isspace()
        if prev_space and not is_space:
            word_count += 1
        prev_space = is_space

    return int(word_count * 1.3)


def build_chunks_from_lines(lines, max_tokens=1400):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = int(len(line.split()) * 1.3)

        if current_tokens + line_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def ai_analysis_summary_check(row, ai_trends, ai_summ, ai_change, ai_change_previous, cursor,
                             chunk_token_threshold=3000, chunk_size=2200,
                             chunk_output_tokens=800, consolidation_output_tokens=1500):
	elastic_query, ai_analysis_summary_id, search_form_type = row
	search_form_type = search_form_type.lower()
	model_prompt_dict = None
	if search_form_type != 'profile analysis':
		model_prompt_dict = get_prompt_and_model(cursor, ai_analysis_summary_id)
	e_response = get_data_from_elastic(elastic_query) if elastic_query else None
	if e_response and e_response['hits']['total']['value'] > 0:
		e_hits = e_response['hits']['hits']
	else:
		e_hits = []
	ai_change_repsonse = ''
	ai_trend_response = ''
	llm_response = ''
	print("--"+search_form_type+"--","Training Areas Analysis".lower())

	if ai_change:
		if e_hits:
			trend_start_time = time.time()
			start_date = find_lt_value(elastic_query)
			if search_form_type.lower() in [x.lower() for x in ["SAM Deployment Analysis", "Training Areas Analysis", "Infra Development Analysis", "AIR Aspects Analysis", "PLA Sitrep Analysis", "Force Disposition Analysis"]]:
				# Get model to use for change detection LLM calls
				_change_llm_model = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL
				print(f"[INFO] Running change detection for {search_form_type} [{ai_analysis_summary_id}] using model: {_change_llm_model}")
				ai_change_repsonse = run_change_detection(elastic_query, start_date=start_date, llm_model_name=_change_llm_model)
	
	if ai_change_previous:

		start_date, end_date = extract_date_range(elastic_query)

		if start_date and end_date:

			start_dt = datetime.fromisoformat(start_date)
			end_dt = datetime.fromisoformat(end_date)

			# Use the model fetched from PostgreSQL (ai_model_master), fall back to LLM_MODEL constant
			_llm_model = model_prompt_dict.get("model_info", LLM_MODEL) if model_prompt_dict else LLM_MODEL

			print(f"[INFO] Running multi-year comparison ({ai_change_previous} year(s)) for {search_form_type} [{ai_analysis_summary_id}] using model: {_llm_model}")
			raw_change_output = run_multi_year_comparison(
				elastic_query,
				start_date=start_dt.strftime("%Y-%m-%d"),
				end_date=end_dt.strftime("%Y-%m-%d"),
				years_back=ai_change_previous,
				llm_model_name=_llm_model
			)
			ai_change_repsonse = raw_change_output
			


	if ai_trends:
		if e_hits:
			trend_start_time = time.time()
			start_date = find_lt_value(elastic_query)
			# if search_form_type == "PLA Sitrep Analysis".lower():
			# 	csv_string = sitrep_main(start_date, elastic_query)
			# elif search_form_type == "Infra Development Analysis".lower():
			# 	csv_string = infra_main(start_date, elastic_query)
			# elif search_form_type == "Force Disposition Analysis".lower():
			# 	csv_string = force_disposition_main(start_date, elastic_query)
			
			if search_form_type == "SAM Deployment Analysis".lower():
				ai_trend_response = trends_sam_main(start_date,elastic_query)
			
			elif search_form_type == "Training Areas Analysis".lower():
				ai_trend_response = training_main(start_date, elastic_query)
			
			elif search_form_type == "Infra Development Analysis".lower():
				ai_trend_response = infra_main(start_date, elastic_query)
			
			elif search_form_type == "AIR Aspects Analysis".lower():
				ai_trend_response = trends_airfield_main(start_date, elastic_query)
			
			elif search_form_type == "PLA Sitrep Analysis".lower():
				ai_trend_response = sitrep_main(start_date, elastic_query)

			elif search_form_type == "Force Disposition Analysis".lower():
				ai_trend_response = force_disposition_main(start_date, elastic_query)

			else:
				print("Trends not created for {}".format(search_form_type))
				ai_trend_response = ""

			print(f"[DONE] Trend Analysis Complete for {search_form_type} [{ai_analysis_summary_id}] [time={(time.time() - trend_start_time):.2f} s]")
				
					

	if ai_summ:
		ai_summ_start_time = time.time()
		if search_form_type == 'profile analysis':
			print("1")
			print("[INFO] for profile analysis", ai_analysis_summary_id)
			if elastic_query:
				# Extracting names
				person_names = []
				organization_names = []
				if 'query' in elastic_query and 'bool' in elastic_query['query'] and 'must' in elastic_query["query"][
					"bool"]:
					for condition in elastic_query["query"]["bool"]["must"]:
						if "match_phrase_prefix" in condition:
							if "person_name" in condition["match_phrase_prefix"]:
								person_names.append(condition["match_phrase_prefix"]["person_name"])
							elif "civil_organization.civil_organization_name" in condition["match_phrase_prefix"]:
								organization_names.append(
									condition["match_phrase_prefix"]["civil_organization.civil_organization_name"])
							elif "civil_organization_name" in condition["match_phrase_prefix"]:
								organization_names.append(condition["match_phrase_prefix"]["civil_organization_name"])
				print("2")
				ai_analysis_text_base64 = 'No results'
				if person_names or organization_names:
					print("3")
					all_descriptions = []
					index = 0
					for single_json in e_hits:
						single_json = single_json['_source']
						if single_json.get("description"):
							if 'file_http_path' in single_json:
								file_http_path = single_json['file_http_path']
							else:
								index = index + 1
								file_http_path = 'file' + str(index)

							activity_date = single_json['@timestamp']
							activity_date = datetime.fromisoformat(activity_date).strftime("%Y-%m-%d")
							file_name = file_http_path.split('/')[-1]
							complete_single_profile = {"file_path": file_http_path, "file_name": file_name,
							                           "date": str(activity_date),
							                           "description": single_json['description']}
							all_descriptions.append(complete_single_profile)

					complete_summary_dict_list = []
					if all_descriptions:
						print("4")
						df = pd.DataFrame(all_descriptions)
						grouped = df.groupby("file_name").agg({
							"file_path": lambda x: list(set(x)),
							"description": lambda x: list(set(x)),
							"date": lambda x: list(set(x))
						}).reset_index()
						complete_summary_dict_list = grouped.to_dict(orient="records")
						llm_prompt = create_llm_prompt_for_profile(complete_summary_dict_list, search_form_type,person_names, organization_names)

						extra_prompt = "\n\n(PLEASE NOTE: The response you will give should be in html(hyper text markup language). Any title in the response should be in h2 tag and paragraph should be in p tag.)"
						llm_prompt += extra_prompt
 
						temp_llm_response = get_llm_response(llm_prompt, LLM_MODEL)
						temp_llm_response = f"<div>{temp_llm_response}</div><br>"
						llm_response += temp_llm_response
						
						
		type_mapping = {
			"infra": {
				"types": ["infra development analysis", "event infra development analysis"],
				"fields": ["infra_type", "location_name", "activity_date", "coordinates", "description"],
				"line_format": "infra_type:{infra_type}, location_name:{location_name}, date: {activity_date}, coordinates: {coordinates}, description:{description}",
			},
			"training": {
				"types": ["training areas analysis", "event training areas analysis"],
				"fields": ["enemy_formation_name", "location_name", "description"],
				"line_format": "enemy_formation_name:{enemy_formation_name}, location_name:{location_name}, description:{description}",
			},
			"general": {
				"types": ["general area analysis", "event general area analysis"],
				"fields": ["location_name", "coordinates", "description"],
				"line_format": "location_name:{location_name}, coordinates:{coordinates}, description:{description}",
			},
			"force": {
				"types": ["force disposition analysis", "event force disposition analysis"],
				"fields": ["location_name", "coordinates", "base_location_name", "base_coordinates", "enemy_formation_name",
				           "orbate_title", "description"],
				"line_format": "location_name:{location_name}, coordinates:{coordinates}, base_location_name:{base_location_name}, base_coordinates:{base_coordinates}, enemy_formation_name:{enemy_formation_name}, orbate_title:{orbate_title}, description:{description}",
			},
			"sitrep": {
				"types": ["pla sitrep analysis", "event pla sitrep analysis"],
				"fields": ["pass_name", "transgression_sighting_type", "sub_activity_type", "description"],
				"line_format": "pass_name:{pass_name},transgression_sighting_type:{transgression_sighting_type}, sub_activity_type:{sub_activity_type}, description:{description}",
			},
			"air_aspects": {
				"types": ["air aspects analysis", "event air aspects analysis"],
				"fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name", "equipement_type",
				           "count", "airfield_type"],
				"line_format": "location_name:{location_name}, coordinates:{coordinates}, infra_name:{infra_name}, infra_type:{infra_type}, equipment_name:{equipment_name}, equipment_name:{equipment_name}, count:{count}, airfield_type:{airfield_type}"
			},
			"sam_deployment_analysis": {
				"types": ["sam deployment analysis", "event sam deployment analysis"],
				"fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name", "equipment_type",
				           "count"],
				"line_format": "location_name:{location_name}, coordinates:{coordinates}, infra_name:{infra_name}, infra_type:{infra_type}, equipment_name:{equipment_name}, equipment_type:{equipment_type}, count:{count}"
			},
			"mobile_interception": {
				"types": ["mobile interception analysis", "Mobile Interception Analysis"],
				"fields": ["start_location_name", "end_location_name", "opposite_to", "mobile_no", "description"],
				"line_format": "start_location_name:{start_location_name}, end_location_name:{end_location_name}, opposite_to:{opposite_to}, mobile_no:{mobile_no}, description:{description}"
        	},
			"internal_security": {
				"types": ["internal security analysis", "Internal Security Analysis"],
				"fields": ["coordinates", "terrorist_casualties_", "security_forces_casualties", "civilian_casualties", "description", "army_name", "force_type_name", "formation_type", "enemy_formation_name", "command_name", "command_coordinates", "comd_tps_loc_name", "comd_tps_coordinates", "terrorists_casualties"],
				"line_format": "coordinates:{coordinates}, terrorist_casualties_:{terrorist_casualties_}, security_forces_casualties:{security_forces_casualties}, civilian_casualties:{civilian_casualties}, description:{description}, army_name:{army_name}, force_type_name:{force_type_name}, formation_type:{formation_type}, enemy_formation_name:{enemy_formation_name}, command_name:{command_name}, command_coordinates:{command_coordinates}, comd_tps_loc_name:{comd_tps_loc_name}, comd_tps_coordinates:{comd_tps_coordinates}, terrorists_casualties:{terrorists_casualties}"
			},
			"elint": {
				"types": ["elint analysis", "Elint Analysis"],
				"fields": ["description", "location_name", "coordinates", "category", "radar_type", "radar_name"],
				"line_format": "description:{description}, location_name:{location_name}, coordinates:{coordinates}, category:{category}, radar_type:{radar_type}, radar_name:{radar_name}"
			},
			"visit": {
				"types": ["visit analysis", "Visit Analysis"],
				"fields": ["description", "visit_name", "purpose", "location_name", "coordinates"],
				"line_format": "description:{description}, visit_name:{visit_name}, purpose:{purpose}, location_name:{location_name}, coordinates:{coordinates}"
			}
		}
		for key, value in type_mapping.items():
			if search_form_type in value["types"]:
				print(f"[INFO] for {key}", ai_analysis_summary_id)
				data_lines = []
				for hit in e_hits:
					source = hit["_source"]
					line = value["line_format"].format(
						**{field: source.get(field, "") for field in value["fields"]}
					)
					if line not in data_lines:
						data_lines.append(line)

				if data_lines:
					print(f"[INFO] Creating LLM prompt for {key} with {len(data_lines)} data lines")
					data_string = "\n\n".join(data_lines)
					llm_prompt = data_string + "\n\n" + model_prompt_dict.get("prompt")
					extra_prompt = "\n\n(PLEASE NOTE: The response you will give should be in html(hyper text markup language). Any title in the response should be in h2 tag and paragraph should be in p tag.)"

					llm_prompt += extra_prompt
					
					

					model_name = model_prompt_dict.get("model_info")
					base_prompt = model_prompt_dict.get("prompt")

					extra_prompt = "\n\n(PLEASE NOTE: The response you will give should be in html(hyper text markup language). Any title in the response should be in h2 tag and paragraph should be in p tag.)"

					full_prompt = data_string + "\n\n" + base_prompt + extra_prompt

					words, total_tokens = count_words_and_tokens(llm_prompt)
					print("[INFO] Words:", words)
					print("[INFO] Approx Tokens:", total_tokens)

					# -------------------------
					# Case 1: Safe Size
					# -------------------------
					if total_tokens < chunk_token_threshold:
						print("[DEBUG] Model used:", model_name)
						temp_llm_response = get_llm_response(full_prompt, model_name)

					# -------------------------
					# Case 2: Large Input → Chunk
					# -------------------------
					else:
						print("[INFO] Large input detected → Using hierarchical chunking")

						chunks = build_chunks_from_lines(data_lines, max_tokens=chunk_size)
						print(f"[INFO] Processing {len(chunks)} chunks...")

						chunk_summaries = []
						chunk_times = []

						chunk_loop_start = time.time()

						for idx, chunk in enumerate(tqdm(chunks, desc="LLM Chunk Processing", unit="chunk")):

							single_chunk_start = time.time()

							chunk_prompt = chunk + "\n\n" + base_prompt + extra_prompt

							summary = get_llm_response(chunk_prompt, model_name, max_tokens=chunk_output_tokens)

							chunk_summaries.append(summary)

							chunk_time = time.time() - single_chunk_start
							chunk_times.append(chunk_time)

							avg_time = sum(chunk_times) / len(chunk_times)
							remaining = len(chunks) - (idx + 1)
							eta = avg_time * remaining

							print(
								f"[CHUNK {idx+1}/{len(chunks)}] "
								f"Time: {chunk_time:.2f}s | "
								f"Avg: {avg_time:.2f}s | "
								f"ETA: {eta/60:.2f}m"
							)

						total_chunk_time = time.time() - chunk_loop_start
						print(f"[INFO] All chunks completed in {total_chunk_time/60:.2f} minutes")

						# Final consolidation
						combined_text = "\n\n".join(chunk_summaries)

						final_prompt = combined_text + "\n\n" + base_prompt + extra_prompt

						temp_llm_response = get_llm_response(final_prompt, model_name, max_tokens=consolidation_output_tokens)

					temp_llm_response = f"<div>{temp_llm_response}</div><br>"
					llm_response += temp_llm_response
									
		print(f"[DONE] ai analysis complete [{ai_analysis_summary_id}] [time={(time.time() - ai_summ_start_time):.2f} s] ")
	
	# with open("llm_response.html", "w", encoding="utf-8") as f:
	# 	f.write(llm_response)
	# with open("ai_change_repsonse.html", "w", encoding="utf-8") as f:
	# 	f.write(ai_change_repsonse)
	# with open("ai_trend_response.html", "w", encoding="utf-8") as f:
	# 	f.write(ai_trend_response)
	tabbed_html = build_tabbed_html([llm_response, ai_trend_response, ai_change_repsonse, ])
	# with open("final_report.html", "w", encoding="utf-8") as f:
	# 	f.write(tabbed_html)

	ai_analysis_text_base64 = base64.b64encode(
						tabbed_html.encode("utf-8")
					).decode("utf-8")

	update_ai_analysis_summary_query_text(
						cursor, ai_analysis_summary_id, ai_analysis_text_base64
					)


def imint_ai_analysis_check(row, cursor):
	elastic_query, imint_ai_analysis_id = row
	model_prompt_dict = get_prompt_and_model(cursor, imint_ai_analysis_id, imint=True)
	print("[INFO] for imint ai analysis", imint_ai_analysis_id)
	if elastic_query:
		comments = []
		e_response = get_data_from_elastic(elastic_query)
		if e_response and e_response['hits']['total']['value'] > 0:
			imint_results = e_response['hits']['hits']
			for single_json in imint_results:
				single_json = single_json['_source']
				if single_json.get("comments"):
					location_name = single_json.get("location_name")
					activity_date = single_json.get("activity_date", "")
					print("['comments']", single_json.get("comments")[-1], location_name, activity_date)
					complete_single_comment = "location_name: " + location_name + ", date:" + str(
						activity_date) + ', comments: ' + single_json['comments'][-1].strip().replace('\n', '. ')

					if complete_single_comment not in comments:
						comments.append(complete_single_comment)
		comments = '\n\n'.join(comments)
		# print("~~~~~~~````comments````````",comments)
		ai_analysis_text_base64 = ''
		if comments:
			llm_prompt = comments + "\n\n" + model_prompt_dict.get("prompt") + "The response you will give should be in html(hyper text markup language). Any title in the response should be in h2 tag and paragraph should be in p tag."
			llm_response = get_llm_response(llm_prompt, model_prompt_dict.get("model_info"))
			corrected_result = correct_json_format(str(llm_response))
			ai_analysis_text = corrected_result.get("summary", "")
			# print("ai_analysis_text: ",ai_analysis_text)
			print("[DONE] imint_ai_analysis_id: ", imint_ai_analysis_id)
			ai_analysis_text_base64 = str(base64.b64encode(str(ai_analysis_text).encode('utf-8')).decode('utf-8'))
		update_imint_ai_analysis(cursor, imint_ai_analysis_id, ai_analysis_text_base64)


def get_prompt_and_model(cursor, primary_key, imint=False):
	"""
		Get Prompt and Model - ADITYA
	"""

	output_dict = {}
	table_name = "imint_ai_analysis" if imint else "ai_analysis_summary"

	query = f"SELECT prompt, prompt_id, model_id FROM {table_name} WHERE {table_name}_id"
	query = query + " = %s;"
	cursor.execute(query, (primary_key,))
	result = cursor.fetchone()

	if result:
		prompt, prompt_id, model_id = result
	else:
		print("No result found for the provided id {}".format(primary_key))
		raise Exception("No result found for the provided id {}".format(primary_key))

	if prompt:
		output_dict["prompt"] = prompt
	elif prompt_id:
		prompt_subquery = "SELECT prompt FROM prompt_master WHERE prompt_id = %s;"
		cursor.execute(prompt_subquery, (prompt_id,))
		result = cursor.fetchone()
		if result:
			output_dict["prompt"] = result[0]
		else:
			raise Exception("Prompt not found for prompt_id {}! Please fix in the database!".format(prompt_id))
	else:
		raise Exception("Prompt ID is None! Please fix in the database!")

	if model_id:
		model_subquery = "SELECT ai_model_info FROM ai_model_master WHERE ai_model_id = %s;"
		cursor.execute(model_subquery, (model_id,))
		result = cursor.fetchone()
		if result:
			output_dict["model_info"] = result[0]
		else:
			raise Exception("Model Info not found for pmodel_id {}! Please fix in the database!".format(model_id))
	else:
		raise Exception("Model ID is None! Please fix in the database!")
	return output_dict


# recursive method to find out lt from the query.
def find_lt_value(data):
	if isinstance(data, dict):
		for key, value in data.items():
			if key == 'lt':
				try:
					return datetime.fromisoformat(value).strftime('%Y-%m-%d')
				except ValueError:
					continue
			else:
				result = find_lt_value(value)
				if result:
					return result
	elif isinstance(data, list):
		for item in data:
			result = find_lt_value(item)
			if result:
				return result
	return None

def extract_date_range(data):
    start = None
    end = None

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "gte":
                start = value
            elif key == "lt":
                end = value
            else:
                s, e = extract_date_range(value)
                start = start or s
                end = end or e

    elif isinstance(data, list):
        for item in data:
            s, e = extract_date_range(item)
            start = start or s
            end = end or e

    return start, end

### new changes ###
def ai_analysis_summary_check_threadsafe(row, ai_trends, ai_summ, ai_change, ai_change_previous,
                                        chunk_token_threshold=3000, chunk_size=2200,
                                        chunk_output_tokens=800, consolidation_output_tokens=1500):
	"""Run ai_analysis_summary_check in a thread-safe way with a new DB connection."""
	try:
		conn = postgres_connection()
		with conn.cursor() as cursor:
			# Unpack tuple into expected values
			filter_json, ai_analysis_summary_id, search_form_type = row
			ai_analysis_summary_check((filter_json, ai_analysis_summary_id, search_form_type), ai_trends, ai_summ,
			                         ai_change, ai_change_previous, cursor, chunk_token_threshold,
			                         chunk_size, chunk_output_tokens, consolidation_output_tokens)
		conn.commit()
		conn.close()
	except Exception as e:
		print(f"[ERROR] Error processing row {row}: {e}")
		traceback.print_exc()


def imint_ai_analysis_summary_check_threadsafe(row):
	"""Run imint_ai_analysis_check in a thread-safe way with a new DB connection."""
	try:
		conn = postgres_connection()
		with conn.cursor() as cursor:
			# Unpack tuple into expected values
			filter_json, imint_ai_analysis_id = row
			imint_ai_analysis_check((filter_json, imint_ai_analysis_id), cursor)
		conn.commit()
		conn.close()
	except Exception as e:
		print(f"[ERROR] Error processing row {row}: {e}")
		traceback.print_exc()


def lock_rows_for_processing_1(conn):
	"""Lock ai_analysis_summary rows for processing."""
	try:
		with conn.cursor() as cursor:
			cursor.execute("""
                UPDATE public.ai_analysis_summary
                SET query_status = 2
                WHERE ai_analysis_summary_id IN (
                    SELECT ai_analysis_summary_id
                    FROM public.ai_analysis_summary
                    WHERE query_status = 0
                    ORDER BY ai_analysis_summary_id DESC
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING filter_json, ai_analysis_summary_id, search_form_type;
            """)
			return cursor.fetchall()
	except psycopg2.ProgrammingError as e:
		print(f"[ERROR] ProgrammingError in lock_rows_for_processing_1: {e}")
		return []


def lock_rows_for_processing_2(conn):
	"""Lock imint_ai_analysis rows for processing."""
	try:
		with conn.cursor() as cursor:
			cursor.execute("""
                UPDATE public.imint_ai_analysis
                SET status = 2
                WHERE imint_ai_analysis_id IN (
                    SELECT imint_ai_analysis_id
                    FROM public.imint_ai_analysis
                    WHERE status = 0
                    ORDER BY imint_ai_analysis_id DESC
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING filter_json, imint_ai_analysis_id;
            """)
			return cursor.fetchall()
	except psycopg2.ProgrammingError as e:
		print(f"[ERROR] ProgrammingError in lock_rows_for_processing_2: {e}")
		return []


def ensure_connection(conn):
	"""Ensure the PostgreSQL connection is alive."""
	try:
		if conn.closed != 0:
			print("[INFO] PostgreSQL connection closed. Reconnecting...")
			return postgres_connection()
		else:
			with conn.cursor() as cursor:
				cursor.execute("SELECT 1;")
		return conn
	except Exception:
		print("[ERROR] Connection check failed. Reconnecting...")
		traceback.print_exc()
		return postgres_connection()


def run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
                                 chunk_token_threshold=3000, chunk_size=2200,
                                 chunk_output_tokens=800, consolidation_output_tokens=1500):
	"""Main loop to poll and process AI analysis summaries."""
	conn = postgres_connection()

	while True:
		try:
			conn = ensure_connection(conn)
			ai_rows = lock_rows_for_processing_1(conn)
			conn.commit()

			conn = ensure_connection(conn)
			imint_rows = lock_rows_for_processing_2(conn)
			conn.commit()

			if ai_rows:
				task_categories = []
				if ai_summ:
					task_categories.append('ai_summary')
				if ai_trends:
					task_categories.append('ai_trends')
				if ai_change:
					task_categories.append('ai_trends')
				if ai_change_previous:
					task_categories.append(f'ai_change_previous({ai_change_previous}y)')
				print(f"[INFO] Found {len(ai_rows)} new rows to process for {', '.join(task_categories)}")
				for row in ai_rows:
					p = multiprocessing.Process(target=ai_analysis_summary_check_threadsafe,
					                    args=(row, ai_trends, ai_summ, ai_change, ai_change_previous,
					                           chunk_token_threshold, chunk_size,
					                           chunk_output_tokens, consolidation_output_tokens))
					p.daemon = True
					p.start()
			# else:
			#     print("[INFO] No new summaries found for ai_analysis")

			if imint_rows:
				print(f"[INFO] Found {len(imint_rows)} new rows to process for imint_ai_analysis")
				for irow in imint_rows:
					p = multiprocessing.Process(target=imint_ai_analysis_summary_check_threadsafe, args=(irow,))
					p.daemon = True
					p.start()
		# else:
		#     print("[INFO] No new summaries found for imint_ai_analysis")

		except Exception as e:
			print("[ERROR] Exception in main loop:")
			traceback.print_exc()

		time.sleep(poll_interval)
	if conn:
		conn.close()



def call_main_func():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ai_summary', action='store_true', help='Enable AI summary mode')
	parser.add_argument('--ai_trends', action='store_true', help='Enable AI trends mode')
	parser.add_argument('--ai_change', action='store_true', help='Enable AI change mode')
	parser.add_argument('--ai_change_previous', type=int, metavar='N', help='Compare selected period with same period N years ago (e.g. --ai_change_previous 3)')
	parser.add_argument('--chunk_token_threshold', type=int, default=3000, metavar='N',
	                    help='Token threshold for chunking; controls all internal token limits (default: 3000)')
	args = parser.parse_args()
	ai_summ = False
	ai_trends = False
	ai_change = False
	ai_change_previous = None

	poll_interval = 5
	if args.ai_summary:
		ai_summ = True
		print("AI summary mode is enabled.")
	else:
		print("AI summary mode is disabled.")

	if args.ai_trends:
		ai_trends = True
		print("AI trends mode is enabled.")
	else:
		print("AI trends mode is disabled.")

	if args.ai_change:
		ai_change = True
		print("AI change mode is enabled.")
	else:
		print("AI change mode is disabled.")

	if args.ai_change_previous is not None:
		ai_change_previous = args.ai_change_previous
		print(f"AI previous-year change detection mode enabled (comparing with {ai_change_previous} year(s) ago).")
	else:
		print("AI previous-year change detection mode disabled.")
	
	chunk_token_threshold = args.chunk_token_threshold

	# Derive internal token limits from the master threshold
	chunk_size = chunk_token_threshold * 3 // 4           # per-chunk input size
	chunk_output_tokens = chunk_token_threshold // 4       # per-chunk output limit
	consolidation_output_tokens = chunk_token_threshold // 2  # final consolidation output

	print(f"Chunk token threshold set to: {chunk_token_threshold}")

	if not ai_summ and not ai_trends and not ai_change and not ai_change_previous:
		sys.exit(1)
	else:
		run_ai_analysis_summary_loop(ai_trends, ai_summ, ai_change, ai_change_previous, poll_interval,
		                             chunk_token_threshold, chunk_size,
		                             chunk_output_tokens, consolidation_output_tokens)


if __name__ == '__main__':
	while True:
		call_main_func()