import logging
import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
from elasticsearch.helpers import scan
from pprint import pprint
import ast
from io import StringIO
from tqdm import tqdm
import json
import os
import importlib.util
import requests

# Setting up the logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

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
CURRENT_INDEX_NAME = constants.CUREENT_INDEX_NAME

# constants values -> will keep in a different file later on.
# ELASTIC_HOST = "192.168.1.103"  # Demo
# ELASTIC_HOST = "192.168.1.16"
# ELASTIC_PORT = 9200
# ELASTIC_CLIENT_SCHEME = "http"
# # CURRENT_INDEX_NAME = "ifc_demo_v22"  # Demo
# CURRENT_INDEX_NAME = "fatboy_ifc", #ifc_demo_103
# LLM_IP = "192.168.1.125"
# LLM_PORT = 11434

es = Elasticsearch(
    [{"host": ELASTIC_HOST, "port": ELASTIC_PORT, "scheme": ELASTIC_CLIENT_SCHEME}],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False
)

# SITREP Important Fields
NON_NUMERIC_FIELDS = ['pass_name', 'sub_activity_type', 'transgression_sighting_type', 'individual_category']
NUMERIC_FIELDS = ["pla_soldiers", "pla_officers", "pla_jco", "enemy_count"]
FIELDS_COMBINATION = [
	['pass_name', 'sub_activity_type'],
	['pass_name', 'transgression_sighting_type'],
	['pass_name', 'pla_soldiers'],
	['pass_name', 'pla_officers'],
	['pass_name', 'pla_jco'],
	['pass_name', 'individual_category', 'enemy_count'],
	['sub_activity_type'],
	['transgression_sighting_type'],
	['individual_category','pla_soldiers'],
	['individual_category','pla_officers'],
	['individual_category','pla_jco'],
	['individual_category', 'enemy_count']
]
OTHER_FIELDS = ["Year", "Month"]
RENAME_DICT = {
	'pass_name': 'Pass Name',
	'sub_activity_type': 'Sub Activity Type',
	'transgression_sighting_type': 'Transgression Sighting Type',
	'individual_category': 'Individual Category',
	'pla_soldiers': 'Total PLA Soldiers',
	'pla_officers': 'Total PLA Officers',
	'pla_jco': 'Total PLA JCO Count',
	'enemy_count': 'Total Enemy Count'
}


import ast

# Reverse RENAME_DICT so: { original_key: new_key }
REVERSE_RENAME_DICT = {v: k for k, v in RENAME_DICT.items()}

def get_llm_response(query, llm_model):
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
	while True:
		r = requests.post(anythingllm_url, headers=headers, json=payload)
		if r.status_code == 200:
			result = r.json()
			if 'textResponse' in result:
				return result['textResponse']

def get_base_records(query):
	all_docs = scan(
		client=es,
		index=CURRENT_INDEX_NAME,
		query=query,
		scroll="2m",
		size=1000  # Batch size
	)
	docs = [doc['_source'] for doc in all_docs]
	df = pd.json_normalize(docs, sep='.')
	return df

def verify_fields(df, NON_NUMERIC_FIELDS, NUMERIC_FIELDS):
	NON_NUMERIC_FIELDS_DEFAULT = ''
	for col in NON_NUMERIC_FIELDS:
		if col not in df.columns:
			df[col] = NON_NUMERIC_FIELDS_DEFAULT

	NUMERIC_FIELDS_DEFAULT = 0
	for col in NUMERIC_FIELDS:
		if col not in df.columns:
			df[col] = NUMERIC_FIELDS_DEFAULT

	return df

def get_filters(base_records):
	current_data = {}
	filters = {}
	for combination in FIELDS_COMBINATION:
		used_columns = [col for col in combination if col not in NUMERIC_FIELDS]

		distinct_rows_x = base_records.drop_duplicates(subset=used_columns)
		# --------------------------
		distinct_rows = distinct_rows_x.copy()

		distinct_rows = distinct_rows[['@timestamp'] + used_columns]
		if "@timestamp" in distinct_rows.columns:
			distinct_rows["@timestamp"] = pd.to_datetime(distinct_rows["@timestamp"], errors="coerce")
			distinct_rows["Year"] = distinct_rows["@timestamp"].dt.year.astype(str)
			distinct_rows["Month"] = distinct_rows["@timestamp"].dt.strftime("%B")
		else:
			distinct_rows["Year"], distinct_rows["Month"] = "", ""
		numeric_fields_exists = False

		for field in NUMERIC_FIELDS:
			if field in distinct_rows.columns:
				numeric_fields_exists = True
				distinct_rows[field] = pd.to_numeric(distinct_rows[field], errors="coerce").fillna(0).astype(int)
			else:
				distinct_rows[field] = 0

		agg_dict = {num_field: "sum" for num_field in NUMERIC_FIELDS}
		agg_dict["@timestamp"] = "count"

		group_by_fields = [each for each in used_columns if each in distinct_rows.columns]
		grouped_df = distinct_rows.groupby(
			group_by_fields + ["Year", "Month"], as_index=False
		).agg(agg_dict)
		grouped_df.rename(columns={"@timestamp": "Count"}, inplace=True)

		if not numeric_fields_exists:
			grouped_df = grouped_df.drop(columns=[col for col in NUMERIC_FIELDS if col in grouped_df.columns])
		current_data[str(combination)] = grouped_df

		result = distinct_rows_x[used_columns].to_dict(orient='records')
		result = [
			{k: ('' if pd.isna(v) else v) for k, v in row.items()}
			for row in result
		]
		filters[str(combination)] = result
	return current_data, filters

def build_dynamic_elastic_query(timestamp_column, start_date, end_date, filters):
	should_clauses = []

	for f in filters:
		must_clauses = []
		for field, value in f.items():
			if value:  # Ignore empty strings or None
				must_clauses.append({
					"match_phrase": {field: value}
				})
		if must_clauses:
			should_clauses.append({"bool": {"must": must_clauses}})

	query = {
		"bool": {
			"filter": [
				{
					"range": {
						timestamp_column: {
							"gte": start_date,
							"lte": end_date,
							"format": "strict_date_optional_time"
						}
					}
				}
			],
			"should": should_clauses,
			"minimum_should_match": 1 if should_clauses else 0
		}
	}

	return {
		"query": query
	}

def get_historical_records(timestamp_field, start_date, end_date, filters_list):
	query = build_dynamic_elastic_query(timestamp_field, start_date, end_date, filters_list)

	all_docs = scan(
		client=es,
		index=CURRENT_INDEX_NAME,
		query=query,
		scroll="2m",
		size=1000
	)
	docs = [doc['_source'] for doc in all_docs]
	df = pd.json_normalize(docs, sep='.')
	return df


def historical_search(timestamp_field, start_date, end_date, filters):
	result = {}
	for combination, filter_list in tqdm(filters.items()):
		combination_list = ast.literal_eval(combination)

		combination_list = [timestamp_field] + combination_list

		merged_df = get_historical_records(timestamp_field, start_date, end_date, filter_list)

		if not merged_df.empty:

			merged_df.sort_values(by='@timestamp', ascending=False, inplace=True)
			missing_cols = [col for col in combination_list if col not in merged_df.columns]
			if missing_cols:
				print(f"Skipping combination due to missing columns: {missing_cols}")
				continue
			merged_df = merged_df[combination_list]
			if "@timestamp" in merged_df.columns:
				merged_df["@timestamp"] = pd.to_datetime(merged_df["@timestamp"], errors="coerce")
				merged_df["Year"] = merged_df["@timestamp"].dt.year.astype(str)
				merged_df["Month"] = merged_df["@timestamp"].dt.strftime("%B")
			else:
				merged_df["Year"], merged_df["Month"] = "", ""

			numeric_fields_exists = False

			for field in NUMERIC_FIELDS:
				if field in merged_df.columns:
					numeric_fields_exists = True
					merged_df[field] = pd.to_numeric(merged_df[field], errors="coerce").fillna(0).astype(int)
				else:
					merged_df[field] = 0

			agg_dict = {num_field: "sum" for num_field in NUMERIC_FIELDS}
			agg_dict["@timestamp"] = "count"

			group_by_fields = [each for each in combination_list if each in merged_df.columns]
			grouped_df = merged_df.groupby(
				group_by_fields + ["Year", "Month"], as_index=False
			).agg(agg_dict)
			grouped_df.rename(columns={"@timestamp": "Count"}, inplace=True)

			if not numeric_fields_exists:
				grouped_df = grouped_df.drop(columns=[col for col in NUMERIC_FIELDS if col in grouped_df.columns])

			if 'year' in grouped_df.columns:
				grouped_df['year'] = grouped_df['year'].astype(str)

			# Identify string columns
			string_columns = grouped_df.select_dtypes(include='object').columns.tolist()

			# Identify integer columns
			int_columns = grouped_df.select_dtypes(include='int').columns.tolist()

			# Group by string columns and sum integer columns
			ndf = grouped_df.groupby(string_columns, as_index=False)[int_columns].sum()
			# print(ndf)
			result[combination] = ndf
		else:
			pass
			# print(f'total 0 rows found')
		# print('-'*50)
	return result

def merge_both_current_and_history(records):
    merged = {}

    for record in records:
        key = tuple(
            (k, json.dumps(record[k], sort_keys=True) if isinstance(record[k], dict) else record[k])
            for k in record if k not in ['Present Data', 'Past Data']
        )

        if key not in merged:
            merged[key] = {
                **{k: v for k, v in record.items() if k not in ['Present Data', 'Past Data']},
                'Present Data': {},
                'Past Data': {}
            }

        merged[key]['Present Data'].update(record.get('Present Data', {}))
        merged[key]['Past Data'].update(record.get('Past Data', {}))

    return list(merged.values())


def detect_trend(current_df, history_df):
	if 'Count' in current_df.columns:
		current_df['Count'] = current_df['Count'].astype(int)

	if 'Count' in history_df.columns:
		history_df['Count'] = history_df['Count'].astype(int)

	values_to_check = ['unknown', 'Unknown', 'UNKNOWN']
	time_columns = OTHER_FIELDS

	col_str = [col for col in current_df.select_dtypes(include='object').columns if col not in time_columns]
	col_int = [col for col in current_df.select_dtypes(include='int').columns if col not in time_columns]

	# Use first integer column as the value source instead of 'Count', if available
	int_column_name = col_int[0] if len(col_int) > 0 else 'Count'

	result = []

	for _, curr_row in current_df.iterrows():
		# Build match condition using string columns
		match_condition = pd.Series([True] * len(history_df))

		for col in col_str:
			match_condition &= (history_df[col] == curr_row[col])

		match = history_df[match_condition]

		current_data = {
			f"{curr_row['Month']} {curr_row['Year']}": curr_row[int_column_name]
		}

		if not match.empty:
			# Sum all int fields from history (optional, not used in this line-by-line version)
			summed_int_fields = {col: match[col].sum() for col in col_int}

			for _, hist_row in match.iterrows():
				history_data = {
					f"{hist_row['Month']} {hist_row['Year']}": hist_row[int_column_name]
				}

				temp_result = {
					**{col: curr_row[col] for col in col_str},
					**summed_int_fields,
					'current_data': current_data,
					'history_data': history_data
				}

				if not any(val in temp_result.values() for val in values_to_check):
					rename_temp_result = {RENAME_DICT.get(k, k): v for k, v in temp_result.items()}
					if 'Count' in rename_temp_result:
						del rename_temp_result['Count']
					result.append(rename_temp_result)

	result = merge_both_current_and_history(result)
	return result

def rename_nested_data(data):
    renamed_result = {}

    for raw_key, records in data.items():
        try:
            fields = ast.literal_eval(raw_key)  # parse list-like string keys
        except Exception:
            fields = [raw_key]

        new_key = ", ".join(fields)  # keep the key unchanged or simplified if you want

        renamed_records = []
        for record in records:
            renamed_record = {}
            for old_field, value in record.items():
                new_field = REVERSE_RENAME_DICT.get(old_field, old_field)
                renamed_record[new_field] = value
            renamed_records.append(renamed_record)

        renamed_result[new_key] = renamed_records

    return renamed_result

def llm_analysis(historical_combinations_data, current_combinations_data):
	trend_data = {}
	for combination in current_combinations_data:
		cat_key = combination
		if combination in historical_combinations_data:
			curr_df = current_combinations_data[combination]
			hist_df = historical_combinations_data[combination]
			trend_result = detect_trend(curr_df, hist_df)
			trend_data[cat_key] = trend_result

	trend_data = rename_nested_data(trend_data)

	return trend_data


def json_to_prompt(json_data, category):
	prompts = []

	for raw_combination in json_data:
		# Safely convert the raw_combination into a list
		if isinstance(raw_combination, str):
			try:
				# Try parsing string as a Python literal
				combination_list = ast.literal_eval(raw_combination)
				if not isinstance(combination_list, (list, tuple)):
					raise ValueError
			except Exception:
				# Fallback: treat as comma-separated
				combination_list = [item.strip() for item in raw_combination.split(',')]
		else:
			combination_list = list(raw_combination)

		# Rename for readability
		combination = [RENAME_DICT.get(item, item) for item in combination_list]
		prompt = ''
		if 'Pass Name' in combination and 'Sub Activity Type' in combination:
			prompt = f"You are a defense analyst. There are {category} activities occurring at various passes with different sub-activity types. The data includes monthly trends of such activities. Please analyze the JSON data and summarize the key trends."

		elif 'Pass Name' in combination and 'Transgression Sighting Type' in combination:
			prompt = f"You are a defense analyst. The {category} activities include different transgression or sighting types across multiple passes. Analyze the trends and summarize key insights based on the data provided."

		elif 'Pass Name' in combination and 'Total PLA Soldiers' in combination:
			prompt = f"You are a defense analyst. The dataset contains {category} activities involving PLA soldiers at various passes. Analyze the monthly trends and share a summary of key patterns."

		elif 'Pass Name' in combination and 'Total PLA Officers' in combination:
			prompt = f"You are a defense analyst. The {category} data includes PLA officer presence at different passes. Analyze this trend and provide a summary of your observations."

		elif 'Pass Name' in combination and 'Total PLA JCO Count' in combination:
			prompt = f"You are a defense analyst. There are {category} activities involving PLA JCOs reported at various passes. Please analyze the monthly data and summarize trends."

		elif 'Pass Name' in combination and 'Individual Category' in combination and 'Total Enemy Count' in combination:
			prompt = f"You are a defense analyst. {category} activities are reported at various passes, categorized by individual types and enemy counts. Please analyze the data and provide a trend summary."

		elif 'Sub Activity Type' in combination:
			prompt = f"You are a defense analyst. The data contains monthly trends of {category} activities grouped by sub-activity type. Analyze and summarize the key patterns."

		elif 'Transgression Sighting Type' in combination:
			prompt = f"You are a defense analyst. The {category} data includes various transgression/sighting types. Please analyze the trends and summarize your findings."

		elif 'Individual Category' in combination and 'Total PLA Soldiers' in combination:
			prompt = f"You are a defense analyst. The data includes {category} activities categorized by individual type and PLA soldier count. Please analyze the trends and summarize insights."

		elif 'Individual Category' in combination and 'Total PLA Officers' in combination:
			prompt = f"You are a defense analyst. {category} activities involve different individual categories and PLA officer counts. Analyze the trends and provide a summary."

		elif 'Individual Category' in combination and 'Total PLA JCO Count' in combination:
			prompt = f"You are a defense analyst. The data includes JCO-related {category} activities across individual categories. Analyze the monthly trends and summarize."

		elif 'Individual Category' in combination and 'Total Enemy Count' in combination:
			prompt = f"You are a defense analyst. {category} activities are grouped by individual type and enemy count. Please analyze the trends and share your summary."

		else:
			# Fallback for unmatched combinations
			prompt = f"You are a defense analyst. This dataset contains monthly trends of {category} activities grouped by: {', '.join(combination)}. Please analyze and summarize the trends."

		prompt += f" Below is the list of JSON data: {json.dumps(json_data[raw_combination])}"
		prompts.append(prompt)

	return prompts

def remove_dotted_and_count_fields(data):
    cleaned = {}

    for key, records in data.items():
        key_fields = key.split(", ")

        # Skip entire key if any part ends with `.count` or contains a `.`
        if any(fld.endswith(".count") or '.' in fld for fld in key_fields):
            continue

        cleaned_records = []
        for record in records:
            # Remove keys that have dots or end with `.count`
            new_record = {
                k: v for k, v in record.items()
                if '.' not in k and not k.endswith(".count")
            }
            cleaned_records.append(new_record)

        cleaned[key] = cleaned_records

    return cleaned

def _build_uniform_html(responses: list[str]) -> str:
    """
    Return only the inner HTML (body content) for the TRAINING report.
    Each response is wrapped in a numbered, styled box:
        • white background, rounded corners, subtle shadow
        • left accent bar that cycles through 6 colours
        • heading “Trends #<idx>”
    """
    colours = ["#0d6efd", "#198754", "#fd7e14", "#6f42c1", "#d63384", "#20c997"]
    parts = []

    for idx, frag in enumerate(responses, 1):
        colour = colours[(idx - 1) % len(colours)]
        parts.append(
            f"""
            <div class="trend-box" style="
                border-left: 5px solid {colour};
                background:#fff;
                border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,.08);
                padding:25px 30px 30px;
                margin-bottom:30px;
            ">
                <h3 style="margin:0 0 15px;font-size:1.3rem;color:{colour};">
                    Trends #{idx}
                </h3>
                {frag}
            </div>
            """
        )
    return "\n".join(parts)

def sitrep_main(start_date, query):
	base_records = get_base_records(query)
	base_records = verify_fields(base_records, NON_NUMERIC_FIELDS, NUMERIC_FIELDS)
	base_records = base_records[
		[col for col in ['@timestamp'] + NON_NUMERIC_FIELDS + NUMERIC_FIELDS if col in base_records.columns]
	]
	current_combinations_data, filters = get_filters(base_records)
	end_date = datetime.strptime(start_date, "%Y-%m-%d")
	start_date = end_date - timedelta(days=365)
	historical_combinations_data = historical_search('@timestamp', start_date, end_date, filters)
	result = llm_analysis(historical_combinations_data, current_combinations_data)
	prompts = json_to_prompt(result, "Training")
	html_fragments = []
	for prompt in tqdm(prompts):
		enriched = (
			f"{prompt}\n\n"
			"Please start with <h1>Trend Analysis</h1>. "
			"Use <h2> for sub-titles and <p> for paragraphs. "
			"Return only valid HTML, no markdown."
		)
		raw = get_llm_response(enriched, LLM_MODEL)
		html_fragments.append(raw)

	return _build_uniform_html(html_fragments)

def sitrep_stats_main(start_date, query):
	base_records = get_base_records(query)
	base_records = verify_fields(base_records, NON_NUMERIC_FIELDS, NUMERIC_FIELDS)
	base_records = base_records[
		[col for col in ['@timestamp'] + NON_NUMERIC_FIELDS + NUMERIC_FIELDS if col in base_records.columns]
	]
	current_combinations_data, filters = get_filters(base_records)
	end_date = datetime.strptime(start_date, "%Y-%m-%d")
	start_date = end_date - timedelta(days=365)
	historical_combinations_data = historical_search('@timestamp', start_date, end_date, filters)
	result = llm_analysis(historical_combinations_data, current_combinations_data)
	result = remove_dotted_and_count_fields(result)
	return result


if __name__ == '__main__':
	start_date = '2025-05-04'
	query = {
	  "aggs": {
	    "unique_descriptions": {
	      "aggs": {
	        "top_hit": {
	          "top_hits": {
	            "size": 100,
	            "_source": True
	          }
	        }
	      },
	      "terms": {
	        "size": 100,
	        "field": "description_hash.keyword"
	      }
	    }
	  },
	  "sort": [
	    {
	      "@timestamp": {
	        "order": "desc"
	      }
	    }
	  ],
	  "query": {
	    "bool": {
	      "must": [
	        {
	          "match": {
	            "form_type": "patrolling"
	          }
	        },
	        {
	          "range": {
	            "activity_date": {
	              "lt": "2025-07-01T00:00:00",
	              "gte": "2025-05-04T00:00:00"
	            }
	          }
	        }
	      ],
	      "filter": [],
	      "must_not": [
	        {
	          "term": {
	            "form_status": 5
	          }
	        }
	      ]
	    }
	  }
	}
	html_report = sitrep_main(start_date, query)
	with open("sitrep_report.html", "w", encoding="utf-8") as f:
		f.write(html_report)