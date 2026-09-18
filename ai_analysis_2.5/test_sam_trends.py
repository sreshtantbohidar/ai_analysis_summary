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
NON_NUMERIC_FIELDS = ['location_name', 'enemy_formation_name', 'formation_type', 'equipment.equipment_name', "equipment.equipment_type"]
NUMERIC_FIELDS = ["equipment.count"]
FIELDS_COMBINATION = [
    ['location_name', 'enemy_formation_name'],
    ['location_name', 'equipment.equipment_name', 'equipment.count'],
    ['location_name', 'equipment.equipment_type', 'equipment.count'],   
    ['location_name', 'formation_type'],
    ['enemy_formation_name', 'equipment.equipment_name', 'equipment.count'],
    ['enemy_formation_name', 'equipment.equipment_type', 'equipment.count'],
    ['location_name','enemy_formation_name', 'equipment.equipment_name', 'equipment.count'],
    ['location_name','enemy_formation_name', 'equipment.equipment_type', 'equipment.count'],
    ['location_name', 'formation_type', 'equipment.equipment_name', 'equipment.count'],
    ['location_name', 'formation_type', 'equipment.equipment_type', 'equipment.count'],
    ['enemy_formation_name'],
    ['equipment.equipment_name'],
    ['equipment.equipment_type'],
    ['location_name'],
]

OTHER_FIELDS = ["Year", "Month"]
RENAME_DICT = {
    'location_name': 'Location of SAM',
    'enemy_formation_name': 'Name of Army Formation',
    'formation_type': 'Type of Army Formation',
    'equipment.equipment_name': 'Equipment Name',
    'equipment.equipment_type': 'Equipment Type',
    'equipment.count': 'Equipment Count',
    'current_data': 'Present Data',
    'history_data': 'Past Data'
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


def merge_both_current_and_history(records):
    merged = {}

    for record in records:
        # Use all keys except 'current_data' and 'history_data' as grouping keys
        key = tuple((k, record[k]) for k in record if k not in ['Present Data', 'Past Data'])

        if key not in merged:
            merged[key] = {
                **{k: v for k, v in record.items() if k not in ['Present Data', 'Past Data']},
                'Present Data': {},
                'Past Data': {}
            }

        # Merge current_data and history_data
        merged[key]['Present Data'].update(record.get('Present Data', {}))
        merged[key]['Past Data'].update(record.get('Past Data', {}))

    return list(merged.values())

def df_to_csv_str(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    return csv_string

def ensure_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            if col in NUMERIC_FIELDS:
                df[col] = 0
            else:
                df[col] = None
    return df

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

def llm_analysis(historical_combinations_data, current_combinations_data):
    trend_data = {}
    for combination in current_combinations_data:
        cat_key = combination
        if combination in historical_combinations_data:
            curr_df = current_combinations_data[combination]
            hist_df = historical_combinations_data[combination]
            trend_result = detect_trend(curr_df,hist_df)
            trend_data[cat_key] = trend_result
    trend_data = rename_nested_data(trend_data)
    # with open('sam.txt', 'w') as f:
    #     json.dump(trend_data, f, indent=4)
    return trend_data

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


def build_dynamic_es_query_with_date(filters, timestamp_field, start_date, end_date, match_type="term"):
    should_clauses = []

    for item in filters:
        must_clauses = []

        for field, value in item.items():
            if value is None:
                continue
            must_clauses.append({
                match_type: {
                    field: str(value).strip()
                }
            })

        if must_clauses:
            should_clauses.append({ "bool": { "must": must_clauses }})

    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": should_clauses,
                            "minimum_should_match": 1
                        }
                    },
                    {
                        "range": {
                            timestamp_field: {
                                "gte": start_date.strftime("%Y-%m-%d"),
                                "lt": end_date.strftime("%Y-%m-%d"),
                                "format": "yyyy-MM-dd"
                            }
                        }
                    }
                ]
            }
        }
    }

    return query

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

def get_filters(base_records):
    current_data = {}
    filters = {}
    for combination in FIELDS_COMBINATION:
        used_columns = [col for col in combination if col not in NUMERIC_FIELDS]
        distinct_rows_x = base_records.drop_duplicates(subset=used_columns)
        distinct_rows = distinct_rows_x.copy()

        distinct_rows = distinct_rows[['@timestamp']+used_columns]
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


def build_dynamic_es_query(filters, match_type="match"):
    """
    Builds a dynamic Elasticsearch bool/should/must query from a list of filter dictionaries.

    :param filters: List of dictionaries with field-value pairs to match.
    :param match_type: Either "match" or "term" for how field values are queried.
    :return: Elasticsearch query as a dictionary.
    """
    should_clauses = []

    for item in filters:
        must_clauses = []

        for field, value in item.items():
            # Skip null or empty fields if undesired
            if value is None:
                continue
            clause = {
                match_type: {
                    field: str(value).strip()
                }
            }
            must_clauses.append(clause)

        if must_clauses:
            should_clauses.append({
                "bool": {
                    "must": must_clauses
                }
            })

    query = {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
    }

    return query


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
            result[combination] = ndf

    return result


def json_to_prompt(json_data, category):
    # TO-DO: In my opinion, we need to think of this logic or change the order or if else, cause it may fail in some scenarios. 

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
        if 'Location of SAM' in combination and 'Name of Army Formation' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening on specific locations by various names of army formations. I have a list of JSON data which has monthly trends of {category} activities happening on different locations by name of army formations. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and  'Equipment Name' in combination and  'Equipment Count' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening using specific Equipment Name by various locations. I have a list of JSON data which has monthly trends of {category} activities happening on different locations by equipment name and equipment count. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and 'Type of Army Formation' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening on specific locations by various types of army formations. I have a list of JSON data which has monthly trends of {category} activities happening on different locations by type of army formations. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and 'Equipment Type' in combination and 'Equipment Count' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening on specific locations by various types of Equipment types. I have a list of JSON data which has monthly trends of {category} activities happening on different locations by type of equipment. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Name of Army Formation' in combination and 'Equipment Name' in combination  and 'Equipment Count' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific Army formation using specific equipment. I have a list of JSON data which has monthly trends of {category} activities happening with formation, equipment names and equipment count as well. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Name of Army Formation' in combination and 'Equipment Type' in combination  and 'Equipment Count' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific Army formation using specific equipment. I have a list of JSON data which has monthly trends of {category} activities happening with formation, equipment names and equipment count as well. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and 'Equipment Name' in combination  and 'Equipment Count' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific location using specific equipment. I have a list of JSON data which has monthly trends of {category} activities happening in various locations with specific equipments and equipment count as well. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and 'Equipment Name' in combination and 'Equipment Type' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific location performing specific types of equipment with particular equipment. I have a list of JSON data which has monthly trends of {category} activities happening in various locations with specific equipments and equipment. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination and 'Type of Army Formation' in combination and 'Equipment Type' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific location using some types of formation and equipment types. I have a list of JSON data which has monthly trends of {category} activities happening in various locations with specific army formation anf equipment type. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Name of Army Formation' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific army formation and their count of occurrences. I have a list of JSON data which has monthly trends of {category} activities happening in various army formation and their count of occurrences. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Equipment Name' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific equipment name and their count of occurrences. I have a list of JSON data which has monthly trends of {category} activities happening in various equipment and their count of occurrences. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Equipment Type' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific equipment type and their count of occurrences. I have a list of JSON data which has monthly trends of {category} activities happening in various equipment type and their count of occurrences. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
            prompts.append(prompt)

        elif 'Location of SAM' in combination:
            prompt += f"You are a defense analyst. There are {category} activities happening with specific locations and their count of occurrences. I have a list of JSON data which has monthly trends of {category} activities happening in various locations and their count of occurrences. Can you analyze the JSON data and perform trend analysis and share the summary. Please find the below list of JSON: {json.dumps(json_data[raw_combination])}"
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

def trends_sam_main(start_date, query):
    llm_response = ''
    base_records = get_base_records(query)
    base_records = verify_fields(base_records, NON_NUMERIC_FIELDS, NUMERIC_FIELDS)
    base_records = base_records[[col for col in ['@timestamp']+NON_NUMERIC_FIELDS+NUMERIC_FIELDS if col in base_records.columns]]
    current_combinations_data, filters = get_filters(base_records)
    end_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)
    historical_combinations_data = historical_search('@timestamp',start_date, end_date, filters)
    result = llm_analysis(historical_combinations_data, current_combinations_data)
    prompts = json_to_prompt(result, "SAM")
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

def trends_stats_sam_main(start_date, query):
    base_records = get_base_records(query)
    base_records = verify_fields(base_records, NON_NUMERIC_FIELDS, NUMERIC_FIELDS)
    base_records = base_records[[col for col in ['@timestamp']+NON_NUMERIC_FIELDS+NUMERIC_FIELDS if col in base_records.columns]]
    current_combinations_data, filters = get_filters(base_records)
    end_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)
    historical_combinations_data = historical_search('@timestamp',start_date, end_date, filters)
    result = llm_analysis(historical_combinations_data, current_combinations_data)
    result = remove_dotted_and_count_fields(result)
    
    return result

if __name__ == '__main__':
    query = {
      "size": 10000,
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
              "exists": {
                "field": "training_type"
              }
            },
            {
              "terms": {
                "daily_activity_type_id": [
                  383
                ]
              }
            },
            {
              "term": {
                "form_type.keyword": "training"
              }
            },
            {
              "exists": {
                "field": "location_name"
              }
            },
            {
              "range": {
                "activity_date": {
                  "lt": "2025-04-29T00:00:00",
                  "gte": "2023-04-21T00:00:00"
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
    html_report = trends_sam_main('2023-04-21', query)
    with open("sam_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)