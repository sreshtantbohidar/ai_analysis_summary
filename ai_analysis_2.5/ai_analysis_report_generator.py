#!/usr/bin/env python3
__author__  = "Sresht"
__version__ = "1.0.0"
__date__    = "2025-10-22"
"""
Stream all documents from the hard-coded index that fall inside the
given date range using the scroll API and generate the AI summary,
change detection and Trends analysis.

If --start or --end are omitted, the script uses:
   start = yesterday
   end   = today   (both inclusive, UTC)

Usage:
	python dump_by_date.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from elasticsearch import Elasticsearch, helpers
from pprint import pprint
import time
import psycopg2
import os
import importlib.util
import requests
from typing import List
import re
import base64
import psycopg2
from psycopg2 import pool, extras
import logging
import traceback

log = logging.getLogger(__name__)

prompt_dict = {}



class PostgresCursorPool:
	"""
	Keeps one real psycopg2 connection open and yields cursors from it.
	If the connection dies (OperationalError) it automatically opens
	a new one (transparent refresh).
	"""
	def __init__(self, dsn, cursor_factory=extras.RealDictCursor):
		self.dsn   = dsn
		self._conn = None
		self.cursor_factory = cursor_factory
		self._ensure_conn()

	# ---------- internal ----------
	def _ensure_conn(self):
		"""Open a new connection only if the current one is dead/None."""
		try:
			if self._conn is None or self._conn.closed:
				raise psycopg2.OperationalError('closed')
			# cheap ping
			with self._conn.cursor() as c:
				c.execute('SELECT 1')
		except (psycopg2.OperationalError, psycopg2.InterfaceError):
			log.info('Refreshing Postgres connection')
			self._conn = psycopg2.connect(self.dsn)
			self._conn.autocommit = True   # or False, your choice

	# ---------- public ----------
	def cursor(self):
		self._ensure_conn()
		return self._conn.cursor(cursor_factory=self.cursor_factory)

	def close(self):
		if self._conn is not None:
			self._conn.close()
			self._conn = None


# Path to constants.py (one folder up)
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'constants.py'))

# Load the module
spec = importlib.util.spec_from_file_location("constants", file_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

# Now manually import specific variables
PG_DB_NAME = constants.PG_DB_NAME
DJANGO_HOST = constants.DJANGO_HOST
CURRENT_INDEX_NAME = constants.CURRENT_INDEX_NAME
ELASTIC_HOST = constants.ELASTIC_HOST
ELASTIC_PORT = constants.ELASTIC_PORT
PG_USER = constants.PG_USER
PG_PORT = constants.PG_PORT
PG_PASSWORD = constants.PG_PASSWORD
LLM_IP = constants.LLM_IP
LLM_PORT = constants.LLM_PORT
LLM_MODEL = constants.LLM_MODEL
# AnythingLLM (per-request workspace workflow)
IFC_LLM_PORT = constants.IFC_LLM_PORT
IFC_LLM_TOKEN = constants.IFC_LLM_TOKEN
ANYTHINGLLM_WORKSPACE_NAME = "ai_news"  # workspace name created per-request, then deleted

# ──────────────────────────────────────────────
# Elasticsearch connection (env-var override -> constants fallback)
# ──────────────────────────────────────────────
# The user-provided env vars take precedence so the script can be retargeted
# without editing constants.py. The explicit Authorization header is required
# because the elasticsearch 7.17 client has a known bug where basic_auth
# silently drops the header when the password contains '+'.
import base64
ES_HOSTS = os.getenv("ES_HOSTS") or f"https://{ELASTIC_HOST}:{ELASTIC_PORT}"
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
# Default to the password from constants.py so the script still works when
# the operator hasn't explicitly exported the env var.
ES_PASSWORD = os.getenv("ES_PASSWORD")
if not ES_PASSWORD:
    try:
        ES_PASSWORD = constants.ELASTICSEARCH_PASSWORD
    except AttributeError:
        ES_PASSWORD = ""

ES_AUTH_HEADER = "Basic " + base64.b64encode(
    f"{ES_USERNAME}:{ES_PASSWORD}".encode("utf-8")
).decode("ascii")
print(f"[INFO] ES connection: hosts={ES_HOSTS}, user={ES_USERNAME}")

# ------------------------------------------------------------------
# Hard-coded connection and query settings
# ------------------------------------------------------------------
OLLAMA_TIMEOUT = 900
MAX_RETRIES = 3
FALLBACK_LLM_MODEL = "gemma2:9b-instruct-q8_0"
DATE_FIELD   = "@timestamp"
SCROLL_SIZE  = 1000
SCROLL_KEEP  = "5m"
# ------------------------------------------------------------------


def postgres_connection():
	connect = psycopg2.connect(
		database=PG_DB_NAME,
		host=DJANGO_HOST,
		user=PG_USER,
		password=PG_PASSWORD,
		port=PG_PORT,
	)
	return connect

def parse_date(s: str) -> str:
	"""Validate ISO date and return it unchanged."""
	datetime.fromisoformat(s)
	return s

def today_str() -> str:
	return datetime.now(timezone.utc).date().isoformat()

def yesterday_str() -> str:
	return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

def classify_document(doc: dict) -> str:
	"""
	Classify an Elasticsearch document into a semantic category
	such as 'force_disposition', 'training', etc.

	Args:
		doc (dict): The _source part of an Elasticsearch hit.

	Returns:
		str: Category name (e.g., 'force_disposition', 'unknown').
	"""
	if not isinstance(doc, dict):
		return "unknown"

	activity_type = doc.get("activity_type", "").strip().lower()
	form_status = doc.get("form_status")

	# -------------------------
	# 1️. force_disposition
	# -------------------------
	if (
		activity_type in ["deployment", "disposition", "movement"]
		and doc.get("enemy_formation_name")
		and doc.get("disposition_status")
		and form_status != 5
	):
		return "force"

	# -------------------------
	# 2️. training
	# -------------------------
	if (
		activity_type in ["training"]
		and doc.get("enemy_formation_name")
		and doc.get("form_status") != 5
	):
		return "training"

	# -------------------------
	# 3️. infra
	# -------------------------
	if (
		activity_type == "infra"
		and doc.get("infra_type")
		and doc.get("count") is not None
		and doc.get("form_status") != 5
	):
		return "infra"
	
	# -------------------------
	# 4. sitrep
	# -------------------------
	if (
		activity_type in ["sitrep", "patrolling"]
		and doc.get("form_status") != 5
	):
		return "sitrep"
	
	# -------------------------
	# 5. imint
	# -------------------------
	if (
		doc.get("injester_name", "").lower() == "imagery"
		and doc.get("file_hash")
		and doc.get("form_status") != 5
	):
		return "imint"
	
	# -------------------------
	# 6. air_aspect
	# -------------------------
	if (
		doc.get("injester_name", "").lower() == "airfield_injester"
		and doc.get("form_status") != 5
	):
		return "air_aspects"
	
	# -------------------------
	# 7. sam
	# -------------------------
	if (
		doc.get("injester_name", "").lower() == "sam_injester"
		and doc.get("form_status") != 5
	):
		return "sam_deployment_analysis"
	
	# -------------------------
	# 8. visit
	# -------------------------
	if (
		activity_type in ['visit']
		and doc.get("form_status") != 5
	):
		return "visit"
	
	# -------------------------
	# 9. elint
	# -------------------------
	if (
		activity_type in ['elint']
		and doc.get("form_status") != 5
	):
		return "elint"
	
	# -------------------------
	# 10. internal_security
	# -------------------------
	if (
		activity_type in ['internal_security']
		and doc.get("form_status") != 5
	):
		return "internal_security"
	
	# -------------------------
	# 11. overall_disposition
	# -------------------------
	if (
		doc.get("injester_name", "").lower() == "orbat_injester"
		and doc.get("form_status") != 5
	):
		return "overall_disposition"
	
	# -------------------------
	# 12. mobile_interception
	# -------------------------
	if (
		doc.get("injester_name", "").strip().lower() == "relevant_injester"
		and doc.get("form_status") != 5
		and doc.get("activity_type", "").strip().lower() != "equipment"
	):
		return "mobile_interception"
	
	# -------------------------
	# 13. overall_data
	# -------------------------
	if (
		doc.get("coordinates") is not None
		and bool(doc.get("location_name", "").strip())
		and bool(doc.get("description_hash", "").strip())
		and bool(doc.get("activity_type", "").strip())
		and doc.get("form_status") != 5
	):
		return "overall_data"


	# -------------------------
	# Default fallback
	# -------------------------
	return "unknown"


def get_prompt_and_model(category, prompt_dict, cursor, gap_type):
	"""Fetch prompt for a given activity/category.

	This attempts to fetch the latest `prompt` from the `news_prompt` table
	(filtered by `activity_type`) using the provided DB cursor. If the DB
	query fails or returns no result, fall back to the in-memory `prompt_dict`.
	"""
	output_dict = {
		"prompt": "",
		"model_info": "gemma2:9b-instruct-q8_0"
	}
	activity_db_name_map = {
		# classifier output -> news_prompt.activity_name
		"infra": "infra development",
		"training": "training",
		"force": "force disposition",
		"overall_disposition": "force disposition",
		"sitrep": "pla sitrep",
		"imint": "imint analysis",
		"air_aspects": "air aspects analysis",
		"sam_deployment_analysis": "sam deployment analysis",
		"mobile_interception": "mobile interception analysis",
		"internal_security": "internal security analysis",
		"elint": "elint analysis",
		"visit": "visit analysis",
		"overall_data": "overall data",
		"unknown": "general area analysis",
	}

	mapped_category = activity_db_name_map.get(category, 'overall data')
	# Try database first (safe, parameterized query)
	try:
		sql = (
			"SELECT prompt "
			"FROM news_prompt "
			"WHERE activity_name = %s AND news_type = %s "
			"ORDER BY updated_date DESC "
			"LIMIT 1"
		)
		cursor.execute(sql, (mapped_category, gap_type, ))
		row = cursor.fetchone()
		if row:
			# support RealDictCursor (dict) or tuple
			prompt_val = None
			if isinstance(row, dict):
				prompt_val = row.get("prompt")
			else:
				prompt_val = row[0] if len(row) > 0 else None
			if prompt_val and str(prompt_val).strip():
				output_dict["prompt"] = str(prompt_val)
				return output_dict
	except Exception:
		log.exception("Failed to fetch prompt from DB for category %s", category)

	# Fallback to prompt_dict
	prompt = prompt_dict.get(category)
	if not prompt:
		log.info("Prompt not found for category '%s' in DB or prompt_dict, using 'unknown' prompt.", category)
		prompt = prompt_dict.get("unknown", "")

	output_dict["prompt"] = prompt
	return output_dict

def get_llm_response(query, llm_model, docs=None, search_form_type=None,
                     doc_label="analysis_data"):
	"""
	Send a query to AnythingLLM using a per-request workspace.

	Workflow (mirrors ai_analysis_summary_v3.1.1.py):
	  1. Convert `docs` (ES sources) to a markdown file.
	  2. Create a fresh AnythingLLM workspace named 'ai_news'.
	  3. Upload the markdown file to AnythingLLM.
	  4. Embed the document into the workspace.
	  5. Set the workspace chat model and send the prompt.
	  6. Always delete the workspace (even on failure).

	When `docs` is None or empty, falls back to direct chat mode against
	a fresh workspace (no document upload) — same per-request create/delete
	lifecycle so we never leave a stale workspace behind.
	"""
	# ── 1. Build markdown from docs (if any) ──
	markdown_content = ""
	if docs:
		md_type = search_form_type or "general area analysis"
		# The convert function expects hit-shaped dicts; callers pass raw _source dicts.
		hits = [{"_source": d} if isinstance(d, dict) else d for d in docs]
		markdown_content = convert_hits_to_markdown(hits, md_type)

	# ── 2-6. AnythingLLM per-request workspace lifecycle ──
	return _anythingllm_per_request_pipeline(
		markdown_content=markdown_content,
		query=query,
		llm_model=llm_model,
		doc_label=doc_label,
	)


# ──────────────────────────────────────────────
# AnythingLLM workspace lifecycle (create / upload / embed / chat / delete)
# Mirrors ai_analysis_summary_v3.1.1.py
# ──────────────────────────────────────────────

def _anythingllm_headers(json_body=True):
	h = {"Authorization": f"Bearer {IFC_LLM_TOKEN}"}
	if json_body:
		h["Content-Type"] = "application/json"
	return h


def _anythingllm_create_workspace(workspace_name):
	"""Create a new AnythingLLM workspace; return its slug on success."""
	url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/new"
	payload = {"name": workspace_name, "openAiTemp": 0.7}
	r = requests.post(url, json=payload, headers=_anythingllm_headers(), timeout=30)
	if r.status_code in (200, 201):
		ws = r.json().get("workspace") or {}
		slug = ws.get("slug") or workspace_name
		print(f"[INFO] AnythingLLM workspace created: '{workspace_name}' (slug={slug})")
		return slug
	print(f"[ERROR] Workspace create failed: {r.status_code} {r.text[:200]}")
	return None


def _anythingllm_delete_workspace(slug):
	"""Delete an AnythingLLM workspace. Best-effort; never raises."""
	url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}"
	for attempt in range(MAX_RETRIES):
		try:
			r = requests.delete(url, headers=_anythingllm_headers(json_body=False), timeout=30)
			if r.status_code in (200, 204):
				print(f"[INFO] AnythingLLM workspace deleted: {slug}")
				return True
			print(f"[WARN] Workspace delete status {r.status_code}: {r.text[:200]}")
		except Exception as e:
			print(f"[WARN] Workspace delete error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
		time.sleep(2 ** attempt)
	return False


def _anythingllm_update_workspace_model(slug, model_name, chat_provider="ollama"):
	"""Set the chat model on the workspace before sending a chat request."""
	url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/update"
	payload = {"chatProvider": chat_provider, "chatModel": model_name}
	r = requests.post(url, json=payload, headers=_anythingllm_headers(), timeout=30)
	if r.status_code in (200, 201, 204):
		print(f"[INFO] Workspace model set: {model_name} (provider={chat_provider})")
		return True
	print(f"[WARN] Workspace model update status {r.status_code}: {r.text[:200]}")
	return False


def _anythingllm_upload_document(markdown_content, filename):
	"""
	Upload a markdown document to AnythingLLM (multipart).

	Returns the document location path (for use with update-embeddings 'adds')
	or None on failure.
	"""
	import tempfile
	upload_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/document/upload"
	headers = {"Authorization": f"Bearer {IFC_LLM_TOKEN}"}

	tmp_path = None
	try:
		with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
			tmp.write(markdown_content)
			tmp_path = tmp.name

		with open(tmp_path, 'rb') as f:
			files = {'file': (filename, f, 'text/markdown')}
			for attempt in range(MAX_RETRIES):
				try:
					r = requests.post(upload_url, headers=headers, files=files, timeout=60)
					if r.status_code in (200, 201):
						result = r.json()
						docs = result.get('documents') or []
						if docs:
							doc = docs[0]
							print(f"[INFO] Document uploaded: {filename} "
								  f"(id={doc.get('id')}, location={doc.get('location')})")
							return doc.get('location')
						print("[WARN] Upload response missing documents[]")
						return None
					print(f"[ERROR] Document upload status {r.status_code}: {r.text[:200]}")
				except requests.exceptions.Timeout:
					print(f"[ERROR] Upload timeout (attempt {attempt+1}/{MAX_RETRIES})")
				except Exception as e:
					print(f"[ERROR] Upload error: {e}")
				time.sleep(2 ** attempt)
		return None
	finally:
		if tmp_path and os.path.exists(tmp_path):
			try:
				os.unlink(tmp_path)
			except OSError:
				pass


def _anythingllm_add_to_embeddings(slug, document_location):
	"""Link an uploaded document into the workspace's vector embeddings."""
	embed_url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/update-embeddings"
	payload = {"adds": [document_location], "deletes": []}
	for attempt in range(MAX_RETRIES):
		try:
			r = requests.post(embed_url, json=payload, headers=_anythingllm_headers(), timeout=300)
			if r.status_code in (200, 201, 204):
				print(f"[INFO] Document '{document_location}' embedded in workspace")
				return True
			print(f"[ERROR] Embedding update status {r.status_code}: {r.text[:200]}")
		except requests.exceptions.Timeout:
			print(f"[ERROR] Embedding timeout (attempt {attempt+1}/{MAX_RETRIES})")
		except Exception as e:
			print(f"[ERROR] Embedding error: {e}")
		time.sleep(2 ** attempt)
	return False


def _anythingllm_chat(slug, message, mode="chat"):
	"""
	Send a chat request to the workspace.

	`mode='query'` for RAG (grounded in embedded documents).
	`mode='chat'`  for direct LLM with the message as the full prompt.
	"""
	url = f"http://{LLM_IP}:{IFC_LLM_PORT}/api/v1/workspace/{slug}/chat"
	payload = {"message": message.replace("\n", " ").strip(), "mode": mode}
	r = requests.post(url, json=payload, headers=_anythingllm_headers(), timeout=OLLAMA_TIMEOUT)
	if r.status_code == 200:
		result = r.json()
		text = result.get("textResponse", "") or ""
		if text:
			return text
		print(f"[WARN] Chat returned empty textResponse")
		return ""
	print(f"[ERROR] Chat status {r.status_code}: {r.text[:200]}")
	return ""


def _anythingllm_per_request_pipeline(markdown_content, query, llm_model,
                                      doc_label="analysis_data"):
	"""
	Full per-request AnythingLLM workflow:
	  create workspace -> upload md -> embed -> chat -> delete workspace.
	"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"{doc_label}_{timestamp}.md"

	# Step 1: Create workspace
	slug = _anythingllm_create_workspace(ANYTHINGLLM_WORKSPACE_NAME)
	if not slug:
		print("[ERROR] Could not create AnythingLLM workspace; aborting pipeline")
		return "<p>AI analysis unavailable — could not create AnythingLLM workspace.</p>"

	try:
		# Step 2: Upload + embed if we have markdown data
		if markdown_content:
			print(f"[INFO] Uploading markdown: {filename} ({len(markdown_content)} chars)")
			doc_location = _anythingllm_upload_document(markdown_content, filename)
			if doc_location:
				_anythingllm_add_to_embeddings(slug, doc_location)
				# Brief pause for embeddings to settle
				time.sleep(2)
			else:
				print("[WARN] Document upload failed; falling back to direct chat mode")

		# Step 3: Set the chat model on the workspace
		_anythingllm_update_workspace_model(slug, llm_model)
		# Try the configured model, fall back on empty response
		model_to_use = llm_model
		text = _anythingllm_chat(slug, query, mode="chat")
		if not text and model_to_use != FALLBACK_LLM_MODEL:
			print(f"[WARN] Model '{model_to_use}' returned empty; trying fallback '{FALLBACK_LLM_MODEL}'")
			_anythingllm_update_workspace_model(slug, FALLBACK_LLM_MODEL)
			text = _anythingllm_chat(slug, query, mode="chat")
		if text:
			return text
		print("[ERROR] AnythingLLM chat returned no response")
		return "<p>AI analysis unavailable — LLM model could not generate a response.</p>"
	finally:
		# Step 4: Always clean up the workspace
		_anythingllm_delete_workspace(slug)


# ──────────────────────────────────────────────
# Markdown conversion (mirrors ai_analysis_summary_v3.1.1.py)
# ──────────────────────────────────────────────

# Mapping of analysis types to field configurations for markdown generation
TYPE_MAPPING_MD = {
	"infra": {
		"types": ["infra development analysis", "event infra development analysis"],
		"fields": ["infra_type", "location_name", "activity_date", "coordinates", "description"],
		"field_labels": ["Infra Type", "Location Name", "Activity Date", "Coordinates", "Description"],
	},
	"training": {
		"types": ["training areas analysis", "event training areas analysis"],
		"fields": ["enemy_formation_name", "location_name", "description"],
		"field_labels": ["Enemy Formation Name", "Location Name", "Description"],
	},
	"general": {
		"types": ["general area analysis", "event general area analysis"],
		"fields": ["location_name", "coordinates", "description"],
		"field_labels": ["Location Name", "Coordinates", "Description"],
	},
	"overall_data": {
		"types": ["general area analysis", "event general area analysis"],
		"fields": ["location_name", "coordinates", "description"],
		"field_labels": ["Location Name", "Coordinates", "Description"],
	},
	"unknown": {
		"types": ["general area analysis", "event general area analysis"],
		"fields": ["location_name", "coordinates", "description"],
		"field_labels": ["Location Name", "Coordinates", "Description"],
	},
	"force": {
		"types": ["force disposition analysis", "event force disposition analysis"],
		"fields": ["location_name", "coordinates", "base_location_name", "base_coordinates",
				   "enemy_formation_name", "orbate_title", "description"],
		"field_labels": ["Location Name", "Coordinates", "Base Location Name", "Base Coordinates",
						 "Enemy Formation Name", "ORBAT Title", "Description"],
	},
	"overall_disposition": {
		"types": ["force disposition analysis", "event force disposition analysis"],
		"fields": ["location_name", "coordinates", "base_location_name", "base_coordinates",
				   "enemy_formation_name", "orbate_title", "description"],
		"field_labels": ["Location Name", "Coordinates", "Base Location Name", "Base Coordinates",
						 "Enemy Formation Name", "ORBAT Title", "Description"],
	},
	"sitrep": {
		"types": ["pla sitrep analysis", "event pla sitrep analysis"],
		"fields": ["pass_name", "transgression_sighting_type", "sub_activity_type", "description"],
		"field_labels": ["Pass Name", "Transgression/Sighting Type", "Sub Activity Type", "Description"],
	},
	"air_aspects": {
		"types": ["air aspects analysis", "event air aspects analysis"],
		"fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name",
				   "equipement_type", "count", "airfield_type"],
		"field_labels": ["Location Name", "Coordinates", "Infra Name", "Infra Type", "Equipment Name",
						 "Equipment Type", "Count", "Airfield Type"],
	},
	"sam_deployment_analysis": {
		"types": ["sam deployment analysis", "event sam deployment analysis"],
		"fields": ["location_name", "coordinates", "infra_name", "infra_type", "equipment_name",
				   "equipment_type", "count"],
		"field_labels": ["Location Name", "Coordinates", "Infra Name", "Infra Type", "Equipment Name",
						 "Equipment Type", "Count"],
	},
	"mobile_interception": {
		"types": ["mobile interception analysis", "Mobile Interception Analysis"],
		"fields": ["start_location_name", "end_location_name", "opposite_to", "mobile_no", "description"],
		"field_labels": ["Start Location Name", "End Location Name", "Opposite To", "Mobile No", "Description"],
	},
	"internal_security": {
		"types": ["internal security analysis", "Internal Security Analysis"],
		"fields": ["coordinates", "terrorist_casualties_", "security_forces_casualties", "civilian_casualties",
				   "description", "army_name", "force_type_name", "formation_type", "enemy_formation_name",
				   "command_name", "command_coordinates", "comd_tps_loc_name", "comd_tps_coordinates",
				   "terrorists_casualties"],
		"field_labels": ["Coordinates", "Terrorist Casualties", "Security Forces Casualties", "Civilian Casualties",
						 "Description", "Army Name", "Force Type Name", "Formation Type", "Enemy Formation Name",
						 "Command Name", "Command Coordinates", "COMd TPS Loc Name", "COMd TPS Coordinates",
						 "Terrorists Casualties"],
	},
	"elint": {
		"types": ["elint analysis", "Elint Analysis"],
		"fields": ["description", "location_name", "coordinates", "category", "radar_type", "radar_name"],
		"field_labels": ["Description", "Location Name", "Coordinates", "Category", "Radar Type", "Radar Name"],
	},
	"visit": {
		"types": ["visit analysis", "Visit Analysis"],
		"fields": ["description", "visit_name", "purpose", "location_name", "coordinates"],
		"field_labels": ["Description", "Visit Name", "Purpose", "Location Name", "Coordinates"],
	},
}


def convert_hits_to_markdown(e_hits, search_form_type):
	"""
	Convert Elasticsearch hits (or raw _source dicts) to a well-structured markdown document.

	Uses TYPE_MAPPING_MD to create a table for known analysis types; otherwise
	falls back to a generic per-record bullet list. Mirrors the same function
	in ai_analysis_summary_v3.1.1.py.
	"""
	search_form_type_lower = (search_form_type or "").lower()
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	lines = [f"# Analysis Report: {search_form_type or 'general'}", "",
			 f"- **Generated:** {now}", f"- **Total Records:** {len(e_hits)}", ""]

	if not e_hits:
		lines.append("_No records found._")
		return "\n".join(lines)

	# Find a matching type config (search by category key OR by any of its 'types')
	type_config = None
	for key, config in TYPE_MAPPING_MD.items():
		if search_form_type_lower == key.lower() or \
		   search_form_type_lower in [t.lower() for t in config["types"]]:
			type_config = config
			break

	if type_config is None:
		# Fallback: generic per-record list
		lines.append("## Data Records\n")
		for idx, hit in enumerate(e_hits, 1):
			source = hit.get("_source", hit) if isinstance(hit, dict) else {}
			lines.append(f"### Record {idx}")
			lines.append("")
			for field, value in source.items():
				if field.startswith("@"):
					continue
				lines.append(f"- **{field}:** {value}")
			lines.append("")
		return "\n".join(lines)

	fields = type_config["fields"]
	field_labels = type_config["field_labels"]
	lines.append("## Summary Statistics")
	lines.append("")
	lines.append(f"- **Analysis Category:** {search_form_type}")
	lines.append(f"- **Fields Tracked:** {', '.join(field_labels)}")
	lines.append("")
	lines.append("## Detail Records")
	lines.append("")
	lines.append("| # | " + " | ".join(field_labels) + " |")
	lines.append("|---" * (len(field_labels) + 1) + "|")
	for idx, hit in enumerate(e_hits, 1):
		source = hit.get("_source", hit) if isinstance(hit, dict) else {}
		row_values = []
		for field in fields:
			value = source.get(field, "") if isinstance(source, dict) else ""
			if value is None:
				value = ""
			value_str = str(value).replace("|", "\\|")
			if len(value_str) > 120:
				value_str = value_str[:117] + "..."
			row_values.append(value_str)
		lines.append(f"| {idx} | " + " | ".join(row_values) + " |")
	lines.append("")
	lines.append("---")
	lines.append(f"*End of report — {len(e_hits)} records*")
	return "\n".join(lines)



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
		print("[!]json load failed:")
		result = {"summary": result}
	return result

def imint_ai_analysis_from_docs(docs: list, cursor, gap_type: str = "daily") -> str:
	"""
	Re-use the *already fetched* ES docs list for IMINT.
	Returns base64-encoded HTML summary.
	"""
	model_prompt_dict = get_prompt_and_model("imint", prompt_dict, cursor, gap_type)
	# Convert comments list to a markdown-friendly per-doc dict (so the AnythingLLM
	# workspace sees a real, structured document rather than a flat prompt).
	imint_docs = []
	for doc in docs:
		src = doc
		if src.get("comments"):
			last_comment = src["comments"][-1].strip().replace(chr(10), ". ")
			imint_docs.append({
				"location_name": src.get("location_name", ""),
				"activity_date": src.get("activity_date", ""),
				"comments": last_comment,
			})

	if not imint_docs:
		return ""

	# Final user prompt (instructions to the model); data goes in the embedded doc.
	llm_prompt = model_prompt_dict["prompt"] + \
		"\n\nThe response you will give should be in html. " \
		"Any title in the response should be in h2 tag and paragraph should be in p tag."

	llm_response = get_llm_response(
		llm_prompt,
		model_prompt_dict["model_info"],
		docs=imint_docs,
		search_form_type="imint analysis",
		doc_label="imint",
	)
	corrected_html = correct_json_format(str(llm_response)).get("summary", "")
	return base64.b64encode(corrected_html.encode("utf-8")).decode("utf-8")

def ai_summary_builder(category: str, docs: list, cursor, gap_type) -> str:
	"""
	Placeholder function to build AI summary for a given category
	and list of documents.

	Args:
		category (str): The category of the documents.
		docs (list): List of document _source dicts.

	Returns:
		str: AI-generated summary.
	"""
	print(f"Building AI summary for category '{category}' with {len(docs)} documents.")
	ai_summary = ''
	llm_response = ''
	ai_trend_response = ''
	ai_change_repsonse = ''
	ai_summ_start_time = time.time()
	model_prompt_dict = get_prompt_and_model(category, prompt_dict, cursor, gap_type)
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
		"overall_data": {
			"types": ["general area analysis", "event general area analysis"],
			"fields": ["location_name", "coordinates", "description"],
			"line_format": "location_name:{location_name}, coordinates:{coordinates}, description:{description}",
		},
		"unknown": {
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
		"overall_disposition": {
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
	if category in type_mapping:
		value = type_mapping[category]
		# Pick a search_form_type that matches the TYPE_MAPPING_MD table builder
		# (use the first declared type for this category).
		search_form_type = value["types"][0] if value.get("types") else None
		# Dedupe docs by formatted line so we don't embed duplicate records
		seen_lines = set()
		unique_docs = []
		for doc in docs:
			source = doc
			line = value["line_format"].format(
				**{field: source.get(field, "") for field in value["fields"]}
			)
			if line in seen_lines:
				continue
			seen_lines.add(line)
			unique_docs.append(source)

		if unique_docs:
			# Final user prompt (instructions to the model); data goes in the embedded doc.
			llm_prompt = model_prompt_dict["prompt"] + \
				"\n\n(PLEASE NOTE: The response you will give should be in html(hyper text markup language). " \
				"Any title in the response should be in h2 tag and paragraph should be in p tag.)"

			temp_llm_response = get_llm_response(
				llm_prompt,
				model_prompt_dict.get("model_info"),
				docs=unique_docs,
				search_form_type=search_form_type,
				doc_label=category,
			)
			temp_llm_response = f"<div>{temp_llm_response}</div><br>"
			llm_response += temp_llm_response
		print(f"[DONE] ai analysis complete for tab {category} [time={(time.time() - ai_summ_start_time):.2f} s] ")

		# with open("llm_response.html", "w", encoding="utf-8") as f:
		# 	f.write(llm_response)
		# with open("ai_change_repsonse.html", "w", encoding="utf-8") as f:
		# 	f.write(ai_change_repsonse)
		# with open("ai_trend_response.html", "w", encoding="utf-8") as f:
		# 	f.write(ai_trend_response)
	else:
		print(f"[!] No type mapping found for category '{category}'.")
		# No known type mapping — just pass the raw docs through; the markdown
		# converter will fall back to a generic per-record list.
		llm_prompt = model_prompt_dict["prompt"] + \
			"\n\n(PLEASE NOTE: The response you will give should be in html(hyper text markup language). " \
			"Any title in the response should be in h2 tag and paragraph should be in p tag.)"
		temp_llm_response = get_llm_response(
			llm_prompt,
			model_prompt_dict.get("model_info"),
			docs=docs,
			search_form_type=category,
			doc_label=category,
		)
		temp_llm_response = f"<div>{temp_llm_response}</div><br>"
		llm_response += temp_llm_response
		print(f"[DONE] ai analysis complete for tab {category} [time={(time.time() - ai_summ_start_time):.2f} s] ")

	return llm_response


def ai_report_builder(category: str, docs: list, start, end, cursor, gap_type):
	"""
	Placeholder function to build AI analysis reports for a given category
	and list of documents.

	Args:
		category (str): The category of the documents.
		docs (list): List of document _source dicts.
	"""
	ai_change_repsonse = ''
	ai_trend_response = ''
	llm_response = ''
	tabbed_html = ''
	if category == "imint":
		llm_response = imint_ai_analysis_from_docs(docs, cursor, gap_type)
	else:
		llm_response = ai_summary_builder(category, docs, cursor, gap_type)
	tabbed_html = build_tabbed_html([llm_response, ai_trend_response, ai_change_repsonse, ])
	encoded_tabbed_html = str(base64.b64encode(str(tabbed_html).encode('utf-8')).decode('utf-8'))
	sql = """
	INSERT INTO ai_summary_report
	(ai_summary_html_report, create_date, update_date,
	activity, report_category, report_title,
	report_type, data_start_date, data_end_date)
	VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
	RETURNING id;
	"""

	report_category = ", ".join([name for val, name in zip([ai_change_repsonse, ai_trend_response, llm_response],["AI Change", "AI Trend", "AI Summary"]) if val.strip()])
	now = datetime.now(timezone.utc)
	cursor.execute(sql, [
		encoded_tabbed_html,
		now,              # create_date
		now,              # update_date
		category,          # activity
		report_category,        # report_category
		f"{gap_type}_{category}_{start}_{end}",     # report_title
		f"{gap_type}",        # report_type
		start,     # data_start_date
		end,     # data_end_date
	])

	


def tab_report_build(start, end, cur, gap_type):
	# Build ES client with explicit Authorization header — works around the
	# 7.17 client's basic_auth bug with '+' in passwords.
	es = Elasticsearch(
		[ES_HOSTS] if isinstance(ES_HOSTS, str) else ES_HOSTS,
		headers={"Authorization": ES_AUTH_HEADER},
		verify_certs=False,
		ssl_show_warn=False,
		request_timeout=60,
	)
	if not es.ping():
		sys.exit("Could not connect to Elasticsearch")

	query = {
		"query": {
			"range": {
				DATE_FIELD: {
					"gte": start,
					"lte": end,
					"format": "yyyy-MM-dd"
				}
			}
		}
	}
	grouped_docs = {}

	for doc in helpers.scan(
		es,
		query=query,
		index=CURRENT_INDEX_NAME,
		size=SCROLL_SIZE,
		scroll=SCROLL_KEEP,
		_source=True,
		clear_scroll=True,
	):
		# pprint(doc["_source"])
		src = doc["_source"]
		category = classify_document(src)
		if category not in grouped_docs:
			grouped_docs[category] = []
		grouped_docs[category].append(src)
	# print counts per category
	for category, docs in grouped_docs.items():
		if len(docs) > 0:
			ai_report_builder(category, docs, start, end, cur, gap_type)
		else:
			print(f"No documents found for category '{category}'.")  


def main():
	parser = argparse.ArgumentParser(description="Scroll-export ES docs by date range")
	parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD). Default: yesterday")
	parser.add_argument("--end",   type=parse_date, help="End date (YYYY-MM-DD).   Default: today")
	parser.add_argument("--weekly", action="store_true", help="Set range to the most recent full week (Mon-Sun) before today")
	parser.add_argument("--monthly", action="store_true", help="Set range to the most recent full month (1st-last) before today")
	parser.add_argument("--daily", action="store_true", help="Run only daily reports (each day in the range). If provided, monthly/weekly flags are not allowed")
	args = parser.parse_args()

	# Mutual exclusion for selection flags
	if (args.monthly and args.weekly) or (args.daily and (args.weekly or args.monthly)):
		log.error("--daily, --weekly and --monthly are mutually exclusive")
		sys.exit("Please pass only one of --daily, --weekly or --monthly")

	# If user explicitly provided both --start and --end but no selection flags, exit
	if args.start is not None and args.end is not None and not (args.daily or args.weekly or args.monthly):
		print("Both --start and --end provided but no report type selected. Use --daily, --weekly or --monthly. Exiting.")
		sys.exit(0)

	# Determine initial start/end based on flags or provided arguments
	if args.monthly:
		# previous calendar month (full month before current date)
		today = datetime.now(timezone.utc).date()
		first_of_current = today.replace(day=1)
		last_of_prev = first_of_current - timedelta(days=1)
		first_of_prev = last_of_prev.replace(day=1)
		start = first_of_prev.isoformat()
		end = last_of_prev.isoformat()
		print(f"--monthly set: start={start}, end={end}")
	elif args.weekly:
		# previous full week (Mon-Sun) before current date
		today = datetime.now(timezone.utc).date()
		current_monday = today - timedelta(days=today.weekday())
		prev_week_start = current_monday - timedelta(days=7)
		prev_week_end = current_monday - timedelta(days=1)
		start = prev_week_start.isoformat()
		end = prev_week_end.isoformat()
		print(f"--weekly set: start={start}, end={end}")
	else:
		# fall back to explicit args or defaults
		start = args.start or (yesterday_str() if args.daily else yesterday_str())
		end   = args.end   or (today_str() if args.daily else today_str())

	print('start=', start, "end=", end)

	# Compute gap and determine which granular reports to run
	start_date = datetime.fromisoformat(start).date()
	end_date = datetime.fromisoformat(end).date()
	if end_date < start_date:
		log.error("End date %s is before start date %s", end, start)
		sys.exit("End date must be >= start date")
	gap_days = (end_date - start_date).days

	# Determine report-level flags.
	# If --daily/--weekly/--monthly was explicitly requested, honor it (mutually exclusive).
	if args.daily:
		daily, weekly, monthly = True, False, False
	elif args.weekly:
		daily, weekly, monthly = False, True, False
	elif args.monthly:
		daily, weekly, monthly = False, False, True
	else:
		# No explicit selection: daily is opt-in only; weekly/monthly decided by gap
		daily = False
		if gap_days < 7:
			weekly = False
			monthly = False
		elif gap_days < 30:
			weekly = True
			monthly = False
		else:
			weekly = True
			monthly = True

	print(f"Date range length: {gap_days} days -> daily={daily}, weekly={weekly}, monthly={monthly}")

	# If no report types were requested explicitly, do nothing and exit
	if not (daily or weekly or monthly):
		print("No report types selected. Use --daily, --weekly or --monthly to request reports. Exiting.")
		sys.exit(0)

	DSN = f"host={DJANGO_HOST} dbname={PG_DB_NAME} user={PG_USER} password={PG_PASSWORD}"
	pool = PostgresCursorPool(DSN)
	with pool.cursor() as cur:
		if daily:
			# If the user passed only --daily (no explicit start/end), treat it as a single
			# report covering yesterday -> today. If start/end were provided, build per-day ranges.
			if args.start is None and args.end is None:
				daily_ranges = [(start, end)]
			else:
				# Build per-day calendar ranges (each day in the range)
				daily_ranges = []
				day_cursor = start_date
				while day_cursor <= end_date:
					daily_ranges.append((day_cursor.isoformat(), day_cursor.isoformat()))
					day_cursor += timedelta(days=1)
			n_days = len(daily_ranges)
			print(f"Building {n_days} daily report(s):")
			for idx, (ds, de) in enumerate(daily_ranges, 1):
				print(f"  {idx}: {ds} -> {de}")
			# invoke daily runs
			for ds, de in daily_ranges:
				print(f"Running daily report for {ds} to {de}")
				tab_report_build(ds, de, cur, gap_type="daily")
		if weekly:
			# Build calendar-aligned FULL weekly ranges (Monday - Sunday) that are completely inside the date range
			weekly_ranges = []
			# first full Monday on or after start_date
			if start_date.weekday() == 0:
				first_monday = start_date
			else:
				first_monday = start_date + timedelta(days=(7 - start_date.weekday()))
			# last full Sunday on or before end_date
			if end_date.weekday() == 6:
				last_sunday = end_date
			else:
				last_sunday = end_date - timedelta(days=(end_date.weekday() + 1))
			if first_monday <= last_sunday:
				week_cursor = first_monday
				while week_cursor <= last_sunday:
					week_start = week_cursor
					week_end = week_cursor + timedelta(days=6)
					weekly_ranges.append((week_start.isoformat(), week_end.isoformat()))
					week_cursor += timedelta(days=7)
			n_weeks = len(weekly_ranges)
			print(f"Building {n_weeks} weekly report(s) (full weeks only):")
			for idx, (ws, we) in enumerate(weekly_ranges, 1):
				print(f"  {idx}: {ws} -> {we}")
			# invoke weekly runs
			for ws, we in weekly_ranges:
				print(f"Running weekly report for {ws} to {we}")
				tab_report_build(ws, we, cur, gap_type="weekly")
			if n_weeks == 0:
				print("No full weeks contained in range; skipping weekly reports.")
		if monthly:
			# Build calendar-aligned FULL monthly ranges (1st day -> last day) that are completely inside the date range
			monthly_ranges = []
			# first full month start
			if start_date.day == 1:
				first_month_start = start_date
			else:
				# first day of next month
				if start_date.month == 12:
					first_month_start = start_date.replace(year=start_date.year + 1, month=1, day=1)
				else:
					first_month_start = start_date.replace(month=start_date.month + 1, day=1)
			# last full month start
			fd_end_month = end_date.replace(day=1)
			# first day of next month for end_date
			if fd_end_month.month == 12:
				fd_next = fd_end_month.replace(year=fd_end_month.year + 1, month=1, day=1)
			else:
				fd_next = fd_end_month.replace(month=fd_end_month.month + 1, day=1)
			last_day_of_end_month = fd_next - timedelta(days=1)
			if end_date == last_day_of_end_month:
				last_month_start = end_date.replace(day=1)
			else:
				prev_month_last_day = fd_end_month - timedelta(days=1)
				last_month_start = prev_month_last_day.replace(day=1)
			# gather months fully inside
			if first_month_start <= last_month_start:
				month_cursor = first_month_start
				while month_cursor <= last_month_start:
					# compute next month start
					if month_cursor.month == 12:
						next_month = month_cursor.replace(year=month_cursor.year + 1, month=1, day=1)
					else:
						next_month = month_cursor.replace(month=month_cursor.month + 1, day=1)
					month_start = month_cursor
					month_end = next_month - timedelta(days=1)
					monthly_ranges.append((month_start.isoformat(), month_end.isoformat()))
					month_cursor = next_month
			n_months = len(monthly_ranges)
			print(f"Building {n_months} monthly report(s) (full months only):")
			for idx, (ms, me) in enumerate(monthly_ranges, 1):
				print(f"  {idx}: {ms} -> {me}")
			# invoke monthly runs
			for ms, me in monthly_ranges:
				print(f"Running monthly report for {ms} to {me}")
				tab_report_build(ms, me, cur, gap_type="monthly")
			if n_months == 0:
				print("No full months contained in range; skipping monthly reports.")
		
		pool.close()

if __name__ == "__main__":
	main()
