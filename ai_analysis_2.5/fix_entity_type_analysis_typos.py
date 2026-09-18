"""
Fix typos in public.ai_analysis_summary.entity_type_analysis (jsonb array).

⭐ Companion to ai_analysis_summary_v3.1.1.13_rag.py (v3.1.1.14): the Overall
Analysis restricts itself to the entity types listed in entity_type_analysis.
Typos such as 'enemey_foramtion_name' would not match any supported field, so
this script rewrites them to the correct canonical field names BEFORE the
analysis runs.

Behavior:
  - Only jsonb ARRAY values are touched (NULL / scalars are left alone).
  - Elements are trimmed, lower-cased, de-duplicated (order preserved).
  - Known typos are mapped to the canonical field names (exact map below).
  - Any other unknown element is corrected with a conservative fuzzy match
    (difflib cutoff 0.85) against the canonical fields, or kept and reported.
  - An array that becomes empty after cleaning is set to NULL (== all fields).

Usage:
  python3 fix_entity_type_analysis_typos.py            # dry run, prints plan
  python3 fix_entity_type_analysis_typos.py --apply    # write corrections
  python3 fix_entity_type_analysis_typos.py --apply --no-fuzzy   # exact map only

NOTE: 'equipement_type' and 'orbate_title' are INTENTIONALLY kept: they are the
real (historically misspelled) field names used in the Elasticsearch index and
across the analysis scripts. They are canonical here.
"""

import argparse
import difflib
import json
import os
import importlib.util

import psycopg2

# ── constants.py (one folder up), same pattern as the analysis scripts ──
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'constants.py'))
spec = importlib.util.spec_from_file_location("constants", file_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

# ── Canonical entity-type fields supported for Overall Analysis ──
# Mirrors OVERALL_ENTITY_TYPE_FIELDS in ai_analysis_summary_v3.1.1.13_rag.py.
CANONICAL_FIELDS = {
    # Location Analysis
    "location_name", "base_location_name", "start_location_name", "end_location_name",
    "pass_name", "visit_name", "comd_tps_loc_name",
    # Force Movement and Deployment Analysis
    "activity_type", "movement_type", "deployment_type", "enemy_formation_name",
    "army_name", "force_type_name", "formation_type", "command_name", "orbate_title",
    "mobile_no",
    # Equipment and Vehicle Analysis
    "equipment_name", "equipment_type", "equipement_type", "vehicle_type", "vehicle_name",
    "count", "weapon_system", "radar_name", "radar_type", "airfield_type",
    # Infrastructure Analysis
    "infra_name", "infra_type", "infrastructure_type", "project_type", "coordinates",
    # Training and Readiness Analysis
    "training_type", "training_area", "exercise_type", "readiness_type", "sub_activity_type",
    # Visits and Inspections Analysis
    "purpose", "visit_type", "inspection_type",
    # Force Composition, Strength and Capability Analysis
    "strength", "personnel_count", "capability", "capabilities",
    # Events, Security and Other Relevant Analysis
    "event_type", "category", "incident_type", "casualties", "civilian_casualties",
    "security_forces_casualties", "terrorists_casualties", "political_event", "security_event",
}

# Values that mean "no restriction" and are left untouched.
ALL_KEYWORDS = {"all", "all fields", "overall", "overall analysis", "*"}

# Known typos -> canonical field (exact, case-insensitive).
TYPO_MAP = {
    # enemy_formation_name
    "enemey_foramtion_name": "enemy_formation_name",
    "enemy_foramtion_name": "enemy_formation_name",
    "enemey_formation_name": "enemy_formation_name",
    "enemey_formtion_name": "enemy_formation_name",
    "enemy_formtion_name": "enemy_formation_name",
    "enmy_formation_name": "enemy_formation_name",
    # location_name
    "locaton_name": "location_name",
    "locaiton_name": "location_name",
    "loction_name": "location_name",
    "location_nam": "location_name",
    # equipment
    "equipmnt_name": "equipment_name",
    "equpment_name": "equipment_name",
    "equiment_name": "equipment_name",
    "equipemnt_type": "equipment_type",
    "eqipment_type": "equipment_type",
    "equipmnt_type": "equipment_type",
    # infrastructure
    "infrastrcture_type": "infrastructure_type",
    "infrastucture_type": "infrastructure_type",
    # training
    "traning_type": "training_type",
    "trianing_type": "training_type",
    "excercise_type": "exercise_type",
    "exersize_type": "exercise_type",
    "exrcise_type": "exercise_type",
    # visits
    "vist_name": "visit_name",
    "visit_typ": "visit_type",
    "inspecion_type": "inspection_type",
    # force composition
    "force_typ_name": "force_type_name",
    "formtion_type": "formation_type",
    "comand_name": "command_name",
    "orbat_title": "orbate_title",
    "moblie_no": "mobile_no",
    # events / casualties
    "casualities": "casualties",
    "civilian_casuality": "civilian_casualties",
    "civillian_casualties": "civilian_casualties",
    "security_force_casualties": "security_forces_casualties",
    "terroist_casualties": "terrorists_casualties",
    # infra
    "infa_name": "infra_name",
    "infr_name": "infra_name",
}

FUZZY_CUTOFF = 0.85


def fuzzy_correct(value: str):
    """Conservative difflib correction against canonical fields, or None."""
    matches = difflib.get_close_matches(value, sorted(CANONICAL_FIELDS), n=1, cutoff=FUZZY_CUTOFF)
    return matches[0] if matches else None


def clean_array(values, use_fuzzy=True):
    """Return (cleaned_list, changes) where changes is a list of (old, new, how)."""
    cleaned = []
    changes = []
    seen = set()
    for raw in values:
        original = str(raw)
        text = original.strip().lower()
        if not text:
            changes.append((original, "<dropped: empty>", "empty"))
            continue
        if text in ALL_KEYWORDS:
            changes.append((original, text, "all-keyword (kept)"))
            new = text
        elif text in CANONICAL_FIELDS:
            new = text
            if text != original.strip():
                changes.append((original, new, "case/whitespace"))
        elif text in TYPO_MAP:
            new = TYPO_MAP[text]
            changes.append((original, new, "typo map"))
        elif use_fuzzy:
            corrected = fuzzy_correct(text)
            if corrected:
                new = corrected
                changes.append((original, new, f"fuzzy>={FUZZY_CUTOFF}"))
            else:
                new = text
                changes.append((original, new, "UNRECOGNIZED (kept as-is)"))
        else:
            new = text
            changes.append((original, new, "UNRECOGNIZED (kept as-is)"))
        if new not in seen:
            seen.add(new)
            cleaned.append(new)
    return cleaned, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="Write corrections to the database (default: dry run)")
    parser.add_argument("--no-fuzzy", action="store_true",
                        help="Only apply the exact typo map, never fuzzy corrections")
    args = parser.parse_args()

    conn = psycopg2.connect(
        database=constants.PG_DB_NAME,
        host=constants.DJANGO_HOST,
        user=constants.PG_USER,
        password=constants.PG_PASSWORD,
        port=constants.PG_PORT,
    )
    updated = 0
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ai_analysis_summary_id, entity_type_analysis
                FROM public.ai_analysis_summary
                WHERE jsonb_typeof(entity_type_analysis) = 'array'
                ORDER BY ai_analysis_summary_id;
            """)
            rows = cur.fetchall()

        print(f"Rows with jsonb array entity_type_analysis: {len(rows)}")
        for row_id, value in rows:
            if not isinstance(value, list):
                continue
            cleaned, changes = clean_array(value, use_fuzzy=not args.no_fuzzy)
            if not changes:
                continue
            print(f"\nRow {row_id}:")
            print(f"  before: {json.dumps(value)}")
            for old, new, how in changes:
                print(f"    [{how}] {old!r} -> {new!r}")
            new_value = json.dumps(cleaned) if cleaned else None
            print(f"  after : {new_value if new_value is not None else 'NULL (all fields)'}")
            if args.apply:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE public.ai_analysis_summary "
                        "SET entity_type_analysis = %s::jsonb "
                        "WHERE ai_analysis_summary_id = %s;",
                        (new_value, row_id),
                    )
                updated += 1

        if args.apply:
            conn.commit()
            print(f"\nAPPLIED: {updated} row(s) updated.")
        else:
            conn.rollback()
            print("\nDRY RUN: no changes written. Re-run with --apply to write corrections.")
    finally:
        conn.close()

    print("\nSupported Overall Analysis fields (canonical):")
    for field in sorted(CANONICAL_FIELDS):
        print(f"  {field}")


if __name__ == "__main__":
    main()
