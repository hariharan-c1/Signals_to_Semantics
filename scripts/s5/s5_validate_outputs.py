#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity checker for LLM outputs (S5).

- Scans a directory for *.json files (each per window).
- Verifies that each file has a valid parsed_result with:
    * actor_matrix (non-empty list with basic structure)
    * primary_responsible_actor key present (value can be null)
    * scenario_classification present & in allowed set
    * final_rationale present & non-empty string
    * ego_window_key present (non-empty)
- Prints summary metrics.
- Writes a CSV with all errors per file.

Usage:
    python scripts/s5/s5_validate_outputs.py \
        --llm-dir artifacts/train650_val50/val50/s5/base_prompt \
        --out-csv artifacts/train650_val50/val50/s5/base_prompt/sanity.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

ALLOWED_SCENARIOS = {
    "cut_in",
    "ped_crossing",
    "obj_crossing",
    "lead_brake",
    "approach_stop",
    "left_oppo",
    "right_ped",
    "other",
}

VALID_ACTORS = {"ACTOR1", "ACTOR2", "ACTOR3"}


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data, errors
    except Exception as e:
        errors.append(f"json_load_error: {type(e).__name__}: {e}")
        return None, errors


def check_actor_matrix(actor_matrix: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(actor_matrix, list):
        errors.append("actor_matrix_not_list")
        return errors
    if len(actor_matrix) == 0:
        errors.append("actor_matrix_empty")
        return errors

    for i, a in enumerate(actor_matrix):
        if not isinstance(a, dict):
            errors.append(f"actor_{i}_not_dict")
            continue
        if "actor_id" not in a:
            errors.append(f"actor_{i}_missing_actor_id")
        if "risk_score" not in a:
            errors.append(f"actor_{i}_missing_risk_score")
        if "reasoning" not in a:
            errors.append(f"actor_{i}_missing_reasoning")
        else:
            reason = a.get("reasoning")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"actor_{i}_empty_reasoning")

    return errors


def check_parsed_result(parsed: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(parsed, dict):
        errors.append("parsed_result_not_dict")
        return errors

    # ego_window_key
    ego_key = parsed.get("ego_window_key")
    if ego_key is None or not isinstance(ego_key, str) or not ego_key.strip():
        errors.append("missing_or_empty_ego_window_key")

    # actor_matrix
    actor_matrix = parsed.get("actor_matrix")
    if actor_matrix is None:
        errors.append("missing_actor_matrix")
    else:
        errors.extend(check_actor_matrix(actor_matrix))

    # primary_responsible_actor (presence, value can be null)
    if "primary_responsible_actor" not in parsed:
        errors.append("missing_primary_responsible_actor_key")
    else:
        pra = parsed.get("primary_responsible_actor")
        if pra is not None and pra not in VALID_ACTORS:
            errors.append(
                f"invalid_primary_responsible_actor_value:{pra}"
            )

    # scenario_classification
    scen = parsed.get("scenario_classification")
    if scen is None:
        errors.append("missing_scenario_classification")
    elif scen not in ALLOWED_SCENARIOS:
        errors.append(f"invalid_scenario_classification:{scen}")

    # final_rationale
    if "final_rationale" not in parsed:
        errors.append("missing_final_rationale")
    else:
        fr = parsed.get("final_rationale")
        if not isinstance(fr, str) or not fr.strip():
            errors.append("empty_final_rationale")

    # confidence_score (optional but nice to check)
    if "confidence_score" not in parsed:
        errors.append("missing_confidence_score")
    else:
        cs = parsed.get("confidence_score")
        if not isinstance(cs, (int, float)):
            errors.append(f"confidence_score_not_numeric:{cs}")

    return errors


def check_single_file(path: Path) -> Dict[str, Any]:
    """
    Returns a dict with:
        {
            "file": <filename>,
            "ego_window_key": <str or ''>,
            "backend": <backend or ''>,
            "errors": [list of error strings]
        }
    """
    record = {
        "file": str(path.name),
        "ego_window_key": "",
        "backend": "",
        "errors": []  # type: List[str]
    }

    data, load_errors = load_json(path)
    if load_errors:
        record["errors"].extend(load_errors)
        return record

    backend = data.get("backend", "")
    record["backend"] = backend if isinstance(backend, str) else ""

    parsed = data.get("parsed_result")
    if parsed is None:
        record["errors"].append("missing_parsed_result")
        return record

    ego_key = parsed.get("ego_window_key")
    if isinstance(ego_key, str):
        record["ego_window_key"] = ego_key

    record["errors"].extend(check_parsed_result(parsed))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm-dir",
        required=True,
        help="Directory containing per-window LLM output JSON files.",
    )
    ap.add_argument(
        "--out-csv",
        default="llm_sanity_errors.csv",
        help="Path to CSV file to write sanity errors.",
    )
    args = ap.parse_args()

    llm_dir = Path(args.llm_dir)
    if not llm_dir.is_dir():
        raise SystemExit(f"[ERROR] LLM dir not found: {llm_dir}")

    json_files = sorted(llm_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"[ERROR] No JSON files found in {llm_dir}")

    print(f"[INFO] Found {len(json_files)} LLM output files in {llm_dir}")

    error_records: List[Dict[str, Any]] = []
    total = 0
    ok = 0

    for path in json_files:
        total += 1
        rec = check_single_file(path)
        if rec["errors"]:
            error_records.append(rec)
        else:
            ok += 1

    num_errors = len(error_records)
    print(f"[RESULT] Total files: {total}")
    print(f"[RESULT] OK files:    {ok}")
    print(f"[RESULT] Error files: {num_errors}")

    if total > 0:
        acc = ok / total
        print(f"[RESULT] Sanity pass rate: {acc:.3f}")

    # Write CSV with error details
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file", "ego_window_key", "backend", "error_count", "errors"]
        )
        for rec in error_records:
            writer.writerow(
                [
                    rec["file"],
                    rec["ego_window_key"],
                    rec["backend"],
                    len(rec["errors"]),
                    ";".join(rec["errors"]),
                ]
            )

    print(f"[INFO] Wrote error report to {out_csv}")


if __name__ == "__main__":
    main()
