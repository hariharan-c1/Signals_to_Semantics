#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

SCENARIO_LABEL_TOKENS = [
    "cut_in",
    "ped_cross",
    "obj_cross",
    "lead_brake",
    "approach_stop",
    # "other" is too generic to safely check
]

def approx_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def check_file(path: Path):
    errors = []
    warnings = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse JSON: {e}")
        return errors, warnings

    # --- basic keys ---
    if "window_key" not in data:
        errors.append("Missing top-level 'window_key'.")
    if "episode" not in data:
        errors.append("Missing top-level 'episode'.")
    if "actors" not in data:
        errors.append("Missing top-level 'actors'.")
    if "llm_input" not in data:
        errors.append("Missing 'llm_input'.")
        return errors, warnings  # no point checking further if this is gone

    llm = data["llm_input"]

    # --- ego_window_key consistency ---
    if llm.get("ego_window_key") != data.get("window_key"):
        errors.append("ego_window_key != window_key.")

    # --- brake episode structure ---
    epi = llm.get("ego_brake_episode", {})
    for key in ["t_start_s", "t_peak_brake_s", "t_end_s",
                "brake_confidence_score", "peak_deceleration_mps2"]:
        if key not in epi:
            errors.append(f"ego_brake_episode missing '{key}'.")

    # --- actors consistency ---
    actors_full = data.get("actors", [])
    actors_llm = llm.get("actors", [])

    if not isinstance(actors_llm, list) or len(actors_llm) == 0:
        errors.append("llm_input.actors is empty or not a list.")
    if len(actors_llm) != len(actors_full):
        warnings.append(
            f"Number of llm_input.actors ({len(actors_llm)}) != "
            f"top-level actors ({len(actors_full)})."
        )

    # Build mapping: rank_s3 -> full_actor
    full_by_rank = {}
    for a in actors_full:
        r = a.get("rank_s3")
        if r is None:
            warnings.append("Top-level actor missing rank_s3.")
            continue
        full_by_rank[int(r)] = a

    # Check each llm actor
    for la in actors_llm:
        gat_rank = la.get("gat_rank")
        if gat_rank is None:
            errors.append("llm actor missing gat_rank.")
            continue

        actor_id = la.get("actor_id", "")
        expected_actor_id = f"ACTOR{gat_rank}"
        if actor_id != expected_actor_id:
            errors.append(
                f"actor_id='{actor_id}' but expected '{expected_actor_id}' for gat_rank={gat_rank}."
            )

        fa = full_by_rank.get(int(gat_rank))
        if fa is None:
            warnings.append(f"No top-level actor for gat_rank={gat_rank}.")
            continue

        # Check a couple of invariants
        if la.get("category") != fa.get("category"):
            errors.append(
                f"Category mismatch for ACTOR{gat_rank}: "
                f"llm='{la.get('category')}', full='{fa.get('category')}'."
            )

        if int(la.get("is_vehicle", 0)) != int(fa.get("is_vehicle", 0)):
            errors.append(
                f"is_vehicle mismatch for ACTOR{gat_rank}."
            )

        if int(la.get("is_vulnerable_road_user", 0)) != int(fa.get("is_vru", 0)):
            errors.append(
                f"is_vulnerable_road_user vs is_vru mismatch for ACTOR{gat_rank}."
            )

        # on_ego_path should be 0 or 1
        oep = la.get("on_ego_path")
        if oep not in (0, 1):
            errors.append(
                f"on_ego_path must be 0 or 1 for ACTOR{gat_rank}, got={oep}."
            )

    # --- ego_map_summary consistency ---
    map_full = data.get("map", {})
    map_llm = llm.get("ego_map_summary", {})

    for key_full, key_llm in [
        ("min_dist_to_crosswalk_m", "min_distance_to_crosswalk_m"),
        ("min_dist_to_stopline_m", "min_distance_to_stopline_m"),
    ]:
        if key_llm in map_llm and key_full in map_full:
            if not approx_equal(map_llm[key_llm], map_full[key_full]):
                warnings.append(
                    f"ego_map_summary.{key_llm} != map.{key_full} ("
                    f"{map_llm[key_llm]} vs {map_full[key_full]})."
                )

    # --- scenario_hints coherence ---
    hints_full = data.get("hints", {})
    hints_llm = llm.get("scenario_hints", {})

    if hints_llm.get("primary_interaction_side") != hints_full.get("primary_side"):
        warnings.append("scenario_hints.primary_interaction_side != hints.primary_side.")

    for k_full, k_llm in [
        ("near_crosswalk", "ego_near_crosswalk"),
        ("near_stopline", "ego_near_stopline"),
        ("has_close_actor", "has_close_actor"),
    ]:
        if k_llm in hints_llm and k_full in hints_full:
            if bool(hints_llm[k_llm]) != bool(hints_full[k_full]):
                warnings.append(
                    f"scenario_hints.{k_llm} != hints.{k_full}."
                )

    # --- context_brief doesn't leak scenario labels ---
    context_brief = llm.get("context_brief", "") or ""
    lower_cb = context_brief.lower()
    for token in SCENARIO_LABEL_TOKENS:
        if token in lower_cb:
            warnings.append(
                f"context_brief contains scenario token '{token}' "
                f"(potential label leakage)."
            )

    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True,
                    help="Directory containing S4.1+S4.2 JSON evidence files.")
    args = ap.parse_args()

    root = Path(args.evidence_dir)
    files = sorted(root.glob("*.json"))

    total = len(files)
    n_ok = 0
    n_err = 0
    n_warn = 0

    print(f"Scanning {total} JSON files in {root}...")

    for path in files:
        errors, warnings = check_file(path)

        if errors:
            n_err += 1
            print(f"[ERROR] {path.name}")
            for e in errors:
                print("   -", e)
        if warnings:
            n_warn += 1
            print(f"[WARN] {path.name}")
            for w in warnings:
                print("   -", w)
        if not errors and not warnings:
            n_ok += 1

    print("\nSummary:")
    print(f"  Total files: {total}")
    print(f"  OK:          {n_ok}")
    print(f"  With errors: {n_err}")
    print(f"  With warns:  {n_warn}")


if __name__ == "__main__":
    main()
