#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S4.2: Build llm_input from S4.1 evidence JSONs.

- Reads each evidence JSON produced by s4.1_build_evidence.py
- Adds a `llm_input` field, containing the compact, LLM-friendly schema.

This version:
- Exposes additional heuristics:
    * is_stationary_actor
    * heuristic_relative_heading_bucket
    * heuristic_distance_trend_bucket
    * heuristic_cross_direction
- Keeps negative t_at_dmin_s as-is (semantically meaningful).
"""

import json
import argparse
from pathlib import Path


def float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def bool_flag(x):
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.lower() in ("1", "true", "yes")
    return False


def build_llm_actor(a):
    """
    Convert a single S4.1 actor dict -> llm_input actor schema.

    Assumes S4.1 has already computed:
      - tags.distance_bucket / side_bucket / ttc_bucket / trajectory_bucket / role_hint
      - is_stationary_actor, relative_heading_bucket, distance_trend_bucket, cross_direction
    """
    tags = a.get("tags", {}) or {}

    # fallback for buckets: prefer top-level field, otherwise tags
    rel_heading_bucket = a.get("relative_heading_bucket") or tags.get("relative_heading_bucket")
    dist_trend_bucket = a.get("distance_trend_bucket") or tags.get("distance_trend_bucket")
    cross_direction = a.get("cross_direction") or tags.get("cross_direction")

    return {
        "actor_id": f"ACTOR{a['rank_s3']}",

        "category": a.get("category"),
        "is_vehicle": a.get("is_vehicle", 0),
        "is_vulnerable_road_user": a.get("is_vru", 0),
        "is_static": a.get("is_static", 0),
        "is_stationary_actor": a.get("is_stationary_actor", 0),

        # --- Evidence geometry / dynamics ---
        "ego_forward_offset_m": float_or_none(a.get("x_rel_m")),
        "ego_lateral_offset_m": float_or_none(a.get("y_rel_m")),
        "lane_center_offset_m": float_or_none(a.get("lat_offset_m")),

        "closest_distance_during_episode_m": float_or_none(a.get("dmin_m")),
        # NOTE: may be negative; we keep this as-is, semantics explained in prompt.
        "time_when_closest_s": float_or_none(a.get("t_at_dmin_s")),

        "relative_closing_speed_mps": float_or_none(a.get("rel_speed_closing_mps")),
        "relative_lateral_speed_mps": float_or_none(a.get("lat_speed_mps")),
        "relative_motion_norm_mps2": float_or_none(a.get("a_norm")),
        "time_to_collision_s": float_or_none(a.get("ttc_s")),

        "ever_within_3m": bool_flag(a.get("dist_lt3")),

        "distance_to_crosswalk_m": float_or_none(a.get("dist_to_crosswalk_m")),
        "distance_to_stopline_m": float_or_none(a.get("dist_to_stopline_m")),
        "in_drivable_area": a.get("in_drivable_area", 1),

        "lane_alignment_cosine": float_or_none(a.get("map_lane_alignment_cos")),
        "heading_alignment_cosine": float_or_none(a.get("heading_align_cos")),

        "length_m": float_or_none(a.get("length_m")),
        "width_m": float_or_none(a.get("width_m")),

        # --- GAT priors ---
        "gat_rank": int(a.get("rank_s3", 0)),
        "gat_score": float_or_none(a.get("s3_score")),

        # 1 or 0 (discrete to save tokens, as agreed)
        "on_ego_path": 1 if float_or_none(a.get("on_path_like")) is not None
                       and float_or_none(a.get("on_path_like")) >= 0.5 else 0,

        # --- Heuristic priors (semantic tags) ---
        "heuristic_distance_bucket": tags.get("distance_bucket"),
        "heuristic_side_bucket": tags.get("side_bucket"),
        "heuristic_ttc_bucket": tags.get("ttc_bucket"),
        "heuristic_trajectory_type": tags.get("trajectory_bucket"),
        "heuristic_role_hint": tags.get("role_hint"),
        "heuristic_is_close": bool_flag(a.get("is_close")),

        # New heuristic buckets:
        "heuristic_relative_heading_bucket": rel_heading_bucket,
        "heuristic_distance_trend_bucket": dist_trend_bucket,
        "heuristic_cross_direction": cross_direction,
    }


def build_llm_input(data: dict) -> dict:
    """
    Turn a full S4.1 evidence JSON -> llm_input schema.

    This is what we later convert to YAML and feed to the LLM.
    """
    episode = data["episode"]
    actors = data["actors"]
    map_sum = data.get("map", {}) or {}
    hints = data.get("hints", {}) or {}

    # Actor conversion
    llm_actors = [build_llm_actor(a) for a in actors]

    # Duration
    t_start = float_or_none(episode.get("t_on"))
    t_end = float_or_none(episode.get("t_off"))
    if t_start is not None and t_end is not None:
        duration = t_end - t_start
    else:
        duration = None

    peak_dec = float_or_none(episode.get("peak_decel_mps2"))
    score_final = float_or_none(episode.get("score_final"))

    # Construct context_brief in a way that is clean and LLM-friendly
    ctx_parts = []

    if duration is not None:
        ctx_parts.append(f"The ego experiences a braking episode of about {duration:.1f} s")
    else:
        ctx_parts.append("The ego experiences a braking episode of unknown duration")

    if peak_dec is not None:
        ctx_parts.append(f"with peak deceleration of {peak_dec:.3f} m/s2")
    else:
        ctx_parts.append("with unknown peak deceleration")

    if score_final is not None:
        ctx_parts.append(f"and final brake_confidence_score is {score_final:.3f}.")
    else:
        ctx_parts.append("and brake confidence score is unknown.")

    road_type = map_sum.get("road_type_hint", "unknown")
    ctx_parts.append(f"The environment is classified as {road_type}.")

    min_dx = map_sum.get("min_dist_to_crosswalk_m")
    min_ds = map_sum.get("min_dist_to_stopline_m")
    ctx_parts.append(
        f"min_distance_to_crosswalk_m is {min_dx}, "
        f"min_distance_to_stopline_m is {min_ds}."
    )

    ctx_parts.append(
        f"primary_interaction_side is {hints.get('primary_side')}."
    )

    ctx_parts.append(
        f"ego_near_crosswalk={hints.get('near_crosswalk')}, "
        f"ego_near_stopline={hints.get('near_stopline')}, "
        f"has_close_actor={hints.get('has_close_actor')}."
    )

    context_brief = " ".join(ctx_parts)

    # Build final llm_input schema
    llm_input = {
        "ego_window_key": data["window_key"],

        "ego_brake_episode": {
            "t_start_s": float_or_none(episode.get("t_on")),
            "t_peak_brake_s": float_or_none(episode.get("t_peak")),
            "t_end_s": float_or_none(episode.get("t_off")),
            "brake_confidence_score": score_final,
            "peak_deceleration_mps2": peak_dec,
        },

        "context_brief": context_brief,

        "actors": llm_actors,

        "ego_map_summary": {
            "min_distance_to_crosswalk_m": float_or_none(
                map_sum.get("min_dist_to_crosswalk_m")
            ),
            "min_distance_to_stopline_m": float_or_none(
                map_sum.get("min_dist_to_stopline_m")
            ),
            "num_actors_in_drivable_area": map_sum.get("num_actors_in_drivable_area"),
            "num_actors_off_drivable_area": map_sum.get("num_actors_off_drivable_area"),
            "num_vru_near_crosswalk": map_sum.get("num_vru_near_crosswalk"),
            "num_ped_near_crosswalk": map_sum.get("num_ped_near_crosswalk"),
            "road_type_hint": map_sum.get("road_type_hint"),
        },

        "scenario_hints": {
            "primary_interaction_side": hints.get("primary_side"),
            "ego_near_crosswalk": hints.get("near_crosswalk"),
            "ego_near_stopline": hints.get("near_stopline"),
            "has_close_actor": hints.get("has_close_actor"),
        },
    }

    return llm_input


def process_all(evidence_dir: str) -> None:
    """
    Walk over all *.json evidence files in `evidence_dir`,
    attach `llm_input` field, and overwrite in place.
    """
    evidence_path = Path(evidence_dir)
    json_files = sorted(evidence_path.glob("*.json"))
    print(f"[S4.2] Found {len(json_files)} evidence files in {evidence_dir}.")

    for js in json_files:
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)

        llm_input = build_llm_input(data)
        data["llm_input"] = llm_input

        with open(js, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[S4.2] Updated: {js.name}")


def main():
    ap = argparse.ArgumentParser(description="Build llm_input from S4.1 evidence JSONs.")
    ap.add_argument("--evidence-dir", required=True, help="Directory with S4.1 evidence JSON files")
    args = ap.parse_args()

    process_all(args.evidence_dir)


if __name__ == "__main__":
    main()
