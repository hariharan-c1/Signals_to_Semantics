#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S4.1 Evidence Pack Builder (thesis version with tags + fixes + extra heuristics)

- Reads:
    * S3 slices:     s3/quasi/slices.parquet
    * S1D scores:    s1d/final_scores.parquet
    * S3 Top-3:      s3/top3.parquet
    * S0 features:   features/<split>_feature.parquet
    * GT labels:     s1a/window_labels.jsonl        (optional)
    * Brake episodes: brakes_all_union.jsonl        (optional)
    * Media manifest: media_manifest.(csv|parquet)  (optional)

- Writes:
    * <outdir>/<split>/<safe-window-id>.json
    * <outdir>/<split>/manifest.parquet

Each evidence JSON has:
    - window_key, log_id
    - episode: {t_on, t_peak, t_off, score_final, peak_decel_mps2}
    - actors: Top-3 actors with S0 features + semantic tags + summary
      (now also: is_stationary_actor, relative_heading_bucket,
       distance_trend_bucket, cross_direction)
    - map: aggregated map context derived from actor map features
    - media: static_map_path / clips (nullable)
    - hints: primary_side, near_crosswalk, near_stopline, has_close_actor
    - ground_truth: {label, scenario_label} (if available)
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from signals_to_semantics.identifiers import safe_window_stem


# ---------------------------- basic utils ---------------------------- #

def read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, sep="," if ext == ".csv" else "\t")
    if ext in [".jsonl", ".json"]:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def clip_pos(x, lo: float):
    if pd.isna(x):
        return x
    return float(x) if x >= lo else lo


def normalize_ttc(ttc_raw: float) -> float:
    """
    TTC normalisation for LLM-friendly semantics.

    - Preserve NaN if missing.
    - Any negative TTC (already passed / diverging) is clamped to 0.01,
      so the LLM interprets it as *extremely urgent* rather than 'safe'.
    - Very tiny positive TTCs are clamped to at least 0.1 (optional, for stability).
    """
    ttc = to_float(ttc_raw)
    if pd.isna(ttc):
        return ttc
    if ttc < 0.0:
        return 0.01
    if ttc < 0.1:
        return 0.1
    return ttc


# ---------------------------- semantic tagging helpers ---------------------------- #

def bucket_distance(r_rel_m: float,
                    dist_lt3: float,
                    dist_lt5: float,
                    dist_lt8: float) -> str:
    """
    Distance buckets, roughly matching previous behaviour:

        - very_close: r<=6m or dist_lt3==1
        - close:      else if dist_lt5==1 or r<=12m
        - far:        else if dist_lt8==1 or r<=18m
        - very_far:   otherwise
    """
    r = to_float(r_rel_m)
    if pd.isna(r):
        return "unknown"

    if dist_lt3 == 1.0 or r <= 6.0:
        return "very_close"
    if dist_lt5 == 1.0 or r <= 12.0:
        return "close"
    if dist_lt8 == 1.0 or r <= 18.0:
        return "far"
    return "very_far"


def bucket_side(x_rel_m: float, y_rel_m: float, lat_offset_m: float) -> str:
    """
    Coarse ego-relative side bucket.

    Released sign convention: y_rel_m < 0 means actor on the ego's right.
    """
    y = to_float(y_rel_m)
    if pd.isna(y):
        # fallback to lat_offset if needed
        y = to_float(lat_offset_m)

    if pd.isna(y):
        return "front"

    # front-left / front-right
    if y > 0.5:
        return "front_left"
    if y < -0.5:
        return "front_right"
    return "front"


def bucket_ttc(ttc_s: float) -> str:
    """
    TTC bucket purely from TTC value (seconds).
        - critical: ttc <= 1.0
        - low:      ttc <= 2.5
        - medium:   ttc <= 5.0
        - high:     ttc  > 5.0
    """
    ttc = to_float(ttc_s)
    if pd.isna(ttc):
        return "unknown"
    if ttc <= 1.0:
        return "critical"
    if ttc <= 2.5:
        return "low"
    if ttc <= 5.0:
        return "medium"
    return "high"


def bucket_heading(heading_align_cos: float) -> str:
    """
    Bucket relative heading from cosine of heading alignment:

        - same:     cos >= 0.5
        - opposite: cos <= -0.5
        - crossing: otherwise
    """
    c = to_float(heading_align_cos)
    if pd.isna(c):
        return "unknown"
    if c >= 0.5:
        return "same"
    if c <= -0.5:
        return "opposite"
    return "crossing"


def bucket_distance_trend(rel_speed_closing_mps: float) -> str:
    """
    Bucket distance trend based on relative closing speed:

        - closing:   v > 0.5
        - diverging: v < -0.5
        - constant:  otherwise
    """
    v = to_float(rel_speed_closing_mps)
    if pd.isna(v):
        return "unknown"
    if v > 0.5:
        return "closing"
    if v < -0.5:
        return "diverging"
    return "constant"


def infer_cross_direction(rel_speed_closing_mps: float,
                          lat_speed_mps: float) -> str:
    """
    Coarse 'trajectory direction' marker:

        - lateral_crossing: lateral motion dominates longitudinal
        - longitudinal:     longitudinal motion dominates or equal
        - unknown:          both NaN
    """
    v_long = abs(to_float(rel_speed_closing_mps))
    v_lat = abs(to_float(lat_speed_mps))

    if pd.isna(v_long) and pd.isna(v_lat):
        return "unknown"

    if pd.isna(v_long):
        v_long = 0.0
    if pd.isna(v_lat):
        v_lat = 0.0

    if v_lat > v_long:
        return "lateral_crossing"
    return "longitudinal"


def infer_trajectory_bucket(on_path_like: float,
                            crossing_like: float,
                            approach_like: float,
                            lat_offset_m: float,
                            r_rel_m: float = None) -> str:
    """
    Coarse interaction type:

        - crossing_like:
            crossing_like high
        - cut_in_like:
            approaching with sizable lateral offset (independent of on_path),
            reasonably not-too-far
        - lead_brake_like:
            on-path, relatively small lateral offset, approaching
        - other: fallback
    """
    onp = to_float(on_path_like)
    crs = to_float(crossing_like)
    appr = to_float(approach_like)
    lat = abs(to_float(lat_offset_m))
    r = to_float(r_rel_m)

    if pd.isna(onp):
        onp = 0.0
    if pd.isna(crs):
        crs = 0.0
    if pd.isna(appr):
        appr = 0.0
    if pd.isna(lat):
        lat = 0.0
    if pd.isna(r):
        r = 999.0  # effectively "far"

    # 1) crossing-like dominates
    if crs >= 0.5:
        return "crossing_like"

    # 2) cut-in-like: approaching + big lateral offset + not absurdly far
    if appr >= 0.5 and lat >= 1.5 and r <= 25.0:
        return "cut_in_like"

    # 3) lead-brake-like: on-path + approaching + small lateral offset
    if onp >= 0.5 and appr >= 0.5 and lat < 1.5:
        return "lead_brake_like"

    return "other"


def infer_role_hint(category: str,
                    is_vru: int,
                    trajectory_bucket: str,
                    side_bucket: str,
                    lat_offset_m: float) -> str:
    """
    Very coarse object role w.r.t. ego:

        - crossing_vehicle
        - lead_vehicle
        - side_pedestrian
        - other
    """
    cat = (category or "").upper()
    vru = int(is_vru or 0)
    lat = abs(to_float(lat_offset_m))
    traj = trajectory_bucket or "other"
    side = side_bucket or "front"

    # side pedestrians / VRUs
    if vru == 1 or "PEDESTRIAN" in cat:
        if lat >= 3.0:
            return "side_pedestrian"
        # front-ish pedestrian will later be handled by the LLM via tags
        return "other"

    # vehicles crossing ego path
    if traj == "crossing_like":
        return "crossing_vehicle"

    # lead vehicle in front
    if traj in ("lead_brake_like", "cut_in_like") and side.startswith("front"):
        return "lead_vehicle"

    return "other"


def refine_role_hint_for_vru(actor: dict) -> str:
    """
    Refine VRU (pedestrian/cyclist) role hints using richer semantics.

    Upgrade VRU roles when they are clearly crossing:
        - is_vru == True
        - crossing_like is True-ish
    """
    role_hint = actor.get("role_hint", "other")
    category = actor.get("category", "")
    is_vru = bool(actor.get("is_vru", 0))
    crossing_like = bool(actor.get("crossing_like", 0.0))
    _ = category  # currently unused, kept for future logic

    # Upgrade crossing VRUs
    if is_vru and crossing_like:
        return "crossing_pedestrian"  # specific VRU crossing role

    return role_hint


def actor_is_close(actor: dict) -> bool:
    """
    'Close' actor semantics for hints.has_close_actor.

    Rules:
        1) Hard-close:
            - dist_lt3 == 1.0  (within ~3 m)
        2) Lead / on-path critical:
            - on_path_like >= 0.5
            - TTC <= 1.5 s
            - r_rel_m <= 30 m
        3) Crossing actor:
            - crossing_like >= 0.5
            - TTC <= 2.0 s
            - dist_to_crosswalk <= 10 m
        4) Generic near-field:
            - distance_bucket in {very_close, close}
            - TTC <= 3.0 s
    """
    r = to_float(actor.get("r_rel_m"))
    ttc = to_float(actor.get("ttc_s"))
    dist_lt3 = to_float(actor.get("dist_lt3"))
    crossing_like = to_float(actor.get("crossing_like"))
    dist_to_xwalk = to_float(actor.get("dist_to_crosswalk_m"))
    on_path_like = to_float(actor.get("on_path_like"))

    # distance bucket (for rule 4)
    dist_bucket = bucket_distance(
        r,
        to_float(actor.get("dist_lt3")),
        to_float(actor.get("dist_lt5")),
        to_float(actor.get("dist_lt8")),
    )

    # 1) Hard close: within 3m
    if dist_lt3 == 1.0:
        return True

    # 2) Lead / on-path critical: ego closing on something in its lane
    if (
        not pd.isna(on_path_like)
        and on_path_like >= 0.5
        and not pd.isna(ttc)
        and ttc <= 1.5
        and not pd.isna(r)
        and r <= 30.0
    ):
        return True

    # 3) Crossing actors near crosswalk
    if (
        not pd.isna(crossing_like)
        and crossing_like >= 0.5
        and not pd.isna(ttc)
        and ttc <= 2.0
        and not pd.isna(dist_to_xwalk)
        and dist_to_xwalk <= 10.0
    ):
        return True

    # 4) Generic near-field: close-ish and short-ish TTC
    if (
        dist_bucket in {"very_close", "close"}
        and not pd.isna(ttc)
        and ttc <= 3.0
    ):
        return True

    return False


def summarize_actor(category: str,
                    r_rel_m: float,
                    side_bucket: str,
                    distance_bucket: str,
                    ttc_s: float,
                    trajectory_bucket: str,
                    role_hint: str) -> str:
    """
    Short natural-language summary for the actor.
    """
    cat = (category or "object").lower().replace("_", " ")
    r = to_float(r_rel_m)
    ttc = to_float(ttc_s)

    if pd.isna(r):
        r_str = "at unknown distance"
    else:
        r_str = f"~{r:.1f} m away"

    if pd.isna(ttc):
        ttc_str = "with unknown TTC"
    else:
        ttc_str = f"with TTC≈{ttc:.1f} s"

    side_str = side_bucket.replace("_", " ")
    dist_str = distance_bucket.replace("_", " ")
    traj_str = trajectory_bucket.replace("_", " ")

    if role_hint and role_hint != "other":
        role_str = f", (role: {role_hint.replace('_', ' ')})"
    else:
        role_str = ""

    return (
        f"A {cat}, {r_str}, on the {side_str}, ({dist_str}), "
        f"{ttc_str}, showing {traj_str} behaviour{role_str}."
    )


def primary_side_from_tags(tags: dict) -> str:
    """
    Map side_bucket -> primary_side used in hints.
        - front_left  -> left
        - front_right -> right
        - front/other -> center
    """
    sb = (tags or {}).get("side_bucket", "")
    if "left" in sb:
        return "left"
    if "right" in sb:
        return "right"
    return "center"


# ---------------------------- map summary ---------------------------- #

def build_map_summary(actors_json: List[dict]) -> dict:
    """
    Aggregate map-derived features from actor rows into a window-level summary.
    This becomes the `map` field in the evidence JSON.
    """
    if not actors_json:
        return {}

    d_stop = []
    d_cross = []
    lane_offsets = []
    lane_aligns = []
    in_drive = 0
    off_drive = 0
    vru_near_cross = 0
    ped_near_cross = 0

    for a in actors_json:
        ds = to_float(a.get("dist_to_stopline_m"))
        dc = to_float(a.get("dist_to_crosswalk_m"))
        lo = to_float(a.get("map_lane_offset_m"))
        la = to_float(a.get("map_lane_alignment_cos"))
        in_da = int(a.get("in_drivable_area", 1))
        cat = str(a.get("category", "") or "")

        if not pd.isna(ds):
            d_stop.append(ds)
        if not pd.isna(dc):
            d_cross.append(dc)
        if not pd.isna(lo):
            lane_offsets.append(lo)
        if not pd.isna(la):
            lane_aligns.append(la)

        if in_da == 1:
            in_drive += 1
        else:
            off_drive += 1

        if dc is not None and not pd.isna(dc) and dc <= 10.0:
            if a.get("is_vru", 0) == 1:
                vru_near_cross += 1
            if "PEDESTRIAN" in cat.upper():
                ped_near_cross += 1

    primary = actors_json[0]
    primary_lane_offset = to_float(primary.get("map_lane_offset_m"))
    primary_lane_align = to_float(primary.get("map_lane_alignment_cos"))

    summary = {
        "min_dist_to_stopline_m": float(min(d_stop)) if d_stop else None,
        "min_dist_to_crosswalk_m": float(min(d_cross)) if d_cross else None,
        "primary_lane_offset_m": primary_lane_offset,
        "primary_lane_alignment_cos": primary_lane_align,
        "num_actors_in_drivable_area": int(in_drive),
        "num_actors_off_drivable_area": int(off_drive),
        "num_vru_near_crosswalk": int(vru_near_cross),
        "num_ped_near_crosswalk": int(ped_near_cross),
    }

    # crude road-type hint
    near_cross_all = (summary["min_dist_to_crosswalk_m"] is not None
                      and summary["min_dist_to_crosswalk_m"] <= 15.0)
    near_stop_all = (summary["min_dist_to_stopline_m"] is not None
                     and summary["min_dist_to_stopline_m"] <= 20.0)
    if near_cross_all or near_stop_all:
        summary["road_type_hint"] = "urban"
    else:
        summary["road_type_hint"] = "highway_like"

    return summary


# ---------------------------- core builder ---------------------------- #

def build_evidence(
    slices_path: str,
    final_scores_path: str,
    top3_path: str,
    features_path: str,
    outdir: str,
    split: str,
    time_tol: float,
    media_manifest_path: Optional[str] = None,
    brakes_path: Optional[str] = None,
    window_labels_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:

    warnings: List[str] = []

    # --- read inputs ---
    slices = read_table(slices_path).copy()
    finals = read_table(final_scores_path).copy()
    top3 = read_table(top3_path).copy()
    feats = read_table(features_path).copy()

    # normalize slices
    if "t_off" not in slices.columns:
        slices["t_off"] = slices.get("window_t_end", slices.get("t1", np.nan))
    if "window_key" not in slices.columns:
        slices["window_key"] = (
            slices["log_id"].astype(str)
            + "|"
            + slices["window_t_start"].map(str)
            + "|"
            + slices["window_t_end"].map(str)
        )

    # --- attach S1D scores to slices ---
    fs = finals.rename(columns={"t_start": "fs_t_start", "t_end": "fs_t_end"})
    fs_j = slices.merge(fs, on="log_id", how="left", suffixes=("", "_fs"))
    fs_j["_dt_start"] = (fs_j["window_t_start"] - fs_j["fs_t_start"]).abs()
    fs_j["_dt_end"] = (fs_j["window_t_end"] - fs_j["fs_t_end"]).abs()
    fs_j["_dt_sum"] = fs_j["_dt_start"] + fs_j["_dt_end"]
    fs_j = fs_j[(fs_j["_dt_start"] <= time_tol) & (fs_j["_dt_end"] <= time_tol)]
    fs_best = fs_j.sort_values("_dt_sum").drop_duplicates("window_key", keep="first")
    slices = slices.merge(
        fs_best[["window_key", "score_final", "post_hmm"]],
        on="window_key",
        how="left",
    )
    slices["score_final"] = slices["score_final"].fillna(slices.get("post_hmm"))

    # --- attach GT labels (optional) ---
    if window_labels_path:
        labels = read_table(window_labels_path)
        labels = labels.rename(
            columns={
                "t_start": "wl_t_start",
                "t_end": "wl_t_end",
                "scenario_label": "gt_scenario_label",
                "label": "gt_label",
            }
        )
        lab = slices.merge(labels, on="log_id", how="left")
        lab["_dt_start"] = (lab["window_t_start"] - lab["wl_t_start"]).abs()
        lab["_dt_end"] = (lab["window_t_end"] - lab["wl_t_end"]).abs()
        lab["_dt_sum"] = lab["_dt_start"] + lab["_dt_end"]
        lab = lab[(lab["_dt_start"] <= time_tol) & (lab["_dt_end"] <= time_tol)]
        lab_best = lab.sort_values("_dt_sum").drop_duplicates("window_key", keep="first")
        slices = slices.merge(
            lab_best[["window_key", "gt_label", "gt_scenario_label"]],
            on="window_key",
            how="left",
        )

    # --- brakes (for peak_decel_mps2) ---
    brakes = None
    if brakes_path:
        brakes = read_table(brakes_path).rename(
            columns={"t_start": "br_t_start", "t_end": "br_t_end"}
        )

    # --- media (optional) ---
    media = None
    if media_manifest_path:
        media = read_table(media_manifest_path)
        if "log_id" not in media.columns:
            warnings.append("media manifest missing 'log_id'; ignoring media.")
            media = None

    # --- features index ---
    key_cols = {"log_id", "window_t_start", "window_t_end", "window_center", "track_uuid"}
    if not key_cols.issubset(feats.columns):
        missing = key_cols - set(feats.columns)
        raise ValueError(f"features file missing required keys: {missing}")

    feats_idx = feats.set_index(["log_id", "track_uuid"]).sort_index()

    # --- top3 prep ---
    t3 = top3.rename(columns={"score": "s3_score", "rank": "rank_s3"}).copy()
    t3["rank_s3"] = t3["rank_s3"].astype(int)

    # --- out dir ---
    out_root = Path(outdir) / split
    ensure_dir(out_root)

    rows = []

    # group by window_key
    for window_key, group in t3.groupby("window_key", sort=False):
        srow = slices.loc[slices["window_key"] == window_key]
        if srow.empty:
            warnings.append(f"[{window_key}] missing in slices; skip window.")
            continue
        srow = srow.iloc[0]

        log_id = srow["log_id"]
        t_peak = float(srow.get("t_peak", srow["window_t_start"]))
        t_on = float(srow.get("t_on", srow["window_t_start"]))
        t_off = float(srow.get("t_off", srow["window_t_end"]))
        score_final = (
            float(srow.get("score_final"))
            if not pd.isna(srow.get("score_final"))
            else None
        )

        # --- peak decel from brakes_all_union (optional) ---
        peak_decel = None
        if brakes is not None:
            b = brakes.loc[brakes["log_id"] == log_id].copy()
            if not b.empty:
                b["_dt_start"] = (b["br_t_start"] - srow["window_t_start"]).abs()
                b["_dt_end"] = (b["br_t_end"] - srow["window_t_end"]).abs()
                b["_dt_sum"] = b["_dt_start"] + b["_dt_end"]
                b = b[(b["_dt_start"] <= time_tol) & (b["_dt_end"] <= time_tol)]
                if not b.empty and "a_min_ms2" in b.columns:
                    peak_decel = abs(float(b.sort_values("_dt_sum").iloc[0]["a_min_ms2"]))

        # --- actor features for Top-3 ---
        actors_json = []
        group_sorted = group.sort_values("rank_s3")
        for _, arow in group_sorted.iterrows():
            track_uuid = arow["track_uuid"]
            rank_s3 = int(arow["rank_s3"])
            s3_score = float(arow["s3_score"])

            try:
                cand = feats_idx.loc[(log_id, track_uuid)]
                if isinstance(cand, pd.Series):
                    cand = cand.to_frame().T
                cand = cand.copy()
                cand["_abs_dt_"] = (cand["window_center"] - t_peak).abs()
                cand = cand.sort_values("_abs_dt_")
                if cand["_abs_dt_"].iloc[0] > time_tol:
                    warnings.append(
                        f"[{window_key}] actor {track_uuid} has no feature row within {time_tol}s; skip actor."
                    )
                    continue
                f = cand.iloc[0].to_dict()

                rel_closing = to_float(f.get("rel_speed_closing_mps"))
                lat_speed = to_float(f.get("lat_speed_mps"))

                # stationarity heuristic: static flag OR very low velocities
                is_static_label = int(f.get("is_static", 0))
                is_stationary_actor = int(
                    (is_static_label == 1)
                    or (
                        (not pd.isna(rel_closing) and abs(rel_closing) < 0.3)
                        and (not pd.isna(lat_speed) and abs(lat_speed) < 0.3)
                    )
                )

                # base actor object
                actor_obj = {
                    "track_uuid": track_uuid,
                    "rank_s3": rank_s3,
                    "s3_score": s3_score,
                    "category": f.get("category"),
                    "is_vehicle": int(f.get("is_vehicle", 0)),
                    "is_vru": int(f.get("is_vru", 0)),
                    "is_static": is_static_label,
                    "is_stationary_actor": is_stationary_actor,
                    "sector_id": int(f.get("sector_id", -1)),
                    "on_path_like": to_float(f.get("on_path_like")),
                    "x_rel_m": to_float(f.get("x_rel_m")),
                    "y_rel_m": to_float(f.get("y_rel_m")),
                    "r_rel_m": clip_pos(to_float(f.get("r_rel_m")), 0.0),
                    "dmin_m": to_float(f.get("dmin_m")),
                    "t_at_dmin_s": to_float(f.get("t_at_dmin_s")),
                    "sustained_tight": to_float(f.get("sustained_tight")),
                    "rel_speed_closing_mps": rel_closing,
                    "lat_speed_mps": lat_speed,
                    "a_norm": to_float(f.get("a_norm")),
                    "length_m": to_float(f.get("length_m")),
                    "width_m": to_float(f.get("width_m")),
                    "bearing_rad": to_float(f.get("bearing_rad")),
                    "ttc_s": normalize_ttc(f.get("ttc_s")),
                    "dist_lt3": to_float(f.get("dist_lt3")),
                    "dist_lt5": to_float(f.get("dist_lt5")),
                    "dist_lt8": to_float(f.get("dist_lt8")),
                    "approach_like": to_float(f.get("approach_like")),
                    "crossing_like": to_float(f.get("crossing_like")),
                    "heading_align_cos": to_float(f.get("heading_align_cos")),
                    "long_gap_m": to_float(f.get("long_gap_m")),
                    "lat_offset_m": to_float(f.get("lat_offset_m")),
                    "map_lane_offset_m": to_float(f.get("map_lane_offset_m")),
                    "map_lane_alignment_cos": to_float(f.get("map_lane_alignment_cos")),
                    "dist_to_stopline_m": to_float(f.get("dist_to_stopline_m")),
                    "dist_to_crosswalk_m": to_float(f.get("dist_to_crosswalk_m")),
                    "in_drivable_area": int(f.get("in_drivable_area", 1)),
                }

                # derived heuristics
                heading_bucket = bucket_heading(actor_obj["heading_align_cos"])
                distance_trend_bucket = bucket_distance_trend(
                    actor_obj["rel_speed_closing_mps"]
                )
                cross_direction = infer_cross_direction(
                    actor_obj["rel_speed_closing_mps"],
                    actor_obj["lat_speed_mps"],
                )

                # semantic tags
                dist_bucket = bucket_distance(
                    actor_obj["r_rel_m"],
                    actor_obj["dist_lt3"],
                    actor_obj["dist_lt5"],
                    actor_obj["dist_lt8"],
                )
                side_bucket = bucket_side(
                    actor_obj["x_rel_m"],
                    actor_obj["y_rel_m"],
                    actor_obj["lat_offset_m"],
                )
                ttc_bucket = bucket_ttc(actor_obj["ttc_s"])
                traj_bucket = infer_trajectory_bucket(
                    actor_obj["on_path_like"],
                    actor_obj["crossing_like"],
                    actor_obj["approach_like"],
                    actor_obj["lat_offset_m"],
                    actor_obj["r_rel_m"],
                )

                base_role_hint = infer_role_hint(
                    actor_obj["category"],
                    actor_obj["is_vru"],
                    traj_bucket,
                    side_bucket,
                    actor_obj["lat_offset_m"],
                )

                # refine role hint for VRUs (e.g., crossing pedestrians)
                actor_obj["role_hint"] = refine_role_hint_for_vru({
                    **actor_obj,
                    "role_hint": base_role_hint,
                })
                role_hint = actor_obj["role_hint"]

                tags = {
                    "distance_bucket": dist_bucket,
                    "side_bucket": side_bucket,
                    "ttc_bucket": ttc_bucket,
                    "trajectory_bucket": traj_bucket,
                    "role_hint": role_hint,
                    "relative_heading_bucket": heading_bucket,
                    "distance_trend_bucket": distance_trend_bucket,
                    "cross_direction": cross_direction,
                }

                summary = summarize_actor(
                    actor_obj["category"],
                    actor_obj["r_rel_m"],
                    side_bucket,
                    dist_bucket,
                    actor_obj["ttc_s"],
                    traj_bucket,
                    role_hint,
                )

                # close-actor flag using upgraded semantics
                is_close_flag = actor_is_close(actor_obj)

                actor_obj["relative_heading_bucket"] = heading_bucket
                actor_obj["distance_trend_bucket"] = distance_trend_bucket
                actor_obj["cross_direction"] = cross_direction
                actor_obj["tags"] = tags
                actor_obj["summary"] = summary
                actor_obj["is_close"] = bool(is_close_flag)

                actors_json.append(actor_obj)
            except KeyError:
                warnings.append(
                    f"[{window_key}] actor {track_uuid} not found in features; skip actor."
                )
                continue

        if not actors_json:
            warnings.append(f"[{window_key}] no actors resolved; skip window.")
            continue

        # --- map summary from actors ---
        map_obj = build_map_summary(actors_json)

        # --- hints ---
        primary_actor = actors_json[0]
        primary_tags = primary_actor.get("tags", {})
        primary_side = primary_side_from_tags(primary_tags)

        d_cross_prim = to_float(primary_actor.get("dist_to_crosswalk_m"))
        d_stop_prim = to_float(primary_actor.get("dist_to_stopline_m"))

        near_crosswalk = (
            (not pd.isna(d_cross_prim)) and (d_cross_prim <= 10.0)
        )
        near_stopline = (
            (not pd.isna(d_stop_prim)) and (d_stop_prim <= 12.0)
        )

        has_close_actor = any(a.get("is_close", False) for a in actors_json)

        # --- media (per log_id) ---
        media_obj = {
            "static_map_path": None,
            "ego_actor1_clip": None,
            "ego_actor2_clip": None,
            "ego_actor3_clip": None,
        }
        if media is not None:
            mrow = media.loc[media["log_id"] == log_id]
            if not mrow.empty:
                md = mrow.iloc[0].to_dict()
                for k in media_obj.keys():
                    if k in md:
                        media_obj[k] = md[k]

        # --- build evidence JSON ---
        evidence = {
            "window_key": window_key,
            "log_id": str(log_id),
            "episode": {
                "t_on": float(t_on),
                "t_peak": float(t_peak),
                "t_off": float(t_off),
                "score_final": float(score_final) if score_final is not None else None,
                "peak_decel_mps2": float(peak_decel) if peak_decel is not None else None,
            },
            "actors": actors_json,
            "map": map_obj,
            "media": media_obj,
            "hints": {
                "primary_side": primary_side,
                "near_crosswalk": bool(near_crosswalk),
                "near_stopline": bool(near_stopline),
                "has_close_actor": bool(has_close_actor),
            },
        }

        # ground truth metadata (if available)
        if "gt_label" in srow and "gt_scenario_label" in srow:
            evidence["ground_truth"] = {
                "label": int(srow["gt_label"])
                if not pd.isna(srow["gt_label"])
                else None,
                "scenario_label": str(srow["gt_scenario_label"])
                if pd.notna(srow["gt_scenario_label"])
                else None,
            }

        # --- write JSON ---
        out_path = out_root / f"{safe_window_stem(window_key)}.json"
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)

        # --- manifest row ---
        rows.append(
            {
                "window_key": window_key,
                "evidence_file": out_path.name,
                "log_id": log_id,
                "n_actors": len(actors_json),
                "has_media": any(v is not None for v in media_obj.values()),
                "score_final": score_final,
                "primary_side": primary_side,
                "near_crosswalk": near_crosswalk,
                "near_stopline": near_stopline,
                "has_close_actor": has_close_actor,
                "gt_scenario_label": evidence.get("ground_truth", {}).get(
                    "scenario_label"
                )
                if "ground_truth" in evidence
                else None,
                "peak_decel_mps2": peak_decel,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = Path(outdir) / split / "manifest.parquet"
    ensure_dir(manifest_path.parent)
    manifest.to_parquet(manifest_path, index=False)

    if manifest.empty:
        warnings.append("No evidence files written. Check inputs / joins.")

    return manifest, warnings


# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--final-scores", required=True)
    ap.add_argument("--top3", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--time-tol", type=float, default=0.25)
    ap.add_argument("--media-manifest", default=None)
    ap.add_argument("--brakes", default=None, help="brakes_all_union.jsonl")
    ap.add_argument("--window-labels", default=None, help="s1a/window_labels.jsonl")
    args = ap.parse_args()

    manifest, warnings = build_evidence(
        slices_path=args.slices,
        final_scores_path=args.final_scores,
        top3_path=args.top3,
        features_path=args.features,
        outdir=args.outdir,
        split=args.split,
        time_tol=args.time_tol,
        media_manifest_path=args.media_manifest,
        brakes_path=args.brakes,
        window_labels_path=args.window_labels,
    )

    print(
        f"[S4.1] Evidence files: {len(manifest)} written to {args.outdir}/{args.split}"
    )
    if not manifest.empty:
        print(manifest.head(5).to_string(index=False))

    if warnings:
        print("\n[S4.1] WARNINGS:")
        for w in warnings[:50]:
            print(" -", w)
        if len(warnings) > 50:
            print(f" ... (+{len(warnings)-50} more)")


if __name__ == "__main__":
    main()
