#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RANK_COLS = ["rank_p_overlap","rank_inv_dmin"]
TOPK_LIST = [1,3,5,10,24]

KEYS = ["log_id","window_t_start","window_t_end","window_center"]

NUM_COLS_RAW = [
    "p_overlap","dmin_m","rel_speed_closing_mps","lat_speed_mps","a_norm","ttc_s",
    "dist_lt3","dist_lt5","dist_lt8",
    "map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m",
    "in_drivable_area"
]
SNAPSHOT_COLS = [
    "category","is_vehicle","is_vru","is_static","sector_id","on_path_like",
    "x_rel_m","y_rel_m","r_rel_m","dmin_m","p_overlap","t_at_dmin_s",
    "sustained_tight","rel_speed_closing_mps","lat_speed_mps","a_norm","length_m","width_m",
    "bearing_rad","ttc_s","heading_align_cos","long_gap_m","lat_offset_m",
    "map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m","in_drivable_area"
]

def agg_block(df_win: pd.DataFrame):
    out = {}
    out["n_actors"]   = len(df_win)
    out["n_vehicle"]  = int((df_win["is_vehicle"]==1).sum())
    out["n_vru"]      = int((df_win["is_vru"]==1).sum())
    # sector buckets: 0=front,1=left,2=back,3=right
    for sid in [0,1,2,3]:
        out[f"n_sector_{sid}"] = int((df_win["sector_id"]==sid).sum())
    out["sustained_any"]  = float((df_win["sustained_tight"]==1).any())
    out["sustained_frac"] = float((df_win["sustained_tight"]==1).mean() if len(df_win)>0 else 0.0)

    # sort by two notions
    df_by_p   = df_win.sort_values("p_overlap", ascending=False).reset_index(drop=True)
    df_by_inv = df_win.assign(inv_d=1.0/(1.0+df_win["dmin_m"].clip(lower=0.0))) \
                      .sort_values("inv_d", ascending=False).reset_index(drop=True)

    # window-level map cues (min & mean across all actors)
    for col in ["map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m","in_drivable_area"]:
        s = df_win[col]
        out[f"{col}_min"]  = float(np.nanmin(s.values)) if s.notna().any() else np.nan
        out[f"{col}_mean"] = float(np.nanmean(s.values)) if s.notna().any() else np.nan
    out["near_stopline"]   = float((df_win["dist_to_stopline_m"].fillna(np.inf) < 8.0).any())
    out["near_crosswalk"]  = float((df_win["dist_to_crosswalk_m"].fillna(np.inf) < 8.0).any())

    # aggregates over top-k by p_overlap
    for K in TOPK_LIST:
        sl = df_by_p.head(K)
        out[f"top{K}_p_max"]   = float(sl["p_overlap"].max()) if len(sl)>0 else 0.0
        out[f"top{K}_p_mean"]  = float(sl["p_overlap"].mean()) if len(sl)>0 else 0.0
        inv = 1.0/(1.0+sl["dmin_m"].clip(lower=0.0))
        out[f"top{K}_inv_d_max"]  = float(inv.max()) if len(sl)>0 else 0.0
        out[f"top{K}_inv_d_mean"] = float(inv.mean()) if len(sl)>0 else 0.0
        out[f"top{K}_rel_close_mean"] = float(sl["rel_speed_closing_mps"].mean()) if len(sl)>0 else 0.0
        out[f"top{K}_a_norm_mean"]    = float(sl["a_norm"].mean()) if len(sl)>0 else 0.0
        out[f"top{K}_lat_speed_mean"] = float(sl["lat_speed_mps"].mean()) if len(sl)>0 else 0.0
        out[f"top{K}_ttc_min"]        = float(sl["ttc_s"].replace([np.inf, -np.inf], np.nan).min()) if len(sl)>0 else np.nan
        out[f"top{K}_dist3_any"]      = float(sl["dist_lt3"].max()) if len(sl)>0 else 0.0
        out[f"top{K}_dist5_any"]      = float(sl["dist_lt5"].max()) if len(sl)>0 else 0.0
        out[f"top{K}_dist8_any"]      = float(sl["dist_lt8"].max()) if len(sl)>0 else 0.0
        out[f"top{K}_crossing_any"]   = float(sl["crossing_like"].max()) if "crossing_like" in sl else 0.0
        out[f"top{K}_approach_any"]   = float(sl["approach_like"].max()) if "approach_like" in sl else 0.0

    # snapshot: best actor by p_overlap
    if len(df_by_p) > 0:
        a1 = df_by_p.iloc[0]
        for c in SNAPSHOT_COLS:
            out[f"a1_{c}"] = a1[c] if pd.notna(a1[c]) else (np.nan)
    else:
        for c in SNAPSHOT_COLS:
            out[f"a1_{c}"] = np.nan

    # stability margins (window-level): from S0 we already have per-actor ranks; keep best margins
    out["best_margin_to_top_p"]    = float(df_by_p["margin_to_top_p"].iloc[0])    if "margin_to_top_p" in df_by_p.columns and len(df_by_p)>0 else 0.0
    out["best_margin_to_top_invd"] = float(df_by_inv["margin_to_top_inv_d"].iloc[0]) if "margin_to_top_inv_d" in df_by_inv.columns and len(df_by_inv)>0 else 0.0

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True, help="Per-actor features parquet (42 cols)")
    ap.add_argument("--out-parquet", required=True, help="Output window-level parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    # basic checks
    for k in KEYS:
        if k not in df.columns:
            raise ValueError(f"Missing key column: {k}")

    # group by window
    grp = df.groupby(KEYS, sort=False)
    rows = []
    for keys, df_win in grp:
        base = dict(zip(KEYS, keys))
        base.update(agg_block(df_win))
        rows.append(base)

    out = pd.DataFrame(rows)

    # deterministic column order: keys first
    key_cols = KEYS
    other_cols = [c for c in out.columns if c not in key_cols]
    out = out[key_cols + sorted(other_cols)]

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"Wrote window table: {args.out_parquet}")
    print(f"Rows: {len(out)}, Cols: {out.shape[1]}")

if __name__ == "__main__":
    main()
