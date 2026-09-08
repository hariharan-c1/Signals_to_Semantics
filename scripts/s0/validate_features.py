#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

NUM_COLS = {
    # geometry/kinematics
    "x_rel_m","y_rel_m","r_rel_m","dmin_m","p_overlap","t_at_dmin_s","sustained_tight",
    "rel_speed_closing_mps","lat_speed_mps","a_norm","length_m","width_m","bearing_rad",
    "ttc_s","dist_lt3","dist_lt5","dist_lt8","approach_like","crossing_like",
    "heading_align_cos","long_gap_m","lat_offset_m",
    # map cues
    "map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m","in_drivable_area",
    # context
    "rank_p_overlap","margin_to_top_p","rank_inv_dmin","margin_to_top_inv_d",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n_rows, n_cols = df.shape

    # NaN rates
    nan_rates = df.isna().mean().sort_values(ascending=False)
    # Basic ranges for numeric cols we care about
    cols = [c for c in df.columns if c in NUM_COLS]
    desc = df[cols].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).T

    # Sanity checks
    problems = []
    if (df["in_drivable_area"].dropna().astype(float) % 1.0 != 0).any():
        problems.append("in_drivable_area not strictly 0/1 for some rows.")
    if (df["map_lane_alignment_cos"].dropna().abs() > 1.0001).any():
        problems.append("map_lane_alignment_cos outside [-1,1] bounds.")

    # Print
    size_mb = Path(args.parquet).stat().st_size / (1024*1024)
    print(f"QC for: {args.parquet}")
    print(f"Rows, Cols: {n_rows}, {n_cols} | Size: {size_mb:.2f} MB")
    print("\nTop NaN columns:")
    print(nan_rates.head(10).to_string())
    print("\nKey numeric ranges (1%, 5%, 50%, 95%, 99%):")
    keep = desc[["min","1%","5%","50%","95%","99%","max"]]
    print(keep.to_string(float_format=lambda x: f"{x:.4f}"))
    if problems:
        print("\nWARNINGS:")
        for p in problems:
            print(" -", p)

if __name__ == "__main__":
    main()
