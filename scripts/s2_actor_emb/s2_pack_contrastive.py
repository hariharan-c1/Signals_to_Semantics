# scripts/s2_pack_contrastive.py
import argparse, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Columns we keep for embeddings (drop IDs and non-feature fields)
ID_COLS = ["log_id","window_t_start","window_t_end","track_uuid","category"]
DROP_COLS = set(ID_COLS + ["window_center"])  # center is redundant here
# Vital numeric feature whitelist (falls back to "all numeric not in DROP_COLS")
VITAL_NUMS = [
    "x_rel_m","y_rel_m","r_rel_m","dmin_m","p_overlap","t_at_dmin_s","sustained_tight",
    "rel_speed_closing_mps","lat_speed_mps","a_norm","length_m","width_m","bearing_rad",
    "ttc_s","dist_lt3","dist_lt5","dist_lt8","approach_like","crossing_like",
    "heading_align_cos","long_gap_m","lat_offset_m","map_lane_offset_m",
    "map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m",
    "in_drivable_area","rank_p_overlap","margin_to_top_p","rank_inv_dmin","margin_to_top_inv_d",
    "post_hmm","score_final"
]

def select_feature_columns(df: pd.DataFrame):
    # keep only numeric (float/int) and in whitelist; if some are missing, use available numeric columns minus drops
    existing = [c for c in VITAL_NUMS if c in df.columns]
    if len(existing) >= 24:  # most made it
        return existing
    # fallback: auto-detect numeric
    auto = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in DROP_COLS]
    return sorted(auto)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g. artifacts/.../dev100")
    ap.add_argument("--index-parquet", default=None,
                    help="defaults to <split-root>/s2/index/actor_index.parquet")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <split-root>/s2/contrastive")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    split_root = Path(args.split_root)
    idx_path = Path(args.index_parquet) if args.index_parquet else (split_root / "s2" / "index" / "actor_index.parquet")
    out_dir = Path(args.out_dir) if args.out_dir else (split_root / "s2" / "contrastive")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(idx_path)
    feat_cols = select_feature_columns(df)

    # Standardize per-split with robust stats
    med = df[feat_cols].median()
    mad = (df[feat_cols] - med).abs().median().replace(0, 1.0)
    dfz = (df[feat_cols] - med) / mad

    # Build keys for grouping negatives (same window are "hard" negatives)
    df_out = pd.DataFrame({
        "log_id": df["log_id"],
        "window_t_start": df["window_t_start"],
        "window_t_end": df["window_t_end"],
        "track_uuid": df["track_uuid"],
        "category": df["category"]
    })
    for c in feat_cols:
        df_out[c] = dfz[c].astype(np.float32)

    # Make a window key
    df_out["window_key"] = (
        df_out["log_id"].astype(str) + "|" +
        df_out["window_t_start"].astype(str) + "|" +
        df_out["window_t_end"].astype(str)
    )

    # Split train/val by window (avoid leakage)
    win_keys = df_out["window_key"].unique()
    win_train, win_val = train_test_split(win_keys, test_size=args.test_size, random_state=args.seed, shuffle=True)
    is_train = df_out["window_key"].isin(win_train)

    train_tbl = df_out[is_train].reset_index(drop=True)
    val_tbl   = df_out[~is_train].reset_index(drop=True)

    # Save
    train_pq = out_dir / "contrastive_train.parquet"
    val_pq   = out_dir / "contrastive_val.parquet"
    meta_js  = out_dir / "meta.json"
    train_tbl.to_parquet(train_pq, index=False)
    val_tbl.to_parquet(val_pq, index=False)

    meta = {
        "split_root": str(split_root),
        "index_parquet": str(idx_path),
        "feat_cols": feat_cols,
        "n_rows": int(len(df_out)),
        "n_windows": int(df_out["window_key"].nunique()),
        "train_rows": int(len(train_tbl)),
        "val_rows": int(len(val_tbl)),
        "train_windows": int(train_tbl["window_key"].nunique()),
        "val_windows": int(val_tbl["window_key"].nunique()),
        "robust_center": med.to_dict(),
        "robust_scale_mad": mad.to_dict(),
        "seed": args.seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(meta_js, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[S2-Pack] wrote:\n  {train_pq} (rows={len(train_tbl)}, windows={train_tbl['window_key'].nunique()})\n  {val_pq}   (rows={len(val_tbl)}, windows={val_tbl['window_key'].nunique()})\n  {meta_js}")

if __name__ == "__main__":
    main()
