# scripts/s2_sanity_check_index.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

VITAL_NUMS = [
    "r_rel_m","x_rel_m","y_rel_m","ttc_s","a_norm","rel_speed_closing_mps",
    "map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m",
    "dist_to_crosswalk_m","in_drivable_area","post_hmm","score_final"
]
ID_KEYS = ["log_id","window_t_start","window_t_end"]

def pct(x): return float(np.round(100.0*x, 3))

def load_scores(split_root: Path):
    # Prefer S1D; fallback to S1C HMM scores
    s1d = split_root / "s1d" / "final_scores.parquet"
    s1c = split_root / "s1c" / "hmm" / "scores_hmm.jsonl"
    if s1d.exists():
        df = pd.read_parquet(s1d).rename(columns={"t_start":"window_t_start","t_end":"window_t_end"})
        df["source"] = "s1d"
        return df[ID_KEYS + ["post_hmm","score_final","source"]]
    elif s1c.exists():
        df = pd.read_json(s1c, lines=True).rename(columns={"t_start":"window_t_start","t_end":"window_t_end"})
        if "score_final" not in df.columns and "post_hmm" in df.columns:
            df["score_final"] = df["post_hmm"]
        df["source"] = "s1c"
        return df[ID_KEYS + ["post_hmm","score_final","source"]]
    else:
        raise FileNotFoundError(f"Could not find S1D or S1C scores under: {split_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g. artifacts/.../dev100 or .../train650")
    ap.add_argument("--index-parquet", default=None, help="Path to s2/index/actor_index.parquet. Default: <split-root>/s2/index/actor_index.parquet")
    ap.add_argument("--r-max", type=float, default=60.0, help="Max allowed r_rel_m after pruning")
    ap.add_argument("--min-actors", type=int, default=1, help="Min actors per kept window")
    ap.add_argument("--max-actors", type=int, default=40, help="Warn if > this per window (graph bloat)")
    ap.add_argument("--out-json", default=None, help="Where to write JSON summary (default under s2/index)")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    idx_path = Path(args.index_parquet) if args.index_parquet else (split_root / "s2" / "index" / "actor_index.parquet")
    if not idx_path.exists():
        raise FileNotFoundError(f"actor_index.parquet not found: {idx_path}")

    df = pd.read_parquet(idx_path)
    scores = load_scores(split_root)

    # ----- Basic presence checks -----
    missing_cols = [c for c in ID_KEYS + ["track_uuid","category","r_rel_m","post_hmm","score_final"] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"actor_index missing required columns: {missing_cols}")

    # ----- r_rel_m pruning sanity -----
    n_total = len(df)
    n_over = int((df["r_rel_m"] > args.r_max).sum())
    over_rows = df.loc[df["r_rel_m"] > args.r_max, ID_KEYS + ["track_uuid","r_rel_m"]]
    # ----- actor counts per window -----
    counts = df.groupby(ID_KEYS)["track_uuid"].nunique().rename("actor_count").reset_index()
    n_windows = counts.shape[0]
    n_too_few = int((counts["actor_count"] < args.min_actors).sum())
    n_too_many = int((counts["actor_count"] > args.max_actors).sum())

    # percentiles of actors per window
    q = counts["actor_count"].quantile([0.01,0.05,0.5,0.95,0.99]).to_dict()

    # ----- NaNs & ranges on vital numeric columns -----
    vital_present = [c for c in VITAL_NUMS if c in df.columns]
    nan_rates = {c: pct(df[c].isna().mean()) for c in vital_present}
    ranges = {}
    for c in vital_present:
        s = df[c].dropna()
        if len(s):
            ranges[c] = {
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.quantile(0.50)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }

    # ----- Check post_hmm/score_final bounds -----
    def frac_out_of_01(s):
        s = s.dropna()
        return 0.0 if not len(s) else pct(((s < 0) | (s > 1)).mean())
    frac_post_out = frac_out_of_01(df["post_hmm"])
    frac_score_out = frac_out_of_01(df["score_final"])

    # ----- Check that windows align with scores-kept -----
    # windows seen in index:
    win_idx = counts[ID_KEYS].drop_duplicates()
    # windows available in scores:
    win_scores = scores[ID_KEYS].drop_duplicates()
    # join to ensure proper subset
    merged = pd.merge(win_idx, win_scores, on=ID_KEYS, how="left", indicator=True)
    n_missing_in_scores = int((merged["_merge"] == "left_only").sum())
    missing_windows = merged.loc[merged["_merge"] == "left_only", ID_KEYS]

    # ----- Category mix -----
    cat_counts = df["category"].value_counts().to_dict()

    # ----- Summaries -----
    summary = {
        "paths": {
            "split_root": str(split_root),
            "actor_index": str(idx_path),
        },
        "rows": {
            "actor_index_rows": n_total,
            "unique_windows_in_index": n_windows
        },
        "r_rel_m_pruning": {
            "r_max": args.r_max,
            "n_over_limit": n_over,
            "pct_over_limit": pct(n_over / max(1, n_total)),
        },
        "actor_count_per_window": {
            "min_required": args.min_actors,
            "max_warn": args.max_actors,
            "n_too_few": n_too_few,
            "n_too_many": n_too_many,
            "quantiles": q
        },
        "nan_rates_pct": nan_rates,
        "ranges": ranges,
        "bounds_checks_pct_out_of_[0,1]": {
            "post_hmm": frac_post_out,
            "score_final": frac_score_out
        },
        "alignment_with_scores": {
            "windows_missing_in_scores": n_missing_in_scores
        },
        "category_mix_counts": cat_counts
    }

    # Write offenders (if any)
    out_dir = idx_path.parent
    bad_over_csv = out_dir / "offenders_r_over_limit.csv"
    bad_counts_csv = out_dir / "offenders_actorcount.csv"
    bad_missing_csv = out_dir / "offenders_windows_missing_scores.csv"

    if n_over > 0:
        over_rows.to_csv(bad_over_csv, index=False)
        summary["offenders_r_over_limit_csv"] = str(bad_over_csv)

    if n_too_few > 0 or n_too_many > 0:
        c_bad = counts[(counts["actor_count"] < args.min_actors) | (counts["actor_count"] > args.max_actors)]
        c_bad.to_csv(bad_counts_csv, index=False)
        summary["offenders_actorcount_csv"] = str(bad_counts_csv)

    if n_missing_in_scores > 0:
        missing_windows.to_csv(bad_missing_csv, index=False)
        summary["offenders_windows_missing_scores_csv"] = str(bad_missing_csv)

    out_json = Path(args.out_json) if args.out_json else (out_dir / "sanity_index_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print concise terminal report ----
    print(f"[S2-SANITY] rows={n_total} | windows={n_windows}")
    print(f"  r_rel_m > {args.r_max}: {n_over} rows ({pct(n_over/max(1,n_total))}%)")
    print(f"  actors/window (q01={q.get(0.01):.1f}, q05={q.get(0.05):.1f}, med={q.get(0.5):.1f}, q95={q.get(0.95):.1f}, q99={q.get(0.99):.1f})")
    print(f"  actor-count too few(<{args.min_actors}): {n_too_few} windows | too many(>{args.max_actors}): {n_too_many} windows")
    print(f"  post_hmm outside [0,1]: {frac_post_out}% | score_final outside [0,1]: {frac_score_out}%")
    print(f"  windows missing in scores: {n_missing_in_scores}")
    print(f"  category mix (top 5): {dict(list(cat_counts.items())[:5])}")
    print(f"[S2-SANITY] summary → {out_json}")
    if n_over: print(f"  offenders r>Rmax → {bad_over_csv}")
    if (n_too_few or n_too_many): print(f"  offenders actor-count → {bad_counts_csv}")
    if n_missing_in_scores: print(f"  offenders windows missing scores → {bad_missing_csv}")

if __name__ == "__main__":
    main()
