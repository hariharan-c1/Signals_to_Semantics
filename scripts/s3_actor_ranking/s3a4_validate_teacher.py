# scripts/s3a4_sanity_teacher.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--teacher-parquet", default=None)
    ap.add_argument("--nodes-parquet", default=None)
    args = ap.parse_args()

    root = Path(args.split_root)
    qdir = root / "s3" / "quasi"
    teach_p = Path(args.teacher_parquet) if args.teacher_parquet else (qdir / "teacher.parquet")
    nodes_p = Path(args.nodes_parquet)   if args.nodes_parquet   else (qdir / "nodes.parquet")

    if not teach_p.exists(): raise FileNotFoundError(teach_p)
    if not nodes_p.exists(): raise FileNotFoundError(nodes_p)

    t = pd.read_parquet(teach_p)
    n = pd.read_parquet(nodes_p)

    non_ego = n[n["track_uuid"] != "EGO"][["window_key","track_uuid"]].drop_duplicates()
    t_idx = t[["window_key","track_uuid"]].drop_duplicates()
    miss = pd.merge(non_ego, t_idx, on=["window_key","track_uuid"], how="left", indicator=True)
    n_missing = int((miss["_merge"] == "left_only").sum())

    sums = t.groupby("window_key")["q"].sum().reset_index()
    dev = float(np.abs(sums["q"] - 1.0).abs().mean())

    top1 = t.sort_values(["window_key","q"], ascending=[True, False]).groupby("window_key").head(1)
    top1_cat = top1["category"].value_counts().to_dict()

    summary = {
        "windows": int(t["window_key"].nunique()),
        "rows": int(len(t)),
        "missing_non_ego_in_teacher": n_missing,
        "mean_abs_dev_from_1": dev,
        "top1_category_mix": top1_cat
    }
    out = qdir / "sanity_teacher.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"[S3-A4-SANITY] windows={summary['windows']} | rows={summary['rows']} | "
          f"missing_non_ego={summary['missing_non_ego_in_teacher']} | mean|Σq-1|={dev:.4e} | "
          f"top1 mix: {top1_cat}")
    print(f"[S3-A4-SANITY] summary → {out}")

if __name__ == "__main__":
    main()
