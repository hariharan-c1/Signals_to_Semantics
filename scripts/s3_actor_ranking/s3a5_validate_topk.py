# scripts/s3a5_sanity_topk.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load_topk(p: Path):
    df = pd.read_parquet(p)
    req = {"window_key","rank","track_uuid","score"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"topk parquet missing columns: {missing}")
    df["window_key"] = df["window_key"].astype(str)
    df["track_uuid"] = df["track_uuid"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype(float)
    return df

def load_manifest(graphs_dir: Path):
    man = pd.read_parquet(graphs_dir / "manifest.parquet")
    if "window_key" not in man.columns:
        raise KeyError("manifest.parquet missing 'window_key'")
    man["window_key"] = man["window_key"].astype(str)
    return man[["window_key"]].drop_duplicates()

def load_teacher(p: Path):
    t = pd.read_parquet(p)
    req = {"window_key","track_uuid","q"}
    if not req.issubset(t.columns):
        raise KeyError(f"teacher.parquet missing {req - set(t.columns)}")
    t["window_key"] = t["window_key"].astype(str)
    t["track_uuid"] = t["track_uuid"].astype(str)
    t["q"] = pd.to_numeric(t["q"], errors="coerce").fillna(0.0).astype(float)
    return t

def check_uniqueness(df, k):
    bad = []
    for wk, grp in df.groupby("window_key", sort=False):
        ranks = sorted(grp["rank"].tolist())
        ok_rank = (ranks == list(range(1, min(k, len(ranks)) + 1)))
        uniq = grp["track_uuid"].nunique()
        no_dups = (uniq == len(grp))
        if not ok_rank or not no_dups:
            bad.append({"window_key": wk, "ok_rank": ok_rank, "no_dups": no_dups,
                        "n_rows": int(len(grp)), "n_unique": int(uniq)})
    return pd.DataFrame(bad)

def check_finites(df):
    return df[~np.isfinite(df["score"].values)]

def teacher_metrics(topk, teacher, k):
    # windows intersection already applied by caller
    t_all_q = teacher.groupby("window_key")["q"].sum().rename("q_sum_all").reset_index()
    t_argmax = teacher.sort_values(["window_key","q"], ascending=[True, False]) \
                      .groupby("window_key").first().reset_index()[["window_key","track_uuid"]]
    t_argmax = t_argmax.rename(columns={"track_uuid":"teacher_top1"})

    pred = topk.sort_values(["window_key","rank"]).drop_duplicates(["window_key","track_uuid"])
    joined = pred.merge(teacher, on=["window_key","track_uuid"], how="left")
    joined["q"] = joined["q"].fillna(0.0)
    cov = joined.groupby("window_key")["q"].sum().rename("q_sum_atK").reset_index()

    m = t_all_q.merge(cov, on="window_key", how="left").fillna({"q_sum_atK":0.0})
    m["mass_atK"] = np.where(m["q_sum_all"] > 0, m["q_sum_atK"] / m["q_sum_all"], np.nan)

    pred_top1 = pred[pred["rank"] == 1][["window_key","track_uuid"]].rename(columns={"track_uuid":"pred_top1"})
    m = m.merge(pred_top1, on="window_key", how="left").merge(t_argmax, on="window_key", how="left")
    m["top1_hit"] = (m["pred_top1"] == m["teacher_top1"]).astype(float)

    summary = {
        "windows_with_teacher": int(m["window_key"].nunique()),
        "mass_atK_mean": float(m["mass_atK"].mean(skipna=True)),
        "mass_atK_median": float(m["mass_atK"].median(skipna=True)),
        "top1_hit_rate": float(m["top1_hit"].mean(skipna=True)),
        "missing_teacher_windows": 0
    }
    return m, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-parquet", required=True)
    ap.add_argument("--graphs-dir", required=True, help=".../s3/quasi/graphs/<split>")
    ap.add_argument("--teacher-parquet", default=None, help="Optional: compute teacher-mass@K & top1 hit")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--restrict-to-topk", type=int, default=1,
                    help="If 1, evaluate only on windows present in Top-K (default=1).")
    args = ap.parse_args()

    topk_p = Path(args.topk_parquet)
    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (topk_p.parent / "sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    topk = load_topk(topk_p)
    man = load_manifest(graphs_dir)

    # Window sets
    win_topk = set(topk["window_key"].unique())
    win_man  = set(man["window_key"].unique())

    if args.restrict_to_topk:
        # Restrict manifest (and teacher later) to the windows that actually appear in Top-K
        man_eval = man[man["window_key"].isin(win_topk)].copy()
        missing = []  # by definition none
        extra   = sorted(list(win_topk - win_man))  # if Top-K has windows not in manifest (shouldn't happen)
    else:
        man_eval = man
        missing = sorted(list(win_man - win_topk))
        extra   = sorted(list(win_topk - win_man))

    # Core checks on the Top-K rows we have
    dupe = check_uniqueness(topk, args.k)
    fin_bad = check_finites(topk)

    report = {
        "topk_path": str(topk_p),
        "graphs_dir": str(graphs_dir),
        "windows_in_manifest": int(man_eval["window_key"].nunique()),
        "windows_in_topk": int(len(win_topk)),
        "rows_in_topk": int(len(topk)),
        "k": int(args.k),
        "duplicates_windows": int(len(dupe)),
        "nonfinite_rows": int(len(fin_bad)),
        "missing_windows": len(missing),
        "extra_windows": len(extra),
        "restricted_to_topk": bool(args.restrict_to_topk),
    }

    if len(dupe):
        dupe.to_csv(out_dir / "offenders_duplicates_or_bad_ranks.csv", index=False)
    if len(fin_bad):
        fin_bad.to_csv(out_dir / "offenders_nonfinite_scores.csv", index=False)
    if len(missing):
        pd.DataFrame({"window_key": missing}).to_csv(out_dir / "offenders_missing_windows.csv", index=False)
    if len(extra):
        pd.DataFrame({"window_key": extra}).to_csv(out_dir / "offenders_extra_windows.csv", index=False)

    # Optional teacher metrics
    if args.teacher_parquet and Path(args.teacher_parquet).exists():
        teacher = load_teacher(Path(args.teacher_parquet))
        # restrict teacher to evaluation windows
        teacher = teacher[teacher["window_key"].isin(man_eval["window_key"])].copy()
        topk_eval = topk[topk["window_key"].isin(man_eval["window_key"])].copy()

        metrics_df, metrics_summary = teacher_metrics(topk_eval, teacher, args.k)
        metrics_df.to_parquet(out_dir / "teacher_metrics_by_window.parquet", index=False)
        report.update(metrics_summary)

    (out_dir / "sanity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[SANITY] report → {out_dir/'sanity_report.json'}")

if __name__ == "__main__":
    main()
