# scripts/s1a_calibrate_tau_cv.py
import argparse, json, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SCORE_CANDIDATES = ["score_pu_xgb", "score", "pu_score", "prob", "yhat"]

def find_scores_path(dev_root: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--scores-json not found: {p}")
        return p
    # default: search under dev_root/** for scores*.jsonl
    matches = list(dev_root.rglob("scores*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No scores*.jsonl found under {dev_root}. "
                                f"Pass --scores-json explicitly.")
    # prefer pu_xgb scores if multiple
    matches = sorted(matches, key=lambda p: ("pu_xgb" not in str(p), str(p)))
    return matches[0]

def load_labels(dev_root: Path) -> pd.DataFrame:
    labels_fp = dev_root / "window_labels.jsonl"
    if not labels_fp.exists():
        raise FileNotFoundError(f"Missing labels: {labels_fp}")
    df = pd.read_json(labels_fp, lines=True)
    # expected keys: log_id, t_start, t_end, label (1 for positive, else None)
    # Normalize types
    for c in ["log_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["t_start", "t_end"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df

def load_scores(scores_fp: Path, score_col_arg: str | None) -> pd.DataFrame:
    df = pd.read_json(scores_fp, lines=True)
    # normalize keys
    rename = {}
    if "t_start" not in df.columns and "window_t_start" in df.columns:
        rename["window_t_start"] = "t_start"
    if "t_end" not in df.columns and "window_t_end" in df.columns:
        rename["window_t_end"] = "t_end"
    if rename:
        df = df.rename(columns=rename)

    # choose score column
    if score_col_arg:
        score_col = score_col_arg
        if score_col not in df.columns:
            raise KeyError(f"Requested --score-col '{score_col}' not found in {scores_fp}. "
                           f"Available: {list(df.columns)}")
    else:
        score_col = None
        for c in SCORE_CANDIDATES:
            if c in df.columns:
                score_col = c
                break
        if score_col is None:
            raise KeyError(f"No score column found. Looked for {SCORE_CANDIDATES}. "
                           f"Columns: {list(df.columns)}")

    keep = ["log_id", "t_start", "t_end", score_col]
    missing = [k for k in keep if k not in df.columns]
    if missing:
        raise KeyError(f"Scores file {scores_fp} missing columns {missing}")

    df = df[keep].rename(columns={score_col: "score"})
    # dtypes
    df["log_id"] = df["log_id"].astype(str)
    df["t_start"] = df["t_start"].astype(float)
    df["t_end"] = df["t_end"].astype(float)
    df["score"] = df["score"].astype(float)
    return df

def quantile_tau_for_recall(pos_scores: np.ndarray, target_recall: float) -> float:
    """
    For positives only: recall(τ) = P(score >= τ | y=1).
    Thus τ is the (1 - target_recall) quantile of positive scores.
    """
    q = max(0.0, min(1.0, 1.0 - float(target_recall)))
    if pos_scores.size == 0:
        return np.nan
    # use "nearest" to pick an actual score-ish threshold
    return float(np.quantile(pos_scores, q, method="nearest"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", required=True, help="Root with window_table.parquet + window_labels.jsonl")
    ap.add_argument("--scores-json", default=None, help="Optional explicit path to scores jsonl")
    ap.add_argument("--score-col", default=None, help=f"Score column name (default auto: {SCORE_CANDIDATES})")
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    dev_root = Path(args.dev_root)
    scores_fp = find_scores_path(dev_root, args.scores_json)
    labels_df = load_labels(dev_root)
    scores_df = load_scores(scores_fp, args.score_col)

    # Merge scores with labels on (log_id, t_start, t_end)
    df = scores_df.merge(labels_df, on=["log_id", "t_start", "t_end"], how="left")
    # Keep only rows with label == 1 for calibration (positives only)
    df_pos = df[df["label"] == 1].copy()
    if df_pos.empty:
        raise RuntimeError("No labeled positives in dev set after merge; cannot calibrate τ.")

    # Grouped K-fold by log_id
    gkf = GroupKFold(n_splits=args.n_splits)
    groups = df_pos["log_id"].values
    idx = np.arange(len(df_pos))
    taus = []
    fold_stats = []

    for k, (itr, ival) in enumerate(gkf.split(idx, groups=groups)):
        dval = df_pos.iloc[ival]
        pos_scores = dval["score"].to_numpy(dtype=float)

        tau_k = quantile_tau_for_recall(pos_scores, args.target_recall)
        taus.append(tau_k)

        # sanity: achieved recall on this fold at tau_k
        rec_k = float((pos_scores >= tau_k).mean()) if np.isfinite(tau_k) else 0.0
        fold_stats.append({
            "fold": k,
            "n_pos": int(len(pos_scores)),
            "tau_k": tau_k,
            "recall_at_tau_k": rec_k,
            "pos_score_min": float(np.min(pos_scores)),
            "pos_score_median": float(np.median(pos_scores)),
            "pos_score_max": float(np.max(pos_scores)),
        })

    taus = np.array([t for t in taus if np.isfinite(t)], dtype=float)
    if taus.size == 0:
        raise RuntimeError("All per-fold taus are NaN; cannot calibrate.")

    tau_median = float(np.median(taus))
    # Achieved recall across ALL positives at tau_median (reporting only)
    rec_global = float((df_pos["score"].to_numpy() >= tau_median).mean())

    out_dir = dev_root / "pu_xgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "tau.json"
    meta = {
        "tau": tau_median,
        "taus_per_fold": fold_stats,
        "target_recall": float(args.target_recall),
        "achieved_recall_allpositives": rec_global,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "scores_json": str(scores_fp),
        "score_col_used": "score",
        "n_labeled_pos": int(len(df_pos)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_fp, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved τ to {out_fp}")
    print(json.dumps({"tau": tau_median, "achieved_recall_allpositives": rec_global}, indent=2))

if __name__ == "__main__":
    main()
