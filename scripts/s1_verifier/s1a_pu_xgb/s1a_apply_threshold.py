#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

def load_scores(scores_path: Path) -> pd.DataFrame:
    # scores JSONL has either 'score' or 'score_pu_xgb'
    df = pd.read_json(scores_path, lines=True)
    if "score" in df.columns:
        df["score"] = df["score"].astype(float)
    elif "score_pu_xgb" in df.columns:
        df = df.rename(columns={"score_pu_xgb": "score"})
    else:
        raise KeyError("Scores file must contain 'score' or 'score_pu_xgb' column.")
    # normalize column names
    rename = {"t_start":"window_t_start","t_end":"window_t_end"}
    for k,v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    keep = ["log_id","window_t_start","window_t_end","score"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Scores file missing columns: {missing}")
    # Dedup by (log_id, window)
    df = df.sort_values("score", ascending=False)
    df = df.drop_duplicates(subset=["log_id","window_t_start","window_t_end"], keep="first")
    return df.reset_index(drop=True)

def load_cv_taus(tau_json: Path):
    with open(tau_json, "r") as f:
        J = json.load(f)
    # Try to read taus from fold list if present; else fall back to top-level tau
    taus = []
    if "taus_per_fold" in J and isinstance(J["taus_per_fold"], list) and J["taus_per_fold"]:
        for item in J["taus_per_fold"]:
            if "tau_k" in item:
                taus.append(float(item["tau_k"]))
    if not taus and "tau" in J:
        taus = [float(J["tau"])]
    if not taus:
        raise ValueError(f"No taus found in {tau_json}")
    taus = np.array(sorted(taus))
    tau_lo = np.quantile(taus, 0.25) if len(taus) > 1 else taus[0]
    tau_hi = np.quantile(taus, 0.50) if len(taus) > 1 else taus[0]
    return float(tau_lo), float(tau_hi), taus.tolist()

def maybe_load_exclude_logs(p: Path):
    if p is None:
        return set()
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"--exclude-logs file not found: {p}")
    logs = []
    if p.suffix.lower() in [".json", ".jsonl"]:
        # accept a json list or jsonl with {log_id: ...}
        if p.suffix.lower() == ".json":
            data = json.load(open(p))
            if isinstance(data, list):
                logs = [str(x) for x in data]
            else:
                raise ValueError("JSON exclude file must be a list of log_ids.")
        else:
            for line in open(p):
                try:
                    obj = json.loads(line)
                    if "log_id" in obj: logs.append(str(obj["log_id"]))
                except Exception:
                    continue
    else:
        # plain text, one log_id per line
        logs = [ln.strip() for ln in open(p) if ln.strip()]
    return set(logs)

def main():
    ap = argparse.ArgumentParser(description="Apply PU-XGB thresholds to create pseudo labels.")
    ap.add_argument("--split-root", required=True,
                    help="Root folder for this split, e.g. artifacts/.../train650/s1a")
    ap.add_argument("--scores", default="pu_xgb/scores_pu_xgb.jsonl",
                    help="Path to scores jsonl (relative to split-root or absolute).")
    ap.add_argument("--tau-json", default="tau_cv.json",
                    help="CV calibration JSON (relative to split-root or absolute).")
    ap.add_argument("--mode", choices=["cv","single"], default="cv",
                    help="Use CV-derived (q25, median) or a single tau.")
    ap.add_argument("--tau-single", type=float, default=None,
                    help="If mode=single, this tau is used for both lo & hi.")
    ap.add_argument("--w-hi", type=float, default=1.0, help="Weight for hi-confidence pseudo positives.")
    ap.add_argument("--w-lo", type=float, default=0.6, help="Weight for lo-confidence pseudo positives.")
    ap.add_argument("--exclude-logs", default=None,
                    help="Optional path to a list/json/jsonl of log_ids to exclude when creating pseudo labels.")
    ap.add_argument("--out-jsonl", default="pu_xgb/pseudo_labels.jsonl",
                    help="Output JSONL path (relative to split-root or absolute).")
    ap.add_argument("--out-summary", default="pu_xgb/pseudo_labels_summary.csv",
                    help="Output CSV summary (relative to split-root or absolute).")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    scores_path = Path(args.scores)
    tau_path = Path(args.tau_json)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)

    if not scores_path.is_absolute():
        scores_path = split_root / scores_path
    if not tau_path.is_absolute():
        tau_path = split_root / tau_path
    if not out_jsonl.is_absolute():
        out_jsonl = split_root / out_jsonl
    if not out_summary.is_absolute():
        out_summary = split_root / out_summary

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    df_scores = load_scores(scores_path)

    # Exclude logs if provided
    excl = maybe_load_exclude_logs(Path(args.exclude_logs)) if args.exclude_logs else set()
    if excl:
        before = len(df_scores)
        df_scores = df_scores[~df_scores["log_id"].isin(excl)].reset_index(drop=True)
        print(f"Excluded {before - len(df_scores)} windows from {len(excl)} logs (exclude set).")

    if args.mode == "cv":
        tau_lo, tau_hi, taus_all = load_cv_taus(tau_path)
        mode_meta = {"mode":"cv", "tau_lo":tau_lo, "tau_hi":tau_hi, "taus_all":taus_all}
    else:
        if args.tau_single is None:
            raise ValueError("--tau-single must be provided when --mode=single")
        tau_lo = float(args.tau_single)
        tau_hi = float(args.tau_single)
        mode_meta = {"mode":"single", "tau":tau_lo}

    # Assign tiers
    def tier_for(s):
        if s >= tau_hi:
            return "hi"
        elif s >= tau_lo:
            return "lo"
        else:
            return "none"

    df_scores["conf_level"] = df_scores["score"].apply(tier_for)
    df_keep = df_scores[df_scores["conf_level"].isin(["hi","lo"])].copy()
    df_keep["label"] = 1
    df_keep["weight"] = df_keep["conf_level"].map({"hi": args.w_hi, "lo": args.w_lo})

    # Summary
    n_total = len(df_scores)
    n_hi = int((df_scores["conf_level"]=="hi").sum())
    n_lo = int((df_scores["conf_level"]=="lo").sum())
    n_keep = len(df_keep)
    print(f"Thresholds: tau_lo={tau_lo:.6f}, tau_hi={tau_hi:.6f}  [{mode_meta['mode']}]")
    print(f"Windows total: {n_total} | kept: {n_keep}  (hi={n_hi}, lo={n_lo})")
    if n_keep > 0:
        print("Score stats kept:",
              f"min={df_keep['score'].min():.3f} med={df_keep['score'].median():.3f} max={df_keep['score'].max():.3f}")

    # Write JSONL for pseudo labels
    with open(out_jsonl, "w") as f:
        for r in df_keep.itertuples(index=False):
            obj = {
                "log_id": r.log_id,
                "t_start": float(r.window_t_start),
                "t_end": float(r.window_t_end),
                "score": float(r.score),
                "label": 1,
                "conf_level": r.conf_level,
                "weight": float(r.weight),
                "source": "pu_xgb",
            }
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote pseudo-labels: {out_jsonl}")

    # Write summary CSV (by conf_level)
    summ = (df_scores
            .groupby("conf_level")
            .agg(n_windows=("score","size"),
                 score_min=("score","min"),
                 score_median=("score","median"),
                 score_max=("score","max"))
            .reset_index()
           )
    summ.to_csv(out_summary, index=False)
    print(f"Wrote summary: {out_summary}")

    # Also drop a small meta json
    meta = {
        "thresholds": mode_meta,
        "w_hi": args.w_hi,
        "w_lo": args.w_lo,
        "n_total": n_total,
        "n_hi": n_hi,
        "n_lo": n_lo,
        "n_keep": n_keep,
        "scores_path": str(scores_path),
        "exclude_logs": sorted(list(excl)) if excl else [],
    }
    with open(out_jsonl.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta: {out_jsonl.with_suffix('.meta.json')}")

if __name__ == "__main__":
    main()
