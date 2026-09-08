#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

def load_meta(train_root: Path):
    meta_fp = train_root / "meta.json"
    if meta_fp.exists():
        with open(meta_fp) as f:
            meta = json.load(f)
    else:
        meta = {}
    return meta

def ensure_imputer(train_root: Path, train_window_table_fp: Path, feature_cols):
    """Ensure we have train medians for imputation. Save into meta.json if missing."""
    meta = load_meta(train_root)
    imputer_key = "imputer_medians"
    if imputer_key in meta and set(meta[imputer_key].keys()) == set(feature_cols):
        return meta[imputer_key]

    # compute on train650 window_table
    df_train = pd.read_parquet(train_window_table_fp)
    med = df_train[feature_cols].median(numeric_only=True).to_dict()
    # persist
    meta.setdefault("feature_cols", feature_cols)
    meta[imputer_key] = med
    with open(train_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return med

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True,
                    help="e.g., artifacts/.../dev100/s1a or artifacts/.../val50/s1a")
    ap.add_argument(
        "--train-root",
        required=True,
        help="Directory containing model_xgb.json and meta.json from Train650 training.",
    )
    ap.add_argument(
        "--train-window-table",
        default=None,
        help=(
            "Train650 window table used to fit missing-value medians. "
            "Default: <train-root>/../window_table.parquet"
        ),
    )
    ap.add_argument("--window-table", default="window_table.parquet")
    ap.add_argument("--out-jsonl", default="scores_pu_xgb.jsonl")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    train_root = Path(args.train_root)

    model = xgb.Booster()
    model.load_model(str(train_root / "model_xgb.json"))

    # meta may or may not contain feature_cols & imputers; handle both cases robustly
    meta = load_meta(train_root)
    feature_cols = meta.get("feature_cols", None)
    win_fp = split_root / args.window_table
    df = pd.read_parquet(win_fp)

    # If feature list is missing (older meta), infer from columns that start with a[1-3]_ and aggregates
    if feature_cols is None:
        # heuristic: drop id/time cols, keep numeric
        drop = {"log_id","window_t_start","window_t_end","window_center"}
        cand = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        feature_cols = sorted(cand)
        meta["feature_cols"] = feature_cols
        with open(train_root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ensure imputers
    train_win_fp = (
        Path(args.train_window_table)
        if args.train_window_table
        else train_root.parent / args.window_table
    )
    if not train_win_fp.is_file():
        raise FileNotFoundError(
            f"Training window table not found: {train_win_fp}. "
            "Pass --train-window-table explicitly."
        )
    imputer_medians = ensure_imputer(train_root, train_win_fp, feature_cols)

    X = df[feature_cols].copy()
    for c, m in imputer_medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    # safety for any leftover NaNs
    X = X.fillna(0.0)

    dmat = xgb.DMatrix(X)
    scores = model.predict(dmat)

    out_fp = split_root / "pu_xgb" / args.out_jsonl
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as f:
        for i, row in df.iterrows():
            rec = {
                "log_id": row["log_id"],
                "t_start": float(row["window_t_start"]),
                "t_end": float(row["window_t_end"]),
                "score_pu_xgb": float(scores[i]),
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote scores: {out_fp} | N={len(scores)}")

if __name__ == "__main__":
    main()
