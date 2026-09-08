#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

KEYS = ["log_id","window_t_start","window_t_end","window_center"]

def load_labels(labels_jsonl: str) -> pd.DataFrame:
    """
    Expect a JSONL with at least: log_id, t_start, t_end, label (1 or null/0),
    optionally scenario_label.
    """
    recs = []
    with open(labels_jsonl, "r") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            recs.append({
                "log_id": obj["log_id"],
                "window_t_start": float(obj["t_start"]),
                "window_t_end": float(obj["t_end"]),
                "label": 1 if (obj.get("label", None) in [1, True]) else 0,
                "scenario_label": obj.get("scenario_label", None)
            })
    return pd.DataFrame(recs)

def elkan_noto_c(pos_probas: np.ndarray) -> float:
    """ c = P(s=1 | y=1) ≈ mean(ŷ | positives) """
    c = float(np.clip(pos_probas.mean(), 1e-6, 0.999999))
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-parquet", required=True, help="Window-level parquet from s1a_build_window_table.py")
    ap.add_argument("--labels-jsonl", required=True, help="Window labels jsonl (label=1 or 0/None)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--split-name", default="train650", help="Tag for bookkeeping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    win_df = pd.read_parquet(args.window_parquet)
    lbl_df = load_labels(args.labels_jsonl)

    # join on window keys
    merged = win_df.merge(lbl_df, on=["log_id","window_t_start","window_t_end"], how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)

    # P: labeled positives, U: unlabeled (0)
    # NB: we DO NOT treat explicit negatives differently; all non-positives are unlabeled per PU.
    P = merged.loc[merged["label"]==1].copy()
    U = merged.loc[merged["label"]==0].copy()

    # pick features: drop keys and any string columns
    drop_cols = set(KEYS + ["label","scenario_label"])
    feat_cols = [c for c in merged.columns if c not in drop_cols and (np.issubdtype(merged[c].dtype, np.number))]
    if len(feat_cols)==0:
        raise ValueError("No numeric features found after filtering.")

    X = merged[feat_cols].copy()
    y = merged["label"].astype(int).values
    groups = merged["log_id"].values

    # impute NaN with column median (and keep distribution friendly for XGB)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols)

    # split by log_id to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(X_imp, y, groups))
    X_tr, y_tr = X_imp.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_imp.iloc[va_idx], y[va_idx]

    # Model — non-linear, robust defaults
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="logloss",
        random_state=args.seed,
        tree_method="hist",
    )
    clf.fit(X_tr, y_tr)

    # Elkan–Noto c on VALIDATION POSITIVES ONLY
    if (y_va==1).sum() == 0:
        # fallback: compute on train positives (rare)
        yhat_pos = clf.predict_proba(X_tr[y_tr==1])[:,1]
    else:
        yhat_pos = clf.predict_proba(X_va[y_va==1])[:,1]
    c = elkan_noto_c(yhat_pos)

    # score EVERY window with corrected probability
    yhat_all = clf.predict_proba(X_imp)[:,1]
    yhat_cal = np.clip(yhat_all / c, 0.0, 1.0)

    # quick metrics on labeled subset (just sanity, not definitive)
    if (merged["label"]==1).any():
        ap = average_precision_score(merged["label"].values, yhat_cal)
        auc = roc_auc_score(merged["label"].values, yhat_cal)
    else:
        ap = float("nan"); auc = float("nan")

    # save model + meta + scores + the exact features used
    with open(outdir/"meta.json", "w") as f:
        json.dump({
            "split": args.split_name,
            "seed": args.seed,
            "n_rows": int(len(merged)),
            "n_pos": int((merged["label"]==1).sum()),
            "n_unlabeled": int((merged["label"]==0).sum()),
            "feat_cols": feat_cols,
            "imputer": "median",
            "elkan_noto_c": c,
            "ap_on_labeled": ap,
            "auc_on_labeled": auc
        }, f, indent=2)

    clf.save_model(str(outdir/"model_xgb.json"))
    X_imp.to_parquet(outdir/"window_features_imputed.parquet", index=False)

    # write scores jsonl
    with open(outdir/"scores_pu_xgb.jsonl", "w") as f:
        for i in range(len(merged)):
            rec = {
                "log_id": str(merged.iloc[i]["log_id"]),
                "t_start": float(merged.iloc[i]["window_t_start"]),
                "t_end": float(merged.iloc[i]["window_t_end"]),
                "score_pu_xgb": float(yhat_cal[i])
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Saved: {outdir/'model_xgb.json'}")
    print(f"Saved: {outdir/'meta.json'}  (c={c:.6f}, AP(labeled)={ap:.4f}, AUC(labeled)={auc:.4f})")
    print(f"Saved: {outdir/'scores_pu_xgb.jsonl'}")
    print(f"Saved: {outdir/'window_features_imputed.parquet'}")

if __name__ == "__main__":
    main()
