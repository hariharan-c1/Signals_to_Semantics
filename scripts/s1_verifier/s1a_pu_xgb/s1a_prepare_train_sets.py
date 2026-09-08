# scripts/s1a_prepare_train_sets.py
import argparse, json
from pathlib import Path
import pandas as pd

def read_jsonl(fp):
    rows = []
    with open(fp, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True,
                    help="e.g., artifacts/.../train650/s1a")
    ap.add_argument("--window-table", default="window_table.parquet",
                    help="Path (relative to split-root) to window-level features parquet")
    ap.add_argument("--gt-labels", default="window_labels.jsonl",
                    help="JSONL with ground-truth positives (label==1)")
    ap.add_argument("--pseudo-labels", default="pu_xgb/pseudo_labels.jsonl",
                    help="JSONL with pseudo positives (label==1, weight, score)")
    ap.add_argument("--out-parquet", default="s1a_train_targets.parquet",
                    help="Output parquet with labels/weights joined onto window table")
    ap.add_argument("--out-dir", default="cons",
                    help="Directory (relative to split-root) for outputs (summary, ids)")
    args = ap.parse_args()

    root = Path(args.split_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load window-level features (1 row per window) ---
    dfW = pd.read_parquet(root / args.window_table)

    # Normalize window key columns
    key_rename = {"window_t_start": "t_start", "window_t_end": "t_end"}
    dfW = dfW.rename(columns={k: v for k, v in key_rename.items() if k in dfW.columns})

    # Verify required keys
    for k in ["log_id", "t_start", "t_end"]:
        if k not in dfW.columns:
            raise ValueError(f"Window table missing required column: {k}")

    # --- Load GT positives (label==1) ---
    gt_fp = root / args.gt_labels
    if gt_fp.exists():
        dfGT = pd.DataFrame(read_jsonl(gt_fp))
        if len(dfGT):
            # We only need positives (label==1)
            dfGT = dfGT.loc[dfGT["label"] == 1, ["log_id", "t_start", "t_end", "label"]]
            dfGT["weight"] = 1.0
            dfGT["source"] = "gt"
        else:
            dfGT = pd.DataFrame(columns=["log_id","t_start","t_end","label","weight","source"])
    else:
        dfGT = pd.DataFrame(columns=["log_id","t_start","t_end","label","weight","source"])

    # --- Load pseudo positives (hi/lo) ---
    pseudo_fp = root / args.pseudo_labels
    if pseudo_fp.exists():
        dfP = pd.DataFrame(read_jsonl(pseudo_fp))
        if len(dfP):
            # keep only positive rows
            keep_cols = ["log_id","t_start","t_end","label","weight","conf_level","score","source"]
            missing = [c for c in keep_cols if c not in dfP.columns]
            if missing:
                raise ValueError(f"Pseudo file missing columns: {missing}")
            dfP = dfP.loc[dfP["label"] == 1, keep_cols]
        else:
            dfP = pd.DataFrame(columns=["log_id","t_start","t_end","label","weight","conf_level","score","source"])
    else:
        dfP = pd.DataFrame(columns=["log_id","t_start","t_end","label","weight","conf_level","score","source"])

    # --- Merge all positives (gt + pseudo) ---
    dfPos = pd.concat([dfGT, dfP], ignore_index=True)
    dfPos = dfPos.drop_duplicates(subset=["log_id","t_start","t_end"])

    # --- Join with window keys to align labels/weights ---
    key = ["log_id","t_start","t_end"]
    dfW_key = dfW[key].copy()
    dfW_pos = dfW_key.merge(dfPos, on=key, how="left")

    # Unlabeled = NaN -> None
    dfW_pos["label"] = dfW_pos["label"].where(dfW_pos["label"].notna(), None)
    # weights: gt/pseudo carry weight; unlabeled = 0.0 (supervised loss); nnPU uses unlabeled separately
    dfW_pos["weight"] = dfW_pos["weight"].fillna(0.0)
    dfW_pos["source"] = dfW_pos["source"].fillna("unlabeled")

    # --- Merge back to full window-level features ---
    df_all = dfW.merge(dfW_pos[["log_id","t_start","t_end","label","weight","source"]],
                       on=key, how="left")
    df_all["label"] = df_all["label"].where(df_all["label"].notna(), None)
    df_all["weight"] = df_all["weight"].fillna(0.0)
    df_all["source"] = df_all["source"].fillna("unlabeled")

    # --- Write outputs ---
    out_parquet = out_dir / args.out_parquet
    df_all.to_parquet(out_parquet, index=False)

    # Summary
    n_all = len(df_all)
    n_pos = int((df_all["label"] == 1).sum())
    n_unl = int(df_all["label"].isna().sum())
    n_gt = int((df_all["source"] == "gt").sum())
    n_pseudo = int((df_all["source"] == "pu_xgb").sum())
    summary = {
        "total_windows": n_all,
        "positives_total": n_pos,
        "positives_gt": n_gt,
        "positives_pseudo": n_pseudo,
        "unlabeled": n_unl,
        "out_parquet": str(out_parquet),
        "window_table": str(root / args.window_table),
        "gt_labels": str(gt_fp),
        "pseudo_labels": str(pseudo_fp),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # IDs kept (for audit)
    with open(out_dir / "ids_kept.jsonl", "w") as f:
        for _, r in df_all[key].drop_duplicates().iterrows():
            f.write(json.dumps({"log_id": r["log_id"], "t_start": float(r["t_start"]), "t_end": float(r["t_end"])}) + "\n")

    print(f"Wrote: {out_parquet}")
    print(f"Summary: {out_dir/'summary.json'}")
    print(f"IDs: {out_dir/'ids_kept.jsonl'}")
    print(f"Total windows: {n_all} | Positives: {n_pos} (gt={n_gt}, pseudo={n_pseudo}) | Unlabeled: {n_unl}")

if __name__ == "__main__":
    main()
