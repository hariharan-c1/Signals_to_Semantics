# scripts/s2_nn_recall.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def cosine_mat(X):
    X = X.astype(np.float32)
    # embeddings are already L2-normalized; guard anyway
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / n
    return Xn @ Xn.T

def eval_file(path):
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if "window_key" not in df.columns:
        df["window_key"] = (
            df["log_id"].astype(str) + "|" +
            df["window_t_start"].astype(str) + "|" +
            df["window_t_end"].astype(str)
        )
    X = df[emb_cols].to_numpy()
    S = cosine_mat(X)
    np.fill_diagonal(S, -1.0)  # exclude self
    nn_idx = S.argmax(axis=1)
    same = (df["window_key"].to_numpy() == df["window_key"].to_numpy()[nn_idx])
    top1_same = float(same.mean()) * 100.0

    # per-window breakdown (median & 90th pct)
    by_win = []
    for k, g in df.groupby("window_key"):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        sub = (df.loc[idx, "window_key"].to_numpy() == df.loc[nn_idx[idx], "window_key"].to_numpy())
        by_win.append(sub.mean())
    med_win = float(np.median(by_win)*100) if by_win else 0.0
    p90_win = float(np.quantile(by_win, 0.9)*100) if by_win else 0.0

    print(f"[S2-NN] {path}")
    print(f"  rows={len(df)} windows={df['window_key'].nunique()}")
    print(f"  Top-1 same-window recall = {top1_same:.1f}% (median per-window={med_win:.1f}%, p90={p90_win:.1f}%)")
    return top1_same

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g. artifacts/.../dev100 or .../train650")
    args = ap.parse_args()
    root = Path(args.split_root) / "s2" / "contrastive"
    eval_file(root / "embeddings_train.parquet")
    eval_file(root / "embeddings_val.parquet")

if __name__ == "__main__":
    main()
