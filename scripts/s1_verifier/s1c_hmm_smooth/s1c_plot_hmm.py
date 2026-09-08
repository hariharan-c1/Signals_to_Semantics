# scripts/s1c_plot_hmm.py
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_scores(split_root: str):
    # prefer HMM outputs; fall back to plain fused/ema if needed
    hmm_path = Path(split_root) / "s1c" / "hmm" / "scores_hmm.jsonl"
    if hmm_path.exists():
        df = pd.read_json(hmm_path, lines=True)
        src = str(hmm_path)
    else:
        # older layout (if any)
        raise FileNotFoundError(f"Did not find {hmm_path}. Run s1c_hmm_smooth.py first.")
    return df, src

def segments_from_viterbi(dfl):
    """Return list of (t0, t1) where viterbi==1"""
    v = dfl["viterbi"].astype(int).values
    t = dfl["t_rel"].values
    segs = []
    if len(v) == 0: return segs
    # run-length encode
    start = 0
    for i in range(1, len(v)+1):
        if i == len(v) or v[i] != v[i-1]:
            # segment [start, i)
            val = v[start]
            t0 = t[start]
            t1 = t[i-1] if i-1 >= start else t[start]
            # expand to the next time for nicer spans
            if i < len(t): t1 = t[i]
            if val == 1:
                segs.append((t0, t1))
            start = i
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True,
                    help="e.g., artifacts/train650_val50/train650")
    ap.add_argument("--log-id", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="optional custom plot dir; default <split-root>/s1c/hmm/plots")
    args = ap.parse_args()

    df, src = load_scores(args.split_root)
    # rename columns if needed (legacy compatibility)
    colmap = {
        "t_start": "t_start",
        "t_end": "t_end",
        "score_pu_xgb": "score_pu_xgb",
        "score_nnpu": "score_nnpu",
        "score_fused": "score_fused",
        "score_ema": "score_ema",
        "post_hmm": "post_hmm",
        "viterbi": "viterbi",
    }
    for k in list(colmap):
        if colmap[k] not in df.columns and k in df.columns:
            # keep original if already present
            colmap[k] = k

    # filter by log and sort by time
    dfl = df[df["log_id"] == args.log_id].copy()
    if dfl.empty:
        print(f"[WARN] No rows for log_id={args.log_id} in {src}")
        return

    # sort by start time and build a clean relative time axis
    dfl = dfl.sort_values(by=colmap["t_start"])
    t0 = float(dfl[colmap["t_start"]].iloc[0])
    dfl["t_rel"] = dfl[colmap["t_start"]].astype(float) - t0

    # keep only valid numeric rows for plotting
    def safe_col(name, default=np.nan):
        return dfl[name].astype(float) if name in dfl.columns else pd.Series(default, index=dfl.index, dtype=float)

    s_fused  = safe_col(colmap["score_fused"])
    s_ema    = safe_col(colmap["score_ema"])
    s_hmm    = safe_col(colmap["post_hmm"])
    s_xgb    = safe_col(colmap["score_pu_xgb"])
    s_nnpu   = safe_col(colmap["score_nnpu"])

    # drop rows where *all* signals are NaN (shouldn’t happen after s1c, but be safe)
    mask_keep = ~(s_fused.isna() & s_ema.isna() & s_hmm.isna() & s_xgb.isna() & s_nnpu.isna())
    dfl = dfl.loc[mask_keep]
    s_fused = s_fused.loc[mask_keep]
    s_ema   = s_ema.loc[mask_keep]
    s_hmm   = s_hmm.loc[mask_keep]
    s_xgb   = s_xgb.loc[mask_keep]
    s_nnpu  = s_nnpu.loc[mask_keep]

    # recompute t_rel after mask
    dfl["t_rel"] = dfl[colmap["t_start"]].astype(float) - float(dfl[colmap["t_start"]].iloc[0])

    # --- Debug print so you can sanity check quickly in terminal ---
    def minmax(s):
        if s.dropna().empty: return (np.nan, np.nan)
        return (float(s.min()), float(s.max()))
    print(f"[PLOT] log_id={args.log_id} | N={len(dfl)} | "
          f"t_rel=[{dfl['t_rel'].min():.2f}, {dfl['t_rel'].max():.2f}] s | "
          f"fused(min,max)={minmax(s_fused)} | ema(min,max)={minmax(s_ema)} | hmm(min,max)={minmax(s_hmm)}")

    # Plot
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.split_root) / "s1c" / "hmm" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"plot_{args.log_id}.png"

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # shaded Viterbi segments (after we have t_rel)
    if "viterbi" in dfl.columns:
        segs = segments_from_viterbi(dfl)
        for a, b in segs:
            ax.axvspan(a, b, color="#ff77aa", alpha=0.15, zorder=0)

    # curves (ensure they are plotted on top)
    ax.plot(dfl["t_rel"], s_xgb,  label="PU-XGB (raw)", color="#999999", linewidth=1.5, alpha=0.6, zorder=2)
    ax.plot(dfl["t_rel"], s_nnpu, label="nnPU (raw)",  color="#ff6666", linewidth=1.5, alpha=0.6, zorder=2)
    ax.plot(dfl["t_rel"], s_fused, label="Fused (logit-sum)", color="#2ca02c", linewidth=2.0, zorder=3)
    ax.plot(dfl["t_rel"], s_ema,   label="EMA(fused)", linestyle="--", linewidth=2.0, zorder=4)
    ax.plot(dfl["t_rel"], s_hmm,   label="HMM posterior", color="#1f77b4", linewidth=2.0, zorder=5)

    ax.set_xlim(left=0, right=max(1e-6, float(dfl["t_rel"].max())))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time since first window (s)")
    ax.set_ylabel("Score / Posterior")
    ax.set_title(f"Temporal Smoothing — {args.log_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")

if __name__ == "__main__":
    main()
