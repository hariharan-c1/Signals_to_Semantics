# scripts/s1d_finalize_scores.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load_hmm(split_root: str):
    p = Path(split_root) / "s1c" / "hmm" / "scores_hmm.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run s1c_hmm_smooth.py first.")
    return pd.read_json(p, lines=True)

def segments_from_viterbi(df_log: pd.DataFrame):
    # assumes df_log sorted by t_start and has columns: t_start, t_end, post_hmm, viterbi
    segs = []
    v = df_log["viterbi"].astype(int).to_numpy()
    if len(v) == 0: return segs
    start = 0
    for i in range(1, len(v)+1):
        if i == len(v) or v[i] != v[i-1]:
            if v[start] == 1:
                chunk = df_log.iloc[start:i]
                t0 = float(chunk["t_start"].iloc[0])
                t1 = float(chunk["t_end"].iloc[-1])
                dur = t1 - t0
                segs.append({
                    "log_id": chunk["log_id"].iloc[0],
                    "t0": t0,
                    "t1": t1,
                    "duration_s": dur,
                    "post_max": float(chunk["post_hmm"].max()),
                    "post_mean": float(chunk["post_hmm"].mean()),
                    "n_windows": int(len(chunk)),
                })
            start = i
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g., artifacts/train650_val50/train650")
    ap.add_argument("--mode", default="hmm", choices=["hmm","ema","fused"],
                    help="which score to export as score_final (default hmm)")
    args = ap.parse_args()

    df = load_hmm(args.split_root)
    # ensure required cols exist
    needed = ["log_id","t_start","t_end","score_pu_xgb","score_nnpu","score_fused","score_ema","post_hmm"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Column {c} missing in HMM scores. Re-run s1c_hmm_smooth.py.")

    # choose final
    if args.mode == "hmm":
        df["score_final"] = df["post_hmm"].astype(float)
    elif args.mode == "ema":
        df["score_final"] = df["score_ema"].astype(float)
    else:
        df["score_final"] = df["score_fused"].astype(float)

    # sort for segment building
    df = df.sort_values(["log_id","t_start"]).reset_index(drop=True)

    # write final scores parquet
    out_dir = Path(args.split_root) / "s1d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_scores = out_dir / "final_scores.parquet"
    cols_out = ["log_id","t_start","t_end",
                "score_pu_xgb","score_nnpu","score_fused","score_ema","post_hmm","score_final"]
    df[cols_out].to_parquet(out_scores, index=False)
    print(f"[S1D] wrote {out_scores} | rows={len(df)}")

    # episodes (per-log Viterbi segments)
    if "viterbi" in df.columns:
        seg_rows = []
        for lid, dlog in df.groupby("log_id"):
            dlog = dlog.sort_values("t_start")
            seg_rows += segments_from_viterbi(dlog)
        ep = pd.DataFrame(seg_rows)
        out_ep = out_dir / "episodes.csv"
        ep.to_csv(out_ep, index=False)
        print(f"[S1D] wrote {out_ep} | segments={len(ep)}")
    else:
        print("[S1D] viterbi column missing; episodes not produced.")

    # small meta
    meta = {
        "split_root": args.split_root,
        "mode": args.mode,
        "rows": int(len(df)),
        "generated_from": "s1c/hmm/scores_hmm.jsonl",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[S1D] meta saved → {out_dir / 'meta.json'}")

if __name__ == "__main__":
    main()
