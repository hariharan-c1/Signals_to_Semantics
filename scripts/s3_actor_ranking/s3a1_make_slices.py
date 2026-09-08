# scripts/s3a1_make_slices_from_index.py
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

def mk_window_key(df):
    return (df["log_id"].astype(str) + "|" +
            df["window_t_start"].astype(str) + "|" +
            df["window_t_end"].astype(str))

def iou_intervals(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    uni   = max(a1, b1) - min(a0, b0)
    return inter / (uni + 1e-9)

def load_hmm_jsonl_many(patterns):
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            with open(f, "r") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    r = json.loads(s)
                    # expected keys (extra keys ignored)
                    rows.append({
                        "log_id": str(r["log_id"]),
                        "t_start": float(r["t_start"]),
                        "t_end": float(r["t_end"]),
                        "post_hmm": float(r.get("post_hmm", 0.0)),
                        "viterbi": int(r.get("viterbi", 0)),
                    })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.sort_values(["log_id","t_start","t_end"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--s2-index", default=None,
                    help="defaults to <split_root>/s2/index/actor_index.parquet")
    ap.add_argument("--hmm-jsonl", default=None,
                    help='Path or GLOB to HMM jsonl(s), e.g. ".../s1c/hmm/*.jsonl"')
    ap.add_argument("--w-pre", type=float, default=0.2)
    ap.add_argument("--w-peak", type=float, default=0.6)
    ap.add_argument("--w-post", type=float, default=0.2)

    # Matching robustness knobs
    ap.add_argument("--round-decimals", type=int, default=6,
                    help="Round t0/t1 before matching (mirror S2 index precision).")
    ap.add_argument("--match-eps-sec", type=float, default=0.25,
                    help="Endpoints tolerance (seconds).")
    ap.add_argument("--iou-thr", type=float, default=0.50,
                    help="Min IoU to accept a match (if endpoints not within eps).")
    args = ap.parse_args()

    root = Path(args.split_root)
    s2_index_p = Path(args.s2_index) if args.s2_index else (root / "s2" / "index" / "actor_index.parquet")
    out_dir = root / "s3" / "quasi"
    out_dir.mkdir(parents=True, exist_ok=True)


    # ------- Load windows from S2 index (unique per window_key) -------
    idx = pd.read_parquet(s2_index_p)
    win_cols = ["log_id", "window_t_start", "window_t_end", "window_key"]
    wins = idx[win_cols].drop_duplicates().reset_index(drop=True).copy()
    # Prepare base slice columns
    wins["t0"] = wins["window_t_start"].astype(float)
    wins["t1"] = wins["window_t_end"].astype(float)

    # Optional rounding to match S2 precision
    if args.round_decimals is not None and args.round_decimals >= 0:
        wins["t0"] = wins["t0"].round(args.round_decimals)
        wins["t1"] = wins["t1"].round(args.round_decimals)

    wins["t_on"] = wins["t0"]                                  # quasi-temporal: onset at start
    wins["t_peak"] = 0.5 * (wins["t0"] + wins["t1"])           # default = center
    wins["w_pre"] = float(args.w_pre)
    wins["w_peak"] = float(args.w_peak)
    wins["w_post"] = float(args.w_post)
    wins["source"] = "s2_index"
    wins["peak_source"] = "fallback_center"

    # ------- Robust HMM window matching (optional) -------
    hmm_used = 0
    unmatched = []

    if args.hmm_jsonl:
        hmm_df = load_hmm_jsonl_many([args.hmm_jsonl])
        if hmm_df is not None and len(hmm_df):
            # Group HMM windows per log
            hmm_by_log = {lg: df.reset_index(drop=True) for lg, df in hmm_df.groupby("log_id")}
            for i, r in wins.iterrows():
                lg = str(r["log_id"])
                if lg not in hmm_by_log:
                    unmatched.append((r["window_key"], "no_log_in_hmm"))
                    continue

                cand = hmm_by_log[lg]
                a0, a1 = float(r["t0"]), float(r["t1"])

                # Candidate rows matching either endpoint closeness OR IoU threshold
                def match_row(hr):
                    b0, b1 = float(hr["t_start"]), float(hr["t_end"])
                    near = (abs(a0 - b0) <= args.match_eps_sec and abs(a1 - b1) <= args.match_eps_sec)
                    ok_iou = iou_intervals(a0, a1, b0, b1) >= args.iou_thr
                    return near or ok_iou

                sub = cand[cand.apply(match_row, axis=1)]
                if len(sub) == 0:
                    unmatched.append((r["window_key"], "no_overlap_match"))
                    continue

                # Choose best by (IoU desc, post_hmm desc)
                b_ious = sub.apply(lambda hr: iou_intervals(a0, a1, float(hr["t_start"]), float(hr["t_end"])), axis=1)
                sub = sub.assign(_iou=b_ious.values)
                sub = sub.sort_values(["_iou","post_hmm"], ascending=[False, False]).reset_index(drop=True)
                hr = sub.iloc[0]

                # Set peak to the HMM interval center (consistent & reproducible)
                b0, b1 = float(hr["t_start"]), float(hr["t_end"])
                tp = 0.5 * (b0 + b1)
                wins.at[i, "t_peak"] = tp
                wins.at[i, "peak_source"] = "hmm_max_post"
                hmm_used += 1
        else:
            print("[S3-A1] WARN: HMM jsonl(s) not found or empty → using fallback_center for all.")

    # ------- Save -------
    cols = ["log_id","window_t_start","window_t_end","t0","t1","t_on","t_peak",
            "w_pre","w_peak","w_post","source","peak_source","window_key"]
    wins = wins[cols]
    out_p = out_dir / "slices.parquet"
    wins.to_parquet(out_p, index=False)

    meta = {
        "split_root": str(root),
        "n_windows": int(len(wins)),
        "weights": {"pre": args.w_pre, "peak": args.w_peak, "post": args.w_post},
        "used_sources": ["s2_index"],
        "try_hmm": bool(args.hmm_jsonl),
        "hmm_refined_peaks": int(hmm_used),
        "paths": {
            "slices_parquet": str(out_p),
            "hmm_jsonl": args.hmm_jsonl or ""
        },
        "matching": {
            "round_decimals": args.round_decimals,
            "match_eps_sec": args.match_eps_sec,
            "iou_thr": args.iou_thr
        }
    }
    (out_dir / "meta_slices.json").write_text(json.dumps(meta, indent=2))

    if unmatched:
        drop_p = out_dir / "slices_unmatched_hmm.csv"
        pd.DataFrame(unmatched, columns=["window_key","reason"]).to_csv(drop_p, index=False)

    print(f"[S3-A1] from S2-index → wrote {out_p} | windows={len(wins)}")
    print(f"         weights=(pre={args.w_pre:.2f}, peak={args.w_peak:.2f}, post={args.w_post:.2f})")
    if args.hmm_jsonl:
        print(f"         HMM peak refinement used for {hmm_used} windows "
              f"(round={args.round_decimals}d, eps={args.match_eps_sec:.2f}s, IoU≥{args.iou_thr:.2f})")
        if unmatched:
            print(f"         unmatched list → {out_dir/'slices_unmatched_hmm.csv'}")

if __name__ == "__main__":
    main()
