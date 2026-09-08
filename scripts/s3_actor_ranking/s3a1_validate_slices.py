import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQ = ["log_id","window_t_start","window_t_end","t0","t1","t_on","t_peak",
       "w_pre","w_peak","w_post","source","peak_source","window_key"]

def pct(x): return float(np.round(100.0*x, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--slices-parquet", default=None,
                    help="Override path to slices.parquet (default: <split>/s3/quasi/slices.parquet)")
    ap.add_argument("--hmm-jsonl", default=None,
                    help="Override path to scores_hmm.jsonl (default: <split>/s1c/hmm/scores_hmm.jsonl)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    root = Path(args.split_root)
    slices_path = Path(args.slices_parquet) if args.slices_parquet \
        else root / "s3" / "quasi" / "slices.parquet"
    hmm_path = Path(args.hmm_jsonl) if args.hmm_jsonl \
        else root / "s1c" / "hmm" / "scores_hmm.jsonl"
    index_path = root / "s2" / "index" / "actor_index.parquet"

    if not slices_path.exists():
        raise FileNotFoundError(slices_path)

    df = pd.read_parquet(slices_path).copy()
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise KeyError(f"slices missing columns: {miss}")

    n = len(df)
    # uniqueness of window_key
    dup = int(df["window_key"].duplicated().sum())
    # duration & ordering
    dur = (df["t1"] - df["t0"]).astype(float)
    n_bad_order = int((dur <= 0).sum())
    n_on_mismatch = int((df["t_on"].astype(float) - df["t0"].astype(float)).abs().gt(args.tol).sum())
    # peak in [t0,t1]
    n_peak_out = int(((df["t_peak"] < df["t0"] - args.tol) | (df["t_peak"] > df["t1"] + args.tol)).sum())
    # weights
    wsum = (df["w_pre"] + df["w_peak"] + df["w_post"]).astype(float)
    n_bad_wsum = int((wsum - 1.0).abs().gt(1e-6).sum())
    n_neg_w = int(((df[["w_pre","w_peak","w_post"]] < 0).any(axis=1)).sum())

    # try HMM overlap for those with peak_source == hmm_max_post
    hmm_ok = None
    n_hmm_needed = int((df["peak_source"] == "hmm_max_post").sum())
    n_hmm_verified = 0
    if hmm_path.exists() and n_hmm_needed > 0:
        hmm = pd.read_json(hmm_path, lines=True)
        if {"log_id","t_start","t_end"}.issubset(hmm.columns):
            # center of each hmm row
            hmm = hmm.assign(t_mid = 0.5*(hmm["t_start"].astype(float) + hmm["t_end"].astype(float)))
            # verify each hmm_max_post window has at least 1 hmm row whose t_mid inside [t0,t1]
            need = df[df["peak_source"] == "hmm_max_post"][["log_id","t0","t1","window_key"]].copy()
            need = need.merge(hmm[["log_id","t_mid"]], on="log_id", how="left")
            inside = need[(need["t_mid"] >= need["t0"] - args.tol) & (need["t_mid"] <= need["t1"] + args.tol)]
            n_hmm_verified = int(inside["window_key"].nunique())
            hmm_ok = (n_hmm_verified == n_hmm_needed)

    # cross-check windows exist in S2 index
    n_in_index, n_missing_in_index = None, None
    if index_path.exists():
        idx = pd.read_parquet(index_path)[["log_id","window_t_start","window_t_end"]].drop_duplicates()
        chk = df.merge(idx, on=["log_id","window_t_start","window_t_end"], how="left", indicator=True)
        n_missing_in_index = int((chk["_merge"] == "left_only").sum())
        n_in_index = n - n_missing_in_index

    summary = {
        "paths": {
            "split_root": str(root),
            "slices_parquet": str(slices_path),
            "hmm_scores": str(hmm_path) if hmm_path.exists() else None,
            "actor_index": str(index_path) if index_path.exists() else None
        },
        "rows": {
            "windows": n,
            "duplicate_window_keys": dup
        },
        "durations_s": {
            "min": float(dur.min()),
            "p05": float(dur.quantile(0.05)),
            "p50": float(dur.quantile(0.5)),
            "p95": float(dur.quantile(0.95)),
            "max": float(dur.max())
        },
        "checks": {
            "bad_time_order_or_zero_dur": n_bad_order,
            "t_on_equals_t0_mismatches": n_on_mismatch,
            "t_peak_outside_slice": n_peak_out,
            "neg_weights_rows": n_neg_w,
            "weights_not_sum_1_rows": n_bad_wsum
        },
        "index_alignment": {
            "present_in_index": n_in_index,
            "missing_in_index": n_missing_in_index
        },
        "hmm_peak_verification": {
            "needed": n_hmm_needed,
            "verified": n_hmm_verified,
            "ok": hmm_ok
        }
    }

    # offenders CSVs
    out_dir = slices_path.parent
    if n_bad_order:
        df[(dur <= 0)].to_csv(out_dir / "offenders_bad_order.csv", index=False)
        summary["offenders_bad_order_csv"] = str(out_dir / "offenders_bad_order.csv")
    if n_on_mismatch:
        df[(df["t_on"] - df["t0"]).abs() > args.tol].to_csv(out_dir / "offenders_ton_not_t0.csv", index=False)
        summary["offenders_ton_not_t0_csv"] = str(out_dir / "offenders_ton_not_t0.csv")
    if n_peak_out:
        df[(df["t_peak"] < df["t0"] - args.tol) | (df["t_peak"] > df["t1"] + args.tol)].to_csv(
            out_dir / "offenders_peak_outside.csv", index=False)
        summary["offenders_peak_outside_csv"] = str(out_dir / "offenders_peak_outside.csv")
    if n_neg_w or n_bad_wsum:
        df[((df[["w_pre","w_peak","w_post"]] < 0).any(axis=1)) | ((wsum - 1.0).abs() > 1e-6)].to_csv(
            out_dir / "offenders_bad_weights.csv", index=False)
        summary["offenders_bad_weights_csv"] = str(out_dir / "offenders_bad_weights.csv")
    if index_path.exists() and n_missing_in_index and n_missing_in_index > 0:
        (df.merge(idx, on=["log_id","window_t_start","window_t_end"], how="left", indicator=True)
           .loc[lambda d: d["_merge"]=="left_only", ["log_id","window_t_start","window_t_end","window_key"]]
           .to_csv(out_dir / "offenders_missing_in_index.csv", index=False))
        summary["offenders_missing_in_index_csv"] = str(out_dir / "offenders_missing_in_index.csv")

    out_json = Path(args.out_json) if args.out_json else (out_dir / "sanity_slices_summary.json")
    out_json.write_text(json.dumps(summary, indent=2))

    # terse print
    print(f"[S3-A1-SANITY] windows={n} | dup_keys={dup}")
    print(f"  dur[s] min/p50/max = {summary['durations_s']['min']:.2f} / "
          f"{summary['durations_s']['p50']:.2f} / {summary['durations_s']['max']:.2f}")
    print(f"  bad_order={n_bad_order} | t_on!=t0={n_on_mismatch} | peak_out={n_peak_out}")
    print(f"  neg_w={n_neg_w} | wsum!=1={n_bad_wsum}")
    if n_in_index is not None:
        print(f"  in_index={n_in_index} | missing_in_index={n_missing_in_index}")
    if hmm_path.exists():
        print(f"  hmm peaks needed={n_hmm_needed} | verified={n_hmm_verified} | ok={hmm_ok}")
    print(f"[S3-A1-SANITY] summary → {out_json}")

if __name__ == "__main__":
    main()
