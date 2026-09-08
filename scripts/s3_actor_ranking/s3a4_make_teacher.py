# scripts/s3a4_make_teacher.py
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

# Map arbitrary phase names to slice weights
def phase_weight(phase: str, w_pre: float, w_peak: float, w_post: float) -> float:
    p = (phase or "").lower()
    if p == "pre":
        return w_pre
    if p in ("onset", "rise"):
        return 0.5 * (w_pre + w_peak)
    if p in ("mid", "middle"):
        return w_peak
    if p == "peak":
        return w_peak
    if p == "post":
        return w_post
    # default: treat as central if unknown
    return w_peak

def safe_num(x, default=0.0, lo=None, hi=None) -> float:
    try:
        if pd.isna(x): v = default
        else: v = float(x)
    except Exception:
        v = default
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def actor_base_prior(category: str) -> float:
    cat = (category or "").upper()
    if cat in ("REGULAR_VEHICLE","TRUCK","BUS","LARGE_VEHICLE","BOX_TRUCK"):
        return 1.0
    if cat in ("PEDESTRIAN","BICYCLE","MOTORCYCLE"):
        return 0.9
    if cat in ("BOLLARD","STOP_SIGN","SIGN"):
        return 0.6
    return 0.8

def per_phase_score(row,
                    ttc_tau=2.0,
                    dist_rho=20.0,
                    closing_gain=0.6,
                    align_gain=0.2,
                    approach_gain=0.2) -> float:
    # lower TTC, smaller distance, more closing, more alignment, approach-like ⇒ higher
    ttc  = safe_num(row.get("ttc_s"), 30.0, lo=0.0, hi=30.0)
    dist = safe_num(row.get("r_rel_m"), 80.0, lo=0.0, hi=80.0)

    dterm = math.exp(-dist / max(1e-6, dist_rho))
    tterm = math.exp(-ttc  / max(1e-6, ttc_tau))

    rel_close = safe_num(row.get("rel_speed_closing_mps"), 0.0)   # closing if negative
    closing_term = math.exp(closing_gain * max(0.0, -rel_close))

    align = safe_num(row.get("map_lane_alignment_cos"), 0.0)      # [-1,1]
    align_term = 1.0 + align_gain * max(0.0, align)

    approach_flag = 1.0 + approach_gain * (safe_num(row.get("approach_like"), 0.0) > 0.5)

    cat_prior = actor_base_prior(str(row.get("category","")))
    score = dterm * tterm * closing_term * align_term * approach_flag * cat_prior
    return float(score)

def softmax_temp(x: np.ndarray, tau: float) -> np.ndarray:
    if x.size == 0: return x
    x = x.astype(np.float64)
    x = x - x.max()
    z = np.exp(x / max(1e-6, tau))
    s = z.sum()
    return z / max(s, 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--nodes-parquet", default=None)
    ap.add_argument("--slices-parquet", default=None)
    ap.add_argument("--tau", type=float, default=0.8, help="softmax temperature")
    ap.add_argument("--eps", type=float, default=0.02, help="uniform smoothing mass")
    ap.add_argument("--out-parquet", default=None)
    # scoring knobs (rarely need to change)
    ap.add_argument("--ttc-tau", type=float, default=2.0)
    ap.add_argument("--dist-rho", type=float, default=20.0)
    ap.add_argument("--closing-gain", type=float, default=0.6)
    ap.add_argument("--align-gain", type=float, default=0.2)
    ap.add_argument("--approach-gain", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.split_root)
    qdir = root / "s3" / "quasi"
    nodes_p  = Path(args.nodes_parquet)  if args.nodes_parquet  else (qdir / "nodes.parquet")
    slices_p = Path(args.slices_parquet) if args.slices_parquet else (qdir / "slices.parquet")
    out_p    = Path(args.out_parquet)    if args.out_parquet    else (qdir / "teacher.parquet")

    if not nodes_p.exists():  raise FileNotFoundError(nodes_p)
    if not slices_p.exists(): raise FileNotFoundError(slices_p)

    nodes = pd.read_parquet(nodes_p)
    slices = pd.read_parquet(slices_p)[["window_key","w_pre","w_peak","w_post"]].drop_duplicates()

    # Non-EGO only
    nodes = nodes[nodes["track_uuid"] != "EGO"].copy()
    if "phase" not in nodes.columns:
        raise KeyError("nodes.parquet is missing 'phase' column.")
    nodes["phase"] = nodes["phase"].astype(str)

    # Compute per-(actor,phase) physics score
    nodes["score_phase"] = nodes.apply(
        lambda r: per_phase_score(
            r,
            ttc_tau=args.ttc_tau,
            dist_rho=args.dist_rho,
            closing_gain=args.closing_gain,
            align_gain=args.align_gain,
            approach_gain=args.approach_gain,
        ),
        axis=1
    )

    # Attach slice weights and map to per-row phase weight
    nodes = nodes.merge(slices, on="window_key", how="left")
    if nodes[["w_pre","w_peak","w_post"]].isna().any().any():
        raise ValueError("Missing slice weights after merge. Check window_key alignment.")

    nodes["phase_weight"] = nodes.apply(
        lambda r: phase_weight(r["phase"], r["w_pre"], r["w_peak"], r["w_post"]), axis=1
    )

    # Weighted sum across phases per actor
    nodes["contrib"] = nodes["score_phase"] * nodes["phase_weight"]
    agg = (nodes
           .groupby(["window_key","track_uuid","category"], as_index=False)["contrib"]
           .sum()
           .rename(columns={"contrib":"score_window"}))

    # Softmax + eps per window to get q(a)
    rows = []
    eps = max(0.0, min(0.5, float(args.eps)))
    for wkey, g in agg.groupby("window_key", sort=False):
        s = g["score_window"].to_numpy(dtype=np.float64) + 1e-9
        p = softmax_temp(s, args.tau)
        p = (1.0 - eps) * p + eps * (1.0 / len(p))
        p = p / max(p.sum(), 1e-12)
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({
                "window_key": wkey,
                "track_uuid": r["track_uuid"],
                "category":   r["category"],
                "q":          float(p[i]),
                "score_window_raw": float(r["score_window"])
            })
    teacher = pd.DataFrame(rows)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    teacher.to_parquet(out_p, index=False)

    meta = {
        "split_root": str(root),
        "nodes_parquet": str(nodes_p),
        "slices_parquet": str(slices_p),
        "out_parquet": str(out_p),
        "tau": args.tau,
        "eps": eps,
        "n_windows": int(teacher["window_key"].nunique()),
        "n_actor_rows": int(len(teacher))
    }
    (qdir / "meta_teacher.json").write_text(json.dumps(meta, indent=2))
    print(f"[S3-A4] teacher → {out_p} | windows={meta['n_windows']} | rows={meta['n_actor_rows']} "
          f"(tau={args.tau}, eps={eps})")

if __name__ == "__main__":
    main()
