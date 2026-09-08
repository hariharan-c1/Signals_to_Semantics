# Tag candidate actors for each detected braking window.
import argparse, json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

from signals_to_semantics.tagging.prob_overlap import prob_traj_overlap_for_window

# ---- union-then-fill (always returns exactly K) ----
def _fused_score(sc):
    inv1p = 1.0/(1.0 + max(float(sc.dmin_m), 0.0))
    base = 0.50*float(sc.p_overlap) + 0.25*inv1p + 0.20*(1.0 if sc.sustained_tight else 0.0) + 0.05*min(1.0, max(float(sc.a_norm), 0.0)/2.0)
    # tiny tie-break nudges: prefer front & large vehicles slightly
    front_bonus = 0.03 if abs(float(sc.bearing_rad)) <= (70.0*np.pi/180.0) else 0.0
    cat = str(sc.category)
    large_bonus = 0.02 if cat in ("BUS","BOX_TRUCK","TRUCK","TRAILER","LARGE_VEHICLE") else 0.0
    return base + front_bonus + large_bonus

def _union_then_fill(scores, K, front_deg=70.0):
    front_rad = np.radians(front_deg)
    hard, rest = [], []
    for s in scores:
        is_front = (abs(float(s.bearing_rad)) <= front_rad)
        cat = str(s.category)
        is_large_vehicle = (cat in ("BUS","BOX_TRUCK","TRUCK","TRAILER","LARGE_VEHICLE"))
        include = (
            (float(s.p_overlap) >= 0.08) or
            (float(s.dmin_m) <= 12.0) or
            (is_front and float(s.rel_speed_closing_mps) >= 0.5) or
            (int(s.is_static)==1 and (int(s.on_path_like)==1 or float(s.p_overlap) >= 0.05)) or
            (is_front and is_large_vehicle)  # safeguard for front large actors (approach-stop)
        )
        (hard if include else rest).append(s)

    hard.sort(key=_fused_score, reverse=True)
    rest.sort(key=_fused_score, reverse=True)

    if len(hard) >= K:
        return hard[:K]
    need = K - len(hard)
    return hard + rest[:need]

def _tag_record(args_tuple):
    paths_yaml, tag_cfg, row, top_k, front_deg = args_tuple
    log_id = row["log_id"]
    t_start = float(row["t_start"])
    t_end   = float(row.get("t_end", t_start))
    # request full candidate list; YAML should set return_all: true
    scores = prob_traj_overlap_for_window(
        paths_yaml, log_id,
        t_start_s=t_start,
        t_end_s=t_end,
        cfg_path=tag_cfg,
        top_k=None
    )
    K = int(top_k) if top_k is not None else 24
    kept = _union_then_fill(scores, K, front_deg=front_deg)
    return {
        "log_id": log_id,
        "i_start": row.get("i_start"),
        "i_end": row.get("i_end"),
        "t_start": row.get("t_start"),
        "t_end": row.get("t_end"),
        "dur_s": row.get("dur_s"),
        "a_min_ms2": row.get("a_min_ms2"),
        "j_min_ms3": row.get("j_min_ms3"),
        "rescued": bool(row.get("rescued", False)),
        "top_actors": [
            {
                "track_uuid": s.track_uuid,
                "category": s.category,
                "p_overlap": float(s.p_overlap),
                "dmin_m": float(s.dmin_m),
                "penetration_m": float(s.penetration_m),
                "t_at_dmin_s": float(s.t_at_dmin_s),
                "sustained_tight": bool(s.sustained_tight),
                "a_norm": float(s.a_norm),
                "length_m": float(s.actor_length_m),
                "width_m": float(s.actor_width_m),
                "is_static": int(s.is_static),
                "on_path_like": int(s.on_path_like),
                "bearing_rad": float(s.bearing_rad),
                "rel_speed_closing_mps": float(s.rel_speed_closing_mps),
                "sector_id": int(s.sector_id),
            } for s in kept
        ],
    }

def main(paths_yaml: str, tag_cfg: str, in_fp: str, out_fp: str, top_k: int | None, workers: int, front_deg: float):
    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(in_fp) as f_in:
        for line in f_in:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))

    if workers <= 0:
        workers = max(1, cpu_count() - 1)

    args_iter = [(paths_yaml, tag_cfg, r, top_k, front_deg) for r in rows]

    if workers == 1:
        out = [_tag_record(a) for a in args_iter]
    else:
        with Pool(processes=workers) as pool:
            out = list(pool.imap_unordered(_tag_record, args_iter, chunksize=8))
        # restore original order
        key = lambda rec: (rec["log_id"], rec.get("i_start"), rec.get("i_end"), rec.get("t_start"))
        m = {key(rec): rec for rec in out}
        out = [m[key(r)] for r in rows]

    with open(out_fp, "w") as f_out:
        for rec in out:
            f_out.write(json.dumps(rec) + "\n")
    print(f"Tagged {len(out)}/{len(rows)} windows -> {out_fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--tag", required=True, help="tagging YAML")
    ap.add_argument("--in", dest="in_fp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=24)
    ap.add_argument("--workers", type=int, default=0, help="0 => use (CPU-1), 1 => no parallelism")
    ap.add_argument("--front_deg", type=float, default=70.0, help="front sector half-angle in degrees")
    args = ap.parse_args()
    main(args.paths, args.tag, args.in_fp, args.out, top_k=args.top_k, workers=args.workers, front_deg=args.front_deg)
