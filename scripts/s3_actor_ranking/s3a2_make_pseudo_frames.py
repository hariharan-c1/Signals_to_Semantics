# scripts/s3a2_make_pseudo_frames.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PHASES_5 = ["pre","onset","mid","peak","post"]
C_5 = [0.2, 0.4, 0.7, 1.0, 0.4]

FEAT_NUM = [
    "x_rel_m","y_rel_m","r_rel_m","dmin_m","p_overlap","t_at_dmin_s","sustained_tight",
    "rel_speed_closing_mps","lat_speed_mps","a_norm","length_m","width_m","bearing_rad",
    "ttc_s","dist_lt3","dist_lt5","dist_lt8","approach_like","crossing_like",
    "heading_align_cos","long_gap_m","lat_offset_m","map_lane_offset_m","map_lane_alignment_cos",
    "dist_to_stopline_m","dist_to_crosswalk_m","in_drivable_area","rank_p_overlap",
    "margin_to_top_p","rank_inv_dmin","margin_to_top_inv_d","post_hmm","score_final"
]
ID_KEYS = ["log_id","window_t_start","window_t_end","track_uuid"]

def window_key(df: pd.DataFrame) -> pd.Series:
    return (df["log_id"].astype(str) + "|" +
            df["window_t_start"].astype(str) + "|" +
            df["window_t_end"].astype(str))

def _clamp01(x): return np.minimum(1.0, np.maximum(0.0, x))
def _safe_float(row, name, default=0.0):
    return float(row[name]) if (name in row and pd.notna(row[name])) else float(default)

def _coerce_emb_df_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all emb_* cols to float32; non-parsable become NaN."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        return df
    num = df[emb_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df = df.drop(columns=emb_cols).join(num)
    return df

def load_embeddings(emb_dir: Path):
    """Union train/val embeddings, de-dup on ID_KEYS. Coerce to float32."""
    out = []
    for name in ["embeddings_train.parquet","embeddings_val.parquet"]:
        p = emb_dir / name
        if p.exists():
            df = pd.read_parquet(p)
            df = _coerce_emb_df_to_float32(df)
            cols = [c for c in df.columns if c.startswith("emb_")]
            keep = ["log_id","window_t_start","window_t_end","track_uuid"] + cols
            out.append(df[keep].copy())
    if not out:
        return None, 0
    emb = pd.concat(out, axis=0, ignore_index=True)
    # sort + dedup ensures we keep a stable first occurrence
    emb = emb.sort_values(ID_KEYS).drop_duplicates(subset=ID_KEYS, keep="first")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    # ensure column order & float32 dtype
    emb[emb_cols] = emb[emb_cols].astype(np.float32)
    return emb, len(emb_cols)

def _extract_emb(row: pd.Series, emb_dim: int):
    """Return (float32 vector, ok) for emb_* in row; ok=False if any NaN or missing."""
    if emb_dim <= 0:
        return None, False
    cols = [f"emb_{j:03d}" for j in range(emb_dim)]
    if not all(c in row.index for c in cols):
        return None, False
    # Convert to numeric array (already coerced at DF level but be safe)
    arr = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if arr.shape[0] != emb_dim or np.isnan(arr).any():
        return None, False
    return arr, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--slices-parquet", default=None, help="default: <split>/s3/quasi/slices.parquet")
    ap.add_argument("--index-parquet", default=None, help="default: <split>/s2/index/actor_index.parquet")
    ap.add_argument("--emb-dir", default=None, help="default: <split>/s2/contrastive")
    ap.add_argument("--phases", default="pre,onset,mid,peak,post")
    ap.add_argument("--phase-coef", default="0.2,0.4,0.7,1.0,0.4")
    ap.add_argument("--max-actors", type=int, default=15)
    ap.add_argument("--prox-m", type=float, default=30.0)
    ap.add_argument("--ego-node", action="store_true", default=True)
    ap.add_argument("--ego-emb-zero", action="store_true", help="fill EGO emb_* with zeros if embeddings exist")
    ap.add_argument("--emb-noise", type=float, default=0.0, help="Gaussian noise on actor embeddings (e.g., 0.01)")
    ap.add_argument("--out-nodes", default="nodes.parquet")
    ap.add_argument("--out-edges", default="edges.parquet")
    args = ap.parse_args()

    root = Path(args.split_root)
    s3q = root / "s3" / "quasi"
    s3q.mkdir(parents=True, exist_ok=True)

    slices_p = Path(args.slices_parquet) if args.slices_parquet else (s3q / "slices.parquet")
    idx_p    = Path(args.index_parquet) if args.index_parquet else (root / "s2" / "index" / "actor_index.parquet")
    emb_dir  = Path(args.emb_dir) if args.emb_dir else (root / "s2" / "contrastive")

    slices = pd.read_parquet(slices_p)
    idx = pd.read_parquet(idx_p)

    # Keep only windows present in slices
    key_slices = set(window_key(slices).tolist())
    idx["window_key"] = window_key(idx)
    idx = idx[idx["window_key"].isin(key_slices)].copy()

    # Trim to needed columns
    cols_keep = list(dict.fromkeys(ID_KEYS + [c for c in FEAT_NUM if c in idx.columns] + ["category"]))
    idx = idx[cols_keep].copy()

    # Top-K actors per window by proximity
    idx["rank_by_r"] = idx.groupby(["log_id","window_t_start","window_t_end"])["r_rel_m"].rank(method="first", ascending=True)
    kept = idx[idx["rank_by_r"] <= args.max_actors].drop(columns=["rank_by_r"]).copy()

    # Load embeddings (coerced to float32)
    emb, emb_dim = load_embeddings(emb_dir)
    if emb is not None and emb_dim > 0:
        kept = kept.merge(emb, on=ID_KEYS, how="left")
    else:
        print("[S3-A2][WARN] embeddings not found; proceeding without emb_*")
        emb_dim = 0

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    coef = [float(x) for x in args.phase_cof.split(",")] if hasattr(args, "phase_cof") else [float(x) for x in args.phase_coef.split(",")]
    assert len(phases) == len(coef), "phases and phase-coef length mismatch"

    # Trend hyperparams
    k_r, k_ttc, k_d = 0.15, 0.20, 0.20
    k_pov, k_v, k_align = 0.15, 0.25, 0.10
    k_pull_xy = 0.05

    node_rows, edge_rows = [], []
    missing_emb_rows = 0

    for (lid, t0, t1), actors in kept.groupby(["log_id","window_t_start","window_t_end"], sort=False):
        wkey = f"{lid}|{t0}|{t1}"
        actors = actors.reset_index(drop=True)
        actor_ids = actors["track_uuid"].astype(str).tolist()

        for p_idx, (p_name, c) in enumerate(zip(phases, coef)):
            # EGO node
            if args.ego_node:
                ego_row = {
                    "log_id": lid, "window_t_start": t0, "window_t_end": t1, "window_key": wkey,
                    "phase_idx": p_idx, "phase": p_name, "track_uuid": "EGO",
                    "x_rel_m": 0.0, "y_rel_m": 0.0, "r_rel_m": 0.0, "ttc_s": np.nan,
                    "dmin_m": np.nan, "p_overlap": np.nan, "rel_speed_closing_mps": np.nan,
                    "heading_align_cos": np.nan, "lat_speed_mps": np.nan, "a_norm": np.nan,
                    "length_m": np.nan, "width_m": np.nan, "map_lane_offset_m": np.nan,
                    "map_lane_alignment_cos": np.nan, "dist_to_stopline_m": np.nan,
                    "dist_to_crosswalk_m": np.nan, "approach_like": np.nan,
                    "crossing_like": np.nan, "post_hmm": np.nan, "score_final": np.nan,
                    "is_ego": 1, "category": "EGO"
                }
                # Optional zero-emb for EGO to avoid NaNs downstream
                if emb_dim > 0 and args.ego_emb_zero:
                    for j in range(emb_dim):
                        ego_row[f"emb_{j:03d}"] = 0.0
                node_rows.append(ego_row)

            # ACTOR nodes
            for _, r in actors.iterrows():
                is_app = (_safe_float(r, "approach_like", 0.0) >= 0.5)
                x = _safe_float(r,"x_rel_m"); y = _safe_float(r,"y_rel_m")
                rrel = max(0.0, _safe_float(r,"r_rel_m"))
                ttc  = max(1e-3, _safe_float(r,"ttc_s", 999.0))
                dmin = max(0.0, _safe_float(r,"dmin_m", rrel))
                pov  = _safe_float(r,"p_overlap", 0.0)
                vcl  = _safe_float(r,"rel_speed_closing_mps", 0.0)
                align= _safe_float(r,"heading_align_cos", 0.0)

                rrel_p = max(0.0, rrel * (1.0 - k_r * c))
                ttc_p  = max(1e-3, ttc  * (1.0 - k_ttc * c))
                dmin_p = max(0.0, dmin * (1.0 - k_d * c))
                pov_p  = _clamp01(pov + k_pov * c)
                vcl_p  = vcl + k_v * c
                align_p= np.clip((1.0 - k_align*c)*align + (k_align*c)*1.0, -1.0, 1.0)

                if is_app:
                    x_p = x * (1.0 - k_pull_xy * c)
                    y_p = y * (1.0 - 0.5*k_pull_xy * c)
                else:
                    x_p, y_p = x, y

                row_out = {
                    "log_id": lid, "window_t_start": t0, "window_t_end": t1, "window_key": wkey,
                    "phase_idx": p_idx, "phase": p_name, "track_uuid": r["track_uuid"],
                    "category": r.get("category","UNKNOWN"), "is_ego": 0,
                    "x_rel_m": x_p, "y_rel_m": y_p, "r_rel_m": rrel_p, "ttc_s": ttc_p,
                    "dmin_m": dmin_p, "p_overlap": pov_p, "rel_speed_closing_mps": vcl_p,
                    "heading_align_cos": align_p,
                    "lat_speed_mps": _safe_float(r,"lat_speed_mps", 0.0),
                    "a_norm": _safe_float(r,"a_norm", 0.0),
                    "length_m": _safe_float(r,"length_m", 0.0),
                    "width_m": _safe_float(r,"width_m", 0.0),
                    "map_lane_offset_m": _safe_float(r,"map_lane_offset_m", 0.0),
                    "map_lane_alignment_cos": _safe_float(r,"map_lane_alignment_cos", 0.0),
                    "dist_to_stopline_m": _safe_float(r,"dist_to_stopline_m", np.nan),
                    "dist_to_crosswalk_m": _safe_float(r,"dist_to_crosswalk_m", np.nan),
                    "approach_like": _safe_float(r,"approach_like", 0.0),
                    "crossing_like": _safe_float(r,"crossing_like", 0.0),
                    "post_hmm": _safe_float(r,"post_hmm", 0.0),
                    "score_final": _safe_float(r,"score_final", 0.0),
                }

                # Attach embeddings if present and valid
                if emb_dim > 0:
                    vec, ok = _extract_emb(r, emb_dim)
                    if ok:
                        if args.emb_noise > 0:
                            noise = np.random.normal(0.0, args.emb_noise, size=vec.shape).astype(np.float32)
                            vec = vec + noise
                            nrm = np.linalg.norm(vec)
                            if nrm > 1e-8: vec = vec / nrm
                        for j in range(emb_dim):
                            row_out[f"emb_{j:03d}"] = float(vec[j])
                    else:
                        missing_emb_rows += 1

                node_rows.append(row_out)

            # Build spatial edges for this phase
            # Gather just-created rows for this window+phase
            span = len(actors) + (1 if args.ego_node else 0)
            phase_nodes = node_rows[-span:]
            ego_present = args.ego_node
            act_pos = [(nr["track_uuid"], nr["x_rel_m"], nr["y_rel_m"])
                       for nr in phase_nodes if nr["is_ego"] == 0]

            # ego ↔ actor edges
            if ego_present:
                for aid, ax, ay in act_pos:
                    dist = float(np.hypot(ax, ay))
                    for et in ["ego_to_actor","actor_to_ego"]:
                        src, dst = ("EGO", aid) if et=="ego_to_actor" else (aid, "EGO")
                        edge_rows.append({
                            "log_id": lid, "window_t_start": t0, "window_t_end": t1, "window_key": wkey,
                            "phase_idx": p_idx, "src_id": src, "dst_id": dst,
                            "edge_type": et, "dist_m": dist, "is_temporal": 0
                        })
            # actor ↔ actor spatial (bi-directional)
            if len(act_pos) >= 2:
                A = np.array([[ax, ay] for (_, ax, ay) in act_pos], dtype=np.float32)
                ids = [aid for (aid,_,_) in act_pos]
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        dist = float(np.linalg.norm(A[i] - A[j]))
                        if dist <= args.prox_m:
                            for src, dst in [(ids[i],ids[j]), (ids[j],ids[i])]:
                                edge_rows.append({
                                    "log_id": lid, "window_t_start": t0, "window_t_end": t1, "window_key": wkey,
                                    "phase_idx": p_idx, "src_id": src, "dst_id": dst,
                                    "edge_type": "actor_to_actor_spatial", "dist_m": dist, "is_temporal": 0
                                })

        # temporal edges (actor self-links across phases)
        if len(phases) >= 2:
            for p_idx in range(len(phases)-1):
                for aid in actor_ids:
                    edge_rows.append({
                        "log_id": lid, "window_t_start": t0, "window_t_end": t1, "window_key": wkey,
                        "phase_idx": p_idx, "src_id": aid, "dst_id": aid,
                        "edge_type": "actor_temporal", "dist_m": 0.0, "is_temporal": 1
                    })

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows)

    nodes_path = s3q / args.out_nodes
    edges_path = s3q / args.out_edges
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)

    meta = {
        "split_root": str(root),
        "slices_parquet": str(slices_p),
        "index_parquet": str(idx_p),
        "emb_dir": str(emb_dir),
        "n_windows": int(slices.shape[0]),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "phases": phases, "phase_coef": coef,
        "max_actors": int(args.max_actors), "prox_m": float(args.prox_m),
        "ego_node": bool(args.ego_node), "ego_emb_zero": bool(args.ego_emb_zero),
        "emb_noise": float(args.emb_noise),
        "emb_dim": int(emb_dim),
        "missing_embedding_rows": int(missing_emb_rows)
    }
    (s3q / "meta_nodes_edges.json").write_text(json.dumps(meta, indent=2))
    print(f"[S3-A2] nodes → {nodes_path} (rows={len(nodes)})")
    print(f"[S3-A2] edges → {edges_path} (rows={len(edges)})")
    print(f"[S3-A2] meta  → {s3q/'meta_nodes_edges.json'}")
    if missing_emb_rows:
        print(f"[S3-A2][WARN] non-EGO rows missing embeddings: {missing_emb_rows}")

if __name__ == "__main__":
    main()
