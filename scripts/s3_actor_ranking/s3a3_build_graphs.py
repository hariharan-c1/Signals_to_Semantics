# scripts/s3a3_build_graphs.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("PyG (torch_geometric) is required for S3-A3. Activate your PyG env.") from e

ID_KEYS = ["log_id","window_t_start","window_t_end","window_key","phase_idx","track_uuid"]

def _collect_emb_cols(df):
    return [c for c in df.columns if c.startswith("emb_")]

# Optional aux numeric features to append after embeddings (disabled by default)
AUX_NUM_FEATS = [
    "x_rel_m","y_rel_m","r_rel_m","ttc_s","dmin_m","p_overlap",
    "rel_speed_closing_mps","heading_align_cos","lat_speed_mps","a_norm",
    "map_lane_offset_m","map_lane_alignment_cos","approach_like","crossing_like"
]

EDGE_TYPES_ORDER = ["ego_to_actor","actor_to_ego","actor_to_actor_spatial","actor_temporal"]
EDGE_TYPE_TO_ID = {t:i for i,t in enumerate(EDGE_TYPES_ORDER)}

def _cap_by_distance(nodes_df, max_nodes: int):
    """Keep EGO + closest (max_nodes-1) actors by r_rel_m per window-phase set; applied on the union list."""
    if len(nodes_df) <= max_nodes:
        return nodes_df

    # EGO rows have track_uuid=="EGO", r_rel_m=0; ensure we keep one EGO per phase
    ego_mask = (nodes_df["track_uuid"] == "EGO")
    keep_idx = nodes_df[ego_mask].index.tolist()

    # Non-EGO: sort by r_rel_m ascending, take up to remaining budget globally
    non_ego = nodes_df[~ego_mask].copy()
    non_ego = non_ego.sort_values("r_rel_m", kind="mergesort")
    budget = max(0, max_nodes - len(keep_idx))
    keep_idx += non_ego.index[:budget].tolist()

    return nodes_df.loc[sorted(keep_idx)].copy()

def _build_one_graph(win_nodes: pd.DataFrame,
                     win_edges: pd.DataFrame,
                     emb_cols,
                     use_aux: bool,
                     max_nodes: int,
                     max_spatial_edges: int):
    """Return (pyg Data, stats dict)."""

    # Cap nodes (global per window).
    nodes_capped = _cap_by_distance(win_nodes, max_nodes)

    # Reindex edges to the capped node set
    allowed_ids = set(nodes_capped["track_uuid"].astype(str) + "|" + nodes_capped["phase_idx"].astype(str))
    def _key(sid, ph): return f"{sid}|{ph}"

    edges_capped = win_edges[
        win_edges.apply(lambda r: (_key(str(r["src_id"]), int(r["phase_idx"])) in allowed_ids) and
                                  (_key(str(r["dst_id"]), int(r["phase_idx"])) in allowed_ids), axis=1)
    ].copy()

    # Node index map: unique by (track_uuid, phase_idx) to make one node per actor per phase
    nodes_capped = nodes_capped.copy()
    nodes_capped["node_key"] = nodes_capped["track_uuid"].astype(str) + "|" + nodes_capped["phase_idx"].astype(str)
    node_keys = nodes_capped["node_key"].tolist()
    idx_map = {k:i for i,k in enumerate(node_keys)}

    # --- Build node feature matrix and per-node IDs aligned with X row order ---
    X_list = []
    node_ids = []          # <=== actor track_uuid per node (no phase suffix)
    node_phase_idx = []    # <=== parallel phase index (for inspection)
    for _, r in nodes_capped.iterrows():
        x_vec = r[emb_cols].to_numpy(dtype=np.float32, copy=True)
        if use_aux:
            aux = [r.get(c, np.nan) for c in AUX_NUM_FEATS]
            aux = [0.0 if (a is None or pd.isna(a)) else float(a) for a in aux]
            x_vec = np.concatenate([x_vec, np.asarray(aux, dtype=np.float32)], axis=0)
        X_list.append(x_vec)
        node_ids.append(str(r["track_uuid"]))
        node_phase_idx.append(int(r["phase_idx"]))
    X = torch.from_numpy(np.vstack(X_list))  # [N, D]

    # Edge arrays
    # Split spatial vs temporal to cap spatial edges only (keep all temporal self-links)
    spat_mask = (edges_capped["is_temporal"].astype(int) == 0)
    temp_mask = ~spat_mask

    spat = edges_capped[spat_mask].copy()
    if len(spat) > max_spatial_edges:
        spat = spat.sort_values("dist_m", kind="mergesort").head(max_spatial_edges)

    temp = edges_capped[temp_mask].copy()

    edges_final = pd.concat([spat, temp], axis=0, ignore_index=True)

    def _idx_of(track_uuid, phase_idx):
        return idx_map[f"{str(track_uuid)}|{int(phase_idx)}"]

    src, dst, etype_id, dist = [], [], [], []
    for _, e in edges_final.iterrows():
        s = _idx_of(e["src_id"], e["phase_idx"])
        d = _idx_of(e["dst_id"], e["phase_idx"])
        src.append(s); dst.append(d)
        dist.append(float(e.get("dist_m", 0.0)))
        et = str(e.get("edge_type", "actor_to_actor_spatial"))
        etype_id.append(EDGE_TYPE_TO_ID.get(et, EDGE_TYPE_TO_ID["actor_to_actor_spatial"]))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.from_numpy(np.stack([np.asarray(dist, np.float32),
                                           np.asarray(etype_id, np.int64)], axis=1))

    data = Data(x=X, edge_index=edge_index)
    # Store attributes
    data.edge_attr = edge_attr[:, :1]        # distance as 1D feature
    data.edge_type = edge_attr[:, 1].long()  # categorical id
    data.num_nodes = X.size(0)

    # --- NEW: persist per-node IDs aligned with x row order ---
    data.node_ids = node_ids                 # list[str] actor IDs (track_uuid or "EGO")
    data.node_phase_idx = node_phase_idx     # list[int] phases (0..4)

    # Keep some bookkeeping (strings) on CPU object
    any_row = nodes_capped.iloc[0]
    data.window_key = str(any_row["window_key"])
    data.log_id = str(any_row["log_id"])
    data.window_t_start = float(any_row["window_t_start"])
    data.window_t_end = float(any_row["window_t_end"])

    stats = {
        "n_nodes": int(data.num_nodes),
        "n_edges": int(edge_index.size(1)),
        "n_spatial": int(len(spat)),
        "n_temporal": int(len(temp)),
    }
    return data, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--nodes-parquet", default=None)
    ap.add_argument("--edges-parquet", default=None)
    ap.add_argument("--out-dir", default=None, help="default: <split>/s3/quasi/graphs/<split_name>")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-spatial-edges", type=int, default=120)
    ap.add_argument("--use-aux", action="store_true", help="append aux numeric features after embeddings")
    args = ap.parse_args()

    root = Path(args.split_root)
    s3q = root / "s3" / "quasi"
    nodes_p = Path(args.nodes_parquet) if args.nodes_parquet else (s3q / "nodes.parquet")
    edges_p = Path(args.edges_parquet) if args.edges_parquet else (s3q / "edges.parquet")

    # default out dir
    if args.out_dir is None:
        split_name = Path(args.split_root).name
        out_dir = s3q / "graphs" / split_name
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(nodes_p)
    edges = pd.read_parquet(edges_p)

    # Ensure required cols
    for c in ID_KEYS:
        if c not in nodes.columns:
            raise KeyError(f"nodes.parquet missing '{c}'")
    for c in ["phase_idx","src_id","dst_id","edge_type","is_temporal","dist_m"]:
        if c not in edges.columns:
            raise KeyError(f"edges.parquet missing '{c}'")

    # Embedding columns
    emb_cols = _collect_emb_cols(nodes)
    if not emb_cols:
        raise RuntimeError("No emb_* columns found in nodes.parquet — embeddings are required for S3-A3.")

    # Group by window_key
    man_rows = []
    for widx, (wkey, win_nodes) in enumerate(nodes.groupby("window_key", sort=False)):
        win_edges = edges[edges["window_key"] == wkey]
        data, st = _build_one_graph(
            win_nodes=win_nodes,
            win_edges=win_edges,
            emb_cols=emb_cols,
            use_aux=args.use_aux,
            max_nodes=args.max_nodes,
            max_spatial_edges=args.max_spatial_edges
        )
        out_path = out_dir / f"window_{widx:04d}.pt"
        torch.save(data, out_path)

        man_rows.append({
            "window_key": wkey,
            "file": str(out_path),
            "n_nodes": st["n_nodes"],
            "n_edges": st["n_edges"],
            "n_spatial": st["n_spatial"],
            "n_temporal": st["n_temporal"],
            "emb_dim": len(emb_cols)
        })

    manifest = pd.DataFrame(man_rows)
    manifest_p = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_p, index=False)

    meta = {
        "split_root": str(root),
        "nodes_parquet": str(nodes_p),
        "edges_parquet": str(edges_p),
        "out_dir": str(out_dir),
        "max_nodes": int(args.max_nodes),
        "max_spatial_edges": int(args.max_spatial_edges),
        "use_aux": bool(args.use_aux),
        "n_graphs": int(len(manifest)),
        "emb_dim": int(len(_collect_emb_cols(nodes)))
    }
    (out_dir / "meta_graphs.json").write_text(json.dumps(meta, indent=2))

    print(f"[S3-A3] graphs → {out_dir} | n_graphs={len(manifest)}")
    print(f"[S3-A3] manifest → {manifest_p}")
    print(f"[S3-A3] meta → {out_dir/'meta_graphs.json'}")

if __name__ == "__main__":
    main()
