# scripts/s3a3_sanity_graphs.py
import argparse, json
from pathlib import Path
import pandas as pd
import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("PyG (torch_geometric) is required to load graphs.") from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", required=True, help="e.g. artifacts/.../s3/quasi/graphs/dev100")
    args = ap.parse_args()

    gdir = Path(args.graphs_dir)
    man_p = gdir / "manifest.parquet"
    if not man_p.exists():
        raise FileNotFoundError(f"manifest not found: {man_p}")

    man = pd.read_parquet(man_p)
    n = len(man)
    if n == 0:
        raise RuntimeError("manifest is empty")

    sizes = []
    edge_splits = {"n_spatial":0, "n_temporal":0}
    deg_zeros = 0
    emb_dim = None
    bad_edge_idx = 0

    for _, r in man.iterrows():
        d: Data = torch.load(r["file"], map_location="cpu")
        N = int(d.num_nodes); E = int(d.edge_index.size(1))
        sizes.append((N,E))
        # degrees
        deg = torch.bincount(d.edge_index[0], minlength=N) + torch.bincount(d.edge_index[1], minlength=N)
        deg_zeros += int((deg == 0).sum().item())
        # counts
        edge_splits["n_spatial"] += int((d.edge_type != 3).sum().item())
        edge_splits["n_temporal"] += int((d.edge_type == 3).sum().item())
        # emb dim
        if emb_dim is None:
            emb_dim = int(d.x.size(1))

        # sanity on indices
        if d.edge_index.min() < 0 or d.edge_index.max() >= N:
            bad_edge_idx += 1

    Ns = [s[0] for s in sizes]; Es = [s[1] for s in sizes]
    summary = {
        "graphs_dir": str(gdir),
        "n_graphs": int(n),
        "nodes_min": int(min(Ns)), "nodes_med": float(pd.Series(Ns).median()), "nodes_max": int(max(Ns)),
        "edges_min": int(min(Es)), "edges_med": float(pd.Series(Es).median()), "edges_max": int(max(Es)),
        "n_graphs_with_isolated_nodes": int(deg_zeros > 0),
        "total_spatial_edges": int(edge_splits["n_spatial"]),
        "total_temporal_edges": int(edge_splits["n_temporal"]),
        "emb_dim_first_graph": emb_dim,
        "bad_edge_index_graphs": int(bad_edge_idx),
    }
    out = gdir / "sanity_graphs.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"[S3-A3-SANITY] graphs={n} | "
          f"N(min/med/max)={summary['nodes_min']}/{summary['nodes_med']}/{summary['nodes_max']} | "
          f"E(min/med/max)={summary['edges_min']}/{summary['edges_med']}/{summary['edges_max']} | "
          f"iso_nodes={'YES' if deg_zeros>0 else 'NO'} | "
          f"bad_edge_idx={summary['bad_edge_index_graphs']}")
    print(f"[S3-A3-SANITY] summary → {out}")

if __name__ == "__main__":
    main()
