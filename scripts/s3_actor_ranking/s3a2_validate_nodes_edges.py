# scripts/s3a2_sanity_nodes_edges.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def pct(x): return float(np.round(100.0*x, 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--nodes", default=None, help="default: <split>/s3/quasi/nodes.parquet")
    ap.add_argument("--edges", default=None, help="default: <split>/s3/quasi/edges.parquet")
    ap.add_argument("--expect-phases", default="pre,onset,mid,peak,post")
    args = ap.parse_args()

    root = Path(args.split_root)
    s3q = root / "s3" / "quasi"
    nodes_p = Path(args.nodes) if args.nodes else (s3q / "nodes.parquet")
    edges_p = Path(args.edges) if args.edges else (s3q / "edges.parquet")

    nodes = pd.read_parquet(nodes_p)
    edges = pd.read_parquet(edges_p)

    phases = [p.strip() for p in args.expect_phases.split(",") if p.strip()]

    # Basic counts
    n_nodes, n_edges = len(nodes), len(edges)
    n_windows = nodes[["log_id","window_t_start","window_t_end"]].drop_duplicates().shape[0]

    # Embedding coverage (non-EGO only)
    emb_cols = [c for c in nodes.columns if c.startswith("emb_")]
    if emb_cols:
        non_ego = nodes[nodes["is_ego"]==0]
        has_emb = non_ego[emb_cols].notna().all(axis=1)
        emb_cov = pct(has_emb.mean()) if len(non_ego) else 0.0
    else:
        emb_cov = 0.0

    # Phase presence per window
    by_win = nodes.groupby(["log_id","window_t_start","window_t_end"])["phase"].unique().apply(set)
    missing_phase_windows = int((~by_win.apply(lambda s: set(phases).issubset(s))).sum())

    # Phase trend check (pre vs peak)
    def q_median(df, col, phase):
        d = df[df["phase"]==phase][col].dropna()
        return float(d.median()) if len(d) else np.nan

    r_trend_ok = 0; ttc_trend_ok = 0; win_count = 0
    for (lid,t0,t1), dfw in nodes[nodes["is_ego"]==0].groupby(["log_id","window_t_start","window_t_end"]):
        if {"pre","peak"}.issubset(set(dfw["phase"].unique())):
            r_pre = q_median(dfw,"r_rel_m","pre")
            r_peak= q_median(dfw,"r_rel_m","peak")
            t_pre = q_median(dfw,"ttc_s","pre")
            t_peak= q_median(dfw,"ttc_s","peak")
            if not (np.isnan(r_pre) or np.isnan(r_peak)): r_trend_ok += int(r_pre > r_peak)
            if not (np.isnan(t_pre) or np.isnan(t_peak)): ttc_trend_ok += int(t_pre > t_peak)
            win_count += 1
    r_trend_pct = pct(r_trend_ok / max(1, win_count))
    ttc_trend_pct = pct(ttc_trend_ok / max(1, win_count))

    # Edge integrity
    # Build node keys: (window_key, phase_idx, track_uuid)
    nodes["nk"] = nodes["window_key"].astype(str) + "|" + nodes["phase_idx"].astype(str) + "|" + nodes["track_uuid"].astype(str)
    node_keys = set(nodes["nk"].tolist())
    def ek(row):
        return str(row["window_key"]) + "|" + str(row["phase_idx"]) + "|" + str(row["src_id"]), \
               str(row["window_key"]) + "|" + str(row["phase_idx"]) + "|" + str(row["dst_id"])

    bad_endpoints = 0
    for _, e in edges.iterrows():
        s, d = ek(e)
        if (s not in node_keys) or (d not in node_keys):
            bad_endpoints += 1

    # Spatial distance sanity
    spatial = edges[edges["edge_type"].isin(["ego_to_actor","actor_to_ego","actor_to_actor_spatial"])]
    neg_d = int((spatial["dist_m"] < -1e-6).sum())
    big_ego = 0
    # for ego_to_actor edges, dist should match hypot(x,y) within tolerance → we won't recompute here,
    # just ensure non-negative already covered.

    # Temporal edges sanity (self-links)
    temporal = edges[edges["edge_type"]=="actor_temporal"]
    self_links = int((temporal["src_id"] == temporal["dst_id"]).sum())
    temp_ok = (self_links == len(temporal))

    summary = {
        "paths": {"nodes": str(nodes_p), "edges": str(edges_p)},
        "counts": {"windows": int(n_windows), "nodes": int(n_nodes), "edges": int(n_edges)},
        "embeddings": {"cols": emb_cols[:3] + (["..."] if len(emb_cols)>3 else []), "coverage_non_ego_pct": emb_cov},
        "phases": {"expected": phases, "windows_missing_any_phase": int(missing_phase_windows)},
        "phase_trends_pct": {"median_r_rel_m_pre_gt_peak": r_trend_pct, "median_ttc_pre_gt_peak": ttc_trend_pct},
        "edges_checks": {"bad_endpoints": int(bad_endpoints), "neg_dist_spatial": int(neg_d), "temporal_self_links_ok": bool(temp_ok)}
    }
    out = s3q / "sanity_nodes_edges.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"[S3-A2-SANITY] windows={n_windows} | nodes={n_nodes} | edges={n_edges}")
    print(f"  emb(non-ego) coverage={emb_cov}% | windows missing any expected phase={missing_phase_windows}")
    print(f"  trends: median r_rel_m(pre>peak)={r_trend_pct}% | median ttc(pre>peak)={ttc_trend_pct}%")
    print(f"  edges: bad_endpoints={bad_endpoints} | neg_dist_spatial={neg_d} | temporal_self_links_ok={temp_ok}")
    print(f"[S3-A2-SANITY] summary → {out}")

if __name__ == "__main__":
    main()
