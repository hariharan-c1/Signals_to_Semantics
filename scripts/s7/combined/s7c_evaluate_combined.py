#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S7C-Combined v2: Query-based scenario retrieval evaluation on combined corpus (val50 + train650)

Adds over v1
------------
1) Dual evaluation runs:
   - Main run: candidate pool controlled by cfg.eval.main_candidate_pool ("open" or "gt_only")
   - GT-constrained run: always candidate pool = gt_only (apples-to-apples baseline comparison)

2) UMAP coloring fix:
   - One color per scenario label (GT and LLM share same color)
   - Marker encodes provenance: GT=circle, LLM=triangle, unlabeled=x
   - Legend split: (A) label colors, (B) provenance markers

Config knobs you can change safely:
- corpus.strategy (winner_per_split vs logical_best)
- eval.query_mode (single vs multi)
- retrieval.dedup_mode (none | window_key | log_id)
- eval.main_candidate_pool (open | gt_only)
- eval.run_gt_constrained (true/false)

Outputs (out_dir)
-----------------
For each run we write:
- <prefix>_per_label.csv
- <prefix>_summary.json
- <prefix>_per_query.csv (only if query_mode=multi)

UMAP:
- umap_combined.png
- umap_points.csv

Assumes v_scenario_trace contains:
  split_name, backend_id, window_key, embedding,
  gt_label_canonical, llm_label_canonical
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ----------------------------
# IO / Config
# ----------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_db_env(env_file: Path) -> Dict[str, Any]:
    load_dotenv(dotenv_path=str(env_file), override=True)
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing DB env vars in {env_file}: {missing}")
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

def connect_db(db_params: Dict[str, Any]):
    return psycopg2.connect(**db_params)


# ----------------------------
# Helpers
# ----------------------------
def parse_log_id(window_key: str) -> str:
    return str(window_key).split("|", 1)[0]

def vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

def safe_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s and s.lower() != "none" else None


# ----------------------------
# Ollama embedding (robust)
# ----------------------------
def embed_text_ollama(text: str, base_url: str, model: str, timeout_s: int) -> List[float]:
    """
    Tries Ollama endpoints:
      - /api/embeddings  (prompt)
      - /api/embed       (input)
    Returns embedding list[float].
    """
    text = str(text)

    # 1) /api/embeddings
    url1 = base_url.rstrip("/") + "/api/embeddings"
    try:
        r = requests.post(url1, json={"model": model, "prompt": text}, timeout=timeout_s)
        if r.status_code == 200:
            j = r.json()
            if "embedding" in j and isinstance(j["embedding"], list):
                return j["embedding"]
    except Exception:
        pass

    # 2) /api/embed
    url2 = base_url.rstrip("/") + "/api/embed"
    r = requests.post(url2, json={"model": model, "input": text}, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()

    # some versions return {"embeddings":[[...]]}
    if "embeddings" in j and isinstance(j["embeddings"], list) and j["embeddings"]:
        return j["embeddings"][0]
    if "embedding" in j and isinstance(j["embedding"], list):
        return j["embedding"]

    raise RuntimeError(f"Ollama embed response missing embedding keys. Keys={list(j.keys())}")


# ----------------------------
# Corpus selection
# ----------------------------
def resolve_backend_map(cfg: Dict[str, Any], for_umap: bool = False) -> Dict[str, int]:
    corpus = cfg["corpus"]
    strategy = corpus["strategy"]

    if for_umap:
        umap_cfg = cfg.get("umap", {})
        mode = str(umap_cfg.get("corpus_for_umap", "strategy"))
        if mode == "winners_per_split":
            return {k: int(v) for k, v in corpus["backend_map_winners"].items()}

    if strategy == "winner_per_split":
        return {k: int(v) for k, v in corpus["backend_map_winners"].items()}
    if strategy == "logical_best":
        return {k: int(v) for k, v in corpus["backend_map_default"].items()}

    raise ValueError(f"Unknown corpus.strategy={strategy}")

def corpus_where_clause(backend_map: Dict[str, int]) -> Tuple[str, List[Any]]:
    """
    Builds: ( (split_name=%s AND backend_id=%s) OR ... )
    Returns SQL string and params.
    """
    parts = []
    params: List[Any] = []
    for split, bid in backend_map.items():
        parts.append("(split_name=%s AND backend_id=%s)")
        params.extend([split, int(bid)])
    return "(" + " OR ".join(parts) + ")", params

def count_gt_support(conn, view_name: str, backend_map: Dict[str, int]) -> Dict[str, int]:
    where, params = corpus_where_clause(backend_map)
    q = f"""
    SELECT gt_label_canonical AS label, COUNT(*) AS n
    FROM {view_name}
    WHERE embedding IS NOT NULL
      AND gt_label_canonical IS NOT NULL
      AND {where}
    GROUP BY gt_label_canonical
    ORDER BY gt_label_canonical;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
    return {str(r["label"]): int(r["n"]) for r in rows}


# ----------------------------
# Retrieval
# ----------------------------
def retrieve(
    conn,
    view_name: str,
    backend_map: Dict[str, int],
    qvec: List[float],
    top_k: int,
    method: str,
    oversample_factor: int,
    dedup_mode: str,
    candidate_pool: str,  # "open" or "gt_only"
) -> List[Dict[str, Any]]:
    """
    Returns list of rows (ranked) after dedup, length == top_k (or less if corpus small).
    Dedup is applied AFTER fetching oversampled list.
    candidate_pool:
      - "open"   : allow GT-missing candidates
      - "gt_only": restrict candidates to gt_label_canonical IS NOT NULL
    """
    assert method in ("ann", "exact")
    assert dedup_mode in ("none", "window_key", "log_id")
    assert candidate_pool in ("open", "gt_only")

    where, params = corpus_where_clause(backend_map)
    limit_fetch = int(max(top_k, 1) * max(1, oversample_factor))
    qvec_lit = vector_literal(qvec)

    extra = ""
    if candidate_pool == "gt_only":
        extra = " AND gt_label_canonical IS NOT NULL\n"

    sql = f"""
    SELECT
      split_name, backend_id, window_key,
      gt_label_canonical, llm_label_canonical
    FROM {view_name}
    WHERE embedding IS NOT NULL
      AND {where}
      {extra}
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    params2 = params + [qvec_lit, int(limit_fetch)]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if method == "exact":
            cur.execute("SET enable_indexscan=off; SET enable_bitmapscan=off; SET enable_seqscan=on;")
        else:
            cur.execute("SET enable_indexscan=on; SET enable_bitmapscan=on; SET enable_seqscan=on;")

        cur.execute(sql, tuple(params2))
        rows = cur.fetchall()

    if not rows:
        return []

    if dedup_mode == "none":
        return rows[:top_k]

    out = []
    seen = set()

    for r in rows:
        wk = str(r["window_key"])
        key = wk if dedup_mode == "window_key" else parse_log_id(wk)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= top_k:
            break

    return out


# ----------------------------
# Metrics (query-based)
# ----------------------------
def precision_recall_f1_at_k(rels: List[int], n_gt: int, k: int) -> Tuple[float, float, float]:
    if k <= 0:
        return (0.0, 0.0, 0.0)
    rels_k = rels[:k] + [0] * max(0, k - len(rels))
    tp = int(sum(rels_k))
    p = tp / float(k)
    r = (tp / float(n_gt)) if n_gt > 0 else 0.0
    f1 = (2*p*r / (p+r)) if (p+r) > 0 else 0.0
    return (p, r, f1)

def recall_at_k(rels: List[int], n_gt: int, k: int) -> float:
    if n_gt <= 0:
        return 0.0
    rels_k = rels[:k]
    return float(sum(rels_k) / float(n_gt))


# ----------------------------
# Evaluation
# ----------------------------
def eval_queries(
    conn,
    cfg: Dict[str, Any],
    backend_map: Dict[str, int],
    candidate_pool: str,   # "open" or "gt_only"
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]:
    view_name = cfg["db"].get("view_name", "v_scenario_trace")

    method = cfg["retrieval"]["method"]
    k_p = int(cfg["retrieval"]["top_k_precision"])
    k_r = int(cfg["retrieval"]["top_k_recall"])
    oversample = int(cfg["retrieval"].get("oversample_factor", 6))
    dedup_mode = str(cfg["retrieval"].get("dedup_mode", "none"))

    query_mode = str(cfg["eval"].get("query_mode", "multi"))
    gt_support_min = int(cfg["eval"].get("gt_support_min", 2))

    ollama = cfg["ollama"]
    base_url = ollama["base_url"]
    emb_model = ollama["embedding_model"]
    timeout_s = int(ollama.get("timeout_s", 120))
    embed_dim = int(ollama.get("embed_dim", 768))

    gt_counts = count_gt_support(conn, view_name, backend_map)

    per_label_rows = []
    per_query_rows = []

    for label, queries in cfg["queries"].items():
        label = str(label)
        n_gt = int(gt_counts.get(label, 0))

        if not isinstance(queries, list) or len(queries) == 0:
            continue

        use_queries = [queries[0]] if query_mode == "single" else queries

        q_metrics = []
        for qi, qtext in enumerate(use_queries):
            qvec = embed_text_ollama(qtext, base_url, emb_model, timeout_s)
            if len(qvec) != embed_dim:
                raise RuntimeError(f"Embedding dim mismatch: got {len(qvec)} expected {embed_dim}")

            retrieved = retrieve(
                conn=conn,
                view_name=view_name,
                backend_map=backend_map,
                qvec=qvec,
                top_k=max(k_p, k_r),
                method=method,
                oversample_factor=oversample,
                dedup_mode=dedup_mode,
                candidate_pool=candidate_pool,
            )

            # Relevance uses GT label only (baseline-style)
            rels = [1 if safe_label(r.get("gt_label_canonical")) == label else 0 for r in retrieved]

            p10, r10, f110 = precision_recall_f1_at_k(rels, n_gt=n_gt, k=k_p)
            r50 = recall_at_k(rels, n_gt=n_gt, k=k_r)

            q_metrics.append((p10, r10, f110, r50))

            per_query_rows.append({
                "label": label,
                "candidate_pool": candidate_pool,
                "query_idx": int(qi),
                "query_text": str(qtext),
                "N_GT": int(n_gt),
                "P@10": float(p10),
                "R@10": float(r10),
                "F1@10": float(f110),
                "R@50": float(r50),
                "retrieved_count": int(len(retrieved)),
            })

        arr = np.asarray(q_metrics, dtype=float)  # shape (Q, 4)
        p10_m, r10_m, f1_m, r50_m = arr.mean(axis=0) if len(arr) else (0.0, 0.0, 0.0, 0.0)

        per_label_rows.append({
            "label": label,
            "candidate_pool": candidate_pool,
            "N_GT": int(n_gt),
            "n_queries_used": int(len(use_queries)),
            "P@10": float(p10_m),
            "R@10": float(r10_m),
            "F1@10": float(f1_m),
            "R@50": float(r50_m),
        })

    df_label = (
        pd.DataFrame(per_label_rows)
        .sort_values(by=["R@50", "P@10"], ascending=[False, False])
        .reset_index(drop=True)
    )
    df_query = pd.DataFrame(per_query_rows) if per_query_rows else None

    df_macro_base = df_label[df_label["N_GT"] >= gt_support_min].copy()
    summary = {
        "backend_map_used": backend_map,
        "candidate_pool": candidate_pool,
        "query_mode": query_mode,
        "method": method,
        "dedup_mode": dedup_mode,
        "k_precision": k_p,
        "k_recall": k_r,
        "gt_support_min": gt_support_min,
        "n_labels_total": int(len(df_label)),
        "n_labels_macro": int(len(df_macro_base)),
        "macro_P@10": float(df_macro_base["P@10"].mean()) if len(df_macro_base) else None,
        "macro_R@10": float(df_macro_base["R@10"].mean()) if len(df_macro_base) else None,
        "macro_F1@10": float(df_macro_base["F1@10"].mean()) if len(df_macro_base) else None,
        "macro_R@50": float(df_macro_base["R@50"].mean()) if len(df_macro_base) else None,
        "gt_counts": gt_counts,
    }

    return df_label, summary, df_query


# ----------------------------
# UMAP (winner-per-split combined)
# ----------------------------
def fetch_embeddings_for_umap(
    conn,
    view_name: str,
    backend_map: Dict[str, int],
    sample_n: int,
) -> pd.DataFrame:
    where, params = corpus_where_clause(backend_map)
    q = f"""
    SELECT
      split_name, backend_id, window_key,
      embedding,
      gt_label_canonical,
      llm_label_canonical
    FROM {view_name}
    WHERE embedding IS NOT NULL
      AND {where}
    ;
    """
    df = pd.read_sql(q, conn, params=tuple(params))

    if df.empty:
        raise RuntimeError("UMAP: No embeddings returned for selected corpus mapping.")

    if sample_n and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    return df

def parse_pgvector_str(s: Any, expected_dim: int) -> np.ndarray:
    if s is None:
        raise ValueError("NULL embedding")
    if isinstance(s, (list, tuple, np.ndarray)):
        arr = np.asarray(s, dtype=np.float32)
    else:
        t = str(s).strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1].strip()
        parts = [p.strip() for p in t.split(",") if p.strip()]
        arr = np.asarray([float(p) for p in parts], dtype=np.float32)

    if arr.shape[0] != expected_dim:
        raise ValueError(f"dim mismatch got={arr.shape[0]} expected={expected_dim}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in embedding")
    return arr

def run_umap_plot(
    conn,
    cfg: Dict[str, Any],
    backend_map_umap: Dict[str, int],
    out_dir: Path,
) -> Tuple[str, str]:
    view_name = cfg["db"].get("view_name", "v_scenario_trace")
    embed_dim = int(cfg["ollama"].get("embed_dim", 768))

    um = cfg.get("umap", {})
    sample_n = int(um.get("sample_n", 0))
    n_neighbors = int(um.get("n_neighbors", 15))
    min_dist = float(um.get("min_dist", 0.10))
    metric = str(um.get("metric", "cosine"))
    random_state = int(um.get("random_state", 42))

    df = fetch_embeddings_for_umap(conn, view_name, backend_map_umap, sample_n=sample_n)

    X_list = []
    label_for_color = []
    provenance = []
    log_ids = []
    window_keys = []

    bad = 0
    for _, r in df.iterrows():
        try:
            vec = parse_pgvector_str(r["embedding"], expected_dim=embed_dim)
        except Exception:
            bad += 1
            continue

        gt = safe_label(r.get("gt_label_canonical"))
        llm = safe_label(r.get("llm_label_canonical"))

        # Color label: GT if present else LLM else unlabeled
        if gt is not None:
            lab = gt
            src = "GT"
        elif llm is not None:
            lab = llm
            src = "LLM"
        else:
            lab = "unlabeled"
            src = "unlabeled"

        X_list.append(vec)
        label_for_color.append(lab)
        provenance.append(src)
        wk = str(r["window_key"])
        window_keys.append(wk)
        log_ids.append(parse_log_id(wk))

    if len(X_list) < 20:
        raise RuntimeError(f"UMAP: too few valid vectors (ok={len(X_list)}, bad={bad})")

    X = np.vstack(X_list)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X2 = reducer.fit_transform(X)

    # Sidecar CSV
    pts_csv = out_dir / "umap_points.csv"
    pd.DataFrame({
        "window_key": window_keys,
        "log_id": log_ids,
        "label": label_for_color,
        "provenance": provenance,
        "umap1": X2[:, 0],
        "umap2": X2[:, 1],
    }).to_csv(pts_csv, index=False)

    # ---- Plot: one color per label; marker by provenance ----
    plt.figure(figsize=(12, 9))

    labs = np.asarray(label_for_color)
    prov = np.asarray(provenance)

    unique_labels = sorted(set(label_for_color))

    # Use Matplotlib default color cycle, mapped deterministically per label
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    color_map = {lab: cycle[i % len(cycle)] for i, lab in enumerate(unique_labels)}

    def scatter_group(label: str, mask: np.ndarray, marker: str, edge: bool):
        if not np.any(mask):
            return
        xs = X2[mask, 0]
        ys = X2[mask, 1]
        plt.scatter(
            xs, ys,
            s=18, alpha=0.65,
            marker=marker,
            c=[color_map[label]],
            edgecolors="black" if edge else "none",
            linewidths=0.35 if edge else 0.0,
        )

    for lab in unique_labels:
        m_lab = (labs == lab)
        scatter_group(lab, m_lab & (prov == "GT"), marker="o", edge=True)
        scatter_group(lab, m_lab & (prov == "LLM"), marker="^", edge=False)
        scatter_group(lab, m_lab & (prov == "unlabeled"), marker="x", edge=False)

    plt.title("S7 Combined Corpus: Scenario Embedding Space (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Legend A: labels (colors)
    label_handles = [Patch(facecolor=color_map[lab], edgecolor="none", label=lab) for lab in unique_labels]
    leg1 = plt.legend(
        handles=label_handles,
        title="Scenario label (color)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.gca().add_artist(leg1)

    # Legend B: provenance (markers)
    marker_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=7, label="GT (circle)"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=7, label="LLM-only (triangle)"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=7, label="unlabeled (x)"),
    ]
    plt.legend(
        handles=marker_handles,
        title="Provenance (marker)",
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()

    fig_path = out_dir / "umap_combined.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return str(fig_path), str(pts_csv)


# ----------------------------
# Main
# ----------------------------
def write_run_outputs(out_dir: Path, prefix: str, df_label: pd.DataFrame, summary: Dict[str, Any], df_query: Optional[pd.DataFrame]) -> None:
    out_csv = out_dir / f"{prefix}_per_label.csv"
    df_label.to_csv(out_csv, index=False)

    out_json = out_dir / f"{prefix}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if df_query is not None and len(df_query) > 0:
        out_q = out_dir / f"{prefix}_per_query.csv"
        df_query.to_csv(out_q, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/s7_eval_combined.yaml")
    ap.add_argument("--mode", default="both", choices=["eval", "umap", "both"], help="Run evaluation, UMAP, or both")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    out_dir = Path(cfg["eval"]["out_dir"])
    ensure_dir(out_dir)

    db_params = load_db_env(Path(cfg["db"]["env_file"]))
    conn = connect_db(db_params)

    try:
        backend_map_eval = resolve_backend_map(cfg, for_umap=False)

        if args.mode in ("eval", "both"):
            main_pool = str(cfg["eval"].get("main_candidate_pool", "open")).strip().lower()
            if main_pool not in ("open", "gt_only"):
                raise ValueError("eval.main_candidate_pool must be 'open' or 'gt_only'")

            prefix_main = str(cfg["eval"].get("output_prefix_main", f"s7c_query_eval_{main_pool}"))

            print(f"[INFO] Running combined-corpus query evaluation (main pool={main_pool})...")
            df_label, summary, df_query = eval_queries(conn, cfg, backend_map_eval, candidate_pool=main_pool)
            write_run_outputs(out_dir, prefix_main, df_label, summary, df_query)

            print(f"[INFO] Wrote: {out_dir / (prefix_main + '_per_label.csv')}")
            print(f"[INFO] Wrote: {out_dir / (prefix_main + '_summary.json')}")
            if df_query is not None:
                print(f"[INFO] Wrote: {out_dir / (prefix_main + '_per_query.csv')}")

            print("[INFO] Macro (labels with GT support >= min):")
            print(f"  macro_P@10 = {summary['macro_P@10']}")
            print(f"  macro_R@10 = {summary['macro_R@10']}")
            print(f"  macro_F1@10 = {summary['macro_F1@10']}")
            print(f"  macro_R@50 = {summary['macro_R@50']}")

            # ---- GT-constrained apples-to-apples run (optional but recommended) ----
            if bool(cfg["eval"].get("run_gt_constrained", True)):
                prefix_gt = str(cfg["eval"].get("output_prefix_gt", "s7c_query_eval_gt_only"))
                print("[INFO] Running GT-constrained candidate-pool evaluation (apples-to-apples baseline)...")
                df_label2, summary2, df_query2 = eval_queries(conn, cfg, backend_map_eval, candidate_pool="gt_only")
                write_run_outputs(out_dir, prefix_gt, df_label2, summary2, df_query2)

                print(f"[INFO] Wrote: {out_dir / (prefix_gt + '_per_label.csv')}")
                print(f"[INFO] Wrote: {out_dir / (prefix_gt + '_summary.json')}")
                if df_query2 is not None:
                    print(f"[INFO] Wrote: {out_dir / (prefix_gt + '_per_query.csv')}")

        if cfg.get("umap", {}).get("enabled", True) and args.mode in ("umap", "both"):
            print("[INFO] Building combined UMAP...")
            backend_map_umap = resolve_backend_map(cfg, for_umap=True)
            fig, pts = run_umap_plot(conn, cfg, backend_map_umap, out_dir=out_dir)
            print(f"[INFO] UMAP saved: {fig}")
            print(f"[INFO] UMAP points CSV: {pts}")

        print("[INFO] Done.")
    finally:
        conn.close()
        print("[INFO] DB connection closed.")


if __name__ == "__main__":
    main()
