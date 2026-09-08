#!/usr/bin/env python3
"""
S7 Evaluation Suite (Production / Thesis-grade)

Implements your agreed evaluation framing (Tier-1 + Tier-2) and adds
APPENDIX-only "experimental" metrics behind a switch.

Main (thesis-safe, conservative):
- S7.1 GT availability per label (N_GT)
- S7.4 Query-centric retrieval (4 semantic queries per label, no filters)
  * GT-Precision@K and GT-Coverage@K (coverage = retrieved GT hits / N_GT(label))
  * Precision fairness: K_eff = min(K, N_GT(label)) for precision only
  * best / mean / worst across queries per label
- Tier-2: RRF fusion across the 4 query runs (stability)
- Discovery metrics (no GT required):
  * LLM-consistency@K, new_windows@K, high_conf_new@K (based on thresholds)
- Failure analysis:
  * Confusion report (what GT labels appear instead among GT-present retrieved windows)
- Embedding plot:
  * UMAP with stable color per label and marker per provenance:
    - GT: circle
    - LLM-only: triangle
    - unlabeled: x
  * Clean legends: one for labels (colors), one for provenance (markers)
  * Sidecar CSV: embedding_umap_points.csv

Appendix-only (enable with --experimental or eval.experimental=true):
- GT-constrained upper bound metrics:
  * retrieval pool restricted to GT-only candidates (gt_present=true)
  * output: gt_only_metrics_constrained.csv
- Query extended pseudo-label metrics (DO NOT call these recall):
  * llm_hit_rate@K (LLM predicted label match rate)
  * consensus_hit_rate@K (LLM label + thresholds)
  * output: query_metrics_extended_experimental.csv

Assumes DB view: v_scenario_trace (as you posted).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv

import matplotlib.pyplot as plt

try:
    import umap  # type: ignore
except Exception:
    umap = None


# -------------------------
# IO / Utils
# -------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_db(cfg: Dict[str, Any]):
    env_file = cfg.get("env_file")
    if env_file:
        load_dotenv(dotenv_path=str(env_file), override=False)
        cfg = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "dbname": os.getenv("DB_NAME"),
        }

    required = ("host", "port", "user", "password", "dbname")
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing database configuration fields: {', '.join(missing)}")

    return psycopg2.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        user=cfg["user"],
        password=cfg["password"],
        dbname=cfg["dbname"],
    )


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fetch_df(conn, query: str, params: Tuple[Any, ...]) -> pd.DataFrame:
    """Stable DB fetch without pandas read_sql warnings (psycopg2 cursor)."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


# -------------------------
# pgvector helpers
# -------------------------
def parse_pgvector(v: Any, expected_dim: Optional[int] = None) -> np.ndarray:
    if v is None:
        raise ValueError("embedding is NULL")

    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=np.float32)
    elif isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        arr = np.asarray([float(x) for x in parts], dtype=np.float32)
    else:
        return parse_pgvector(str(v), expected_dim=expected_dim)

    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(f"dim mismatch: got {arr.shape[0]} expected {expected_dim}")
    if not np.isfinite(arr).all():
        raise ValueError("embedding contains non-finite values")
    return arr


def vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec.tolist()) + "]"


# -------------------------
# Ollama embeddings
# -------------------------
def ollama_embed(base_url: str, model: str, text: str, timeout_s: int = 60) -> np.ndarray:
    url = base_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding", None)
    if emb is None:
        raise RuntimeError(f"Ollama embeddings API returned no 'embedding' for model={model}")
    return np.asarray(emb, dtype=np.float32)


# -------------------------
# Feature helpers
# -------------------------
def extract_gat_top1_score(gat_top3_json: Any) -> float:
    """gat_top3_json is jsonb array like [{'rank':1,'score':...}, ...]"""
    if gat_top3_json is None:
        return 0.0
    try:
        obj = json.loads(gat_top3_json) if isinstance(gat_top3_json, str) else gat_top3_json
        if not isinstance(obj, list) or len(obj) == 0:
            return 0.0
        for it in obj:
            if isinstance(it, dict) and it.get("rank") == 1:
                return safe_float(it.get("score"), 0.0)
        return safe_float(obj[0].get("score"), 0.0)
    except Exception:
        return 0.0


def normalize_series(s: pd.Series) -> pd.Series:
    """Robust normalization to [0,1] (winsorized via percentiles)."""
    if s.empty:
        return s
    x = s.astype(float)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x)), index=s.index)
    y = (x - lo) / (hi - lo)
    return y.clip(0.0, 1.0)


def apply_rerank(df: pd.DataFrame, rerank_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Evidence-aware rerank. final_score = sim_score + w_evidence * evidence_score"""
    if df.empty:
        return df

    enabled = bool(rerank_cfg.get("enabled", False))
    if not enabled:
        out = df.copy()
        out["evidence_score"] = 0.0
        out["final_score"] = out["sim_score"].astype(float)
        return out

    w_ev = float(rerank_cfg.get("w_evidence", 0.25))
    w = rerank_cfg.get("weights", {})

    out = df.copy()
    ev = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)

    # booleans
    if "near_crosswalk" in w:
        ev += float(w["near_crosswalk"]) * out["near_crosswalk"].fillna(False).astype(int)
    if "near_stopline" in w:
        ev += float(w["near_stopline"]) * out["near_stopline"].fillna(False).astype(int)
    if "has_close_actor" in w:
        ev += float(w["has_close_actor"]) * out["has_close_actor"].fillna(False).astype(int)

    # counts (normalized)
    if "num_ped_near_crosswalk" in w and "num_ped_near_crosswalk" in out.columns:
        ev += float(w["num_ped_near_crosswalk"]) * normalize_series(out["num_ped_near_crosswalk"].fillna(0.0))
    if "num_vru_near_crosswalk" in w and "num_vru_near_crosswalk" in out.columns:
        ev += float(w["num_vru_near_crosswalk"]) * normalize_series(out["num_vru_near_crosswalk"].fillna(0.0))

    # continuous (normalized)
    if "episode_score_final" in w:
        ev += float(w["episode_score_final"]) * normalize_series(out["episode_score_final"].fillna(0.0))
    if "peak_decel_mps2" in w:
        ev += float(w["peak_decel_mps2"]) * normalize_series(out["peak_decel_mps2"].fillna(0.0))
    if "gat_top1_score" in w:
        ev += float(w["gat_top1_score"]) * normalize_series(out["gat_top1_score"].fillna(0.0))

    out["evidence_score"] = ev
    out["final_score"] = out["sim_score"].astype(float) + w_ev * out["evidence_score"].astype(float)
    return out


def dedup_by_log(df: pd.DataFrame) -> pd.DataFrame:
    """Keep best candidate per log_id by final_score."""
    if df.empty:
        return df
    df2 = df.sort_values("final_score", ascending=False).drop_duplicates(subset=["log_id"], keep="first")
    return df2.reset_index(drop=True)


# -------------------------
# Retrieval
# -------------------------
def retrieve_candidates(
    conn,
    split: str,
    query_vec: np.ndarray,
    top_k: int,
    backend_id: int,
    backend_scope_all: bool,
    exclude_log_id: Optional[str],
    gt_present_only: bool,
) -> pd.DataFrame:
    """
    Returns candidates with distance and metadata.
    dist uses pgvector <=> (cosine distance if index/operator class configured).
    We convert to sim_score = 1 - dist (monotone ranking surrogate).
    """
    qvec_lit = vector_literal(query_vec)

    if backend_scope_all:
        q = """
        SELECT
          window_key, log_id,
          gt_label_canonical, llm_label_canonical, llm_confidence,
          episode_score_final, peak_decel_mps2, near_crosswalk, near_stopline,
          has_close_actor, num_ped_near_crosswalk, num_vru_near_crosswalk,
          gat_top3_json,
          backend_id,
          (embedding <=> %s::vector) AS dist
        FROM v_scenario_trace
        WHERE split_name=%s AND embedding IS NOT NULL
        """
        params: List[Any] = [qvec_lit, split]
    else:
        q = """
        SELECT
          window_key, log_id,
          gt_label_canonical, llm_label_canonical, llm_confidence,
          episode_score_final, peak_decel_mps2, near_crosswalk, near_stopline,
          has_close_actor, num_ped_near_crosswalk, num_vru_near_crosswalk,
          gat_top3_json,
          backend_id,
          (embedding <=> %s::vector) AS dist
        FROM v_scenario_trace
        WHERE split_name=%s AND backend_id=%s AND embedding IS NOT NULL
        """
        params = [qvec_lit, split, backend_id]

    if gt_present_only:
        q += " AND gt_label_canonical IS NOT NULL "

    if exclude_log_id:
        q += " AND log_id::text != %s "
        params.append(str(exclude_log_id))

    q += " ORDER BY embedding <=> %s::vector LIMIT %s; "
    params.extend([qvec_lit, int(top_k)])

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q, tuple(params))
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["dist"] = df["dist"].astype(float)
    df["sim_score"] = 1.0 - df["dist"]
    df["gat_top1_score"] = df["gat_top3_json"].apply(extract_gat_top1_score)

    # ensure numeric columns exist and are numeric
    for col in ["llm_confidence", "episode_score_final", "peak_decel_mps2", "gat_top1_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)

    return df


# -------------------------
# Metrics
# -------------------------
def gt_precision_at_k(gt_labels: List[Optional[str]], target: str) -> float:
    if not gt_labels:
        return 0.0
    return float(sum(1 for x in gt_labels if x == target) / len(gt_labels))


def gt_coverage_at_k(gt_labels: List[Optional[str]], target: str, n_gt: int) -> float:
    """Coverage is 'how many GT positives recovered' / N_GT(label)."""
    if n_gt <= 0:
        return float("nan")
    return float(sum(1 for x in gt_labels if x == target) / n_gt)


def rrf_fuse(rank_lists: List[List[str]], k: int, rrf_k: int = 60) -> List[str]:
    """
    Reciprocal Rank Fusion for window_key lists.
    score(d) = sum_i 1/(rrf_k + rank_i(d))
    """
    scores: Dict[str, float] = {}
    for lst in rank_lists:
        for idx, wk in enumerate(lst, start=1):
            scores[wk] = scores.get(wk, 0.0) + 1.0 / (rrf_k + idx)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [wk for wk, _ in fused[:k]]


# -------------------------
# UMAP data
# -------------------------
def fetch_embeddings_for_umap(conn, split: str, backend_id: int, embed_dim: int) -> pd.DataFrame:
    q = """
    SELECT
      window_key, log_id,
      gt_label_canonical, llm_label_canonical,
      embedding
    FROM v_scenario_trace
    WHERE split_name=%s AND backend_id=%s AND embedding IS NOT NULL;
    """
    df = fetch_df(conn, q, (split, backend_id))
    if df.empty:
        return df

    def provenance(row) -> str:
        if row.get("gt_label_canonical"):
            return "GT"
        if row.get("llm_label_canonical"):
            return "LLM"
        return "unlabeled"

    def final_label(row) -> str:
        return row.get("gt_label_canonical") or row.get("llm_label_canonical") or "unlabeled"

    df["provenance"] = df.apply(provenance, axis=1)
    df["label_final"] = df.apply(final_label, axis=1)

    X_list: List[np.ndarray] = []
    keep: List[bool] = []
    for _, r in df.iterrows():
        try:
            v = parse_pgvector(r["embedding"], expected_dim=embed_dim)
            X_list.append(v)
            keep.append(True)
        except Exception:
            keep.append(False)

    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        return df
    df["_emb"] = list(X_list)
    return df


# -------------------------
# Core evaluation
# -------------------------
def run_suite(cfg: Dict[str, Any]) -> None:
    db_cfg = cfg["db"]
    ev = cfg["eval"]
    out = cfg["outputs"]
    oll = cfg["ollama"]
    rerank_cfg = cfg.get("rerank", {"enabled": False})
    queries_cfg = cfg["queries"]

    split = str(ev["split_name"])
    backend_id = int(ev["backend_id"])
    backend_scope_all = bool(ev.get("backend_scope_all", False))
    dedup_logs = bool(ev.get("dedup_by_log_id", True))
    exclude_same_log = bool(ev.get("exclude_same_log", False))
    top_k_list = [int(x) for x in ev["top_k_list"]]

    gt_support_min = int(ev.get("gt_support_min", 2))

    # discovery / consensus thresholds
    llm_conf_min = float(ev.get("llm_conf_min", 0.85))
    s1_min = float(ev.get("s1_min", 0.0))
    gat_top1_min = float(ev.get("gat_top1_min", 0.0))

    experimental = bool(ev.get("experimental", False))
    gt_constrained_metrics = bool(ev.get("gt_constrained_metrics", True))

    out_dir = str(out["out_dir"])
    ensure_dir(out_dir)

    print("[INFO] Connecting to DB...")
    conn = connect_db(db_cfg)

    try:
        # ---------------- S7.1 GT support ----------------
        q_support = """
        SELECT gt_label_canonical AS label, COUNT(*)::int AS n_gt
        FROM v_scenario_trace
        WHERE split_name=%s AND backend_id=%s AND gt_label_canonical IS NOT NULL
        GROUP BY gt_label_canonical
        ORDER BY 1;
        """
        df_support = fetch_df(conn, q_support, (split, backend_id))

        # include all labels (even if n_gt==0) from YAML queries
        all_labels = sorted(set(queries_cfg.keys()))
        support_map = {str(r["label"]): int(r["n_gt"]) for _, r in df_support.iterrows()} if not df_support.empty else {}
        df_support_full = pd.DataFrame([{"label": lab, "n_gt": int(support_map.get(lab, 0))} for lab in all_labels])
        df_support_full.to_csv(os.path.join(out_dir, "label_support.csv"), index=False)

        n_gt_by_label = {r["label"]: int(r["n_gt"]) for _, r in df_support_full.iterrows()}

        # optional GT log list (for exclude_same_log)
        q_gt_logs = """
        SELECT gt_label_canonical AS label, log_id::text AS log_id
        FROM v_scenario_trace
        WHERE split_name=%s AND backend_id=%s AND gt_label_canonical IS NOT NULL;
        """
        df_gt_logs = fetch_df(conn, q_gt_logs, (split, backend_id))

        # ---------------- Query-centric metrics ----------------
        per_label_rows: List[Dict[str, Any]] = []
        per_query_rows_extended: List[Dict[str, Any]] = []  # experimental
        discovery_rows: List[Dict[str, Any]] = []
        confusion_rows: List[Dict[str, Any]] = []

        Kmax = max(top_k_list)

        # helper: compute one query run and return ranked topK frame + ranklist
        def run_one_query(qtext: str, exclude_log_id: Optional[str], gt_present_only: bool) -> Tuple[pd.DataFrame, List[str]]:
            qvec = ollama_embed(oll["base_url"], oll["embedding_model"], qtext)
            df_ret = retrieve_candidates(
                conn=conn,
                split=split,
                query_vec=qvec,
                top_k=(Kmax * (5 if backend_scope_all else 3)),
                backend_id=backend_id,
                backend_scope_all=backend_scope_all,
                exclude_log_id=exclude_log_id,
                gt_present_only=gt_present_only,
            )
            if df_ret.empty:
                return df_ret, []
            df_ret = apply_rerank(df_ret, rerank_cfg)
            df_rank = df_ret.sort_values("final_score", ascending=False).reset_index(drop=True)
            if dedup_logs:
                df_rank = dedup_by_log(df_rank)
            df_top = df_rank.head(Kmax).reset_index(drop=True)
            return df_top, df_top["window_key"].astype(str).tolist()

        # per-label aggregation buffers
        # {label: {K: [values...]}}
        p_by_label: Dict[str, Dict[int, List[float]]] = {lab: {K: [] for K in top_k_list} for lab in all_labels}
        cov_by_label: Dict[str, Dict[int, List[float]]] = {lab: {K: [] for K in top_k_list} for lab in all_labels}

        # for confusion counts
        confusion_bags: Dict[str, List[str]] = {lab: [] for lab in all_labels}

        for label in all_labels:
            qlist = queries_cfg[label]
            n_gt = int(n_gt_by_label.get(label, 0))
            support_ok = (n_gt >= gt_support_min)

            rank_lists_for_rrf: List[List[str]] = []

            for qtext in qlist:
                # exclude same log only for supported labels (otherwise meaningless)
                exclude_log_id = None
                if exclude_same_log and support_ok:
                    sub = df_gt_logs[df_gt_logs["label"] == label]
                    if not sub.empty:
                        exclude_log_id = str(sub.iloc[0]["log_id"])

                df_top, rank_list = run_one_query(qtext, exclude_log_id=exclude_log_id, gt_present_only=False)
                rank_lists_for_rrf.append(rank_list)

                # handle empty retrieval
                if df_top.empty:
                    for K in top_k_list:
                        p_by_label[label][K].append(0.0)
                        cov_by_label[label][K].append(0.0 if n_gt > 0 else float("nan"))
                    # discovery placeholder
                    discovery_rows.append({
                        "query_label": label,
                        "query_text": qtext,
                        "top_k": Kmax,
                        "dominant_llm_label": None,
                        "llm_consistency_at_k": 0.0,
                        "new_windows_in_topk": 0,
                        "high_conf_new_in_topk": 0,
                        "backend_scope": "all" if backend_scope_all else str(backend_id),
                        "llm_conf_min": llm_conf_min,
                        "s1_min": s1_min,
                        "gat_top1_min": gat_top1_min,
                    })
                    continue

                # Discovery metrics @Kmax
                llm_labs = df_top["llm_label_canonical"].fillna("unlabeled").astype(str).tolist()
                dom = max(set(llm_labs), key=lambda x: llm_labs.count(x)) if llm_labs else None
                cons = (llm_labs.count(dom) / len(llm_labs)) if (llm_labs and dom is not None) else 0.0
                new_windows = int(df_top["gt_label_canonical"].isna().sum())

                df_new = df_top[df_top["gt_label_canonical"].isna()].copy()
                df_high = df_new[
                    (df_new["llm_confidence"] >= llm_conf_min) &
                    (df_new["episode_score_final"] >= s1_min) &
                    (df_new["gat_top1_score"] >= gat_top1_min)
                ]
                high_conf_new = int(len(df_high))

                discovery_rows.append({
                    "query_label": label,
                    "query_text": qtext,
                    "top_k": Kmax,
                    "dominant_llm_label": dom,
                    "llm_consistency_at_k": float(cons),
                    "new_windows_in_topk": int(new_windows),
                    "high_conf_new_in_topk": int(high_conf_new),
                    "backend_scope": "all" if backend_scope_all else str(backend_id),
                    "llm_conf_min": llm_conf_min,
                    "s1_min": s1_min,
                    "gat_top1_min": gat_top1_min,
                })

                # Confusion bag: look only at GT-present retrieved windows, record their GT labels
                df_gtpresent = df_top[df_top["gt_label_canonical"].notna()].copy()
                confusion_bags[label].extend(df_gtpresent["gt_label_canonical"].astype(str).tolist())

                # Metrics per K (conservative GT-based)
                for K in top_k_list:
                    df_k = df_top.head(K)
                    gt_labels = df_k["gt_label_canonical"].tolist()

                    # precision fairness: K_eff = min(K, N_GT(label)) for precision only
                    k_eff = min(K, n_gt) if n_gt > 0 else K
                    if k_eff != K:
                        df_ke = df_top.head(k_eff)
                        p = gt_precision_at_k(df_ke["gt_label_canonical"].tolist(), label)
                    else:
                        p = gt_precision_at_k(gt_labels, label)

                    cov = gt_coverage_at_k(gt_labels, label, n_gt) if n_gt > 0 else float("nan")

                    p_by_label[label][K].append(float(p))
                    cov_by_label[label][K].append(float(cov))

                    # Experimental extended per-query metrics (Appendix-only)
                    if experimental:
                        gt_hits = int((df_k["gt_label_canonical"] == label).sum())
                        llm_hits = int((df_k["llm_label_canonical"] == label).sum())

                        cons_hits = int((
                            (df_k["llm_label_canonical"] == label) &
                            (df_k["llm_confidence"] >= llm_conf_min) &
                            (df_k["episode_score_final"] >= s1_min) &
                            (df_k["gat_top1_score"] >= gat_top1_min)
                        ).sum())

                        new_k = int(df_k["gt_label_canonical"].isna().sum())
                        high_conf_new_k = int((
                            df_k["gt_label_canonical"].isna() &
                            (df_k["llm_confidence"] >= llm_conf_min) &
                            (df_k["episode_score_final"] >= s1_min) &
                            (df_k["gat_top1_score"] >= gat_top1_min)
                        ).sum())

                        per_query_rows_extended.append({
                            "label": label,
                            "query_text": qtext,
                            "K": int(K),
                            "n_gt": int(n_gt),
                            "gt_hits_in_topk": gt_hits,
                            "gt_precision_at_k": float(gt_hits / K),
                            "gt_coverage_at_k": float(gt_hits / n_gt) if n_gt > 0 else float("nan"),
                            "llm_hits_in_topk": llm_hits,
                            "llm_hit_rate_at_k": float(llm_hits / K),
                            "consensus_hits_in_topk": cons_hits,
                            "consensus_hit_rate_at_k": float(cons_hits / K),
                            "new_windows_in_topk": new_k,
                            "high_conf_new_in_topk": high_conf_new_k,
                            "backend_scope": "all" if backend_scope_all else str(backend_id),
                            "llm_conf_min": llm_conf_min,
                            "s1_min": s1_min,
                            "gat_top1_min": gat_top1_min,
                        })

            # aggregate best/mean/worst over the 4 queries per label
            for K in top_k_list:
                ps = p_by_label[label][K]
                cvs = cov_by_label[label][K]
                if len(ps) == 0:
                    continue
                per_label_rows.append({
                    "label": label,
                    "K": int(K),
                    "n_gt": int(n_gt),
                    "gt_precision_best": float(np.nanmax(ps)),
                    "gt_precision_mean": float(np.nanmean(ps)),
                    "gt_precision_worst": float(np.nanmin(ps)),
                    "gt_coverage_best": float(np.nanmax(cvs)) if n_gt > 0 else float("nan"),
                    "gt_coverage_mean": float(np.nanmean(cvs)) if n_gt > 0 else float("nan"),
                    "gt_coverage_worst": float(np.nanmin(cvs)) if n_gt > 0 else float("nan"),
                    "gt_support_ok": bool(support_ok),
                })

            # confusion report: only meaningful if there exists any GT-present retrieved windows
            if confusion_bags[label]:
                counts = pd.Series(confusion_bags[label]).value_counts()
                # remove self to report what it confuses with
                if label in counts.index:
                    counts = counts.drop(index=label)
                for conf_lab, c in counts.head(5).items():
                    confusion_rows.append({
                        "label": label,
                        "confused_with": str(conf_lab),
                        "count": int(c),
                        "n_gt_label": int(n_gt),
                    })

        df_per_label = pd.DataFrame(per_label_rows)
        df_discovery = pd.DataFrame(discovery_rows)
        df_conf = pd.DataFrame(confusion_rows)

        df_per_label.to_csv(os.path.join(out_dir, "query_metrics_per_label.csv"), index=False)
        df_discovery.to_csv(os.path.join(out_dir, "discovery_metrics.csv"), index=False)
        df_conf.to_csv(os.path.join(out_dir, "confusion_report.csv"), index=False)

        if experimental and per_query_rows_extended:
            pd.DataFrame(per_query_rows_extended).to_csv(
                os.path.join(out_dir, "query_metrics_extended_experimental.csv"),
                index=False
            )

        # ---------------- Macro/Micro headline (supported labels only) ----------------
        df_supp = df_per_label[df_per_label["gt_support_ok"] == True].copy()
        included_labels = sorted(df_supp["label"].unique().tolist())

        macro_micro_rows: List[Dict[str, Any]] = []
        for K in sorted(set(df_supp["K"].tolist())):
            dfk = df_supp[df_supp["K"] == K].copy()
            if dfk.empty:
                continue

            # Macro: mean over labels (using mean over queries)
            macro_p = float(dfk["gt_precision_mean"].mean())
            macro_cov = float(dfk["gt_coverage_mean"].mean())

            # Micro: weighted by n_gt (overall behavior)
            w = dfk["n_gt"].astype(float).clip(lower=1.0)
            micro_p = float(np.average(dfk["gt_precision_mean"], weights=w))
            micro_cov = float(np.average(dfk["gt_coverage_mean"], weights=w))

            macro_micro_rows.append({
                "K": int(K),
                "macro_gt_precision_mean_over_queries": macro_p,
                "macro_gt_coverage_mean_over_queries": macro_cov,
                "micro_gt_precision_mean_over_queries": micro_p,
                "micro_gt_coverage_mean_over_queries": micro_cov,
                "labels_included": int(len(included_labels)),
            })

        pd.DataFrame(macro_micro_rows).to_csv(os.path.join(out_dir, "macro_micro.csv"), index=False)

        # ---------------- RRF fusion (Tier-2) ----------------
        # mapping window_key -> gt_label for scoring (backend_id fixed)
        q_map = """
        SELECT window_key, gt_label_canonical
        FROM v_scenario_trace
        WHERE split_name=%s AND backend_id=%s;
        """
        df_map = fetch_df(conn, q_map, (split, backend_id))
        wk_to_gt = dict(zip(df_map["window_key"].astype(str), df_map["gt_label_canonical"]))

        rrf_rows: List[Dict[str, Any]] = []
        for label in all_labels:
            qlist = queries_cfg[label]
            n_gt = int(n_gt_by_label.get(label, 0))
            support_ok = (n_gt >= gt_support_min)

            rank_lists_for_rrf: List[List[str]] = []
            for qtext in qlist:
                df_top, rank_list = run_one_query(qtext, exclude_log_id=None, gt_present_only=False)
                rank_lists_for_rrf.append(rank_list)

            fused = rrf_fuse(rank_lists_for_rrf, k=Kmax)

            for K in top_k_list:
                top = fused[:K]
                gt_labels = [wk_to_gt.get(wk, None) for wk in top]

                # precision fairness
                k_eff = min(K, n_gt) if n_gt > 0 else K
                if k_eff != K:
                    top_eff = fused[:k_eff]
                    gt_eff = [wk_to_gt.get(wk, None) for wk in top_eff]
                    p = gt_precision_at_k(gt_eff, label)
                else:
                    p = gt_precision_at_k(gt_labels, label)

                cov = gt_coverage_at_k(gt_labels, label, n_gt) if n_gt > 0 else float("nan")

                rrf_rows.append({
                    "label": label,
                    "K": int(K),
                    "n_gt": int(n_gt),
                    "gt_precision_rrf": float(p),
                    "gt_coverage_rrf": float(cov) if n_gt > 0 else float("nan"),
                    "gt_support_ok": bool(support_ok),
                })

        pd.DataFrame(rrf_rows).to_csv(os.path.join(out_dir, "query_metrics_rrf_fused.csv"), index=False)

        # ---------------- Appendix: GT-constrained upper bound ----------------
        if experimental and gt_constrained_metrics:
            gt_con_rows: List[Dict[str, Any]] = []
            for label in all_labels:
                n_gt = int(n_gt_by_label.get(label, 0))
                support_ok = (n_gt >= gt_support_min)
                qlist = queries_cfg[label]

                # For each query: retrieve only within GT subset, compute GT precision / GT coverage.
                for qtext in qlist:
                    df_top, _ = run_one_query(qtext, exclude_log_id=None, gt_present_only=True)
                    if df_top.empty:
                        for K in top_k_list:
                            gt_con_rows.append({
                                "label": label,
                                "query_text": qtext,
                                "K": int(K),
                                "candidate_pool": "gt_only",
                                "n_gt": int(n_gt),
                                "gt_precision_at_k": 0.0,
                                "gt_coverage_at_k": 0.0 if n_gt > 0 else float("nan"),
                                "gt_support_ok": bool(support_ok),
                            })
                        continue

                    for K in top_k_list:
                        df_k = df_top.head(K)
                        gt_labels = df_k["gt_label_canonical"].tolist()

                        # fairness
                        k_eff = min(K, n_gt) if n_gt > 0 else K
                        if k_eff != K:
                            df_ke = df_top.head(k_eff)
                            p = gt_precision_at_k(df_ke["gt_label_canonical"].tolist(), label)
                        else:
                            p = gt_precision_at_k(gt_labels, label)

                        cov = gt_coverage_at_k(gt_labels, label, n_gt) if n_gt > 0 else float("nan")

                        gt_con_rows.append({
                            "label": label,
                            "query_text": qtext,
                            "K": int(K),
                            "candidate_pool": "gt_only",
                            "n_gt": int(n_gt),
                            "gt_precision_at_k": float(p),
                            "gt_coverage_at_k": float(cov),
                            "gt_support_ok": bool(support_ok),
                        })

            pd.DataFrame(gt_con_rows).to_csv(os.path.join(out_dir, "gt_only_metrics_constrained.csv"), index=False)

        # ---------------- UMAP plot + sidecar ----------------
        if out.get("make_umap", True):
            if umap is None:
                print("[WARN] umap-learn not available; skipping UMAP.")
            else:
                print("[INFO] Generating UMAP plot + sidecar CSV...")
                embed_dim = int(oll["embed_dim"])
                df_um = fetch_embeddings_for_umap(conn, split, backend_id, embed_dim)

                if not df_um.empty:
                    X = np.vstack(df_um["_emb"].to_list())
                    umcfg = out.get("umap", {})
                    reducer = umap.UMAP(
                        n_neighbors=int(umcfg.get("n_neighbors", 15)),
                        min_dist=float(umcfg.get("min_dist", 0.10)),
                        metric=str(umcfg.get("metric", "cosine")),
                        random_state=int(umcfg.get("random_state", 42)),
                    )
                    X2 = reducer.fit_transform(X)
                    df_um["umap1"] = X2[:, 0]
                    df_um["umap2"] = X2[:, 1]

                    # sidecar CSV
                    sidecar_cols = ["window_key", "log_id", "label_final", "provenance", "umap1", "umap2"]
                    df_um[sidecar_cols].to_csv(os.path.join(out_dir, "embedding_umap_points.csv"), index=False)

                    # Stable label colors (committee-friendly)
                    # You can add more labels later without changing semantics.
                    label_colors = {
                        "cut_in": "red",
                        "lead_brake": "blue",
                        "approach_stop": "green",
                        "obj_crossing": "orange",
                        "ped_crossing": "purple",
                        "left_oppo": "brown",
                        "unlabeled": "gray",
                    }
                    labels_unique = sorted(df_um["label_final"].astype(str).unique().tolist())
                    prov_mark = {"GT": "o", "LLM": "^", "unlabeled": "x"}

                    plt.figure(figsize=(12, 9))
                    ax = plt.gca()
                    ax.set_facecolor("white")

                    # Plot by (label, provenance) so both encodings work
                    for lab in labels_unique:
                        c = label_colors.get(lab, "black")

                        for prov, mk in prov_mark.items():
                            dfP = df_um[(df_um["label_final"] == lab) & (df_um["provenance"] == prov)]
                            if dfP.empty:
                                continue
                            ax.scatter(
                                dfP["umap1"].values,
                                dfP["umap2"].values,
                                s=18,
                                alpha=0.70 if prov != "unlabeled" else 0.45,
                                marker=mk,
                                color=c,
                                linewidths=0.4 if mk != "x" else 0.8,
                            )

                    ax.set_title("S7: Scenario Embedding Space (UMAP)")
                    ax.set_xlabel("UMAP-1")
                    ax.set_ylabel("UMAP-2")

                    # Legends: label colors and provenance markers (clean + non-duplicated)
                    from matplotlib.lines import Line2D

                    label_handles = []
                    for lab in labels_unique:
                        c = label_colors.get(lab, "black")
                        label_handles.append(Line2D([0], [0], marker="o", color=c, linestyle="None", label=lab, markersize=7))

                    prov_handles = [
                        Line2D([0], [0], marker="o", color="black", linestyle="None", label="GT", markersize=7),
                        Line2D([0], [0], marker="^", color="black", linestyle="None", label="LLM-only", markersize=7),
                        Line2D([0], [0], marker="x", color="black", linestyle="None", label="unlabeled", markersize=7),
                    ]

                    leg1 = ax.legend(handles=label_handles, title="Label (color)", loc="upper right")
                    ax.add_artist(leg1)
                    ax.legend(handles=prov_handles, title="Provenance (marker)", loc="lower right")

                    plt.tight_layout()
                    out_path = os.path.join(out_dir, "embedding_umap.png")
                    plt.savefig(out_path, dpi=300, facecolor="white")
                    plt.close()
                    print(f"[INFO] Saved UMAP: {out_path}")
                else:
                    print("[WARN] No embeddings to plot UMAP.")

        print(f"[INFO] Done. Outputs in: {out_dir}")

    finally:
        conn.close()
        print("[INFO] DB connection closed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/s7_eval.yaml")
    ap.add_argument("--backend_scope_all", action="store_true", help="Override: use all backends")
    ap.add_argument("--exclude_same_log", action="store_true", help="Override: exclude same log_id")
    ap.add_argument("--experimental", action="store_true", help="Appendix-only: GT-constrained + extended pseudo-label metrics")
    args = ap.parse_args()

    cfg = read_yaml(args.config)

    if args.backend_scope_all:
        cfg["eval"]["backend_scope_all"] = True
        cfg["eval"]["dedup_by_log_id"] = True  # strongly recommended

    if args.exclude_same_log:
        cfg["eval"]["exclude_same_log"] = True

    if args.experimental:
        cfg["eval"]["experimental"] = True

    run_suite(cfg)


if __name__ == "__main__":
    main()
