#!/usr/bin/env python3
"""
S7B: scenario retrieval engine (production, view-contract-safe)

Supports:
- Semantic retrieval (text query -> embedding -> nearest neighbors)
- By-example retrieval (seed window_key -> use stored embedding)
- Backend scoping:
    * single backend_id (demo-safe default)
    * prompt_type scope
    * backend_dir scope
    * all backends
- Constraints via filters_json (only applies if columns exist in v_scenario_trace)
- Optional reranking (interpretable, uses llm_confidence + score_final if present)
- Exact baseline vs ANN retrieval
- CSV/JSON reports + optional DB logging into s7_retrieval_runs/results

Contract:
- All retrieval metadata is read from v_scenario_trace (S7A view).
- Embedding table is used for seed embedding lookup (by-example) and dim sanity.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv


# ----------------------------
# Logging
# ----------------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"[WARNING] {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


# ----------------------------
# Config
# ----------------------------
@dataclasses.dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str


@dataclasses.dataclass
class S7BConfig:
    split_name: str
    results_root: str

    # Embeddings (table exists; view also contains embedding)
    embeddings_table: str          # e.g., scenario_embeddings_gemma
    embedding_model: str           # e.g., embeddinggemma:latest
    embed_dim: int                 # e.g., 768

    # Ollama
    ollama_url: str
    timeout_s: int

    # Retrieval
    top_k: int
    method: str                    # "ann" or "exact"
    store_to_db: bool
    enable_rerank: bool
    rerank_alpha_sim: float
    rerank_beta_llm_conf: float
    rerank_gamma_s1: float

    # Backend scope defaults
    backend_id: Optional[int]
    prompt_type: Optional[str]
    backend_dir: Optional[str]
    backend_scope_all: bool


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cfg(cfg_y: Dict[str, Any]) -> Tuple[DBConfig, S7BConfig]:
    db = cfg_y["db"]
    s7 = cfg_y["s7b"]

    env_file = db.get("env_file")
    if env_file:
        load_dotenv(dotenv_path=str(env_file), override=False)
        db = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "dbname": os.getenv("DB_NAME"),
        }

    required = ("host", "port", "user", "password", "dbname")
    missing = [key for key in required if db.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing database configuration fields: {', '.join(missing)}")

    db_cfg = DBConfig(
        host=db["host"],
        port=int(db["port"]),
        user=db["user"],
        password=db.get("password", ""),
        dbname=db["dbname"],
    )

    s7_cfg = S7BConfig(
        split_name=s7["split_name"],
        results_root=s7.get("results_root", "results"),
        embeddings_table=s7["embeddings_table"],
        embedding_model=s7["embedding_model"],
        embed_dim=int(s7["embed_dim"]),
        ollama_url=s7["ollama_url"].rstrip("/"),
        timeout_s=int(s7.get("timeout_s", 120)),
        top_k=int(s7.get("top_k", 10)),
        method=str(s7.get("method", "ann")).lower(),
        store_to_db=bool(s7.get("store_to_db", True)),
        enable_rerank=bool(s7.get("enable_rerank", True)),
        rerank_alpha_sim=float(s7.get("rerank_alpha_sim", 1.0)),
        rerank_beta_llm_conf=float(s7.get("rerank_beta_llm_conf", 0.1)),
        rerank_gamma_s1=float(s7.get("rerank_gamma_s1", 0.1)),
        backend_id=s7.get("backend_id", None),
        prompt_type=s7.get("prompt_type", None),
        backend_dir=s7.get("backend_dir", None),
        backend_scope_all=bool(s7.get("backend_scope_all", False)),
    )
    return db_cfg, s7_cfg


# ----------------------------
# Ollama embeddings
# ----------------------------
def ollama_embed(ollama_url: str, model: str, text: str, timeout_s: int) -> List[float]:
    url = f"{ollama_url}/api/embeddings"
    payload = {"model": model, "prompt": text}
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    j = resp.json()
    emb = j.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError(f"Ollama returned invalid embedding payload keys={list(j.keys())}")
    return [float(x) for x in emb]


# ----------------------------
# DB helpers
# ----------------------------
def connect(db: DBConfig):
    return psycopg2.connect(
        host=db.host,
        port=db.port,
        user=db.user,
        password=db.password,
        dbname=db.dbname,
    )


def fetch_columns(conn, schema: str, table_or_view: str) -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema=%s AND table_name=%s
    ORDER BY ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table_or_view))
        return [r[0] for r in cur.fetchall()]


def table_exists(conn, name: str) -> bool:
    q = """
    SELECT EXISTS (
      SELECT 1 FROM information_schema.tables
      WHERE table_schema='public' AND table_name=%s
    );
    """
    with conn.cursor() as cur:
        cur.execute(q, (name,))
        return bool(cur.fetchone()[0])


def view_exists(conn, name: str) -> bool:
    q = """
    SELECT EXISTS (
      SELECT 1 FROM information_schema.views
      WHERE table_schema='public' AND table_name=%s
    );
    """
    with conn.cursor() as cur:
        cur.execute(q, (name,))
        return bool(cur.fetchone()[0])


def ensure_prereqs(conn, cfg: S7BConfig) -> None:
    if not view_exists(conn, "v_scenario_trace"):
        raise RuntimeError("Missing view: v_scenario_trace (run S7A first)")

    if not table_exists(conn, cfg.embeddings_table):
        raise RuntimeError(f"Missing embeddings table: {cfg.embeddings_table}")

    if cfg.store_to_db:
        if not table_exists(conn, "s7_retrieval_runs"):
            raise RuntimeError("Missing table: s7_retrieval_runs (run S7A first)")
        if not table_exists(conn, "s7_retrieval_results"):
            raise RuntimeError("Missing table: s7_retrieval_results (run S7A first)")

    # Dim sanity from embeddings table
    q = f"""
    SELECT COUNT(*) AS n, MIN(embed_dim) AS min_dim, MAX(embed_dim) AS max_dim
    FROM {cfg.embeddings_table}
    WHERE split_name=%s AND embedding_model=%s;
    """
    with conn.cursor() as cur:
        cur.execute(q, (cfg.split_name, cfg.embedding_model))
        n, min_dim, max_dim = cur.fetchone()

    if n == 0:
        raise RuntimeError(
            f"No embeddings found in {cfg.embeddings_table} for split={cfg.split_name} model={cfg.embedding_model}"
        )
    if min_dim != max_dim or int(min_dim) != int(cfg.embed_dim):
        raise RuntimeError(
            f"Embedding dim mismatch: table has [{min_dim},{max_dim}] but cfg.embed_dim={cfg.embed_dim}"
        )

    # View must expose embedding + key columns
    cols = set(fetch_columns(conn, "public", "v_scenario_trace"))
    required = {"split_name", "backend_id", "window_key", "embedding", "embedding_model", "embed_dim"}
    missing = sorted(list(required - cols))
    if missing:
        raise RuntimeError(f"v_scenario_trace is missing required columns: {missing}")


def parse_filters_json(filters_json: Optional[str]) -> Dict[str, Any]:
    if not filters_json:
        return {}
    try:
        return json.loads(filters_json)
    except Exception as e:
        raise ValueError(f"Invalid --filters_json: {e}")


def build_backend_where(cfg: S7BConfig, view_cols: set) -> Tuple[str, List[Any], Dict[str, Any]]:
    """
    Backend scoping is done via v_scenario_trace columns (backend_id, prompt_type, backend_dir).
    Demo-safety default: backend_id=1 if user didn't specify any scope and backend_scope_all=False.
    Returns:
      where_sql (prefixed with AND if non-empty)
      params
      scope_meta
    """
    if cfg.backend_scope_all:
        return "", [], {"backend_scope": "all", "backend_ids": None}

    clauses = []
    params: List[Any] = []
    scope = {"backend_scope": "scoped", "backend_ids": None}

    if cfg.backend_id is not None:
        clauses.append("t.backend_id = %s")
        params.append(int(cfg.backend_id))
        scope["backend_scope"] = "single"
        scope["backend_ids"] = [int(cfg.backend_id)]

    if cfg.prompt_type is not None:
        if "prompt_type" in view_cols:
            clauses.append("t.prompt_type = %s")
            params.append(str(cfg.prompt_type))
            scope["backend_scope"] = "prompt_type"
        else:
            log_warn("v_scenario_trace has no prompt_type; ignoring prompt_type scope.")

    if cfg.backend_dir is not None:
        if "backend_dir" in view_cols:
            clauses.append("t.backend_dir = %s")
            params.append(str(cfg.backend_dir))
            scope["backend_scope"] = "backend_dir"
        else:
            log_warn("v_scenario_trace has no backend_dir; ignoring backend_dir scope.")

    if not clauses:
        # demo safety default
        clauses.append("t.backend_id = %s")
        params.append(1)
        scope["backend_scope"] = "single_default"
        scope["backend_ids"] = [1]
        log_warn("No backend scope specified; defaulting to backend_id=1 for demo safety.")

    return " AND " + " AND ".join(clauses), params, scope


def build_filters_where(filters: Dict[str, Any], cols: set) -> Tuple[str, List[Any]]:
    """
    Applies filters only if column exists in v_scenario_trace.
    Uses the canonical naming you already established in the view.
    """
    clauses = []
    params: List[Any] = []

    def add_eq(col: str, val: Any):
        if col in cols and val is not None:
            clauses.append(f"t.{col} = %s")
            params.append(val)

    def add_ge(col: str, val: Any):
        if col in cols and val is not None:
            clauses.append(f"t.{col} >= %s")
            params.append(val)

    def add_le(col: str, val: Any):
        if col in cols and val is not None:
            clauses.append(f"t.{col} <= %s")
            params.append(val)

    # LLM filters (view uses llm_* fields)
    add_eq("llm_label_canonical", filters.get("llm_label"))
    add_eq("llm_label_raw", filters.get("llm_label_raw"))
    add_ge("llm_confidence", filters.get("llm_conf_min"))
    add_le("llm_confidence", filters.get("llm_conf_max"))

    # GT filters
    add_eq("gt_label_canonical", filters.get("gt_label"))
    add_eq("gt_label_raw", filters.get("gt_label_raw"))

    if "gt_present" in filters and "gt_label_canonical" in cols:
        if filters["gt_present"] is True:
            clauses.append("t.gt_label_canonical IS NOT NULL")
        elif filters["gt_present"] is False:
            clauses.append("t.gt_label_canonical IS NULL")

    # Evidence / map / hints
    add_eq("road_type_hint", filters.get("road_type_hint"))
    add_eq("near_crosswalk", filters.get("near_crosswalk"))
    add_eq("near_stopline", filters.get("near_stopline"))
    add_eq("has_close_actor", filters.get("has_close_actor"))
    add_eq("primary_side", filters.get("primary_side"))

    # S1 verifier score
    add_ge("score_final", filters.get("s1_score_min"))
    add_le("score_final", filters.get("s1_score_max"))

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def set_exact_mode(conn) -> None:
    """
    Encourage brute-force scan behavior for baseline comparisons.
    """
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET enable_seqscan = on;")
    conn.commit()


# ----------------------------
# By-example: seed embedding
# ----------------------------
def get_seed_embedding(conn, cfg: S7BConfig, backend_id: int, window_key: str) -> List[float]:
    q = f"""
    SELECT embedding
    FROM {cfg.embeddings_table}
    WHERE split_name=%s AND backend_id=%s AND window_key=%s AND embedding_model=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(q, (cfg.split_name, backend_id, window_key, cfg.embedding_model))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"No seed embedding found for backend_id={backend_id}, window_key={window_key}")
    emb_str = row[0]
    # vector often comes back as string like '[...]'
    if isinstance(emb_str, str):
        nums = emb_str.strip("[]").split(",")
        return [float(x) for x in nums if x.strip()]
    return list(emb_str)


# ----------------------------
# Retrieval core
# ----------------------------
def retrieve(
    conn,
    cfg: S7BConfig,
    query_vec: List[float],
    query_meta: Dict[str, Any],
    filters: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    view_cols = fetch_columns(conn, "public", "v_scenario_trace")
    cols_set = set(view_cols)

    backend_where, backend_params, scope_meta = build_backend_where(cfg, cols_set)
    filters_where, filters_params = build_filters_where(filters, cols_set)

    # fields for display (only if exist)
    display_fields = [
        "log_id", "t_start", "t_end", "dur_s",
        "score_final",
        "road_type_hint", "near_crosswalk", "near_stopline", "has_close_actor",
        "gat_top3_json",
        "gt_label_canonical", "gt_label_raw", "gt_has_guest", "gt_guest_track_uuid",
        "prompt_type", "backend_dir", "model_name", "provider_hint",
        "llm_label_raw", "llm_label_canonical", "llm_confidence",
        "llm_primary_actor_raw", "llm_primary_track_uuid",
        "llm_actor_matrix_json", "llm_final_rationale",
    ]
    select_parts = ["t.split_name", "t.backend_id", "t.window_key", "t.embedding_model", "t.embed_dim"]
    for f in display_fields:
        if f in cols_set:
            select_parts.append(f"t.{f}")

    # Similarity: cosine distance (<=>) (smaller better). We also report cosine_sim = 1 - dist.
    # We pass query_vec as a pgvector literal string: '[...]'
    qvec_lit = "[" + ",".join(f"{x:.8f}" for x in query_vec) + "]"

    sql = f"""
    SELECT
      {", ".join(select_parts)},
      (t.embedding <=> %s::vector) AS cosine_dist,
      (1 - (t.embedding <=> %s::vector)) AS cosine_sim
    FROM v_scenario_trace t
    WHERE t.split_name = %s
    AND t.embedding_model = %s
    {backend_where}
    {filters_where}
    ORDER BY cosine_dist ASC
    LIMIT %s;
    """

    params: List[Any] = []
    params.append(qvec_lit)
    params.append(qvec_lit)
    params.append(cfg.split_name)
    params.append(cfg.embedding_model)
    params += backend_params
    params += filters_params
    params.append(cfg.top_k)

    t0 = time.time()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    latency_ms = int((time.time() - t0) * 1000)

    # Rerank (interpretable)
    # Only uses signals if present.
    def safe_float(v) -> float:
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    out: List[Dict[str, Any]] = []
    for r in rows:
        sim = safe_float(r.get("cosine_sim"))
        llm_conf = safe_float(r.get("llm_confidence")) if "llm_confidence" in cols_set else 0.0
        s1 = safe_float(r.get("score_final")) if "score_final" in cols_set else 0.0

        rerank = sim
        breakdown = {"cosine_sim": sim, "llm_confidence": llm_conf, "score_final": s1}

        if cfg.enable_rerank:
            rerank = (
                cfg.rerank_alpha_sim * sim
                + cfg.rerank_beta_llm_conf * llm_conf
                + cfg.rerank_gamma_s1 * s1
            )
            breakdown["rerank"] = {
                "alpha_sim": cfg.rerank_alpha_sim,
                "beta_llm_conf": cfg.rerank_beta_llm_conf,
                "gamma_s1": cfg.rerank_gamma_s1,
            }

        rr = dict(r)
        rr["rerank_score"] = rerank
        rr["score_breakdown_json"] = breakdown
        rr["_meta"] = {
            "latency_ms": latency_ms,
            "method": cfg.method,
            "embedding_model": cfg.embedding_model,
            "backend_scope": scope_meta.get("backend_scope"),
        }
        out.append(rr)

    if cfg.enable_rerank:
        out.sort(key=lambda x: safe_float(x.get("rerank_score")), reverse=True)
    else:
        out.sort(key=lambda x: safe_float(x.get("cosine_sim")), reverse=True)

    for i, r in enumerate(out, start=1):
        r["rank"] = i

    scope_meta["latency_ms"] = latency_ms
    scope_meta["n_results"] = len(out)
    return out, scope_meta


# ----------------------------
# Run logging (DB + files)
# ----------------------------
def discover_backend_ids_for_scope(conn, cfg: S7BConfig) -> Optional[List[int]]:
    """
    For reproducibility: record which backends were in scope.
    If backend_scope_all=True -> return all backend_ids present for split+embedding_model.
    Else if prompt_type/backend_dir scope -> return ids matching.
    Else -> backend_id (or default 1).
    """
    cols = set(fetch_columns(conn, "public", "v_scenario_trace"))
    where = ["split_name=%s", "embedding_model=%s"]
    params: List[Any] = [cfg.split_name, cfg.embedding_model]

    if cfg.backend_scope_all:
        pass
    else:
        if cfg.backend_id is not None:
            where.append("backend_id=%s")
            params.append(int(cfg.backend_id))
        else:
            # demo default is backend_id=1
            where.append("backend_id=%s")
            params.append(1)

        if cfg.prompt_type is not None and "prompt_type" in cols:
            where.append("prompt_type=%s")
            params.append(cfg.prompt_type)

        if cfg.backend_dir is not None and "backend_dir" in cols:
            where.append("backend_dir=%s")
            params.append(cfg.backend_dir)

    q = f"SELECT DISTINCT backend_id FROM v_scenario_trace WHERE {' AND '.join(where)} ORDER BY backend_id;"
    with conn.cursor() as cur:
        cur.execute(q, params)
        ids = [int(r[0]) for r in cur.fetchall()]
    return ids if ids else None


def insert_run(conn, cfg: S7BConfig, query_meta: Dict[str, Any], filters: Dict[str, Any]) -> int:
    """
    Insert into s7_retrieval_runs using the released table schema.
    """
    backend_ids = discover_backend_ids_for_scope(conn, cfg)

    backend_scope = "all" if cfg.backend_scope_all else (
        "single" if cfg.backend_id is not None else "single_default"
    )
    if cfg.prompt_type is not None:
        backend_scope = "prompt_type"
    if cfg.backend_dir is not None:
        backend_scope = "backend_dir"
    if cfg.backend_scope_all:
        backend_scope = "all"

    q = """
    INSERT INTO s7_retrieval_runs (
      split_name,
      query_type,
      query_text,
      seed_window_key,
      backend_id,
      backend_scope,
      backend_ids_json,
      embedding_table,
      embedding_model,
      filters_json,
      top_k,
      notes
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s)
    RETURNING run_id;
    """
    with conn.cursor() as cur:
        cur.execute(
            q,
            (
                cfg.split_name,
                query_meta.get("query_type"),
                query_meta.get("query_text"),
                query_meta.get("seed_window_key"),
                cfg.backend_id if cfg.backend_id is not None else (None if cfg.backend_scope_all else 1),
                backend_scope,
                json.dumps(backend_ids),
                cfg.embeddings_table,
                cfg.embedding_model,
                json.dumps(filters),
                cfg.top_k,
                None,
            ),
        )
        run_id = int(cur.fetchone()[0])
    conn.commit()
    return run_id


def insert_results(conn, run_id: int, results: List[Dict[str, Any]]) -> None:
    q = """
    INSERT INTO s7_retrieval_results (
      run_id, rank, window_key, backend_id, similarity_score, rerank_score, score_breakdown_json, notes
    )
    VALUES %s
    ON CONFLICT DO NOTHING;
    """
    rows = []
    for r in results:
        rows.append(
            (
                run_id,
                int(r["rank"]),
                r["window_key"],
                int(r.get("backend_id")) if r.get("backend_id") is not None else None,
                float(r.get("cosine_sim", 0.0)),
                float(r.get("rerank_score", 0.0)) if r.get("rerank_score") is not None else None,
                json.dumps(r.get("score_breakdown_json", {})),
                None,
            )
        )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, q, rows, page_size=2000)
    conn.commit()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    flat_rows = []
    for r in rows:
        rr = dict(r)
        meta = rr.pop("_meta", {})
        for k, v in meta.items():
            rr[f"meta_{k}"] = v
        # keep JSON fields compact in CSV
        for jcol in ["gat_top3_json", "llm_actor_matrix_json", "score_breakdown_json"]:
            if jcol in rr and rr[jcol] is not None and not isinstance(rr[jcol], str):
                rr[jcol] = json.dumps(rr[jcol], ensure_ascii=False)
        flat_rows.append(rr)

    keys = sorted({k for r in flat_rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to s7_db.yaml (S7B section)")
    ap.add_argument("--query_text", default=None, help="Free text query for semantic retrieval")
    ap.add_argument("--seed_window_key", default=None, help="Seed window_key for by-example retrieval")
    ap.add_argument("--backend_id", type=int, default=None, help="Override backend_id scope (recommended for demos)")
    ap.add_argument("--prompt_type", default=None, help="Scope by prompt_type (analysis mode)")
    ap.add_argument("--backend_dir", default=None, help="Scope by backend_dir (analysis mode)")
    ap.add_argument("--backend_scope_all", action="store_true", help="Search across all backends (careful)")
    ap.add_argument("--filters_json", default=None, help="JSON string of filters (LLM/GT/evidence/S1)")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--method", choices=["ann", "exact"], default=None, help="ANN (HNSW) or exact baseline")
    ap.add_argument("--no_store_to_db", action="store_true", help="Do not write run/results to DB tables")
    ap.add_argument("--no_rerank", action="store_true", help="Disable reranking; pure similarity order")
    args = ap.parse_args()

    db_cfg, cfg = build_cfg(load_yaml(args.config))

    # overrides
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.method is not None:
        cfg.method = str(args.method)
    if args.backend_id is not None:
        cfg.backend_id = int(args.backend_id)
    if args.prompt_type is not None:
        cfg.prompt_type = args.prompt_type
    if args.backend_dir is not None:
        cfg.backend_dir = args.backend_dir
    if args.backend_scope_all:
        cfg.backend_scope_all = True
    if args.no_store_to_db:
        cfg.store_to_db = False
    if args.no_rerank:
        cfg.enable_rerank = False

    filters = parse_filters_json(args.filters_json)

    # query mode
    if args.query_text and args.seed_window_key:
        raise ValueError("Provide either --query_text OR --seed_window_key, not both.")
    if not args.query_text and not args.seed_window_key:
        raise ValueError("Provide one of: --query_text or --seed_window_key")

    query_type = "text" if args.query_text else "by_example"
    query_text = args.query_text
    seed_window_key = args.seed_window_key

    log_info("Connecting to DB...")
    conn = connect(db_cfg)

    try:
        log_info("S7B scenario retrieval")
        log_info(f"  split_name       : {cfg.split_name}")
        log_info(f"  embeddings_table : {cfg.embeddings_table}")
        log_info(f"  embedding_model  : {cfg.embedding_model}")
        log_info(f"  embed_dim        : {cfg.embed_dim}")
        log_info(f"  method           : {cfg.method}")
        log_info(f"  top_k            : {cfg.top_k}")
        log_info(f"  store_to_db      : {cfg.store_to_db}")
        log_info(f"  enable_rerank    : {cfg.enable_rerank}")
        log_info(f"  backend_scope_all: {cfg.backend_scope_all}")
        log_info(f"  backend_id       : {cfg.backend_id}")
        log_info(f"  prompt_type      : {cfg.prompt_type}")
        log_info(f"  backend_dir      : {cfg.backend_dir}")
        log_info(f"  filters          : {filters}")

        ensure_prereqs(conn, cfg)

        if cfg.method == "exact":
            log_info("Exact baseline mode enabled (encouraging seqscan).")
            set_exact_mode(conn)

        query_meta: Dict[str, Any] = {
            "query_type": query_type,
            "query_text": query_text,
            "seed_window_key": seed_window_key,
        }

        # Build query embedding
        if query_type == "text":
            log_info("Embedding query text via Ollama...")
            qvec = ollama_embed(cfg.ollama_url, cfg.embedding_model, query_text, cfg.timeout_s)
            if len(qvec) != cfg.embed_dim:
                raise RuntimeError(f"Query embedding dim={len(qvec)} != cfg.embed_dim={cfg.embed_dim}")
        else:
            seed_backend = cfg.backend_id if cfg.backend_id is not None else 1
            log_info(f"Fetching seed embedding from DB (backend_id={seed_backend})...")
            qvec = get_seed_embedding(conn, cfg, seed_backend, seed_window_key)
            if len(qvec) != cfg.embed_dim:
                raise RuntimeError(f"Seed embedding dim={len(qvec)} != cfg.embed_dim={cfg.embed_dim}")

        results, scope_meta = retrieve(conn, cfg, qvec, query_meta, filters)

        # Persist run + results
        run_id = None
        if cfg.store_to_db:
            run_id = insert_run(conn, cfg, query_meta, filters)
            insert_results(conn, run_id, results)

        # Write reports
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"run_{run_id}" if run_id is not None else f"run_local_{ts}"
        out_dir = os.path.join(cfg.results_root, "s7", "retrieval_runs")
        ensure_dir(out_dir)

        out_json = os.path.join(out_dir, f"{run_tag}.json")
        out_csv = os.path.join(out_dir, f"{run_tag}.csv")

        payload = {
            "run_id": run_id,
            "timestamp": ts,
            "config": dataclasses.asdict(cfg),
            "query": query_meta,
            "filters": filters,
            "scope_meta": scope_meta,
            "n_results": len(results),
            "results": results,
        }
        write_json(out_json, payload)
        write_csv(out_csv, results)

        # Terminal preview
        log_info(f"Run tag: {run_tag}")
        log_info(f"JSON report: {out_json}")
        log_info(f"CSV report : {out_csv}")
        log_info("Top results preview:")
        for r in results[: min(10, len(results))]:
            wk = r.get("window_key")
            sim = float(r.get("cosine_sim", 0.0))
            rr = float(r.get("rerank_score", 0.0)) if r.get("rerank_score") is not None else 0.0
            lab = r.get("llm_label_canonical") or r.get("llm_label_raw")
            gt = r.get("gt_label_canonical")
            conf = r.get("llm_confidence")
            log_info(f"  rank={r['rank']:>2} sim={sim:.4f} rerank={rr:.4f} llm={lab} gt={gt} conf={conf} wk={wk}")

        log_info("Done.")
    finally:
        conn.close()
        log_info("DB connection closed.")


if __name__ == "__main__":
    main()
