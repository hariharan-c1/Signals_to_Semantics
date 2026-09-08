#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6H - Build scenario embeddings from llm_predictions and store into a fixed-dim pgvector table.

Key idea:
- For each LLM output row (backend_id, window_key), we build a canonical text payload from:
  scenario_label_*, confidence_score, primary_actor_*, final_rationale, actor_matrix_json / parsed_result_json.
- We embed that payload with a chosen embedding model (Ollama embeddings endpoint).
- We store into a table with FIXED vector dimension: vector(embed_dim).
  => This enforces consistent embedding lengths and makes indexing/retrieval sane.

Reads:
  - YAML config (s6h section + runtime settings)
  - .env with DB credentials
  - llm_predictions and llm_backends tables in Postgres

Writes:
  - scenario_embeddings_* table (name configurable), e.g. scenario_embeddings_gemma
    PK: (backend_id, window_key, embedding_model)
    Upsert: if text changes, we overwrite embedding + update timestamps.

Run:
  python scripts/s6/s6h_build_scenario_embeddings.py --config configs/s6_db.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# requests is the simplest reliable HTTP client for Ollama
import requests


# ---------------- logging ---------------- #

def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s] %(message)s")


# ---------------- config ---------------- #

def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# ---------------- helpers ---------------- #

def safe_json_dumps(x: Any) -> str:
    # ensure no NaN/Infinity leaks into JSON
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize_label(raw: Optional[str], canon: Optional[str]) -> str:
    # prefer canonical if present; else raw; else empty
    return (canon or raw or "").strip()


def build_embedding_text(row: Dict[str, Any], include_parsed: bool, include_actor_matrix: bool) -> str:
    """
    Build the text that will be embedded.
    Keep it deterministic (stable ordering) so re-runs are reproducible.
    """
    label = normalize_label(row.get("scenario_label_raw"), row.get("scenario_label_canonical"))
    conf = row.get("confidence_score", None)

    primary_actor_raw = row.get("primary_actor_raw", None)
    primary_track_uuid = row.get("primary_track_uuid", None)

    final_rationale = (row.get("final_rationale") or "").strip()

    parts: List[str] = []
    parts.append(f"scenario_label={label}")
    if conf is not None:
        parts.append(f"confidence_score={float(conf):.6f}")
    else:
        parts.append("confidence_score=")

    parts.append(f"primary_actor_raw={primary_actor_raw if primary_actor_raw is not None else ''}")
    parts.append(f"primary_track_uuid={str(primary_track_uuid) if primary_track_uuid is not None else ''}")

    if include_actor_matrix:
        am = row.get("actor_matrix_json", None)
        if am is not None:
            parts.append(f"actor_matrix_json={safe_json_dumps(am)}")
        else:
            parts.append("actor_matrix_json=")

    if include_parsed:
        pr = row.get("parsed_result_json", None)
        if pr is not None:
            parts.append(f"parsed_result_json={safe_json_dumps(pr)}")
        else:
            parts.append("parsed_result_json=")

    parts.append(f"final_rationale={final_rationale}")

    # Important: keep delimiter consistent
    return "\n".join(parts).strip() + "\n"


def to_pgvector_literal(vec: List[float], decimals: int = 8) -> str:
    # pgvector accepts: '[0.1,0.2,...]'
    return "[" + ",".join(f"{float(v):.{decimals}f}" for v in vec) + "]"


def validate_embed_dim(embed_dim: int) -> None:
    if not isinstance(embed_dim, int) or embed_dim <= 0 or embed_dim > 32768:
        raise ValueError(f"embed_dim looks invalid: {embed_dim}")


# ---------------- DB DDL ---------------- #

def ensure_table(conn, table_name: str, embed_dim: int) -> None:
    """
    Create a fixed-dimension embeddings table.
    We enforce vector(embed_dim) to prevent mixing dimensions.
    """
    validate_embed_dim(embed_dim)

    # NOTE: table_name is controlled via YAML; still validate mild constraints.
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe table_name: {table_name}")

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      split_name      TEXT NOT NULL,
      backend_id      BIGINT NOT NULL,
      window_key      TEXT NOT NULL,
      embedding_model TEXT NOT NULL,
      embed_dim       INT NOT NULL,
      embedding       vector({embed_dim}) NOT NULL,
      text_used       TEXT NOT NULL,
      source          TEXT NOT NULL DEFAULT 'llm_predictions',
      created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (backend_id, window_key, embedding_model),
      CONSTRAINT fk_{table_name}_backend
        FOREIGN KEY (backend_id) REFERENCES llm_backends(backend_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_{table_name}_split_window
      ON {table_name} (split_name, window_key);

    CREATE INDEX IF NOT EXISTS idx_{table_name}_split_backend
      ON {table_name} (split_name, backend_id);

    CREATE INDEX IF NOT EXISTS idx_{table_name}_model
      ON {table_name} (embedding_model);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def fetch_llm_rows(conn, split_name: str) -> List[Dict[str, Any]]:
    """
    Pull everything needed from llm_predictions.
    """
    sql = """
    SELECT
      split_name,
      backend_id,
      window_key,
      scenario_label_raw,
      scenario_label_canonical,
      confidence_score,
      primary_actor_raw,
      primary_track_uuid,
      actor_matrix_json,
      final_rationale,
      parsed_result_json
    FROM llm_predictions
    WHERE split_name = %s
    ORDER BY backend_id, window_key
    """
    with conn.cursor() as cur:
        cur.execute(sql, (split_name,))
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            rows.append({cols[i]: r[i] for i in range(len(cols))})
    return rows


def fetch_existing_keys(conn, table_name: str, split_name: str, embedding_model: str) -> set:
    """
    Return existing (backend_id, window_key) pairs for this split+embedding_model.
    Used to skip embedding calls when already present.
    """
    sql = f"""
    SELECT backend_id, window_key
    FROM {table_name}
    WHERE split_name=%s AND embedding_model=%s
    """
    s = set()
    with conn.cursor() as cur:
        cur.execute(sql, (split_name, embedding_model))
        for backend_id, window_key in cur.fetchall():
            s.add((int(backend_id), str(window_key)))
    return s


def upsert_embeddings(
    conn,
    table_name: str,
    records: List[Tuple],
    batch_size: int,
) -> None:
    """
    Upsert embeddings: if same PK exists, overwrite embedding + text_used + updated_at.
    """
    if not records:
        return

    sql = f"""
    INSERT INTO {table_name} (
      split_name, backend_id, window_key, embedding_model, embed_dim, embedding, text_used
    )
    VALUES %s
    ON CONFLICT (backend_id, window_key, embedding_model)
    DO UPDATE SET
      embed_dim   = EXCLUDED.embed_dim,
      embedding   = EXCLUDED.embedding,
      text_used   = EXCLUDED.text_used,
      updated_at  = NOW();
    """

    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(
                cur,
                sql,
                chunk,
                template="(%s,%s,%s,%s,%s,%s::vector,%s)",
                page_size=min(len(chunk), 1000),
            )
    conn.commit()


# ---------------- Ollama embeddings ---------------- #

def ollama_embed(
    ollama_url: str,
    model: str,
    text: str,
    timeout_s: int,
) -> List[float]:
    """
    Calls Ollama embeddings endpoint.
    Compatible with: POST {ollama_url}/api/embeddings
      payload: {"model": "...", "prompt": "..."}
      response: {"embedding": [..]}
    """
    url = ollama_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if "embedding" not in data or not isinstance(data["embedding"], list):
        raise RuntimeError(f"Ollama response missing embedding list. Keys={list(data.keys())}")
    return data["embedding"]


# ---------------- main ingest ---------------- #

def run_s6h(conn, cfg: Dict[str, Any]) -> None:
    s6h = cfg["s6h"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6h["split_name"])
    ollama_url = str(s6h.get("ollama_url", "http://127.0.0.1:11434"))
    embed_dim = int(s6h["embed_dim"])
    embedding_models = list(s6h["embedding_models"])
    table_name = str(s6h["table_name"])
    timeout_s = int(s6h.get("timeout_s", 120))

    include_parsed = bool(s6h.get("include_parsed_result_json", True))
    include_actor_matrix = bool(s6h.get("include_actor_matrix_json", True))

    batch_size = int(runtime.get("batch_size", 2000))

    validate_embed_dim(embed_dim)

    logging.info("S6H build scenario embeddings (fixed-dim table)")
    logging.info(f"  split_name      : {split_name}")
    logging.info(f"  table_name      : {table_name}")
    logging.info(f"  ollama_url      : {ollama_url}")
    logging.info(f"  embed_dim       : {embed_dim}")
    logging.info(f"  embedding_models: {embedding_models}")
    logging.info(f"  timeout_s       : {timeout_s}")
    logging.info(f"  batch_size      : {batch_size}")
    logging.info(f"  include_actor_matrix_json : {include_actor_matrix}")
    logging.info(f"  include_parsed_result_json: {include_parsed}")

    # Health check
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s", (split_name,))
        (nwin,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM llm_predictions WHERE split_name=%s", (split_name,))
        (nllm,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {int(nwin)}")
    logging.info(f"DB check: llm_predictions  rows for split = {int(nllm)}")

    if nllm == 0:
        logging.warning("No llm_predictions rows found; nothing to embed.")
        return

    # Ensure embeddings table exists (fixed dim)
    ensure_table(conn, table_name, embed_dim)

    # Load all LLM rows once
    llm_rows = fetch_llm_rows(conn, split_name)
    logging.info(f"Loaded llm_predictions rows = {len(llm_rows)}")

    for model in embedding_models:
        model = str(model)

        # Existing key set to skip redundant calls
        try:
            existing = fetch_existing_keys(conn, table_name, split_name, model)
        except Exception:
            existing = set()

        logging.info(f"[{model}] existing embeddings rows = {len(existing)}")

        attempted = 0
        skipped_existing = 0
        embedded_ok = 0
        embedded_fail = 0
        dim_mismatch = 0

        # Accumulate DB records in memory, flush periodically
        db_records: List[Tuple] = []
        last_flush = time.time()

        for idx, row in enumerate(llm_rows, start=1):
            backend_id = int(row["backend_id"])
            window_key = str(row["window_key"])

            attempted += 1

            if (backend_id, window_key) in existing:
                skipped_existing += 1
                continue

            text = build_embedding_text(row, include_parsed=include_parsed, include_actor_matrix=include_actor_matrix)

            try:
                emb = ollama_embed(ollama_url, model, text, timeout_s=timeout_s)
                if len(emb) != embed_dim:
                    dim_mismatch += 1
                    raise ValueError(f"Embedding dim mismatch: got {len(emb)} expected {embed_dim}")

                emb_str = to_pgvector_literal(emb, decimals=8)
                db_records.append((
                    split_name, backend_id, window_key, model, embed_dim, emb_str, text
                ))
                embedded_ok += 1

            except Exception as e:
                embedded_fail += 1
                # keep going; we want maximum salvage
                logging.warning(f"[{model}] embed failed idx={idx}/{len(llm_rows)} backend_id={backend_id} window_key={window_key} err={type(e).__name__}: {e}")

            # periodic flush to DB (prevents huge RAM & keeps progress)
            if len(db_records) >= 200 or (time.time() - last_flush) > 30:
                upsert_embeddings(conn, table_name, db_records, batch_size=batch_size)
                db_records = []
                last_flush = time.time()

            if idx % 100 == 0:
                logging.info(f"[{model}] progress {idx}/{len(llm_rows)} ok={embedded_ok} fail={embedded_fail} skipped_existing={skipped_existing}")

        # final flush
        if db_records:
            upsert_embeddings(conn, table_name, db_records, batch_size=batch_size)

        # report
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE split_name=%s AND embedding_model=%s",
                (split_name, model),
            )
            (n_in_db,) = cur.fetchone()

        logging.info(f"[{model}] summary:")
        logging.info(f"  attempted         : {attempted}")
        logging.info(f"  skipped_existing  : {skipped_existing}")
        logging.info(f"  embedded_ok       : {embedded_ok}")
        logging.info(f"  embedded_fail     : {embedded_fail}")
        logging.info(f"  dim_mismatch      : {dim_mismatch}")
        logging.info(f"  rows_in_table_now : {int(n_in_db)}")

    logging.info("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/s6_db.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    env_file = Path(cfg["db"]["env_file"])
    db_params = load_db_env(env_file)

    logging.info("Connecting to DB...")
    conn = psycopg2.connect(**db_params)
    conn.autocommit = False

    try:
        # no-op: tables created inside run_s6h (because we need embed_dim & table_name)
        run_s6h(conn, cfg)
    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
