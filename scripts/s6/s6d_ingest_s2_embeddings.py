#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6D (fixed) - Ingest S2 actor embeddings with canonical window_key reconstruction.

Core fix:
  - NEVER trust window_key from S2 parquet for joins (float precision mismatch).
  - Always reconstruct canonical window_key from (log_id, window_t_start, window_t_end)
    using the same t_precision rule as S6A/S6B, with fixed-decimal formatting.

Reads:
  - YAML config (paths, split name, runtime settings)
  - .env with DB credentials
  - embeddings_train.parquet and embeddings_val.parquet from embeddings_dir

Writes:
  - s2_actor_embeddings table
      split_name, embed_split, window_key (canonical), track_uuid
      plus log_id, t_start, t_end, source_window_key, category, embedding

Idempotence:
  - UNIQUE (split_name, embed_split, window_key, track_uuid)
  - INSERT ... ON CONFLICT DO NOTHING

Sanity checks:
  - row counts per subset
  - NaN handling policy
  - overlap stats with scenario_actors (should be >0 typically if actor IDs align)

Run:
  python scripts/s6/s6d_ingest_s2_embeddings.py --config configs/s6_db.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values


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


# ---------------- db ddl ---------------- #

DDL_S2_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS s2_actor_embeddings (
  split_name        TEXT NOT NULL,
  embed_split       TEXT NOT NULL, -- train / val

  -- canonical identity (join-safe)
  window_key        TEXT NOT NULL,
  track_uuid        UUID NOT NULL,

  -- debug + provenance
  log_id            UUID,
  t_start           DOUBLE PRECISION,
  t_end             DOUBLE PRECISION,
  source_window_key TEXT,
  category          TEXT,

  embedding_dim     INT NOT NULL,
  embedding         vector(128) NOT NULL,

  source_path       TEXT NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotence key:
CREATE UNIQUE INDEX IF NOT EXISTS uq_s2_actor_embeddings
  ON s2_actor_embeddings (split_name, embed_split, window_key, track_uuid);

CREATE INDEX IF NOT EXISTS idx_s2_embed_window
  ON s2_actor_embeddings (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_s2_embed_track
  ON s2_actor_embeddings (split_name, track_uuid);

CREATE INDEX IF NOT EXISTS idx_s2_embed_log
  ON s2_actor_embeddings (split_name, log_id);
"""


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_S2_EMBEDDINGS)
    conn.commit()


# ---------------- helpers ---------------- #

def parse_nan_policy(x: str) -> str:
    x = str(x).strip().lower()
    if x not in {"error", "drop", "zero"}:
        raise ValueError("nan_policy must be one of: error, drop, zero")
    return x


def embedding_cols(dim: int) -> List[str]:
    return [f"emb_{i:03d}" for i in range(dim)]


def canonical_window_key(log_id: Any, t_start: Any, t_end: Any, t_precision: int) -> Tuple[str, float, float]:
    """
    Construct canonical window_key with:
      - rounding to t_precision
      - fixed-decimal formatting (CRITICAL for stable joins)
    Returns: (window_key, t_start_rounded, t_end_rounded)
    """
    lid = str(log_id)
    ts = round(float(t_start), t_precision)
    te = round(float(t_end), t_precision)

    ts_s = format(ts, f".{t_precision}f")
    te_s = format(te, f".{t_precision}f")

    return f"{lid}|{ts_s}|{te_s}", ts, te


# ---------------- ingest ---------------- #

INSERT_SQL = """
INSERT INTO s2_actor_embeddings (
  split_name, embed_split,
  window_key, track_uuid,
  log_id, t_start, t_end, source_window_key, category,
  embedding_dim, embedding, source_path
)
VALUES %s
ON CONFLICT (split_name, embed_split, window_key, track_uuid) DO NOTHING;
"""


def build_records_from_parquet(
    parquet_path: Path,
    split_name: str,
    embed_split: str,
    dim: int,
    t_precision: int,
    nan_policy: str,
) -> Tuple[List[Tuple], Dict[str, int]]:
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded parquet: {parquet_path.name} rows={len(df)}, cols={len(df.columns)}")

    # We REQUIRE numeric columns to rebuild canonical window_key (do not trust string window_key)
    required = {"log_id", "window_t_start", "window_t_end", "track_uuid"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"{parquet_path.name} missing required columns: {missing}")

    emb_cols = embedding_cols(dim)
    missing_emb = [c for c in emb_cols if c not in df.columns]
    if missing_emb:
        raise ValueError(
            f"{parquet_path.name} missing embedding columns (expected dim={dim}). "
            f"Missing examples: {missing_emb[:5]}..."
        )

    total = len(df)
    invalid = 0
    nan_rows = 0
    built = 0

    # Keep original window_key if present (for debugging only)
    if "window_key" in df.columns:
        df["source_window_key"] = df["window_key"].astype(str)
    else:
        df["source_window_key"] = None

    # category optional
    if "category" not in df.columns:
        df["category"] = None

    # Normalize ids
    df["log_id"] = df["log_id"].astype(str)
    df["track_uuid"] = df["track_uuid"].astype(str)

    # Build canonical window_key + rounded times
    # Vectorized approach would be faster, but this is clearer and safer for now.
    canon_keys: List[str] = []
    ts_list: List[float] = []
    te_list: List[float] = []

    for i in range(len(df)):
        try:
            wkey, ts, te = canonical_window_key(
                df.iloc[i]["log_id"],
                df.iloc[i]["window_t_start"],
                df.iloc[i]["window_t_end"],
                t_precision,
            )
            canon_keys.append(wkey)
            ts_list.append(ts)
            te_list.append(te)
        except Exception:
            canon_keys.append(None)
            ts_list.append(np.nan)
            te_list.append(np.nan)

    df["window_key_canon"] = canon_keys
    df["t_start_round"] = ts_list
    df["t_end_round"] = te_list

    # Convert embeddings to float32 numpy arrays
    E = df[emb_cols].to_numpy(dtype=np.float32, copy=False)

    row_has_nan = np.isnan(E).any(axis=1)
    nan_rows = int(row_has_nan.sum())

    if nan_policy == "error" and nan_rows > 0:
        bad_idx = int(np.where(row_has_nan)[0][0])
        raise ValueError(
            f"Found NaNs in embeddings ({nan_rows}/{total}) under nan_policy=error. "
            f"Example idx={bad_idx}, log_id={df.iloc[bad_idx]['log_id']}, track_uuid={df.iloc[bad_idx]['track_uuid']}"
        )

    if nan_policy == "drop" and nan_rows > 0:
        keep = ~row_has_nan
        df = df.loc[keep].copy()
        E = E[keep]
        total = len(df)

    if nan_policy == "zero" and nan_rows > 0:
        E = np.nan_to_num(E, nan=0.0)

    records: List[Tuple] = []
    src = str(parquet_path)

    for i in range(len(df)):
        try:
            wkey = df.iloc[i]["window_key_canon"]
            tu = df.iloc[i]["track_uuid"]
            lid = df.iloc[i]["log_id"]
            ts = df.iloc[i]["t_start_round"]
            te = df.iloc[i]["t_end_round"]
            swk = df.iloc[i]["source_window_key"]
            cat = df.iloc[i]["category"]

            if not wkey or not tu or not lid:
                invalid += 1
                continue

            emb = E[i].tolist()
            emb_str = "[" + ",".join(f"{float(v):.8f}" for v in emb) + "]"

            records.append((
                split_name, embed_split,
                str(wkey), str(tu),
                str(lid), float(ts), float(te),
                (None if swk is None or (isinstance(swk, float) and np.isnan(swk)) else str(swk)),
                (None if pd.isna(cat) else str(cat)),
                dim, emb_str, src
            ))
            built += 1
        except Exception:
            invalid += 1
            continue

    stats = {
        "total_rows": total,
        "built_records": built,
        "invalid_rows": invalid,
        "nan_rows": nan_rows,
    }
    return records, stats


def insert_records(conn, records: List[Tuple], batch_size: int) -> None:
    if not records:
        return
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(
                cur,
                INSERT_SQL,
                chunk,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)",
                page_size=min(len(chunk), 1000),
            )
    conn.commit()


def overlap_stats_with_scenario_actors(conn, split_name: str, embed_split: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COUNT(*) AS n_embeddings,
              SUM(CASE WHEN a.window_key IS NOT NULL THEN 1 ELSE 0 END) AS n_overlap
            FROM s2_actor_embeddings e
            LEFT JOIN scenario_actors a
              ON a.split_name=e.split_name
             AND a.window_key=e.window_key
             AND a.track_uuid=e.track_uuid
            WHERE e.split_name=%s AND e.embed_split=%s
            """,
            (split_name, embed_split),
        )
        n_embeddings, n_overlap = cur.fetchone()
    logging.info(f"[{embed_split}] overlap with scenario_actors: {int(n_overlap)}/{int(n_embeddings)}")


def ingest_s2_embeddings(conn, cfg: Dict[str, Any]) -> None:
    s6d = cfg["s6d"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6d["split_name"])
    embeddings_dir = Path(s6d["embeddings_dir"])
    dim = int(s6d.get("embedding_dim", 128))
    nan_policy = parse_nan_policy(s6d.get("nan_policy", "error"))
    t_precision = int(s6d.get("window_key", {}).get("t_precision", 6))
    batch_size = int(runtime.get("batch_size", 2000))

    if dim != 128:
        raise ValueError("This script expects embedding_dim=128 (table column is vector(128)).")

    train_path = embeddings_dir / "embeddings_train.parquet"
    val_path = embeddings_dir / "embeddings_val.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing file: {val_path}")

    logging.info("S6D ingest S2 actor embeddings (canonical window_key)")
    logging.info(f"  split_name        : {split_name}")
    logging.info(f"  embeddings_dir    : {embeddings_dir}")
    logging.info(f"  t_precision       : {t_precision}")
    logging.info(f"  embedding_dim     : {dim}")
    logging.info(f"  nan_policy        : {nan_policy}")
    logging.info(f"  batch_size        : {batch_size}")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s", (split_name,))
        (nwin,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {int(nwin)}")

    # Train
    rec_train, stats_train = build_records_from_parquet(
        train_path, split_name, "train", dim, t_precision, nan_policy
    )
    logging.info("[train] Parquet summary:")
    for k, v in stats_train.items():
        logging.info(f"  {k:12s} : {v}")
    insert_records(conn, rec_train, batch_size)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM s2_actor_embeddings WHERE split_name=%s AND embed_split='train'",
            (split_name,),
        )
        (count_train,) = cur.fetchone()
    logging.info("[train] ingest result:")
    logging.info(f"  rows_present_for_subset : {int(count_train)}")
    overlap_stats_with_scenario_actors(conn, split_name, "train")

    # Val
    rec_val, stats_val = build_records_from_parquet(
        val_path, split_name, "val", dim, t_precision, nan_policy
    )
    logging.info("[val] Parquet summary:")
    for k, v in stats_val.items():
        logging.info(f"  {k:12s} : {v}")
    insert_records(conn, rec_val, batch_size)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM s2_actor_embeddings WHERE split_name=%s AND embed_split='val'",
            (split_name,),
        )
        (count_val,) = cur.fetchone()
    logging.info("[val] ingest result:")
    logging.info(f"  rows_present_for_subset : {int(count_val)}")
    overlap_stats_with_scenario_actors(conn, split_name, "val")

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
        if bool(cfg.get("runtime", {}).get("create_tables", True)):
            logging.info("Ensuring tables exist...")
            ensure_tables(conn)

        ingest_s2_embeddings(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
