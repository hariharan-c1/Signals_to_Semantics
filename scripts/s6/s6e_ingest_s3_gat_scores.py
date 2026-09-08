#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6E - Ingest GAT teacher scores and GAT inference (top-3) into PostgreSQL, idempotently.

Reads:
  - YAML config (paths, split name, runtime settings)
  - .env with DB credentials
  - teacher.parquet (soft q(a) per (window_key, track_uuid))
  - top3_infer.parquet (rank, score per (window_key, track_uuid))

Writes:
  - gat_teacher_scores
  - gat_inference_scores

Key detail:
  The parquet window_key uses high-precision timestamps. We canonicalize it to match the DB
  (same rounding policy as S6A/S6B/S6C/S6D), and store the original as source_window_key.

Idempotence:
  - gat_teacher_scores UNIQUE(split_name, window_key, track_uuid)
  - gat_inference_scores UNIQUE(split_name, window_key, track_uuid)
  - UPSERT to allow safe reruns if values change (still idempotent)

Run:
  python scripts/s6/s6e_ingest_gat_scores.py --config configs/s6_db.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd
import numpy as np
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

DDL_GAT_TEACHER = """
CREATE TABLE IF NOT EXISTS gat_teacher_scores (
  split_name        TEXT NOT NULL,
  window_key        TEXT NOT NULL,
  track_uuid        UUID NOT NULL,
  category          TEXT,
  q                DOUBLE PRECISION,
  score_window_raw  DOUBLE PRECISION,
  source_path       TEXT NOT NULL,
  source_window_key TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gat_teacher
  ON gat_teacher_scores (split_name, window_key, track_uuid);

CREATE INDEX IF NOT EXISTS idx_gat_teacher_window
  ON gat_teacher_scores (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_gat_teacher_track
  ON gat_teacher_scores (split_name, track_uuid);
"""

DDL_GAT_INFER = """
CREATE TABLE IF NOT EXISTS gat_inference_scores (
  split_name        TEXT NOT NULL,
  window_key        TEXT NOT NULL,
  track_uuid        UUID NOT NULL,
  rank             INT NOT NULL,
  score            DOUBLE PRECISION,
  source_path       TEXT NOT NULL,
  source_window_key TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (rank >= 1 AND rank <= 10)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gat_infer
  ON gat_inference_scores (split_name, window_key, track_uuid);

CREATE INDEX IF NOT EXISTS idx_gat_infer_window
  ON gat_inference_scores (split_name, window_key);

CREATE INDEX IF NOT EXISTS idx_gat_infer_rank
  ON gat_inference_scores (split_name, rank);
"""


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_GAT_TEACHER)
        cur.execute(DDL_GAT_INFER)
    conn.commit()


# ---------------- canonicalization ---------------- #

def _fmt_rounded(x: float, p: int) -> str:
    """
    Round to p decimals, format deterministically, and strip trailing zeros/dot
    so it matches prior window_key behavior (no useless trailing zeros).
    """
    s = format(round(float(x), p), f".{p}f")
    s = s.rstrip("0").rstrip(".")
    return s


def canonicalize_window_key(source_window_key: str, t_precision: int) -> Tuple[str, str, float, float]:
    """
    Parse 'log_id|t_start|t_end' and return canonical window_key plus components.

    Returns:
      (canonical_window_key, log_id, t_start_float, t_end_float)
    """
    parts = str(source_window_key).split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid window_key format: {source_window_key}")

    log_id = parts[0]
    t_start = float(parts[1])
    t_end = float(parts[2])

    ts_str = _fmt_rounded(t_start, t_precision)
    te_str = _fmt_rounded(t_end, t_precision)

    canonical = f"{log_id}|{ts_str}|{te_str}"
    return canonical, log_id, float(ts_str), float(te_str)


# ---------------- ingest helpers ---------------- #

def read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def nan_to_none(x: Any) -> Any:
    if x is None:
        return None
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
    except Exception:
        pass
    return x


UPSERT_TEACHER = """
INSERT INTO gat_teacher_scores (
  split_name, window_key, track_uuid, category, q, score_window_raw,
  source_path, source_window_key
)
VALUES %s
ON CONFLICT (split_name, window_key, track_uuid) DO UPDATE SET
  category = EXCLUDED.category,
  q = EXCLUDED.q,
  score_window_raw = EXCLUDED.score_window_raw,
  source_path = EXCLUDED.source_path,
  source_window_key = EXCLUDED.source_window_key;
"""

UPSERT_INFER = """
INSERT INTO gat_inference_scores (
  split_name, window_key, track_uuid, rank, score,
  source_path, source_window_key
)
VALUES %s
ON CONFLICT (split_name, window_key, track_uuid) DO UPDATE SET
  rank = EXCLUDED.rank,
  score = EXCLUDED.score,
  source_path = EXCLUDED.source_path,
  source_window_key = EXCLUDED.source_window_key;
"""


def insert_values(conn, sql: str, records: List[Tuple], batch_size: int) -> None:
    if not records:
        return
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(cur, sql, chunk, page_size=min(len(chunk), 1000))
    conn.commit()


# ---------------- S6E ingestion ---------------- #

def build_teacher_records(
    df: pd.DataFrame,
    split_name: str,
    source_path: str,
    t_precision: int
) -> Tuple[List[Tuple], Dict[str, int]]:
    required = {"window_key", "track_uuid", "q", "score_window_raw"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"teacher.parquet missing required columns: {missing}")

    total = len(df)
    invalid = 0
    built = 0
    key_fail = 0

    # optional
    if "category" not in df.columns:
        df["category"] = None

    records: List[Tuple] = []
    for _, r in df.iterrows():
        try:
            src_wkey = str(r["window_key"])
            wkey, _, _, _ = canonicalize_window_key(src_wkey, t_precision)

            track_uuid = str(r["track_uuid"])
            cat = r.get("category", None)
            q = nan_to_none(r.get("q", None))
            swr = nan_to_none(r.get("score_window_raw", None))

            if not wkey or not track_uuid:
                invalid += 1
                continue

            records.append((
                split_name, wkey, track_uuid,
                None if pd.isna(cat) else str(cat),
                None if q is None else float(q),
                None if swr is None else float(swr),
                source_path, src_wkey
            ))
            built += 1
        except ValueError:
            key_fail += 1
            invalid += 1
        except Exception:
            invalid += 1

    stats = {
        "total_rows": total,
        "built_records": built,
        "invalid_rows": invalid,
        "window_key_parse_fail": key_fail,
    }
    return records, stats


def build_infer_records(
    df: pd.DataFrame,
    split_name: str,
    source_path: str,
    t_precision: int
) -> Tuple[List[Tuple], Dict[str, int]]:
    required = {"window_key", "rank", "track_uuid", "score"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"top3_infer.parquet missing required columns: {missing}")

    total = len(df)
    invalid = 0
    built = 0
    key_fail = 0

    records: List[Tuple] = []
    for _, r in df.iterrows():
        try:
            src_wkey = str(r["window_key"])
            wkey, _, _, _ = canonicalize_window_key(src_wkey, t_precision)

            track_uuid = str(r["track_uuid"])
            rank = int(r["rank"])
            score = nan_to_none(r.get("score", None))

            if not wkey or not track_uuid:
                invalid += 1
                continue
            if rank <= 0:
                invalid += 1
                continue

            records.append((
                split_name, wkey, track_uuid, rank,
                None if score is None else float(score),
                source_path, src_wkey
            ))
            built += 1
        except ValueError:
            key_fail += 1
            invalid += 1
        except Exception:
            invalid += 1

    stats = {
        "total_rows": total,
        "built_records": built,
        "invalid_rows": invalid,
        "window_key_parse_fail": key_fail,
    }
    return records, stats


def sanity_overlap(conn, split_name: str) -> None:
    """
    Informational sanity: how many rows can join to scenario_actors.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM gat_teacher_scores WHERE split_name=%s) AS n_teacher,
              (SELECT COUNT(*) FROM gat_inference_scores WHERE split_name=%s) AS n_infer
            """,
            (split_name, split_name),
        )
        n_teacher, n_infer = cur.fetchone()

        cur.execute(
            """
            SELECT COUNT(*) FROM gat_teacher_scores t
            LEFT JOIN scenario_actors a
              ON a.split_name=t.split_name
             AND a.window_key=t.window_key
             AND a.track_uuid=t.track_uuid
            WHERE t.split_name=%s AND a.window_key IS NULL
            """,
            (split_name,),
        )
        (teacher_without_actor,) = cur.fetchone()

        cur.execute(
            """
            SELECT COUNT(*) FROM gat_inference_scores g
            LEFT JOIN scenario_actors a
              ON a.split_name=g.split_name
             AND a.window_key=g.window_key
             AND a.track_uuid=g.track_uuid
            WHERE g.split_name=%s AND a.window_key IS NULL
            """,
            (split_name,),
        )
        (infer_without_actor,) = cur.fetchone()

        # rank sanity: count duplicates of (window_key, rank)
        cur.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT window_key, rank, COUNT(*) c
              FROM gat_inference_scores
              WHERE split_name=%s
              GROUP BY window_key, rank
              HAVING COUNT(*) > 1
            ) x
            """,
            (split_name,),
        )
        (dup_rank_pairs,) = cur.fetchone()

    logging.info("Sanity (join + rank):")
    logging.info(f"  teacher_rows_total        : {int(n_teacher)}")
    logging.info(f"  inference_rows_total      : {int(n_infer)}")
    logging.info(f"  teacher_without_actor     : {int(teacher_without_actor)}")
    logging.info(f"  inference_without_actor   : {int(infer_without_actor)}")
    logging.info(f"  duplicate_(window,rank)   : {int(dup_rank_pairs)}")


def ingest_gat(conn, cfg: Dict[str, Any]) -> None:
    s6e = cfg["s6e"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6e["split_name"])
    teacher_path = Path(s6e["teacher_parquet"])
    infer_path = Path(s6e["top3_parquet"])

    t_precision = int(s6e.get("window_key", {}).get("t_precision", 6))
    batch_size = int(runtime.get("batch_size", 2000))

    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing teacher parquet: {teacher_path}")
    if not infer_path.exists():
        raise FileNotFoundError(f"Missing top3 parquet: {infer_path}")

    logging.info("S6E ingest GAT teacher + inference")
    logging.info(f"  split_name      : {split_name}")
    logging.info(f"  teacher_parquet : {teacher_path}")
    logging.info(f"  top3_parquet    : {infer_path}")
    logging.info(f"  t_precision     : {t_precision}")
    logging.info(f"  batch_size      : {batch_size}")

    # health check
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s", (split_name,))
        (nwin,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {int(nwin)}")

    # ---- teacher ----
    df_t = read_parquet(teacher_path)
    logging.info(f"Loaded parquet: {teacher_path.name} rows={len(df_t)}, cols={len(df_t.columns)}")

    teacher_records, teacher_stats = build_teacher_records(
        df_t, split_name, str(teacher_path), t_precision
    )

    logging.info("Teacher parquet summary:")
    for k, v in teacher_stats.items():
        logging.info(f"  {k:20s} : {v}")

    insert_values(conn, UPSERT_TEACHER, teacher_records, batch_size)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gat_teacher_scores WHERE split_name=%s", (split_name,))
        (n_teacher,) = cur.fetchone()
    logging.info(f"gat_teacher_scores rows_present_for_split : {int(n_teacher)}")

    # ---- inference ----
    df_i = read_parquet(infer_path)
    logging.info(f"Loaded parquet: {infer_path.name} rows={len(df_i)}, cols={len(df_i.columns)}")

    infer_records, infer_stats = build_infer_records(
        df_i, split_name, str(infer_path), t_precision
    )

    logging.info("Inference parquet summary:")
    for k, v in infer_stats.items():
        logging.info(f"  {k:20s} : {v}")

    insert_values(conn, UPSERT_INFER, infer_records, batch_size)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gat_inference_scores WHERE split_name=%s", (split_name,))
        (n_infer,) = cur.fetchone()
    logging.info(f"gat_inference_scores rows_present_for_split : {int(n_infer)}")

    # sanity joins
    sanity_overlap(conn, split_name)

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

        ingest_gat(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
