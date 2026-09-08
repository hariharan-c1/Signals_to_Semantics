#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6A - Ingest braking windows into PostgreSQL (user-space), idempotently.

Reads:
  - YAML config (paths, split name, runtime settings)
  - .env with DB credentials

Writes:
  - scenario_windows table (one row per detected braking window)

Idempotence:
  - PRIMARY KEY(window_key)
  - INSERT ... ON CONFLICT DO NOTHING

Sanity checks:
  - counts rows, duplicates (by window_key), invalid rows (missing fields, t_end <= t_start)
  - logs summary stats at the end

Run:
  python scripts/s6/s6a_ingest_windows.py --config configs/s6_db.yaml
"""

import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extras import execute_values


# ---------------- logging ---------------- #

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="[%(levelname)s] %(message)s"
    )


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

DDL_SCENARIO_WINDOWS = """
CREATE TABLE IF NOT EXISTS scenario_windows (
  window_key      TEXT PRIMARY KEY,
  log_id          UUID NOT NULL,
  i_start         INT,
  i_end           INT,
  t_start         DOUBLE PRECISION NOT NULL,
  t_end           DOUBLE PRECISION NOT NULL,
  dur_s           DOUBLE PRECISION,
  a_min_ms2       DOUBLE PRECISION,
  j_min_ms3       DOUBLE PRECISION,
  split_name      TEXT NOT NULL,
  detector_name   TEXT NOT NULL,
  source_path     TEXT NOT NULL,
  row_hash        TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_windows_log_id ON scenario_windows (log_id);
CREATE INDEX IF NOT EXISTS idx_windows_t_start ON scenario_windows (t_start);
"""


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_SCENARIO_WINDOWS)
    conn.commit()


# ---------------- helpers ---------------- #

def round_t(x: Any, precision: int) -> float:
    # JSON may contain int/float; make it float consistently
    return round(float(x), precision)


def make_window_key(log_id: str, t_start: float, t_end: float) -> str:
    return f"{log_id}|{t_start}|{t_end}"


def stable_row_hash(row: Dict[str, Any]) -> str:
    # stable hash of JSON row (sorted keys)
    s = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------- ingest ---------------- #

INSERT_SQL = """
INSERT INTO scenario_windows (
  window_key, log_id, i_start, i_end,
  t_start, t_end, dur_s, a_min_ms2, j_min_ms3,
  split_name, detector_name, source_path, row_hash
)
VALUES %s
ON CONFLICT (window_key) DO NOTHING;
"""


def ingest_windows(conn, cfg: Dict[str, Any]) -> None:
    s6 = cfg["s6a_windows"]
    runtime = cfg.get("runtime", {})
    batch_size = int(runtime.get("batch_size", 2000))
    t_precision = int(s6.get("window_key", {}).get("t_precision", 6))

    input_path = Path(s6["input_jsonl"])
    split_name = str(s6["split_name"])
    detector_name = str(s6["detector_name"])
    source_path = str(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    logging.info(f"S6A ingest windows")
    logging.info(f"  split_name     : {split_name}")
    logging.info(f"  detector_name  : {detector_name}")
    logging.info(f"  input_jsonl    : {input_path}")
    logging.info(f"  t_precision    : {t_precision}")
    logging.info(f"  batch_size     : {batch_size}")

    raw_rows = read_jsonl(input_path)
    logging.info(f"Loaded {len(raw_rows)} rows from JSONL")

    # Build records + sanity checks
    total = 0
    invalid = 0
    built = 0
    dup_in_file = 0

    seen_keys = set()
    records: List[Tuple] = []

    for row in raw_rows:
        total += 1

        log_id = row.get("log_id")
        t_start_raw = row.get("t_start")
        t_end_raw = row.get("t_end")

        if not log_id or t_start_raw is None or t_end_raw is None:
            invalid += 1
            continue

        t_start = round_t(t_start_raw, t_precision)
        t_end = round_t(t_end_raw, t_precision)

        if t_end <= t_start:
            invalid += 1
            continue

        wkey = make_window_key(str(log_id), t_start, t_end)

        if wkey in seen_keys:
            dup_in_file += 1
            continue
        seen_keys.add(wkey)

        i_start = row.get("i_start")
        i_end = row.get("i_end")
        dur_s = row.get("dur_s")
        a_min = row.get("a_min_ms2")
        j_min = row.get("j_min_ms3")

        rh = stable_row_hash(row)

        records.append((
            wkey, log_id, i_start, i_end,
            t_start, t_end, dur_s, a_min, j_min,
            split_name, detector_name, source_path, rh
        ))
        built += 1

    logging.info(f"Sanity summary (file-level):")
    logging.info(f"  total_rows      : {total}")
    logging.info(f"  built_records   : {built}")
    logging.info(f"  invalid_rows    : {invalid}")
    logging.info(f"  dup_in_file     : {dup_in_file}")

    # Insert in batches
    inserted_total = 0
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(cur, INSERT_SQL, chunk, page_size=len(chunk))
            # rowcount after execute_values is not reliable for ON CONFLICT; we measure via a query later
        conn.commit()

    # Measure effect: how many rows exist for this split+source
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM scenario_windows
            WHERE split_name = %s AND detector_name = %s AND source_path = %s
            """,
            (split_name, detector_name, source_path)
        )
        (count_in_db,) = cur.fetchone()

    logging.info(f"DB check:")
    logging.info(f"  rows_present_for_this_ingest : {count_in_db}")
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

        ingest_windows(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
