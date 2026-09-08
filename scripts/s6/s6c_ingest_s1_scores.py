#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6C - Ingest S1 verifier window-level scores into PostgreSQL (user-space), idempotently.

Reads:
  - YAML config (paths, split name, runtime settings)
  - .env with DB credentials
  - final_scores.parquet (window-level scores)

Writes:
  - window_scores table (one row per detected braking window)

Idempotence:
  - PRIMARY KEY(window_key)  (NOTE: split isolation assumes window_key is split-unique)
  - INSERT ... ON CONFLICT DO UPDATE (safe re-runs; updates values deterministically)

Fix (2025-12):
  - Deduplicate records by window_key BEFORE execute_values to avoid:
      psycopg2.errors.CardinalityViolation:
      "ON CONFLICT DO UPDATE command cannot affect row a second time"

Sanity checks:
  - counts rows loaded/built/invalid
  - verifies window_key exists in scenario_windows (reports missing)
  - logs score ranges
  - logs DB row counts for this split after ingest

Run:
  python scripts/s6/s6c_ingest_s1_scores.py --config configs/s6_db_train650.yaml
"""

import argparse
import json
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

import pandas as pd
import numpy as np


# ---------------- logging ---------------- #

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
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

def to_float_or_none(x: Any) -> float:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def stable_row_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def canonicalize_window_key(log_id: str, t_start: float, t_end: float, t_precision: int) -> str:
    """
    Canonical window_key string with fixed decimal formatting to avoid:
      1.0 vs 1.000000 mismatches across stages.
    """
    fmt = f"{{:.{t_precision}f}}"
    return f"{str(log_id).strip()}|{fmt.format(float(t_start))}|{fmt.format(float(t_end))}"


# ---------------- db ddl ---------------- #

DDL_WINDOW_SCORES = """
CREATE TABLE IF NOT EXISTS window_scores (
  window_key     TEXT PRIMARY KEY,
  split_name     TEXT NOT NULL,
  log_id         UUID NOT NULL,
  t_start        DOUBLE PRECISION NOT NULL,
  t_end          DOUBLE PRECISION NOT NULL,

  score_pu_xgb   DOUBLE PRECISION,
  score_nnpu     DOUBLE PRECISION,
  score_fused    DOUBLE PRECISION,
  score_ema      DOUBLE PRECISION,
  post_hmm       DOUBLE PRECISION,
  score_final    DOUBLE PRECISION,

  source_path    TEXT NOT NULL,
  row_hash       TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_window_scores_split ON window_scores (split_name);
CREATE INDEX IF NOT EXISTS idx_window_scores_log   ON window_scores (log_id);
"""

def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_WINDOW_SCORES)
    conn.commit()


# ---------------- ingest ---------------- #

UPSERT_SQL = """
INSERT INTO window_scores (
  window_key, split_name, log_id, t_start, t_end,
  score_pu_xgb, score_nnpu, score_fused, score_ema, post_hmm, score_final,
  source_path, row_hash
)
VALUES %s
ON CONFLICT (window_key) DO UPDATE SET
  split_name   = EXCLUDED.split_name,
  log_id       = EXCLUDED.log_id,
  t_start      = EXCLUDED.t_start,
  t_end        = EXCLUDED.t_end,
  score_pu_xgb = EXCLUDED.score_pu_xgb,
  score_nnpu   = EXCLUDED.score_nnpu,
  score_fused  = EXCLUDED.score_fused,
  score_ema    = EXCLUDED.score_ema,
  post_hmm     = EXCLUDED.post_hmm,
  score_final  = EXCLUDED.score_final,
  source_path  = EXCLUDED.source_path,
  row_hash     = EXCLUDED.row_hash,
  updated_at   = NOW();
"""

EXPECTED_TUPLE_LEN = 13


def _pick_better_record(existing: Tuple, candidate: Tuple) -> Tuple:
    """
    Choose one record when window_key duplicates exist.

    Tuple layout (fixed):
      (wkey, split_name, log_id, t_start, t_end,
       score_pu_xgb, score_nnpu, score_fused, score_ema, post_hmm, score_final,
       source_path, row_hash)

    Policy:
      - prefer higher score_final (if both not None)
      - else keep candidate (last seen) deterministically
    """
    ex_sf = existing[10]
    ca_sf = candidate[10]
    try:
        ex_v = -1.0 if ex_sf is None else float(ex_sf)
    except Exception:
        ex_v = -1.0
    try:
        ca_v = -1.0 if ca_sf is None else float(ca_sf)
    except Exception:
        ca_v = -1.0

    if ca_v > ex_v:
        return candidate
    if ca_v < ex_v:
        return existing
    # tie: keep candidate (last seen)
    return candidate


def ingest_s1_scores(conn, cfg: Dict[str, Any]) -> None:
    s6 = cfg["s6c"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6.get("split_name"))
    scores_path = Path(s6["scores_parquet"])

    # Allow either s6c.t_precision or runtime.window_key.t_precision
    t_precision = int(
        s6.get("t_precision", cfg.get("runtime", {}).get("window_key", {}).get("t_precision", 6))
    )
    batch_size = int(s6.get("batch_size", runtime.get("batch_size", 2000)))

    if not scores_path.exists():
        raise FileNotFoundError(f"S6C scores parquet not found: {scores_path}")

    logging.info("S6C ingest S1 verifier window scores")
    logging.info(f"  split_name      : {split_name}")
    logging.info(f"  scores_parquet  : {scores_path}")
    logging.info(f"  t_precision     : {t_precision}")
    logging.info(f"  batch_size      : {batch_size}")

    # DB pre-check: windows exist?
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s;", (split_name,))
        (n_windows,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {n_windows}")

    df = pd.read_parquet(scores_path)
    logging.info(f"Loaded parquet: rows={len(df)}, cols={len(df.columns)}")

    required = {
        "log_id", "t_start", "t_end",
        "score_pu_xgb", "score_nnpu", "score_fused", "score_ema", "post_hmm", "score_final"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"final_scores.parquet missing required columns: {sorted(missing)}")

    # Build records + sanity
    total = len(df)
    invalid = 0
    built = 0
    source_path = str(scores_path)

    # We'll build a dict keyed by window_key to deduplicate safely
    rec_by_wkey: Dict[str, Tuple] = {}
    hash_by_wkey: Dict[str, str] = {}

    dup_total = 0
    dup_conflicting = 0

    wkeys_for_join: List[str] = []

    for _, r in df.iterrows():
        log_id = r.get("log_id")
        t_start_raw = r.get("t_start")
        t_end_raw = r.get("t_end")

        if pd.isna(log_id) or pd.isna(t_start_raw) or pd.isna(t_end_raw):
            invalid += 1
            continue

        try:
            t_start = round(float(t_start_raw), t_precision)
            t_end = round(float(t_end_raw), t_precision)
        except Exception:
            invalid += 1
            continue

        if t_end <= t_start:
            invalid += 1
            continue

        wkey = canonicalize_window_key(str(log_id), t_start, t_end, t_precision)
        wkeys_for_join.append(wkey)

        row_dict = {
            "log_id": str(log_id),
            "t_start": t_start,
            "t_end": t_end,
            "score_pu_xgb": to_float_or_none(r.get("score_pu_xgb")),
            "score_nnpu": to_float_or_none(r.get("score_nnpu")),
            "score_fused": to_float_or_none(r.get("score_fused")),
            "score_ema": to_float_or_none(r.get("score_ema")),
            "post_hmm": to_float_or_none(r.get("post_hmm")),
            "score_final": to_float_or_none(r.get("score_final")),
        }
        rh = stable_row_hash(row_dict)

        rec = (
            wkey, split_name, str(log_id), t_start, t_end,
            row_dict["score_pu_xgb"], row_dict["score_nnpu"], row_dict["score_fused"],
            row_dict["score_ema"], row_dict["post_hmm"], row_dict["score_final"],
            source_path, rh
        )

        if wkey in rec_by_wkey:
            dup_total += 1
            # detect whether it is a "conflicting" duplicate (different content)
            if hash_by_wkey.get(wkey) != rh:
                dup_conflicting += 1
            # choose deterministically
            chosen = _pick_better_record(rec_by_wkey[wkey], rec)
            rec_by_wkey[wkey] = chosen
            hash_by_wkey[wkey] = chosen[-1]
        else:
            rec_by_wkey[wkey] = rec
            hash_by_wkey[wkey] = rh

        built += 1

    records: List[Tuple] = list(rec_by_wkey.values())

    logging.info("Parquet summary:")
    logging.info(f"  total_rows          : {total}")
    logging.info(f"  built_records       : {built}")
    logging.info(f"  invalid_rows        : {invalid}")
    logging.info(f"  unique_window_keys  : {len(records)}")
    logging.info(f"  duplicate_wkeys     : {dup_total} (conflicting={dup_conflicting})")

    # Score range sanity (informational)
    for col in ["score_pu_xgb", "score_nnpu", "score_fused", "score_ema", "post_hmm", "score_final"]:
        vals = df[col].astype(float)
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            logging.info(f"  {col:12s} : (all null/NaN)")
        else:
            logging.info(f"  {col:12s} : min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Sanity: how many of these window_keys exist in scenario_windows?
    missing_windows = 0
    if records:
        # Use unique wkeys for the join check
        uniq_wkeys = list(rec_by_wkey.keys())
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT UNNEST(%s::text[]) AS window_key
                ) w
                LEFT JOIN scenario_windows sw
                  ON sw.window_key = w.window_key
                 AND sw.split_name = %s
                WHERE sw.window_key IS NULL;
                """,
                (uniq_wkeys, split_name)
            )
            (missing_windows,) = cur.fetchone()

    if missing_windows > 0:
        logging.warning(f"{missing_windows} score rows have window_key not found in scenario_windows for split={split_name}.")
        logging.warning("This usually means timestamp rounding/precision mismatch between S6A and S6C.")
    else:
        logging.info("All score window_keys exist in scenario_windows (join OK).")

    # Guard: tuple length must match INSERT columns count (13)
    if records and len(records[0]) != EXPECTED_TUPLE_LEN:
        raise RuntimeError(
            f"BUG: record tuple length={len(records[0])} but expected {EXPECTED_TUPLE_LEN}. "
            "Fix INSERT column list or tuple packing."
        )

    # Upsert (now safe because records are unique by window_key)
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(cur, UPSERT_SQL, chunk, page_size=min(len(chunk), 1000))
        conn.commit()

    # Post-check
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM window_scores WHERE split_name=%s;", (split_name,))
        (count_scores,) = cur.fetchone()

    logging.info("window_scores ingest result:")
    logging.info(f"  rows_present_for_split : {count_scores}")
    logging.info("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/s6_db_train650.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
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

        ingest_s1_scores(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
