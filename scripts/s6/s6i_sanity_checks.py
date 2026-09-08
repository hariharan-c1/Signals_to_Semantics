#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6I - Scenario DB sanity checks & audit report (schema-introspective).

Reads:
  - YAML config (split name, output path, runtime)
  - .env with DB credentials

Does:
  - introspects which tables/columns exist in public schema
  - reports row counts (split-aware if split_name column exists)
  - computes coverage + join integrity checks across S6 tables
  - reports LLM backend coverage per prompt/model
  - validates scenario embeddings coverage vs llm_predictions

Writes:
  - CSV report with sections (table_counts, coverage, joins, backends, notes)

Run:
  python scripts/s6/s6i_sanity_checks.py --config configs/s6_db.yaml
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
import psycopg2


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


# ---------------- db introspection ---------------- #

def table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_schema='public' AND table_name=%s
            )
            """,
            (table,),
        )
        (ok,) = cur.fetchone()
    return bool(ok)


def list_tables(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
              AND table_type='BASE TABLE'
            ORDER BY table_name
            """
        )
        return [r[0] for r in cur.fetchall()]


def get_columns(conn, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def has_column(conn, table: str, col: str) -> bool:
    return col in set(get_columns(conn, table))


# ---------------- reporting helpers ---------------- #

def add_row(rows: List[Dict[str, Any]], section: str, metric: str, value: Any, details: Optional[Dict[str, Any]] = None) -> None:
    rows.append({
        "section": section,
        "metric": metric,
        "value": value,
        "details_json": json.dumps(details or {}, ensure_ascii=False),
    })


def scalar(conn, sql: str, params: Tuple = ()) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def fetchall(conn, sql: str, params: Tuple = ()) -> List[Tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------- core checks ---------------- #

S6_TABLE_CANDIDATES = [
    "scenario_windows",
    "scenario_actors",
    "actor_features",
    "window_scores",
    "s2_actor_embeddings",
    "gat_teacher_scores",
    "gat_inference_scores",
    "gt_window_truth",
    "llm_backends",
    "llm_predictions",
    # scenario embeddings tables may vary by model name
    "scenario_embeddings_gemma",
    "scenario_embeddings_qwen",
]


def row_count(conn, table: str, split_name: Optional[str]) -> int:
    cols = set(get_columns(conn, table))
    if split_name and "split_name" in cols:
        return int(scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE split_name=%s", (split_name,)))
    return int(scalar(conn, f"SELECT COUNT(*) FROM {table}"))


def distinct_windows(conn, table: str, split_name: Optional[str]) -> Optional[int]:
    cols = set(get_columns(conn, table))
    if "window_key" not in cols:
        return None
    if split_name and "split_name" in cols:
        return int(scalar(conn, f"SELECT COUNT(DISTINCT window_key) FROM {table} WHERE split_name=%s", (split_name,)))
    return int(scalar(conn, f"SELECT COUNT(DISTINCT window_key) FROM {table}"))


def join_missing_count(conn, left_table: str, right_table: str, join_keys: List[str], split_name: Optional[str]) -> Optional[int]:
    if not table_exists(conn, left_table) or not table_exists(conn, right_table):
        return None
    lcols = set(get_columns(conn, left_table))
    rcols = set(get_columns(conn, right_table))
    if any(k not in lcols for k in join_keys) or any(k not in rcols for k in join_keys):
        return None

    where = []
    params: List[Any] = []

    if split_name and "split_name" in lcols and "split_name" in rcols:
        where.append("l.split_name = %s")
        params.append(split_name)

    on = " AND ".join([f"l.{k}=r.{k}" for k in join_keys])
    sql = f"""
    SELECT COUNT(*)
    FROM {left_table} l
    LEFT JOIN {right_table} r
      ON {on}
    {"WHERE " + " AND ".join(where + ["r." + join_keys[0] + " IS NULL"]) if where else "WHERE r." + join_keys[0] + " IS NULL"}
    """
    return int(scalar(conn, sql, tuple(params)))


def backend_coverage(conn, split_name: str) -> List[Dict[str, Any]]:
    if not table_exists(conn, "llm_predictions") or not table_exists(conn, "llm_backends"):
        return []
    # Don’t assume columns beyond the ones we know exist in your schema.
    bcols = set(get_columns(conn, "llm_backends"))
    pcols = set(get_columns(conn, "llm_predictions"))
    needed_b = {"backend_id", "prompt_type", "backend_dir"}
    needed_p = {"backend_id", "window_key", "split_name"}

    if not needed_b.issubset(bcols) or not needed_p.issubset(pcols):
        return []

    sql = """
    SELECT
      b.prompt_type,
      b.backend_dir,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT p.window_key) AS distinct_windows
    FROM llm_predictions p
    JOIN llm_backends b ON b.backend_id=p.backend_id
    WHERE p.split_name=%s
    GROUP BY 1,2
    ORDER BY 1,2
    """
    out = []
    for prompt_type, backend_dir, n_rows, distinct_w in fetchall(conn, sql, (split_name,)):
        out.append({
            "prompt_type": prompt_type,
            "backend_dir": backend_dir,
            "n_rows": int(n_rows),
            "distinct_windows": int(distinct_w),
        })
    return out


def embedding_coverage(conn, split_name: str, embed_table: str) -> Dict[str, Any]:
    if not table_exists(conn, embed_table):
        return {}

    cols = set(get_columns(conn, embed_table))
    required = {"split_name", "backend_id", "window_key", "embedding_model", "embed_dim"}
    if not required.issubset(cols):
        return {}

    total = int(scalar(conn, f"SELECT COUNT(*) FROM {embed_table} WHERE split_name=%s", (split_name,)))
    dims = fetchall(conn, f"SELECT embed_dim, COUNT(*) FROM {embed_table} WHERE split_name=%s GROUP BY 1 ORDER BY 1", (split_name,))
    by_dim = {int(d): int(c) for d, c in dims}

    # Check embeddings without llm_predictions
    if table_exists(conn, "llm_predictions"):
        missing = int(scalar(conn, f"""
        SELECT COUNT(*)
        FROM {embed_table} e
        LEFT JOIN llm_predictions p
          ON p.backend_id=e.backend_id
         AND p.window_key=e.window_key
         AND p.split_name=e.split_name
        WHERE e.split_name=%s AND p.window_key IS NULL
        """, (split_name,)))
    else:
        missing = None

    return {"total_rows": total, "by_dim": by_dim, "embeddings_without_llm": missing}


def write_report_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "metric", "value", "details_json"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------- main ---------------- #

def run_sanity(conn, cfg: Dict[str, Any]) -> Path:
    s6i = cfg.get("s6i", {})
    runtime = cfg.get("runtime", {})

    split_name = str(s6i.get("split_name", "val50"))
    out_csv = Path(s6i.get("out_csv", f"artifacts/s6/s6i_report_{split_name}.csv"))

    logging.info("S6I scenario DB sanity checks")
    logging.info(f"  split_name : {split_name}")
    logging.info(f"  out_csv    : {out_csv}")

    report_rows: List[Dict[str, Any]] = []

    # 0) Schema snapshot
    tables = list_tables(conn)
    add_row(report_rows, "schema", "tables_in_public_schema", len(tables), {"tables": tables})

    # 1) Table counts (only for tables that exist)
    add_row(report_rows, "table_counts", "note", "split-aware where split_name exists")
    for t in S6_TABLE_CANDIDATES:
        if table_exists(conn, t):
            cnt = row_count(conn, t, split_name)
            dw = distinct_windows(conn, t, split_name)
            add_row(report_rows, "table_counts", f"{t}.rows", cnt)
            if dw is not None:
                add_row(report_rows, "table_counts", f"{t}.distinct_window_keys", dw)

    # 2) Coverage metrics anchored on scenario_windows
    if table_exists(conn, "scenario_windows"):
        nwin = row_count(conn, "scenario_windows", split_name)
        add_row(report_rows, "coverage", "scenario_windows.total_windows", nwin)

        # Windows with GT
        if table_exists(conn, "gt_window_truth"):
            gt_dw = distinct_windows(conn, "gt_window_truth", split_name) or 0
            add_row(report_rows, "coverage", "windows_with_ground_truth", gt_dw, {"pct": (gt_dw / max(nwin, 1))})

        # Windows with verifier scores
        if table_exists(conn, "window_scores"):
            s1_dw = distinct_windows(conn, "window_scores", split_name) or 0
            add_row(report_rows, "coverage", "windows_with_s1_scores", s1_dw, {"pct": (s1_dw / max(nwin, 1))})

        # Windows with GAT inference
        if table_exists(conn, "gat_inference_scores"):
            gat_dw = distinct_windows(conn, "gat_inference_scores", split_name) or 0
            add_row(report_rows, "coverage", "windows_with_gat_inference", gat_dw, {"pct": (gat_dw / max(nwin, 1))})

        # Windows with LLM predictions
        if table_exists(conn, "llm_predictions"):
            llm_dw = distinct_windows(conn, "llm_predictions", split_name) or 0
            add_row(report_rows, "coverage", "windows_with_any_llm_prediction", llm_dw, {"pct": (llm_dw / max(nwin, 1))})

    # 3) Join integrity checks
    add_row(report_rows, "joins", "note", "counts of left rows that fail to join right on canonical keys")

    checks = [
        ("scenario_actors", "scenario_windows", ["split_name", "window_key"]),
        ("actor_features", "scenario_actors", ["split_name", "window_key", "track_uuid"]),
        ("window_scores", "scenario_windows", ["split_name", "window_key"]),
        ("gat_teacher_scores", "scenario_actors", ["split_name", "window_key", "track_uuid"]),
        ("gat_inference_scores", "scenario_actors", ["split_name", "window_key", "track_uuid"]),
        ("gt_window_truth", "scenario_windows", ["split_name", "window_key"]),
        ("llm_predictions", "scenario_windows", ["split_name", "window_key"]),
    ]

    for lt, rt, keys in checks:
        if not table_exists(conn, lt) or not table_exists(conn, rt):
            continue
        # remove split_name from keys if either side doesn't have it
        lcols = set(get_columns(conn, lt))
        rcols = set(get_columns(conn, rt))
        join_keys = [k for k in keys if (k in lcols and k in rcols)]
        # we need at least window_key; otherwise skip
        if "window_key" not in join_keys:
            continue
        missing = join_missing_count(conn, lt, rt, join_keys, split_name)
        if missing is not None:
            add_row(report_rows, "joins", f"{lt} -> {rt} missing", missing, {"join_keys": join_keys})

    # embeddings join check (if present)
    for embed_table in ["scenario_embeddings_gemma", "scenario_embeddings_qwen"]:
        if table_exists(conn, embed_table):
            cov = embedding_coverage(conn, split_name, embed_table)
            if cov:
                add_row(report_rows, "embeddings", f"{embed_table}.rows", cov["total_rows"])
                add_row(report_rows, "embeddings", f"{embed_table}.by_dim", "see details", {"by_dim": cov["by_dim"]})
                add_row(report_rows, "embeddings", f"{embed_table}.embeddings_without_llm", cov["embeddings_without_llm"])

    # 4) LLM backend coverage table
    bc = backend_coverage(conn, split_name)
    if bc:
        add_row(report_rows, "llm_backends", "configs_present", len(bc), {"configs": bc})
        # Also compute min/max windows across configs (useful quick warning)
        dws = [x["distinct_windows"] for x in bc]
        add_row(report_rows, "llm_backends", "distinct_windows.min", min(dws))
        add_row(report_rows, "llm_backends", "distinct_windows.max", max(dws))

    # Write report
    write_report_csv(out_csv, report_rows)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/s6_db.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    env_file = Path(cfg["db"]["env_file"])
    db_params = load_db_env(env_file)

    logging.info("Connecting to DB...")
    conn = psycopg2.connect(**db_params)
    conn.autocommit = False
    try:
        out_csv = run_sanity(conn, cfg)
        logging.info(f"Report written: {out_csv}")
        logging.info("Done.")
    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
