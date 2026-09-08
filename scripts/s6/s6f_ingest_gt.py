#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6F - Ingest Ground Truth (GT) labels into PostgreSQL, idempotently.

Fixes:
  - Canonicalizes window_key with fixed decimal formatting (t_precision)
  - Deduplicates by (split_name, window_key) BEFORE execute_values to avoid:
        psycopg2.errors.CardinalityViolation
  - Optionally drops GT rows whose window_key is not in scenario_windows for the split
    (default: drop, because those GT rows are not joinable downstream anyway)

Run:
  python scripts/s6/s6f_ingest_gt.py --config configs/s6_db_train650.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import yaml
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

DDL_GT = """
CREATE TABLE IF NOT EXISTS gt_window_truth (
  split_name          TEXT NOT NULL,
  window_key          TEXT NOT NULL,
  log_id              UUID NOT NULL,

  -- from window_labels.jsonl
  window_label        INT,
  label_raw           TEXT,
  label_canonical     TEXT NOT NULL,

  -- from tags_norm.jsonl (matched by (log_id, canonical_label))
  gt_tag_raw          TEXT,
  has_guest           BOOLEAN NOT NULL DEFAULT FALSE,
  guest_id_raw        TEXT,
  guest_track_uuid    UUID,
  gt_event_id         TEXT,

  -- provenance
  source_window_labels_path  TEXT NOT NULL,
  source_tags_norm_path      TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (split_name, window_key)
);

CREATE INDEX IF NOT EXISTS idx_gt_log_id
  ON gt_window_truth (split_name, log_id);

CREATE INDEX IF NOT EXISTS idx_gt_label
  ON gt_window_truth (split_name, label_canonical);

CREATE INDEX IF NOT EXISTS idx_gt_guest
  ON gt_window_truth (split_name, guest_track_uuid);
"""


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_GT)
    conn.commit()


# ---------------- helpers ---------------- #

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonicalize_window_key(log_id: str, t_start: Any, t_end: Any, t_precision: int) -> str:
    """
    Canonical window_key with fixed decimal formatting.
    This MUST match S6A’s formatting.
    """
    ts = round(float(t_start), t_precision)
    te = round(float(t_end), t_precision)
    fmt = f"{{:.{t_precision}f}}"
    return f"{str(log_id).strip()}|{fmt.format(ts)}|{fmt.format(te)}"


def canonicalize_label(lbl: Optional[str]) -> Optional[str]:
    if lbl is None:
        return None
    x = str(lbl).strip()
    if x == "" or x.lower() in {"none", "null", "nan"}:
        return None

    mapping = {
        "right_ped": "ped_crossing",
        "traffic_sign": "approach_stop",
    }
    return mapping.get(x, x)


def try_parse_uuid(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    st = str(s).strip()
    if len(st) != 36:
        return None
    return st


# ---------------- ingest ---------------- #

UPSERT_SQL = """
INSERT INTO gt_window_truth (
  split_name, window_key, log_id,
  window_label, label_raw, label_canonical,
  gt_tag_raw, has_guest, guest_id_raw, guest_track_uuid, gt_event_id,
  source_window_labels_path, source_tags_norm_path,
  updated_at
)
VALUES %s
ON CONFLICT (split_name, window_key) DO UPDATE SET
  log_id = EXCLUDED.log_id,
  window_label = EXCLUDED.window_label,
  label_raw = EXCLUDED.label_raw,
  label_canonical = EXCLUDED.label_canonical,
  gt_tag_raw = EXCLUDED.gt_tag_raw,
  has_guest = EXCLUDED.has_guest,
  guest_id_raw = EXCLUDED.guest_id_raw,
  guest_track_uuid = EXCLUDED.guest_track_uuid,
  gt_event_id = EXCLUDED.gt_event_id,
  source_window_labels_path = EXCLUDED.source_window_labels_path,
  source_tags_norm_path = EXCLUDED.source_tags_norm_path,
  updated_at = NOW();
"""


def build_gt_index(tags_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Index by (log_id, canonical_label) -> "best" GT row.
    Preference: has_guest == true.
    """
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in tags_rows:
        log_id = r.get("log_id")
        tag_raw = r.get("tag")
        if not log_id or not tag_raw:
            continue

        tag_canon = canonicalize_label(tag_raw)
        if tag_canon is None:
            continue

        key = (str(log_id), str(tag_canon))
        cur = idx.get(key)

        has_guest = bool(r.get("has_guest", False))
        if cur is None:
            idx[key] = r
        else:
            cur_has_guest = bool(cur.get("has_guest", False))
            if has_guest and not cur_has_guest:
                idx[key] = r
    return idx


def _pick_better(existing: Tuple, candidate: Tuple) -> Tuple:
    """
    Resolve duplicates for the same (split_name, window_key).

    Policy:
      - prefer rows that have_guest=True
      - then prefer window_label==1 (if provided)
      - else keep candidate (last seen)
    """
    ex_has = bool(existing[7])
    ca_has = bool(candidate[7])
    if ca_has and not ex_has:
        return candidate
    if ex_has and not ca_has:
        return existing

    ex_wl = existing[3]
    ca_wl = candidate[3]
    try:
        ex_pos = 1 if int(ex_wl) == 1 else 0
    except Exception:
        ex_pos = 0
    try:
        ca_pos = 1 if int(ca_wl) == 1 else 0
    except Exception:
        ca_pos = 0

    if ca_pos > ex_pos:
        return candidate
    if ca_pos < ex_pos:
        return existing

    return candidate  # deterministic: last seen


def _fetch_missing_window_keys(conn, split_name: str, window_keys: List[str]) -> Set[str]:
    if not window_keys:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH w AS (
              SELECT DISTINCT UNNEST(%s::text[]) AS window_key
            )
            SELECT w.window_key
            FROM w
            LEFT JOIN scenario_windows sw
              ON sw.window_key = w.window_key
             AND sw.split_name = %s
            WHERE sw.window_key IS NULL;
            """,
            (window_keys, split_name)
        )
        rows = cur.fetchall()
    return {r[0] for r in rows}


def ingest_gt(conn, cfg: Dict[str, Any]) -> None:
    s6f = cfg["s6f"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6f["split_name"])
    tags_path = Path(s6f["tags_norm_jsonl"])
    winlabels_path = Path(s6f["window_labels_jsonl"])

    # IMPORTANT: t_precision must be consistent across S6 stages
    t_precision = int(
        s6f.get("t_precision", cfg.get("runtime", {}).get("window_key", {}).get("t_precision", 6))
    )
    batch_size = int(runtime.get("batch_size", 2000))

    # default behavior: drop rows that do not exist in scenario_windows
    drop_missing_windows = bool(s6f.get("drop_missing_windows", True))

    if not tags_path.exists():
        raise FileNotFoundError(f"tags_norm_jsonl not found: {tags_path}")
    if not winlabels_path.exists():
        raise FileNotFoundError(f"window_labels_jsonl not found: {winlabels_path}")

    logging.info("S6F ingest Ground Truth (window + actor)")
    logging.info(f"  split_name          : {split_name}")
    logging.info(f"  tags_norm_jsonl     : {tags_path}")
    logging.info(f"  window_labels_jsonl : {winlabels_path}")
    logging.info(f"  t_precision         : {t_precision}")
    logging.info(f"  batch_size          : {batch_size}")
    logging.info(f"  drop_missing_windows: {drop_missing_windows}")

    # quick pipeline health
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s", (split_name,))
        (nwin,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {int(nwin)}")

    tags_rows = read_jsonl(tags_path)
    wl_rows = read_jsonl(winlabels_path)

    logging.info(f"Loaded tags_norm.jsonl rows   : {len(tags_rows)}")
    logging.info(f"Loaded window_labels.jsonl rows: {len(wl_rows)}")

    gt_index = build_gt_index(tags_rows)

    # Build + dedupe records
    total = 0
    invalid = 0
    built = 0
    missing_guest_match = 0

    remap_counts: Dict[str, int] = {}

    rec_by_key: Dict[Tuple[str, str], Tuple] = {}
    dup_total = 0
    dup_conflicting = 0

    # track for later missing-window check
    all_wkeys: List[str] = []

    for r in wl_rows:
        total += 1
        log_id = r.get("log_id")
        t_start = r.get("t_start")
        t_end = r.get("t_end")
        lbl_raw = r.get("scenario_label")
        win_label = r.get("label")

        if not log_id or t_start is None or t_end is None or lbl_raw is None:
            invalid += 1
            continue

        lbl_canon = canonicalize_label(lbl_raw)
        if lbl_canon is None:
            invalid += 1
            continue

        if str(lbl_raw) != str(lbl_canon):
            remap_counts[str(lbl_raw)] = remap_counts.get(str(lbl_raw), 0) + 1

        try:
            wkey = canonicalize_window_key(str(log_id), t_start, t_end, t_precision)
        except Exception:
            invalid += 1
            continue

        all_wkeys.append(wkey)

        gt = gt_index.get((str(log_id), str(lbl_canon)))

        gt_tag_raw = None
        has_guest = False
        guest_id_raw = None
        guest_uuid = None
        gt_event_id = None

        if gt is None:
            missing_guest_match += 1
        else:
            gt_tag_raw = gt.get("tag")
            has_guest = bool(gt.get("has_guest", False))
            guest_id_raw = gt.get("guest_id")
            gt_event_id = gt.get("id")
            if has_guest:
                guest_uuid = try_parse_uuid(guest_id_raw)

        rec = (
            split_name, wkey, str(log_id),
            int(win_label) if win_label is not None else None,
            str(lbl_raw), str(lbl_canon),
            (None if gt_tag_raw is None else str(gt_tag_raw)),
            bool(has_guest),
            (None if guest_id_raw is None else str(guest_id_raw)),
            guest_uuid,
            (None if gt_event_id is None else str(gt_event_id)),
            str(winlabels_path), str(tags_path),
        )

        key = (split_name, wkey)
        if key in rec_by_key:
            dup_total += 1
            # conflicting if canonical label differs, or guest differs, etc.
            if rec_by_key[key][5] != rec[5] or rec_by_key[key][9] != rec[9]:
                dup_conflicting += 1
            rec_by_key[key] = _pick_better(rec_by_key[key], rec)
        else:
            rec_by_key[key] = rec

        built += 1

    if remap_counts:
        logging.info("Label remaps (raw -> canonical):")
        for k, v in sorted(remap_counts.items(), key=lambda kv: -kv[1])[:20]:
            logging.info(f"  {k} -> {canonicalize_label(k)} : {v}")

    logging.info("Window-labels summary:")
    logging.info(f"  total_rows            : {total}")
    logging.info(f"  built_records         : {built}")
    logging.info(f"  invalid_rows          : {invalid}")
    logging.info(f"  windows_missing_gt_match : {missing_guest_match}")
    logging.info(f"  unique_gt_rows        : {len(rec_by_key)}")
    logging.info(f"  duplicate_keys        : {dup_total} (conflicting={dup_conflicting})")

    # Sanity: check missing window_keys against scenario_windows
    unique_wkeys = list({k[1] for k in rec_by_key.keys()})
    missing_set = _fetch_missing_window_keys(conn, split_name, unique_wkeys)

    if missing_set:
        logging.warning(
            f"Sanity: {len(missing_set)} GT window_keys do not exist in scenario_windows for split={split_name}."
        )
        ex = sorted(list(missing_set))[:15]
        logging.warning(f"Example missing window_keys (first {len(ex)}):")
        for w in ex:
            logging.warning(f"  {w}")

        if drop_missing_windows:
            before = len(rec_by_key)
            rec_by_key = {k: v for k, v in rec_by_key.items() if k[1] not in missing_set}
            after = len(rec_by_key)
            logging.warning(f"Dropping missing-window GT rows: {before} -> {after}")
    else:
        logging.info("Sanity: all GT window_keys exist in scenario_windows (join OK).")

    records = list(rec_by_key.values())
    if not records:
        logging.warning("No GT records to ingest after filtering. Exiting.")
        return

    # Upsert (now safe: no duplicate (split_name, window_key) in same command)
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(
                cur,
                UPSERT_SQL,
                chunk,
                template="(%s,%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s::uuid,%s,%s,%s,NOW())",
                page_size=min(len(chunk), 1000),
            )
    conn.commit()

    # Post-ingest counts
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gt_window_truth WHERE split_name=%s", (split_name,))
        (count_gt,) = cur.fetchone()
    logging.info(f"gt_window_truth rows_present_for_split : {int(count_gt)}")

    # Optional sanity: guest in tagged actors?
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM gt_window_truth g
            LEFT JOIN scenario_actors a
              ON a.split_name=g.split_name
             AND a.window_key=g.window_key
             AND a.track_uuid=g.guest_track_uuid
            WHERE g.split_name=%s
              AND g.has_guest=TRUE
              AND g.guest_track_uuid IS NOT NULL
              AND a.window_key IS NULL
            """,
            (split_name,)
        )
        (guest_not_tagged,) = cur.fetchone()

    logging.info("Sanity (guest overlap with tagged actors):")
    logging.info(f"  guest_not_in_scenario_actors : {int(guest_not_tagged)}")

    logging.info("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/s6_db_train650.yaml")
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

        ingest_gt(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
