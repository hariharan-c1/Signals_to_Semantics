#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6B: Ingest tagged actors + per-actor feature rows into PostgreSQL.

Fix in this version:
- val50_features.parquet may NOT contain window_key.
  If missing, we construct it from (log_id, window_t_start, window_t_end)
  using the same canonical formatting rules as S6A.

Idempotence:
- PRIMARY KEY (split_name, window_key, track_uuid)
- ON CONFLICT DO NOTHING
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ----------------------------
# logging
# ----------------------------

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}")


# ----------------------------
# config
# ----------------------------

@dataclass
class RuntimeCfg:
    batch_size: int
    create_tables: bool
    t_precision: int


@dataclass
class DbCfg:
    env_file: Path


@dataclass
class S6BCfg:
    tagged_jsonl: Path
    features_parquet: Path


@dataclass
class FullCfg:
    db: DbCfg
    runtime: RuntimeCfg
    s6a_split_name: str
    s6b: S6BCfg


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg_path: Path) -> FullCfg:
    raw = load_yaml(cfg_path)

    env_file = Path(raw["db"]["env_file"])

    runtime = raw.get("runtime", {})
    batch_size = int(runtime.get("batch_size", 2000))
    create_tables = bool(runtime.get("create_tables", True))
    wk = runtime.get("window_key", {}) or {}
    t_precision = int(wk.get("t_precision", 6))

    split_name = str(raw["s6a_windows"]["split_name"])

    s6b = raw["s6b"]
    tagged_jsonl = Path(s6b["tagged_jsonl"])
    features_parquet = Path(s6b["features_parquet"])

    return FullCfg(
        db=DbCfg(env_file=env_file),
        runtime=RuntimeCfg(batch_size=batch_size, create_tables=create_tables, t_precision=t_precision),
        s6a_split_name=split_name,
        s6b=S6BCfg(tagged_jsonl=tagged_jsonl, features_parquet=features_parquet),
    )


# ----------------------------
# DB connection
# ----------------------------

def connect_db(env_file: Path):
    if not env_file.exists():
        raise FileNotFoundError(f"DB env_file not found: {env_file}")

    load_dotenv(env_file, override=True)

    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing DB env vars in {env_file}: {missing}")

    host = os.environ["DB_HOST"]
    port = int(os.environ["DB_PORT"])
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
    conn.autocommit = False
    return conn


# ----------------------------
# window_key formatting
# ----------------------------

def fmt_t(x: Any, t_precision: int) -> str:
    if x is None:
        return "nan"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if t_precision <= 0:
        return str(int(round(v)))
    s = f"{round(v, t_precision):.{t_precision}f}"
    return s.rstrip("0").rstrip(".")


def build_window_key(log_id: str, t_start: Any, t_end: Any, t_precision: int) -> str:
    return f"{log_id}|{fmt_t(t_start, t_precision)}|{fmt_t(t_end, t_precision)}"


# ----------------------------
# table creation
# ----------------------------

DDL_SCENARIO_ACTORS = """
CREATE TABLE IF NOT EXISTS scenario_actors (
  split_name TEXT NOT NULL,
  window_key TEXT NOT NULL,
  track_uuid UUID NOT NULL,

  category TEXT,
  tag_rank SMALLINT,

  p_overlap DOUBLE PRECISION,
  dmin_m DOUBLE PRECISION,
  penetration_m DOUBLE PRECISION,
  t_at_dmin_s DOUBLE PRECISION,
  sustained_tight BOOLEAN,
  a_norm DOUBLE PRECISION,
  length_m DOUBLE PRECISION,
  width_m DOUBLE PRECISION,
  is_static SMALLINT,
  on_path_like SMALLINT,
  bearing_rad DOUBLE PRECISION,
  rel_speed_closing_mps DOUBLE PRECISION,
  sector_id SMALLINT,
  x_rel DOUBLE PRECISION,
  y_rel DOUBLE PRECISION,
  v_lat_ego_mps DOUBLE PRECISION,
  lat_conv DOUBLE PRECISION,
  is_vehicle SMALLINT,
  is_vru SMALLINT,

  actor_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (split_name, window_key, track_uuid)
);
"""

DDL_ACTOR_FEATURES = """
CREATE TABLE IF NOT EXISTS actor_features (
  split_name TEXT NOT NULL,
  window_key TEXT NOT NULL,
  track_uuid UUID NOT NULL,

  category TEXT,

  is_vehicle SMALLINT,
  is_vru SMALLINT,
  is_static SMALLINT,
  sector_id SMALLINT,
  on_path_like SMALLINT,

  x_rel_m DOUBLE PRECISION,
  y_rel_m DOUBLE PRECISION,
  r_rel_m DOUBLE PRECISION,
  dmin_m DOUBLE PRECISION,
  p_overlap DOUBLE PRECISION,
  t_at_dmin_s DOUBLE PRECISION,
  sustained_tight DOUBLE PRECISION,
  rel_speed_closing_mps DOUBLE PRECISION,
  lat_speed_mps DOUBLE PRECISION,
  a_norm DOUBLE PRECISION,
  bearing_rad DOUBLE PRECISION,
  ttc_s DOUBLE PRECISION,

  dist_to_stopline_m DOUBLE PRECISION,
  dist_to_crosswalk_m DOUBLE PRECISION,
  map_lane_offset_m DOUBLE PRECISION,
  map_lane_alignment_cos DOUBLE PRECISION,

  rank_p_overlap DOUBLE PRECISION,
  rank_inv_dmin DOUBLE PRECISION,

  feature_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  PRIMARY KEY (split_name, window_key, track_uuid)
);
"""


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_SCENARIO_ACTORS)
        cur.execute(DDL_ACTOR_FEATURES)
    conn.commit()


# ----------------------------
# helpers
# ----------------------------

def jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def count_rows(conn, table: str, split_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE split_name = %s;", (split_name,))
        return int(cur.fetchone()[0])


def window_keys_missing_in_windows(conn, split_name: str, window_keys: List[str]) -> int:
    if not window_keys:
        return 0
    uniq = sorted(set(window_keys))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT UNNEST(%s::text[]) AS window_key
            ) x
            LEFT JOIN scenario_windows w
              ON w.split_name = %s AND w.window_key = x.window_key
            WHERE w.window_key IS NULL;
            """,
            (uniq, split_name),
        )
        return int(cur.fetchone()[0])


# ----------------------------
# ingest scenario_actors from tagged_jsonl
# ----------------------------

def build_actor_records(split_name: str, tagged_jsonl: Path, t_precision: int) -> Tuple[List[Tuple], Dict[str, int]]:
    records: List[Tuple] = []
    stats = {"total_windows": 0, "total_actor_rows": 0, "invalid_rows": 0, "missing_top_actors": 0}

    for obj in jsonl_iter(tagged_jsonl):
        stats["total_windows"] += 1
        log_id = safe_get(obj, "log_id")
        t_start = safe_get(obj, "t_start")
        t_end = safe_get(obj, "t_end")

        if not log_id or t_start is None or t_end is None:
            stats["invalid_rows"] += 1
            continue

        wkey = build_window_key(str(log_id), t_start, t_end, t_precision)

        top_actors = safe_get(obj, "top_actors", [])
        if not isinstance(top_actors, list) or len(top_actors) == 0:
            stats["missing_top_actors"] += 1
            continue

        for idx, a in enumerate(top_actors, start=1):
            track_uuid = safe_get(a, "track_uuid")
            if not track_uuid:
                continue

            rec = (
                split_name,
                wkey,
                track_uuid,
                safe_get(a, "category"),
                idx,
                safe_get(a, "p_overlap"),
                safe_get(a, "dmin_m"),
                safe_get(a, "penetration_m"),
                safe_get(a, "t_at_dmin_s"),
                bool(safe_get(a, "sustained_tight")) if safe_get(a, "sustained_tight") is not None else None,
                safe_get(a, "a_norm"),
                safe_get(a, "length_m"),
                safe_get(a, "width_m"),
                safe_get(a, "is_static"),
                safe_get(a, "on_path_like"),
                safe_get(a, "bearing_rad"),
                safe_get(a, "rel_speed_closing_mps"),
                safe_get(a, "sector_id"),
                safe_get(a, "x_rel"),
                safe_get(a, "y_rel"),
                safe_get(a, "v_lat_ego_mps"),
                safe_get(a, "lat_conv"),
                safe_get(a, "is_vehicle"),
                safe_get(a, "is_vru"),
                json.dumps(a),
            )
            records.append(rec)
            stats["total_actor_rows"] += 1

    return records, stats


def insert_scenario_actors(conn, records: List[Tuple], batch_size: int) -> int:
    if not records:
        return 0

    sql = """
    INSERT INTO scenario_actors (
      split_name, window_key, track_uuid,
      category, tag_rank,
      p_overlap, dmin_m, penetration_m, t_at_dmin_s, sustained_tight,
      a_norm, length_m, width_m, is_static, on_path_like,
      bearing_rad, rel_speed_closing_mps, sector_id, x_rel, y_rel,
      v_lat_ego_mps, lat_conv, is_vehicle, is_vru,
      actor_json
    ) VALUES %s
    ON CONFLICT (split_name, window_key, track_uuid) DO NOTHING;
    """

    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            execute_values(cur, sql, chunk, page_size=min(len(chunk), 1000))
    conn.commit()
    return len(records)


# ----------------------------
# ingest actor_features from parquet
# ----------------------------

def build_feature_records(
    split_name: str, features_parquet: Path, t_precision: int
) -> Tuple[List[Tuple], Dict[str, int]]:
    df = pd.read_parquet(features_parquet)
    stats = {"total_rows": int(len(df)), "invalid_rows": 0, "built_records": 0, "window_key_built": 0}

    required_base = {"track_uuid"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"features parquet missing required columns: {sorted(missing_base)}")

    # Build window_key if missing
    if "window_key" not in df.columns:
        needed = {"log_id", "window_t_start", "window_t_end"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(
                f"features parquet missing 'window_key' and also missing columns needed to construct it: {sorted(missing)}"
            )

        log("features parquet has no window_key; constructing window_key from (log_id, window_t_start, window_t_end)...")
        df["window_key"] = df.apply(
            lambda r: build_window_key(str(r["log_id"]), r["window_t_start"], r["window_t_end"], t_precision),
            axis=1,
        )
        stats["window_key_built"] = int(len(df))

    # ---- JSON-safe sanitization ----
    def sanitize(v):
        # pandas NA / numpy NaN / None -> None (JSON null)
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        # convert numpy scalars to python scalars if present
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass

        # handle pandas Timestamp
        if isinstance(v, pd.Timestamp):
            return v.isoformat()

        return v

    # normalize missing to None for regular columns too
    df = df.where(pd.notnull(df), None)

    def col(row, c):
        return sanitize(row[c]) if c in df.columns else None

    records: List[Tuple] = []
    for _, row in df.iterrows():
        wkey = row["window_key"]
        tuid = row["track_uuid"]
        if not wkey or not tuid:
            stats["invalid_rows"] += 1
            continue

        # build JSONB payload but guarantee strict JSON (no NaN/Inf)
        feature_dict = {}
        for c in df.columns:
            if c in ("split_name", "window_key", "track_uuid"):
                continue
            feature_dict[c] = sanitize(row[c])

        try:
            feature_json = json.dumps(feature_dict, allow_nan=False)
        except ValueError as e:
            # pinpoint offender
            bad = [(k, feature_dict[k]) for k in feature_dict.keys() if str(feature_dict[k]) in ("nan", "inf", "-inf")]
            raise ValueError(f"feature_json contains non-JSON values. Examples: {bad[:5]}") from e

        rec = (
            split_name,
            wkey,
            tuid,
            col(row, "category"),
            col(row, "is_vehicle"),
            col(row, "is_vru"),
            col(row, "is_static"),
            col(row, "sector_id"),
            col(row, "on_path_like"),
            col(row, "x_rel_m"),
            col(row, "y_rel_m"),
            col(row, "r_rel_m"),
            col(row, "dmin_m"),
            col(row, "p_overlap"),
            col(row, "t_at_dmin_s"),
            col(row, "sustained_tight"),
            col(row, "rel_speed_closing_mps"),
            col(row, "lat_speed_mps"),
            col(row, "a_norm"),
            col(row, "bearing_rad"),
            col(row, "ttc_s"),
            col(row, "dist_to_stopline_m"),
            col(row, "dist_to_crosswalk_m"),
            col(row, "map_lane_offset_m"),
            col(row, "map_lane_alignment_cos"),
            col(row, "rank_p_overlap"),
            col(row, "rank_inv_dmin"),
            feature_json,
        )

        records.append(rec)
        stats["built_records"] += 1

    return records, stats


def insert_actor_features(conn, records: List[Tuple], batch_size: int) -> int:
    if not records:
        return 0

    sql = """
    INSERT INTO actor_features (
      split_name, window_key, track_uuid,
      category,
      is_vehicle, is_vru, is_static, sector_id, on_path_like,
      x_rel_m, y_rel_m, r_rel_m, dmin_m, p_overlap, t_at_dmin_s, sustained_tight,
      rel_speed_closing_mps, lat_speed_mps, a_norm, bearing_rad, ttc_s,
      dist_to_stopline_m, dist_to_crosswalk_m, map_lane_offset_m, map_lane_alignment_cos,
      rank_p_overlap, rank_inv_dmin,
      feature_json
    ) VALUES %s
    ON CONFLICT (split_name, window_key, track_uuid) DO NOTHING;
    """

    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            execute_values(cur, sql, chunk, page_size=min(len(chunk), 1000))
    conn.commit()
    return len(records)


# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = parse_config(Path(args.config))

    split_name = cfg.s6a_split_name
    tagged_jsonl = cfg.s6b.tagged_jsonl
    features_parquet = cfg.s6b.features_parquet
    t_precision = cfg.runtime.t_precision
    batch_size = cfg.runtime.batch_size

    log("Connecting to DB...")
    conn = connect_db(cfg.db.env_file)

    try:
        if cfg.runtime.create_tables:
            log("Ensuring tables exist...")
            ensure_tables(conn)

        log("S6B ingest actors + features")
        log(f"  split_name        : {split_name}")
        log(f"  tagged_jsonl      : {tagged_jsonl}")
        log(f"  features_parquet  : {features_parquet}")
        log(f"  t_precision       : {t_precision}")
        log(f"  batch_size        : {batch_size}")

        # sanity: scenario_windows exists for split
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name = %s;", (split_name,))
            n_windows = int(cur.fetchone()[0])
        log(f"DB check: scenario_windows rows for split = {n_windows}")
        if n_windows == 0:
            raise RuntimeError("Missing scenario_windows rows; run S6A first.")

        # counts before
        before_actors = count_rows(conn, "scenario_actors", split_name)
        before_feats = count_rows(conn, "actor_features", split_name)

        # 1) scenario_actors
        actor_records, actor_stats = build_actor_records(split_name, tagged_jsonl, t_precision)
        log("Tagged JSONL summary:")
        for k, v in actor_stats.items():
            log(f"  {k:>18} : {v}")

        missing_actor_wk = window_keys_missing_in_windows(conn, split_name, [r[1] for r in actor_records])
        if missing_actor_wk > 0:
            log(f"[WARN] scenario_actors: {missing_actor_wk} distinct window_keys missing in scenario_windows", "WARN")

        attempted_actors = insert_scenario_actors(conn, actor_records, batch_size)
        after_actors = count_rows(conn, "scenario_actors", split_name)
        inserted_actors = after_actors - before_actors
        skipped_actors = max(0, attempted_actors - inserted_actors)

        log("scenario_actors ingest result:")
        log(f"  attempted_rows : {attempted_actors}")
        log(f"  inserted_rows  : {inserted_actors}")
        log(f"  skipped_rows   : {skipped_actors} (duplicates / conflicts)")

        # 2) actor_features
        feat_records, feat_stats = build_feature_records(split_name, features_parquet, t_precision)
        log("Features parquet summary:")
        for k, v in feat_stats.items():
            log(f"  {k:>18} : {v}")

        missing_feat_wk = window_keys_missing_in_windows(conn, split_name, [r[1] for r in feat_records])
        if missing_feat_wk > 0:
            log(f"[WARN] actor_features: {missing_feat_wk} distinct window_keys missing in scenario_windows", "WARN")

        attempted_feats = insert_actor_features(conn, feat_records, batch_size)
        after_feats = count_rows(conn, "actor_features", split_name)
        inserted_feats = after_feats - before_feats
        skipped_feats = max(0, attempted_feats - inserted_feats)

        log("actor_features ingest result:")
        log(f"  attempted_rows : {attempted_feats}")
        log(f"  inserted_rows  : {inserted_feats}")
        log(f"  skipped_rows   : {skipped_feats} (duplicates / conflicts)")

        log("Done.")

    finally:
        try:
            conn.close()
            log("DB connection closed.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
