#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S6G - Ingest S5 LLM predictions into PostgreSQL with locked allowlists + S4 evidence mapping.

Reads:
  - YAML config (paths, split, allowlists)
  - .env DB credentials
  - S5 JSON outputs under s5_root/<prompt>/<backend>/*.json
  - S4 evidence JSON files under s4_evidence_root/*.json (named by window_key)

Writes:
  - s4_window_evidence (compact evidence + ACTOR1/2/3 -> track_uuid mapping)
  - llm_backends
  - llm_predictions

Idempotence:
  - s4_window_evidence: PRIMARY KEY(split_name, window_key)
  - llm_backends: UNIQUE(split_name, prompt_type, backend_dir)
  - llm_predictions: PRIMARY KEY(backend_id, window_key) with UPSERT

Run:
  python scripts/s6/s6g_ingest_llm_predictions.py --config configs/s6_db.yaml
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------- helpers ---------------- #

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

def is_uuid(s: Any) -> bool:
    if s is None:
        return False
    return bool(_UUID_RE.match(str(s)))

def to_json_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return json.dumps({"_unserializable": str(x)}, ensure_ascii=False, separators=(",", ":"))

def safe_json_load(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_window_key_from_filename(p: Path) -> Optional[str]:
    stem = p.stem.strip()
    if stem.count("|") == 2:
        return stem
    return None

def canonicalize_window_key(window_key: str, t_precision: int) -> str:
    parts = str(window_key).split("|")
    if len(parts) != 3:
        return str(window_key).strip()
    log_id = parts[0].strip()
    try:
        t_start = round(float(parts[1]), t_precision)
        t_end = round(float(parts[2]), t_precision)
    except Exception:
        return str(window_key).strip()
    fmt = f"{{:.{t_precision}f}}"
    return f"{log_id}|{fmt.format(t_start)}|{fmt.format(t_end)}"

def canonicalize_label(raw: Optional[str], label_map: Dict[str, str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return label_map.get(s, s)


# ---------------- allowlists ---------------- #

@dataclass(frozen=True)
class Allowlists:
    prompt_types: Tuple[str, ...]
    backend_dirs: Tuple[str, ...]


def load_allowlists(cfg: Dict[str, Any]) -> Allowlists:
    s6g = cfg["s6g"]
    prompt_types = tuple(s6g.get("prompt_allowlist", ["base_prompt", "cp", "CoT", "ALLINONE"]))
    backend_dirs = tuple(s6g.get("backend_allowlist", ["llm_gpt-5-chat", "llm_gpt-5-mini", "llm_ollama_gpt-oss"]))
    return Allowlists(prompt_types=prompt_types, backend_dirs=backend_dirs)


# ---------------- DB schema ---------------- #

DDL = """
CREATE TABLE IF NOT EXISTS s4_window_evidence (
  split_name                TEXT NOT NULL,
  window_key                TEXT NOT NULL,
  log_id                    UUID,
  t_on                      DOUBLE PRECISION,
  t_peak                    DOUBLE PRECISION,
  t_off                     DOUBLE PRECISION,
  episode_score_final       DOUBLE PRECISION,
  peak_decel_mps2           DOUBLE PRECISION,

  near_crosswalk            BOOLEAN,
  near_stopline             BOOLEAN,
  has_close_actor           BOOLEAN,
  primary_side              TEXT,
  road_type_hint            TEXT,

  min_dist_to_crosswalk_m   DOUBLE PRECISION,
  min_dist_to_stopline_m    DOUBLE PRECISION,
  num_actors_in_drivable    INT,
  num_actors_off_drivable   INT,
  num_vru_near_crosswalk    INT,
  num_ped_near_crosswalk    INT,

  actor1_track_uuid         UUID,
  actor2_track_uuid         UUID,
  actor3_track_uuid         UUID,

  actor1_s3_score           DOUBLE PRECISION,
  actor2_s3_score           DOUBLE PRECISION,
  actor3_s3_score           DOUBLE PRECISION,

  actor1_category           TEXT,
  actor2_category           TEXT,
  actor3_category           TEXT,

  source_path               TEXT NOT NULL,
  created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (split_name, window_key)
);

CREATE TABLE IF NOT EXISTS llm_backends (
  backend_id      BIGSERIAL PRIMARY KEY,
  split_name      TEXT NOT NULL,
  prompt_type     TEXT NOT NULL,
  backend_dir     TEXT NOT NULL,
  model_name      TEXT NOT NULL,
  provider_hint   TEXT,
  source_root     TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_backends
  ON llm_backends (split_name, prompt_type, backend_dir);

CREATE TABLE IF NOT EXISTS llm_predictions (
  split_name               TEXT NOT NULL,
  backend_id               BIGINT NOT NULL REFERENCES llm_backends(backend_id) ON DELETE CASCADE,
  window_key               TEXT NOT NULL,

  scenario_label_raw       TEXT,
  scenario_label_canonical TEXT,
  confidence_score         DOUBLE PRECISION,

  primary_actor_raw        TEXT,
  primary_track_uuid       UUID,

  actor1_track_uuid        UUID,
  actor2_track_uuid        UUID,
  actor3_track_uuid        UUID,

  actor_matrix_json        JSONB,
  final_rationale          TEXT,
  parsed_result_json       JSONB,
  raw_backend_json         JSONB,

  source_path              TEXT NOT NULL,
  created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (backend_id, window_key)
);

CREATE INDEX IF NOT EXISTS idx_llm_predictions_split_window
  ON llm_predictions (split_name, window_key);
"""

def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


# ---------------- SQL ---------------- #

UPSERT_BACKEND_SQL = """
INSERT INTO llm_backends (split_name, prompt_type, backend_dir, model_name, provider_hint, source_root)
VALUES %s
ON CONFLICT (split_name, prompt_type, backend_dir)
DO UPDATE SET
  provider_hint = EXCLUDED.provider_hint,
  updated_at = NOW()
RETURNING backend_id;
"""

UPSERT_EVIDENCE_SQL = """
INSERT INTO s4_window_evidence (
  split_name, window_key, log_id,
  t_on, t_peak, t_off, episode_score_final, peak_decel_mps2,
  near_crosswalk, near_stopline, has_close_actor, primary_side, road_type_hint,
  min_dist_to_crosswalk_m, min_dist_to_stopline_m,
  num_actors_in_drivable, num_actors_off_drivable, num_vru_near_crosswalk, num_ped_near_crosswalk,
  actor1_track_uuid, actor2_track_uuid, actor3_track_uuid,
  actor1_s3_score, actor2_s3_score, actor3_s3_score,
  actor1_category, actor2_category, actor3_category,
  source_path
)
VALUES %s
ON CONFLICT (split_name, window_key)
DO UPDATE SET
  log_id=EXCLUDED.log_id,
  t_on=EXCLUDED.t_on, t_peak=EXCLUDED.t_peak, t_off=EXCLUDED.t_off,
  episode_score_final=EXCLUDED.episode_score_final,
  peak_decel_mps2=EXCLUDED.peak_decel_mps2,
  near_crosswalk=EXCLUDED.near_crosswalk,
  near_stopline=EXCLUDED.near_stopline,
  has_close_actor=EXCLUDED.has_close_actor,
  primary_side=EXCLUDED.primary_side,
  road_type_hint=EXCLUDED.road_type_hint,
  min_dist_to_crosswalk_m=EXCLUDED.min_dist_to_crosswalk_m,
  min_dist_to_stopline_m=EXCLUDED.min_dist_to_stopline_m,
  num_actors_in_drivable=EXCLUDED.num_actors_in_drivable,
  num_actors_off_drivable=EXCLUDED.num_actors_off_drivable,
  num_vru_near_crosswalk=EXCLUDED.num_vru_near_crosswalk,
  num_ped_near_crosswalk=EXCLUDED.num_ped_near_crosswalk,
  actor1_track_uuid=EXCLUDED.actor1_track_uuid,
  actor2_track_uuid=EXCLUDED.actor2_track_uuid,
  actor3_track_uuid=EXCLUDED.actor3_track_uuid,
  actor1_s3_score=EXCLUDED.actor1_s3_score,
  actor2_s3_score=EXCLUDED.actor2_s3_score,
  actor3_s3_score=EXCLUDED.actor3_s3_score,
  actor1_category=EXCLUDED.actor1_category,
  actor2_category=EXCLUDED.actor2_category,
  actor3_category=EXCLUDED.actor3_category,
  source_path=EXCLUDED.source_path,
  updated_at=NOW();
"""

UPSERT_PRED_SQL = """
INSERT INTO llm_predictions (
  split_name, backend_id, window_key,
  scenario_label_raw, scenario_label_canonical,
  confidence_score,
  primary_actor_raw, primary_track_uuid,
  actor1_track_uuid, actor2_track_uuid, actor3_track_uuid,
  actor_matrix_json, final_rationale,
  parsed_result_json, raw_backend_json,
  source_path
)
VALUES %s
ON CONFLICT (backend_id, window_key)
DO UPDATE SET
  scenario_label_raw       = EXCLUDED.scenario_label_raw,
  scenario_label_canonical = EXCLUDED.scenario_label_canonical,
  confidence_score         = EXCLUDED.confidence_score,
  primary_actor_raw        = EXCLUDED.primary_actor_raw,
  primary_track_uuid       = EXCLUDED.primary_track_uuid,
  actor1_track_uuid        = EXCLUDED.actor1_track_uuid,
  actor2_track_uuid        = EXCLUDED.actor2_track_uuid,
  actor3_track_uuid        = EXCLUDED.actor3_track_uuid,
  actor_matrix_json        = EXCLUDED.actor_matrix_json,
  final_rationale          = EXCLUDED.final_rationale,
  parsed_result_json       = EXCLUDED.parsed_result_json,
  raw_backend_json         = EXCLUDED.raw_backend_json,
  source_path              = EXCLUDED.source_path,
  updated_at               = NOW();
"""


# ---------------- evidence parsing ---------------- #

def extract_top3_mapping(evd: Dict[str, Any]) -> Dict[str, Any]:
    """
    From evidence:
      actors: [{track_uuid, rank_s3, s3_score, category, ...}, ...]
    Build mapping:
      ACTOR1/2/3 -> track_uuid (+ s3_score, category)
    """
    out: Dict[str, Any] = {
        "actor1_track_uuid": None, "actor2_track_uuid": None, "actor3_track_uuid": None,
        "actor1_s3_score": None, "actor2_s3_score": None, "actor3_s3_score": None,
        "actor1_category": None, "actor2_category": None, "actor3_category": None,
    }
    actors = evd.get("actors") or []
    # find rank_s3 1..3
    for a in actors:
        r = a.get("rank_s3")
        if r not in (1, 2, 3):
            continue
        tu = a.get("track_uuid")
        sc = a.get("s3_score")
        cat = a.get("category")
        if r == 1:
            out["actor1_track_uuid"] = tu
            out["actor1_s3_score"] = sc
            out["actor1_category"] = cat
        elif r == 2:
            out["actor2_track_uuid"] = tu
            out["actor2_s3_score"] = sc
            out["actor2_category"] = cat
        elif r == 3:
            out["actor3_track_uuid"] = tu
            out["actor3_s3_score"] = sc
            out["actor3_category"] = cat
    return out


def load_evidence_map(
    evidence_root: Path,
    split_name: str,
    t_precision: int,
    batch_size: int,
    conn
) -> Dict[str, Dict[str, Any]]:
    """
    Load all evidence JSON files under evidence_root and upsert into s4_window_evidence.
    Return mapping: canonical_window_key -> {actor1_track_uuid, actor2_track_uuid, actor3_track_uuid}
    """
    files = sorted(evidence_root.glob("*.json"))
    if not files:
        logging.warning(f"No evidence JSON files found in: {evidence_root}")
        return {}

    records: List[Tuple] = []
    mapping: Dict[str, Dict[str, Any]] = {}

    invalid = 0
    for fp in files:
        try:
            evd = safe_json_load(fp)
            w_raw = evd.get("window_key") or parse_window_key_from_filename(fp)
            if not w_raw:
                invalid += 1
                continue
            wkey = canonicalize_window_key(str(w_raw), t_precision)

            log_id = evd.get("log_id")
            ep = evd.get("episode") or {}
            hints = evd.get("hints") or {}
            mp = evd.get("map") or {}

            top3 = extract_top3_mapping(evd)

            rec = (
                split_name, wkey, log_id,
                ep.get("t_on"), ep.get("t_peak"), ep.get("t_off"),
                ep.get("score_final"), ep.get("peak_decel_mps2"),

                hints.get("near_crosswalk"), hints.get("near_stopline"), hints.get("has_close_actor"),
                hints.get("primary_side"),
                mp.get("road_type_hint"),

                mp.get("min_dist_to_crosswalk_m"), mp.get("min_dist_to_stopline_m"),
                mp.get("num_actors_in_drivable_area"), mp.get("num_actors_off_drivable_area"),
                mp.get("num_vru_near_crosswalk"), mp.get("num_ped_near_crosswalk"),

                top3["actor1_track_uuid"], top3["actor2_track_uuid"], top3["actor3_track_uuid"],
                top3["actor1_s3_score"], top3["actor2_s3_score"], top3["actor3_s3_score"],
                top3["actor1_category"], top3["actor2_category"], top3["actor3_category"],

                str(fp),
            )
            records.append(rec)
            mapping[wkey] = top3
        except Exception:
            invalid += 1
            continue

    # upsert evidence table
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            execute_values(cur, UPSERT_EVIDENCE_SQL, chunk, page_size=min(len(chunk), 1000))
    conn.commit()

    logging.info("Evidence summary:")
    logging.info(f"  evidence_files_total : {len(files)}")
    logging.info(f"  evidence_upserted    : {len(records)}")
    logging.info(f"  evidence_invalid     : {invalid}")

    return mapping


# ---------------- ingest main ---------------- #

def upsert_backend(conn, rec: Tuple) -> int:
    with conn.cursor() as cur:
        execute_values(cur, UPSERT_BACKEND_SQL, [rec], page_size=1)
        backend_id = cur.fetchone()[0]
    conn.commit()
    return int(backend_id)


def resolve_primary_from_evidence(primary_raw: Any, top3: Dict[str, Any]) -> Optional[str]:
    if primary_raw is None:
        return None
    s = str(primary_raw).strip()
    if not s:
        return None
    if is_uuid(s):
        return s

    sU = s.upper()
    if sU == "ACTOR1":
        return top3.get("actor1_track_uuid")
    if sU == "ACTOR2":
        return top3.get("actor2_track_uuid")
    if sU == "ACTOR3":
        return top3.get("actor3_track_uuid")
    return None


def ingest_llm(conn, cfg: Dict[str, Any]) -> None:
    s6g = cfg["s6g"]
    runtime = cfg.get("runtime", {})

    split_name = str(s6g["split_name"])
    s5_root = Path(s6g["s5_root"])
    evidence_root = Path(s6g["s4_evidence_root"])
    t_precision = int(s6g.get("t_precision", 6))
    batch_size = int(runtime.get("batch_size", 2000))

    allow = load_allowlists(cfg)
    label_map = s6g.get("label_map", {}) or {}

    if not s5_root.exists():
        raise FileNotFoundError(f"s5_root not found: {s5_root}")
    if not evidence_root.exists():
        raise FileNotFoundError(f"s4_evidence_root not found: {evidence_root}")

    logging.info("S6G ingest LLM predictions (allowlists + evidence mapping)")
    loggingA = list(allow.prompt_types)
    B = list(allow.backend_dirs)
    R = list(allow.prompt_types)
    logging.info(f"  split_name       : {split_name}")
    logging.info(f"  s5_root          : {s5_root}")
    logging.info(f"  s4_evidence_root : {evidence_root}")
    logging.info(f"  t_precision      : {t_precision}")
    logging.info(f"  batch_size       : {batch_size}")
    logging.info(f"  prompt_allowlist : {R}")
    logging.info(f"  backend_allowlist: {B}")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM scenario_windows WHERE split_name=%s", (split_name,))
        (nwin,) = cur.fetchone()
    logging.info(f"DB check: scenario_windows rows for split = {int(nwin)}")

    # Load evidence + build per-window mapping
    evidence_map = load_evidence_map(evidence_root, split_name, t_precision, batch_size, conn)

    total_json = 0
    ingested = 0
    invalid = 0
    missing_window = 0
    missing_evidence = 0
    unresolved_primary = 0

    for prompt_dir in sorted([d for d in s5_root.iterdir() if d.is_dir() and d.name in allow.prompt_types]):
        prompt_type = prompt_dir.name

        for backend_dir in sorted([d for d in prompt_dir.iterdir() if d.is_dir() and d.name in allow.backend_dirs]):
            backend_dir_name = backend_dir.name
            model_name = backend_dir_name.replace("llm_", "")

            json_files = sorted(backend_dir.glob("*.json"))
            if not json_files:
                continue

            provider_hint = None
            try:
                sample = safe_json_load(json_files[0])
                provider_hint = sample.get("backend")
            except Exception:
                provider_hint = None

            backend_id = upsert_backend(
                conn,
                (split_name, prompt_type, backend_dir_name, model_name, provider_hint, str(backend_dir)),
            )
            logging.info(f"[{prompt_type} / {backend_dir_name}] backend_id={backend_id} json_files={len(json_files)}")

            pred_records: List[Tuple] = []

            for jf in json_files:
                total_json += 1
                try:
                    data = safe_json_load(jf)
                    parsed = data.get("parsed_result") if isinstance(data.get("parsed_result"), dict) else None

                    # window_key extraction
                    w_raw = None
                    if parsed and parsed.get("ego_window_key"):
                        w_raw = parsed.get("ego_window_key")
                    if not w_raw:
                        w_raw = parse_window_key_from_filename(jf)
                    if not w_raw:
                        invalid += 1
                        continue

                    wkey = canonicalize_window_key(str(w_raw), t_precision)

                    # ensure window exists
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT 1 FROM scenario_windows WHERE split_name=%s AND window_key=%s LIMIT 1",
                            (split_name, wkey),
                        )
                        ok = cur.fetchone() is not None
                    if not ok:
                        missing_window += 1
                        continue

                    top3 = evidence_map.get(wkey)
                    if top3 is None:
                        missing_evidence += 1
                        top3 = {"actor1_track_uuid": None, "actor2_track_uuid": None, "actor3_track_uuid": None}

                    scenario_raw = parsed.get("scenario_classification") if parsed else None
                    conf = parsed.get("confidence_score") if parsed else None
                    primary_raw = parsed.get("primary_responsible_actor") if parsed else None
                    actor_matrix = parsed.get("actor_matrix") if parsed else None
                    rationale = parsed.get("final_rationale") if parsed else None

                    scenario_canon = canonicalize_label(scenario_raw, label_map)

                    primary_uuid = resolve_primary_from_evidence(primary_raw, top3)
                    if primary_raw is not None and str(primary_raw).strip() and primary_uuid is None:
                        unresolved_primary += 1

                    # jsonb fields MUST be text (psycopg2 can't adapt dict/list in execute_values)
                    actor_matrix_jsons = to_json_str(actor_matrix)
                    parsed_jsons = to_json_str(parsed)
                    raw_backend_jsons = to_json_str({"backend": data.get("backend"), "file": jf.name})

                    pred_records.append((
                        split_name, backend_id, wkey,
                        (None if scenario_raw is None else str(scenario_raw)),
                        (None if scenario_canon is None else str(scenario_canon)),
                        (None if conf is None else float(conf)),
                        (None if primary_raw is None else str(primary_raw)),
                        (None if primary_uuid is None else primary_uuid),

                        top3.get("actor1_track_uuid"), top3.get("actor2_track_uuid"), top3.get("actor3_track_uuid"),

                        actor_matrix_jsons,
                        (None if rationale is None else str(rationale)),
                        parsed_jsons,
                        raw_backend_jsons,
                        str(jf),
                    ))

                    ingested += 1

                except Exception:
                    invalid += 1
                    continue

            if pred_records:
                with conn.cursor() as cur:
                    for i in range(0, len(pred_records), batch_size):
                        chunk = pred_records[i:i + batch_size]
                        execute_values(
                            cur,
                            UPSERT_PRED_SQL,
                            chunk,
                            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s)",
                            page_size=min(len(chunk), 1000),
                        )
                conn.commit()

    # Summary
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM llm_backends WHERE split_name=%s", (split_name,))
        (nb,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM llm_predictions WHERE split_name=%s", (split_name,))
        (npred,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM s4_window_evidence WHERE split_name=%s", (split_name,))
        (nev,) = cur.fetchone()

    logging.info("S6G summary:")
    logging.info(f"  evidence_rows              : {int(nev)}")
    logging.info(f"  total_json_scanned         : {total_json}")
    logging.info(f"  ingested_predictions       : {ingested}")
    logging.info(f"  invalid_json               : {invalid}")
    logging.info(f"  missing_window             : {missing_window}")
    logging.info(f"  missing_evidence_for_window: {missing_evidence}")
    logging.info(f"  unresolved_primary_actor   : {unresolved_primary}")
    logging.info(f"  llm_backends_rows          : {int(nb)}")
    logging.info(f"  llm_predictions_rows       : {int(npred)}")
    logging.info("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/s6_db.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    db_params = load_db_env(Path(cfg["db"]["env_file"]))

    logging.info("Connecting to DB...")
    conn = psycopg2.connect(**db_params)
    conn.autocommit = False

    try:
        if bool(cfg.get("runtime", {}).get("create_tables", True)):
            logging.info("Ensuring tables exist...")
            ensure_tables(conn)

        ingest_llm(conn, cfg)

    finally:
        conn.close()
        logging.info("DB connection closed.")


if __name__ == "__main__":
    main()
