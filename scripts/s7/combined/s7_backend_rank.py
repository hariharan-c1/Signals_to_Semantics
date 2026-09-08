#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S7X - Backend ranking (GT alignment) across all LLM backends.

Outputs:
  1) backend_ranking_backend_id.csv  (24 rows: per split_name + backend_id)
  2) backend_ranking_logical_backend.csv (12 rows: per (prompt_type, backend_dir), combined + per-split)
  3) backend_ranking_summary.json (light summary + "winners")

Metrics (computed on rows with GT present):
  - accuracy
  - macro_precision / macro_recall / macro_f1   (macro over labels with GT support > 0)
  - weighted_f1 (weighted by label support)
  - support counts: n_gt_rows, n_labels_gt
  - coverage: n_total_rows_in_view, gt_coverage_frac

Assumptions:
  - v_scenario_trace includes: split_name, backend_id, gt_label_canonical, llm_label_canonical
  - llm_backends includes: backend_id, split_name, prompt_type, backend_dir
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


# ----------------------------
# DB
# ----------------------------
def connect_db_from_env(env_file: Path) -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path=str(env_file), override=True)
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing DB env vars in {env_file}: {missing}")

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


# ----------------------------
# Metrics
# ----------------------------
@dataclass
class ClassStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0  # GT count for this class

def safe_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s and s.lower() != "none" else None

def compute_metrics(y_true: List[Optional[str]], y_pred: List[Optional[str]]) -> Dict[str, Any]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "n_gt_rows": 0,
            "n_labels_gt": 0,
        }

    labels = sorted({t for t in y_true if t is not None})
    stats = {lab: ClassStats() for lab in labels}

    correct = 0
    for t, p in zip(y_true, y_pred):
        t = safe_label(t)
        p = safe_label(p)

        if t is None:
            continue  # should not happen since we filter to GT-present rows, but keep safe.

        stats[t].support += 1
        if p == t:
            correct += 1
            stats[t].tp += 1
        else:
            stats[t].fn += 1
            if p is not None and p in stats:
                stats[p].fp += 1
            # if p is None or unseen label, we don't assign fp to any known GT label

    acc = correct / float(n) if n > 0 else np.nan

    per_p, per_r, per_f1, supports = [], [], [], []
    for lab in labels:
        s = stats[lab]
        # Precision: tp/(tp+fp) ; Recall: tp/(tp+fn)
        prec = (s.tp / (s.tp + s.fp)) if (s.tp + s.fp) > 0 else np.nan
        rec  = (s.tp / (s.tp + s.fn)) if (s.tp + s.fn) > 0 else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not np.nan and rec is not np.nan and (prec + rec) > 0) else np.nan

        per_p.append(prec)
        per_r.append(rec)
        per_f1.append(f1)
        supports.append(s.support)

    macro_p = float(np.nanmean(per_p)) if len(per_p) else np.nan
    macro_r = float(np.nanmean(per_r)) if len(per_r) else np.nan
    macro_f1 = float(np.nanmean(per_f1)) if len(per_f1) else np.nan

    # Weighted F1 (by GT support)
    supports_arr = np.asarray(supports, dtype=float)
    f1_arr = np.asarray(per_f1, dtype=float)
    mask = np.isfinite(f1_arr) & (supports_arr > 0)
    weighted_f1 = float(np.sum(f1_arr[mask] * supports_arr[mask]) / np.sum(supports_arr[mask])) if mask.any() else np.nan

    return {
        "accuracy": float(acc),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_gt_rows": int(n),
        "n_labels_gt": int(len(labels)),
    }


# ----------------------------
# Fetch + Rank
# ----------------------------
def fetch_trace_rows(conn, view_name: str) -> pd.DataFrame:
    """
    Pull minimal required columns from v_scenario_trace and join backend metadata.
    We do NOT filter to a single split here: this script ranks across all splits present.
    """
    q = f"""
    SELECT
      t.split_name,
      t.backend_id,
      b.prompt_type,
      b.backend_dir,
      t.gt_label_canonical,
      t.llm_label_canonical
    FROM {view_name} t
    JOIN llm_backends b ON b.backend_id = t.backend_id
    ;
    """
    return pd.read_sql(q, conn)

def build_backend_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Total rows in the view per backend (coverage context)
    total_counts = (
        df.groupby(["split_name", "backend_id", "prompt_type", "backend_dir"])
          .size()
          .reset_index(name="n_total_rows_in_view")
    )

    # GT-only subset for metrics
    dfg = df[df["gt_label_canonical"].notna()].copy()

    rows = []
    for (split_name, backend_id, prompt_type, backend_dir), g in dfg.groupby(["split_name", "backend_id", "prompt_type", "backend_dir"]):
        y_true = [safe_label(x) for x in g["gt_label_canonical"].tolist()]
        y_pred = [safe_label(x) for x in g["llm_label_canonical"].tolist()]
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "split_name": split_name,
            "backend_id": int(backend_id),
            "prompt_type": prompt_type,
            "backend_dir": backend_dir,
            **m,
        })

    df_backend = pd.DataFrame(rows)

    # Merge coverage info
    df_backend = df_backend.merge(
        total_counts,
        on=["split_name", "backend_id", "prompt_type", "backend_dir"],
        how="left",
    )
    df_backend["gt_coverage_frac"] = df_backend["n_gt_rows"] / df_backend["n_total_rows_in_view"].replace({0: np.nan})
    df_backend["gt_coverage_frac"] = df_backend["gt_coverage_frac"].astype(float)

    # Sort: primary by macro_f1 desc, then coverage desc, then n_gt desc
    df_backend = df_backend.sort_values(
        by=["macro_f1", "gt_coverage_frac", "n_gt_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Logical-backend aggregation (12 rows): group by (prompt_type, backend_dir)
    # We'll compute combined metrics by concatenating y_true/y_pred across splits.
    logical_rows = []
    for (prompt_type, backend_dir), g in dfg.groupby(["prompt_type", "backend_dir"]):
        y_true_all = [safe_label(x) for x in g["gt_label_canonical"].tolist()]
        y_pred_all = [safe_label(x) for x in g["llm_label_canonical"].tolist()]
        m_all = compute_metrics(y_true_all, y_pred_all)

        # Per-split metrics (helpful for “winner per split” later)
        per_split = {}
        for split_name, gs in g.groupby("split_name"):
            yt = [safe_label(x) for x in gs["gt_label_canonical"].tolist()]
            yp = [safe_label(x) for x in gs["llm_label_canonical"].tolist()]
            ms = compute_metrics(yt, yp)
            per_split[split_name] = ms

        # Coverage counts per split (total rows in view)
        gtot = df[(df["prompt_type"] == prompt_type) & (df["backend_dir"] == backend_dir)].copy()
        tot_by_split = gtot.groupby("split_name").size().to_dict()
        gt_by_split = g.groupby("split_name").size().to_dict()

        logical_rows.append({
            "prompt_type": prompt_type,
            "backend_dir": backend_dir,

            # combined
            "macro_f1_combined": m_all["macro_f1"],
            "macro_precision_combined": m_all["macro_precision"],
            "macro_recall_combined": m_all["macro_recall"],
            "accuracy_combined": m_all["accuracy"],
            "weighted_f1_combined": m_all["weighted_f1"],
            "n_gt_rows_combined": m_all["n_gt_rows"],
            "n_labels_gt_combined": m_all["n_labels_gt"],

            # per split (fill missing safely)
            "macro_f1_val50": per_split.get("val50", {}).get("macro_f1", np.nan),
            "macro_f1_train650": per_split.get("train650", {}).get("macro_f1", np.nan),
            "n_gt_rows_val50": per_split.get("val50", {}).get("n_gt_rows", 0),
            "n_gt_rows_train650": per_split.get("train650", {}).get("n_gt_rows", 0),

            "n_total_rows_val50": int(tot_by_split.get("val50", 0)),
            "n_total_rows_train650": int(tot_by_split.get("train650", 0)),
            "gt_coverage_frac_val50": (gt_by_split.get("val50", 0) / float(tot_by_split.get("val50", 1))) if tot_by_split.get("val50", 0) > 0 else np.nan,
            "gt_coverage_frac_train650": (gt_by_split.get("train650", 0) / float(tot_by_split.get("train650", 1))) if tot_by_split.get("train650", 0) > 0 else np.nan,
        })

    df_logical = pd.DataFrame(logical_rows).sort_values(
        by=["macro_f1_combined", "n_gt_rows_combined"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Winners (best backend_id per split based on macro_f1)
    winners = {}
    for split_name, gs in df_backend.groupby("split_name"):
        if len(gs) == 0:
            continue
        best = gs.iloc[0].to_dict()
        winners[split_name] = {
            "backend_id": int(best["backend_id"]),
            "prompt_type": best["prompt_type"],
            "backend_dir": best["backend_dir"],
            "macro_f1": float(best["macro_f1"]),
            "n_gt_rows": int(best["n_gt_rows"]),
            "gt_coverage_frac": float(best["gt_coverage_frac"]) if pd.notna(best["gt_coverage_frac"]) else None,
        }

    summary = {
        "n_rows_in_view_total": int(len(df)),
        "n_rows_with_gt_total": int(df["gt_label_canonical"].notna().sum()),
        "n_backends_found": int(df_backend[["split_name", "backend_id"]].drop_duplicates().shape[0]),
        "n_logical_backends_found": int(df_logical[["prompt_type", "backend_dir"]].drop_duplicates().shape[0]),
        "winners_per_split": winners,
    }

    return df_backend, df_logical, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default="configs/db.env", help="Path to db.env with DB_HOST/PORT/NAME/USER/PASSWORD")
    ap.add_argument("--view_name", default="v_scenario_trace", help="Retrieval trace view name")
    ap.add_argument("--out_dir", default="artifacts/backend_rank", help="Output directory for CSV/JSON")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Connecting to DB...")
    conn = connect_db_from_env(Path(args.env_file))
    try:
        print(f"[INFO] Fetching rows from view: {args.view_name}")
        df = fetch_trace_rows(conn, args.view_name)

        if df.empty:
            raise RuntimeError("No rows returned from v_scenario_trace. Check that S7A view exists and has rows.")

        df_backend, df_logical, summary = build_backend_tables(df)

        out1 = out_dir / "backend_ranking_backend_id.csv"
        out2 = out_dir / "backend_ranking_logical_backend.csv"
        out3 = out_dir / "backend_ranking_summary.json"

        df_backend.to_csv(out1, index=False)
        df_logical.to_csv(out2, index=False)
        with out3.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("[INFO] Wrote:")
        print(f"  - {out1}")
        print(f"  - {out2}")
        print(f"  - {out3}")

        print("\n[INFO] Quick sanity:")
        print(f"  total rows in view      : {summary['n_rows_in_view_total']}")
        print(f"  total rows with GT      : {summary['n_rows_with_gt_total']}")
        print(f"  backends found (split+id): {summary['n_backends_found']}")
        print(f"  logical backends found   : {summary['n_logical_backends_found']}")

        print("\n[INFO] Winners per split (by macro_f1):")
        for split_name, w in summary["winners_per_split"].items():
            print(f"  - {split_name}: backend_id={w['backend_id']} ({w['prompt_type']} + {w['backend_dir']}), "
                  f"macro_f1={w['macro_f1']:.4f}, n_gt={w['n_gt_rows']}")

        # Show top-5 backend_id rows
        print("\n[INFO] Top-5 backends (backend_id table):")
        cols = ["split_name", "backend_id", "prompt_type", "backend_dir", "macro_f1", "accuracy", "n_gt_rows", "gt_coverage_frac"]
        print(df_backend[cols].head(5).to_string(index=False))

        print("\n[INFO] Top-5 logical backends (combined):")
        cols2 = ["prompt_type", "backend_dir", "macro_f1_combined", "macro_f1_val50", "macro_f1_train650", "n_gt_rows_combined"]
        print(df_logical[cols2].head(5).to_string(index=False))

        print("\n[INFO] Done.")
    finally:
        conn.close()
        print("[INFO] DB connection closed.")


if __name__ == "__main__":
    main()
