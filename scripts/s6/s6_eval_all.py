#!/usr/bin/env python3
"""
S6 Evaluation Suite (Scenario Reasoning) — v3

Evaluates 12 backends (4 prompts × 3 models) on val split using:
- GT-only strict agreement (label correctness)
- Cross-prompt stability (no GT required)
- Cross-model agreement (no GT required)
- Confidence calibration (GT-only) + High-confidence precision (ALL backends)
- Discovery power (no GT required): "New Disc." per backend
- Human-validated reasoning/causality (JSONL subset)

New in v3 (requested upgrades):
1) High-confidence precision computed for ALL backends (not only best).
2) New Disc metric per backend:
     NewDisc = #GT-missing windows with (conf>=0.9 AND prompt-consensus>=X AND pred==modal_label).
3) Confusion matrices (CSV + PNG) produced for ALL backends.
   Best/Worst still also get the special filenames.
4) From confusion matrix: per-label precision/recall/F1 + macro/weighted per backend.
5) Retrieval removed from S6 (handled separately in S7).

Outputs:
- s6_backend_scoreboard.csv
- s6_l2_prompt_stability.csv (model-level)
- s6_l2_prompt_window_consensus.csv (window-level; used by New Disc.)
- s6_l3_model_consensus.csv
- s6_highconf_precision_all_backends.csv
- s6_new_disc_all_backends.csv
- s6_prf_scoreboard.csv
- confusion matrices: s6_confusion_backend_{id}.csv/.png (+ best/worst special copies)
- reliability plots/bins for best and worst (unchanged)

Assumes:
- Postgres with llm_backends + v_scenario_trace already populated (S6G done).
"""

from __future__ import annotations

import os
import glob
import json
import math
import yaml
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_pg_engine(db_cfg: Dict[str, Any]) -> Engine:
    env_file = db_cfg.get("env_file")
    if env_file:
        load_dotenv(dotenv_path=str(env_file), override=False)
        values = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "dbname": os.getenv("DB_NAME"),
        }
    else:
        values = {
            "host": db_cfg.get("host"),
            "port": db_cfg.get("port"),
            "user": db_cfg.get("user"),
            "password": db_cfg.get("password"),
            "dbname": db_cfg.get("dbname"),
        }

    missing = [key for key, value in values.items() if value in (None, "")]
    if missing:
        raise ValueError(f"Missing database configuration fields: {', '.join(missing)}")

    url = URL.create(
        "postgresql+psycopg2",
        username=str(values["user"]),
        password=str(values["password"]),
        host=str(values["host"]),
        port=int(str(values["port"])),
        database=str(values["dbname"]),
    )
    return create_engine(url, pool_pre_ping=True)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def normalize_label(raw: Any, aliases: Dict[str, str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None
    s0 = s.lower().strip()
    s0 = s0.replace(" ", "_").replace("-", "_")
    return aliases.get(s0, s0)


def canonicalize_window_key(wk: Any, t_precision: int = 6) -> str:
    wk = ("" if wk is None else str(wk)).strip()
    parts = wk.split("|")
    if len(parts) != 3:
        return wk
    log_id = parts[0].strip()
    try:
        t0 = round(float(parts[1]), t_precision)
        t1 = round(float(parts[2]), t_precision)
    except Exception:
        return wk
    fmt = f"{{:.{t_precision}f}}"
    return f"{log_id}|{fmt.format(t0)}|{fmt.format(t1)}"


def entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return float("nan")
    return float(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true))


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    other = "other"
    allowed = set(labels)
    yt = [(t if t in allowed else other) for t in y_true]
    yp = [(p if p in allowed else other) for p in y_pred]
    all_labs = labels + [other]

    f1s = []
    for lab in all_labs:
        tp = sum((t == lab and p == lab) for t, p in zip(yt, yp))
        fp = sum((t != lab and p == lab) for t, p in zip(yt, yp))
        fn = sum((t == lab and p != lab) for t, p in zip(yt, yp))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)

    head = f1s[: len(labels)]
    return float(np.mean(head)) if head else 0.0


# -------------------------
# Data loading (DB)
# -------------------------
def fetch_backends(engine: Engine, split: str) -> pd.DataFrame:
    q = """
    SELECT backend_id, prompt_type, model_name, backend_dir
    FROM llm_backends
    WHERE split_name=:split
    ORDER BY backend_id;
    """
    return pd.read_sql(text(q), engine, params={"split": split})


def fetch_trace(engine: Engine, split: str, backend_id: int) -> pd.DataFrame:
    q = """
    SELECT
      split_name, backend_id, prompt_type, model_name, backend_dir,
      log_id::text AS log_id,
      window_key,
      gt_label_canonical,
      llm_label_canonical,
      llm_label_raw,
      llm_confidence,
      llm_primary_track_uuid::text AS llm_primary_track_uuid,
      llm_actor1_track_uuid::text AS llm_actor1_track_uuid,
      llm_actor2_track_uuid::text AS llm_actor2_track_uuid,
      llm_actor3_track_uuid::text AS llm_actor3_track_uuid
    FROM v_scenario_trace
    WHERE split_name=:split AND backend_id=:bid;
    """
    df = pd.read_sql(text(q), engine, params={"split": split, "bid": int(backend_id)})
    if not df.empty:
        df["llm_confidence"] = df["llm_confidence"].apply(lambda x: safe_float(x, float("nan")))
        df["window_key"] = df["window_key"].apply(lambda x: canonicalize_window_key(x, 6))
    return df


# -------------------------
# Human JSONL parsing
# -------------------------
def load_human_jsonl(input_dir: str, glob_pat: str, aliases: Dict[str, str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, glob_pat)))
    rows: List[Dict[str, Any]] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                log_id = str(obj.get("log_id", "")).strip()
                window_key_raw = str(obj.get("window_key", "")).strip()
                if not window_key_raw:
                    continue
                window_key = canonicalize_window_key(window_key_raw, 6)

                scenario = obj.get("scenario", {}) or {}
                scenario_label_raw = scenario.get("scenario_label", None)
                scenario_label = normalize_label(scenario_label_raw, aliases)

                fields = scenario.get("fields", {}) or {}
                primary_trigger = fields.get("primary_trigger", None)
                overall_conf = (fields.get("overall_confidence", None) or "").strip()

                actor_uuid_by_label: Dict[str, str] = {}
                for a in (obj.get("actors", []) or []):
                    lab = str(a.get("label", "")).strip()
                    uuid = a.get("uuid", None)
                    if lab and uuid is not None:
                        actor_uuid_by_label[lab] = str(uuid).strip()

                human_primary_uuid = None
                if primary_trigger:
                    human_primary_uuid = actor_uuid_by_label.get(str(primary_trigger).strip(), None)

                rows.append({
                    "log_id": log_id,
                    "window_key": window_key,
                    "human_scenario_label_raw": scenario_label_raw,
                    "human_scenario_label": scenario_label,
                    "human_primary_trigger": primary_trigger,
                    "human_primary_track_uuid": human_primary_uuid,
                    "human_overall_confidence": overall_conf,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # CRITICAL: one row per validated window_key
    df = df.drop_duplicates(subset=["window_key"], keep="first").reset_index(drop=True)
    return df


def eval_human_alignment(
    df_pred: pd.DataFrame,
    df_human: pd.DataFrame,
    aliases: Dict[str, str],
    canonical_labels: List[str],
) -> Dict[str, Any]:
    if df_human is None or df_human.empty:
        return {
            "n_human": 0,
            "coverage": 0.0,
            "scenario_acc": float("nan"),
            "scenario_macro_f1": float("nan"),
            "actor1_acc": float("nan"),
            "actor3_hit": float("nan"),
            "joint_correct": float("nan"),
        }

    d = df_pred.copy()
    d["window_key"] = d["window_key"].apply(lambda x: canonicalize_window_key(x, 6))

    h = df_human.copy()
    h["window_key"] = h["window_key"].apply(lambda x: canonicalize_window_key(x, 6))
    h = h.drop_duplicates(subset=["window_key"], keep="first").reset_index(drop=True)

    merged = h.merge(d, on="window_key", how="left", suffixes=("_h", "_llm"))
    n = int(len(merged))

    pred_norm = merged["llm_label_canonical"].apply(lambda x: normalize_label(x, aliases))
    cov = float(pred_norm.notna().mean()) if n > 0 else 0.0

    y_true = merged["human_scenario_label"].apply(lambda x: normalize_label(x, aliases)).tolist()
    y_pred = pred_norm.tolist()

    y_true_fill = [(t if t is not None else "other") for t in y_true]
    y_pred_fill = [(p if p is not None else "other") for p in y_pred]

    scen_acc = accuracy(y_true_fill, y_pred_fill)
    scen_mf1 = macro_f1(y_true_fill, y_pred_fill, canonical_labels)

    human_uuid = merged["human_primary_track_uuid"].astype(str).replace({"None": None, "nan": None})
    llm_primary = merged["llm_primary_track_uuid"].astype(str).replace({"None": None, "nan": None})

    actor1 = []
    for hu, lp in zip(human_uuid.tolist(), llm_primary.tolist()):
        if hu is None or hu in {"None", "nan", ""}:
            actor1.append(np.nan)
        else:
            actor1.append(1.0 if lp == hu else 0.0)
    actor1_acc = float(np.nanmean(actor1)) if len(actor1) else float("nan")

    a1 = merged["llm_actor1_track_uuid"].astype(str).replace({"None": None, "nan": None})
    a2 = merged["llm_actor2_track_uuid"].astype(str).replace({"None": None, "nan": None})
    a3 = merged["llm_actor3_track_uuid"].astype(str).replace({"None": None, "nan": None})
    hit = []
    for hu, x1, x2, x3 in zip(human_uuid.tolist(), a1.tolist(), a2.tolist(), a3.tolist()):
        if hu is None or hu in {"None", "nan", ""}:
            hit.append(np.nan)
        else:
            hit.append(1.0 if hu in {x1, x2, x3} else 0.0)
    actor3_hit = float(np.nanmean(hit)) if len(hit) else float("nan")

    joint = []
    for t, p, a in zip(y_true_fill, y_pred_fill, actor1):
        if np.isnan(a):
            joint.append(np.nan)
        else:
            joint.append(1.0 if (t == p and a == 1.0) else 0.0)
    joint_correct = float(np.nanmean(joint)) if len(joint) else float("nan")

    return {
        "n_human": int(n),
        "coverage": float(cov),
        "scenario_acc": float(scen_acc),
        "scenario_macro_f1": float(scen_mf1),
        "actor1_acc": float(actor1_acc),
        "actor3_hit": float(actor3_hit),
        "joint_correct": float(joint_correct),
    }


# -------------------------
# S6.1 Strict GT Agreement + Confusion/PRF helpers
# -------------------------
def eval_strict_vs_gt(
    df: pd.DataFrame,
    canonical_labels: List[str],
    aliases: Dict[str, str],
    use_llm_field: str,
    normalize_raw: bool,
) -> Dict[str, Any]:
    dfg = df[df["gt_label_canonical"].notna()].copy()
    total = len(dfg)
    if total == 0:
        return {"n_gt": 0, "coverage": 0.0, "accuracy": float("nan"), "macro_f1": float("nan")}

    y_true = dfg["gt_label_canonical"].astype(str).tolist()
    y_pred_raw = dfg[use_llm_field].tolist()
    if normalize_raw:
        y_pred = [normalize_label(x, aliases) for x in y_pred_raw]
    else:
        y_pred = [None if x is None else str(x) for x in y_pred_raw]

    cov = sum(p is not None and str(p).strip() != "" for p in y_pred) / total
    y_pred_fill = [(p if p is not None and str(p).strip() != "" else "other") for p in y_pred]
    y_true_fill = [(t if t is not None else "other") for t in y_true]

    acc = accuracy(y_true_fill, y_pred_fill)
    mf1 = macro_f1(y_true_fill, y_pred_fill, canonical_labels)

    return {"n_gt": int(total), "coverage": float(cov), "accuracy": float(acc), "macro_f1": float(mf1)}


def confusion_matrix_counts(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    # include "other" so counts always tally
    labs = labels + ["other"]
    m = {t: {p: 0 for p in labs} for t in labs}
    for t, p in zip(y_true, y_pred):
        tt = t if t in labs else "other"
        pp = p if p in labs else "other"
        m[tt][pp] += 1
    df = pd.DataFrame(m).T
    df.index.name = "gt"
    df.columns.name = "pred"
    return df


def plot_confusion(df_cm: pd.DataFrame, title: str, out_path: str, dpi: int = 250) -> None:
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.imshow(df_cm.values, aspect="auto")
    ax.set_xticks(range(len(df_cm.columns)))
    ax.set_yticks(range(len(df_cm.index)))
    ax.set_xticklabels(df_cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(df_cm.index)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            ax.text(j, i, str(int(df_cm.values[i, j])), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def prf_from_confusion(df_cm: pd.DataFrame, labels: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    labs = labels + ["other"]
    rows = []
    supports = df_cm.sum(axis=1).to_dict()

    for lab in labs:
        tp = float(df_cm.loc[lab, lab]) if (lab in df_cm.index and lab in df_cm.columns) else 0.0
        fp = float(df_cm[lab].sum() - tp) if lab in df_cm.columns else 0.0
        fn = float(df_cm.loc[lab].sum() - tp) if lab in df_cm.index else 0.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        rows.append({
            "label": lab,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(supports.get(lab, 0.0)),
        })

    df_prf = pd.DataFrame(rows)

    # Macro/weighted headline over canonical labels ONLY (exclude "other")
    df_head = df_prf[df_prf["label"].isin(labels)].copy()
    macro_p = float(df_head["precision"].mean()) if not df_head.empty else float("nan")
    macro_r = float(df_head["recall"].mean()) if not df_head.empty else float("nan")
    macro_f = float(df_head["f1"].mean()) if not df_head.empty else float("nan")

    total_support = float(df_head["support"].sum()) if not df_head.empty else 0.0
    if total_support > 0:
        w_p = float((df_head["precision"] * df_head["support"]).sum() / total_support)
        w_r = float((df_head["recall"] * df_head["support"]).sum() / total_support)
        w_f = float((df_head["f1"] * df_head["support"]).sum() / total_support)
    else:
        w_p = w_r = w_f = float("nan")

    # Micro accuracy on canonical GT rows (still includes errors into other preds)
    correct = float(sum(df_cm.loc[l, l] for l in labels if l in df_cm.index and l in df_cm.columns))
    denom = float(sum(df_cm.loc[l].sum() for l in labels if l in df_cm.index))
    micro_acc = (correct / denom) if denom > 0 else float("nan")

    summary = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "weighted_precision": w_p,
        "weighted_recall": w_r,
        "weighted_f1": w_f,
        "micro_acc": micro_acc,
    }
    return df_prf, summary


# -------------------------
# S6.4 Calibration helpers
# -------------------------
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    conf = np.clip(conf, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    if N == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (mask.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: List[float]) -> pd.DataFrame:
    rows = []
    N = len(conf)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < len(bins) - 2 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else float("nan")
        cmean = float(conf[mask].mean()) if n > 0 else float("nan")
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": n,
            "accuracy": acc, "mean_conf": cmean,
            "frac": (n / N if N > 0 else 0.0),
        })
    return pd.DataFrame(rows)


def plot_reliability(df_bins: pd.DataFrame, title: str, out_path: str, dpi: int = 250) -> None:
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1])
    x = df_bins["mean_conf"].to_numpy()
    y = df_bins["accuracy"].to_numpy()
    ax.scatter(x, y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


# -------------------------
# S6.2 Prompt stability (window + model aggregates)
# -------------------------
def prompt_window_consensus(df_all: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()

    df = df_all.copy()
    df["pred_norm"] = df["llm_label_canonical"].apply(lambda x: normalize_label(x, aliases))
    df["conf"] = df["llm_confidence"].astype(float)

    rows = []
    for (model, wk), g in df.groupby(["model_name", "window_key"]):
        labs = [l for l in g["pred_norm"].tolist() if l is not None]
        if len(labs) == 0:
            continue
        vals, cnts = np.unique(labs, return_counts=True)
        modal_idx = int(np.argmax(cnts))
        modal = str(vals[modal_idx])
        agree = float(cnts[modal_idx] / len(labs))
        ent = entropy_from_counts(cnts.tolist())

        conf_modal = g[g["pred_norm"] == modal]["conf"].dropna()
        ccons = float(conf_modal.mean()) if len(conf_modal) else float("nan")

        rows.append({
            "model_name": model,
            "window_key": wk,
            "prompt_votes": int(len(labs)),
            "modal_label": modal,
            "agreement_rate": agree,
            "entropy": ent,
            "consensus_confidence": ccons,
        })

    return pd.DataFrame(rows)


def eval_cross_prompt_stability(df_prompt_window: pd.DataFrame) -> pd.DataFrame:
    if df_prompt_window.empty:
        return pd.DataFrame()
    out = df_prompt_window.groupby("model_name").agg(
        mean_agreement=("agreement_rate", "mean"),
        low_entropy_frac=("entropy", lambda s: float((s <= np.log(2)).mean())),
        mean_consensus_conf=("consensus_confidence", "mean"),
        n_windows=("window_key", "nunique"),
    ).reset_index()
    return out


# -------------------------
# S6.3 Cross-model agreement
# -------------------------
def eval_cross_model_consensus(df_all: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame()

    df = df_all.copy()
    df["pred_norm"] = df["llm_label_canonical"].apply(lambda x: normalize_label(x, aliases))

    rows = []
    for (prompt, wk), g in df.groupby(["prompt_type", "window_key"]):
        labs = [l for l in g["pred_norm"].tolist() if l is not None]
        if len(labs) == 0:
            continue
        vals, cnts = np.unique(labs, return_counts=True)
        m = int(np.max(cnts))
        agree1 = 1.0 if (len(labs) >= 3 and m == 3) else 0.0
        agree2 = 1.0 if m >= 2 else 0.0
        ent = entropy_from_counts(cnts.tolist())
        rows.append({
            "prompt_type": prompt,
            "window_key": wk,
            "model_votes": int(len(labs)),
            "agreement_at_1": agree1,
            "agreement_at_2": agree2,
            "entropy": ent,
        })

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    out = d.groupby("prompt_type").agg(
        agreement_at_1=("agreement_at_1", "mean"),
        agreement_at_2=("agreement_at_2", "mean"),
        mean_entropy=("entropy", "mean"),
        n_windows=("window_key", "nunique"),
    ).reset_index()
    return out


# -------------------------
# Backend selection
# -------------------------
def pick_best_worst(df_backend_scores: pd.DataFrame) -> Tuple[int, int]:
    d = df_backend_scores.copy()
    d["macro_f1_canonical_gt"] = d["macro_f1_canonical_gt"].astype(float)
    d["accuracy_canonical_gt"] = d["accuracy_canonical_gt"].astype(float)
    d["coverage_gt"] = d["coverage_gt"].astype(float)

    d_sorted = d.sort_values(
        ["macro_f1_canonical_gt", "accuracy_canonical_gt", "coverage_gt"],
        ascending=[False, False, False],
    )
    best = int(d_sorted.iloc[0]["backend_id"])

    d_nonzero = d[d["coverage_gt"] > 0.0].copy()
    if d_nonzero.empty:
        worst = best
    else:
        d_w = d_nonzero.sort_values(
            ["macro_f1_canonical_gt", "accuracy_canonical_gt", "coverage_gt"],
            ascending=[True, True, True],
        )
        worst = int(d_w.iloc[0]["backend_id"])
    return best, worst


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/s6_eval.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    db_cfg = cfg["db"]
    ev = cfg["eval"]
    out_cfg = cfg["outputs"]

    split = ev["split_name"]
    canonical_labels = [str(x).lower() for x in ev["canonical_labels"]]
    aliases = {str(k).lower(): str(v).lower() for k, v in (ev.get("label_aliases", {}) or {}).items()}

    out_dir = out_cfg["out_dir"]
    ensure_dir(out_dir)

    engine = make_pg_engine(db_cfg)

    # Backends
    df_backends = fetch_backends(engine, split)
    backend_ids_cfg = ev.get("backend_ids", []) or []
    if backend_ids_cfg:
        backend_ids = [int(x) for x in backend_ids_cfg]
        df_backends = df_backends[df_backends["backend_id"].isin(backend_ids)].copy()
    else:
        backend_ids = df_backends["backend_id"].astype(int).tolist()

    # Human
    hv_cfg = ev.get("human_validation", {}) or {}
    df_human = None
    if hv_cfg.get("enabled", True):
        input_dir = hv_cfg["input_dir"]
        glob_pat = hv_cfg.get("glob", "*.jsonl")
        if os.path.isdir(input_dir):
            df_human = load_human_jsonl(input_dir, glob_pat, aliases)
            df_human.to_csv(os.path.join(out_dir, "s6_human_loaded.csv"), index=False)
        else:
            print(f"[WARN] human_validation input_dir not found: {input_dir}")

    # Discovery config
    disc_cfg = ev.get("discovery", {}) or {}
    disc_enabled = bool(disc_cfg.get("enabled", True))
    disc_conf_thr = float(disc_cfg.get("conf_threshold", 0.90))
    disc_cons_thr = float(disc_cfg.get("consensus_threshold", 0.75))
    disc_excl = set([str(x).lower() for x in (disc_cfg.get("exclude_labels", ["other"]) or [])])

    # Confidence config
    conf_cfg = ev.get("confidence", {}) or {}
    high_conf_thresholds = [float(x) for x in (conf_cfg.get("high_conf_thresholds", [0.7, 0.8, 0.9]) or [])]
    rel_bins = conf_cfg.get("bins", [0.0, 0.5, 0.7, 0.8, 0.9, 1.01])

    # Calibration config
    cal_cfg = ev.get("calibration", {}) or {}
    n_bins_ece = int(cal_cfg.get("n_bins", 10))

    dpi = int(out_cfg.get("plot_dpi", 250))
    make_plots = bool(out_cfg.get("make_plots", True))

    # Collect
    backend_rows = []
    df_all_preds = []
    confusion_cache: Dict[int, Tuple[List[str], List[str], pd.DataFrame, Dict[str, float]]] = {}
    highconf_rows = []
    newdisc_rows = []

    # Loop backends
    for bid in backend_ids:
        df = fetch_trace(engine, split, bid)
        if df.empty:
            continue

        df_all_preds.append(df)

        # strict canonical vs GT
        m_can = eval_strict_vs_gt(df, canonical_labels, aliases, use_llm_field="llm_label_canonical", normalize_raw=False)
        # strict raw-normalized vs GT
        m_raw = eval_strict_vs_gt(df, canonical_labels, aliases, use_llm_field="llm_label_raw", normalize_raw=True)

        # Human alignment
        m_h = eval_human_alignment(df, df_human, aliases, canonical_labels) if df_human is not None else {
            "n_human": 0, "coverage": float("nan"), "scenario_acc": float("nan"), "scenario_macro_f1": float("nan"),
            "actor1_acc": float("nan"), "actor3_hit": float("nan"), "joint_correct": float("nan"),
        }

        # Confusion + PRF (GT-only)
        dfg = df[df["gt_label_canonical"].notna()].copy()
        y_true = dfg["gt_label_canonical"].astype(str).str.lower().tolist()
        pred_norm = dfg["llm_label_canonical"].apply(lambda x: normalize_label(x, aliases))
        y_pred = [(p if (p is not None and p in canonical_labels) else "other") for p in pred_norm.tolist()]

        # Ensure GT is in canonical; otherwise map to other (should not happen in your setup)
        y_true = [(t if t in canonical_labels else "other") for t in y_true]

        df_cm = confusion_matrix_counts(y_true, y_pred, canonical_labels)
        df_prf, prf_sum = prf_from_confusion(df_cm, canonical_labels)

        confusion_cache[int(bid)] = (y_true, y_pred, df_cm, prf_sum)

        # Save per-backend PRF detail now (useful for committee surprise questions)
        df_prf.to_csv(os.path.join(out_dir, f"s6_prf_backend_{int(bid)}.csv"), index=False)

        # High-confidence precision for THIS backend (GT-only)
        if len(dfg) > 0:
            conf = dfg["llm_confidence"].astype(float).fillna(0.0)
            gt = dfg["gt_label_canonical"].astype(str).str.lower()
            pred_fill = pred_norm.fillna("other").astype(str).str.lower()
            pred_fill = pred_fill.where(pred_fill.isin(canonical_labels), "other")
            correct = (pred_fill == gt)

            for thr in high_conf_thresholds:
                mask = conf >= float(thr)
                if int(mask.sum()) == 0:
                    highconf_rows.append({
                        "backend_id": int(bid),
                        "threshold": float(thr),
                        "precision": float("nan"),
                        "coverage": 0.0,
                        "n": 0,
                    })
                else:
                    highconf_rows.append({
                        "backend_id": int(bid),
                        "threshold": float(thr),
                        "precision": float(correct[mask].mean()),
                        "coverage": float(mask.mean()),
                        "n": int(mask.sum()),
                    })

        # Scoreboard row
        backend_rows.append({
            "backend_id": int(bid),
            "prompt_type": str(df["prompt_type"].iloc[0]),
            "model_name": str(df["model_name"].iloc[0]),
            "backend_dir": str(df["backend_dir"].iloc[0]),

            "n_gt": m_can["n_gt"],
            "coverage_gt": m_can["coverage"],
            "accuracy_canonical_gt": m_can["accuracy"],
            "macro_f1_canonical_gt": m_can["macro_f1"],

            "accuracy_rawnorm_gt": m_raw["accuracy"],
            "macro_f1_rawnorm_gt": m_raw["macro_f1"],

            # PRF headline from confusion matrix
            "macro_precision_canonical_gt": prf_sum["macro_precision"],
            "macro_recall_canonical_gt": prf_sum["macro_recall"],
            "weighted_f1_canonical_gt": prf_sum["weighted_f1"],
            "micro_acc_canonical_gt": prf_sum["micro_acc"],

            "n_human": m_h["n_human"],
            "coverage_human": m_h["coverage"],
            "human_scenario_acc": m_h["scenario_acc"],
            "human_scenario_macro_f1": m_h["scenario_macro_f1"],
            "human_actor1_acc": m_h["actor1_acc"],
            "human_actor3_hit": m_h["actor3_hit"],
            "human_joint_correct": m_h["joint_correct"],
        })

    # Write backend scoreboard
    df_backend = pd.DataFrame(backend_rows)
    df_backend.to_csv(os.path.join(out_dir, "s6_backend_scoreboard.csv"), index=False)

    # Aggregate: PRF scoreboard (macro/weighted/micro per backend)
    prf_score_rows = []
    for bid, (_, _, _, prf_sum) in confusion_cache.items():
        prf_score_rows.append({"backend_id": int(bid), **prf_sum})
    pd.DataFrame(prf_score_rows).sort_values("backend_id").to_csv(
        os.path.join(out_dir, "s6_prf_scoreboard.csv"), index=False
    )

    # High-confidence precision for ALL backends
    pd.DataFrame(highconf_rows).sort_values(["backend_id", "threshold"]).to_csv(
        os.path.join(out_dir, "s6_highconf_precision_all_backends.csv"), index=False
    )

    # Cross-prompt / cross-model / discovery require df_all
    df_all = pd.concat(df_all_preds, ignore_index=True) if df_all_preds else pd.DataFrame()

    # Prompt window consensus (window-level)
    df_pw = prompt_window_consensus(df_all, aliases) if not df_all.empty else pd.DataFrame()
    if not df_pw.empty:
        df_pw.to_csv(os.path.join(out_dir, "s6_l2_prompt_window_consensus.csv"), index=False)

        # Model-level stability
        df_prompt = eval_cross_prompt_stability(df_pw)
        df_prompt.to_csv(os.path.join(out_dir, "s6_l2_prompt_stability.csv"), index=False)

    # Cross-model agreement (prompt-level)
    if not df_all.empty:
        df_model = eval_cross_model_consensus(df_all, aliases)
        df_model.to_csv(os.path.join(out_dir, "s6_l3_model_consensus.csv"), index=False)

    # New Disc metric per backend
    if disc_enabled and (not df_all.empty) and (not df_pw.empty):
        # fast lookup: (model_name, window_key) -> (agreement_rate, modal_label)
        key2 = {}
        for _, r in df_pw.iterrows():
            key2[(str(r["model_name"]), str(r["window_key"]))] = (float(r["agreement_rate"]), str(r["modal_label"]))

        for bid in backend_ids:
            df = next((x for x in df_all_preds if int(x["backend_id"].iloc[0]) == int(bid)), None)
            if df is None or df.empty:
                continue

            model_name = str(df["model_name"].iloc[0])

            # only GT-missing windows
            d0 = df[df["gt_label_canonical"].isna()].copy()
            n_no_gt = int(d0["window_key"].nunique())

            if d0.empty:
                newdisc_rows.append({
                    "backend_id": int(bid),
                    "prompt_type": str(df["prompt_type"].iloc[0]),
                    "model_name": model_name,
                    "n_no_gt": n_no_gt,
                    "new_disc_count": 0,
                    "new_disc_rate": 0.0,
                    "conf_thr": disc_conf_thr,
                    "cons_thr": disc_cons_thr,
                })
                continue

            # compute per window
            keep_w = set()
            for _, row in d0.iterrows():
                wk = str(row["window_key"])
                conf = safe_float(row["llm_confidence"], float("nan"))
                pred = normalize_label(row["llm_label_canonical"], aliases)
                if pred is None:
                    continue
                if pred in disc_excl:
                    continue
                if not (conf >= disc_conf_thr):
                    continue

                k = (model_name, wk)
                if k not in key2:
                    continue
                agree, modal = key2[k]
                if agree < disc_cons_thr:
                    continue
                if pred != modal:
                    continue
                keep_w.add(wk)

            cnt = int(len(keep_w))
            rate = float(cnt / n_no_gt) if n_no_gt > 0 else 0.0

            newdisc_rows.append({
                "backend_id": int(bid),
                "prompt_type": str(df["prompt_type"].iloc[0]),
                "model_name": model_name,
                "n_no_gt": n_no_gt,
                "new_disc_count": cnt,
                "new_disc_rate": rate,
                "conf_thr": disc_conf_thr,
                "cons_thr": disc_cons_thr,
            })

        pd.DataFrame(newdisc_rows).sort_values("backend_id").to_csv(
            os.path.join(out_dir, "s6_new_disc_all_backends.csv"), index=False
        )

    # Confusion matrices for ALL backends (CSV + PNG)
    if make_plots and confusion_cache:
        for bid, (_, _, df_cm, _) in confusion_cache.items():
            df_cm.to_csv(os.path.join(out_dir, f"s6_confusion_backend_{bid}.csv"))
            plot_confusion(
                df_cm,
                title=f"S6 Confusion Matrix (GT-only) — backend={bid}",
                out_path=os.path.join(out_dir, f"s6_confusion_backend_{bid}.png"),
                dpi=dpi,
            )

    # Best/worst (special copies + reliability plots/bins)
    if make_plots and (not df_backend.empty):
        best_id, worst_id = pick_best_worst(df_backend)

        # Special confusion copies
        for tag, bid in [("best", best_id), ("worst", worst_id)]:
            if bid in confusion_cache:
                _, _, df_cm, _ = confusion_cache[bid]
                df_cm.to_csv(os.path.join(out_dir, f"s6_confusion_{tag}_backend_{bid}.csv"))
                plot_confusion(
                    df_cm,
                    title=f"S6 Confusion Matrix (GT-only) — {tag.upper()} backend={bid}",
                    out_path=os.path.join(out_dir, f"s6_confusion_{tag}_backend_{bid}.png"),
                    dpi=dpi,
                )

        # Reliability plots for best/worst
        for tag, bid in [("best", best_id), ("worst", worst_id)]:
            df = fetch_trace(engine, split, bid)
            dfg = df[df["gt_label_canonical"].notna()].copy()
            if dfg.empty:
                continue

            pred = dfg["llm_label_canonical"].apply(lambda x: normalize_label(x, aliases))
            conf = dfg["llm_confidence"].astype(float).fillna(0.0).to_numpy()
            gt = dfg["gt_label_canonical"].astype(str).str.lower().to_numpy()

            pred_fill = pred.fillna("other").astype(str).str.lower().to_numpy()
            pred_fill = np.array([p if p in canonical_labels else "other" for p in pred_fill], dtype=object)

            correct = (pred_fill == gt).astype(float)

            ece = expected_calibration_error(conf, correct, n_bins=n_bins_ece)
            df_bins = reliability_bins(conf, correct, bins=rel_bins)
            df_bins["ece"] = ece
            df_bins.to_csv(os.path.join(out_dir, f"s6_reliability_bins_{tag}_backend_{bid}.csv"), index=False)

            plot_reliability(
                df_bins,
                title=f"S6 Reliability Diagram (GT-only) — {tag.upper()} backend={bid} (ECE={ece:.3f})",
                out_path=os.path.join(out_dir, f"s6_reliability_{tag}_backend_{bid}.png"),
                dpi=dpi,
            )

    print(f"[INFO] S6 done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
