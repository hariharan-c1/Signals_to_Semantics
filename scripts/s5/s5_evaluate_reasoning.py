#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Evaluation Script (tags_norm-based, multi-backend)

- Uses tags_norm.jsonl as authoritative GT for:
    * scenario label: `tag`
    * primary actor: `guest_id` (if has_guest == True)

- Optionally uses window_labels.jsonl to restrict GT logs to current split.

- Evaluation per backend (e.g. azure / ollama):

  1) Scenario classification metrics:
     - One representative window per GT log:
         * Pick window whose predicted scenario matches GT tag.
         * If multiple, prefer window where LLM primary actor == GT guest_id.
         * If none match the GT tag, pick the first window (counts as FN).
     - "Window-level" scenario accuracy (per-log)
     - Log-level hit accuracy (any window for that log predicts GT tag)
     - Confusion matrix (GT vs predicted)
     - Per-class TP, FP, FN, TN, precision, recall, F1, support
     - Micro accuracy, macro precision/recall/F1

  2) Actor metrics:
     - Overall (all windows with LLM primary actor):
         * LLM primary actor == GAT rank-1 track_uuid
         * LLM primary actor in top-2 (ACTOR1 or ACTOR2)
     - On GT logs (representative window only):
         * GT actor rank (1/2/3/None)
         * LLM primary actor == GT actor (track_uuid)

  3) New-discovery logs:
     - Logs in this split with no GT tag but at least one non-'other'
       scenario predicted by the backend.

  4) Cross-backend agreement:
     - Agreement rate between backend A and B on scenario_classification
       for shared windows.

Outputs per backend to: <llm_root>/eval/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic IO helpers
# ---------------------------------------------------------------------

def read_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return pd.read_json(path, lines=True)


def load_evidences(evidence_dir: str) -> pd.DataFrame:
    """
    Load S4.1 evidence JSONs.

    Returns DataFrame with:
        - window_key
        - log_id
        - actors (list[dict])
    """
    evidence_dir = Path(evidence_dir)
    rows = []
    for js in sorted(evidence_dir.glob("*.json")):
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "window_key": data.get("window_key"),
            "log_id": data.get("log_id"),
            "actors": data.get("actors", []),
        })
    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} evidence windows from {evidence_dir}")
    return df


def load_llm_backend(backend_dir: Path, backend_name: str) -> pd.DataFrame:
    """
    Load LLM outputs for a single backend.

    Expects per-window JSON with structure:
        {
          "backend": "...",
          "parsed_result": {
             "actor_matrix": [...],
             "primary_responsible_actor": "...",
             "scenario_classification": "...",
             "confidence_score": ...,
             "ego_window_key": "...",
             ...
          }
        }
    """
    rows = []
    n_files = 0
    for js in sorted(backend_dir.glob("*.json")):
        n_files += 1
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        pr = data.get("parsed_result") or {}
        window_key = pr.get("ego_window_key")
        if window_key is None:
            # Fall back: maybe top-level window_key
            window_key = data.get("ego_window_key") or data.get("window_key")

        rows.append({
            "window_key": window_key,
            "backend": backend_name,
            "scenario_classification": pr.get("scenario_classification"),
            "primary_responsible_actor": pr.get("primary_responsible_actor"),
            "confidence_score": pr.get("confidence_score"),
            "actor_matrix": pr.get("actor_matrix", []),
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} LLM outputs for backend '{backend_name}' "
          f"from {backend_dir} (files seen: {n_files})")
    return df


def load_backends(llm_root: str) -> dict:
    """
    Discover and load all backends under llm_root.

    Convention: each subdir under llm_root is one backend, e.g.
        llm_gpt5mini, llm_ollama_batch, etc.
    """
    root = Path(llm_root)
    if not root.exists():
        raise FileNotFoundError(f"LLM root not found: {llm_root}")

    backends = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if not name.startswith("llm_"):
            # ignore non-LLM folders
            continue
        backend_name = name.replace("llm_", "", 1)
        print(f"[INFO] Detected backend: {backend_name} (dir: {sub})")
        backends[backend_name] = load_llm_backend(sub, backend_name)

    if not backends:
        raise RuntimeError(f"No backend directories found under {llm_root}")
    return backends


def load_tags_norm(tags_norm_path: str,
                   evidence_logs: set,
                   window_labels_path: str | None = None) -> pd.DataFrame:
    """
    Load tags_norm.jsonl and restrict to logs present in this split
    (and optionally to logs listed in window_labels.jsonl).
    """
    df = read_jsonl(tags_norm_path)
    print(f"[INFO] Loaded {len(df)} rows from tags_norm.jsonl")

    # Only keep host_id == 'ego' just in case there are other host types.
    if "host_id" in df.columns:
        df = df[df["host_id"] == "ego"].copy()

    # Filter by logs actually present in evidence.
    df = df[df["log_id"].isin(evidence_logs)].copy()
    print(f"[INFO] Tags after filtering to evidence logs: {len(df)} rows, "
          f"{df['log_id'].nunique()} distinct GT logs")

    # Optional: restrict to logs from window_labels for this split.
    if window_labels_path is not None:
        wl = read_jsonl(window_labels_path)
        if "log_id" not in wl.columns:
            raise ValueError("window_labels.jsonl must contain 'log_id' column")
        split_gt_logs = set(wl["log_id"].astype(str).unique())
        before = len(df)
        df = df[df["log_id"].isin(split_gt_logs)].copy()
        print(f"[INFO] Tags after intersecting with window_labels "
              f"({len(split_gt_logs)} GT logs in split): "
              f"{before} -> {len(df)} rows, {df['log_id'].nunique()} GT logs")

    # If multiple tags per log_id, we keep the first and warn.
    dup_logs = df["log_id"].value_counts()
    multi = dup_logs[dup_logs > 1]
    if not multi.empty:
        print("[WARN] Multiple tags found for some log_ids in tags_norm. "
              "Using the first tag per log_id:")
        for lid, cnt in multi.items():
            print(f"  - {lid}: {cnt} tags")

    df = df.sort_values("log_id").drop_duplicates("log_id", keep="first")
    print(f"[INFO] Final GT tag set: {len(df)} rows (1 per GT log)")
    return df


# ---------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------

def map_actors_for_window(actors: list[dict]) -> dict:
    """
    Given evidence['actors'] list, build:
      - actor_id_to_uuid: ACTOR1/2/3 -> track_uuid
      - uuid_to_rank: track_uuid -> rank_s3
    """
    actor_id_to_uuid = {}
    uuid_to_rank = {}
    for a in actors:
        rank = int(a.get("rank_s3", 0))
        track_uuid = a.get("track_uuid")
        if not track_uuid or rank <= 0:
            continue
        actor_id = f"ACTOR{rank}"
        actor_id_to_uuid[actor_id] = track_uuid
        uuid_to_rank[track_uuid] = rank
    return actor_id_to_uuid, uuid_to_rank


def select_representative_window_for_log(
    log_id: str,
    gt_label: str,
    gt_actor_uuid: str | None,
    cand_df: pd.DataFrame,
    evid_df: pd.DataFrame,
) -> dict:
    """
    For a given GT log, select a single 'representative window' among
    all candidate windows with LLM outputs.

    Steps:
      1) Among candidate windows, find those where predicted scenario
         matches gt_label.
      2) If gt_actor_uuid is available, among those label-matches
         prefer windows where LLM primary actor == GT actor (track_uuid).
      3) If no label-matches at all, fall back to the first candidate.
         (That log will count as FN for confusion matrix / accuracy.)

    Returns a dict with:
      - log_id
      - representative_window_key
      - pred_label_rep
      - any_hit_flag (True if ANY window matched the gt_label)
      - actor_match_rep (bool or None)
      - gt_actor_rank (1/2/3/None)
      - llm_primary_rank_rep (1/2/3/None)
      - llm_primary_track_uuid_rep
      - has_llm (bool)
    """
    result = {
        "log_id": log_id,
        "representative_window_key": None,
        "pred_label_rep": None,
        "any_hit_flag": False,
        "actor_match_rep": None,
        "gt_actor_rank": None,
        "llm_primary_rank_rep": None,
        "llm_primary_track_uuid_rep": None,
        "has_llm": False,
    }

    if cand_df.empty:
        # No LLM outputs for this log.
        return result

    result["has_llm"] = True

    # Merge candidate windows with evidence to access actors/track_uuids.
    merged = cand_df.merge(evid_df, on=["window_key", "log_id"], how="left")

    # Pre-compute per-window info.
    rows_info = []
    any_hit_flag = False

    for _, row in merged.iterrows():
        wk = row["window_key"]
        pred_label = row["scenario_classification"]
        pa_id = row["primary_responsible_actor"]
        actors = row.get("actors", []) or []

        actor_id_to_uuid, uuid_to_rank = map_actors_for_window(actors)

        # LLM primary rank & track_uuid
        pa_rank = None
        pa_uuid = None
        if isinstance(pa_id, str) and pa_id.startswith("ACTOR"):
            try:
                pa_rank = int(pa_id.replace("ACTOR", ""))
            except ValueError:
                pa_rank = None
            pa_uuid = actor_id_to_uuid.get(pa_id)

        # GT actor rank (if GT actor is present among top-3)
        gt_rank = None
        actor_match = None
        if gt_actor_uuid is not None:
            gt_rank = uuid_to_rank.get(gt_actor_uuid)
            if pa_uuid is not None:
                actor_match = (pa_uuid == gt_actor_uuid)

        label_match = (pred_label == gt_label)
        if label_match:
            any_hit_flag = True

        rows_info.append({
            "window_key": wk,
            "pred_label": pred_label,
            "pa_id": pa_id,
            "pa_rank": pa_rank,
            "pa_uuid": pa_uuid,
            "gt_actor_rank": gt_rank,
            "actor_match": actor_match,
            "label_match": label_match,
        })

    result["any_hit_flag"] = any_hit_flag

    # Select representative window.
    # 1) Label matches (pred == gt_label)
    label_matches = [r for r in rows_info if r["label_match"]]
    selected = None

    if label_matches:
        # 2) If GT actor exists, prefer actor_match windows among label_matches.
        if gt_actor_uuid is not None:
            actor_label_matches = [r for r in label_matches if r["actor_match"] is True]
            if actor_label_matches:
                selected = actor_label_matches[0]
            else:
                selected = label_matches[0]
        else:
            selected = label_matches[0]
    else:
        # 3) No label-match -> just pick the first candidate.
        selected = rows_info[0]

    result["representative_window_key"] = selected["window_key"]
    result["pred_label_rep"] = selected["pred_label"]
    result["actor_match_rep"] = selected["actor_match"]
    result["gt_actor_rank"] = selected["gt_actor_rank"]
    result["llm_primary_rank_rep"] = selected["pa_rank"]
    result["llm_primary_track_uuid_rep"] = selected["pa_uuid"]

    return result


def compute_confusion_and_metrics(gt_labels: list[str],
                                  pred_labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute confusion matrix and per-class metrics.

    Returns:
      - confusion_df: DataFrame (rows=GT, cols=Pred)
      - per_class_df: DataFrame with TP,FN,FP,TN,precision,recall,F1,support
      - global_metrics: dict with macro/micro values
    """
    assert len(gt_labels) == len(pred_labels)
    labels = sorted(set(gt_labels) | set(pred_labels))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for g, p in zip(gt_labels, pred_labels):
        gi = label_to_idx[g]
        pi = label_to_idx[p]
        cm[gi, pi] += 1

    confusion_df = pd.DataFrame(cm, index=labels, columns=labels)

    total = cm.sum()
    per_rows = []
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for i, c in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FP - FN
        support = cm[i, :].sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        if precision + recall > 0 and not np.isnan(precision) and not np.isnan(recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan

        if support > 0:
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1s.append(f1)

        per_rows.append({
            "class": c,
            "TP": int(TP),
            "FN": int(FN),
            "FP": int(FP),
            "TN": int(TN),
            "support": int(support),
            "precision": float(precision) if not np.isnan(precision) else None,
            "recall": float(recall) if not np.isnan(recall) else None,
            "f1": float(f1) if not np.isnan(f1) else None,
        })

    per_class_df = pd.DataFrame(per_rows)

    micro_accuracy = np.trace(cm) / total if total > 0 else 0.0
    macro_precision = np.nanmean(macro_precisions) if macro_precisions else np.nan
    macro_recall = np.nanmean(macro_recalls) if macro_recalls else np.nan
    macro_f1 = np.nanmean(macro_f1s) if macro_f1s else np.nan

    global_metrics = {
        "micro_accuracy": micro_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    return confusion_df, per_class_df, global_metrics


def evaluate_backend(
    backend_name: str,
    llm_df: pd.DataFrame,
    evid_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    all_logs_in_split: set[str],
    eval_dir: Path,
) -> dict:
    """
    Evaluate one backend given:
      - llm_df: LLM outputs (window_key, log_id, scenario, primary_actor, ...)
      - evid_df: evidence (window_key, log_id, actors)
      - tags_df: GT per log_id (log_id, tag, guest_id, has_guest)
      - all_logs_in_split: all distinct log_ids present in evidence (for new-discovery)
    """
    print(f"\n[INFO] ==== Evaluating backend: {backend_name} ====")

    # Merge LLM outputs with evidence to attach log_id.
    joined = llm_df.merge(evid_df[["window_key", "log_id", "actors"]],
                          on="window_key", how="left")

    if joined["log_id"].isna().any():
        missing = joined[joined["log_id"].isna()]
        print(f"[WARN] {len(missing)} LLM outputs have window_key not found in evidence "
              f"(backend={backend_name}). They will be ignored in some metrics.")

    # ---------------- GT-based evaluation ----------------

    gt_logs = list(tags_df["log_id"].astype(str).unique())
    print(f"[INFO] GT logs for this split (from tags_norm): {len(gt_logs)}")

    gt_rows = []
    for _, gt_row in tags_df.iterrows():
        log_id = str(gt_row["log_id"])
        gt_label = gt_row["tag"]
        gt_actor_uuid = None
        if bool(gt_row.get("has_guest", False)):
            gt_actor_uuid = gt_row.get("guest_id")

        cand = joined[joined["log_id"] == log_id].copy()
        rep_info = select_representative_window_for_log(
            log_id=log_id,
            gt_label=gt_label,
            gt_actor_uuid=gt_actor_uuid,
            cand_df=cand,
            evid_df=evid_df,
        )

        gt_rows.append({
            "log_id": log_id,
            "gt_label": gt_label,
            "gt_actor_uuid": gt_actor_uuid,
            "representative_window_key": rep_info["representative_window_key"],
            "pred_label_rep": rep_info["pred_label_rep"],
            "has_llm": rep_info["has_llm"],
            "any_hit_flag": rep_info["any_hit_flag"],
            "actor_match_rep": rep_info["actor_match_rep"],
            "gt_actor_rank": rep_info["gt_actor_rank"],
            "llm_primary_rank_rep": rep_info["llm_primary_rank_rep"],
            "llm_primary_track_uuid_rep": rep_info["llm_primary_track_uuid_rep"],
        })

    gt_eval_df = pd.DataFrame(gt_rows)

    # Restrict to logs that have any LLM outputs.
    gt_eval_with_llm = gt_eval_df[gt_eval_df["has_llm"]].copy()
    n_gt_logs = len(gt_eval_df)
    n_gt_with_llm = len(gt_eval_with_llm)
    print(f"[INFO] GT logs with any LLM outputs for backend '{backend_name}': "
          f"{n_gt_with_llm}/{n_gt_logs}")

    # Scenario metrics (representative window per GT log)
    valid_rows = gt_eval_with_llm[gt_eval_with_llm["pred_label_rep"].notna()].copy()
    gt_labels = valid_rows["gt_label"].tolist()
    pred_labels_rep = valid_rows["pred_label_rep"].tolist()

    # Basic scenario accuracy (per GT log, representative window)
    n_correct = sum(g == p for g, p in zip(gt_labels, pred_labels_rep))
    scenario_acc = n_correct / len(valid_rows) if len(valid_rows) > 0 else 0.0
    print(f"[RESULT] Backend {backend_name}: "
          f"scenario accuracy (per GT log, representative window) "
          f"= {scenario_acc:.3f} ({n_correct}/{len(valid_rows)})")

    confusion_df, per_class_df, global_metrics = compute_confusion_and_metrics(
        gt_labels, pred_labels_rep
    )

    # Log-level "hit" accuracy (any window matches GT label)
    hit_rows = gt_eval_with_llm.copy()
    n_hit = hit_rows["any_hit_flag"].sum()
    log_hit_acc = n_hit / len(hit_rows) if len(hit_rows) > 0 else 0.0
    print(f"[RESULT] Backend {backend_name}: log-level HIT accuracy "
          f"(any window has correct scenario) = {log_hit_acc:.3f} "
          f"({n_hit}/{len(hit_rows)})")

    # ---------------- Actor metrics ----------------

    # 1) Overall LLM vs GAT alignment (all windows with primary actor)
    actor_rows = []
    for _, row in joined.iterrows():
        wk = row["window_key"]
        lid = row["log_id"]
        if pd.isna(lid):
            continue
        pa_id = row["primary_responsible_actor"]
        actors = row.get("actors", []) or []
        actor_id_to_uuid, _ = map_actors_for_window(actors)

        pa_uuid = None
        if isinstance(pa_id, str) and pa_id.startswith("ACTOR"):
            pa_uuid = actor_id_to_uuid.get(pa_id)

        gat_rank1_uuid = actor_id_to_uuid.get("ACTOR1")

        actor_rows.append({
            "window_key": wk,
            "log_id": lid,
            "llm_primary_actor_id": pa_id,
            "llm_primary_actor_uuid": pa_uuid,
            "gat_rank1_uuid": gat_rank1_uuid,
        })

    actor_df = pd.DataFrame(actor_rows)
    actor_df_non_null = actor_df[actor_df["llm_primary_actor_id"].notna()].copy()
    n_total_pa = len(actor_df_non_null)

    n_pa_equals_gat1 = (
        actor_df_non_null["llm_primary_actor_uuid"] ==
        actor_df_non_null["gat_rank1_uuid"]
    ).sum()

    # primary in top-2: ACTOR1 or ACTOR2
    def in_top2(aid):
        if not isinstance(aid, str):
            return False
        return aid in ("ACTOR1", "ACTOR2")

    n_pa_in_top2 = actor_df_non_null["llm_primary_actor_id"].apply(in_top2).sum()

    ratio_pa_equals_gat1 = n_pa_equals_gat1 / n_total_pa if n_total_pa > 0 else 0.0
    ratio_pa_in_top2 = n_pa_in_top2 / n_total_pa if n_total_pa > 0 else 0.0

    print(f"[RESULT] Backend {backend_name}: LLM primary actor = GAT rank-1 "
          f"track_uuid in {n_pa_equals_gat1}/{n_total_pa} "
          f"({ratio_pa_equals_gat1:.3f})")
    print(f"[RESULT] Backend {backend_name}: LLM primary actor in GAT top-2 "
          f"(ACTOR1 or ACTOR2) in {n_pa_in_top2}/{n_total_pa} "
          f"({ratio_pa_in_top2:.3f})")

    # 2) On GT logs: LLM vs GT actor (representative window only)
    gt_actor_rows = []
    for _, row in valid_rows.iterrows():
        log_id = row["log_id"]
        rep_wk = row["representative_window_key"]
        gt_actor_uuid = row["gt_actor_uuid"]

        # Find that window in joined
        cand = joined[(joined["log_id"] == log_id) &
                      (joined["window_key"] == rep_wk)]
        if cand.empty:
            continue
        j = cand.iloc[0]
        pa_id = j["primary_responsible_actor"]
        actors = j.get("actors", []) or []
        actor_id_to_uuid, uuid_to_rank = map_actors_for_window(actors)

        pa_uuid = None
        if isinstance(pa_id, str) and pa_id.startswith("ACTOR"):
            pa_uuid = actor_id_to_uuid.get(pa_id)

        gt_rank = None
        if gt_actor_uuid is not None:
            gt_rank = uuid_to_rank.get(gt_actor_uuid)

        actor_match = (gt_actor_uuid is not None
                       and pa_uuid is not None
                       and pa_uuid == gt_actor_uuid)

        gt_actor_rows.append({
            "log_id": log_id,
            "window_key": rep_wk,
            "gt_label": row["gt_label"],
            "gt_actor_uuid": gt_actor_uuid,
            "gt_actor_rank": gt_rank,
            "llm_primary_actor_id": pa_id,
            "llm_primary_actor_uuid": pa_uuid,
            "actor_match": actor_match,
        })

    gt_actor_df = pd.DataFrame(gt_actor_rows)
    # Only logs where GT actor exists AND is within top-3
    gt_actor_valid = gt_actor_df[gt_actor_df["gt_actor_uuid"].notna()].copy()
    n_gt_actor_logs = len(gt_actor_valid)
    n_llm_correct_actor = gt_actor_valid["actor_match"].sum()

    actor_gt_ratio = (n_llm_correct_actor / n_gt_actor_logs
                      if n_gt_actor_logs > 0 else 0.0)
    print(f"[RESULT] Backend {backend_name}: LLM primary actor matches "
          f"GT actor (track_uuid) in {n_llm_correct_actor}/{n_gt_actor_logs} "
          f"({actor_gt_ratio:.3f})")

    # ---------------- New-discovery logs ----------------

    gt_log_set = set(gt_logs)
    logs_with_llm = set(joined["log_id"].dropna().astype(str).unique())
    # Logs in split with no GT tag (for this split) but with evidence.
    candidate_new_logs = (all_logs_in_split - gt_log_set)
    new_rows = []
    for log_id in sorted(candidate_new_logs):
        log_llm = joined[joined["log_id"] == log_id]
        if log_llm.empty:
            continue
        preds = log_llm["scenario_classification"].dropna().tolist()
        # Consider as new-discovery if at least one non-'other' scenario appears.
        non_other = [p for p in preds if p is not None and p != "other"]
        if not non_other:
            continue
        # Summarize majority or just list
        counts = Counter(non_other)
        top_pred, top_cnt = counts.most_common(1)[0]
        new_rows.append({
            "log_id": log_id,
            "distinct_non_other_scenarios": sorted(set(non_other)),
            "top_scenario": top_pred,
            "top_scenario_count": int(top_cnt),
            "num_windows_with_predictions": int(len(preds)),
        })

    new_discovery_df = pd.DataFrame(new_rows)
    n_new_logs = len(new_discovery_df)
    print(f"[RESULT] Backend {backend_name}: new-discovery logs = {n_new_logs}")

    # ---------------- Save all artifacts ----------------

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Confusion & per-class metrics
    confusion_path = eval_dir / f"confusion_matrix_{backend_name}.csv"
    scenario_metrics_path = eval_dir / f"scenario_metrics_{backend_name}.csv"
    confusion_df.to_csv(confusion_path)
    per_class_df.to_csv(scenario_metrics_path, index=False)
    print(f"[INFO] Wrote per-class scenario metrics to {scenario_metrics_path}")
    print(f"[INFO] Wrote confusion matrix to {confusion_path}")

    # Log-level eval table
    log_eval_path = eval_dir / f"log_level_eval_{backend_name}.csv"
    gt_eval_with_llm.to_csv(log_eval_path, index=False)
    print(f"[INFO] Wrote log-level eval table to {log_eval_path}")

    # Actor alignment tables
    actor_align_path = eval_dir / f"actor_alignment_{backend_name}.csv"
    actor_df.to_csv(actor_align_path, index=False)
    print(f"[INFO] Wrote LLM vs GAT actor alignment table to {actor_align_path}")

    gt_actor_eval_path = eval_dir / f"gt_actor_eval_{backend_name}.csv"
    gt_actor_df.to_csv(gt_actor_eval_path, index=False)
    print(f"[INFO] Wrote GT-actor vs LLM primary actor table to {gt_actor_eval_path}")

    # New-discovery summary
    new_disc_path = eval_dir / f"new_discoveries_{backend_name}.csv"
    new_discovery_df.to_csv(new_disc_path, index=False)
    print(f"[INFO] Wrote new-discovery summary to {new_disc_path}")

    # Summary metrics CSV
    summary_rows = [
        {"metric": "n_gt_logs", "value": n_gt_logs},
        {"metric": "n_gt_logs_with_llm", "value": n_gt_with_llm},
        {"metric": "scenario_accuracy_rep", "value": scenario_acc},
        {"metric": "log_hit_accuracy_any_window", "value": log_hit_acc},
        {"metric": "micro_accuracy", "value": global_metrics["micro_accuracy"]},
        {"metric": "macro_precision", "value": global_metrics["macro_precision"]},
        {"metric": "macro_recall", "value": global_metrics["macro_recall"]},
        {"metric": "macro_f1", "value": global_metrics["macro_f1"]},
        {"metric": "n_windows_with_primary_actor", "value": n_total_pa},
        {"metric": "n_pa_equals_gat_rank1", "value": n_pa_equals_gat1},
        {"metric": "ratio_pa_equals_gat_rank1", "value": ratio_pa_equals_gat1},
        {"metric": "n_pa_in_top2", "value": n_pa_in_top2},
        {"metric": "ratio_pa_in_top2", "value": ratio_pa_in_top2},
        {"metric": "n_gt_logs_with_gt_actor", "value": n_gt_actor_logs},
        {"metric": "n_llm_correct_gt_actor", "value": n_llm_correct_actor},
        {"metric": "ratio_llm_correct_gt_actor", "value": actor_gt_ratio},
        {"metric": "n_new_discovery_logs", "value": n_new_logs},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = eval_dir / f"summary_metrics_{backend_name}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Wrote summary metrics to {summary_path}")

    # Return some key metrics for overall / cross-backend info.
    return {
        "backend": backend_name,
        "scenario_accuracy_rep": scenario_acc,
        "log_hit_accuracy": log_hit_acc,
        "micro_accuracy": global_metrics["micro_accuracy"],
        "macro_f1": global_metrics["macro_f1"],
        "n_new_discovery_logs": n_new_logs,
    }


def compute_cross_backend_agreement(backends: dict, eval_dir: Path) -> None:
    """
    backends: dict[name -> llm_df]
    Compute scenario_classification agreement on windows shared by both
    backends (e.g. azure vs ollama).
    """
    if len(backends) < 2:
        print("[INFO] Only one backend; skipping cross-backend agreement.")
        return

    # Build one big table: window_key x backend -> scenario
    dfs = []
    for name, df in backends.items():
        dfs.append(df[["window_key", "scenario_classification"]].rename(
            columns={"scenario_classification": f"scenario_{name}"}
        ))
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="window_key", how="outer")

    # For now: only compare first two backends (typical: azure vs ollama)
    names = list(backends.keys())
    b1, b2 = names[0], names[1]
    col1, col2 = f"scenario_{b1}", f"scenario_{b2}"
    sub = merged[[ "window_key", col1, col2 ]].dropna(subset=[col1, col2])

    n_total = len(sub)
    n_agree = (sub[col1] == sub[col2]).sum()
    agreement = n_agree / n_total if n_total > 0 else 0.0

    print(f"\n[INFO] Computing cross-backend agreement between "
          f"{b1} and {b2}.")
    print(f"[RESULT] Cross-backend agreement ({b1} vs {b2}) "
          f"= {n_agree}/{n_total} ({agreement:.3f})")

    sub["agree"] = (sub[col1] == sub[col2])
    cross_path = eval_dir / "cross_backend_agreement.csv"
    sub.to_csv(cross_path, index=False)
    print(f"[INFO] Wrote cross-backend comparison table to {cross_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True,
                    help="Directory with S4.1 evidence JSONs for this split.")
    ap.add_argument("--llm-root", required=True,
                    help="Root dir containing llm_<backend> subdirs.")
    ap.add_argument("--tags-norm", required=True,
                    help="Path to tags_norm.jsonl (global GT).")
    ap.add_argument("--window-labels", default=None,
                    help="Optional window_labels.jsonl for this split "
                         "(to restrict GT logs).")
    args = ap.parse_args()

    evidence_dir = args.evidence_dir
    llm_root = args.llm_root
    tags_norm_path = args.tags_norm
    window_labels_path = args.window_labels

    eval_dir = Path(llm_root) / "eval_updated"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load evidence
    evid_df = load_evidences(evidence_dir)
    all_logs_in_split = set(evid_df["log_id"].astype(str).unique())
    print(f"[INFO] Distinct logs in this split (from evidence): "
          f"{len(all_logs_in_split)}")

    # Load GT tags from tags_norm
    tags_df = load_tags_norm(tags_norm_path, all_logs_in_split, window_labels_path)

    # Load LLM backends
    backends_llm = load_backends(llm_root)

    # Evaluate each backend
    backend_metrics = []
    for backend_name, llm_df in backends_llm.items():
        metrics = evaluate_backend(
            backend_name=backend_name,
            llm_df=llm_df,
            evid_df=evid_df,
            tags_df=tags_df,
            all_logs_in_split=all_logs_in_split,
            eval_dir=eval_dir,
        )
        backend_metrics.append(metrics)

    # Cross-backend agreement
    compute_cross_backend_agreement(backends_llm, eval_dir)

    # Also write a global summary of backends
    global_summary_path = eval_dir / "backend_summary.csv"
    pd.DataFrame(backend_metrics).to_csv(global_summary_path, index=False)
    print(f"\n[INFO] Wrote backend summary to {global_summary_path}")


if __name__ == "__main__":
    main()
