#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final, pipeline-faithful evaluation (ROBUST + CLEAN OUTPUT)

S1 Event Detection (ALL S1 windows):
  - Universe: s1d/final_scores.parquet (N_total)
  - GT positives: s1a/window_labels.jsonl (label==1 rows)
  - Score: post-HMM / score_final column from s1d/final_scores.parquet
  - Metrics (Solution 1 - No True Negatives):
      * GT Performance:
          * recall_total (coverage of GT by any S1 window)
          * temporal (vs GT): latency median (s), IoU median, fragmentation mean
      * Discovery Analysis (GT-overlap vs New-discovery):
          * n_total_s1_windows, n_gt_overlap_windows, n_new_discovery_windows
          * score_gt_overlap_mean/median
          * score_new_discovery_mean/median
      * Per-scenario: recall only
  - Operational counts (NOT a confusion matrix):
      * n_total, n_kept (true_windows.jsonl), n_discarded

S3 Actor Ranking:
  - Robust GT matching (nearest-with-GT -> any-with-GT -> max-IoU fallback)
  - GT metrics on considered rows: Top-1, R@K, AvgBestRank, MRR, PairwiseAcc
  - Teacher NDCG@K on ALL predicted windows + NEW-DISCOVERY windows
  - Per-scenario from considered rows

Catalogs (lists at end of metrics.json):
  - total_windows_by_log (with #windows)
  - gt_windows_by_log (with #windows)
  - new_discovery_windows_by_log (kept∖GT, with #windows)
  - discarded_windows_by_log (with #windows) — empty if none

Run example (VAL50):
  python scripts/s3_actor_ranking/evaluate_s1_to_s3.py \
    --split-root artifacts/train650_val50/val50 \
    --graphs-split val50 \
    --pred-parquet artifacts/train650_val50/val50/s3/quasi/graphs/gat_final_eval_ckpt2/top3_infer.parquet \
    --window-labels artifacts/train650_val50/val50/s1a/window_labels.jsonl \
    --tags-norm artifacts/train650_val50/val50/s1a/tags_norm.jsonl \
    --s1d-final-scores artifacts/train650_val50/val50/s1d/final_scores.parquet \
    --s1d-true-windows artifacts/train650_val50/val50/s1d/true_windows.jsonl \
    --teacher-parquet artifacts/train650_val50/val50/s3/quasi/teacher.parquet \
    --k 3 \
    --near-sec 3.0 \
    --outdir artifacts/train650_val50/val50/eval_pipeline_final_clean
"""

import argparse, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------- Utils ----------------
def parse_wk(wk: str) -> Tuple[str, float, float]:
    p = str(wk).split("|")
    if len(p) != 3:
        return str(wk), float("nan"), float("nan")
    return p[0], float(p[1]), float(p[2])

def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = max(0.0, right - left)
    union = max(1e-9, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union

def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    k = max(1, min(k, len(scores)))
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    # log2 discount, gains already "q" (no 2^g - 1 trick needed since teacher q is [0,1])
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg

def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_json(path, lines=True)

def coalesce_cols(df: pd.DataFrame, candidates: List[str], new_name: str) -> pd.DataFrame:
    """Create/overwrite new_name by first-non-null across candidates that exist."""
    vals = None
    for c in candidates:
        if c in df.columns:
            v = df[c]
            if vals is None:
                vals = v
            else:
                vals = vals.where(vals.notna(), v)
    if vals is None:
        # create empty column
        df[new_name] = np.nan
    else:
        df[new_name] = vals
    return df

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------------- Loaders ----------------
def load_window_labels(path: Path) -> pd.DataFrame:
    df = read_jsonl(path).copy()
    df = df[df.get("label", 1) == 1].copy()
    need = {"log_id","t_start","t_end"}
    if not need.issubset(df.columns):
        raise KeyError(f"{path} missing {need}")
    if "scenario_label" not in df.columns:
        df["scenario_label"] = "unknown"
    df["log_id"] = df["log_id"].astype(str)
    df["t_start"] = df["t_start"].astype(float)
    df["t_end"]   = df["t_end"].astype(float)
    df["_center"] = 0.5 * (df["t_start"] + df["t_end"])
    return df[["log_id","t_start","t_end","_center","scenario_label"]]

def load_tags_norm(path: Path) -> pd.DataFrame:
    df = read_jsonl(path).copy()
    df = df[(df.get("has_guest", True) == True) & df["guest_id"].notna()].copy()
    df["log_id"] = df["log_id"].astype(str)
    df["guest_id"] = df["guest_id"].astype(str)
    df.rename(columns={"tag":"scenario"}, inplace=True)
    if "scenario" not in df.columns:
        df["scenario"] = "unknown"
    return df[["log_id","guest_id","scenario"]]

def load_s1d_scores(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    col_log = "log_id"
    col_ts  = "t_start" if "t_start" in df.columns else ("start" if "start" in df.columns else None)
    col_te  = "t_end"   if "t_end"   in df.columns else ("end"   if "end"   in df.columns else None)
    col_sc  = "post_hmm" if "post_hmm" in df.columns else ("score_final" if "score_final" in df.columns else None)
    for c in [col_log, col_ts, col_te, col_sc]:
        if c is None:
            raise KeyError(f"{path} missing required columns (need log_id, t_start, t_end, post_hmm/score_final)")
    out = df[[col_log, col_ts, col_te, col_sc]].copy()
    out.columns = ["log_id","t_start","t_end","score"]
    out["log_id"] = out["log_id"].astype(str)
    out["t_start"] = out["t_start"].astype(float)
    out["t_end"]   = out["t_end"].astype(float)
    out["_center"] = 0.5 * (out["t_start"] + out["t_end"])
    out["window_key"] = out.apply(lambda r: f"{r['log_id']}|{r['t_start']}|{r['t_end']}", axis=1)
    return out

def load_true_windows(path: Path) -> pd.DataFrame:
    df = read_jsonl(path).copy()
    # May contain window_key or explicit fields or both; standardize robustly
    if "window_key" in df.columns:
        df["window_key"] = df["window_key"].astype(str)
        parsed = df["window_key"].map(lambda s: s.split("|"))
        df["log_id_from_wk"] = parsed.map(lambda x: x[0] if len(x) > 0 else np.nan)
        df["t_start_from_wk"] = parsed.map(lambda x: float(x[1]) if len(x) > 1 else np.nan)
        df["t_end_from_wk"]   = parsed.map(lambda x: float(x[2]) if len(x) > 2 else np.nan)
    else:
        df["window_key"] = np.nan

    # Coalesce any provided explicit fields with parsed ones
    for c in ["log_id","t_start","t_end"]:
        if c not in df.columns:
            df[c] = np.nan
    df = coalesce_cols(df, ["log_id","log_id_from_wk"], "log_id")
    df = coalesce_cols(df, ["t_start","t_start_from_wk"], "t_start")
    df = coalesce_cols(df, ["t_end","t_end_from_wk"], "t_end")

    # Construct window_key if missing
    m_wk = df["window_key"].isna() | (df["window_key"] == "")
    if m_wk.any():
        df.loc[m_wk, "window_key"] = df.loc[m_wk].apply(
            lambda r: f"{str(r['log_id'])}|{float(r['t_start'])}|{float(r['t_end'])}", axis=1
        )
    df["_center"] = 0.5 * (df["t_start"].astype(float) + df["t_end"].astype(float))
    return df[["window_key","log_id","t_start","t_end","_center"]].drop_duplicates()

def load_pred_topk(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    need = {"window_key","rank","track_uuid","score"}
    if not need.issubset(df.columns):
        raise KeyError(f"{path} missing {need}")
    parsed = df["window_key"].map(parse_wk)
    df["log_id"]  = parsed.map(lambda t: t[0]).astype(str)
    df["t_start"] = parsed.map(lambda t: t[1]).astype(float)
    df["t_end"]   = parsed.map(lambda t: t[2]).astype(float)
    df["_center"] = 0.5 * (df["t_start"] + df["t_end"])
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    return df

# ---------------- S1 evaluation (UPDATED) ----------------
def s1_eval_all(s1d: pd.DataFrame,
                kept: pd.DataFrame,
                gt_wl: pd.DataFrame) -> Dict:
    """
    Compute S1 metrics based on Solution 1 (Practical Fix).
    - We only have positive labels (GT), not confirmed negatives.
    - We evaluate Recall & Temporal Quality on GT windows.
    - We analyze the count and score distribution of "New Discovery" windows.
    """
    gt = gt_wl.copy()
    s1d = s1d.copy() # Make a copy to add a new column

    # --- 1. Label each S1 window as GT-overlap (True) vs New-discovery (False) ---
    s1d["is_gt_overlap"] = False
    for lid, sub in gt.groupby("log_id"):
        cand = s1d[s1d["log_id"] == lid]
        if cand.empty:
            continue
        gt_ts = sub["t_start"].to_numpy()
        gt_te = sub["t_end"].to_numpy()
        c_ts  = cand["t_start"].to_numpy()
        c_te  = cand["t_end"].to_numpy()

        overlaps = np.zeros(len(cand), dtype=bool)
        for i in range(len(gt_ts)):
            inter = np.minimum(c_te, gt_te[i]) - np.maximum(c_ts, gt_ts[i])
            overlaps |= (inter > 0)

        # Mark overlapping windows in the main s1d dataframe
        if np.any(overlaps):
            s1d.loc[cand.index[overlaps], "is_gt_overlap"] = True

    # --- 2. Analyze Discovery Windows (GT-overlap vs New-discovery) ---
    s1d_gt_overlap = s1d[s1d["is_gt_overlap"] == True]
    s1d_new_discovery = s1d[s1d["is_gt_overlap"] == False]

    n_total = len(s1d)
    n_gt_overlap = len(s1d_gt_overlap)
    n_new_discovery = len(s1d_new_discovery)

    s1d_gt_overlap_scores = s1d_gt_overlap["score"].dropna()
    s1d_new_discovery_scores = s1d_new_discovery["score"].dropna()

    discovery_analysis = {
        "n_total_s1_windows": n_total,
        "n_gt_overlap_windows": n_gt_overlap,
        "n_new_discovery_windows": n_new_discovery,
        "score_gt_overlap_mean": float(s1d_gt_overlap_scores.mean()) if not s1d_gt_overlap_scores.empty else 0.0,
        "score_gt_overlap_median": float(s1d_gt_overlap_scores.median()) if not s1d_gt_overlap_scores.empty else 0.0,
        "score_new_discovery_mean": float(s1d_new_discovery_scores.mean()) if not s1d_new_discovery_scores.empty else 0.0,
        "score_new_discovery_median": float(s1d_new_discovery_scores.median()) if not s1d_new_discovery_scores.empty else 0.0,
    }

    # --- 3. Temporal stats vs GT (using ALL S1 windows) ---
    # This logic remains valid, as it iterates over GT and finds the best match.
    latencies, best_ious, frags = [], [], []
    for _, r in gt.iterrows():
        lid = r["log_id"]; ts = r["t_start"]; te = r["t_end"]
        g_center = 0.5*(ts+te)
        cand = s1d[s1d["log_id"] == lid]
        if cand.empty:
            continue
        inter = np.minimum(cand["t_end"].to_numpy(), te) - np.maximum(cand["t_start"].to_numpy(), ts)
        m = inter > 0
        if not np.any(m):
            continue

        centers = 0.5*(cand["t_start"].to_numpy()+cand["t_end"].to_numpy())
        deltas = np.abs(centers - g_center)
        nearest = np.where(m)[0][np.argmin(deltas[m])]
        latencies.append(float(centers[nearest] - g_center))

        unions = (cand["t_end"].to_numpy()-cand["t_start"].to_numpy()) + (te-ts) - np.clip(inter, a_min=0.0, a_max=None)
        ious = np.clip(inter, a_min=0.0, a_max=None) / np.maximum(unions, 1e-9)
        best_ious.append(float(np.max(ious[m])))
        frags.append(int(np.sum(m)))

    temporal = {
        "latency_median_s": float(np.median(latencies)) if latencies else 0.0,
        "iou_median": float(np.median(best_ious)) if best_ious else 0.0,
        "fragmentation_mean": float(np.mean(frags)) if frags else 0.0
    }

    # --- 4. Per-scenario recall & Total Recall (coverage by ANY S1 window) ---
    # This logic also remains valid.
    per_scen = {}
    total_gt_windows = 0
    total_gt_covered = 0
    for scen, gsub in gt.groupby("scenario_label"):
        covered = 0
        total_gt_windows += len(gsub)
        for _, rr in gsub.iterrows():
            lid = rr["log_id"]; ts = rr["t_start"]; te = rr["t_end"]
            cand = s1d[s1d["log_id"] == lid]
            if cand.empty:
                continue
            inter = np.minimum(cand["t_end"].to_numpy(), te) - np.maximum(cand["t_start"].to_numpy(), ts)
            if np.any(inter > 0):
                covered += 1

        total_gt_covered += covered
        per_scen[scen] = {
            "n_windows": int(len(gsub)),
            "recall": covered / max(1, len(gsub))
        }

    gt_recall_total = total_gt_covered / max(1, total_gt_windows)

    # --- 5. Operational counts (NOT a confusion matrix) ---
    # This logic is independent and still valuable.
    n_total_op = int(len(s1d)) # Use n_total from discovery analysis
    kept_set = set(kept["window_key"].astype(str).tolist())
    n_kept = int(len(kept_set))
    n_discarded = int(max(0, n_total_op - n_kept))

    operational_counts = {
        "n_s1_windows_total": n_total_op,
        "n_kept": n_kept,
        "n_discarded": n_discarded
    }

    # --- Assemble Final Report ---
    return {
        "discovery_analysis": discovery_analysis,
        "gt_performance": {
            "gt_recall_total": gt_recall_total,
            "temporal": temporal
        },
        "per_scenario": per_scen,
        "operational_counts": operational_counts
    }

# ---------------- S3 evaluation (UNCHANGED) ----------------
def build_gt_actor_map(tags_norm: pd.DataFrame) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    for lid, sub in tags_norm.groupby("log_id"):
        mp[lid] = [str(x) for x in sub["guest_id"].tolist()]
    return mp

def s3_eval(gt_wl: pd.DataFrame,
            preds: pd.DataFrame,
            tags_norm: pd.DataFrame,
            teacher: Optional[pd.DataFrame],
            k: int,
            near_sec: float) -> Dict:
    gt = gt_wl.copy()
    preds = preds.copy()
    gt_map = build_gt_actor_map(tags_norm)

    # Discovery split: which predicted windows overlap any GT window
    wk_with_gt_overlap = set()
    for lid, gsub in gt.groupby("log_id"):
        ps = preds[preds["log_id"] == lid]
        if ps.empty:
            continue
        for _, gr in gsub.iterrows():
            ts, te = float(gr["t_start"]), float(gr["t_end"])
            ss = ps.copy()
            inter = np.minimum(ss["t_end"].to_numpy(), te) - np.maximum(ss["t_start"].to_numpy(), ts)
            iou = np.clip(inter, 0.0, None) / ((ss["t_end"]-ss["t_start"]) + (te-ts) - np.clip(inter, 0.0, None) + 1e-9)
            wk_with_gt_overlap |= set(ss.loc[iou > 0, "window_key"].tolist())
    all_pred_wk = set(preds["window_key"].unique().tolist())
    wk_new_discovery = all_pred_wk - wk_with_gt_overlap

    # Considered GT rows with robust policy
    gt["center"] = 0.5*(gt["t_start"]+gt["t_end"])
    preds["center"] = 0.5*(preds["t_start"]+preds["t_end"])
    man = preds[["window_key","log_id","t_start","t_end","_center"]].drop_duplicates()

    considered = []
    pred_wk_set = set(preds["window_key"].unique())
    for idx, g in gt.iterrows():
        lid, gc = g["log_id"], float(g["center"])
        cands = man[man["log_id"] == lid].copy()
        if cands.empty:
            continue
        cands["near"] = (cands["_center"] - gc).abs() <= near_sec

        def choose_best_with_gt(keys: List[str]):
            if not keys:
                return "", 999
            sub = preds[preds["window_key"].isin(keys)].copy()
            if sub.empty:
                return "", 999
            gt_actors = set(gt_map.get(lid, []))
            best_rank, best_wk, best_score, best_dt = 999, "", -1.0, 1e9
            centers = cands.set_index("window_key")["_center"].to_dict()
            for wk in keys:
                rows = sub[sub["window_key"] == wk].sort_values("rank")
                if rows.empty:
                    continue
                rmin = 999; top_gt_score = -1.0
                for a in gt_actors:
                    rr = rows[rows["track_uuid"] == a]
                    if not rr.empty:
                        rmin = min(rmin, int(rr["rank"].min()))
                        top_gt_score = max(top_gt_score, float(rr["score"].max()))
                if rmin == 999:
                    continue
                dt = abs(centers.get(wk, gc) - gc)
                better = (rmin < best_rank) or \
                         (rmin == best_rank and top_gt_score > best_score) or \
                         (rmin == best_rank and abs(top_gt_score - best_score) < 1e-12 and dt < best_dt)
                if better:
                    best_rank, best_wk, best_score, best_dt = rmin, wk, top_gt_score, dt
            return best_wk, best_rank

        near_wks = cands[cands["near"]]["window_key"].tolist()
        used_wk, best_rank = choose_best_with_gt(near_wks)
        if not used_wk:
            any_wks = cands["window_key"].tolist()
            used_wk, best_rank = choose_best_with_gt(any_wks)
        if not used_wk:
            cands["iou"] = cands.apply(lambda r: interval_iou(g["t_start"], g["t_end"], r["t_start"], r["t_end"]), axis=1)
            pick = cands.sort_values(["iou","_center"], ascending=[False, True]).head(1)
            if not pick.empty:
                used_wk = str(pick["window_key"].iloc[0]); best_rank = 999

        if used_wk and used_wk in pred_wk_set:
            topk = preds[preds["window_key"] == used_wk].sort_values("rank").head(k)
            considered.append((idx, used_wk, topk))

    n_total = int(len(gt))
    n_considered = int(len(considered))

    # GT metrics
    top1_hits = 0; hit_at_k = 0
    ranks_list = []; mrr_list = []; pairwise_acc_list = []
    per_scen = {}
    for idx, wk, topk in considered:
        g = gt.loc[idx]
        scen = g["scenario_label"]; lid = g["log_id"]
        gt_actors = set(gt_map.get(lid, []))
        actors = [str(x) for x in topk["track_uuid"].tolist()]
        scores = topk["score"].to_numpy(dtype=float)
        rank_gt = None
        for r, a in enumerate(actors, start=1):
            if a in gt_actors:
                rank_gt = r
                break
        if rank_gt is not None:
            if rank_gt == 1: top1_hits += 1
            if rank_gt <= k: hit_at_k += 1
            ranks_list.append(rank_gt)
            mrr_list.append(1.0/rank_gt)
            s_gt = float(scores[rank_gt-1])
            neg_scores = [float(scores[j]) for j,a in enumerate(actors) if a not in gt_actors]
            if neg_scores:
                pairwise_acc_list.append(sum(1 for s in neg_scores if s_gt > s)/len(neg_scores))
        else:
            mrr_list.append(0.0); pairwise_acc_list.append(0.0)
        d = per_scen.setdefault(scen, {"n":0,"top1":0,"hitk":0,"ranks":[],"mrr":[]})
        d["n"] += 1
        if rank_gt is not None:
            d["ranks"].append(rank_gt); d["mrr"].append(1.0/rank_gt)
            if rank_gt == 1: d["top1"] += 1
            if rank_gt <= k: d["hitk"] += 1
        else:
            d["mrr"].append(0.0)

    s3_gt = {
        "n_windows_total": n_total,
        "n_windows_considered": n_considered,
        "top1": top1_hits / max(1, n_considered),
        "r@{}".format(k): hit_at_k / max(1, n_considered),
        "avg_best_rank": float(np.mean(ranks_list)) if ranks_list else 0.0,
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "pairwise_acc": float(np.mean(pairwise_acc_list)) if pairwise_acc_list else 0.0,
    }
    s3_per_scen = {}
    for scen, d in per_scen.items():
        n = d["n"]
        s3_per_scen[scen] = {
            "n_windows": int(n),
            "gt_top1": d["top1"]/max(1,n),
            "gt_r@{}".format(k): d["hitk"]/max(1,n),
            "gt_avg_best_rank": float(np.mean(d["ranks"])) if d["ranks"] else 0.0,
            "gt_mrr": float(np.mean(d["mrr"])) if d["mrr"] else 0.0,
        }

    # Teacher NDCG@K (all + new-discovery)
    teacher_block = None
    if teacher is not None and not teacher.empty:
        teacher = teacher.copy()
        teacher["window_key"] = teacher["window_key"].astype(str)
        teacher["track_uuid"] = teacher["track_uuid"].astype(str)
        teacher["q"] = teacher["q"].astype(float)
        tmap = {wk: sub.set_index("track_uuid")["q"].to_dict() for wk, sub in teacher.groupby("window_key")}
        ndcgs_all = []; ndcgs_new = []
        wk_all = set(); wk_new = set()
        for wk, grp in preds.groupby("window_key"):
            if wk not in tmap:
                continue
            rows = grp.sort_values("rank")
            actors = [str(x) for x in rows["track_uuid"].tolist()]
            gains = np.array([float(tmap[wk].get(a, 0.0)) for a in actors], dtype=float)
            if not np.any(gains > 0):
                continue
            scores = rows["score"].to_numpy(dtype=float)
            val = ndcg_at_k(scores, gains, k)
            ndcgs_all.append(val); wk_all.add(wk)
            if wk in wk_new_discovery:
                ndcgs_new.append(val); wk_new.add(wk)
        teacher_block = {
            "n_windows_all": int(len(wk_all)),
            "ndcg@{}".format(k): float(np.mean(ndcgs_all)) if ndcgs_all else 0.0,
            "n_windows_new_discovery": int(len(wk_new)),
            "ndcg@{}_new_discovery".format(k): float(np.mean(ndcgs_new)) if ndcgs_new else 0.0
        }

    return {
        "gt": s3_gt,
        "teacher": teacher_block,
        "per_scenario": s3_per_scen,
        "counts": {
            "predicted_windows_total": int(len(preds["window_key"].unique())),
            "predicted_windows_with_gt_overlap": int(len(wk_with_gt_overlap)),
            "predicted_windows_new_discovery": int(len(all_pred_wk - wk_with_gt_overlap))
        }
    }

# ---------------- Catalog builders (UNCHANGED) ----------------
def list_windows_by_log(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty:
        return []
    need = ["log_id","t_start","t_end"]
    for c in need:
        if c not in df.columns:
            return []
    g = df.groupby("log_id").size().reset_index(name="n_windows").sort_values(
        ["n_windows","log_id"], ascending=[False, True]
    )
    out = []
    for _, r in g.iterrows():
        out.append({"log_id": str(r["log_id"]), "count": int(r["n_windows"])})
    return out

# ---------------- Main function (UNCHANGED) ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--graphs-split", required=True)
    ap.add_argument("--pred-parquet", required=True)
    ap.add_argument("--window-labels", required=True)
    ap.add_argument("--tags-norm", required=True)
    ap.add_argument("--s1d-final-scores", required=True)
    ap.add_argument("--s1d-true-windows", required=True)
    ap.add_argument("--teacher-parquet", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--near-sec", type=float, default=3.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    split_root = Path(args.split_root)
    graphs_dir = split_root / "s3" / "quasi" / "graphs" / args.graphs_split
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    wl = load_window_labels(Path(args.window_labels))
    tn = load_tags_norm(Path(args.tags_norm))
    s1d = load_s1d_scores(Path(args.s1d_final_scores))
    kept = load_true_windows(Path(args.s1d_true_windows))
    preds = load_pred_topk(Path(args.pred_parquet))
    teacher = pd.read_parquet(args.teacher_parquet) if args.teacher_parquet else None

    # ---------- S1 (Uses new eval function) ----------
    S1 = s1_eval_all(s1d, kept, wl)

    # ---------- S3 (Uses original eval function) ----------
    S3 = s3_eval(wl, preds, tn, teacher, k=args.k, near_sec=args.near_sec)

    # ---------- Catalogs (Unchanged) ----------
    # New discoveries = kept that do not overlap any GT window
    # First, make a robust kept_df with consistent column names after potential merges
    # Merge kept with s1d on window_key to guarantee columns; handle suffixes safely.
    s1d_min = s1d[["window_key","log_id","t_start","t_end"]].copy()
    kept_df = kept.copy()
    # If kept already has these columns, we will coalesce after merge.
    merged = kept_df.merge(s1d_min, on="window_key", how="left", suffixes=("_kept","_s1d"))
    # Coalesce to canonical column names
    merged = coalesce_cols(merged, ["log_id_kept","log_id","log_id_s1d"], "log_id")
    merged = coalesce_cols(merged, ["t_start_kept","t_start","t_start_s1d"], "t_start")
    merged = coalesce_cols(merged, ["t_end_kept","t_end","t_end_s1d"], "t_end")
    merged = ensure_cols(merged, ["log_id","t_start","t_end"])
    # Keep only canonical columns
    kept_df_std = merged[["window_key","log_id","t_start","t_end"]].copy()
    # Drop rows that still have NaNs after coalescing (if any)
    kept_df_std = kept_df_std.dropna(subset=["log_id","t_start","t_end"]).copy()
    if kept_df_std.empty:
        # still provide valid empty frame
        kept_df_std = pd.DataFrame(columns=["window_key","log_id","t_start","t_end"])

    # Discarded = s1d windows that are not in kept
    kept_set = set(kept["window_key"].astype(str).tolist())
    discarded_df = s1d[~s1d["window_key"].isin(kept_set)][["log_id","t_start","t_end"]].copy()
    if discarded_df.empty:
        discarded_df = pd.DataFrame(columns=["log_id","t_start","t_end"])

    # Overlap function
    def overlap_any(win_row, gt_rows):
        lid, ts, te = win_row["log_id"], float(win_row["t_start"]), float(win_row["t_end"])
        sub = wl[wl["log_id"] == lid]
        if sub.empty:
            return False
        inter = np.minimum(sub["t_end"].to_numpy(), te) - np.maximum(sub["t_start"].to_numpy(), ts)
        return bool(np.any(inter > 0))

    if kept_df_std.empty:
        kept_df_std["is_new"] = pd.Series(dtype=bool)
    else:
        kept_df_std["is_new"] = kept_df_std.apply(lambda r: not overlap_any(r, wl), axis=1)

    catalogs = {
        "total_windows_by_log": list_windows_by_log(s1d[["log_id","t_start","t_end"]]),
        "gt_windows_by_log": list_windows_by_log(wl[["log_id","t_start","t_end"]]),
        "new_discovery_windows_by_log": list_windows_by_log(kept_df_std[kept_df_std["is_new"]][["log_id","t_start","t_end"]]),
        "discarded_windows_by_log": list_windows_by_log(discarded_df)
    }

    report = {
        "split_root": str(split_root),
        "graphs_dir": str(graphs_dir),
        "pred_parquet": str(Path(args.pred_parquet)),
        "near_sec": float(args.near_sec),
        "S1_event_detection": S1,
        "S3_actor_ranking": S3,
        "catalogs": catalogs,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
