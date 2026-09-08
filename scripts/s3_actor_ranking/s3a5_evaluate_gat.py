#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAL50 Final Evaluation (window-true, predictions-only)
[MODIFIED to include MRR and Pairwise Accuracy]

- Each GT window from window_labels.jsonl is treated independently.
- For each GT row, we select one manifest window in the same log:
    1) prefer windows within ±NEAR_SEC that contain any GT actor in Top-3 (break ties by best rank -> best score -> nearest center)
    2) else any window in the log that contains a GT actor in Top-3
    3) else fallback to max-IoU (counts as miss if no GT actor in Top-3)

- IMPORTANT: A GT row is INCLUDED in metrics ONLY IF the chosen used_window exists
  in the provided Top-3 predictions parquet (pred_parquet).
  This ensures we evaluate only on windows for which we have predictions.

Outputs:
  A configurable evaluation directory containing metrics and diagnostics.
    - metrics.json
    - gt_match_diag.csv  (per-GT row diagnostics)
    - top3_per_gt.parquet (optional: Top-3 rows for the used window of each *considered* GT row)
"""

import argparse, json, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------- Utils ----------------
def parse_wk(wk: str) -> Tuple[str, float, float]:
    p = str(wk).split("|")
    return p[0], float(p[1]), float(p[2])

def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = max(0.0, right - left)
    union = max(1e-6, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union

def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    k = max(1, min(k, len(scores)))
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    gains_ranked = (2.0 ** gains[order] - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains_ranked * discounts))
    idcg = float(np.sum((2.0 ** ideal - 1.0) * discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g. artifacts/train650_val50/val50")
    ap.add_argument("--pred-parquet", required=True, help="Top-3 predictions parquet (top3_infer.parquet)")
    ap.add_argument("--graphs-dir", default=None,
                    help="Graph directory; defaults to <split-root>/s3/quasi/graphs/<split-name>")
    ap.add_argument("--window-labels", default=None,
                    help="GT window labels; defaults to <split-root>/s1a/window_labels.jsonl")
    ap.add_argument("--tags-norm", default=None,
                    help="Normalized actor tags; defaults to <split-root>/s1a/tags_norm.jsonl")
    ap.add_argument("--teacher-parquet", default=None,
                    help="Teacher scores; defaults to <split-root>/s3/quasi/teacher.parquet")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory; defaults to <graphs-dir>/evaluation")
    ap.add_argument("--near-sec", type=float, default=3.0, help="± seconds around GT center")
    ap.add_argument("--export-top3-per-gt", action="store_true")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    split_name = split_root.name
    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else (
        split_root / "s3" / "quasi" / "graphs" / split_name
    )
    wl_path = Path(args.window_labels) if args.window_labels else (
        split_root / "s1a" / "window_labels.jsonl"
    )
    tn_path = Path(args.tags_norm) if args.tags_norm else (
        split_root / "s1a" / "tags_norm.jsonl"
    )
    man_path = graphs_dir / "manifest.parquet"
    pred_path = Path(args.pred_parquet)

    for p in (wl_path, tn_path, man_path, pred_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    wl = pd.read_json(wl_path, lines=True)
    tn = pd.read_json(tn_path, lines=True)
    man = pd.read_parquet(man_path)
    pred = pd.read_parquet(pred_path)

    # --- normalize / index
    wl = wl.copy()
    wl["log_id"] = wl["log_id"].astype(str)
    wl["scenario_label"] = wl["scenario_label"].astype(str)
    wl["t_start"] = wl["t_start"].astype(float)
    wl["t_end"]   = wl["t_end"].astype(float)
    wl["_center"] = 0.5 * (wl["t_start"] + wl["t_end"])

    tn = tn.copy()
    tn["log_id"] = tn["log_id"].astype(str)
    tn["guest_id"] = tn["guest_id"].astype(str)
    gt_actors_by_log: Dict[str, List[str]] = (
        tn.groupby("log_id")["guest_id"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    )

    parsed = man["window_key"].map(parse_wk)
    man = man.copy()
    man["window_key"] = man["window_key"].astype(str)
    man["_log"] = parsed.map(lambda t: t[0])
    man["_tsr"] = parsed.map(lambda t: t[1])
    man["_ter"] = parsed.map(lambda t: t[2])
    man["_center"] = 0.5 * (man["_tsr"] + man["_ter"])

    pred = pred.copy()
    pred["window_key"] = pred["window_key"].astype(str)
    pred["track_uuid"] = pred["track_uuid"].astype(str)
    pred["rank"] = pred["rank"].astype(int)
    pred["score"] = pred["score"].astype(float)
    pred_wk_set = set(pred["window_key"].unique())
    # <-- NEW --> Create indexed predictions for faster lookup
    pred_i = pred.set_index("window_key")

    # <-- NEW --> Load Teacher Q-map early so it's available for per-scenario metrics
    teacher_parquet = Path(args.teacher_parquet) if args.teacher_parquet else (
        split_root / "s3" / "quasi" / "teacher.parquet"
    )
    qmap = {}
    if teacher_parquet.exists():
        tdf = pd.read_parquet(teacher_parquet)
        tdf["window_key"] = tdf["window_key"].astype(str)
        tdf["track_uuid"] = tdf["track_uuid"].astype(str)
        tdf["q"] = tdf["q"].astype(float)
        qmap = {(wk, tu): float(q) for wk, tu, q in zip(tdf["window_key"], tdf["track_uuid"], tdf["q"])}
    else:
        print(f"Warning: Teacher parquet not found at {teacher_parquet}. NDCG metrics will be 0.")

    # containers
    diag_rows = []
    # keep per-GT window results (we'll filter to considered ones later)
    gt_rows_all = []
    top3_rows_considered = []

    for _, r in wl.iterrows():
        log_id = r["log_id"]
        scen = r["scenario_label"]
        gt_ts = float(r["t_start"]); gt_te = float(r["t_end"])
        gt_center = float(r["_center"])
        gt_actors = gt_actors_by_log.get(log_id, [])

        # candidates in this log
        cands = man.loc[man["_log"] == log_id, ["window_key", "_tsr", "_ter", "_center"]].copy()
        if cands.empty:
            diag_rows.append({
                "log_id": log_id, "scenario": scen,
                "t_start": gt_ts, "t_end": gt_te,
                "status": "no_manifest_for_log",
                "considered_windows": "", "used_window": ""
            })
            gt_rows_all.append({
                "log_id": log_id, "scenario": scen, "used_window": "", "best_rank": 999, "top1": 0, "r@3": 0
            })
            continue

        # annotate near + IoU (use .loc to avoid SettingWithCopy)
        cands = cands.copy()
        cands.loc[:, "near"] = (cands["_center"] - gt_center).abs() <= args.near_sec
        cands.loc[:, "iou"] = cands.apply(
            lambda x: interval_iou(gt_ts, gt_te, x["_tsr"], x["_ter"]), axis=1
        )

        # helper: choose best window (among given keys) that has any GT actor in Top-3
        def choose_best_with_gt(window_keys: List[str]):
            if not gt_actors or not window_keys:
                return "", 999, -1.0, 1e9
            # <-- MODIFIED --> Use indexed predictions
            wks_with_preds = [wk for wk in window_keys if wk in pred_i.index]
            if not wks_with_preds:
                 return "", 999, -1.0, 1e9
            sub = pred.loc[pred["window_key"].isin(wks_with_preds), ["window_key", "track_uuid", "rank", "score"]]
            if sub.empty:
                return "", 999, -1.0, 1e9

            best_rank, best_wk, best_score, best_dt = 999, "", -1.0, 1e9
            centers = cands.set_index("window_key")["_center"].to_dict()

            # Use groupby for efficiency
            sub_grouped = sub.groupby("window_key")
            for wk, rows in sub_grouped:
                # min rank among GT actors
                rmin = 999
                top_gt_score = -1.0
                for a in gt_actors:
                    aa = rows.loc[rows["track_uuid"] == a]
                    if len(aa) > 0:
                        rmin = min(rmin, int(aa["rank"].min()))
                        top_gt_score = max(top_gt_score, float(aa["score"].max()))

                if rmin == 999:  # GT actor not in Top-3
                    continue

                dt = abs(centers.get(wk, gt_center) - gt_center)
                better = (rmin < best_rank) or \
                         (rmin == best_rank and top_gt_score > best_score) or \
                         (rmin == best_rank and abs(top_gt_score - best_score) < 1e-12 and dt < best_dt)
                if better:
                    best_rank, best_wk, best_score, best_dt = rmin, wk, top_gt_score, dt
            return best_wk, best_rank, best_score, best_dt

        # 1) near with GT
        near_wks = cands.loc[cands["near"], "window_key"].tolist()
        used_wk, best_rank, _, _ = choose_best_with_gt(near_wks)
        status = "chose_near_with_gt" if used_wk else ""

        # 2) any with GT
        if not used_wk:
            any_wks = cands["window_key"].tolist()
            used_wk, best_rank, _, _ = choose_best_with_gt(any_wks)
            if used_wk:
                status = "chose_any_with_gt"

        # 3) fallback to max-IoU (even if GT actor not in Top-3)
        if not used_wk:
            pick = cands.sort_values(["iou", "_center"], ascending=[False, True]).head(1)
            if not pick.empty:
                used_wk = str(pick["window_key"].iloc[0])
                best_rank = 999  # GT actor not in Top-3
                status = "fallback_max_iou_no_gt"
            else:
                used_wk = ""
                best_rank = 999
                status = "no_candidates"

        diag_rows.append({
            "log_id": log_id, "scenario": scen,
            "t_start": gt_ts, "t_end": gt_te,
            "status": status,
            "considered_windows": ";".join(cands["window_key"].tolist()),
            "used_window": used_wk
        })

        # store raw (we will filter by presence in PRED later)
        top1 = 1 if best_rank == 1 else 0
        r3   = 1 if best_rank <= 3 else 0
        gt_rows_all.append({
            "log_id": log_id, "scenario": scen,
            "used_window": used_wk,
            "best_rank": best_rank, "top1": top1, "r@3": r3
        })

    diag_df = pd.DataFrame(diag_rows)
    gt_all_df = pd.DataFrame(gt_rows_all)

    # --------- FILTER to "considered" GT rows with predictions ----------
    # (1) exclude rows with no manifest / no candidates (used_window == "")
    # (2) require used_window to be present in pred parquet
    mask_considered = (gt_all_df["used_window"] != "") & (gt_all_df["used_window"].isin(pred_wk_set))
    gt_considered = gt_all_df.loc[mask_considered].copy()

    # <-- NEW --> Get set of considered windows for discovery calculation
    gt_considered_wks = set(gt_considered["used_window"].unique())

    # Optional export: exact Top-3 for the used_window of each considered GT row
    out_dir = Path(args.out_dir) if args.out_dir else (graphs_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.export_top3_per_gt and not gt_considered.empty:
        rows = []
        # <-- MODIFIED --> Use pre-built pred_i
        for _, rr in gt_considered.iterrows():
            wk = rr["used_window"]
            if wk in pred_i.index:
                sub = pred_i.loc[[wk]] # Use [[wk]] to ensure DataFrame
                if isinstance(sub, pd.Series):
                    sub = sub.to_frame().T
                for __, s in sub.sort_values("rank").iterrows():
                    rows.append({
                        "log_id": rr["log_id"],
                        "scenario": rr["scenario"],
                        "used_window": wk,
                        "rank": int(s["rank"]),
                        "track_uuid": str(s["track_uuid"]),
                        "score": float(s["score"]),
                    })
        if rows:
            pd.DataFrame(rows).to_parquet(out_dir / "top3_per_gt.parquet", index=False)

    # --------- Compute GT metrics ONLY on considered rows ----------
    # <-- MODIFIED BLOCK START -->
    n_gt_total = int(len(gt_all_df))
    n_gt_considered = int(len(gt_considered))

    if n_gt_considered > 0:
        top1 = float(gt_considered["top1"].mean())
        r3 = float(gt_considered["r@3"].mean())
        valid = gt_considered.loc[gt_considered["best_rank"] < 999, "best_rank"]
        avg_rank = float(valid.mean()) if len(valid) > 0 else 0.0

        # --- Add MRR and Pairwise Acc ---
        gt_considered = gt_considered.copy() # Avoid SettingWithCopyWarning
        gt_considered['mrr'] = 0.0
        gt_considered['pairwise_acc'] = 0.0

        for idx, row in gt_considered.iterrows():
            wk = row['used_window']
            lid = row['log_id']
            gt_actors = set(gt_actors_by_log.get(lid, []))

            if not gt_actors or wk not in pred_i.index:
                continue

            topk = pred_i.loc[[wk]].sort_values("rank")
            actors = topk["track_uuid"].tolist()
            scores = topk["score"].to_numpy(dtype=float)

            rank_gt = None
            for r_idx, a in enumerate(actors):
                if a in gt_actors:
                    rank_gt = r_idx + 1  # 1-based rank
                    break

            if rank_gt is not None:
                mrr_val = 1.0 / rank_gt
                s_gt = scores[rank_gt - 1]
                neg_scores = [scores[j] for j, a in enumerate(actors) if a not in gt_actors]

                pairwise_val = 0.0
                if neg_scores:
                    pairwise_val = sum(1 for s in neg_scores if s_gt > s) / len(neg_scores)

                gt_considered.loc[idx, 'mrr'] = mrr_val
                gt_considered.loc[idx, 'pairwise_acc'] = pairwise_val

        mrr = float(gt_considered["mrr"].mean())
        pairwise_acc = float(gt_considered["pairwise_acc"].mean())

        print(f"[GT] considered={n_gt_considered}/{n_gt_total}  Top1={top1:.3f}  R@3={r3:.3f}  AvgBestRank={avg_rank:.3f}  MRR={mrr:.3f}  PairwiseAcc={pairwise_acc:.3f}")
        # --- End Add MRR and Pairwise Acc ---

    else:
        top1 = r3 = avg_rank = mrr = pairwise_acc = 0.0
        print(f"[GT] considered=0/{n_gt_total}  (no evaluable GT rows)")

    # <-- MODIFIED BLOCK END -->


    # --------- Per-scenario metrics on considered set ----------
    # <-- MODIFIED BLOCK START -->
    per_scen = {}
    for scen, g in gt_considered.groupby("scenario"):
        n = int(len(g))
        t1 = float(g["top1"].mean())
        r_3 = float(g["r@3"].mean())
        v = g.loc[g["best_rank"] < 999, "best_rank"]
        avr = float(v.mean()) if len(v) > 0 else 0.0

        # Add MRR per scenario
        mrr_scen = float(g["mrr"].mean())

        # <-- This is the per-scenario teacher_ndcg@3 you asked for -->
        scen_ndcg_vals = []
        scenario_wks = set(g["used_window"].unique())
        if qmap: # Only if teacher scores exist
            for wk in scenario_wks:
                if wk not in pred_i.index:
                    continue
                grp = pred_i.loc[[wk]] # Get rows for this window
                scores = grp.sort_values("rank")["score"].to_numpy(dtype=float)
                gains = np.array([qmap.get((wk, tu), 0.0) for tu in grp.sort_values("rank")["track_uuid"].tolist()], dtype=float)
                if np.any(gains > 0):
                    scen_ndcg_vals.append(ndcg_at_k(scores, gains, k=3))

        scen_ndcg = float(np.mean(scen_ndcg_vals)) if scen_ndcg_vals else 0.0

        per_scen[scen] = {
            "n_windows": n,
            "gt_top1": t1,
            "gt_r@3": r_3,
            "gt_avg_best_rank": avr,
            "gt_mrr": mrr_scen,           # <-- ADDED
            "teacher_ndcg@3": scen_ndcg   # <-- Already present
        }
    # <-- MODIFIED BLOCK END -->

    # --------- Teacher NDCG@3 (and Discovery) ----------
    # <-- MODIFIED --> Calculate for all, gt-used, and discovery
    ndcg_vals_all = []
    ndcg_vals_discovery = []

    if qmap: # Only if teacher scores exist
        # Compute NDCG on *all* predicted windows that have any teacher mass on Top-3 actors
        for wk, grp in pred.groupby("window_key"):
            scores = grp.sort_values("rank")["score"].to_numpy(dtype=float)
            gains = np.array([qmap.get((wk, tu), 0.0) for tu in grp.sort_values("rank")["track_uuid"].tolist()], dtype=float)
            if np.any(gains > 0):
                ndcg_val = ndcg_at_k(scores, gains, k=3)
                ndcg_vals_all.append(ndcg_val)

                # Check if this window was used for a GT row
                if wk not in gt_considered_wks:
                    ndcg_vals_discovery.append(ndcg_val)

    teacher_ndcg = float(np.mean(ndcg_vals_all)) if ndcg_vals_all else 0.0
    discovery_ndcg = float(np.mean(ndcg_vals_discovery)) if ndcg_vals_discovery else 0.0

    # --------- Save diagnostics + metrics ----------
    diag_df.to_csv(out_dir / "gt_match_diag.csv", index=False)

    metrics = {
        "split_root": str(split_root),
        "graphs_dir": str(graphs_dir),
        "pred_parquet": str(pred_path),
        "near_sec": float(args.near_sec),
        "gt": {
            "n_windows_total": n_gt_total,
            "n_windows_considered": n_gt_considered,
            "top1": top1,
            "r@3": r3,
            "avg_best_rank": avg_rank,
            "mrr": mrr,                     # <-- ADDED
            "pairwise_acc": pairwise_acc    # <-- ADDED
        },
        "teacher": {
            "n_windows": len(ndcg_vals_all), # <-- MODIFIED -->
            "ndcg@3": teacher_ndcg
        },
        # <-- NEW --> Discovery metrics
        "discovery": {
            "n_windows": len(ndcg_vals_discovery),
            "teacher_ndcg@3": discovery_ndcg
        },
        "per_scenario": per_scen,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[OK] wrote {out_dir/'metrics.json'} and diagnostics")

if __name__ == "__main__":
    main()
