#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thesis-ready visualizations for:
  S1: Event detection
  S3: Actor ranking

Schemas supported (as provided by user):
- episodes.csv (S1D):
    log_id,t0,t1,duration_s,post_max,post_mean,n_windows
- s1c/hmm/scores_hmm.jsonl:
    log_id,t_start,t_end,score_pu_xgb,score_nnpu,score_fused,score_ema,post_hmm,viterbi
- s1a/window_labels.jsonl:
    log_id,t_start,t_end,label,scenario_label
- top3_infer.parquet:
    window_key,rank,track_uuid,score
- s1a/tags_norm.jsonl:
    log_id,tag,guest_id,has_guest,host_id,id
- s3/quasi/teacher.parquet:
    window_key,track_uuid,category,q,score_window_raw

Also consumes:
- S1 metrics JSON (to get operating threshold op_threshold)
- S3 metrics JSON (final_report.json with per-scenario metrics, ndcg@3 overall, etc.)
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Matplotlib defaults ----------
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ---------- Utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)

def parse_wk(wk: str) -> Tuple[str, float, float]:
    p = str(wk).split("|")
    return p[0], float(p[1]), float(p[2])

def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = max(0.0, right - left)
    union = max(1e-9, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    half = (z/denom) * np.sqrt((p*(p-1)/-n) + (z**2)/(4*n**2))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    accs, confs, counts = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if m.sum() == 0:
            accs.append(np.nan); confs.append((bins[b]+bins[b+1])/2); counts.append(0)
        else:
            accs.append(y_true[m].mean())
            confs.append(y_prob[m].mean())
            counts.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(counts)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=n_bins)
    n = np.sum(counts)
    if n == 0:
        return 0.0
    return float(np.nansum((counts/n) * np.abs(accs - confs)))

def brier_score(y_true: np.ndarray, y_prob: np.ndarray):
    return float(np.mean((y_prob - y_true) ** 2))

def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    if len(scores) == 0 or len(gains) == 0: return 0.0
    k = max(1, min(k, len(scores)))
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0 else dcg / idcg

# ---------- S1 loaders & computations ----------
def load_s1_inputs(scores_path: Path, episodes_path: Path, wl_path: Path):
    # scores
    scores = read_jsonl(scores_path).copy()
    need = {"log_id","t_start","t_end","post_hmm"}
    if not need.issubset(scores.columns):
        raise KeyError(f"{scores_path} missing {need}")
    scores["log_id"] = scores["log_id"].astype(str)
    scores["t_start"] = scores["t_start"].astype(float)
    scores["t_end"]   = scores["t_end"].astype(float)
    scores["_center"] = 0.5*(scores["t_start"]+scores["t_end"])

    # episodes (accept t0/t1 or seg_start/seg_end)
    episodes = pd.read_csv(episodes_path).copy()
    episodes["log_id"] = episodes["log_id"].astype(str)
    if {"t0","t1"}.issubset(episodes.columns):
        episodes["seg_start"] = pd.to_numeric(episodes["t0"], errors="coerce")
        episodes["seg_end"]   = pd.to_numeric(episodes["t1"], errors="coerce")
    elif {"seg_start","seg_end"}.issubset(episodes.columns):
        episodes["seg_start"] = pd.to_numeric(episodes["seg_start"], errors="coerce")
        episodes["seg_end"]   = pd.to_numeric(episodes["seg_end"], errors="coerce")
    else:
        raise KeyError("episodes.csv must have t0,t1 or seg_start,seg_end columns.")
    episodes = episodes.dropna(subset=["seg_start","seg_end"]).copy()

    # GT windows
    wl = read_jsonl(wl_path)
    wl = wl[wl.get("label",1) == 1].copy()
    wl["log_id"] = wl["log_id"].astype(str)
    wl["t_start"] = wl["t_start"].astype(float)
    wl["t_end"]   = wl["t_end"].astype(float)
    wl["_center"] = 0.5*(wl["t_start"]+wl["t_end"])
    if "scenario_label" not in wl.columns:
        wl["scenario_label"] = "unknown"
    wl["scenario_label"] = wl["scenario_label"].astype(str)

    return scores, episodes, wl

def s1_build_pr(scores: pd.DataFrame, wl: pd.DataFrame, op_threshold: float):
    # Positive = overlaps any GT window in same log
    gt_map = wl.groupby("log_id")[["t_start","t_end"]].apply(lambda g: g.values.tolist()).to_dict()
    y_true, y_prob = [], []
    for _, r in scores.iterrows():
        lid, ts, te = r["log_id"], r["t_start"], r["t_end"]
        pos = 0
        for (gs, ge) in gt_map.get(lid, []):
            if interval_iou(ts, te, gs, ge) > 0.0:
                pos = 1; break
        y_true.append(pos)
        y_prob.append(float(r.get("post_hmm", 0.0)))
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    ths = np.linspace(0, 1, 201)
    P, R = [], []
    for t in ths:
        pred = (y_prob >= t).astype(int)
        tp = int(np.sum((pred==1)&(y_true==1)))
        fp = int(np.sum((pred==1)&(y_true==0)))
        fn = int(np.sum((pred==0)&(y_true==1)))
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        recall    = tp / (tp+fn) if (tp+fn)>0 else 0.0
        P.append(precision); R.append(recall)
    order = np.argsort(R)
    pr_auc = float(np.trapz(np.array(P)[order], np.array(R)[order]))

    pred_op = (y_prob >= op_threshold).astype(int)
    tp = int(np.sum((pred_op==1)&(y_true==1)))
    fp = int(np.sum((pred_op==1)&(y_true==0)))
    fn = int(np.sum((pred_op==0)&(y_true==1)))
    prec_op = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec_op  = tp / (tp+fn) if (tp+fn)>0 else 0.0

    ece  = expected_calibration_error(y_true, y_prob, n_bins=10)
    brier= brier_score(y_true, y_prob)

    return {
        "ths": ths, "P": np.array(P), "R": np.array(R), "pr_auc": pr_auc,
        "op": {"threshold": op_threshold, "precision": prec_op, "recall": rec_op},
        "calib": {"ece": ece, "brier": brier, "y_true": y_true, "y_prob": y_prob}
    }

def s1_latency_iou_fragmentation(episodes: pd.DataFrame, wl: pd.DataFrame):
    """
    For each GT window:
      - fragmentation = # of episode segments overlapping this GT
      - choose nearest (by center) among overlapping segments (or nearest overall if none overlap)
      - latency = episode_center - gt_center
      - IoU = IoU(GT, chosen segment)
    Returns df with columns: log_id, scenario, latency_s, iou, frag
    """
    rows = []
    epi_by_log = {lid: g for lid, g in episodes.groupby("log_id")}
    for _, r in wl.iterrows():
        lid = r["log_id"]; ts=r["t_start"]; te=r["t_end"]; cen=r["_center"]
        scen = r["scenario_label"]
        ep = epi_by_log.get(lid)
        if ep is None:
            continue
        g = ep[["seg_start","seg_end"]].dropna().copy()
        if g.empty:
            continue
        g["center"] = 0.5*(g["seg_start"]+g["seg_end"])
        g["iou"] = g.apply(lambda x: interval_iou(ts, te, float(x["seg_start"]), float(x["seg_end"])), axis=1)
        g["dt"] = (g["center"] - cen).abs()
        frag = int((g["iou"] > 0).sum())
        choose = g[g["iou"]>0].sort_values("dt").head(1) if (g["iou"]>0).any() else g.sort_values("dt").head(1)
        if not choose.empty:
            latency = float(choose["center"].iloc[0] - cen)
            iou = float(choose["iou"].iloc[0])
            rows.append({"log_id": lid, "scenario": scen, "latency_s": latency, "iou": iou, "frag": frag})
    return pd.DataFrame(rows)

# ---------- S3 helpers ----------
def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())

def map_window_key_to_scenario_from_wl(pred_parquet: Path, wl: pd.DataFrame, tol: float = 0.05) -> Dict[str, str]:
    """
    Map each predicted window_key -> scenario by matching times to WL (within tol).
    If multiple WL match, pick the one with max IoU.
    """
    pred = pd.read_parquet(pred_parquet)
    pred["window_key"] = pred["window_key"].astype(str)
    wk_times = pred["window_key"].drop_duplicates().map(parse_wk)
    wk_df = pd.DataFrame({
        "window_key": pred["window_key"].drop_duplicates().tolist(),
        "log_id": [t[0] for t in wk_times],
        "t_start": [t[1] for t in wk_times],
        "t_end":   [t[2] for t in wk_times],
    })
    wk_df["_center"] = 0.5*(wk_df["t_start"]+wk_df["t_end"])

    # join by log then match by IoU
    lut = {}
    for lg, g in wk_df.groupby("log_id"):
        g = g.copy()
        cand = wl[wl["log_id"]==lg]
        if cand.empty:
            for wk in g["window_key"]:
                lut[str(wk)] = "unknown"
            continue
        for _, rr in g.iterrows():
            ts, te = float(rr["t_start"]), float(rr["t_end"])
            c = cand.copy()
            c["iou"] = c.apply(lambda x: interval_iou(ts, te, float(x["t_start"]), float(x["t_end"])), axis=1)
            c = c[c["iou"] > 0.0]
            if c.empty:
                lut[str(rr["window_key"])] = "unknown"
            else:
                pick = c.sort_values("iou", ascending=False).head(1)
                lut[str(rr["window_key"])] = str(pick["scenario_label"].iloc[0])
    return lut

def best_ranks_from_preds(split_root: Path, pred_parquet: Path, tags_norm: Path, near_sec: float=3.0, k:int=3):
    wl = read_jsonl(split_root / "s1a" / "window_labels.jsonl")
    wl = wl[wl.get("label",1) == 1].copy()
    wl["log_id"] = wl["log_id"].astype(str)
    wl["t_start"] = wl["t_start"].astype(float)
    wl["t_end"]   = wl["t_end"].astype(float)
    wl["_center"] = 0.5*(wl["t_start"]+wl["t_end"])
    if "scenario_label" not in wl.columns:
        wl["scenario_label"] = "unknown"

    tn = read_jsonl(tags_norm)
    tn["log_id"] = tn["log_id"].astype(str)
    tn["guest_id"] = tn["guest_id"].astype(str)
    tn = tn[(tn.get("has_guest", False) == True) & (tn.get("tag","")!="not_relevant")]
    gt_map = tn.groupby("log_id")["guest_id"].apply(lambda s: set(s.astype(str).tolist())).to_dict()

    pred = pd.read_parquet(pred_parquet).copy()
    pred["window_key"] = pred["window_key"].astype(str)
    pred["track_uuid"] = pred["track_uuid"].astype(str)
    pred["rank"] = pred["rank"].astype(int)
    pred["score"] = pred["score"].astype(float)
    parsed = pred["window_key"].map(parse_wk)
    pred["log_id"]  = parsed.map(lambda t: t[0]).astype(str)
    pred["t_start"] = parsed.map(lambda t: t[1]).astype(float)
    pred["t_end"]   = parsed.map(lambda t: t[2]).astype(float)
    pred["_center"] = 0.5*(pred["t_start"]+pred["t_end"])

    rows = []
    for _, r in wl.iterrows():
        lid, ts, te, cen = r["log_id"], float(r["t_start"]), float(r["t_end"]), float(r["_center"])
        scen = r.get("scenario_label", "unknown")
        gt_actors = list(gt_map.get(lid, []))
        sub = pred[pred["log_id"] == lid].copy()
        if sub.empty:
            rows.append({"scenario": scen, "best_rank": 999}); continue
        sub["near"] = (sub["_center"] - cen).abs() <= near_sec

        def pick_best_with_gt(df):
            br, sc = 999, -1.0
            for wk_i, grp in df.groupby("window_key"):
                grp = grp.sort_values("rank")
                rmin = 999; scgt = -1.0
                for a in gt_actors:
                    aa = grp[grp["track_uuid"] == a]
                    if not aa.empty:
                        rmin = min(rmin, int(aa["rank"].min()))
                        scgt = max(scgt, float(aa["score"].max()))
                if rmin == 999:
                    continue
                # prefer better rank then higher GT score
                better = (rmin < br) or (rmin == br and scgt > sc)
                if better:
                    br, sc = rmin, scgt
            return br

        br = pick_best_with_gt(sub[sub["near"]])
        if br == 999: br = pick_best_with_gt(sub)
        # IoU fallback doesn't change rank bucket (still miss if >3)
        rows.append({"scenario": scen, "best_rank": int(br)})
    return pd.DataFrame(rows)

# ---------- Plotters ----------
def plot_pr(ths, P, R, pr_auc, op, out):
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(R, P, lw=2, label=f"PR curve (AUC={pr_auc:.3f})")
    plt.scatter([op["recall"]],[op["precision"]], s=60, marker='o', zorder=5, label=f"OP τ={op['threshold']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.xlim(0,1); plt.ylim(0,1)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_latency_violin(df_lat, out):
    if df_lat.empty: return
    plt.figure(figsize=(7.2, 4.8))
    sns.violinplot(data=df_lat, x="scenario", y="latency_s", inner="quartile", cut=0)
    plt.axhline(0, ls="--", lw=1, color="k", alpha=0.4)
    plt.ylabel("Latency (s)  [episode center − GT center]")
    plt.xlabel("Scenario"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_iou_box(df_lat, out):
    if df_lat.empty: return
    plt.figure(figsize=(7.2, 4.8))
    sns.boxplot(data=df_lat, x="scenario", y="iou")
    plt.ylabel("Episode IoU vs GT window")
    plt.xlabel("Scenario"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_fragmentation_bar(df_lat, out):
    if df_lat.empty: return
    agg = df_lat.groupby("scenario")["frag"].mean().reset_index()
    plt.figure(figsize=(7.2, 4.8))
    sns.barplot(data=agg, x="scenario", y="frag")
    plt.ylabel("Mean #segments per GT window (fragmentation)")
    plt.xlabel("Scenario"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_reliability(y_true, y_prob, ece, brier, out):
    confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6.2, 4.8))
    plt.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
    plt.plot(confs, accs, marker='o', lw=2)
    if np.any(counts):
        sizes = 100*(counts/np.max(counts[np.nonzero(counts)]))
        plt.scatter(confs, accs, s=sizes, alpha=0.7)
    plt.xlabel("Confidence"); plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability: ECE={ece:.3f}, Brier={brier:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_prop_bars_with_ci(rows, ylabel, out):
    if not rows: return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6.8, 4.6))
    ax = sns.barplot(data=df, x="label", y="prop", ci=None)
    for i, r in enumerate(rows):
        ax.plot([i, i], [r["lo"], r["hi"]], color="k", lw=1.5)
    plt.ylim(0,1)
    plt.ylabel(ylabel); plt.xlabel("")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_grouped_per_scenario(per_scen_df, out):
    if per_scen_df.empty: return
    melted = per_scen_df.melt(id_vars=["scenario"], value_vars=["top1","r3"], var_name="metric", value_name="value")
    plt.figure(figsize=(7.5, 4.8))
    sns.barplot(data=melted, x="scenario", y="value", hue="metric")
    plt.ylim(0,1); plt.ylabel("Proportion"); plt.xlabel("Scenario")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title=""); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_rank_histogram(best_ranks_df, out):
    if best_ranks_df.empty: return
    ranks = best_ranks_df["best_rank"].tolist()
    buckets = {"1":0,"2":0,"3":0,"miss":0}
    for r in ranks:
        if r == 1: buckets["1"]+=1
        elif r == 2: buckets["2"]+=1
        elif r == 3: buckets["3"]+=1
        else: buckets["miss"]+=1
    labs = list(buckets.keys()); vals = [buckets[k] for k in labs]
    total = sum(vals) if sum(vals)>0 else 1
    pct = [v/total for v in vals]
    plt.figure(figsize=(6.2, 4.6))
    sns.barplot(x=labs, y=pct)
    plt.ylabel("Share of GT cases"); plt.xlabel("GT best rank in Top-3 (miss if >3)")
    for i,v in enumerate(pct):
        plt.text(i, v+0.02, f"{v*100:.1f}%", ha="center")
    plt.ylim(0,1)
    plt.tight_layout(); plt.savefig(out); plt.close()

def plot_simple_bars(df, x, y, out, ylim=(0,1), ylabel=""):
    if df.empty: return
    plt.figure(figsize=(7.2, 4.8))
    sns.barplot(data=df, x=x, y=y, ci=None)
    if ylim: plt.ylim(*ylim)
    plt.ylabel(ylabel or y); plt.xlabel(x)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--s1-scores", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--window-labels", required=True)
    ap.add_argument("--s1-metrics", required=True)
    ap.add_argument("--s3-metrics-val50", required=True)
    ap.add_argument("--s3-metrics-trainval", default="")
    ap.add_argument("--pred-parquet", required=True)
    ap.add_argument("--tags-norm", required=True)
    ap.add_argument("--teacher-parquet", default="")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    split_root = Path(args.split_root)
    outdir = Path(args.outdir); ensure_dir(outdir)

    # ---- S1 load + PR/Calibration
    scores, episodes, wl = load_s1_inputs(Path(args.s1_scores), Path(args.episodes), Path(args.window_labels))
    s1m = load_json(Path(args.s1_metrics))
    op_thresh = float(s1m.get("S1_event_detection", {}).get("op_threshold", 0.5))

    pr = s1_build_pr(scores, wl, op_thresh)
    plot_pr(pr["ths"], pr["P"], pr["R"], pr["pr_auc"], pr["op"], outdir / "s1_pr_curve.png")
    plot_reliability(pr["calib"]["y_true"], pr["calib"]["y_prob"], pr["calib"]["ece"], pr["calib"]["brier"], outdir / "s1_calibration_reliability.png")

    # ---- S1 latency/IoU/fragmentation
    df_lat = s1_latency_iou_fragmentation(episodes, wl)
    df_lat.to_csv(outdir / "s1_per_window_latency_iou_fragmentation.csv", index=False)
    if "scenario" in df_lat.columns and not df_lat.empty:
        plot_latency_violin(df_lat, outdir / "s1_latency_violin_per_scenario.png")
        plot_iou_box(df_lat, outdir / "s1_iou_boxplot_per_scenario.png")
        plot_fragmentation_bar(df_lat, outdir / "s1_fragmentation_bar_per_scenario.png")

    # ---- S1 Table
    rows = []
    prec_op = pr["op"]["precision"]; rec_op = pr["op"]["recall"]
    f1_op = (2*prec_op*rec_op)/(prec_op+rec_op) if (prec_op+rec_op)>0 else 0.0
    med_lat = float(df_lat["latency_s"].median()) if not df_lat.empty else 0.0
    med_iou = float(df_lat["iou"].median()) if not df_lat.empty else 0.0
    mean_frag = float(df_lat["frag"].mean()) if not df_lat.empty else 0.0
    rows.append({
        "scenario": "OVERALL",
        "precision_at_OP": prec_op, "recall_at_OP": rec_op, "f1_at_OP": f1_op,
        "pr_auc": pr["pr_auc"], "latency_median_s": med_lat,
        "iou_median": med_iou, "fragmentation_mean": mean_frag
    })
    if "scenario" in df_lat.columns and not df_lat.empty:
        for scen, g in df_lat.groupby("scenario"):
            rows.append({
                "scenario": scen,
                "precision_at_OP": np.nan, "recall_at_OP": np.nan, "f1_at_OP": np.nan,
                "pr_auc": np.nan,
                "latency_median_s": float(g["latency_s"].median()),
                "iou_median": float(g["iou"].median()),
                "fragmentation_mean": float(g["frag"].mean()),
            })
    pd.DataFrame(rows).to_csv(outdir / "table_s1_metrics.csv", index=False)

    # ---- S3 (VAL50 & optional train-val)
    s3_val = load_json(Path(args.s3_metrics_val50))
    s3_trv = load_json(Path(args.s3_metrics_trainval)) if args.s3_metrics_trainval else None

    # Top-1 & R@3 (with Wilson CIs)
    rows_top1 = []
    rows_r3   = []
    n_cons = int(s3_val.get("gt", {}).get("n_windows_considered", 0))
    t1 = float(s3_val.get("gt", {}).get("top1", 0.0))
    r3 = float(s3_val.get("gt", {}).get("r@3", 0.0))
    t1_lo, t1_hi = wilson_ci(t1, n_cons)
    r3_lo, r3_hi = wilson_ci(r3, n_cons)
    rows_top1.append({"label": "VAL50", "prop": t1, "lo": t1_lo, "hi": t1_hi})
    rows_r3.append({"label": "VAL50", "prop": r3, "lo": r3_lo, "hi": r3_hi})
    if s3_trv is not None and "gt" in s3_trv:
        n_cons_tv = int(s3_trv["gt"].get("n_windows_considered", 0))
        t1_tv = float(s3_trv["gt"].get("top1", 0.0))
        r3_tv = float(s3_trv["gt"].get("r@3", 0.0))
        t1_lo_tv, t1_hi_tv = wilson_ci(t1_tv, n_cons_tv)
        r3_lo_tv, r3_hi_tv = wilson_ci(r3_tv, n_cons_tv)
        rows_top1.append({"label": "Train-Val", "prop": t1_tv, "lo": t1_lo_tv, "hi": t1_hi_tv})
        rows_r3.append({"label": "Train-Val", "prop": r3_tv, "lo": r3_lo_tv, "hi": r3_hi_tv})

    plot_prop_bars_with_ci(rows_top1, ylabel="Top-1", out=outdir / "s3_top1_bars_ci.png")
    plot_prop_bars_with_ci(rows_r3,   ylabel="Recall@3", out=outdir / "s3_r3_bars_ci.png")

    # Per-scenario grouped bars + table (AvgBestRank/MRR)
    per_s = s3_val.get("per_scenario", {})
    per_rows, table_rows = [], []
    for scen, vals in per_s.items():
        per_rows.append({
            "scenario": scen,
            "top1": float(vals.get("gt_top1", 0.0)),
            "r3": float(vals.get("gt_r@3", 0.0))
        })
        table_rows.append({
            "scenario": scen,
            "avg_best_rank": float(vals.get("gt_avg_best_rank", 0.0)),
            "mrr": float(vals.get("gt_mrr", np.nan)) if "gt_mrr" in vals else np.nan
        })
    per_df = pd.DataFrame(per_rows)
    if not per_df.empty:
        plot_grouped_per_scenario(per_df, outdir / "s3_per_scenario_grouped.png")
    pd.DataFrame(table_rows).to_csv(outdir / "s3_avg_best_mrr_table.csv", index=False)

    # Rank histogram (rebuild from preds + tags_norm)
    best_df = best_ranks_from_preds(split_root, Path(args.pred_parquet), Path(args.tags_norm), near_sec=3.0, k=3)
    best_df.to_csv(outdir / "s3_best_ranks_per_gt.csv", index=False)
    plot_rank_histogram(best_df, outdir / "s3_rank_histogram.png")

    # Teacher NDCG bars (overall + per-scenario recomputed if needed)
    teacher_overall = float(s3_val.get("teacher", {}).get("ndcg@3", 0.0))
    t_rows = [{"scenario": "OVERALL", "ndcg@3": teacher_overall}]

    # If per-scenario teacher_ndcg@3 not in metrics, compute from teacher.parquet:
    per_has_ndcg = any("teacher_ndcg@3" in v for v in per_s.values()) if per_s else False
    if per_has_ndcg:
        for scen, vals in per_s.items():
            t_rows.append({"scenario": scen, "ndcg@3": float(vals.get("teacher_ndcg@3", 0.0))})
    else:
        teacher_pq = Path(args.teacher_parquet) if args.teacher_parquet else None
        if teacher_pq and teacher_pq.exists():
            # map window_key to scenario via WL
            scen_lut = map_window_key_to_scenario_from_wl(Path(args.pred_parquet), wl, tol=0.05)
            tdf = pd.read_parquet(teacher_pq)
            tdf["window_key"] = tdf["window_key"].astype(str)
            tdf["track_uuid"] = tdf["track_uuid"].astype(str)
            # build gains lookup
            gains = {(wk, tu): float(q) for wk, tu, q in zip(tdf["window_key"], tdf["track_uuid"], tdf["q"])}
            pred = pd.read_parquet(Path(args.pred_parquet)).copy()
            pred["window_key"] = pred["window_key"].astype(str)
            pred["track_uuid"] = pred["track_uuid"].astype(str)
            # compute per-window ndcg then aggregate by scenario
            rows_nd = []
            for wk, grp in pred.groupby("window_key"):
                sc = scen_lut.get(wk, "unknown")
                sc_arr = grp.sort_values("rank")
                scores = sc_arr["score"].to_numpy(dtype=float)
                g = np.array([gains.get((wk, tu), 0.0) for tu in sc_arr["track_uuid"].tolist()], dtype=float)
                if np.any(g>0):
                    rows_nd.append({"scenario": sc, "ndcg": ndcg_at_k(scores, g, k=3)})
            if rows_nd:
                nd_df = pd.DataFrame(rows_nd).groupby("scenario")["ndcg"].mean().reset_index()
                for _, rr in nd_df.iterrows():
                    t_rows.append({"scenario": rr["scenario"], "ndcg@3": float(rr["ndcg"])})
    tdf = pd.DataFrame(t_rows)
    if not tdf.empty:
        plot_simple_bars(tdf, x="scenario", y="ndcg@3", out=outdir / "s3_teacher_ndcg_bars.png", ylim=(0,1), ylabel="Teacher NDCG@3")

    # Pairwise accuracy bars (overall + per-scenario if available)
    pair_overall = float(s3_val.get("gt", {}).get("pairwise_acc", 0.0))
    prow = [{"scenario": "OVERALL", "pairwise_acc": pair_overall}]
    for scen, vals in per_s.items():
        if "gt_pairwise_acc" in vals:
            prow.append({"scenario": scen, "pairwise_acc": float(vals["gt_pairwise_acc"])})
    pdf = pd.DataFrame(prow).dropna(subset=["pairwise_acc"])
    if not pdf.empty:
        plot_simple_bars(pdf, x="scenario", y="pairwise_acc", out=outdir / "s3_pairwise_accuracy_bars.png", ylim=(0,1), ylabel="Pairwise accuracy")

    # Coverage bars: (n_considered / n_total) per scenario
    total_per_scen = wl.groupby("scenario_label").size().rename("n_total").reset_index().rename(columns={"scenario_label":"scenario"})
    cons_rows = []
    for _, rr in total_per_scen.iterrows():
        scen = rr["scenario"]
        n_total = int(rr["n_total"])
        n_cons  = int(per_s.get(scen, {}).get("n_windows", 0)) if per_s else 0
        prop = n_cons / n_total if n_total>0 else 0.0
        cons_rows.append({"scenario": scen, "coverage": prop})
    cov_df = pd.DataFrame(cons_rows)
    if not cov_df.empty:
        plot_simple_bars(cov_df, x="scenario", y="coverage", out=outdir / "s3_coverage_bars.png", ylim=(0,1), ylabel="Coverage (n_considered / n_total)")

    # ---- Table S3 (overall + per-scenario)
    rows_s3 = [{
        "scenario": "OVERALL",
        "top1": t1, "r@3": r3,
        "avg_best_rank": float(s3_val.get("gt", {}).get("avg_best_rank", 0.0)),
        "mrr": float(s3_val.get("gt", {}).get("mrr", 0.0)),
        "pairwise_acc": pair_overall,
        "ndcg@3_teacher": float(s3_val.get("teacher", {}).get("ndcg@3", 0.0)),
        "n_considered": int(s3_val.get("gt", {}).get("n_windows_considered", 0)),
        "n_total": int(s3_val.get("gt", {}).get("n_windows_total", 0))
    }]
    for scen, vals in per_s.items():
        rows_s3.append({
            "scenario": scen,
            "top1": float(vals.get("gt_top1", 0.0)),
            "r@3": float(vals.get("gt_r@3", 0.0)),
            "avg_best_rank": float(vals.get("gt_avg_best_rank", 0.0)),
            "mrr": float(vals.get("gt_mrr", np.nan)) if "gt_mrr" in vals else np.nan,
            "pairwise_acc": float(vals.get("gt_pairwise_acc", np.nan)) if "gt_pairwise_acc" in vals else np.nan,
            "ndcg@3_teacher": float(vals.get("teacher_ndcg@3", np.nan)) if "teacher_ndcg@3" in vals else np.nan,
            "n_considered": int(vals.get("n_windows", 0)),
            "n_total": int(total_per_scen.loc[total_per_scen["scenario"]==scen, "n_total"].sum())
        })
    pd.DataFrame(rows_s3).to_csv(outdir / "table_s3_metrics.csv", index=False)

    print(f"[OK] Wrote figures & tables to: {outdir}")

if __name__ == "__main__":
    main()
