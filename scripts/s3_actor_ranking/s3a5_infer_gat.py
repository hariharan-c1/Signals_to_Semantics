#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


# ---------------- Utils ----------------
def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def safe_device(name: str):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name

def replace_nan_inf_(x: torch.Tensor, value: float = 0.0):
    if x.is_floating_point():
        x[torch.isnan(x)] = value
        x[torch.isinf(x)] = value
    return x

def split_batch_indices(batch_vec: torch.Tensor) -> List[torch.Tensor]:
    if batch_vec.numel() == 0:
        return [torch.arange(0, 0, dtype=torch.long, device=batch_vec.device)]
    num = int(batch_vec.max().item()) + 1
    return [(batch_vec == g).nonzero(as_tuple=False).view(-1) for g in range(num)]

def parse_wk(wk: str, decimals: int = 6) -> Tuple[str, float, float]:
    parts = str(wk).split("|")
    if len(parts) != 3:
        return str(wk), float("nan"), float("nan")
    lg = parts[0]
    try:
        ts = round(float(parts[1]), decimals)
        te = round(float(parts[2]), decimals)
    except Exception:
        ts, te = float("nan"), float("nan")
    return lg, ts, te

def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = max(0.0, right - left)
    union = max(1e-9, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union


# ---------------- Teacher ----------------
def build_teacher_lookup(teacher_parquet: Optional[Path]) -> Dict[Tuple[str, str], float]:
    if teacher_parquet is None or (not teacher_parquet.exists()):
        print("[Teacher] WARN: teacher.parquet not found → teacher metrics disabled.")
        return {}
    tdf = pd.read_parquet(teacher_parquet)
    need = {"window_key","track_uuid","q"}
    if not need.issubset(tdf.columns):
        print("[Teacher] WARN: teacher.parquet missing columns → teacher metrics disabled.")
        return {}
    tdf["window_key"] = tdf["window_key"].astype(str)
    tdf["track_uuid"] = tdf["track_uuid"].astype(str)
    lut = {(wk, tu): float(q) for wk, tu, q in zip(tdf["window_key"], tdf["track_uuid"], tdf["q"])}
    print(f"[Teacher] Loaded {len(lut)} window-actor entries.")
    return lut

def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    if len(scores) == 0 or len(gains) == 0:
        return 0.0
    k = max(1, min(k, len(scores)))
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    # teacher q is already [0,1]; no 2^g-1 transformation needed
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0 else dcg / idcg


# ---------------- Dataset ----------------
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs_dir: Path):
        self.graphs_dir = graphs_dir
        man = pd.read_parquet(graphs_dir / "manifest.parquet")
        raw_files = [str(p) for p in man["file"].tolist()]
        files: List[Path] = []
        for p in raw_files:
            p1 = Path(p); p2 = graphs_dir / Path(p)
            files.append(p1 if p1.exists() else p2 if p2.exists() else p1)
        self.files = files
        if not self.files:
            raise FileNotFoundError(f"No graphs under {graphs_dir}")

        self.manifest = man.copy()
        self.manifest["window_key"] = self.manifest["window_key"].astype(str)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        g: Data = torch.load(self.files[idx], map_location="cpu")
        for req in ("x","edge_index","edge_type","node_ids","window_key"):
            if not hasattr(g, req):
                raise KeyError(f"{self.files[idx]} missing '{req}'")
        x = g.x.to(torch.float32).clone()
        replace_nan_inf_(x, 0.0)
        edge_index = g.edge_index.clone().to(torch.long)
        edge_type = g.edge_type.clone().to(torch.long)
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        data.window_key = str(getattr(g, "window_key", ""))
        data.node_ids = list(map(str, g.node_ids))
        return data


# ---------------- Model ----------------
class EdgeTypeEmbed(nn.Module):
    def __init__(self, num_types: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max(1, num_types), dim)
    def forward(self, edge_type: torch.Tensor):
        edge_type = edge_type.clamp_min(0)
        edge_type = torch.remainder(edge_type, max(1, self.emb.num_embeddings))
        return self.emb(edge_type)

class GATQuasi(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, layers: int, dropout: float,
                 edge_type_dim: int, num_edge_types: int):
        super().__init__()
        self.et = EdgeTypeEmbed(num_edge_types, edge_type_dim)
        self.proj_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GATv2Conv(
                in_channels=hidden,
                out_channels=hidden // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=edge_type_dim,
                share_weights=False,
            ))
            self.norms.append(nn.LayerNorm(hidden))
        self.readout = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None):
        h = self.proj_in(x)
        for conv, ln in zip(self.layers, self.norms):
            eattr = self.et(edge_type) if edge_type is not None else None
            h2 = conv(h, edge_index, eattr)
            h = ln(F.elu(h2) + h)
            h = self.dropout(h)
        s = self.readout(h).squeeze(-1)
        return s


# ---------------- Actor aggregation ----------------
def aggregate_to_actors(node_ids: List[str],
                        scores_1d: np.ndarray,
                        pool_mode: str = "max") -> Tuple[List[str], np.ndarray]:
    bucket: Dict[str, List[int]] = {}
    for i, nid in enumerate(node_ids):
        if nid == "EGO" or str(nid).startswith("UNK_"):
            continue
        bucket.setdefault(str(nid), []).append(i)
    if not bucket:
        return [], np.zeros((0,), dtype=np.float32)
    actors, s_list = [], []
    for tu, idxs in bucket.items():
        actors.append(tu)
        if pool_mode == "mean":
            s_list.append(float(np.mean(scores_1d[idxs])))
        else:
            s_list.append(float(np.max(scores_1d[idxs])))
    return actors, np.asarray(s_list, dtype=np.float32)


# ---------------- Robust S3 evaluation (VAL50) ----------------
def load_window_labels(split_root: Path) -> pd.DataFrame:
    wl_path = split_root / "s1a" / "window_labels.jsonl"
    if not wl_path.exists():
        raise FileNotFoundError(str(wl_path))
    df = pd.read_json(wl_path, lines=True)
    df = df[df.get("label", 1) == 1].copy()
    need = {"log_id","t_start","t_end"}
    if not need.issubset(df.columns):
        raise KeyError("window_labels.jsonl missing required columns")
    if "scenario_label" not in df.columns:
        df["scenario_label"] = "unknown"
    df["log_id"] = df["log_id"].astype(str)
    df["t_start"] = df["t_start"].astype(float)
    df["t_end"]   = df["t_end"].astype(float)
    df["_center"] = 0.5*(df["t_start"]+df["t_end"])
    return df[["log_id","t_start","t_end","_center","scenario_label"]]

def load_tags_norm(tags_norm_path: Path) -> Dict[str, List[str]]:
    tn = pd.read_json(tags_norm_path, lines=True)
    tn["log_id"] = tn["log_id"].astype(str)
    tn["guest_id"] = tn["guest_id"].astype(str)
    valid = (tn.get("has_guest", True) == True) & tn["guest_id"].notna()
    mp: Dict[str, List[str]] = {}
    for lid, sub in tn[valid].groupby("log_id"):
        mp[lid] = sorted(set(sub["guest_id"].astype(str).tolist()))
    return mp

def robust_s3_eval_from_preds(
    split_root: Path,
    tags_norm_path: Path,
    pred_parquet: Path,
    teacher_parquet: Optional[Path],
    near_sec: float = 3.0,
    k: int = 3
) -> Dict:
    """Nearest-with-GT → any-with-GT → max-IoU fallback; computes GT metrics,
       overall teacher NDCG@k, and per-scenario including teacher_ndcg@k."""
    wl = load_window_labels(split_root)  # GT windows (independent rows)
    gt_map = load_tags_norm(tags_norm_path)  # log_id → [gt actors]
    pred = pd.read_parquet(pred_parquet).copy()
    need = {"window_key","rank","track_uuid","score"}
    if not need.issubset(pred.columns):
        raise KeyError(f"{pred_parquet} missing {need}")
    pred["window_key"] = pred["window_key"].astype(str)
    pred["track_uuid"] = pred["track_uuid"].astype(str)
    pred["rank"] = pred["rank"].astype(int)
    pred["score"] = pred["score"].astype(float)
    # parse times from window_key
    parsed = pred["window_key"].map(parse_wk)
    pred["log_id"]  = parsed.map(lambda t: t[0]).astype(str)
    pred["t_start"] = parsed.map(lambda t: t[1]).astype(float)
    pred["t_end"]   = parsed.map(lambda t: t[2]).astype(float)
    pred["_center"] = 0.5*(pred["t_start"]+pred["t_end"])

    pred_wk_set = set(pred["window_key"].unique())

    # helper: choose best window that contains any GT actor in Top-K
    def choose_best_with_gt(sub_df: pd.DataFrame, gt_actors: List[str], gt_center: float) -> Tuple[str, int, float]:
        if sub_df.empty or not gt_actors:
            return "", 999, -1.0
        best_rank, best_wk, best_score = 999, "", -1.0
        for wk, rows in sub_df.groupby("window_key"):
            rows = rows.sort_values("rank")
            rmin = 999; top_gt_score = -1.0
            for a in gt_actors:
                aa = rows[rows["track_uuid"] == a]
                if not aa.empty:
                    rmin = min(rmin, int(aa["rank"].min()))
                    top_gt_score = max(top_gt_score, float(aa["score"].max()))
            if rmin == 999:
                continue
            # tiebreak: better rank → higher GT score → nearer
            dt = abs(float(rows["_center"].iloc[0]) - gt_center)
            better = (rmin < best_rank) or \
                     (rmin == best_rank and top_gt_score > best_score) or \
                     (rmin == best_rank and abs(top_gt_score - best_score) < 1e-12 and dt < abs(best_score - best_score + 0.0))
            if better:
                best_rank, best_wk, best_score = rmin, str(wk), top_gt_score
        return best_wk, best_rank, best_score

    # select a "used window" per GT row
    considered_rows = []   # list of dicts per GT row that we can evaluate against predictions
    used_windows = []      # the set of used window_keys (for teacher per-scenario averaging)
    for _, r in wl.iterrows():
        lid = r["log_id"]; ts = float(r["t_start"]); te = float(r["t_end"])
        center = float(r["_center"]); scen = str(r["scenario_label"])
        gt_actors = gt_map.get(lid, [])
        cands = pred[pred["log_id"] == lid].copy()
        if cands.empty:
            continue
        cands["near"] = (cands["_center"] - center).abs() <= near_sec

        # 1) near-with-GT
        w_near = cands[cands["near"]]
        used_wk, best_rank, _ = choose_best_with_gt(w_near, gt_actors, center)

        # 2) any-with-GT
        if not used_wk:
            used_wk, best_rank, _ = choose_best_with_gt(cands, gt_actors, center)

        # 3) fallback: max IoU (even if GT actor not in Top-K)
        if not used_wk:
            tmp = cands[["window_key","t_start","t_end","_center"]].drop_duplicates().copy()
            tmp["iou"] = tmp.apply(lambda x: interval_iou(ts, te, x["t_start"], x["t_end"]), axis=1)
            pick = tmp.sort_values(["iou","_center"], ascending=[False, True]).head(1)
            if not pick.empty:
                used_wk = str(pick["window_key"].iloc[0])
                best_rank = 999

        # confirm prediction exists
        if used_wk and used_wk in pred_wk_set:
            sub = pred[pred["window_key"] == used_wk].sort_values("rank")
            actors = sub["track_uuid"].tolist()
            scores = sub["score"].to_numpy(dtype=float)
            # compute Top1 / R@k / best rank
            rank_gt = None
            for rr, a in enumerate(actors, start=1):
                if a in gt_actors:
                    rank_gt = rr
                    break
            t1 = 1.0 if rank_gt == 1 else 0.0
            r_at_k = 1.0 if (rank_gt is not None and rank_gt <= k) else 0.0
            best_r = float(rank_gt) if rank_gt is not None else 999.0

            considered_rows.append({
                "log_id": lid, "scenario": scen, "used_window": used_wk,
                "top1": t1, "r@k": r_at_k, "best_rank": best_r
            })
            used_windows.append(used_wk)

    # aggregate GT metrics across considered rows
    gt_considered = pd.DataFrame(considered_rows)
    n_total_gt = int(len(wl))
    n_considered = int(len(gt_considered))
    if n_considered > 0:
        top1 = float(gt_considered["top1"].mean())
        r_k = float(gt_considered["r@k"].mean())
        valid = gt_considered[gt_considered["best_rank"] < 999]["best_rank"].to_numpy(dtype=float)
        avg_best_rank = float(valid.mean()) if valid.size > 0 else 0.0
        # MRR (only where GT in top-k list; treat miss as 0)
        mrr = float(np.mean([1.0/x if x < 999 else 0.0 for x in gt_considered["best_rank"].tolist()]))
        # simple pairwise acc vs other Top-K items (proxy)
        pairwise = []
        for _, row in gt_considered.iterrows():
            if row["best_rank"] < 999 and row["best_rank"] <= k:
                # if GT is present, we approximate its pairwise win-rate among (k-1) others
                pairwise.append((k - (row["best_rank"] - 1)) / max(1, k - 1))
            else:
                pairwise.append(0.0)
        pairwise_acc = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        top1 = r_k = avg_best_rank = mrr = pairwise_acc = 0.0

    # per-scenario GT metrics (on considered set)
    per_scenario: Dict[str, Dict] = {}
    if n_considered > 0:
        for scen, g in gt_considered.groupby("scenario"):
            vv = g["best_rank"].to_numpy(dtype=float)
            per_scenario[scen] = {
                "n_windows": int(len(g)),
                "gt_top1": float(g["top1"].mean()),
                "gt_r@{}".format(k): float(g["r@k"].mean()),
                "gt_avg_best_rank": float(np.mean(vv[vv<999])) if np.any(vv<999) else 0.0,
            }

    # teacher NDCG@k (overall + per-scenario over used windows only)
    teacher_lut = build_teacher_lookup(teacher_parquet) if teacher_parquet is not None else {}
    teacher_ndcg = 0.0; teacher_n = 0
    per_scen_teacher: Dict[str, List[float]] = {sc: [] for sc in per_scenario.keys()}

    if teacher_lut and n_considered > 0:
        used_set = set(used_windows)
        for wk, sub in pred.groupby("window_key"):
            if wk not in used_set:
                continue
            gains = np.array([teacher_lut.get((wk, tu), 0.0) for tu in sub.sort_values("rank")["track_uuid"].tolist()], dtype=float)
            if not np.any(gains > 0):
                continue
            scores = sub.sort_values("rank")["score"].to_numpy(dtype=float)
            nd = ndcg_at_k(scores, gains, k=k)
            teacher_ndcg += nd; teacher_n += 1
            # per-scenario assignment via the GT row that used this window
            scen_rows = gt_considered[gt_considered["used_window"] == wk]["scenario"].unique().tolist()
            scen_name = scen_rows[0] if scen_rows else "unknown"
            per_scen_teacher.setdefault(scen_name, []).append(nd)

    # attach teacher_per_scenario to per_scenario dict
    for scen in list(per_scenario.keys()) + list(per_scen_teacher.keys()):
        arr = per_scen_teacher.get(scen, [])
        per_scenario.setdefault(scen, {"n_windows": 0})
        per_scenario[scen]["teacher_ndcg@{}".format(k)] = float(np.mean(arr)) if arr else 0.0

    s3_report = {
        "gt": {
            "n_windows_total": n_total_gt,
            "n_windows_considered": n_considered,
            "top1": top1,
            "r@{}".format(k): r_k,
            "avg_best_rank": avg_best_rank,
            "mrr": mrr,
            "pairwise_acc": pairwise_acc
        },
        "teacher": {
            "n_windows_considered": int(teacher_n),
            "ndcg@{}".format(k): (teacher_ndcg/teacher_n) if teacher_n > 0 else 0.0
        },
        "per_scenario": per_scenario
    }
    return s3_report


# ---------------- Inference & (upgraded) Eval ----------------
@torch.no_grad()
def run_inference_and_eval(
    split_root: Path,
    graphs_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    pool_mode: str,
    batch_size: int,
    seed: int,
    device: torch.device,
    # model dims (may be auto-loaded from meta)
    hidden: int, heads: int, layers: int, dropout: float, edge_type_dim: int,
    teacher_parquet: Optional[Path],
    tags_norm_path: Path,
    gt_time_decimals: int,      # kept for compatibility (not used in upgraded eval)
    gt_time_tol_s: float,       # kept for compatibility (not used in upgraded eval)
    gt_allow_log_fallback: int, # kept for compatibility (not used in upgraded eval)
    export_k: int = 3,
):
    # --- inference (unchanged) ---
    ds = GraphDataset(graphs_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    g0: Data = torch.load(ds.files[0], map_location="cpu")
    in_dim = int(g0.x.size(1))
    et0 = g0.edge_type if hasattr(g0, "edge_type") else torch.zeros(g0.edge_index.size(1), dtype=torch.long)
    num_edge_types = int(torch.unique(et0).numel())

    meta_path = checkpoint.parent / "meta_train.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            hidden = int(meta.get("hidden", hidden))
            heads = int(meta.get("heads", heads))
            layers = int(meta.get("layers", layers))
            dropout = float(meta.get("dropout", dropout))
            edge_type_dim = int(meta.get("edge_type_dim", edge_type_dim))
            print(f"[Meta] Loaded model dims from {meta_path}")
        except Exception as e:
            print(f"[Meta] WARN: failed to parse {meta_path}: {e} → falling back to CLI dims.")

    model = GATQuasi(
        in_dim=in_dim, hidden=hidden, heads=heads, layers=layers, dropout=dropout,
        edge_type_dim=edge_type_dim, num_edge_types=num_edge_types
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    rows_topk = []
    model.eval()
    for data in dl:
        data = data.to(device)
        s = model(data.x, data.edge_index, getattr(data, "edge_type", None)).detach().cpu().numpy()
        batch_vec = data.batch.cpu()

        if isinstance(data.window_key, (list, tuple)):
            wkeys = list(map(str, data.window_key))
        else:
            n_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 1
            wkeys = [str(data.window_key)] * n_graphs

        ids_per_graph = []
        if isinstance(data.node_ids, list) and data.node_ids and isinstance(data.node_ids[0], list):
            ids_per_graph = data.node_ids
        else:
            flat = list(map(str, data.node_ids))
            for idx in split_batch_indices(batch_vec):
                ids_per_graph.append([flat[int(i)] for i in idx])

        for g, idx in enumerate(split_batch_indices(batch_vec)):
            wkey = wkeys[g]
            ids_g = ids_per_graph[g]
            s_g = s[idx]

            actors, s_actor = aggregate_to_actors(ids_g, s_g, pool_mode=pool_mode)
            if len(actors) == 0:
                continue

            order = np.argsort(-s_actor)[:export_k]
            for rank, j in enumerate(order, 1):
                rows_topk.append({
                    "window_key": wkey,
                    "rank": rank,
                    "track_uuid": actors[j],
                    "score": float(s_actor[j]),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"top{export_k}_infer.parquet"
    pd.DataFrame(rows_topk).to_parquet(out_parquet, index=False)
    print(f"[S3-A5][INFER] Exported Top-{export_k} → {out_parquet}")

    # --- upgraded S3 eval (robust matcher + per-scenario teacher NDCG) ---
    s3_eval_report = robust_s3_eval_from_preds(
        split_root=split_root,
        tags_norm_path=tags_norm_path,
        pred_parquet=out_parquet,
        teacher_parquet=teacher_parquet,
        near_sec=3.0,
        k=export_k
    )

    report = {
        "split_root": str(split_root),
        "graphs_dir": str(graphs_dir),
        "checkpoint": str(checkpoint),
        "export_k": export_k,
        "pool_mode": pool_mode,
        "gt": s3_eval_report["gt"],
        "teacher": s3_eval_report["teacher"],
        "per_scenario": s3_eval_report["per_scenario"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    print(f"[Report] Wrote → {out_dir / 'final_report.json'}")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True, help="e.g. artifacts/train650_val50/val50")
    ap.add_argument("--graphs-split", required=True, help="e.g. val50")
    ap.add_argument("--checkpoint", required=True)

    # model dims (auto-read from meta_train.json if present; else use these)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--edge-type-dim", type=int, default=16)

    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--export-k", type=int, default=3)
    ap.add_argument("--pool-mode", choices=["max", "mean"], default="max")

    # evaluation inputs
    ap.add_argument("--teacher-parquet", default="", help="Optional path to teacher.parquet for this split")
    ap.add_argument("--tags-norm", required=True, help="Path to tags_norm.jsonl")
    ap.add_argument("--gt-time-decimals", type=int, default=6)       # kept for compatibility (unused)
    ap.add_argument("--gt-time-tol-s", type=float, default=0.05)     # kept for compatibility (unused)
    ap.add_argument("--gt-allow-log-fallback", type=int, default=0)  # kept for compatibility (unused)

    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(safe_device(args.device))

    split_root = Path(args.split_root)
    graphs_dir = split_root / "s3" / "quasi" / "graphs" / args.graphs_split
    out_dir = graphs_dir / "gat_final_eval_upgraded_v9_f0_max"
    teacher_parquet = Path(args.teacher_parquet) if args.teacher_parquet else None

    run_inference_and_eval(
        split_root=split_root,
        graphs_dir=graphs_dir,
        checkpoint=Path(args.checkpoint),
        out_dir=out_dir,
        pool_mode=args.pool_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        hidden=args.hidden, heads=args.heads, layers=args.layers, dropout=args.dropout, edge_type_dim=args.edge_type_dim,
        teacher_parquet=teacher_parquet,
        tags_norm_path=Path(args.tags_norm),
        gt_time_decimals=args.gt_time_decimals,
        gt_time_tol_s=args.gt_time_tol_s,
        gt_allow_log_fallback=args.gt_allow_log_fallback,
        export_k=args.export_k,
    )

if __name__ == "__main__":
    main()
