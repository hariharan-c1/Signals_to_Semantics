# scripts/s3a5_train_gat_multitask_v6.py
import argparse, json, random, time, math, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# ----------------------------- utils -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_device(name: str):
    if name == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    return name

def replace_nan_inf_(x: torch.Tensor, value: float = 0.0):
    if x.is_floating_point():
        x[torch.isnan(x)] = value
        x[torch.isinf(x)] = value
    return x

def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    k = max(1, min(k, len(scores)))
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    gains = (2**gains[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    idcg = float(np.sum((2**ideal - 1) * discounts))
    return 0.0 if idcg == 0 else dcg / idcg

def listnet_ce(scores: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Listwise CE: - sum q * log softmax(s) on masked subset (q normalized within mask)."""
    s = scores[mask]
    q = probs[mask]
    if s.numel() == 0: return scores.new_zeros(())
    logp = F.log_softmax(s, dim=0)
    qn = F.softmax(q, dim=0)  # normalize teacher mass within mask
    return -(qn * logp).sum()

# ----------------------- teacher / GT inputs ----------------------
def build_teacher_lookup(teacher_parquet: Path) -> Dict[Tuple[str, str], float]:
    tdf = pd.read_parquet(teacher_parquet)
    need = {"window_key","track_uuid","q"}
    if not need.issubset(tdf.columns):
        raise KeyError(f"teacher.parquet missing {need}")
    tdf["window_key"] = tdf["window_key"].astype(str)
    tdf["track_uuid"] = tdf["track_uuid"].astype(str)
    return {(wk, tu): float(q) for wk, tu, q in zip(tdf["window_key"], tdf["track_uuid"], tdf["q"])}

def _parse_window_key_round(wk: str, decimals: int) -> Tuple[str, float, float]:
    parts = wk.split("|")
    if len(parts) != 3: return wk, float("nan"), float("nan")
    lg = parts[0]
    try:
        ts = round(float(parts[1]), decimals)
        te = round(float(parts[2]), decimals)
    except Exception:
        ts, te = float("nan"), float("nan")
    return lg, ts, te

def build_gt_lookup(
    split_root: Path,
    graphs_dir: Path,
    tags_norm_path: Path,
    time_decimals: int = 6,
    time_tol_s: float = 0.05,
    allow_log_fallback: int = 0
) -> Dict[str, Set[str]]:
    """
    Returns: gt_lut[window_key] = set(track_uuid) (multi-positives supported).
    Uses exact/tolerant match on (log_id, t_start, t_end). Fallback expands to all windows in tagged logs.
    """
    wl_path = split_root / "s1a" / "window_labels.jsonl"
    if not wl_path.exists():
        print(f"[GT] WARN: window_labels.jsonl not found at {wl_path} → GT disabled.")
        return {}

    wl = pd.read_json(wl_path, lines=True)
    need = {"log_id","t_start","t_end","label"}
    if not need.issubset(wl.columns):
        print("[GT] WARN: window_labels.jsonl missing required columns → GT disabled.")
        return {}
    wl = wl[wl["label"] == 1].copy()
    wl["log_id"] = wl["log_id"].astype(str)
    wl["t_start_r"] = pd.to_numeric(wl["t_start"], errors="coerce").round(time_decimals)
    wl["t_end_r"]   = pd.to_numeric(wl["t_end"],   errors="coerce").round(time_decimals)

    by_log: Dict[str, np.ndarray] = {}
    for lg, g in wl.groupby("log_id"):
        by_log[str(lg)] = g[["t_start_r","t_end_r"]].to_numpy(dtype=float)

    man = pd.read_parquet(graphs_dir / "manifest.parquet").copy()
    if "window_key" not in man.columns:
        raise KeyError("manifest.parquet must contain 'window_key'")
    man["window_key"] = man["window_key"].astype(str)

    parsed = man["window_key"].map(lambda wk: _parse_window_key_round(wk, time_decimals))
    man["_log"] = parsed.map(lambda t: t[0])
    man["_tsr"] = parsed.map(lambda t: t[1])
    man["_ter"] = parsed.map(lambda t: t[2])

    tags = pd.read_json(tags_norm_path, lines=True)
    tags["log_id"] = tags["log_id"].astype(str)
    valid = (tags.get("has_guest", False) == True) & (tags.get("tag","") != "not_relevant") & (tags["guest_id"].astype(str).str.len() > 0)
    tags = tags.loc[valid, ["log_id","guest_id"]].dropna()
    guests_by_log: Dict[str, Set[str]] = {}
    for lg, g in tags.groupby("log_id"):
        guests_by_log[str(lg)] = set(map(str, g["guest_id"].tolist()))

    gt_lut: Dict[str, Set[str]] = {}
    tol = float(time_tol_s)
    used_fallback = False

    for wk, lg, tsr, ter in zip(man["window_key"], man["_log"], man["_tsr"], man["_ter"]):
        lg = str(lg)
        if lg not in by_log:
            continue
        arr = by_log[lg]  # (M,2)
        hit = False
        if np.isfinite(tsr) and np.isfinite(ter):
            diffs = np.max(np.abs(arr - np.array([tsr, ter])[None, :]), axis=1)
            if (diffs <= tol).any():
                gs = guests_by_log.get(lg, set())
                if gs:
                    gt_lut[wk] = set(gs)
                    hit = True
        if not hit and allow_log_fallback:
            gs = guests_by_log.get(lg, set())
            if gs:
                gt_lut[wk] = set(gs)
                used_fallback = True

    print(f"[GT] windows with GT (exact/tolerant) = {len(gt_lut)} "
          f"(time_tol={time_tol_s}s, decimals={time_decimals}, fallback={bool(allow_log_fallback)})")
    if used_fallback:
        print("[GT] Note: log_id-only fallback used for some windows.")
    return gt_lut

# ---------------------------- dataset ----------------------------
def collect_files_from_manifest(graphs_dir: Path) -> List[Path]:
    man = pd.read_parquet(graphs_dir / "manifest.parquet")
    raw_files = [str(p) for p in man["file"].tolist()]
    files: List[Path] = []
    for p in raw_files:
        p1 = Path(p)
        p2 = graphs_dir / Path(p)
        files.append(p1 if p1.exists() else p2 if p2.exists() else p1)
    if not files:
        raise FileNotFoundError(f"No graph files under {graphs_dir}")
    return files

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[Path]):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i: int) -> Data:
        g: Data = torch.load(self.files[i], map_location="cpu")
        for req in ("x","edge_index","edge_type","node_ids","window_key"):
            if not hasattr(g, req):
                raise KeyError(f"Graph {self.files[i]} missing '{req}'")
        g.x = g.x.to(torch.float32)
        replace_nan_inf_(g.x, 0.0)
        g.edge_index = g.edge_index.to(torch.long)
        g.edge_type = g.edge_type.to(torch.long)
        # keep node_ids as Python list; window_key as string
        return g

# ----------------------------- model -----------------------------
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

# ------------------------- helpers (pooling etc.) -------------------------
def split_by_keys(files: List[Path], manifest_path: Path, mode: str, val_frac: float, seed: int):
    if mode == "fixed80_20":
        man = pd.read_parquet(manifest_path)
        keys = [str(k) for k in man["window_key"].tolist()]
        keyed = [(int(hashlib.md5(k.encode()).hexdigest(), 16), i) for i, k in enumerate(keys)]
        keyed.sort()
        order = [i for _, i in keyed]
        nv = max(1, int(round(val_frac * len(files))))
        val_idx = sorted(order[:nv]); tr_idx = sorted(order[nv:])
        return tr_idx, val_idx
    elif mode == "all_train":
        idx = list(range(len(files)))
        return idx, []
    else:
        raise ValueError(f"Unknown split mode: {mode}")

def batch_slices(batch_vec: torch.Tensor) -> List[torch.Tensor]:
    if batch_vec.numel() == 0:
        return [torch.arange(0, 0, dtype=torch.long, device=batch_vec.device)]
    num = int(batch_vec.max().item()) + 1
    return [(batch_vec == g).nonzero(as_tuple=False).view(-1) for g in range(num)]

def local_subgraph_edges(edge_index: torch.Tensor,
                         edge_type: Optional[torch.Tensor],
                         idx: torch.Tensor,
                         num_nodes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if edge_index.numel() == 0:
        return edge_index.new_zeros((2,0), dtype=torch.long), (edge_type[:0] if edge_type is not None else None)
    idx = idx.to(edge_index.device)
    idx_map = -torch.ones(num_nodes, dtype=torch.long, device=edge_index.device)
    idx_map[idx] = torch.arange(idx.numel(), device=edge_index.device)
    ei = idx_map[edge_index]
    mask = (ei[0] >= 0) & (ei[1] >= 0)
    ei = ei[:, mask]
    et = edge_type[mask] if edge_type is not None else None
    return ei, et

def group_actor_nodes(ids_g: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Returns:
      actors: list of unique track_uuid (excluding EGO/UNK_)
      groups: list of lists of *local* node indices per actor
    """
    actors: List[str] = []
    groups: List[List[int]] = []
    idx_map: Dict[str, int] = {}
    for j, nid in enumerate(ids_g):
        if (nid == "EGO") or str(nid).startswith("UNK_"):
            continue
        a = str(nid)
        if a not in idx_map:
            idx_map[a] = len(actors)
            actors.append(a)
            groups.append([j])
        else:
            groups[idx_map[a]].append(j)
    return actors, groups

def pool_actor_scores(s_local: torch.Tensor,
                      groups: List[List[int]],
                      mode: str = "max") -> torch.Tensor:
    if not groups:
        return s_local[:0]
    outs = []
    for g in groups:
        if len(g) == 0:  # safety
            outs.append(s_local.new_zeros(()))
            continue
        idx = torch.tensor(g, device=s_local.device, dtype=torch.long)
        idx = idx.clamp_min(0)
        idx = idx[idx < s_local.numel()]  # guard
        if idx.numel() == 0:
            outs.append(s_local.new_zeros(()))
            continue
        if mode == "max":
            outs.append(torch.max(s_local[idx]))
        elif mode == "mean":
            outs.append(torch.mean(s_local[idx]))
        else:
            outs.append(torch.max(s_local[idx]))
    return torch.stack(outs, dim=0)

# ------------------------- extra losses -------------------------
def loss_pairwise_margin(pooled_scores: torch.Tensor, actor_ids: List[str], gt_set: Set[str], margin: float):
    """Multi-positive hinge vs hardest negative."""
    if pooled_scores.numel() == 0 or not gt_set:
        return pooled_scores.new_zeros(())
    P = [i for i,a in enumerate(actor_ids) if a in gt_set]
    N = [i for i,a in enumerate(actor_ids) if a not in gt_set]
    if len(P) == 0 or len(N) == 0:
        return pooled_scores.new_zeros(())
    s = pooled_scores
    sN_max = s[N].max()
    loss = 0.0
    for i in P:
        loss += F.relu(margin - (s[i] - sN_max))
    return loss / float(len(P))

# ------------------------------ training ------------------------------
def run_epoch_actor_level(loader: DataLoader,
                          model: nn.Module,
                          opt,
                          device,
                          teacher_lut: Dict[Tuple[str,str], float],
                          gt_lut: Dict[str, Set[str]],
                          lambda_gt: float,
                          lambda_pair: float,
                          pair_margin: float,
                          temp_w: float,
                          temporal_edge_type_id: Optional[int],
                          pool_mode: str,
                          train: bool) -> Tuple[float, float, float, float, float]:
    if train: model.train()
    else: model.eval()

    total_loss = 0.0
    n_graphs = 0
    ndcgs = []
    gt_top1_hits = 0
    gt_p3_hits = 0
    gt_r3_hits = 0
    gt_total_with_targets = 0
    gt_total_pos = 0

    for data in loader:
        data = data.to(device)
        s = model(data.x, data.edge_index, getattr(data, "edge_type", None))
        batch_vec = data.batch
        idx_list = batch_slices(batch_vec)
        num_nodes = data.x.size(0)
        edge_type_all = getattr(data, "edge_type", None)

        loss_sum = s.new_zeros(())
        for gid, idx in enumerate(idx_list):
            # window_key & node_ids per-graph (batch-safe)
            if isinstance(data.window_key, (list, tuple)):
                wkey = str(data.window_key[gid])
            else:
                wkey = str(data.window_key)
            if isinstance(data.node_ids, list) and data.node_ids and isinstance(data.node_ids[0], list):
                ids_g = list(map(str, data.node_ids[gid]))
            else:
                flat = list(map(str, data.node_ids))
                ids_g = [flat[int(i)] for i in idx.detach().cpu().tolist()]
            # make sure local sizes agree
            s_loc = s[idx]
            # ---- actor groups & pooled scores (LOCAL indices only) ----
            actors, groups = group_actor_nodes(ids_g)
            s_act = pool_actor_scores(s_loc, groups, mode=pool_mode)  # [A] (A can be 0)

            # ---- teacher target @ actor-level ----
            if len(actors) > 0:
                q_act = torch.zeros(len(actors), dtype=torch.float32, device=device)
                sup_m = torch.zeros(len(actors), dtype=torch.bool, device=device)
                for a_i, a in enumerate(actors):
                    val = teacher_lut.get((wkey, a), None)
                    if val is not None and val > 0:
                        q_act[a_i] = float(val); sup_m[a_i] = True
                L_t = listnet_ce(s_act, q_act, sup_m) if sup_m.any() else s.new_zeros(())
            else:
                L_t = s.new_zeros(())

            # ---- GT target @ actor-level (multi-positive uniform) ----
            gt_set = gt_lut.get(wkey, None)
            if len(actors) > 0 and gt_set:
                present = [a for a in actors if a in gt_set]
                if len(present) > 0:
                    k = len(present)
                    # listwise CE with uniform mass over present GT actors
                    tgt = torch.zeros(len(actors), dtype=torch.float32, device=device)
                    for a in present:
                        tgt[actors.index(a)] = 1.0 / k
                    valid_m = torch.ones(len(actors), dtype=torch.bool, device=device)
                    L_g = listnet_ce(s_act, tgt, valid_m)

                    # pairwise margin (positives vs hardest negative in this graph)
                    L_pair = loss_pairwise_margin(s_act, actors, set(present), pair_margin) if lambda_pair > 0 else s.new_zeros(())
                else:
                    L_g = s.new_zeros(())
                    L_pair = s.new_zeros(())
            else:
                L_g = s.new_zeros(())
                L_pair = s.new_zeros(())

            # ---- node-level temporal smoothness (localized) ----
            ei_loc, et_loc = local_subgraph_edges(
                edge_index=data.edge_index, edge_type=edge_type_all,
                idx=idx, num_nodes=num_nodes
            )
            def temporal_smoothness_loss(scores: torch.Tensor,
                                         edge_index: torch.Tensor,
                                         edge_type: Optional[torch.Tensor],
                                         temporal_edge_type_id: Optional[int],
                                         weight: float = 1e-3) -> torch.Tensor:
                if weight <= 0 or edge_index is None:
                    return scores.new_zeros(())
                if edge_index.numel() == 0:
                    return scores.new_zeros(())
                if edge_type is None or temporal_edge_type_id is None:
                    return scores.new_zeros(())
                mask = (edge_type == int(temporal_edge_type_id))
                if not mask.any():
                    return scores.new_zeros(())
                ei = edge_index[:, mask]
                if ei.numel() == 0:
                    return scores.new_zeros(())
                si = scores[ei[0]]
                sj = scores[ei[1]]
                return weight * ((si - sj) ** 2).mean()

            L_temp = temporal_smoothness_loss(
                scores=s_loc, edge_index=ei_loc, edge_type=et_loc,
                temporal_edge_type_id=temporal_edge_type_id, weight=temp_w
            )

            loss_g = L_t + lambda_gt * L_g + lambda_pair * L_pair + L_temp
            loss_sum = loss_sum + loss_g

            # ---- teacher metric @ actor-level ----
            if len(actors) > 0:
                gains = np.array([teacher_lut.get((wkey, a), 0.0) for a in actors], dtype=float)
                if gains.sum() > 0:
                    ndcgs.append(
                        ndcg_at_k(
                            s_act.detach().cpu().numpy(),
                            gains,
                            k=3
                        )
                    )

            # ---- GT metrics @ actor-level ----
            if gt_set and len(actors) > 0:
                present = [a for a in actors if a in gt_set]
                if len(present) > 0:
                    gt_total_with_targets += 1
                    order = torch.argsort(s_act, descending=True).tolist()
                    topk = [actors[j] for j in order[:3]]
                    hit_top1 = 1 if (len(topk) >= 1 and topk[0] in gt_set) else 0
                    gt_top1_hits += hit_top1
                    hits_at3 = sum(1 for a in topk if a in gt_set)
                    gt_p3_hits += hits_at3
                    gt_total_pos += len(set(present))
                    gt_r3_hits += hits_at3

            n_graphs += 1

        if train and opt is not None:
            opt.zero_grad()
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += float(loss_sum.detach().cpu())

    mean_loss = total_loss / max(1, n_graphs)
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    gt_top1 = (gt_top1_hits / max(1, gt_total_with_targets)) if gt_total_with_targets else 0.0
    gt_p3   = (gt_p3_hits   / max(1, 3 * gt_total_with_targets)) if gt_total_with_targets else 0.0
    gt_r3   = (gt_r3_hits   / max(1, gt_total_pos)) if gt_total_pos else 0.0
    return mean_loss, mean_ndcg, gt_top1, gt_p3, gt_r3

@torch.no_grad()
def export_topk_actor_level(loader: DataLoader, model: nn.Module, device, out_parquet: Path, k: int = 3, pool_mode: str = "max", temp_T: float = 1.0):
    model.eval()
    rows = []
    for data in loader:
        data = data.to(device)
        s = model(data.x, data.edge_index, getattr(data, "edge_type", None)).detach()
        if temp_T > 0:
            s = s / float(temp_T)  # calibration rescales logits; ranking unchanged
        s = s.cpu()
        batch_vec = data.batch.cpu()
        if isinstance(data.window_key, (list, tuple)):
            wkeys = [str(w) for w in data.window_key]
        else:
            n_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 1
            wkeys = [str(data.window_key)] * n_graphs

        ids_per_graph = []
        if isinstance(data.node_ids, list) and data.node_ids and isinstance(data.node_ids[0], list):
            ids_per_graph = [list(map(str, sub)) for sub in data.node_ids]
        else:
            flat = list(map(str, data.node_ids))
            for idx in batch_slices(batch_vec):
                ids_per_graph.append([flat[int(i)] for i in idx])

        for g, idx in enumerate(batch_slices(batch_vec)):
            wkey = wkeys[g]
            ids_g = ids_per_graph[g]
            actors, groups = group_actor_nodes(ids_g)
            if len(actors) == 0:
                continue
            s_act = pool_actor_scores(s[idx], groups, mode=pool_mode).numpy()
            order = np.argsort(-s_act)[:k]
            for rank, j in enumerate(order, 1):
                rows.append({
                    "window_key": wkey,
                    "rank": rank,
                    "track_uuid": actors[j],
                    "score": float(s_act[j]),
                })

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)

# ---------------- temperature calibration (VAL, multi-positive) ---------------
@torch.no_grad()
def _collect_val_actor_logits(loader, model, device, gt_lut, pool_mode):
    model.eval()
    blobs = []
    for data in loader:
        data = data.to(device)
        s = model(data.x, data.edge_index, getattr(data, "edge_type", None))
        batch = data.batch
        idx_list = batch_slices(batch)
        for gid, idx in enumerate(idx_list):
            if isinstance(data.window_key, (list, tuple)):
                wk = str(data.window_key[gid])
            else:
                wk = str(data.window_key)
            gt_set = gt_lut.get(wk, set())
            if not gt_set: continue
            if isinstance(data.node_ids, list) and data.node_ids and isinstance(data.node_ids[0], list):
                ids_g = list(map(str, data.node_ids[gid]))
            else:
                flat = list(map(str, data.node_ids))
                ids_g = [flat[int(i)] for i in idx.detach().cpu().tolist()]
            actors, groups = group_actor_nodes(ids_g)
            if not actors: continue
            s_act = pool_actor_scores(s[idx], groups, mode=pool_mode).detach().cpu().numpy().astype(np.float64)
            gt_mask = np.array([a in gt_set for a in actors], dtype=bool)
            if not gt_mask.any(): continue
            blobs.append((s_act, gt_mask))
    return blobs

def _nll_multi_pos_at_T(blobs, T: float):
    total, ct = 0.0, 0
    for logits, gt_mask in blobs:
        z = logits / max(1e-6, T)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        prob_gt = float(p[gt_mask].sum())
        prob_gt = max(prob_gt, 1e-12)
        total += -math.log(prob_gt)
        ct += 1
    return total / max(1, ct)

def calibrate_temperature(blobs, T_init=1.0, steps=200, lr=0.05):
    logT = math.log(max(1e-3, T_init))
    for _ in range(steps):
        T = math.exp(logT)
        eps = 1e-4
        f = _nll_multi_pos_at_T(blobs, T)
        f_eps = _nll_multi_pos_at_T(blobs, T * math.exp(eps))
        grad = (f_eps - f) / eps
        logT -= lr * grad
        logT = min(max(logT, math.log(1e-3)), math.log(100.0))
    return float(math.exp(logT))

# ------------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)
    ap.add_argument("--graphs-split", required=True, help="Subdir under s3/quasi/graphs/ (e.g., train650, dev100, val50)")
    ap.add_argument("--split-mode", choices=["fixed80_20","all_train"], default="fixed80_20")
    ap.add_argument("--val-frac", type=float, default=0.20)

    # model
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--edge-type-dim", type=int, default=16)

    # multi-task weights (curriculum)
    ap.add_argument("--lambda-gt", type=float, default=0.7, help="Final GT loss weight after warmup")
    ap.add_argument("--lambda-gt-warmup-epochs", type=int, default=8, help="Warmup length in epochs")

    # pairwise margin
    ap.add_argument("--lambda-pair", type=float, default=0.3, help="Weight for pairwise margin loss")
    ap.add_argument("--pair-margin", type=float, default=0.15, help="Margin for positives vs hardest negative")

    # temporal smoothness
    ap.add_argument("--lambda-temp", type=float, default=1e-3, help="Temporal smoothness weight")
    ap.add_argument("--temporal-edge-type-id", type=int, default=3, help="Edge type id that denotes temporal edges")

    # teacher / GT sources
    ap.add_argument("--tags-norm", required=True, help="Path to tags_norm.jsonl")
    ap.add_argument("--gt-time-decimals", type=int, default=6)
    ap.add_argument("--gt-time-tol-s", type=float, default=0.05)
    ap.add_argument("--gt-allow-log-fallback", type=int, default=0)

    # pooling
    ap.add_argument("--pool-mode", choices=["max","mean"], default="max", help="Actor pooling across nodes")

    # temperature calibration
    ap.add_argument("--calibrate-temp", type=int, default=1)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(safe_device(args.device))

    root = Path(args.split_root)
    graphs_dir = root / "s3" / "quasi" / "graphs" / args.graphs_split
    teacher_parquet = root / "s3" / "quasi" / "teacher.parquet"

    print(f"[S3-A5][MT-ACTOR] graphs_dir={graphs_dir}")
    files = collect_files_from_manifest(graphs_dir)
    print(f"[S3-A5][MT-ACTOR] first_graph={files[0]}")
    if len(files) > 1:
        print(f"[S3-A5][MT-ACTOR] second_graph={files[1]}")

    # splits
    tr_idx, va_idx = split_by_keys(files, graphs_dir / "manifest.parquet", args.split_mode, args.val_frac, args.seed)
    ds_tr = GraphDataset([files[i] for i in tr_idx])
    ds_va = GraphDataset([files[i] for i in va_idx]) if va_idx else GraphDataset([files[i] for i in tr_idx[-max(1,len(tr_idx)//5):]])

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # model dims & edge types
    g0: Data = torch.load(files[0], map_location="cpu")
    in_dim = int(g0.x.size(1))
    uniq_types = set()
    for p in files[: min(8, len(files))]:
        gg: Data = torch.load(p, map_location="cpu")
        if hasattr(gg, "edge_type"):
            uniq_types.update(set(gg.edge_type.detach().cpu().numpy().tolist()))
    num_edge_types = max(1, len(uniq_types) if uniq_types else 1)

    model = GATQuasi(
        in_dim=in_dim,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        edge_type_dim=args.edge_type_dim,
        num_edge_types=num_edge_types,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    teacher_lut = build_teacher_lookup(teacher_parquet)
    gt_lut = build_gt_lookup(
        split_root=root,
        graphs_dir=graphs_dir,
        tags_norm_path=Path(args.tags_norm),
        time_decimals=args.gt_time_decimals,
        time_tol_s=args.gt_time_tol_s,
        allow_log_fallback=int(args.gt_allow_log_fallback)
    )
    print(f"[GT] windows with GT = {len(gt_lut)} (multi-positives supported)")

    def gt_lambda_schedule(ep: int) -> float:
        if args.lambda_gt <= 0: return 0.0
        W = max(1, int(args.lambda_gt_warmup_epochs))
        return float(args.lambda_gt) * min(1.0, ep / W)

    best_combo = -1.0
    out_dir = graphs_dir.parent / "gat_multitask_v9_pair_temp_f_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    for ep in range(1, args.epochs + 1):
        lam = gt_lambda_schedule(ep)
        tr_loss, tr_ndcg, tr_top1, tr_p3, tr_r3 = run_epoch_actor_level(
            dl_tr, model, opt, device, teacher_lut, gt_lut,
            lambda_gt=lam, lambda_pair=args.lambda_pair, pair_margin=args.pair_margin,
            temp_w=args.lambda_temp, temporal_edge_type_id=int(args.temporal_edge_type_id),
            pool_mode=args.pool_mode, train=True
        )
        va_loss, va_ndcg, va_top1, va_p3, va_r3 = run_epoch_actor_level(
            dl_va, model, None, device, teacher_lut, gt_lut,
            lambda_gt=lam, lambda_pair=args.lambda_pair, pair_margin=args.pair_margin,
            temp_w=args.lambda_temp, temporal_edge_type_id=int(args.temporal_edge_type_id),
            pool_mode=args.pool_mode, train=False
        )

        combo = 0.5 * va_top1 + 0.3 * va_p3 + 0.2 * va_ndcg
        print(f"[S3-A5][MT-ACTOR] ep {ep:02d}/{args.epochs}  "
              f"trainLoss={tr_loss:.4f}  valNDCG@3={va_ndcg:.4f}  "
              f"GT_top1={va_top1:.4f}  GT_P@3={va_p3:.4f}  GT_R@3={va_r3:.4f}  "
              f"in_dim={in_dim}  device={device.type}  λ_GT={lam:.3f}  pool={args.pool_mode}")

        if combo > best_combo:
            best_combo = combo
            torch.save(model.state_dict(), best_path)

    # Temperature calibration on VAL (optional)
    temp_T = 1.0
    if args.calibrate_temp and len(va_idx) > 0:
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        blobs = _collect_val_actor_logits(dl_va, model, device, gt_lut, pool_mode=args.pool_mode)
        if blobs:
            temp_T = calibrate_temperature(blobs, T_init=1.0, steps=200, lr=0.05)
            print(f"[S3-A5][MT-ACTOR] Calibrated temperature T = {temp_T:.4f} on VAL.")
        else:
            print("[S3-A5][MT-ACTOR] No GT windows on VAL for temperature calibration (T=1.0).")

    # --------------------------- INFERENCE ON ALL FILES ---------------------------
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

        # Create loader for ALL files (Train + Val)
        print(f"[S3-A5][MT-ACTOR] Loading ALL {len(files)} graphs for inference...")
        ds_all = GraphDataset(files)
        dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, drop_last=False)

        out_inference = out_dir / "top3_inference.parquet"
        export_topk_actor_level(dl_all, model, device, out_inference, k=3, pool_mode=args.pool_mode, temp_T=temp_T)

        print(f"[S3-A5][MT-ACTOR] Saved best model → {best_path}")
        print(f"[S3-A5][MT-ACTOR] Exported Top-3 for ALL logs (Train+Val) → {out_inference}")

    (out_dir / "meta_train.json").write_text(json.dumps({
        "split_root": str(root),
        "graphs_dir": str(graphs_dir),
        "n_train": len(tr_idx), "n_val": len(va_idx),
        "in_dim": in_dim, "hidden": args.hidden, "heads": args.heads, "layers": args.layers,
        "dropout": args.dropout, "edge_type_dim": args.edge_type_dim, "num_edge_types": num_edge_types,
        "lambda_gt": args.lambda_gt, "lambda_gt_warmup_epochs": args.lambda_gt_warmup_epochs,
        "lambda_pair": args.lambda_pair, "pair_margin": args.pair_margin,
        "lambda_temp": args.lambda_temp, "temporal_edge_type_id": int(args.temporal_edge_type_id),
        "gt_time_decimals": args.gt_time_decimals, "gt_time_tol_s": args.gt_time_tol_s,
        "gt_allow_log_fallback": int(args.gt_allow_log_fallback),
        "pool_mode": args.pool_mode,
        "best_combo_metric": best_combo,
        "calibrated_T": temp_T,
        "seed": args.seed, "device": device.type,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

if __name__ == "__main__":
    main()
