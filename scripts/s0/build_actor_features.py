#!/usr/bin/env python3
"""
S0 – Freeze & Pack: build per-(window, actor) features with AV2 map cues.

Changes:
- AV2 root is auto-read from paths.yaml (key: av2_sensor_root / DATA_DIR / av2_sensor_dir).
- in_drivable_area: 1/0 using drivable area polygons if present; else fallback to lane proximity (<3.5 m).
- dist_to_stopline_m: proxy as min distance to a crosswalk edge ahead of ego heading; fallback to
  min distance to lane centerlines marked is_intersection=true; else NaN.

Outputs Parquet and prints shape, size, and first row.
"""

from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

# Optional AV2 API
try:
    from av2.map.map_api import ArgoverseStaticMap  # type: ignore
    HAVE_AV2_MAP = True
except Exception:
    ArgoverseStaticMap = None
    HAVE_AV2_MAP = False

# Ego pose helper
try:
    from signals_to_semantics.tagging.state_est import _quat_to_yaw
except Exception:
    _quat_to_yaw = None


# ---------------- Pose / root utils ----------------
def _get_root_from_cfg(cfg: dict):
    for k in ("av2_sensor_root", "DATA_DIR", "av2_sensor_dir"):
        if k in cfg:
            return cfg[k]
    raise KeyError("Expected one of av2_sensor_root/DATA_DIR in paths yaml")

def _load_pose_feather(paths_yaml: str, log_id: str):
    root = Path(_get_root_from_cfg(yaml.safe_load(open(paths_yaml))))
    df_pose = pd.read_feather(root / log_id / "city_SE3_egovehicle.feather")
    t_s = (df_pose["timestamp_ns"].to_numpy(np.int64).astype(np.float64) * 1e-9)
    qw = df_pose.get("qw", pd.Series(np.zeros(len(df_pose)))).to_numpy(np.float64)
    qx = df_pose.get("qx", pd.Series(np.zeros(len(df_pose)))).to_numpy(np.float64)
    qy = df_pose.get("qy", pd.Series(np.zeros(len(df_pose)))).to_numpy(np.float64)
    qz = df_pose.get("qz", pd.Series(np.ones (len(df_pose)))).to_numpy(np.float64)
    tx = df_pose.get("tx_m", pd.Series(np.zeros(len(df_pose)))).to_numpy(np.float64)
    ty = df_pose.get("ty_m", pd.Series(np.zeros(len(df_pose)))).to_numpy(np.float64)
    if _quat_to_yaw is None:
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    else:
        yaw = np.array([_quat_to_yaw(a,b,c,d) for a,b,c,d in zip(qw,qx,qy,qz)], dtype=np.float64)
    return t_s, yaw, tx, ty

def _interp_pose_at(t_query: float, pose_ref: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float,float,float]:
    t_s, yaw, tx, ty = pose_ref
    yaw_q = float(np.interp(t_query, t_s, yaw))
    tx_q  = float(np.interp(t_query, t_s, tx))
    ty_q  = float(np.interp(t_query, t_s, ty))
    return yaw_q, tx_q, ty_q

def _ego_to_city(x_rel: float, y_rel: float, ego_yaw: float, ego_tx: float, ego_ty: float) -> Tuple[float,float]:
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    x_city = ego_tx + c * x_rel - s * y_rel
    y_city = ego_ty + s * x_rel + c * y_rel
    return x_city, y_city


# ---------------- Map loading helpers ----------------
def _choose_map_json(map_dir: Path) -> Optional[Path]:
    if not map_dir.exists(): return None
    cand = sorted(map_dir.glob("log_map_archive_*city_*.json"))
    if cand: return cand[0]
    cand = sorted(map_dir.glob("*_img_Sim2_city.json"))
    if cand: return cand[0]
    cand = sorted(map_dir.glob("*.json"))
    return cand[0] if cand else None

def _load_av2_map_api(log_id: str, base_root: Optional[str]) -> Optional[Any]:
    if not HAVE_AV2_MAP or base_root is None:
        return None
    map_json = _choose_map_json(Path(base_root) / log_id / "map")
    if map_json is None:
        return None
    try:
        return ArgoverseStaticMap.from_json(map_json)
    except Exception:
        return None

class LightMap:
    def __init__(self,
                 lane_centerlines: List[np.ndarray],
                 crosswalk_edges: List[np.ndarray],
                 drivable_polys: List[np.ndarray],
                 intersection_centerlines: List[np.ndarray]):
        self.lane_centerlines = lane_centerlines
        self.crosswalk_edges  = crosswalk_edges
        self.drivable_polys   = drivable_polys
        self.intersection_centerlines = intersection_centerlines

def _to_xy(arr) -> np.ndarray:
    if not arr:
        return np.zeros((0,2), dtype=float)
    f = arr[0]
    if isinstance(f, dict):
        xs = [p.get("x", 0.0) for p in arr]
        ys = [p.get("y", 0.0) for p in arr]
        return np.stack([xs, ys], axis=1).astype(float)
    if isinstance(f, (list, tuple)) and len(f) >= 2:
        return np.array([[p[0], p[1]] for p in arr], dtype=float)
    return np.zeros((0,2), dtype=float)

def _centerline_from_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] < 2 or right.shape[0] < 2: return np.zeros((0,2), float)
    def sample(poly: np.ndarray, k=50) -> np.ndarray:
        seg = np.diff(poly, axis=0)
        seglen = np.hypot(seg[:,0], seg[:,1])
        L = float(np.sum(seglen))
        if L < 1e-6: return np.repeat(poly[:1], k, axis=0)
        t = np.linspace(0.0, L, k)
        out = []
        acc = 0.0; j = 0
        for ti in t:
            while j < len(seglen) and (acc + seglen[j]) < ti:
                acc += seglen[j]; j += 1
            if j >= len(seglen): out.append(poly[-1]); continue
            r = (ti - acc) / max(seglen[j], 1e-9)
            p = poly[j] + r * (poly[j+1] - poly[j])
            out.append(p)
        return np.array(out, float)
    return 0.5*(sample(left,50) + sample(right,50))

def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return d

def _point_to_polyline_nearest(P: np.ndarray, poly: np.ndarray) -> Tuple[float, int, float]:
    """Return (distance, idx, seg_theta) where idx is nearest segment start index."""
    if poly.shape[0] < 2:
        return float("inf"), -1, float("nan")
    d_best = float("inf"); idx_best = -1; seg_theta = float("nan")
    for i in range(len(poly)-1):
        A = poly[i]; B = poly[i+1]; AB = B-A; AP = P-A
        t = np.clip(np.dot(AP, AB) / (np.dot(AB, AB) + 1e-12), 0.0, 1.0)
        Q = A + t*AB
        d = float(np.hypot(*(P-Q)))
        if d < d_best:
            d_best = d; idx_best = i
            dx, dy = (B - A)
            seg_theta = math.atan2(dy, dx)
    return d_best, idx_best, seg_theta

def _point_to_polyline_distance(P: np.ndarray, poly: np.ndarray) -> float:
    d, _, _ = _point_to_polyline_nearest(P, poly)
    return d

def _point_in_polygon(P: np.ndarray, poly: np.ndarray) -> bool:
    x, y = P
    inside = False
    n = len(poly)
    if n < 3: return False
    x0, y0 = poly[-1]
    for i in range(n):
        x1, y1 = poly[i]
        if ((y1 > y) != (y0 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0):
            inside = not inside
        x0, y0 = x1, y1
    return inside

def _load_light_map(log_id: str, base_root: Optional[str]) -> Optional[LightMap]:
    if base_root is None: return None
    map_json = _choose_map_json(Path(base_root) / log_id / "map")
    if map_json is None: return None
    try:
        data = json.load(open(map_json, "r"))
    except Exception:
        return None

    lane_centerlines : List[np.ndarray] = []
    cross_edges      : List[np.ndarray] = []
    drivable_polys   : List[np.ndarray] = []
    inter_center     : List[np.ndarray] = []

    lanes = data.get("lane_segments") or data.get("lanes") or {}
    if isinstance(lanes, dict):
        for _lid, lane in lanes.items():
            left  = _to_xy(lane.get("left_lane_boundary",  []))
            right = _to_xy(lane.get("right_lane_boundary", []))
            C = _centerline_from_boundaries(left, right)
            if C.shape[0] >= 2:
                lane_centerlines.append(C)
                if bool(lane.get("is_intersection", False)):
                    inter_center.append(C)

    peds = data.get("pedestrian_crossings") or {}
    if isinstance(peds, dict):
        for _cid, cw in peds.items():
            e1 = _to_xy(cw.get("edge1", []))
            e2 = _to_xy(cw.get("edge2", []))
            if e1.shape[0] >= 2: cross_edges.append(e1)
            if e2.shape[0] >= 2: cross_edges.append(e2)

    drs = data.get("drivable_areas") or data.get("drivable_area") or {}
    if isinstance(drs, dict):
        for _did, da in drs.items():
            poly = _to_xy(da.get("area_boundary", []))
            if poly.shape[0] >= 3:
                drivable_polys.append(poly)

    if not lane_centerlines and not cross_edges and not drivable_polys:
        return None
    return LightMap(lane_centerlines, cross_edges, drivable_polys, inter_center)


# ---------------- Map feature front-ends ----------------
def _map_features_for_point_with_av2(amap, x_city: float, y_city: float, heading_rad: Optional[float]) -> Dict[str, float]:
    out = dict(
        map_lane_offset_m=np.nan,
        map_lane_alignment_cos=np.nan,
        dist_to_stopline_m=np.nan,
        dist_to_crosswalk_m=np.nan,
        in_drivable_area=np.nan,
    )
    if amap is None:
        return out
    try:
        lane = amap.get_nearest_centerline([x_city, y_city])
        if lane is not None:
            cl = np.array(lane.xyz_centerline)[:, :2]
            d, idx, seg_th = _point_to_polyline_nearest(np.array([x_city, y_city]), cl)
            out["map_lane_offset_m"] = d
            out["map_lane_alignment_cos"] = float(math.cos((heading_rad or 0.0) - seg_th)) if heading_rad is not None else np.nan

        # crosswalk distance (polygon or polyline)
        d_cw = float("inf")
        for cw in getattr(amap, "crosswalks", []) or []:
            poly = None
            if hasattr(cw, "polygon") and cw.polygon is not None:
                poly = np.array(cw.polygon)[:,:2]
            elif hasattr(cw, "xyz_polyline") and cw.xyz_polyline is not None:
                poly = np.array(cw.xyz_polyline)[:,:2]
            if poly is not None and len(poly) > 1:
                d_cw = min(d_cw, _point_to_polyline_distance(np.array([x_city, y_city]), poly))
        out["dist_to_crosswalk_m"] = d_cw if math.isfinite(d_cw) else np.nan

        # drivable area
        try:
            out["in_drivable_area"] = float(1.0 if amap.is_in_drivable_area([x_city, y_city]) else 0.0)
        except Exception:
            out["in_drivable_area"] = np.nan

        # stopline proxy not available in AV2 helper here; leave NaN.
        return out
    except Exception:
        return out

def _map_features_for_point_light(lmap: Optional[LightMap], x_city: float, y_city: float, heading_rad: Optional[float]) -> Dict[str, float]:
    out = dict(
        map_lane_offset_m=np.nan,
        map_lane_alignment_cos=np.nan,
        dist_to_stopline_m=np.nan,
        dist_to_crosswalk_m=np.nan,
        in_drivable_area=np.nan,
    )
    if lmap is None:
        return out

    P = np.array([x_city, y_city], dtype=float)

    # lane distance + alignment
    d_best = float("inf"); align_best = float("nan")
    for cl in lmap.lane_centerlines:
        d, idx, seg_th = _point_to_polyline_nearest(P, cl)
        if d < d_best:
            d_best = d
            align_best = float(math.cos((heading_rad or 0.0) - seg_th)) if heading_rad is not None else float("nan")
    if math.isfinite(d_best): out["map_lane_offset_m"] = d_best
    if math.isfinite(align_best): out["map_lane_alignment_cos"] = align_best

    # crosswalk distance (min to edges)
    d_cw = float("inf")
    for e in lmap.crosswalk_edges:
        d, _, _ = _point_to_polyline_nearest(P, e)
        d_cw = min(d_cw, d)
    if math.isfinite(d_cw): out["dist_to_crosswalk_m"] = d_cw

    # stopline proxy: nearest crosswalk edge "ahead" of ego heading; else nearest intersection centerline
    stop_proxy = float("inf")
    if heading_rad is not None:
        hvec = np.array([math.cos(heading_rad), math.sin(heading_rad)], float)
        for e in lmap.crosswalk_edges:
            d, idx, seg_th = _point_to_polyline_nearest(P, e)
            if idx >= 0:
                # "ahead" if segment direction roughly aligned with heading (within 90°)
                if abs(_angle_diff(seg_th, heading_rad)) < (math.pi/2):
                    stop_proxy = min(stop_proxy, d)
    if not math.isfinite(stop_proxy) or stop_proxy == float("inf"):
        for cl in lmap.intersection_centerlines:
            d = _point_to_polyline_distance(P, cl)
            stop_proxy = min(stop_proxy, d)
    if math.isfinite(stop_proxy) and stop_proxy != float("inf"):
        out["dist_to_stopline_m"] = stop_proxy

    # drivable membership: polygons → 1 if inside any, else 0
    if len(lmap.drivable_polys) > 0:
        inside_any = any(_point_in_polygon(P, poly) for poly in lmap.drivable_polys)
        out["in_drivable_area"] = 1.0 if inside_any else 0.0
    else:
        # fallback: near any lane centerline (<3.5m) → treat as drivable
        near_lane = (math.isfinite(d_best) and d_best < 3.5)
        out["in_drivable_area"] = 1.0 if near_lane else 0.0

    return out


# ---------------- Feature engineering ----------------
def _rank_and_margin(vals: np.ndarray, higher_is_better=True) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(vals)[::-1] if higher_is_better else np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(vals)+1, dtype=np.int64)
    top = np.nanmax(vals) if higher_is_better else np.nanmin(vals)
    margin = (top - vals) if higher_is_better else (vals - top)
    return ranks, margin

def _features_from_actor(window: Dict[str, Any],
                         actor: Dict[str, Any],
                         pose_ref: Optional[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]],
                         amap_obj: Optional[Any],
                         lmap_obj: Optional[LightMap],
                         av2_root: Optional[str]) -> Dict[str, Any]:
    log_id = window["log_id"]
    t_start = float(window["t_start"]); t_end = float(window["t_end"])
    t_center = 0.5*(t_start + t_end)

    def g(name, default=np.nan):
        return actor.get(name, actor.get(name.replace("_mps","_mps"), default))

    x_rel = float(g("x_rel_m", g("x_rel", np.nan)))
    y_rel = float(g("y_rel_m", g("y_rel", np.nan)))

    dmin = float(g("dmin_m", np.nan))
    p_overlap = float(g("p_overlap", np.nan))
    t_at_dmin = float(g("t_at_dmin_s", np.nan))
    sustained = bool(g("sustained_tight", False))
    rel_close = float(g("rel_speed_closing_mps", np.nan))
    lat_speed = float(g("lat_speed_mps", g("v_lat_ego_mps", np.nan)))
    a_norm    = float(g("a_norm", np.nan))
    sector_id = int(g("sector_id", -1))
    on_path   = int(g("on_path_like", 0))
    bearing   = float(g("bearing_rad", np.nan))
    length_m  = float(g("length_m", np.nan))
    width_m   = float(g("width_m",  np.nan))
    is_vehicle = int(g("is_vehicle", 0))
    is_vru     = int(g("is_vru", 0))
    is_static  = int(g("is_static", 0))
    category   = str(g("category", "UNKNOWN"))
    track_uuid = str(g("track_uuid", "NA"))

    ttc = np.nan
    if rel_close > 1e-6:
        ttc = dmin / rel_close
    elif rel_close <= -1e-6:
        ttc = -dmin / rel_close

    dist_lt3 = float(1.0 if dmin <= 3.0 else 0.0)
    dist_lt5 = float(1.0 if dmin <= 5.0 else 0.0)
    dist_lt8 = float(1.0 if dmin <= 8.0 else 0.0)

    r = math.hypot(x_rel, y_rel) if (np.isfinite(x_rel) and np.isfinite(y_rel)) else np.nan
    long_gap = x_rel
    lat_off  = y_rel
    heading_align = float(math.cos(bearing)) if np.isfinite(bearing) else np.nan
    approach_like = float(1.0 if (rel_close > 0.5 and long_gap > -2.0) else 0.0)
    crossing_like = float(1.0 if (abs(lat_speed) > 0.4 and abs(lat_off) < 8.5) else 0.0)

    map_feats = dict(map_lane_offset_m=np.nan, map_lane_alignment_cos=np.nan,
                     dist_to_stopline_m=np.nan, dist_to_crosswalk_m=np.nan,
                     in_drivable_area=np.nan)

    if pose_ref is not None and np.isfinite(x_rel) and np.isfinite(y_rel):
        yaw, tx, ty = _interp_pose_at(t_center, pose_ref)
        ax_city, ay_city = _ego_to_city(x_rel, y_rel, yaw, tx, ty)
        heading_for_map = yaw  # use ego heading as a proxy
        if amap_obj is not None:
            map_feats = _map_features_for_point_with_av2(amap_obj, ax_city, ay_city, heading_for_map)
        else:
            map_feats = _map_features_for_point_light(lmap_obj, ax_city, ay_city, heading_for_map)

    feat = dict(
        log_id=log_id, window_t_start=t_start, window_t_end=t_end, window_center=t_center,
        track_uuid=track_uuid, category=category, is_vehicle=is_vehicle, is_vru=is_vru, is_static=is_static,
        sector_id=sector_id, on_path_like=on_path,
        x_rel_m=x_rel, y_rel_m=y_rel, r_rel_m=r,
        dmin_m=dmin, p_overlap=p_overlap, t_at_dmin_s=t_at_dmin, sustained_tight=float(1.0 if sustained else 0.0),
        rel_speed_closing_mps=rel_close, lat_speed_mps=lat_speed, a_norm=a_norm,
        length_m=length_m, width_m=width_m, bearing_rad=bearing,
        ttc_s=ttc, dist_lt3=dist_lt3, dist_lt5=dist_lt5, dist_lt8=dist_lt8,
        approach_like=approach_like, crossing_like=crossing_like,
        heading_align_cos=heading_align, long_gap_m=long_gap, lat_offset_m=lat_off,
        map_lane_offset_m=map_feats["map_lane_offset_m"],
        map_lane_alignment_cos=map_feats["map_lane_alignment_cos"],
        dist_to_stopline_m=map_feats["dist_to_stopline_m"],
        dist_to_crosswalk_m=map_feats["dist_to_crosswalk_m"],
        in_drivable_area=map_feats["in_drivable_area"],
    )
    return feat


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-jsonl", required=True)
    ap.add_argument("--paths-yaml", required=True)
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    # discover AV2 root from paths.yaml
    try:
        av2_root = _get_root_from_cfg(yaml.safe_load(open(args.paths_yaml)))
    except Exception:
        av2_root = None

    pose_cache : Dict[str, Optional[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]] = {}
    amap_cache : Dict[str, Any] = {}
    lmap_cache : Dict[str, Optional[LightMap]] = {}

    rows_out : List[Dict[str,Any]] = []

    with open(args.windows_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            win = json.loads(line)
            log_id = win["log_id"]
            acts = win.get("top_actors", [])

            # pose
            if log_id not in pose_cache:
                try:
                    pose_cache[log_id] = _load_pose_feather(args.paths_yaml, log_id)
                except Exception:
                    pose_cache[log_id] = None
            pose_ref = pose_cache[log_id]

            # map objects (prefer AV2; else lightweight)
            if log_id not in amap_cache:
                amap_cache[log_id] = _load_av2_map_api(log_id, av2_root)
            amap = amap_cache[log_id]
            if amap is None:
                if log_id not in lmap_cache:
                    lmap_cache[log_id] = _load_light_map(log_id, av2_root)
                lmap = lmap_cache[log_id]
            else:
                lmap = None

            if isinstance(acts, list) and acts:
                p_all = np.array([float(a.get("p_overlap", np.nan)) for a in acts], dtype=float)
                d_all = np.array([float(a.get("dmin_m",     np.nan)) for a in acts], dtype=float)
                p_ranks, p_margin = _rank_and_margin(p_all, higher_is_better=True)
                invd_ranks, invd_margin = _rank_and_margin(-d_all, higher_is_better=True)

                for i, a in enumerate(acts):
                    feat = _features_from_actor(win, a, pose_ref, amap, lmap, av2_root)
                    feat["rank_p_overlap"] = int(p_ranks[i]) if np.isfinite(p_all[i]) else np.nan
                    feat["margin_to_top_p"] = float(p_margin[i]) if np.isfinite(p_all[i]) else np.nan
                    feat["rank_inv_dmin"]   = int(invd_ranks[i]) if np.isfinite(d_all[i]) else np.nan
                    feat["margin_to_top_inv_d"] = float(invd_margin[i]) if np.isfinite(d_all[i]) else np.nan
                    rows_out.append(feat)

    if not rows_out:
        print("No rows parsed. Check inputs.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows_out)

    meta_cols = ["log_id","window_t_start","window_t_end","window_center","track_uuid","category",
                 "is_vehicle","is_vru","is_static","sector_id","on_path_like"]
    geom_cols = ["x_rel_m","y_rel_m","r_rel_m","dmin_m","p_overlap","t_at_dmin_s","sustained_tight",
                 "rel_speed_closing_mps","lat_speed_mps","a_norm","length_m","width_m","bearing_rad"]
    kin_cols  = ["ttc_s","dist_lt3","dist_lt5","dist_lt8","approach_like","crossing_like",
                 "heading_align_cos","long_gap_m","lat_offset_m"]
    map_cols  = ["map_lane_offset_m","map_lane_alignment_cos","dist_to_stopline_m","dist_to_crosswalk_m","in_drivable_area"]
    ctx_cols  = ["rank_p_overlap","margin_to_top_p","rank_inv_dmin","margin_to_top_inv_d"]
    ordered = [c for c in (meta_cols+geom_cols+kin_cols+map_cols+ctx_cols) if c in df.columns] + \
              [c for c in df.columns if c not in (meta_cols+geom_cols+kin_cols+map_cols+ctx_cols)]
    df = df[ordered]

    outp = Path(args.out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    n_rows, n_cols = df.shape
    try:
        size_mb = outp.stat().st_size / (1024*1024)
    except Exception:
        size_mb = float("nan")
    print(f"Wrote parquet: {outp}")
    print(f"Data Dimensions (Rows, Columns): ({n_rows}, {n_cols})")
    print(f"File size: {size_mb:.2f} MB")
    first_row = df.iloc[0].to_dict()
    kv = ", ".join([f"{k}={first_row[k]!r}" for k in df.columns])
    print("First row (all columns):")
    print(kv)

if __name__ == "__main__":
    main()
