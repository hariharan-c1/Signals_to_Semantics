"""Probabilistic actor-trajectory overlap scoring."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import yaml
from math import atan2, cos, sin, hypot, pi

from shapely.geometry import Polygon
from signals_to_semantics.io_av2.av2_loader import load_ego_series
from signals_to_semantics.tagging.geom import oriented_box_polygon, box_distance
from signals_to_semantics.tagging.state_est import estimate_state_city, _quat_to_yaw

# ---------- Utilities ----------
def _get_root_from_cfg(cfg: dict):
    for k in ("av2_sensor_root", "DATA_DIR", "av2_sensor_dir"):
        if k in cfg: return cfg[k]
    raise KeyError("Expected one of av2_sensor_root/DATA_DIR in paths yaml")

def _interp_pose_yaw_tx_ty(t_query: np.ndarray, pose_ref):
    t_s, qw, qx, qy, qz, tx, ty = pose_ref
    yaw_all = np.array([_quat_to_yaw(a,b,c,d) for a,b,c,d in zip(qw,qx,qy,qz)], dtype=np.float64)
    yaw_q = np.interp(t_query, t_s, yaw_all)
    tx_q  = np.interp(t_query, t_s, tx)
    ty_q  = np.interp(t_query, t_s, ty)
    return yaw_q, tx_q, ty_q

def _build_time_grid_centered(t_center_s: float, horizon_s: float, step_s: float) -> np.ndarray:
    # inclusive grid with symmetric bounds
    n = int(np.floor(horizon_s / step_s))
    t0 = t_center_s - 0.5 * horizon_s
    return t0 + np.arange(n + 1, dtype=np.float64) * step_s

def _interp_ego_xy(t_grid: np.ndarray, s_ego):
    x = np.interp(t_grid, s_ego.timestamps_s, s_ego.ego_tx_m)
    y = np.interp(t_grid, s_ego.timestamps_s, s_ego.ego_ty_m)
    return x, y

def _softmin_weighted(m_cv: np.ndarray, m_ca: np.ndarray, tau: float, w: float) -> np.ndarray:
    tau = max(tau, 1e-6)
    a = (1.0 - w) * np.exp(-m_cv / tau)
    b = w * np.exp(-m_ca / tau)
    return -tau * np.log(a + b + 1e-12)

def _sustained(mask: np.ndarray, need: int) -> bool:
    run = 0
    for f in mask:
        run = run + 1 if bool(f) else 0
        if run >= need:
            return True
    return False

def _load_pose_feather(paths_yaml: str, log_id: str):
    root = Path(_get_root_from_cfg(yaml.safe_load(open(paths_yaml))))
    df_pose = pd.read_feather(root / log_id / "city_SE3_egovehicle.feather")
    t_s = (df_pose["timestamp_ns"].to_numpy(np.int64).astype(np.float64) * 1e-9)
    qw = df_pose["qw"].to_numpy(np.float64)
    qx = df_pose["qx"].to_numpy(np.float64)
    qy = df_pose["qy"].to_numpy(np.float64)
    qz = df_pose["qz"].to_numpy(np.float64)
    tx = df_pose["tx_m"].to_numpy(np.float64)
    ty = df_pose["ty_m"].to_numpy(np.float64)
    return t_s, qw, qx, qy, qz, tx, ty

# ---------- Data classes ----------
@dataclass
class ActorScore:
    track_uuid: str
    category: str
    p_overlap: float
    dmin_m: float
    penetration_m: float
    t_at_dmin_s: float
    sustained_tight: bool
    a_norm: float
    actor_length_m: float
    actor_width_m: float
    is_static: int
    on_path_like: int
    # NEW features
    bearing_rad: float
    rel_speed_closing_mps: float
    sector_id: int  # 0=front,1=left,2=right,3=back

@dataclass
class _Cfg:
    horizon_s: float
    step_s: float
    top_k: int
    ego_length_m: float
    ego_width_m: float
    sigma_m: float
    use_obd: bool
    rollout_model: str
    accel_weight_a0: float
    softmin_tau_m: float
    sustained_margin_m: float
    sustained_len: int
    unsustained_decay: float
    radius_factor: float
    touch_buffer_m: float
    clamp_speed_mps: float
    clamp_accel_mps2: float
    max_extrap_s: float
    include_categories: Optional[set] = None
    min_track_len_s: float = 0.0
    max_init_dist_m: float = 1e9
    # NEW gates
    front_sector_deg: float = 70.0
    min_static_keep_dist_m: float = 40.0
    static_keep_if_poverlap_ge: float = 0.05
    # NEW: return all candidates (caller will do union-then-fill)
    return_all: bool = False

# ---------- Core ----------
def _score_tracks_at_event(paths_yaml: str, log_id: str, t_center_s: float, cfg: _Cfg) -> List[ActorScore]:
    root = Path(_get_root_from_cfg(yaml.safe_load(open(paths_yaml))))
    ann_fp = root / log_id / "annotations.feather"
    df = pd.read_feather(ann_fp)

    pose_ref = _load_pose_feather(paths_yaml, log_id)
    t_grid = _build_time_grid_centered(t_center_s, cfg.horizon_s, cfg.step_s)

    s_ego = load_ego_series(paths_yaml, log_id)
    ego_x, ego_y = _interp_ego_xy(t_grid, s_ego)
    ego_yaw, _, _ = _interp_pose_yaw_tx_ty(t_grid, pose_ref)

    mid_idx = len(t_grid) // 2
    ego_xc, ego_yc, ego_yaw_c = float(ego_x[mid_idx]), float(ego_y[mid_idx]), float(ego_yaw[mid_idx])
    cos_y, sin_y = cos(-ego_yaw_c), sin(-ego_yaw_c)
    v_ego_center = float(np.interp(t_center_s, s_ego.timestamps_s, s_ego.ego_speed_mps))

    # actor slice near event time (WIDER: ±(H/2 + 4s) => ±6s when H=4)
    t_ns = df["timestamp_ns"].to_numpy(np.int64)
    t_s_all = t_ns.astype(np.float64) * 1e-9
    pad = 4.0 + 0.5 * cfg.horizon_s
    mask_near = (t_s_all >= t_center_s - pad) & (t_s_all <= t_center_s + pad)
    df_near = df[mask_near] if mask_near.any() else df

    STATIC_CATS = {
        "STOP_SIGN","TRAFFIC_SIGN","SIGN","BOLLARD","CONE","BARRIER","POLE","UNKNOWN_STATIC"
    }

    front_rad = np.deg2rad(cfg.front_sector_deg)
    scores: List[ActorScore] = []

    for track_uuid, df_tr in df_near.groupby("track_uuid"):
        if len(df_tr) < 2:
            continue

        L = float(df_tr["length_m"].iloc[0])
        W = float(df_tr["width_m"].iloc[0])
        category = str(df_tr["category"].iloc[0]) if "category" in df_tr.columns else "UNKNOWN"
        if cfg.include_categories and (category not in cfg.include_categories):
            continue

        t_first = float(df_tr["timestamp_ns"].iloc[0]) * 1e-9
        t_last  = float(df_tr["timestamp_ns"].iloc[-1]) * 1e-9
        if (t_last - t_first) < cfg.min_track_len_s:
            continue

        # state estimate at center
        try:
            st = estimate_state_city(df_tr, t_center_s, pose_ref, window_s=0.8)
        except Exception:
            continue
        x0,y0,vx,vy,ax,ay = float(st["x0"]), float(st["y0"]), float(st["vx"]), float(st["vy"]), float(st["ax"]), float(st["ay"])
        a_norm = float(st["a_norm"])

        # guards
        ax = float(np.clip(ax, -cfg.clamp_accel_mps2, cfg.clamp_accel_mps2))
        ay = float(np.clip(ay, -cfg.clamp_accel_mps2, cfg.clamp_accel_mps2))
        vx = float(np.clip(vx, -cfg.clamp_speed_mps,  cfg.clamp_speed_mps))
        vy = float(np.clip(vy, -cfg.clamp_speed_mps,  cfg.clamp_speed_mps))

        dx, dy = (x0 - ego_xc), (y0 - ego_yc)
        x_rel = cos_y*dx - sin_y*dy
        y_rel = sin_y*dx + cos_y*dy
        on_path_like = int((abs(y_rel) <= 3.0) and (x_rel >= -5.0))

        if hypot(dx, dy) > cfg.max_init_dist_m:
            continue

        # rollout CV/CA
        dt = t_grid - t_center_s
        yaw_cv = float(atan2(vy, vx)) if (abs(vx)+abs(vy) > 1e-4) else 0.0
        vel_ca_x = vx + ax*dt
        vel_ca_y = vy + ay*dt
        yaw_ca = np.where((np.hypot(vel_ca_x, vel_ca_y) > 1e-3), np.arctan2(vel_ca_y, vel_ca_x), yaw_cv)

        ax_cv = x0 + vx*dt
        ay_cv = y0 + vy*dt
        ax_ca = x0 + vx*dt + 0.5*ax*(dt**2)
        ay_ca = y0 + vy*dt + 0.5*ay*(dt**2)

        d_cv = np.empty_like(dt)
        d_ca = np.empty_like(dt)
        for i in range(len(t_grid)):
            ego_poly = oriented_box_polygon(ego_x[i], ego_y[i], cfg.ego_length_m, cfg.ego_width_m, float(ego_yaw[i]))
            actor_poly_cv = oriented_box_polygon(ax_cv[i], ay_cv[i], L, W, yaw_cv)
            actor_poly_ca = oriented_box_polygon(ax_ca[i], ay_ca[i], L, W, float(yaw_ca[i]))
            d_cv[i] = box_distance(ego_poly, actor_poly_cv)
            d_ca[i] = box_distance(ego_poly, actor_poly_ca)

        w_acc = float(a_norm / (a_norm + max(1e-6, cfg.accel_weight_a0)))
        m_eff = _softmin_weighted(d_cv, d_ca, tau=cfg.softmin_tau_m, w=w_acc)

        dmin = float(m_eff.min())                 # negative => penetration
        imin = int(np.argmin(m_eff))
        t_at_abs = float(t_grid[imin])
        t_at_rel = t_at_abs - t_center_s

        sustained = _sustained(m_eff < cfg.sustained_margin_m, cfg.sustained_len)

        sigma = max(1e-6, cfg.sigma_m)
        p_raw = float(np.exp(- (max(dmin, 0.0) / sigma) ** 2))
        p = p_raw if sustained else cfg.unsustained_decay * p_raw

        # bearings/sector & closing speed
        bearing = float(atan2(y_rel, x_rel))  # [-pi, pi]
        if abs(bearing) <= front_rad:
            sector_id = 0
        elif bearing > 0:
            sector_id = 1
        else:
            sector_id = 2
        if abs(bearing) > pi - front_rad:
            sector_id = 3

        v_act_forward = float(vx*cos(ego_yaw_c) + vy*sin(ego_yaw_c))
        rel_speed_closing = max(0.0, v_ego_center - v_act_forward)

        is_static_int = int(category in {
            "STOP_SIGN","TRAFFIC_SIGN","SIGN","BOLLARD","CONE","BARRIER","POLE","UNKNOWN_STATIC"
        })
        dist_center = float(hypot(dx, dy))
        # static gate (late, uses p)
        if is_static_int and dist_center > cfg.min_static_keep_dist_m:
            if not (on_path_like or (p >= cfg.static_keep_if_poverlap_ge)):
                continue

        scores.append(ActorScore(
            track_uuid=str(track_uuid),
            category=category,
            p_overlap=p,
            dmin_m=max(dmin, 0.0),
            penetration_m=max(-dmin, 0.0),
            t_at_dmin_s=t_at_rel,
            sustained_tight=bool(sustained),
            a_norm=float(a_norm),
            actor_length_m=L,
            actor_width_m=W,
            is_static=is_static_int,
            on_path_like=on_path_like,
            bearing_rad=bearing,
            rel_speed_closing_mps=rel_speed_closing,
            sector_id=sector_id
        ))

    # sort by proximity/probability
    scores.sort(key=lambda r: (r.p_overlap, -1.0/(1.0 + r.dmin_m), r.penetration_m), reverse=True)
    return scores

def prob_traj_overlap_for_window(
    paths_yaml: str,
    log_id: str,
    t_start_s: float,
    cfg_path: str,
    top_k: Optional[int] = None,
    t_end_s: Optional[float] = None
):
    cfg_raw = yaml.safe_load(open(cfg_path))
    include_cats = cfg_raw.get("filters", {}).get("include_categories", None)
    cfg = _Cfg(
        horizon_s=float(cfg_raw["horizon_s"]),
        step_s=float(cfg_raw["step_s"]),
        top_k=int(cfg_raw.get("top_k", 3)),
        ego_length_m=float(cfg_raw["ego_length_m"]),
        ego_width_m=float(cfg_raw["ego_width_m"]),
        sigma_m=float(cfg_raw.get("sigma_m", 2.0)),
        use_obd=bool(cfg_raw.get("use_obd", True)),
        rollout_model=str(cfg_raw.get("rollout_model", "hybrid")),
        accel_weight_a0=float(cfg_raw.get("accel_weight_a0", 0.8)),
        softmin_tau_m=float(cfg_raw.get("softmin_tau_m", 0.3)),
        sustained_margin_m=float(cfg_raw.get("sustained_margin_m", 0.60)),
        sustained_len=int(cfg_raw.get("sustained_len", 2)),
        unsustained_decay=float(cfg_raw.get("unsustained_decay", 0.6)),
        radius_factor=float(cfg_raw.get("radius_factor", 0.35)),
        touch_buffer_m=float(cfg_raw.get("touch_buffer_m", 0.0)),
        clamp_speed_mps=float(cfg_raw.get("clamp_speed_mps", 100.0)),
        clamp_accel_mps2=float(cfg_raw.get("clamp_accel_mps2", 20.0)),
        max_extrap_s=float(cfg_raw.get("max_extrap_s", 4.0)),
        include_categories=set(include_cats) if include_cats else None,
        min_track_len_s=float(cfg_raw.get("filters", {}).get("min_track_len_s", 0.0)),
        max_init_dist_m=float(cfg_raw.get("filters", {}).get("max_init_dist_m", 200.0)),  # RAISED default
        front_sector_deg=float(cfg_raw.get("front_sector_deg", 70.0)),
        min_static_keep_dist_m=float(cfg_raw.get("min_static_keep_dist_m", 40.0)),
        static_keep_if_poverlap_ge=float(cfg_raw.get("static_keep_if_poverlap_ge", 0.05)),
        return_all=bool(cfg_raw.get("return_all", False)),
    )

    t_center = 0.5 * (float(t_start_s) + float(t_end_s)) if (t_end_s is not None) else float(t_start_s)
    scores = _score_tracks_at_event(paths_yaml, log_id, t_center, cfg)

    if cfg.return_all:
        return scores
    k = int(top_k) if top_k is not None else cfg.top_k
    return scores[:k]
