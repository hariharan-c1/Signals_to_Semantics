from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def _get_root_from_cfg(cfg: dict):
    for k in ("av2_sensor_root", "DATA_DIR", "av2_sensor_dir"):
        if k in cfg: return cfg[k]
    raise KeyError("Expected one of av2_sensor_root/DATA_DIR in paths yaml")


TARGET_HZ = 20.0  # resample rate for stable derivatives (dt=0.05s)

@dataclass
class EgoSeries:
    log_id: str
    timestamps_s: np.ndarray   # [T], strictly increasing, uniform @ TARGET_HZ
    ego_tx_m: np.ndarray       # [T], city frame X
    ego_ty_m: np.ndarray       # [T], city frame Y
    ego_speed_mps: np.ndarray  # [T]

def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.empty_like(x, dtype=np.float64)
    if len(x) >= 3:
        dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
    elif len(x) == 2:
        dx[0] = dx[1] = (x[1] - x[0]) / dt
    else:
        dx[...] = 0.0
    return dx

def _sort_and_unique(t_s: np.ndarray, *arrays: np.ndarray):
    order = np.argsort(t_s, kind="mergesort")
    t_sorted = t_s[order]
    arrays_sorted = [arr[order] for arr in arrays]
    if len(t_sorted) == 0:
        return t_sorted, *arrays_sorted
    keep = np.ones_like(t_sorted, dtype=bool)
    keep[1:] = t_sorted[1:] > t_sorted[:-1]
    t_unique = t_sorted[keep]
    arrays_unique = [arr[keep] for arr in arrays_sorted]
    return (t_unique, *arrays_unique)

def _resample_to_target_hz(t_s: np.ndarray, tx: np.ndarray, ty: np.ndarray, hz: float):
    # sort/dedup all together
    t_s, tx, ty = _sort_and_unique(t_s, tx, ty)
    if len(t_s) < 3:
        dt = 1.0 / hz
        return t_s.astype(np.float64), tx.astype(np.float64), ty.astype(np.float64), dt
    dt = 1.0 / hz
    t0, t1 = float(t_s[0]), float(t_s[-1])
    n = int(np.floor((t1 - t0) / dt)) + 1
    t_u = t0 + np.arange(n, dtype=np.float64) * dt
    # linear interpolation onto target grid
    tx_u = np.interp(t_u, t_s, tx)
    ty_u = np.interp(t_u, t_s, ty)
    return t_u, tx_u, ty_u, dt

def load_ego_series(paths_yaml: str, log_id: str) -> EgoSeries:
    """
    Loads ego pose for a log, resamples to a uniform timeline (TARGET_HZ),
    and returns x,y + speed on that grid.
    """
    cfg = yaml.safe_load(open(paths_yaml))
    root = Path(_get_root_from_cfg(cfg))
    df = pd.read_feather(root / log_id / "city_SE3_egovehicle.feather")

    t_ns = df["timestamp_ns"].to_numpy(np.int64)
    t_s_raw = t_ns.astype(np.float64) * 1e-9
    tx = df["tx_m"].to_numpy(np.float64)
    ty = df["ty_m"].to_numpy(np.float64)

    # resample to TARGET_HZ for robust derivatives
    t_u, tx_u, ty_u, dt = _resample_to_target_hz(t_s_raw, tx, ty, TARGET_HZ)

    vx = _central_diff(tx_u, dt)
    vy = _central_diff(ty_u, dt)
    speed = np.hypot(vx, vy)

    return EgoSeries(
        log_id=log_id,
        timestamps_s=t_u,
        ego_tx_m=tx_u,
        ego_ty_m=ty_u,
        ego_speed_mps=speed,
    )
