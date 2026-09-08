from typing import Tuple, Dict
import numpy as np
import pandas as pd

def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # Z-up yaw from quaternion
    return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))

def _interp_pose_yaw_tx_ty(t_query: np.ndarray, t_s: np.ndarray, qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, tx: np.ndarray, ty: np.ndarray):
    yaw_all = np.array([_quat_to_yaw(a,b,c,d) for a,b,c,d in zip(qw,qx,qy,qz)], dtype=np.float64)
    yaw_q = np.interp(t_query, t_s, yaw_all)
    tx_q  = np.interp(t_query, t_s, tx)
    ty_q  = np.interp(t_query, t_s, ty)
    return yaw_q, tx_q, ty_q

def _ego_to_city_xy(x_e: np.ndarray, y_e: np.ndarray, yaw: np.ndarray, tx: np.ndarray, ty: np.ndarray):
    cy = np.cos(yaw); sy = np.sin(yaw)
    x_c = cy * x_e - sy * y_e + tx
    y_c = sy * x_e + cy * y_e + ty
    return x_c, y_c

def estimate_state_city(df_tr: pd.DataFrame, t0_s: float, pose_ref, window_s: float=0.8) -> Dict[str,float]:
    """
    Fit x(t), y(t) around t0 in CITY frame. Prefer quadratic (CA); fallback to linear (CV).
    Returns dict {x0,y0,vx,vy,ax,ay,a_norm,n_samples}.
    """
    t_pose_s, qw, qx, qy, qz, tx_pose, ty_pose = pose_ref

    t_all = (df_tr["timestamp_ns"].to_numpy("int64").astype("float64") * 1e-9)
    x_e = df_tr["tx_m"].to_numpy("float64")
    y_e = df_tr["ty_m"].to_numpy("float64")

    if t_all.size < 2:
        raise ValueError("not enough samples")

    # local window around t0
    mask = (t_all >= t0_s - window_s) & (t_all <= t0_s + window_s)
    if mask.sum() < 3:
        # widen once
        mask = (t_all >= t0_s - 1.2) & (t_all <= t0_s + 1.2)
    t_loc = t_all[mask]; xe = x_e[mask]; ye = y_e[mask]
    if t_loc.size < 2:
        # pick nearest 2 for CV fallback
        idx = np.argsort(np.abs(t_all - t0_s))[:2]
        t_loc = t_all[idx]; xe = x_e[idx]; ye = y_e[idx]

    # transform to city
    yaw_q, tx_q, ty_q = _interp_pose_yaw_tx_ty(t_loc, t_pose_s, qw, qx, qy, qz, tx_pose, ty_pose)
    xc, yc = _ego_to_city_xy(xe, ye, yaw_q, tx_q, ty_q)

    dt = t_loc - t0_s
    # design matrices
    X_lin = np.stack([np.ones_like(dt), dt], axis=1)
    X_quad = np.stack([np.ones_like(dt), dt, 0.5*dt**2], axis=1)

    def fit(X, y):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    if t_loc.size >= 3:
        cx = fit(X_quad, xc)  # x0, vx, ax
        cy = fit(X_quad, yc)
        x0, vx, ax = float(cx[0]), float(cx[1]), float(cx[2])
        y0, vy, ay = float(cy[0]), float(cy[1]), float(cy[2])
    else:
        cx = fit(X_lin, xc); cy = fit(X_lin, yc)
        x0, vx = float(cx[0]), float(cx[1]); ax = 0.0
        y0, vy = float(cy[0]), float(cy[1]); ay = 0.0

    a_norm = float(np.hypot(ax, ay))
    return {"x0":x0, "y0":y0, "vx":vx, "vy":vy, "ax":ax, "ay":ay, "a_norm":a_norm, "n_samples": int(t_loc.size)}
