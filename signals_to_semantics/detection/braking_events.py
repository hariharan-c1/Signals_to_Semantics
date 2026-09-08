import numpy as np
from typing import List, Tuple
from scipy.signal import savgol_filter

def smooth_series(x: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    n = len(x)
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5 or w > n:
        return x.astype(float, copy=True)
    return savgol_filter(x, window_length=w, polyorder=min(poly, w - 1), mode="interp")

def ego_kinematics(speed: np.ndarray, t_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.gradient(speed, t_s)   # m/s^2
    j = np.gradient(a,     t_s)   # m/s^3
    return a, j

def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    starts, ends, on = [], [], False
    for i, m in enumerate(mask):
        if m and not on:
            starts.append(i); on = True
        if on and (not m or i == len(mask) - 1):
            ends.append(i); on = False
    return list(zip(starts, ends))

def _merge_and_filter(segs: List[Tuple[int, int]], t_s: np.ndarray,
                      min_dur_s: float, max_gap_s: float) -> List[Tuple[int, int]]:
    if not segs:
        return []
    merged: List[List[int]] = [list(segs[0])]
    for s, e in segs[1:]:
        ls, le = merged[-1]
        gap = t_s[s] - t_s[le]
        if gap <= max_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out: List[Tuple[int, int]] = []
    for s, e in merged:
        if t_s[e] - t_s[s] >= min_dur_s:
            out.append((s, e))
    return out

def detect_brakes(
    speed: np.ndarray,
    t_s: np.ndarray,
    # thresholds can be float (fixed) or strings like "pct:10" meaning 10th percentile
    a_min: float | str = -1.0,
    j_min: float | str = -5.0,
    min_dur_s: float = 0.25,
    max_gap_s: float = 0.15,
    smooth_window: int = 9,
    smooth_poly: int = 2,
    mode: str = "SOFT_AND",     # {"AND","OR","SOFT_AND"}
    frac_required: float = 0.3, # fraction of samples inside seg that must satisfy both thresholds in SOFT_AND
    min_delta_v: float = 0.5,   # minimum speed drop (m/s) across the seg
):
    """
    Return (segments, a, j). Segments pass:
      - windowing/merging,
      - mode logic on thresholds (fixed or percentile),
      - and min speed drop Δv.
    """
    v = smooth_series(speed, window=smooth_window, poly=smooth_poly)
    a, j = ego_kinematics(v, t_s)

    # resolve percentile thresholds if requested
    def _resolve(th, arr):
        if isinstance(th, str) and th.startswith("pct:"):
            p = float(th.split(":", 1)[1])
            return float(np.percentile(arr, p))
        return float(th)

    a_thr = _resolve(a_min, a)
    j_thr = _resolve(j_min, j)

    mask_and = (a <= a_thr) & (j <= j_thr)
    mask_or  = (a <= a_thr) | (j <= j_thr)
    raw_mask = mask_and if mode == "AND" else (mask_or if mode == "OR" else mask_or)

    segs = _segments_from_mask(raw_mask)
    segs = _merge_and_filter(segs, t_s, min_dur_s=min_dur_s, max_gap_s=max_gap_s)

    kept: List[Tuple[int,int]] = []
    for s_idx, e_idx in segs:
        if mode == "SOFT_AND":
            seg_len = max(1, e_idx - s_idx + 1)
            frac = float(np.count_nonzero(mask_and[s_idx:e_idx+1])) / seg_len
            if frac < frac_required:
                continue
        dv = float(v[s_idx] - v[e_idx])
        if dv < min_delta_v:
            continue
        kept.append((s_idx, e_idx))

    return kept, a, j
