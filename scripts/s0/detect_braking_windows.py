import argparse, json, os, sys
from pathlib import Path
import numpy as np
import yaml

from signals_to_semantics.io_av2.av2_loader import load_ego_series
from signals_to_semantics.detection.braking_events import detect_brakes, smooth_series, ego_kinematics

def _resolve_thresh(val, arr) -> float:
    """
    Accepts either a float/int or a string like 'pct:10'.
    If 'pct:x', returns np.percentile(arr, x).
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and val.startswith("pct:"):
        p = float(val.split(":", 1)[1])
        return float(np.percentile(arr, p))
    try:
        return float(val)
    except Exception as e:
        raise ValueError(f"Unrecognized threshold value: {val!r}") from e

def _load_brake_cfg(thresh_fp: str):
    """
    Read thresholds from YAML. Allows floats or strings like 'pct:10'.
    Expected keys either flat or nested under 'brake':
      a_min_ms2 / a_min
      j_min_ms3 / j_min
      min_dur_s, max_gap_s, smooth_window, smooth_poly, mode, min_delta_v
    """
    try:
        cfg = yaml.safe_load(open(thresh_fp))
    except Exception:
        cfg = None

    br = (cfg or {}).get("brake", {}) if isinstance(cfg, dict) else (cfg or {})

    a_min_raw = br.get("a_min_ms2", br.get("a_min", -1.0))
    j_min_raw = br.get("j_min_ms3", br.get("j_min", -5.0))

    params = {
        "min_dur_s": float(br.get("min_dur_s", 0.25)),
        "max_gap_s": float(br.get("max_gap_s", 0.20)),
        "smooth_window": int(br.get("smooth_window", 9)),
        "smooth_poly": int(br.get("smooth_poly", 2)),
        "mode": str(br.get("mode", "OR")),
        "min_delta_v": float(br.get("min_delta_v", 0.8)),
    }
    return a_min_raw, j_min_raw, params

def detect_one_log(paths_yaml: str, log_id: str, thresh_fp: str, out_fp: Path):
    """Run detection for a single log and write its JSONL."""
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    s = load_ego_series(paths_yaml, log_id)

    a_min_raw, j_min_raw, P = _load_brake_cfg(thresh_fp)

    # compute smoothed speed + this-log a/j to resolve percentiles
    v = smooth_series(s.ego_speed_mps, window=P["smooth_window"], poly=P["smooth_poly"])
    a_arr, j_arr = ego_kinematics(v, s.timestamps_s)

    # resolve thresholds
    a_min = _resolve_thresh(a_min_raw, a_arr)
    j_min = _resolve_thresh(j_min_raw, j_arr)

    print(f"[detect] log={log_id} a_min={a_min:.3f} m/s^2, j_min={j_min:.3f} m/s^3")

    segs, a, j = detect_brakes(
        speed=s.ego_speed_mps,
        t_s=s.timestamps_s,
        a_min=a_min,
        j_min=j_min,
        min_dur_s=P["min_dur_s"],
        max_gap_s=P["max_gap_s"],
        smooth_window=P["smooth_window"],
        smooth_poly=P["smooth_poly"],
        mode=P["mode"],
        min_delta_v=P["min_delta_v"],
    )

    n = 0
    with open(out_fp, "w") as f:
        for s_idx, e_idx in segs:
            rec = {
                "log_id": log_id,
                "i_start": int(s_idx),
                "i_end": int(e_idx),
                "t_start": float(s.timestamps_s[s_idx]),
                "t_end": float(s.timestamps_s[e_idx]),
                "dur_s": float(s.timestamps_s[e_idx] - s.timestamps_s[s_idx]),
                "a_min_ms2": float(a[s_idx:e_idx+1].min()),
                "j_min_ms3": float(j[s_idx:e_idx+1].min()),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[detect] wrote {out_fp} with {n} braking windows")
    return n

def read_list_file(list_fp: str):
    with open(list_fp) as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml",
                    help="YAML that lets av2_loader resolve data roots")
    # Single-log mode (backward compatible)
    ap.add_argument("--log_id", help="Run detection for a single log_id (mutually exclusive with --log_list)")
    # Multi-log mode
    ap.add_argument("--log_list", help="Text file: one log_id per line (e.g., configs/splits/val50.txt)")
    ap.add_argument("--out_dir", default="artifacts/mini/detect",
                    help="Directory to write per-log JSONLs (will create subdir 'logs/')")
    ap.add_argument("--thresh", default="configs/thresholds.yaml", help="YAML thresholds file")
    ap.add_argument("--concat_out", default=None,
                    help="If set, concatenates all per-log JSONLs into this file after processing")
    args = ap.parse_args()

    # Validate mutually exclusive inputs
    if (args.log_id is None) == (args.log_list is None):
        print("Provide exactly one of --log_id or --log_list", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_logs = 0
    total_windows = 0
    written_files = []

    if args.log_id:
        # Single-log execution
        out_fp = logs_dir / f"brakes_{args.log_id}.jsonl"
        total_windows += detect_one_log(args.paths, args.log_id, args.thresh, out_fp)
        total_logs = 1
        written_files.append(out_fp)
    else:
        # Multi-log execution
        ids = read_list_file(args.log_list)
        for log_id in ids:
            out_fp = logs_dir / f"brakes_{log_id}.jsonl"
            n = detect_one_log(args.paths, log_id, args.thresh, out_fp)
            total_windows += n
            total_logs += 1
            written_files.append(out_fp)

    print(f"[detect] summary logs_processed={total_logs} total_windows={total_windows}")

    # Optional concatenation
    if args.concat_out:
        concat_fp = Path(args.concat_out)
        concat_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(concat_fp, "w") as out:
            for fp in sorted(written_files):
                with open(fp, "r") as f:
                    for line in f:
                        out.write(line)
        print(f"[detect] concatenated -> {concat_fp} ({len(written_files)} files)")

if __name__ == "__main__":
    main()
