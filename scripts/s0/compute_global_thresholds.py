# scripts/compute_global_percentiles.py
import argparse, json, sys
from pathlib import Path
import numpy as np

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{ln} JSON decode error: {e}", file=sys.stderr)

def pick_float(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default

def main(input_path: str, out_path: str):
    p = Path(input_path)
    files = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        print(f"[ERR] Input not found: {p}", file=sys.stderr); sys.exit(2)

    a_vals, j_vals = [], []
    uniq = set()
    for fp in files:
        for obj in iter_jsonl(fp):
            # Deduplicate by (log_id, i_start, i_end) when available
            key = (obj.get("log_id"), obj.get("i_start"), obj.get("i_end"))
            if key in uniq:
                continue
            uniq.add(key)

            a = pick_float(obj, "a_min_ms2", "a_min", default=None)
            j = pick_float(obj, "j_min_ms3", "j_min", default=None)
            if a is not None: a_vals.append(a)
            if j is not None: j_vals.append(j)

    if not a_vals or not j_vals:
        print("[ERR] No a_min/j_min values found.", file=sys.stderr); sys.exit(3)

    a = np.array(a_vals, dtype=float)
    j = np.array(j_vals, dtype=float)

    # More negative = stronger braking; we still report standard percentiles of the raw values.
    pct_list = [1, 5, 10, 12.5, 15, 20, 25, 50]
    stats = {
        "count": int(len(a_vals)),
        "a_min": {f"p{p}": float(np.percentile(a, p)) for p in pct_list},
        "j_min": {f"p{p}": float(np.percentile(j, p)) for p in pct_list},
        "notes": "Percentiles over per-window minima; use OR mode with Δv filter."
    }

    print(json.dumps(stats, indent=2))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="JSONL file OR directory of JSONLs")
    ap.add_argument("--output_path", default="", help="Write stats JSON here (optional)")
    args = ap.parse_args()
    main(args.input_path, args.output_path)
