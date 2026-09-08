#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

KEYS = ["log_id", "window_t_start", "window_t_end", "window_center"]

def load_tags(tags_jsonl:str) -> pd.DataFrame:
    """
    Expect per-log normalized tags with fields:
      - log_id (str)
      - has_guest (bool)
      - guest_id (str or None)    # track_uuid if has_guest==True
      - tag (str)                 # scenario taxonomy name
    Extra fields are ignored.
    """
    recs = []
    with open(tags_jsonl, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            o = json.loads(ln)
            recs.append({
                "log_id": o["log_id"],
                "has_guest": bool(o.get("has_guest", False)),
                "guest_id": o.get("guest_id", None),
                "tag": o.get("tag", None),
            })
    df = pd.DataFrame(recs)
    # normalize
    df["guest_id"] = df["guest_id"].astype(object)
    df["tag"] = df["tag"].astype(object)
    return df

def load_rescued_map(windows_jsonl:str):
    """
    Optional: load rescued flags from a detection file that has
    records with keys (log_id, t_start, t_end, rescued: bool).
    Returns dict keyed by (log_id, t_start, t_end) -> rescued(bool)
    """
    rescued = {}
    with open(windows_jsonl, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            o = json.loads(ln)
            log_id = o["log_id"]
            t_start = float(o.get("t_start") or o.get("window_t_start"))
            t_end   = float(o.get("t_end")   or o.get("window_t_end"))
            r = bool(o.get("rescued", False))
            rescued[(log_id, t_start, t_end)] = r
    return rescued

def pick_best_window(df_guest: pd.DataFrame):
    """
    df_guest: rows for one (log_id, guest_id) across windows.
    Choose the best window:
      1) highest sustained_tight (True > False)
      2) then highest p_overlap
      3) then lowest dmin
      4) then earliest window_center
    Returns a single row (as Series)
    """
    df = df_guest.copy()
    df["sustained_rank"] = df["sustained_tight"].astype(float)  # 1.0 or 0.0
    df = df.sort_values(
        by=["sustained_rank","p_overlap","dmin_m","window_center"],
        ascending=[False, True, True, True]
    )
    return df.iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True,
                    help="Per-actor features parquet (42 cols) for a split (e.g., train650_features.parquet)")
    ap.add_argument("--tags-jsonl", required=True,
                    help="Normalized tags jsonl with has_guest, guest_id, tag")
    ap.add_argument("--out-jsonl", required=True,
                    help="Output window_labels.jsonl (positives only)")
    ap.add_argument("--windows-jsonl", default=None,
                    help="(Optional) detection windows jsonl providing rescued flags")
    args = ap.parse_args()

    # Load per-actor features (S0)
    df = pd.read_parquet(args.features_parquet)
    # minimal sanity
    for k in ["log_id","track_uuid","p_overlap","dmin_m","sustained_tight","window_center","window_t_start","window_t_end"]:
        if k not in df.columns:
            raise ValueError(f"Missing required column in features: {k}")

    # We only need the columns to identify guest presence per window:
    # keep memory small
    df_small = df[[
        "log_id","track_uuid","p_overlap","dmin_m","sustained_tight","window_center","window_t_start","window_t_end"
    ]].copy()

    # Deduplicate exact duplicates (rare)
    df_small = df_small.drop_duplicates()

    # Load tags
    tags = load_tags(args.tags_jsonl)

    # Filter to relevant logs present in features
    logs_in_feat = set(df_small["log_id"].unique().tolist())
    tags = tags[tags["log_id"].isin(logs_in_feat)].copy()

    # Optional rescued flags
    rescued_map = {}
    if args.windows_jsonl:
        rescued_map = load_rescued_map(args.windows_jsonl)

    # Build positives
    out_recs = []
    n_total_guests = 0
    n_found_any = 0
    n_selected = 0

    # We allow multiple guests per log (if present)
    pos_tags = tags[(tags["has_guest"]==True) & tags["guest_id"].notna()].copy()

    # fast index by (log_id, guest_id)
    # we’ll filter df_small per (log, guest)
    for _, row in pos_tags.iterrows():
        log_id = row["log_id"]
        guest  = row["guest_id"]
        scen   = row["tag"]
        n_total_guests += 1

        df_g = df_small[(df_small["log_id"]==log_id) & (df_small["track_uuid"]==guest)]
        if len(df_g)==0:
            # guest never appears among top-24 in any window → skip
            continue
        n_found_any += 1

        best = pick_best_window(df_g)
        t_start = float(best["window_t_start"])
        t_end   = float(best["window_t_end"])

        rec = {
            "log_id": log_id,
            "t_start": t_start,
            "t_end": t_end,
            "label": 1,
            "scenario_label": scen
        }
        if rescued_map:
            rec["rescued"] = bool(rescued_map.get((log_id, t_start, t_end), False))

        out_recs.append(rec)
        n_selected += 1

    # Write JSONL
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_recs:
            f.write(json.dumps(r) + "\n")

    # Summary
    print(f"Wrote positives: {args.out_jsonl}")
    print(f"Guests total in tags: {n_total_guests}")
    print(f"Guests found among windows: {n_found_any}")
    print(f"Positive windows selected: {n_selected}")
    if args.windows_jsonl:
        print("Rescued flags attached when available.")

if __name__ == "__main__":
    main()
