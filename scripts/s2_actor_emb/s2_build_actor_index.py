# scripts/s2_build_actor_index.py
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

def read_parquet(p): return pd.read_parquet(p)
def read_jsonl(p):   return pd.read_json(p, lines=True)

def _coerce_times(df):
    """Coerce time columns to float (no rounding) and rename t_* to window_t_* if present."""
    for c in ("t_start","t_end","window_t_start","window_t_end"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    ren = {}
    if "t_start" in df.columns: ren["t_start"] = "window_t_start"
    if "t_end"   in df.columns: ren["t_end"]   = "window_t_end"
    if ren: df = df.rename(columns=ren)
    return df

def _dedup_scores(df):
    key = ["log_id","window_t_start","window_t_end"]
    if not set(key).issubset(df.columns):
        return df
    cols = [c for c in df.columns if c not in key]
    if not cols:
        return df.drop_duplicates(key)
    agg = {}
    for c in cols:
        agg[c] = "max"  # conservative for posteriors and viterbi
    return df.groupby(key, as_index=False).agg(agg)

def _load_s1d(s1d_path):
    if not s1d_path.exists(): return None
    df = read_parquet(s1d_path)
    df = _coerce_times(df)
    keep = ["log_id","window_t_start","window_t_end"]
    extra = [c for c in ("post_hmm","score_final","viterbi") if c in df.columns]
    df = df[keep + extra].copy()
    if "score_final" not in df.columns and "post_hmm" in df.columns:
        df["score_final"] = df["post_hmm"]
    # dev100 sometimes didn't have viterbi in S1D; keep if present
    return _dedup_scores(df)

def _load_s1c(s1c_path):
    if not s1c_path.exists(): return None
    df = read_jsonl(s1c_path)
    df = _coerce_times(df)
    keep = ["log_id","window_t_start","window_t_end"]
    extra = [c for c in ("post_hmm","viterbi","score_fused","score_ema","score_final") if c in df.columns]
    df = df[keep + extra].copy()
    if "score_final" not in df.columns and "post_hmm" in df.columns:
        df["score_final"] = df["post_hmm"]
    if "viterbi" in df.columns:
        df["viterbi"] = df["viterbi"].fillna(0).astype(int)
    return _dedup_scores(df)

def load_scores_dev100_style(split_root: Path):
    """
    dev100 behavior:
      - Prefer S1D (final_scores.parquet).
      - If S1C exists, merge in `viterbi` (so we can do posterior OR viterbi).
      - If S1D missing, fallback to S1C.
    """
    s1d = split_root / "s1d" / "final_scores.parquet"
    s1c = split_root / "s1c" / "hmm" / "scores_hmm.jsonl"

    d = _load_s1d(s1d)
    c = _load_s1c(s1c)

    if d is not None:
        if c is not None and "viterbi" in c.columns:
            key = ["log_id","window_t_start","window_t_end"]
            add = c[key + ["viterbi"]].copy()
            out = pd.merge(d, add, on=key, how="left")
            out["viterbi"] = out.get("viterbi", 0).fillna(0).astype(int)
            return _dedup_scores(out), "s1d+viterbi_from_s1c"
        else:
            # ensure viterbi exists (0) for OR mask to work uniformly
            if "viterbi" not in d.columns:
                d["viterbi"] = 0
            d["viterbi"] = d["viterbi"].fillna(0).astype(int)
            return _dedup_scores(d), "s1d"
    # fallback to S1C
    if c is None:
        raise FileNotFoundError("No scores found (neither S1D nor S1C).")
    # ensure viterbi exists (S1C path usually has it; if not, set 0)
    if "viterbi" not in c.columns:
        c["viterbi"] = 0
    c["viterbi"] = c["viterbi"].fillna(0).astype(int)
    return _dedup_scores(c), "s1c"

def load_features(split_root: Path):
    # dev100 layout: <split-root>/features/<split>_features.parquet
    split_name = split_root.name
    feats_path = split_root / "features" / f"{split_name}_features.parquet"
    if not feats_path.exists():
        # fallback: first parquet under features
        cands = sorted(list((split_root / "features").rglob("*.parquet")))
        if not cands:
            raise FileNotFoundError(f"No features parquet found under {split_root/'features'}")
        feats_path = cands[0]
    df = read_parquet(feats_path)
    for k in ("log_id","window_t_start","window_t_end","track_uuid"):
        if k not in df.columns:
            raise KeyError(f"Features missing key column: {k}")
    # dev100: no rounding, just coerce numeric
    for c in ("window_t_start","window_t_end"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, str(feats_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True,
                    help="artifacts/.../dev100 or .../train650")
    ap.add_argument("--out-dir", default=None)
    # distance pruning (dev100)
    ap.add_argument("--r-max", type=float, default=60.0)
    # keep behavior identical to dev100: no actor cap
    ap.add_argument("--max-actors", type=int, default=1000000)

    # Mask config (dev100: posterior OR viterbi, thr=0.5)
    ap.add_argument("--post-thresh", type=float, default=0.5)

    # --- NEW ARGUMENT ---
    ap.add_argument("--no-mask", action="store_true",
                    help="Disable the posterior/viterbi mask and keep all windows.")

    args = ap.parse_args()

    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir) if args.out_dir else (split_root / "s2_no_filter" / "index")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load scores (dev100 behavior) -----
    scores, used_source = load_scores_dev100_style(split_root)

    # Coerce time columns (no rounding)
    scores = _coerce_times(scores)

    # --- MODIFIED MASKING LOGIC ---
    if not args.no_mask:
        # Build OR mask: (post_hmm >= thr) OR (viterbi == 1)
        # Use post_hmm if present; else score_final.
        score_col = "post_hmm" if "post_hmm" in scores.columns else ("score_final" if "score_final" in scores.columns else None)
        if score_col is None:
            raise KeyError("Neither 'post_hmm' nor 'score_final' available in scores.")

        postmask = (scores[score_col] >= float(args.post_thresh))
        vitmask  = (scores.get("viterbi", 0).fillna(0).astype(int) == 1)
        keep_mask = (postmask | vitmask)

        kept = scores[keep_mask].copy()

        # For metadata
        mask_mode = "both"
        mask_logic = "union"

    else:
        # --no-mask is specified, keep all scores
        kept = scores.copy()

        # For metadata
        score_col = None # No score col was used for selection
        mask_mode = "none"
        mask_logic = "none"
    # --- END MODIFIED MASKING LOGIC ---

    # ----- Load features (dev100 layout) -----
    feats, feats_path = load_features(split_root)

    key = ["log_id","window_t_start","window_t_end"]
    merged = pd.merge(
        feats, kept[key + [c for c in kept.columns if c not in key]],
        on=key, how="inner", validate="many_to_one"
    )

    # distance pruning only
    if "r_rel_m" in merged.columns:
        merged = merged[merged["r_rel_m"] <= float(args.r_max)].copy()

    # optional per-window cap (disabled by default)
    if args.max_actors < 1000000:
        merged = merged.sort_values(key + ["r_rel_m","ttc_s","p_overlap"])
        merged["_rank"] = merged.groupby(key).cumcount()
        merged = merged[merged["_rank"] < int(args.max_actors)].drop(columns=["_rank"])

    # deterministic order
    merged = merged.sort_values(key + ["r_rel_m","ttc_s","p_overlap"]).reset_index(drop=True)

    # Ensure viterbi exists and is last (matching the released dev100 contract).
    if "viterbi" not in merged.columns:
        merged["viterbi"] = 0
    else:
        merged["viterbi"] = merged["viterbi"].fillna(0).astype(int)

    # Reorder columns so viterbi is last
    cols = list(merged.columns)
    if "viterbi" in cols:
        cols.remove("viterbi")
        cols.append("viterbi")
        merged = merged[cols]

    out_parquet = out_dir / "actor_index.parquet"
    merged.to_parquet(out_parquet, index=False)

    # --- MODIFIED METADATA ---
    meta = {
        "split_root": str(split_root),
        "features_parquet": str(feats_path),
        "scores_source": used_source,
        "mask": {
            "mode": mask_mode,    # <-- MODIFIED
            "logic": mask_logic,   # <-- MODIFIED
            "thresh": float(args.post_thresh) if not args.no_mask else None, # <-- MODIFIED
            "prefer_score": score_col # <-- MODIFIED
        },
        "r_max": float(args.r_max),
        "max_actors": int(args.max_actors),
        "rows_scores": int(len(scores)),
        "rows_scores_kept": int(len(kept)),
        "rows_features": int(len(feats)),
        "rows_actor_index": int(len(merged)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[S2-INDEX] wrote {out_parquet} | rows={len(merged)}, cols={len(merged.columns)}")

    # --- MODIFIED PRINT STATEMENT ---
    if not args.no_mask:
        print(f"[S2-INDEX] scores source: {used_source} | mask=posterior OR viterbi | thr={args.post_thresh}")
    else:
        print(f"[S2-INDEX] scores source: {used_source} | mask=DISABLED (all windows kept)")

    print(f"[S2-INDEX] features: {feats_path}")

if __name__ == "__main__":
    main()
