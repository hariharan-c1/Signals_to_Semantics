# scripts/s1c_hmm_smooth.py
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def echo(msg): print(msg, flush=True)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def ema_1d(x, alpha=0.4):
    if len(x) == 0: return np.array([], dtype=float)
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def stable_logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))

def forward_backward_log(em_loglik, logA, logpi):
    """
    em_loglik: (T,2) log p(x_t | z_t=s)
    logA: (2,2) log transition probs, rows: prev->curr
    logpi: (2,) log initial prior
    Returns: gamma (T,2) = posterior per state in prob
    """
    T = em_loglik.shape[0]
    if T == 0:
        return np.zeros((0,2), dtype=float)

    log_alpha = np.zeros((T,2), dtype=float)
    log_beta  = np.zeros((T,2), dtype=float)

    # forward init
    log_alpha[0] = logpi + em_loglik[0]
    # forward recursion
    for t in range(1, T):
        for s in range(2):
            log_alpha[t, s] = em_loglik[t, s] + stable_logsumexp(log_alpha[t-1] + logA[:, s])

    # backward init
    log_beta[T-1] = 0.0
    # backward recursion
    for t in range(T-2, -1, -1):
        for s in range(2):
            log_beta[t, s] = stable_logsumexp(logA[s, :] + em_loglik[t+1, :] + log_beta[t+1, :])

    log_gamma = log_alpha + log_beta
    # normalize
    for t in range(T):
        Zt = stable_logsumexp(log_gamma[t])
        log_gamma[t] -= Zt
    gamma = np.exp(log_gamma)
    return gamma  # (T,2)

def viterbi_log(em_loglik, logA, logpi):
    T = em_loglik.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=int)

    log_delta = np.zeros((T,2), dtype=float)
    psi = np.zeros((T,2), dtype=int)

    log_delta[0] = logpi + em_loglik[0]
    psi[0] = 0

    for t in range(1, T):
        for s in range(2):
            seq = log_delta[t-1] + logA[:, s]
            psi[t, s] = int(np.argmax(seq))
            log_delta[t, s] = em_loglik[t, s] + np.max(seq)

    path = np.zeros(T, dtype=int)
    path[T-1] = int(np.argmax(log_delta[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path  # (T,)

def must_exist(path: Path, label: str):
    if not path.exists():
        echo(f"[ERROR] Missing {label}: {path}")
        sys.exit(2)

def load_scores(split_root: Path, verbose: bool):
    """
    Expects:
      {split}/s1a/window_table.parquet
      {split}/s1a/pu_xgb/scores_pu_xgb.jsonl
      {split}/s1b/nnpu/scores_nnpu_{splitname}.jsonl  (or .../scores_nnpu.jsonl)
    Returns merged df with:
      log_id, window_t_start, window_t_end, score_pu_xgb, score_nnpu
    """
    splitname = split_root.name

    wt_path   = split_root / "s1a" / "window_table.parquet"
    xgb_path  = split_root / "s1a" / "pu_xgb" / "scores_pu_xgb.jsonl"
    nnpu_path = split_root / "s1b" / "nnpu" / f"scores_nnpu_{splitname}.jsonl"
    if not nnpu_path.exists():
        nnpu_path = split_root / "s1b" / "nnpu" / "scores_nnpu.jsonl"

    must_exist(wt_path,   "window_table")
    must_exist(xgb_path,  "PU-XGB scores")
    must_exist(nnpu_path, "nnPU scores")

    if verbose:
        echo(f"[INFO] Loading window_table: {wt_path}")
    wt = pd.read_parquet(wt_path)
    if verbose:
        echo(f"       → {wt.shape[0]} rows")

    if verbose:
        echo(f"[INFO] Loading PU-XGB scores: {xgb_path}")
    sc_xgb = pd.read_json(xgb_path, lines=True)
    # standardize column names
    sc_xgb = sc_xgb.rename(columns={"t_start":"window_t_start","t_end":"window_t_end"})
    if "score" in sc_xgb.columns and "score_pu_xgb" not in sc_xgb.columns:
        sc_xgb = sc_xgb.rename(columns={"score":"score_pu_xgb"})
    must_have = {"log_id","window_t_start","window_t_end","score_pu_xgb"}
    if not must_have.issubset(set(sc_xgb.columns)):
        echo(f"[ERROR] PU-XGB file missing required columns {must_have}. Got: {list(sc_xgb.columns)}")
        sys.exit(2)

    if verbose:
        echo(f"[INFO] Loading nnPU scores: {nnpu_path}")
    sc_nn = pd.read_json(nnpu_path, lines=True)
    sc_nn = sc_nn.rename(columns={"t_start":"window_t_start","t_end":"window_t_end"})
    if "score" in sc_nn.columns and "score_nnpu" not in sc_nn.columns:
        sc_nn = sc_nn.rename(columns={"score":"score_nnpu"})
    if "score_nnpu" not in sc_nn.columns:
        # also allow 'score' or 'score_nn'
        for alt in ["score_nn", "score"]:
            if alt in sc_nn.columns:
                sc_nn = sc_nn.rename(columns={alt:"score_nnpu"})
                break
    must_have = {"log_id","window_t_start","window_t_end","score_nnpu"}
    if not must_have.issubset(set(sc_nn.columns)):
        echo(f"[ERROR] nnPU file missing required columns {must_have}. Got: {list(sc_nn.columns)}")
        sys.exit(2)

    # Join
    keep_cols = ["log_id","window_t_start","window_t_end"]
    df = wt[keep_cols].merge(
        sc_xgb[["log_id","window_t_start","window_t_end","score_pu_xgb"]],
        on=keep_cols, how="left"
    ).merge(
        sc_nn[["log_id","window_t_start","window_t_end","score_nnpu"]],
        on=keep_cols, how="left"
    )

    # sanity: any missing scores?
    missing_x = int(df["score_pu_xgb"].isna().sum())
    missing_n = int(df["score_nnpu"].isna().sum())
    if verbose:
        echo(f"[INFO] After merge: rows={df.shape[0]}, missing PU-XGB={missing_x}, missing nnPU={missing_n}")

    # fill conservative defaults if needed (rare)
    df["score_pu_xgb"] = df["score_pu_xgb"].astype(float).fillna(1e-5)
    df["score_nnpu"]   = df["score_nnpu"].astype(float).fillna(1e-5)
    # clip
    df["score_pu_xgb"] = df["score_pu_xgb"].clip(1e-5, 1-1e-5)
    df["score_nnpu"]   = df["score_nnpu"].clip(1e-5, 1-1e-5)

    return df, wt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True,
                    help="artifacts/.../<split> (e.g., .../train650 or .../dev100)")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir; defaults to {split}/s1c/hmm")
    ap.add_argument("--w-xgb", type=float, default=0.5, help="logit weight for PU-XGB")
    ap.add_argument("--w-nnpu", type=float, default=0.5, help="logit weight for nnPU")
    ap.add_argument("--alpha-ema", type=float, default=0.4, help="EMA smoothing alpha")
    ap.add_argument("--pi", type=float, default=None,
                    help="prior P(z=1). If None, estimated from fused scores")
    ap.add_argument("--p-stay-on", type=float, default=0.85,
                    help="stickiness in state=1 (interaction persists)")
    ap.add_argument("--p-stay-off", type=float, default=0.95,
                    help="stickiness in state=0 (quiet persists)")
    ap.add_argument("--verbose", action="store_true", help="print progress")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    if not split_root.exists():
        echo(f"[ERROR] split_root does not exist: {split_root}")
        sys.exit(2)

    out_dir = Path(args.out_dir) if args.out_dir else split_root / "s1c" / "hmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        echo(f"[RUN] HMM smoother")
        echo(f"      split_root = {split_root}")
        echo(f"      out_dir    = {out_dir}")

    df, wt = load_scores(split_root, verbose=args.verbose)
    n_rows = df.shape[0]
    if n_rows == 0:
        echo("[WARN] No rows found after joining scores; exiting cleanly.")
        # still write empty containers
        pd.DataFrame([], columns=["log_id","t_start","t_end","score_pu_xgb","score_nnpu","score_fused","score_ema","post_hmm","viterbi"]).to_json(out_dir/"scores_hmm.jsonl", orient="records", lines=True)
        pd.DataFrame([], columns=["log_id","state","t_start","t_end","duration_s","n_windows"]).to_csv(out_dir/"segments.csv", index=False)
        with open(out_dir/"meta.json", "w") as f:
            json.dump({"note":"empty run"}, f, indent=2)
        echo(f"[DONE] Wrote empty outputs to {out_dir}")
        sys.exit(0)

    # fused emission prob via logit-sum
    l_xgb = logit(df["score_pu_xgb"].values)
    l_nn  = logit(df["score_nnpu"].values)
    l_fused = args.w_xgb * l_xgb + args.w_nnpu * l_nn
    p_fused = sigmoid(l_fused)
    df["score_fused"] = p_fused

    # estimate prior if not provided
    if args.pi is None:
        pi_hat = float(np.clip(np.nanmean(p_fused), 1e-3, 0.999))
    else:
        pi_hat = float(np.clip(args.pi, 1e-3, 0.999))

    # transitions (sticky)
    p11 = np.clip(args.p_stay_on, 1e-3, 0.999)
    p00 = np.clip(args.p_stay_off, 1e-3, 0.999)
    A = np.array([[p00, 1 - p00],
                  [1 - p11, p11]], dtype=float)     # rows: prev(0/1)->curr(0/1)
    logA = np.log(A)
    logpi = np.log(np.array([1 - pi_hat, pi_hat], dtype=float))

    if args.verbose:
        echo(f"[INFO] Prior pi_hat={pi_hat:.3f} | A=\n{A}")

    rows = []
    seg_rows = []

    gcount = 0
    for log_id, g in df.groupby("log_id"):
        gcount += 1
        g = g.sort_values(["window_t_start","window_t_end"]).reset_index(drop=True)
        T = len(g)
        s_fus = g["score_fused"].values
        s_ema = ema_1d(s_fus, alpha=args.alpha_ema)

        eps = 1e-6
        p1 = np.clip(s_fus, eps, 1 - eps)
        p0 = np.clip(1.0 - s_fus, eps, 1 - eps)
        em_loglik = np.stack([np.log(p0), np.log(p1)], axis=1)  # (T,2)

        gamma = forward_backward_log(em_loglik, logA, logpi)   # (T,2)
        post1 = gamma[:, 1] if T > 0 else np.array([], dtype=float)
        vit = viterbi_log(em_loglik, logA, logpi)              # (T,)

        # segments from Viterbi
        if T > 0 and len(vit) == T:
            curr = vit[0]
            seg_start = 0
            for t in range(1, T):
                if vit[t] != curr:
                    t0 = g.loc[seg_start, "window_t_start"]
                    t1 = g.loc[t-1, "window_t_end"]
                    seg_rows.append({
                        "log_id": log_id,
                        "state": int(curr),
                        "t_start": float(t0),
                        "t_end": float(t1),
                        "duration_s": float(t1 - t0),
                        "n_windows": int(t - seg_start),
                    })
                    curr = vit[t]
                    seg_start = t
            # last segment
            t0 = g.loc[seg_start, "window_t_start"]
            t1 = g.loc[T-1, "window_t_end"]
            seg_rows.append({
                "log_id": log_id,
                "state": int(curr),
                "t_start": float(t0),
                "t_end": float(t1),
                "duration_s": float(t1 - t0),
                "n_windows": int(T - seg_start),
            })

        for i in range(T):
            rows.append({
                "log_id": log_id,
                "t_start": float(g.loc[i, "window_t_start"]),
                "t_end":   float(g.loc[i, "window_t_end"]),
                "score_pu_xgb": float(g.loc[i, "score_pu_xgb"]),
                "score_nnpu":   float(g.loc[i, "score_nnpu"]),
                "score_fused":  float(s_fus[i]),
                "score_ema":    float(s_ema[i]),
                "post_hmm":     float(post1[i]) if T > 0 else 0.0,
                "viterbi":      int(vit[i]) if T > 0 else 0,
            })

    out_scores = out_dir / "scores_hmm.jsonl"
    out_segs = out_dir / "segments.csv"
    out_meta = out_dir / "meta.json"

    pd.DataFrame(rows).to_json(out_scores, orient="records", lines=True)
    pd.DataFrame(seg_rows).to_csv(out_segs, index=False)

    meta = {
        "split_root": str(split_root),
        "w_xgb": args.w_xgb, "w_nnpu": args.w_nnpu,
        "alpha_ema": args.alpha_ema,
        "pi_hat": pi_hat,
        "A": A.tolist(),
        "p_stay_on": float(p11), "p_stay_off": float(p00),
        "rows_out": len(rows),
        "segments_out": len(seg_rows),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    echo(f"[DONE] {split_root.name}: wrote {len(rows)} rows → {out_scores}")
    echo(f"[DONE] {split_root.name}: wrote {len(seg_rows)} segments → {out_segs}")
    echo(f"[DONE] meta → {out_meta}")

if __name__ == "__main__":
    main()
