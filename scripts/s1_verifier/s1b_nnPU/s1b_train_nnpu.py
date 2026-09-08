# scripts/s1b_train_nnpu.py
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------- Utils ---------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_parquet(path):
    return pd.read_parquet(path)

def _normalize_key_cols(df: pd.DataFrame):
    """
    Ensure df has ['log_id','window_t_start','window_t_end'].
    Accepts t_start/t_end or window_t_start/window_t_end (and start/end as last fallback).
    """
    df = df.copy()
    if "log_id" not in df.columns:
        raise KeyError("Missing 'log_id' in dataframe.")

    if ("window_t_start" in df.columns) and ("window_t_end" in df.columns):
        return df

    if ("t_start" in df.columns) and ("t_end" in df.columns):
        return df.rename(columns={"t_start":"window_t_start", "t_end":"window_t_end"})

    if ("start" in df.columns) and ("end" in df.columns):
        return df.rename(columns={"start":"window_t_start", "end":"window_t_end"})

    raise KeyError("Could not find time columns: expected t_start/t_end or window_t_start/window_t_end.")

def select_feature_columns(df: pd.DataFrame):
    drop_cols = {
        "log_id", "window_t_start", "window_t_end", "window_center",
        "label", "weight"
    }
    num_cols = [c for c in df.columns
                if (np.issubdtype(df[c].dtype, np.number) and c not in drop_cols)]
    return num_cols

def load_train_tables(cons_parquet, window_table_parquet):
    tgt = read_parquet(cons_parquet)
    wt  = read_parquet(window_table_parquet)

    tgt = _normalize_key_cols(tgt)
    wt  = _normalize_key_cols(wt)

    # sanity: need label & weight in targets
    for c in ["label","weight"]:
        if c not in tgt.columns:
            raise KeyError(f"Consolidated targets parquet missing '{c}' column.")

    key = ["log_id","window_t_start","window_t_end"]
    df = pd.merge(wt, tgt[key + ["label","weight"]], on=key, how="left")
    return df

def simple_impute_and_standardize(df_feat: pd.DataFrame):
    med = df_feat.median(axis=0)
    df_imp = df_feat.fillna(med)
    mu = df_imp.mean(axis=0)
    sig = df_imp.std(axis=0)
    sig = sig.replace(0, 1.0)
    df_z = (df_imp - mu) / sig
    return df_z.values.astype(np.float32), med, mu, sig

def prepare_puxgb_scores(scores_jsonl_path):
    """
    Load PU-XGB scores; normalize columns; deduplicate per (log_id,window_t_start,window_t_end) using mean.
    Return a compact df: [log_id, window_t_start, window_t_end, score]
    """
    sc = pd.read_json(scores_jsonl_path, lines=True)
    # normalize time cols
    if "window_t_start" not in sc.columns and "t_start" in sc.columns:
        sc = sc.rename(columns={"t_start":"window_t_start"})
    if "window_t_end" not in sc.columns and "t_end" in sc.columns:
        sc = sc.rename(columns={"t_end":"window_t_end"})
    # normalize score col
    if "score" not in sc.columns and "score_pu_xgb" in sc.columns:
        sc = sc.rename(columns={"score_pu_xgb":"score"})

    needed = ["log_id","window_t_start","window_t_end","score"]
    missing = [c for c in ["log_id","window_t_start","window_t_end","score"] if c not in sc.columns]
    if missing:
        raise KeyError(f"PU-XGB scores missing columns: {missing}")

    sc = sc[needed].copy()
    # dedup safeguard
    sc = sc.groupby(["log_id","window_t_start","window_t_end"], as_index=False)["score"].mean()
    return sc

def estimate_prior_elkan_noto(scores_df, train_labels_df):
    """
    Align scores with training labels on keys and compute Elkan–Noto prior:
      c_hat = E[s | labeled positive]
      pi_hat ≈ E[s_all] / c_hat
    """
    scores_df = _normalize_key_cols(scores_df)
    tl = _normalize_key_cols(train_labels_df)

    key = ["log_id","window_t_start","window_t_end"]
    # keep only keys + label for alignment
    tl_small = tl[key + ["label"]].copy()

    sj = pd.merge(scores_df, tl_small, on=key, how="left")  # aligned table
    s_all = sj["score"].fillna(0.0).values.astype(np.float64)
    s_pos = sj.loc[sj["label"] == 1, "score"].fillna(0.0).values.astype(np.float64)

    # compute c_hat and pi_hat robustly
    c_hat = float(np.clip(s_pos.mean() if s_pos.size else 0.5, 0.01, 0.999))
    pi_hat = float(np.clip(s_all.mean() / c_hat if c_hat > 0 else 0.5, 0.01, 0.95))
    return pi_hat, c_hat, float(s_all.mean())

# ---------------- Model ---------------- #

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cons", required=True,
                    help="Train650 consolidated targets parquet (with label, weight)")
    ap.add_argument("--train-window-table", required=True,
                    help="Train650 window_table.parquet used for features")
    ap.add_argument("--puxgb-scores-train", required=True,
                    help="Train650 PU-XGB scores jsonl (for prior estimation)")
    ap.add_argument("--dev-window-table", required=True,
                    help="Dev100 window_table.parquet to score after training (no Val50 here)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ===== Load train features + labels (Train650) =====
    df_train = load_train_tables(args.train_cons, args.train_window_table)
    feat_cols = select_feature_columns(df_train)
    Xz, med, mu, sig = simple_impute_and_standardize(df_train[feat_cols])

    y_is_pos = (df_train["label"] == 1).fillna(False).values
    y_is_unl = (~y_is_pos)
    w = df_train["weight"].fillna(0.0).values.astype(np.float32)

    # ===== Estimate class prior from PU-XGB (aligned to labels by keys) =====
    puxgb_sc = prepare_puxgb_scores(args.puxgb_scores_train)
    # build a labels-only df with keys to align
    labels_for_align = df_train[["log_id","window_t_start","window_t_end","label"]].copy()
    pi_hat, c_hat, sbar = estimate_prior_elkan_noto(puxgb_sc, labels_for_align)

    meta = {
        "seed": args.seed,
        "pi_hat": pi_hat,
        "c_hat": c_hat,
        "E_s_all": sbar,
        "feat_cols": feat_cols,
        "imputer": {"median": {k: float(v) for k,v in med.items()}},
        "scaler": {"mean": {k: float(v) for k,v in mu.items()},
                   "std":  {k: float(v) for k,v in sig.items()}},
        "model": {"hidden": args.hidden, "dropout": args.dropout},
        "train_rows": int(len(df_train)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    device = torch.device(args.device)
    X = torch.from_numpy(Xz).to(device)
    y_pos_mask = torch.from_numpy(y_is_pos.astype(np.bool_)).to(device)
    y_unl_mask = torch.from_numpy(y_is_unl.astype(np.bool_)).to(device)
    w_pos = torch.from_numpy(w).to(device)

    # 1:2 pos:unl sampling per epoch (approx)
    idx_pos = np.where(y_is_pos)[0]
    idx_unl = np.where(y_is_unl)[0]
    pos_ct = max(len(idx_pos), 1)
    unl_ct = max(2*len(idx_pos), 1)
    per_epoch_idx = np.concatenate([
        np.random.choice(idx_pos, size=pos_ct, replace=True),
        np.random.choice(idx_unl, size=unl_ct, replace=True)
    ])

    model = MLP(d_in=X.shape[1], d_hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ===== Train nnPU (Kiryo et al.) with positive weights =====
    model.train()
    for ep in range(args.epochs):
        np.random.shuffle(per_epoch_idx)
        total, steps = 0.0, 0
        for i in range(0, len(per_epoch_idx), args.batch_size):
            batch_idx = per_epoch_idx[i:i+args.batch_size]
            xb = X[batch_idx]
            ypb = y_pos_mask[batch_idx]
            yub = y_unl_mask[batch_idx]
            wb  = w_pos[batch_idx]

            pred = model(xb)
            eps = 1e-6
            pred_c = torch.clamp(pred, eps, 1-eps)
            loss_pos = -torch.log(pred_c)
            loss_neg = -torch.log(1.0 - pred_c)

            if ypb.any():
                wn = wb[ypb]
                wn = wn / (wn.mean() + 1e-8)  # normalize weights to mean 1
                Rp_pos = (loss_pos[ypb] * wn).mean()
                Rp_neg = (loss_neg[ypb] * wn).mean()
            else:
                Rp_pos = torch.tensor(0.0, device=device)
                Rp_neg = torch.tensor(0.0, device=device)

            if yub.any():
                Ru_neg = loss_neg[yub].mean()
            else:
                Ru_neg = torch.tensor(0.0, device=device)

            R_neg = torch.clamp(Ru_neg - pi_hat * Rp_neg, min=0.0)
            risk = pi_hat * Rp_pos + R_neg

            opt.zero_grad()
            risk.backward()
            opt.step()
            total += float(risk.detach().cpu().item())
            steps += 1
        print(f"[ep {ep+1:02d}/{args.epochs}] nnPU risk={total/max(1,steps):.4f}")

    # ===== Save =====
    torch.save(model.state_dict(), os.path.join(args.out_dir, "nnpu_model.pt"))
    with open(os.path.join(args.out_dir, "nnpu_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model + meta to {args.out_dir} (pi_hat={pi_hat:.3f}, c_hat={c_hat:.3f})")

    # ===== Scoring helper (Train650 + Dev100 only) =====
    def score_split(window_table_path, out_jsonl_path):
        dfx = read_parquet(window_table_path)
        dfx = _normalize_key_cols(dfx)

        Xcols = [c for c in meta["feat_cols"] if c in dfx.columns]
        med_s = pd.Series(meta["imputer"]["median"])
        mu_s  = pd.Series(meta["scaler"]["mean"])
        sd_s  = pd.Series(meta["scaler"]["std"])
        for k in Xcols:
            if k not in med_s: med_s[k] = 0.0
            if k not in mu_s:  mu_s[k]  = 0.0
            if k not in sd_s:  sd_s[k]  = 1.0
        Ximp = dfx[Xcols].fillna(med_s)
        Xz = ((Ximp - mu_s) / sd_s.replace(0,1.0)).values.astype(np.float32)

        Xt = torch.from_numpy(Xz).to(device)
        model.eval()
        with torch.no_grad():
            scores = model(Xt).detach().cpu().numpy().astype(float)

        out = pd.DataFrame({
            "log_id": dfx["log_id"],
            "t_start": dfx["window_t_start"],
            "t_end": dfx["window_t_end"],
            "score_nnpu": scores
        })
        out.to_json(out_jsonl_path, orient="records", lines=True)
        print(f"Scored {len(out)} → {out_jsonl_path}")

    score_split(args.train_window_table, os.path.join(args.out_dir, "scores_nnpu_train650.jsonl"))
    score_split(args.dev_window_table,   os.path.join(args.out_dir, "scores_nnpu_dev100.jsonl"))

if __name__ == "__main__":
    main()
