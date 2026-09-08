# scripts/s2_train_contrastive.py
import argparse, os, json, time, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- utils ----------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def robust_standardize(df, cols, center, mad, clip=8.0):
    """
    Robust z-score using per-feature center/MAD from pack step.
    - Fill NaNs with center BEFORE scaling (prevents NaNs).
    - Replace zero MAD by 1.0.
    - Clip to [-clip, clip].
    """
    X = df[cols].astype(np.float32).copy()
    cen = pd.Series(center, index=cols).astype(np.float32)
    sca = pd.Series(mad, index=cols).astype(np.float32).replace(0.0, 1.0)
    # fillna per column with its center
    X = X.fillna(cen)
    Z = (X - cen) / sca
    if clip is not None:
        Z = Z.clip(-clip, clip)
    # final safety: replace inf/NaN with 0
    Z = Z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return Z.values.astype(np.float32)

def build_window_key(df, prefix=""):
    """
    Returns "log|t_start|t_end" using whichever time columns exist.
    Accepts: (t_start,t_end) OR (window_t_start,window_t_end) OR (t0,t1)
    """
    if "t_start" in df.columns and "t_end" in df.columns:
        ts = df["t_start"].astype(str); te = df["t_end"].astype(str)
    elif "window_t_start" in df.columns and "window_t_end" in df.columns:
        ts = df["window_t_start"].astype(str); te = df["window_t_end"].astype(str)
    elif "t0" in df.columns and "t1" in df.columns:
        ts = df["t0"].astype(str); te = df["t1"].astype(str)
    else:
        raise KeyError(f"{prefix} missing t_start/t_end (or window_t_start/window_t_end or t0/t1).")
    return df["log_id"].astype(str) + "|" + ts + "|" + te

def load_episodes(split_root: Path):
    """
    Load s1d/episodes.csv if available and return DataFrame with:
      [window_key, episode_id]
    """
    epi_csv = split_root / "s1d" / "episodes.csv"
    if not epi_csv.exists():
        return None
    df = pd.read_csv(epi_csv)
    df["window_key"] = build_window_key(df, prefix="episodes.csv: ")
    if "episode_id" not in df.columns:
        df["episode_id"] = np.arange(len(df), dtype=int)
    return df[["window_key", "episode_id"]]

# --------------- model -----------------
class MLPProj(nn.Module):
    def __init__(self, d_in, hidden=256, proj_dim=128, dropout=0.1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, proj_dim)
    def forward(self, x):
        h = self.f(x)
        z = self.head(h)
        return F.normalize(z, dim=1, eps=1e-6)

def info_nce(z1, z2, temp=0.10):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2B, d), already L2-normalized
    sim = (z @ z.t()) / temp                 # cosine sim matrix
    sim.fill_diagonal_(-1e9)                 # mask self-sim
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    return F.cross_entropy(sim, pos)

# --------------- sampling --------------
def build_pools(df, pos_mode: str):
    pools = {}
    if pos_mode in ("window", "both"):
        win = {}
        for i, k in enumerate(df["window_key"].astype(str).values):
            win.setdefault(k, []).append(i)
        pools["window"] = {k: np.asarray(v, dtype=int) for k, v in win.items()}
    if pos_mode in ("episode", "both") and "episode_id" in df.columns:
        epi = {}
        keys = (df["log_id"].astype(str) + "|" +
                df["track_uuid"].astype(str) + "|" +
                df["episode_id"].astype(int).astype(str)).values
        for i, k in enumerate(keys):
            epi.setdefault(k, []).append(i)
        pools["episode"] = {k: np.asarray(v, dtype=int) for k, v in epi.items()}
    return pools

def sample_pairs(df, pools, pos_mode: str, batch: int):
    N = len(df)
    order = ["episode", "window"] if pos_mode == "both" else [pos_mode]
    idx = np.random.choice(N, size=batch, replace=N < batch)
    jdx = np.empty_like(idx)
    for t, i in enumerate(idx):
        j = None
        for mode in order:
            if mode == "window" and "window" in pools:
                k = str(df.iloc[i]["window_key"])
                pool = pools["window"].get(k, None)
            elif mode == "episode" and "episode" in pools:
                k = (str(df.iloc[i]["log_id"]) + "|" +
                     str(df.iloc[i]["track_uuid"]) + "|" +
                     str(int(df.iloc[i]["episode_id"])) )
                pool = pools["episode"].get(k, None)
            else:
                pool = None
            if pool is not None and len(pool) >= 2:
                j = np.random.choice(pool)
                if j == i and len(pool) > 1:
                    j = np.random.choice(pool)
                if j != i:
                    break
        if j is None or j == i:
            j = np.random.randint(N)
            if j == i and N > 1:
                j = (j + 1) % N
        jdx[t] = j
    return idx, jdx

# ---------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", required=True)

    # data / mode
    ap.add_argument("--pos-mode", choices=["window", "episode", "both"], default="window")

    # model / train hparams (safer defaults for Train650)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--aug-noise", type=float, default=0.03)

    # scheduler
    ap.add_argument("--sched", choices=["cosine", "none"], default="cosine")
    ap.add_argument("--warmup-steps", type=int, default=500)

    # early stop
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=5)

    # misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    split_root = Path(args.split_root)
    cdir = split_root / "s2" / "contrastive"
    meta = json.loads((cdir / "meta.json").read_text())
    feat_cols = meta["feat_cols"]; center = meta["robust_center"]; mad = meta["robust_scale_mad"]

    df_tr = pd.read_parquet(cdir / "contrastive_train.parquet").copy()
    df_va = pd.read_parquet(cdir / "contrastive_val.parquet").copy()

    # ensure window_key exists
    if "window_key" not in df_tr.columns:
        df_tr["window_key"] = build_window_key(df_tr, prefix="train: ")
    if "window_key" not in df_va.columns:
        df_va["window_key"] = build_window_key(df_va, prefix="val: ")

    # merge episodes if available
    episodes = load_episodes(split_root)
    if episodes is not None:
        df_tr = df_tr.merge(episodes, on="window_key", how="left")
        df_va = df_va.merge(episodes, on="window_key", how="left")
        df_tr["episode_id"] = df_tr["episode_id"].fillna(-1).astype(int)
        df_va["episode_id"] = df_va["episode_id"].fillna(-1).astype(int)
    elif args.pos_mode in ("episode", "both"):
        print("[S2-CL][WARN] episodes.csv not found; falling back to window-only positives.")
        args.pos_mode = "window"

    # standardize with SAME robust stats from packer (now NaN-safe)
    Xtr = robust_standardize(df_tr, feat_cols, center, mad)
    Xva = robust_standardize(df_va, feat_cols, center, mad)

    # quick sanity to catch NaNs early
    for name, X in [("train", Xtr), ("val", Xva)]:
        if not np.isfinite(X).all():
            nbad = int((~np.isfinite(X)).sum())
            print(f"[S2-CL][WARN] {name} has {nbad} non-finite entries after standardize (they were zeroed).")

    device = torch.device(args.device)
    model = MLPProj(d_in=Xtr.shape[1], hidden=args.hidden, proj_dim=args.proj_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler w/ warmup
    steps_per_epoch = max(1, math.ceil(len(Xtr) / max(1, args.batch_size)))
    total_steps = args.epochs * steps_per_epoch
    if args.sched == "cosine":
        warmup_steps = max(0, int(args.warmup_steps))
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        sch = None

    pools_tr = build_pools(df_tr, args.pos_mode)
    pools_va = build_pools(df_va, args.pos_mode)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    best_val = float("inf")
    best_path = cdir / "model_contrastive.pt"
    no_improve = 0
    global_step = 0
    saved_once = False

    for ep in range(1, args.epochs + 1):
        model.train(); losses = []; nan_batches = 0
        for _ in range(steps_per_epoch):
            i, j = sample_pairs(df_tr, pools_tr, args.pos_mode, args.batch_size)
            x1 = Xtr_t[i].clone(); x2 = Xtr_t[j].clone()
            if args.aug_noise > 0:
                noise = args.aug_noise
                x1.add_(torch.randn_like(x1) * noise)
                x2.add_(torch.randn_like(x2) * noise)
            z1 = model(x1); z2 = model(x2)
            loss = info_nce(z1, z2, temp=args.temp)

            if torch.isnan(loss):
                nan_batches += 1
                continue  # skip this batch

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if sch is not None:
                global_step += 1
                sch.step()
            losses.append(float(loss.detach().cpu()))
        # quick val
        model.eval()
        with torch.no_grad():
            i, j = sample_pairs(df_va, pools_va, args.pos_mode, min(args.batch_size, len(Xva)))
            z1 = model(Xva_t[i]); z2 = model(Xva_t[j])
            vloss = info_nce(z1, z2, temp=args.temp)
            vloss_val = float(vloss.detach().cpu())
        lr_now = opt.param_groups[0]["lr"]
        tr_mean = float(np.nan) if not len(losses) else float(np.mean(losses))
        print(f"[S2-CL] ep {ep:02d}/{args.epochs}  train={tr_mean:.4f}  val={vloss_val:.4f}  lr={lr_now:.2e}"
              + (f"  skipped_nan_batches={nan_batches}" if nan_batches else ""))

        # save if first finite val or improvement
        if math.isfinite(vloss_val) and (vloss_val < best_val - 1e-4 or not saved_once):
            best_val = vloss_val
            torch.save(model.state_dict(), best_path)
            saved_once = True
            no_improve = 0
        else:
            no_improve += 1
            if args.early_stop and saved_once and no_improve >= args.patience:
                print(f"[S2-CL] Early stop at epoch {ep} (best val={best_val:.4f})")
                break

    # if nothing saved (all NaNs), save last model to keep pipeline moving
    if not saved_once:
        torch.save(model.state_dict(), best_path)
        print("[S2-CL][WARN] No finite val loss observed; saved last model state anyway.")

    # reload best & export embeddings
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def export(df, X, out_path):
        rows = len(df)
        Z_blocks = []
        with torch.no_grad():
            for k in range(0, rows, 4096):
                xb = torch.from_numpy(X[k:k+4096]).to(device)
                zb = model(xb).cpu().numpy().astype(np.float32)
                Z_blocks.append(zb)
        Z = np.vstack(Z_blocks)
        norms = np.linalg.norm(Z, axis=1)
        tiny = float((norms < 1e-5).mean() * 100)
        print(f"[S2-CL] export → {out_path} | rows={rows} | mean||z||={norms.mean():.4f} | tiny%={tiny:.2f}")
        ids = df[["log_id","window_t_start","window_t_end","track_uuid","category","window_key"]].reset_index(drop=True)
        emb_cols = [f"emb_{j:03d}" for j in range(Z.shape[1])]
        emb_df = pd.DataFrame(Z, columns=emb_cols)
        out_df = pd.concat([ids, emb_df], axis=1)
        out_df.to_parquet(out_path, index=False)

    cdir.mkdir(parents=True, exist_ok=True)
    export(df_tr, Xtr, cdir / "embeddings_train.parquet")
    export(df_va, Xva, cdir / "embeddings_val.parquet")

    (cdir / "meta_trained.json").write_text(json.dumps({
        **meta,
        "proj_dim": args.proj_dim, "hidden": args.hidden, "dropout": args.dropout,
        "temp": args.temp, "epochs": args.epochs, "batch_size": args.batch_size,
        "aug_noise": args.aug_noise, "weight_decay": args.weight_decay,
        "lr": args.lr, "sched": args.sched, "warmup_steps": args.warmup_steps,
        "pos_mode": args.pos_mode, "best_val": best_val, "seed": args.seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }, indent=2))

if __name__ == "__main__":
    main()
