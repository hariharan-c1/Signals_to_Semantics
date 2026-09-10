# scripts/s2_infer_contrastive_val.py
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- utilities (same math as training) ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def robust_standardize(df, cols, center, mad, clip=8.0):
    """
    Robust z-score using per-feature center/MAD (from TRAIN650 pack meta).
    - fillna with center
    - divide by MAD (0 -> 1)
    - clip to [-clip, clip]
    - zero any remaining inf/NaN
    """
    X = df[cols].astype(np.float32).copy()
    cen = pd.Series(center, index=cols).astype(np.float32)
    sca = pd.Series(mad, index=cols).astype(np.float32).replace(0.0, 1.0)
    X = X.fillna(cen)
    Z = (X - cen) / sca
    if clip is not None:
        Z = Z.clip(-clip, clip)
    Z = Z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return Z.values.astype(np.float32)

def build_window_key(df, prefix=""):
    """
    Returns "log|t_start|t_end" using available time columns.
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


# ---------- model (must match training exactly) ----------
class MLPProj(nn.Module):
    def __init__(self, d_in, hidden=256, proj_dim=128, dropout=0.10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, proj_dim)
    def forward(self, x):
        h = self.f(x)
        z = self.head(h)
        return F.normalize(z, dim=1, eps=1e-6)  # match training: L2-normalized


def main():
    ap = argparse.ArgumentParser()
    # Location of the Train650 contrastive artifacts:
    ap.add_argument("--train-contrastive-dir", required=True,
        help="Dir with TRAIN650 S2 contrastive artifacts (meta.json, model_contrastive.pt, meta_trained.json).")
    # VAL50 packed features (same schema as TRAIN650 contrastive_{train/val}.parquet):
    ap.add_argument("--val-contrastive-parquet", required=True,
        help="VAL50 packed contrastive parquet produced by s2_pack_contrastive (e.g., .../val50/s2/contrastive/contrastive_val.parquet).")
    # Output embeddings path:
    ap.add_argument("--out-parquet", required=False,
        help="Output path for embeddings. Default: <val-contrastive-parquet dir>/embeddings_val.parquet")
    # Runtime
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    set_seed(args.seed)

    tdir = Path(args.train_contrastive_dir)
    vparq = Path(args.val_contrastive_parquet)
    out_parq = Path(args.out_parquet) if args.out_parquet else (vparq.parent / "embeddings_val.parquet")
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- Load TRAIN650 meta (robust stats + feature list) ----
    meta_path = tdir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing TRAIN650 meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())
    feat_cols = meta["feat_cols"]
    center   = meta["robust_center"]
    mad      = meta["robust_scale_mad"]

    # Model hparams: prefer meta_trained.json if present (saves hidden/proj_dim/dropout actually used)
    mt_path = tdir / "meta_trained.json"
    if mt_path.exists():
        mt = json.loads(mt_path.read_text())
        hidden   = int(mt.get("hidden", 256))
        proj_dim = int(mt.get("proj_dim", 128))
        dropout  = float(mt.get("dropout", 0.10))
    else:
        hidden, proj_dim, dropout = 256, 128, 0.10

    # ---- Load trained model ----
    model_path = tdir / "model_contrastive.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing TRAIN650 model_contrastive.pt at {model_path}")

    # ---- Read VAL50 contrastive parquet ----
    df = pd.read_parquet(vparq).copy()
    if "window_key" not in df.columns:
        df["window_key"] = build_window_key(df, prefix="VAL50: ")

    # ---- Standardize using TRAIN650 robust stats ----
    X = robust_standardize(df, feat_cols, center, mad)

    # ---- Build model skeleton, load weights ----
    model = MLPProj(d_in=X.shape[1], hidden=hidden, proj_dim=proj_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- Inference (batched) ----
    Z_blocks = []
    with torch.no_grad():
        for k in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[k:k+args.batch_size]).to(device)
            zb = model(xb).detach().cpu().numpy().astype(np.float32)
            Z_blocks.append(zb)
    Z = np.vstack(Z_blocks)
    norms = np.linalg.norm(Z, axis=1)
    tiny = float((norms < 1e-5).mean() * 100.0)

    # ---- Export embeddings (schema matches training exports) ----
    ids = df[["log_id","window_t_start","window_t_end","track_uuid","category","window_key"]].reset_index(drop=True)
    emb_cols = [f"emb_{j:03d}" for j in range(Z.shape[1])]
    emb_df = pd.DataFrame(Z, columns=emb_cols)
    out_df = pd.concat([ids, emb_df], axis=1)
    out_df.to_parquet(out_parq, index=False)

    print(f"[S2-CL-INFER] read={len(df)}  d_in={X.shape[1]}  proj_dim={Z.shape[1]}  device={device.type}")
    print(f"[S2-CL-INFER] mean||z||={norms.mean():.4f}  tiny%(<1e-5)={tiny:.2f}")
    print(f"[S2-CL-INFER] wrote → {out_parq}")

if __name__ == "__main__":
    main()
