# scripts/s1b_infer_nnpu.py
import argparse, json, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------- Model must exactly match training ---------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(1)


# --------- Helpers ---------
def _normalize_key_cols(df: pd.DataFrame):
    """Ensure df has ['log_id','window_t_start','window_t_end']."""
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


def _load_meta(nnpu_dir: Path):
    meta_path = nnpu_dir / "nnpu_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"nnPU meta not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # sanity
    for k in ("feat_cols", "imputer", "scaler", "model"):
        if k not in meta:
            raise KeyError(f"nnPU meta missing '{k}'")
    return meta


def _prepare_features(df: pd.DataFrame, meta: dict) -> np.ndarray:
    """Impute & standardize using training-time statistics from meta."""
    feat_cols = meta["feat_cols"]
    Xcols = [c for c in feat_cols if c in df.columns]
    if not Xcols:
        raise RuntimeError("No overlap between meta['feat_cols'] and VAL window_table columns.")

    # Fill with training medians; if a feature exists in meta but not in this split,
    # we safely ignore (it won't be in Xcols).
    med = pd.Series(meta["imputer"]["median"])
    mu  = pd.Series(meta["scaler"]["mean"])
    sd  = pd.Series(meta["scaler"]["std"])

    # defaults for any column found in VAL but not in meta (rare)
    for k in Xcols:
        if k not in med: med[k] = 0.0
        if k not in mu:  mu[k]  = 0.0
        if k not in sd:  sd[k]  = 1.0

    Ximp = df[Xcols].fillna(med)
    Xz = ((Ximp - mu) / sd.replace(0, 1.0)).values.astype(np.float32)
    return Xz, Xcols


def _batched_predict(model: nn.Module, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    out = np.zeros((len(X),), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            pb = model(xb).detach().cpu().numpy().astype(np.float32)
            out[i:i+batch_size] = pb
    return out


# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-table", required=True,
                    help="VAL50 window_table.parquet (features table).")
    ap.add_argument("--nnpu-dir", required=True,
                    help="Directory with nnpu_model.pt and nnpu_meta.json from training (e.g., Train650 run).")
    ap.add_argument("--out-jsonl", required=True,
                    help="Output JSONL path for scores (scores_nnpu_val50.jsonl).")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    set_seed(args.seed)

    wt_path = Path(args.window_table)
    nnpu_dir = Path(args.nnpu_dir)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # --- Load meta + model ---
    meta = _load_meta(nnpu_dir)
    model_path = nnpu_dir / "nnpu_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"nnPU model not found: {model_path}")

    # Read features
    df = pd.read_parquet(wt_path)
    df = _normalize_key_cols(df)

    # Prepare features
    Xz, used_cols = _prepare_features(df, meta)

    # Build model skeleton with same dims/hparams as training
    d_in = Xz.shape[1]
    hidden = int(meta["model"].get("hidden", 128))
    dropout = float(meta["model"].get("dropout", 0.1))
    model = MLP(d_in=d_in, d_hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Predict
    scores = _batched_predict(model, Xz, device, batch_size=args.batch_size)

    # Write JSONL (mirror training helper)
    out_df = pd.DataFrame({
        "log_id": df["log_id"].astype(str),
        "t_start": df["window_t_start"].astype(float),
        "t_end": df["window_t_end"].astype(float),
        "score_nnpu": scores.astype(float)
    })
    out_df.to_json(out_path, orient="records", lines=True)
    print(f"[nnPU-INFER] read={len(df)}  used_feats={len(used_cols)}  device={device.type}")
    print(f"[nnPU-INFER] wrote → {out_path}")

if __name__ == "__main__":
    main()
