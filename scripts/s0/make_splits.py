#!/usr/bin/env python3
import argparse, os, random
from pathlib import Path

def list_logs(root):
    # one subdir per log, sorted for stable indexing
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])

def write_list(path, rows):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_root",
        required=True,
        help="Local Argoverse 2 Sensor train directory",
    )
    ap.add_argument("--out_dir", default="configs/splits")
    ap.add_argument("--dev_n", type=int, default=100,
                    help="first N logs reserved as Dev (1..N)")
    ap.add_argument("--val_n", type=int, default=50,
                    help="number of Val logs to draw from N+1..end")
    ap.add_argument("--seed", type=int, default=1337,
                    help="PRNG seed for reproducibility")
    args = ap.parse_args()

    logs = list_logs(args.train_root)
    assert len(logs) >= args.dev_n + args.val_n, "Not enough logs available"

    dev = logs[:args.dev_n]              # 1..100 (Dev)
    pool = logs[args.dev_n:]             # 101..700 (Pool)
    rng = random.Random(args.seed)
    val = sorted(rng.sample(pool, args.val_n))  # 50 Val (random, reproducible)
    # Train = all others (Dev + (Pool - Val))
    pool_minus_val = [x for x in pool if x not in set(val)]
    train550 = pool_minus_val            # 600 - 50 = 550
    train650 = dev + train550            # Dev100 + Train550

    write_list(f"{args.out_dir}/dev100.txt", dev)
    write_list(f"{args.out_dir}/val50.txt", val)
    write_list(f"{args.out_dir}/train550.txt", train550)
    write_list(f"{args.out_dir}/train650.txt", train650)

    print(f"Done.\nDev100={len(dev)}  Val50={len(val)}  Train550={len(train550)}  Train650={len(train650)}")
