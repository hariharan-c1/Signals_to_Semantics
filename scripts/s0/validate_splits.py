#!/usr/bin/env python3
import argparse
from pathlib import Path

def read_list(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="configs/splits")
    ap.add_argument("--train_root", required=True)
    args = ap.parse_args()

    dev = set(read_list(Path(args.splits_dir)/"dev100.txt"))
    val = set(read_list(Path(args.splits_dir)/"val50.txt"))
    tr550 = set(read_list(Path(args.splits_dir)/"train550.txt"))
    tr650 = set(read_list(Path(args.splits_dir)/"train650.txt"))

    ok = True
    for a,b,lab in [(dev,val,"dev∩val"), (dev,tr550,"dev∩train550"), (val,tr550,"val∩train550")]:
        inter = a & b
        if inter:
            print(f"[ERROR] overlap {lab}: {len(inter)} e.g. {sorted(list(inter))[:5]}")
            ok = False

    if tr650 != (dev | tr550):
        print("[ERROR] train650 != dev100 ∪ train550"); ok = False

    # existence check
    root = Path(args.train_root)
    missing = [n for n in (dev|val|tr550) if not (root/n).exists()]
    if missing:
        print(f"[ERROR] missing {len(missing)} dirs under train_root, e.g. {missing[:5]}"); ok = False

    print("[OK] splits look fine." if ok else "[FAIL] see errors above.")
