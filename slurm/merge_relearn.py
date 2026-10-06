"""Merge entry_relearn array-task outputs into one CSV per policy.

Usage (login node, after `squeue` shows the array finished):
    python slurm/merge_relearn.py            # merges every policy found in results/relearn_parts/
"""
import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
parts = defaultdict(list)
for f in sorted(glob.glob(str(root / "results/relearn_parts/*.csv"))):
    m = re.match(r"(\w+)_a[0-9.]+_s\d+\.csv$", Path(f).name)
    if m:
        parts[m.group(1)].append(f)
for policy, files in parts.items():
    df = pd.concat(pd.read_csv(f) for f in files)
    out = root / f"results/cfl_exp2_entry_relearn_{policy}.csv"
    df.to_csv(out, index=False)
    n_alpha = df.alpha.nunique(); n_seed = df.seed.nunique()
    print(f"{policy}: {len(files)} parts -> {out.name}  ({n_alpha} alphas x {n_seed} seeds)")
