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
EXPECTED = {(a, s) for a in (0.1, 0.3, 0.5) for s in range(5)}
for policy, files in parts.items():
    df = pd.concat(pd.read_csv(f) for f in files)
    out = root / f"results/cfl_exp2_entry_relearn_{policy}.csv"
    df.to_csv(out, index=False)
    have = {(round(float(a), 2), int(s)) for a, s in df[["alpha", "seed"]].drop_duplicates().itertuples(index=False)}
    missing = sorted(EXPECTED - have)
    status = "complete" if not missing else f"MISSING {missing}"
    print(f"{policy}: {len(files)} parts -> {out.name}  [{status}]")
