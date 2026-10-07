"""Coverage report for recorded or converted trades parquet: rows per coin per hour, the largest
silent gap inside each hour, and hours missing from the sequence.  A laptop that slept, a dropped
websocket, or an archive hole all show up here before they show up as a bad statistic.

Usage:
  python data/coverage.py [--raw data/raw/live] [--gap-s 120]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAT = re.compile(r"trades_(?P<coin>[A-Za-z0-9]+)_(?P<hour>\d{10})\.parquet$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/live")
    ap.add_argument("--gap-s", type=float, default=120.0, help="flag silent gaps longer than this (seconds)")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(Path(args.raw) / "trades_*.parquet"))):
        m = PAT.search(f)
        if not m:
            continue
        df = pd.read_parquet(f, columns=["time_ms", "buyer", "seller"])
        t = np.sort(df.time_ms.to_numpy())
        h0 = pd.Timestamp(m["hour"][:8] + "T" + m["hour"][8:] + ":00:00", tz="UTC").value // 10**6
        edges = np.concatenate([[h0], t, [h0 + 3_600_000]])
        gaps = np.diff(edges) / 1000.0
        rows.append({"coin": m["coin"], "hour": m["hour"], "rows": len(df),
                     "first_s": (t[0] - h0) / 1000.0 if len(t) else np.nan,
                     "last_s": (t[-1] - h0) / 1000.0 if len(t) else np.nan,
                     "max_gap_s": float(gaps.max()) if len(gaps) else 3600.0,
                     "missing_ids": int(df.buyer.isna().sum() + df.seller.isna().sum())})
    if not rows:
        raise SystemExit(f"no trades_*.parquet under {args.raw}")
    cov = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(cov.pivot(index="hour", columns="coin", values="rows").fillna(0).astype(int).to_string())
    print("\nhours with a silent gap >", args.gap_s, "s (any coin):")
    bad = cov[cov.max_gap_s > args.gap_s].sort_values(["hour", "coin"])
    print(bad[["hour", "coin", "rows", "max_gap_s"]].to_string(index=False) if len(bad) else "  none")
    hours = sorted(cov.hour.unique())
    seq = pd.date_range(pd.Timestamp(hours[0][:8] + "T" + hours[0][8:] + ":00", tz="UTC"),
                        pd.Timestamp(hours[-1][:8] + "T" + hours[-1][8:] + ":00", tz="UTC"), freq="h")
    missing = [s.strftime("%Y%m%d%H") for s in seq if s.strftime("%Y%m%d%H") not in set(hours)]
    print("\nmissing hours:", missing if missing else "none")
    print("rows with a missing wallet id:", int(cov.missing_ids.sum()))
    print(f"total: {cov.rows.sum()} prints over {len(hours)} hours, {cov.coin.nunique()} coins")


if __name__ == "__main__":
    main()
