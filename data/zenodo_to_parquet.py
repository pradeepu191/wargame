"""Convert the Zenodo trades archive (Albers et al. 2026, record 18184441) to the recorder's
parquet layout, so analysis/calibrate.py reads both sources the same way.

Archive layout (SCHEMA.md §3): trades_YYYY_MM.tar -> <date>/<H>.gz, gzip JSON lines, one trade per
line, ALL coins interleaved:
    coin, side ("B" buy aggressor / "A" sell aggressor), time (ISO, ns), px, sz, hash,
    trade_dir_override, side_info: [ {user, start_pos, oid, twap_id, cloid}, {...} ]

Who is the taker?  The schema does not say which side_info entry is which.  Order ids are
assigned incrementally by the matching engine, so the AGGRESSOR is the entry with the LARGER
oid (its order arrived later than the resting one).  With side = "B" the taker bought, so
buyer = taker, seller = maker; with "A" the reverse.  The converter also reports how often
side_info[0] turned out to be the buyer, as a check on the convention.

Output: <out>/trades_<COIN>_<YYYYMMDDHH>.parquet with the recorder's columns
    time_ms, coin, side, px, sz, tid, hash, buyer, seller
plus archive-only columns
    taker_oid, maker_oid, taker_start_pos, maker_start_pos, taker_twap, maker_twap, same_block
(same_block: hash is all zeros, i.e. the order crossed in the block it was submitted).

Usage:
  python data/zenodo_to_parquet.py --tar data/raw/zenodo/trades_2026_01.tar \
      --coins BTC,ETH,SOL,HYPE [--dates 20260105-20260107] [--out data/raw/zenodo_parquet]
Streams the tar; memory is one hour-file at a time.  ~1-2 min per day of data for four coins.
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import orjson as _json
    _loads = _json.loads
except ImportError:                       # pragma: no cover
    import json as _json
    _loads = _json.loads

MEMBER = re.compile(r"(?:^|/)(?P<date>\d{8})/(?P<hour>\d{1,2})\.gz$")
ZERO_HASH = re.compile(r"^0x0+$")

COLS = ["time_ms", "coin", "side", "px", "sz", "tid", "hash", "buyer", "seller",
        "taker_oid", "maker_oid", "taker_start_pos", "maker_start_pos", "taker_twap", "maker_twap", "same_block"]


def parse_hour(raw: bytes, coins: set[str] | None, tid0: int) -> tuple[pd.DataFrame, dict]:
    """Parse one gzip JSONL hour file into the canonical frame.  Returns (df, diagnostics)."""
    rows = []
    idx0_buyer = 0
    n_all = 0
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
        for line in f:
            n_all += 1
            if coins is not None:
                # cheap pre-filter before JSON parsing: the coin is the first field
                j = line.find(b'"coin"')
                if j < 0:
                    continue
                k = line.find(b'"', j + 7)
                k2 = line.find(b'"', k + 1)
                if line[k + 1:k2].decode() not in coins:
                    continue
            t = _loads(line)
            si = t["side_info"]
            if len(si) != 2:
                continue
            a, b = si
            # aggressor = larger oid (arrived later than the resting order)
            if a["oid"] >= b["oid"]:
                taker, maker, taker_idx = a, b, 0
            else:
                taker, maker, taker_idx = b, a, 1
            side = t["side"]
            if side == "B":
                buyer, seller = taker["user"], maker["user"]
                idx0_buyer += taker_idx == 0
            else:
                buyer, seller = maker["user"], taker["user"]
                idx0_buyer += taker_idx == 1
            h = t.get("hash") or ""
            rows.append((t["time"], t["coin"], side, float(t["px"]), float(t["sz"]), tid0 + len(rows), h,
                         buyer, seller, int(taker["oid"]), int(maker["oid"]),
                         float(taker["start_pos"]) if taker.get("start_pos") is not None else np.nan,
                         float(maker["start_pos"]) if maker.get("start_pos") is not None else np.nan,
                         taker.get("twap_id") is not None, maker.get("twap_id") is not None,
                         bool(ZERO_HASH.match(h))))
    df = pd.DataFrame(rows, columns=COLS)
    if len(df):
        df["time_ms"] = (pd.to_datetime(df["time_ms"], format="ISO8601", utc=True).astype("int64") // 1_000_000)
    diag = {"n_lines": n_all, "n_kept": len(df), "idx0_is_buyer_share": idx0_buyer / max(len(df), 1)}
    return df, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--coins", default="BTC,ETH,SOL,HYPE", help="comma list, or 'all'")
    ap.add_argument("--dates", default=None, help="YYYYMMDD or YYYYMMDD-YYYYMMDD (inclusive)")
    ap.add_argument("--hours", default=None, help="e.g. 0-23 or 12-14")
    ap.add_argument("--out", default="data/raw/zenodo_parquet")
    args = ap.parse_args()
    coins = None if args.coins == "all" else {c.strip() for c in args.coins.split(",")}
    d0 = d1 = None
    if args.dates:
        d0, d1 = (args.dates.split("-") + [None])[:2]
        d1 = d1 or d0
    h0, h1 = (0, 23) if not args.hours else map(int, args.hours.split("-"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tid = 0
    t_start = time.time()
    n_files = 0
    idx0 = []
    with tarfile.open(args.tar, "r|") as tf:            # streaming: no seeking, one member at a time
        for m in tf:
            mm = MEMBER.search(m.name)
            if not m.isfile() or not mm:
                continue
            date, hour = mm["date"], int(mm["hour"])
            if (d0 and date < d0) or (d1 and date > d1) or hour < h0 or hour > h1:
                continue
            raw = tf.extractfile(m).read()
            df, diag = parse_hour(raw, coins, tid)
            tid += len(df)
            idx0.append(diag["idx0_is_buyer_share"])
            for coin, g in df.groupby("coin", sort=False):
                g.to_parquet(out / f"trades_{coin}_{date}{hour:02d}.parquet", index=False, compression="zstd")
            n_files += 1
            print(f"{date} h{hour:02d}: {diag['n_lines']} lines -> {diag['n_kept']} kept "
                  f"({', '.join(f'{c}:{n}' for c, n in df.coin.value_counts().items())})  "
                  f"[{time.time() - t_start:.0f}s]", flush=True)
    print(f"done: {n_files} hour files, {tid} trades, side_info[0] was the buyer in "
          f"{np.mean(idx0) if idx0 else float('nan'):.3f} of trades (convention check)")


if __name__ == "__main__":
    main()
