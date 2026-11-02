"""Marked fills, cached and marked incrementally.

Every analysis script used to rebuild the same thing from scratch: load every trades_<COIN>_<hour>
file, load every bbo file into one book, assign maker/taker roles, and mark every fill against
the true mid at 1, 10 and 60 s.  On a month of BTC that is a few million fills against a book
with several million updates, four times per run.  This module does it once and keeps it:

    data/cache/book_<COIN>.parquet      the top-of-book stream, one row per change
    data/cache/tm_<COIN>.parquet        the marked fills (analysis.l2.true_mid_markouts output)
    data/cache/manifest_<COIN>.json     which hour files are in the cache, with their sizes

`marked_fills(raw, coin)` returns (tm, book) and only does work for hours that are new or whose
file has grown since the cache was written.  The last cached hour is always re-marked, because
its 60 s marks may now be resolvable against the next hour's book.  Delete the cache directory
to force a full rebuild (`--rebuild` on the scripts does the same).  The cache is derived from
raw data, so it lives under data/ and is gitignored like the raw files.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 10, 60)
PAT = re.compile(r"_(?P<hour>\d{10})\.parquet$")


def _hour_files(raw: Path, stream: str, coin: str) -> dict[str, Path]:
    out = {}
    for f in sorted(glob.glob(str(raw / f"{stream}_{coin}_*.parquet"))):
        m = PAT.search(f)
        if m:
            out[m.group("hour")] = Path(f)
    return out


def _sizes(files: dict[str, Path]) -> dict[str, int]:
    return {h: p.stat().st_size for h, p in files.items()}


def _load_book_files(files: list[Path]) -> pd.DataFrame:
    from analysis.l2 import load_l2
    if not files:
        return pd.DataFrame()
    book = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    book = book.drop_duplicates("time_ms", keep="last").sort_values("time_ms")
    book = book[book.ask > book.bid]
    book["mid"] = (book.bid + book.ask) / 2
    book["spread_bps"] = (book.ask - book.bid) / book.mid * 1e4
    return book.reset_index(drop=True)


def _load_trade_files(files: list[Path]) -> pd.DataFrame:
    from analysis.wallets import with_roles
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["buyer", "seller"]).drop_duplicates(["tid"] if "tid" in df.columns else None)
    return with_roles(df.sort_values("time_ms", kind="stable").reset_index(drop=True))


def marked_fills(raw: Path | str, coin: str, cache_dir: Path | str | None = None, horizons=HORIZONS,
                 rebuild: bool = False, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(marked fills, book) for one coin, from the cache plus whatever is new under raw."""
    from analysis.l2 import true_mid_markouts
    raw = Path(raw)
    cache = Path(cache_dir) if cache_dir else raw.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tm_path, book_path, man_path = cache / f"tm_{coin}.parquet", cache / f"book_{coin}.parquet", cache / f"manifest_{coin}.json"
    trade_files, bbo_files = _hour_files(raw, "trades", coin), _hour_files(raw, "bbo", coin)
    if not trade_files or not bbo_files:
        return pd.DataFrame(), pd.DataFrame()
    man = {"trades": {}, "bbo": {}, "horizons": list(horizons)}
    if not rebuild and man_path.exists() and tm_path.exists() and book_path.exists():
        man = json.loads(man_path.read_text())
        if tuple(man.get("horizons", ())) != tuple(horizons):
            man = {"trades": {}, "bbo": {}, "horizons": list(horizons)}
    cur_t, cur_b = _sizes(trade_files), _sizes(bbo_files)
    new_b = [h for h, s in cur_b.items() if man["bbo"].get(h) != s]
    new_t = [h for h, s in cur_t.items() if man["trades"].get(h) != s]
    cached_hours = sorted(man["trades"])
    if cached_hours and not rebuild:
        new_t.append(cached_hours[-1])                   # re-mark the last cached hour
    new_t = sorted(set(new_t))
    # ---- book: append new bbo hours
    if man["bbo"] and not rebuild:
        book = pd.read_parquet(book_path)
        if new_b:
            add = _load_book_files([bbo_files[h] for h in new_b])
            keep = ~book.time_ms.floordiv(3_600_000).astype(np.int64).isin(
                [int(pd.Timestamp(f"{h[:4]}-{h[4:6]}-{h[6:8]} {h[8:]}:00", tz="UTC").value // 1_000_000 // 3_600_000) for h in new_b])
            book = pd.concat([book[keep], add], ignore_index=True).drop_duplicates("time_ms", keep="last").sort_values("time_ms").reset_index(drop=True)
    else:
        book = _load_book_files(list(bbo_files.values()))
    # ---- fills: mark new hours against the full book
    if man["trades"] and not rebuild:
        tm = pd.read_parquet(tm_path)
    else:
        tm = pd.DataFrame()
        new_t = sorted(trade_files)
    if new_t:
        if verbose:
            print(f"{coin}: marking {len(new_t)} hour(s) ({len(trade_files)} on disk, {len(cached_hours)} cached)", flush=True)
        fresh = _load_trade_files([trade_files[h] for h in new_t])
        fresh = true_mid_markouts(fresh, book, horizons_s=horizons)
        if len(tm):
            tm = tm[~tm.hour.isin(new_t)]
        fresh = fresh.assign(hour=pd.to_datetime(fresh.time_ms, unit="ms", utc=True).dt.strftime("%Y%m%d%H"))
        tm = pd.concat([tm, fresh], ignore_index=True).sort_values("time_ms", kind="stable").reset_index(drop=True)
        tm.to_parquet(tm_path, index=False)
        book.to_parquet(book_path, index=False)
        man_path.write_text(json.dumps({"trades": cur_t, "bbo": cur_b, "horizons": list(horizons)}))
    elif new_b:
        book.to_parquet(book_path, index=False)
        man_path.write_text(json.dumps({"trades": cur_t, "bbo": cur_b, "horizons": list(horizons)}))
    return tm, book
