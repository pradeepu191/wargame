"""True-mid markouts from the recorder's top-of-book stream (data/record.py: bbo_<COIN>_<hour>.parquet,
one row per change of the best bid or ask; falls back to the throttled l2Book snapshots).

The trades-only pipeline uses the next trade price as a mid proxy, which carries bid-ask bounce
(a maker's sale at the ask followed by a print at the bid looks like a profit).  With the book:

    mid_t         = (best bid + best ask) / 2 at the last snapshot STRICTLY before t
    markout_d     = -s (mid_{t+d} - p) / p * 1e4        maker's PnL per unit vs the mid d s later, bps
                                                        (= half the realized spread: it INCLUDES the
                                                        half-spread earned at the fill; s = +1 if the
                                                        maker sold).  impact_d isolates adverse selection.
    effective     = 2 s (p - mid_t) / p * 1e4           what the taker paid vs the mid at the fill
    realized_d    = 2 s (p - mid_{t+d}) / p * 1e4       what the maker kept after d seconds
    effective     = realized_d + impact_d

Two alignment rules matter, both learned from the Oct 2026 live runs:

  * Strictly before, not at-or-before, for the fill's own mid.  Hyperliquid stamps a trade and the
    top-of-book update it caused with the same block time, so the snapshot AT the fill's time is
    the post-trade book; marking a buy at the ask against it gives an effective spread of zero or
    less.  The pre-trade book is the last snapshot with an earlier timestamp.  The marks d seconds
    later use at-or-before, which is what "the mid as of t + d" means.
  * A coverage guard.  Trades run continuously but the bbo stream exists only while a recorder with
    the bbo subscription was alive; outside those windows the "latest snapshot" is hours old and
    the book moved unseen.  A lookup whose latest snapshot is older than `max_stale_s` (60 s by
    default; BTC's top of book changes several times a second) returns NaN, and fills with no
    pre-trade mid are dropped.  `book_summary` reports the covered hours and the number of
    recording sessions.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

MAX_STALE_S = 60.0


def load_l2(raw: Path, coin: str) -> pd.DataFrame:
    """Best bid / ask over time: columns time_ms, bid, ask, mid, spread_bps, bid_sz, ask_sz.
    Prefers bbo_<coin>_*.parquet (every top-of-book change) and falls back to the top level of the
    throttled l2Book snapshots (~one per 5 s), which leaves the mid stale between snapshots."""
    bbo_files = sorted(glob.glob(str(Path(raw) / f"bbo_{coin}_*.parquet")))
    if bbo_files:
        book = pd.concat([pd.read_parquet(f) for f in bbo_files], ignore_index=True)
        book = book.drop_duplicates("time_ms", keep="last").sort_values("time_ms")
    else:
        files = sorted(glob.glob(str(Path(raw) / f"l2_{coin}_*.parquet")))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        top = df[df.level == 0]
        bid = top[top.side == "bid"][["time_ms", "px", "sz"]].rename(columns={"px": "bid", "sz": "bid_sz"})
        ask = top[top.side == "ask"][["time_ms", "px", "sz"]].rename(columns={"px": "ask", "sz": "ask_sz"})
        book = bid.merge(ask, on="time_ms", how="inner").drop_duplicates("time_ms").sort_values("time_ms")
    book = book[book.ask > book.bid]
    book["mid"] = (book.bid + book.ask) / 2
    book["spread_bps"] = (book.ask - book.bid) / book.mid * 1e4
    return book.reset_index(drop=True)


def mid_at(book: pd.DataFrame, times_ms, strict: bool = False, max_stale_ms: float | None = None) -> np.ndarray:
    """Mid as of the last snapshot at or before each time -- strictly before it when strict=True
    (the pre-trade book for a fill; see the module docstring).  NaN before the first snapshot and,
    when max_stale_ms is given, wherever that snapshot is older than max_stale_ms (no recorder
    coverage at that time)."""
    tb = book.time_ms.to_numpy()
    times_ms = np.asarray(times_ms, dtype=np.int64)
    idx = np.searchsorted(tb, times_ms, side="left" if strict else "right") - 1
    ok = idx >= 0
    if max_stale_ms is not None and len(tb):
        age = times_ms - tb[np.clip(idx, 0, len(tb) - 1)]
        ok &= age <= max_stale_ms
    out = np.full(len(times_ms), np.nan)
    out[ok] = book.mid.to_numpy()[idx[ok]]
    return out


def true_mid_markouts(trades: pd.DataFrame, book: pd.DataFrame, horizons_s=(1, 10, 60),
                      max_stale_s: float = MAX_STALE_S) -> pd.DataFrame:
    """Adds mid_0, effective_bps, markout_<d>_bps, realized_<d>_bps, impact_<d>_bps to a roles frame
    (analysis.wallets.with_roles output).  Fills with no pre-trade snapshot inside the coverage
    window are dropped; a mark whose t + d falls outside coverage is NaN for that horizon only."""
    t = trades.copy()
    times = t.time_ms.to_numpy()
    stale = max_stale_s * 1000
    t["mid_0"] = mid_at(book, times, strict=True, max_stale_ms=stale)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["effective_bps"] = 2 * s * (t.px - t.mid_0) / t.px * 1e4
    for d in horizons_s:
        m = mid_at(book, times + d * 1000, strict=False, max_stale_ms=stale)
        t[f"markout_{d}_bps"] = -s * (m - t.px) / t.px * 1e4
        t[f"realized_{d}_bps"] = 2 * s * (t.px - m) / t.px * 1e4
        t[f"impact_{d}_bps"] = 2 * s * (m - t.mid_0) / t.px * 1e4
    return t.dropna(subset=["mid_0"])


def book_summary(book: pd.DataFrame, max_stale_s: float = MAX_STALE_S) -> dict:
    """Snapshot count and spacing, covered hours (gaps longer than max_stale_s are recorder downtime,
    not quiet books), number of recording sessions, and the quoted spread / top depth per snapshot."""
    if book.empty:
        return {}
    gaps = np.diff(book.time_ms.to_numpy()) / 1000.0
    covered = gaps[gaps <= max_stale_s]
    return {"l2_snapshots": len(book), "l2_median_gap_s": float(np.median(gaps)) if len(gaps) else np.nan,
            "l2_coverage_h": float(covered.sum() / 3600.0), "l2_sessions": int(1 + (gaps > max_stale_s).sum()),
            "spread_bps_median": float(book.spread_bps.median()),
            "spread_bps_p90": float(book.spread_bps.quantile(0.9)),
            "top_depth_usd_median": float(((book.bid_sz + book.ask_sz) / 2 * book.mid).median())}
