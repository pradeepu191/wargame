"""True-mid markouts from the recorder's L2 snapshots (data/record.py, l2_<COIN>_<hour>.parquet).

The trades-only pipeline uses the next trade price as a mid proxy, which carries bid-ask bounce
(a maker's sale at the ask followed by a print at the bid looks like a profit).  With the book:

    mid_t         = (best bid + best ask) / 2 at the last snapshot at or before t
    markout_d     = -s (mid_{t+d} - p) / p * 1e4        maker's PnL per unit vs the mid d s later, bps
                                                        (= half the realized spread: it INCLUDES the
                                                        half-spread earned at the fill; s = +1 if the
                                                        maker sold).  impact_d isolates adverse selection.
    effective     = 2 s (p - mid_t) / p * 1e4           what the taker paid vs the mid at the fill
    realized_d    = 2 s (p - mid_{t+d}) / p * 1e4       what the maker kept after d seconds
    effective     = realized_d + impact_d

Snapshots arrive once per block (~1 s), so the mid is block-resolution; a fill inside a block is
marked against the book as of the previous snapshot, which is the book the resting order was in.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


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


def mid_at(book: pd.DataFrame, times_ms: np.ndarray) -> np.ndarray:
    """Mid as of the last snapshot at or before each time (NaN before the first snapshot)."""
    idx = np.searchsorted(book.time_ms.to_numpy(), times_ms, side="right") - 1
    out = np.full(len(times_ms), np.nan)
    ok = idx >= 0
    out[ok] = book.mid.to_numpy()[idx[ok]]
    return out


def true_mid_markouts(trades: pd.DataFrame, book: pd.DataFrame, horizons_s=(1, 10, 60)) -> pd.DataFrame:
    """Adds mid_0, effective_bps, markout_<d>_bps, realized_<d>_bps, impact_<d>_bps to a roles frame
    (analysis.wallets.with_roles output).  Rows without a book snapshot before them are dropped."""
    t = trades.copy()
    times = t.time_ms.to_numpy()
    t["mid_0"] = mid_at(book, times)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["effective_bps"] = 2 * s * (t.px - t.mid_0) / t.px * 1e4
    for d in horizons_s:
        m = mid_at(book, times + d * 1000)
        t[f"markout_{d}_bps"] = -s * (m - t.px) / t.px * 1e4
        t[f"realized_{d}_bps"] = 2 * s * (t.px - m) / t.px * 1e4
        t[f"impact_{d}_bps"] = 2 * s * (m - t.mid_0) / t.px * 1e4
    return t.dropna(subset=["mid_0"])


def book_summary(book: pd.DataFrame) -> dict:
    if book.empty:
        return {}
    gaps = np.diff(book.time_ms.to_numpy()) / 1000.0
    return {"l2_snapshots": len(book), "l2_median_gap_s": float(np.median(gaps)) if len(gaps) else np.nan,
            "spread_bps_median": float(book.spread_bps.median()),
            "spread_bps_p90": float(book.spread_bps.quantile(0.9)),
            "top_depth_usd_median": float(((book.bid_sz + book.ask_sz) / 2 * book.mid).median())}
