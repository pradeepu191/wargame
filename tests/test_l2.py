"""True-mid markouts on a synthetic book + tape where the proxy's bid-ask bounce is known."""
import numpy as np
import pandas as pd

from analysis.l2 import book_summary, load_l2, mid_at, true_mid_markouts
from analysis.wallets import with_roles


def synthetic_book_and_tape(n=4000, seed=0, half_spread_bps=5.0):
    """Mid is a random walk sampled every second; takers alternate buy/sell at the ask/bid (pure
    bounce, no information) so the TRUE markout is ~0 while the trade-price proxy shows a positive
    maker 'profit' of about one spread."""
    rng = np.random.default_rng(seed)
    t0 = 1_759_770_000_000
    snap_t = t0 + np.arange(n) * 1000
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    h = mid * half_spread_bps / 1e4
    rows = []
    for tm, m, hh in zip(snap_t, mid, h):
        rows.append({"time_ms": tm, "coin": "X", "side": "bid", "level": 0, "px": m - hh, "sz": 1.0, "n_orders": 1})
        rows.append({"time_ms": tm, "coin": "X", "side": "ask", "level": 0, "px": m + hh, "sz": 1.0, "n_orders": 1})
        rows.append({"time_ms": tm, "coin": "X", "side": "bid", "level": 1, "px": m - 2 * hh, "sz": 2.0, "n_orders": 3})
    l2 = pd.DataFrame(rows)
    trades = []
    for k in range(n - 1):
        tm = snap_t[k] + 500
        side = "buy" if k % 2 == 0 else "sell"
        px = mid[k] + h[k] if side == "buy" else mid[k] - h[k]
        trades.append({"time_ms": tm, "coin": "X", "side": side, "px": px, "sz": 1.0,
                       "buyer": "taker" if side == "buy" else "mm", "seller": "mm" if side == "buy" else "taker"})
    return l2, pd.DataFrame(trades)


def test_true_mid_removes_bounce(tmp_path):
    l2, tape = synthetic_book_and_tape()
    l2.to_parquet(tmp_path / "l2_X_2026100700.parquet", index=False)
    book = load_l2(tmp_path, "X")
    assert abs(book.spread_bps.median() - 10.0) < 0.1 and book_summary(book)["l2_snapshots"] == len(l2) // 3
    t = with_roles(tape)
    tm = true_mid_markouts(t, book, horizons_s=(1, 10))
    # true-mid: the maker keeps the half spread (markout = +5 bps) and there is no impact
    assert abs(tm.effective_bps.mean() - 10.0) < 0.3
    assert abs(tm.markout_10_bps.mean() - 5.0) < 0.5
    assert abs(tm.impact_10_bps.mean()) < 0.5
    # trade-price proxy on the same tape: alternating sides make the maker look a full spread richer
    from analysis.wallets import mid_proxy
    s = np.where(t.maker_sold, 1.0, -1.0)
    proxy = (-s * (mid_proxy(t, 1000) - t.px) / t.px * 1e4).mean()
    assert proxy > 9.0
    assert np.isnan(mid_at(book, np.array([book.time_ms.iloc[0] - 1]))[0])
