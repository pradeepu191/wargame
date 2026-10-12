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


def _book(rows):
    b = pd.DataFrame(rows, columns=["time_ms", "bid", "ask"])
    b["bid_sz"] = 1.0; b["ask_sz"] = 1.0
    b["mid"] = (b.bid + b.ask) / 2
    b["spread_bps"] = (b.ask - b.bid) / b.mid * 1e4
    return b


def test_fill_mid_is_the_pre_trade_book():
    """A trade and the top-of-book update it caused share a block timestamp: the snapshot AT the
    fill's time is the post-trade book.  Strict alignment picks the previous one."""
    t0 = 1_760_000_000_000
    book = _book([(t0, 99.0, 101.0),            # pre-trade: one-tick book around 100
                  (t0 + 1000, 99.0, 103.0)])    # same block as the buy that cleared the ask
    assert mid_at(book, [t0 + 1000], strict=False)[0] == 101.0
    assert mid_at(book, [t0 + 1000], strict=True)[0] == 100.0
    tape = pd.DataFrame([{"time_ms": t0 + 1000, "coin": "X", "side": "buy", "px": 101.0, "sz": 1.0,
                          "buyer": "taker", "seller": "mm"}])
    tm = true_mid_markouts(with_roles(tape), book, horizons_s=(1,))
    # taker paid the full one-tick quoted spread (2 bps on 100) against the book it hit
    assert abs(tm.effective_bps.iloc[0] - 2 / 101 * 1e4) < 1e-6
    # at-or-before would have said the taker paid nothing
    post = mid_at(book, [t0 + 1000], strict=False)[0]
    assert 2 * (101.0 - post) / 101.0 * 1e4 == 0.0


def test_marks_outside_recorder_coverage_are_dropped():
    """bbo exists only while the recorder ran; trades outside that window must not be marked
    against a frozen last mid."""
    t0 = 1_760_000_000_000
    snaps = [(t0 + k * 1000, 99.0, 101.0) for k in range(101)]              # 100 s of coverage
    snaps += [(t0 + 7_200_000 + k * 1000, 109.0, 111.0) for k in range(101)]  # second session, 2 h later
    book = _book(snaps)
    s = book_summary(book)
    assert s["l2_sessions"] == 2 and abs(s["l2_coverage_h"] - 200 / 3600) < 1e-9
    tape = pd.DataFrame([
        {"time_ms": t0 + 50_500, "side": "buy", "px": 101.0},     # inside session 1, all marks inside
        {"time_ms": t0 + 95_500, "side": "buy", "px": 101.0},     # inside; its 10-s mark is outside
        {"time_ms": t0 + 3_600_000, "side": "buy", "px": 130.0},  # in the gap: no coverage
        {"time_ms": t0 + 7_200_000 + 500, "side": "sell", "px": 109.0},  # session 2
    ])
    tape["coin"] = "X"; tape["sz"] = 1.0
    tape["buyer"] = np.where(tape.side == "buy", "taker", "mm"); tape["seller"] = np.where(tape.side == "buy", "mm", "taker")
    tm = true_mid_markouts(with_roles(tape), book, horizons_s=(1, 10), max_stale_s=5.0)
    assert len(tm) == 3 and t0 + 3_600_000 not in set(tm.time_ms)
    r = tm.set_index("time_ms")
    assert np.isfinite(r.loc[t0 + 50_500, "markout_10_bps"])
    assert np.isfinite(r.loc[t0 + 95_500, "markout_1_bps"]) and np.isnan(r.loc[t0 + 95_500, "markout_10_bps"])
    assert np.isfinite(r.loc[t0 + 7_200_000 + 500, "markout_1_bps"])
    # default guard also kills the frozen-mid lookups directly
    assert np.isnan(mid_at(book, [t0 + 3_600_000], max_stale_ms=60_000)[0])
    assert np.isfinite(mid_at(book, [t0 + 3_600_000])[0])
