"""Event study: what the book looks like, hour by hour, around a dated market event.

Events live in data/events.csv (time_utc, label, source).  For every event and coin, every
hour from -before to +after hours around it (UTC, hour buckets aligned to the clock):

  vol_bps          the mid's mean absolute 10 s move per touch fill
  n_touch          touch fills in the hour
  toxic_fill_share share of touch fills from wallets typed toxic (typed on the data before the
                   largest gap, as in stress.py), benign_fill_share likewise
  top_share        the top makers' share of all touch fills
  benign_top_share / toxic_top_share / gap   the identity-selection gap inside the hour
  top_markout_bps / rest_markout_bps         who paid
  fringe_share     the share of top-10 fills taken by the single worst top-10 wallet of the
                   window (the identity-blind one, if there is one)

So one can see, for the Binance hack: did toxic flow surge, did the selecting makers step out
or stay, did the gap survive inside the hour, and who absorbed it.
    python analysis/events.py --raw data/raw/live --results results/live_2026_10 [--before 24 --after 24]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000


def event_table(tm: pd.DataFrame, top: list[str], events: pd.DataFrame, horizon_s: int = 10,
                n_min_taker: int = 30, before_h: int = 24, after_h: int = 24) -> list[dict]:
    from analysis.longwindow import _first_session_split, _prep, _type_takers
    t, mk = _prep(tm, top, horizon_s)
    if t.empty:
        return []
    a, _ = _first_session_split(t)
    types = _type_takers(a, mk, n_min_taker)
    t["type"] = t.taker.map(types) if types is not None else np.nan
    t["hour_ms"] = (t.time_ms.to_numpy() // HOUR_MS) * HOUR_MS
    imp = f"impact_{horizon_s}_bps"
    # the worst top-10 wallet over the whole window, by mean markout (the fringe candidate)
    tops = t[t.is_top > 0].groupby("maker")[mk].agg(["mean", "size"])
    tops = tops[tops["size"] >= 200]
    worst = tops["mean"].idxmin() if len(tops) else None
    rows = []
    for _, ev in events.iterrows():
        t_ev = int(pd.Timestamp(ev.time_utc).value // 1_000_000)
        h0 = (t_ev // HOUR_MS) * HOUR_MS
        for k in range(-before_h, after_h + 1):
            g = t[t.hour_ms == h0 + k * HOUR_MS]
            if len(g) < 20:
                continue
            ben, tox = g[g.type == 0], g[g.type == 2]
            gt, gr = g[g.is_top > 0], g[g.is_top == 0]
            row = {"event": ev.label, "event_time_utc": ev.time_utc, "k_hours": k,
                   "hour_utc": pd.Timestamp(h0 + k * HOUR_MS, unit="ms", tz="UTC").isoformat(),
                   "n_touch": int(len(g)),
                   "vol_bps": float(g[imp].abs().mean()) if imp in g else np.nan,
                   "benign_fill_share": float(len(ben) / len(g)), "toxic_fill_share": float(len(tox) / len(g)),
                   "top_share": float(g.is_top.mean()),
                   "benign_top_share": float(ben.is_top.mean()) if len(ben) >= 10 else np.nan,
                   "toxic_top_share": float(tox.is_top.mean()) if len(tox) >= 10 else np.nan,
                   "top_markout_bps": float(gt[mk].mean()) if len(gt) else np.nan,
                   "rest_markout_bps": float(gr[mk].mean()) if len(gr) else np.nan,
                   "toxic_realized_bps": float(tox[mk].mean()) if len(tox) else np.nan,
                   "n_top_active": int(gt.maker.nunique()),
                   "fringe_share": float((gt.maker == worst).mean()) if worst is not None and len(gt) else np.nan,
                   "fringe_wallet": worst}
            row["gap"] = row["benign_top_share"] - row["toxic_top_share"] if len(ben) >= 10 and len(tox) >= 10 else np.nan
            rows.append(row)
    return rows


def coin_job(c, args):
    from pathlib import Path
    from analysis.cache import marked_fills
    raw, res = Path(args.raw), Path(args.results)
    events = pd.read_csv(args.events)
    tm, book = marked_fills(raw, c, rebuild=args.rebuild)
    if tm.empty or book.empty or not (res / f"wallets_{c}.csv").exists():
        print(f"{c}: missing trades, book or wallets_{c}.csv"); return None
    feats = pd.read_csv(res / f"wallets_{c}.csv")
    top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(args.top_k))
    rows = event_table(tm, top, events, before_h=args.before, after_h=args.after)
    if not rows:
        print(f"{c}: no fills inside any event window"); return []
    d = pd.DataFrame(rows)
    for ev, g in d.groupby("event", sort=False):
        pre, post = g[g.k_hours < 0], g[(g.k_hours >= 0) & (g.k_hours < 6)]
        print(f"{c} | {ev[:60]}: hours -{args.before}..-1 vs 0..5: vol {pre.vol_bps.mean():.2f} -> {post.vol_bps.mean():.2f} bps, "
              f"toxic fill share {pre.toxic_fill_share.mean():.2f} -> {post.toxic_fill_share.mean():.2f}, top share {pre.top_share.mean():.3f} -> {post.top_share.mean():.3f}, "
              f"gap {pre.gap.mean():+.2f} -> {post.gap.mean():+.2f}, top markout {pre.top_markout_bps.mean():+.2f} -> {post.top_markout_bps.mean():+.2f}, "
              f"rest {pre.rest_markout_bps.mean():+.2f} -> {post.rest_markout_bps.mean():+.2f}, fringe share of top fills {pre.fringe_share.mean():.2f} -> {post.fringe_share.mean():.2f}", flush=True)
    return rows


def main():
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.parallel import per_coin
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/live")
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--coins", default="BTC,ETH,SOL,HYPE")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true", help="ignore data/cache and re-mark every fill")
    ap.add_argument("--before", type=int, default=24)
    ap.add_argument("--after", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=4, help="coins in parallel")
    args = ap.parse_args()
    res = Path(args.results)
    coins = args.coins.split(",")
    results = per_coin(coin_job, coins, args, args.jobs)
    out = [{"coin": c, **r} for c, rows in zip(coins, results) if rows for r in rows]
    pd.DataFrame(out).to_csv(res / "events.csv", index=False)
    print("wrote", res / "events.csv")


if __name__ == "__main__":
    main()
