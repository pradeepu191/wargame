"""The identity-selection result on a continuous recording: where it lives in the day, whether
it holds day after day, and how long a toxic label stays predictive.

Inputs are the same as analysis/stress.py (touch fills marked against the true mid; top makers
= the top-K MM wallets by volume in wallets_<COIN>.csv).  Four cuts, one CSV each:

  longwindow_hour.csv    by UTC hour of day, on fills after the first session: the top makers'
                         share of fills from benign vs toxic takers (typed on the first session,
                         as in stress.py), the gap, both groups' n, and the top / rest markouts
  longwindow_day.csv     the same by calendar day, so the gap is one number per day with its
                         5-minute-cluster SE: a seven-point out-of-sample test instead of one
  longwindow_rolling.csv rolling typing: takers typed on day d (>= n_min fills), scored on day
                         d+k for k = 1..K; the gap and the realized toxicity difference by k,
                         averaged over d.  How long does last night's label stay predictive?
  longwindow_makers.csv  per-maker 10 s markout by day (>= n_min fills) and the Spearman rank
                         correlation of consecutive days and of each day with the first: does the
                         ranking of who earns persist across a week?

    python analysis/longwindow.py --raw data/raw/live --results results/live_2026_10
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
BLOCK_MS = 300_000


def _type_takers(t: pd.DataFrame, mk: str, n_min: int) -> pd.Series | None:
    """Tercile type per taker (0 benign, 1 middle, 2 toxic) from the markout it inflicted."""
    from analysis.stress import _terciles
    typed = t.groupby("taker")[mk].agg(["mean", "size"])
    typed = typed[typed["size"] >= n_min]
    if len(typed) < 9:
        return None
    return pd.Series(_terciles((-typed["mean"]).to_numpy()), index=typed.index)


def _prep(tm: pd.DataFrame, top: list[str], horizon_s: int) -> tuple[pd.DataFrame, str]:
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk])
    t = t[t.at_touch.astype(bool)].copy()
    t["is_top"] = t.maker.isin(set(top)).astype(float)
    t["block"] = (t.time_ms.to_numpy() // BLOCK_MS).astype(np.int64)
    t["day"] = (t.time_ms.to_numpy() // DAY_MS).astype(np.int64)
    t["hour"] = ((t.time_ms.to_numpy() % DAY_MS) // 3_600_000).astype(np.int64)
    return t, mk


def _first_session_split(t: pd.DataFrame):
    gaps = np.diff(t.time_ms.to_numpy())
    cut = t.time_ms.iloc[int(gaps.argmax()) + 1] if len(gaps) and gaps.max() > 3_600_000 else t.time_ms.iloc[len(t) // 2]
    return t[t.time_ms < cut], t[t.time_ms >= cut].copy()


def _gap_rows(b: pd.DataFrame, mk: str, by: str) -> list[dict]:
    from analysis.makers import cluster_se
    rows = []
    for key, g in b.groupby(by, sort=True):
        ben, tox = g[g.type == 0], g[g.type == 2]
        if len(ben) < 20 or len(tox) < 20:
            continue
        gt, gr = g[g.is_top > 0], g[g.is_top == 0]
        # SE of the gap in shares: clustered by 5-minute block, both groups treated as independent
        se_b = cluster_se(ben.is_top.to_numpy(), ben.block.to_numpy()) if len(ben) > 1 else np.nan
        se_t = cluster_se(tox.is_top.to_numpy(), tox.block.to_numpy()) if len(tox) > 1 else np.nan
        rows.append({by: int(key), "n_touch": int(len(g)), "n_benign": int(len(ben)), "n_toxic": int(len(tox)),
                     "benign_top_share": float(ben.is_top.mean()), "toxic_top_share": float(tox.is_top.mean()),
                     "gap": float(ben.is_top.mean() - tox.is_top.mean()), "gap_se": float(np.sqrt(se_b ** 2 + se_t ** 2)),
                     "gap_ratio": float(ben.is_top.mean() / max(tox.is_top.mean(), 1e-9)),
                     "top_markout_bps": float(gt[mk].mean()) if len(gt) else np.nan,
                     "rest_markout_bps": float(gr[mk].mean()) if len(gr) else np.nan,
                     "toxic_realized_bps": float(tox[mk].mean()), "benign_realized_bps": float(ben[mk].mean()),
                     "top_share": float(g.is_top.mean())})
    return rows


def by_hour_and_day(tm: pd.DataFrame, top: list[str], horizon_s: int = 10, n_min_taker: int = 30):
    """Type on the first session, score everything after it, cut by UTC hour and by day."""
    t, mk = _prep(tm, top, horizon_s)
    if t.empty:
        return [], []
    a, b = _first_session_split(t)
    types = _type_takers(a, mk, n_min_taker)
    if types is None:
        return [], []
    b["type"] = b.taker.map(types)
    b = b.dropna(subset=["type"])
    return _gap_rows(b, mk, "hour"), _gap_rows(b, mk, "day")


def rolling_typing(tm: pd.DataFrame, top: list[str], horizon_s: int = 10, n_min_taker: int = 30, max_k: int = 10) -> list[dict]:
    """Type on day d, score on day d + k; one row per (d, k) with enough fills."""
    t, mk = _prep(tm, top, horizon_s)
    rows = []
    days = sorted(t.day.unique())
    for d in days:
        types = _type_takers(t[t.day == d], mk, n_min_taker)
        if types is None:
            continue
        n_typed = int(len(types))
        for k in range(1, max_k + 1):
            g = t[t.day == d + k].copy()
            if g.empty:
                continue
            g["type"] = g.taker.map(types)
            known = g.dropna(subset=["type"])
            ben, tox = known[known.type == 0], known[known.type == 2]
            if len(ben) < 20 or len(tox) < 20:
                continue
            rows.append({"typed_day": int(d), "k": k, "scored_day": int(d + k), "n_typed": n_typed,
                         "n_benign": int(len(ben)), "n_toxic": int(len(tox)),
                         "known_share": float(len(known) / len(g)),
                         "benign_top_share": float(ben.is_top.mean()), "toxic_top_share": float(tox.is_top.mean()),
                         "gap": float(ben.is_top.mean() - tox.is_top.mean()),
                         "type_holds_bps": float(ben[mk].mean() - tox[mk].mean())})
    return rows


def maker_persistence(tm: pd.DataFrame, horizon_s: int = 10, n_min: int = 100) -> tuple[list[dict], list[dict]]:
    """Per-maker markout by day, and rank correlations between days."""
    from scipy.stats import spearmanr
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk])
    t = t[t.at_touch.astype(bool)].copy()
    t["day"] = (t.time_ms.to_numpy() // DAY_MS).astype(np.int64)
    per = t.groupby(["day", "maker"])[mk].agg(["mean", "size"]).reset_index()
    per = per[per["size"] >= n_min]
    days = sorted(per.day.unique())
    corr_rows = []
    if days:
        first = per[per.day == days[0]].set_index("maker")["mean"]
        for i, d in enumerate(days):
            cur = per[per.day == d].set_index("maker")["mean"]
            row = {"day": int(d), "n_makers": int(len(cur))}
            if i > 0:
                prev = per[per.day == days[i - 1]].set_index("maker")["mean"]
                common = cur.index.intersection(prev.index)
                if len(common) >= 8:
                    r, p = spearmanr(cur[common], prev[common])
                    row.update(rho_prev_day=float(r), p_prev_day=float(p), n_common_prev=int(len(common)))
                common = cur.index.intersection(first.index)
                if len(common) >= 8:
                    r, p = spearmanr(cur[common], first[common])
                    row.update(rho_first_day=float(r), p_first_day=float(p), n_common_first=int(len(common)))
            corr_rows.append(row)
    per_rows = per.rename(columns={"mean": "markout_bps", "size": "n_fills"}).to_dict("records")
    return per_rows, corr_rows


def main():
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.calibrate import load_trades
    from analysis.l2 import load_l2, true_mid_markouts
    from analysis.wallets import with_roles
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/live")
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--coins", default="BTC,ETH,SOL,HYPE")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=10)
    args = ap.parse_args()
    raw, res = Path(args.raw), Path(args.results)
    H, D, R, M, MC = [], [], [], [], []
    for c in args.coins.split(","):
        tr, book = load_trades(raw, c), load_l2(raw, c)
        if tr.empty or book.empty or not (res / f"wallets_{c}.csv").exists():
            print(f"{c}: missing trades, book or wallets_{c}.csv"); continue
        feats = pd.read_csv(res / f"wallets_{c}.csv")
        top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(args.top_k))
        tm = true_mid_markouts(with_roles(tr), book, horizons_s=(1, 10))
        h, d = by_hour_and_day(tm, top)
        r = rolling_typing(tm, top, max_k=args.max_k)
        m, mc = maker_persistence(tm)
        for rows, store in ((h, H), (d, D), (r, R), (m, M), (mc, MC)):
            store.extend({"coin": c, **x} for x in rows)
        if d:
            dd = pd.DataFrame(d)
            print(f"{c}: gap by day (benign - toxic top share): " + ", ".join(f"{g:+.2f}±{s:.2f}" for g, s in zip(dd.gap, dd.gap_se))
                  + f"  | days with gap > 2 SE: {(dd.gap > 2 * dd.gap_se).sum()}/{len(dd)}", flush=True)
        if h:
            hh = pd.DataFrame(h)
            print(f"{c}: gap by UTC hour, min/median/max {hh.gap.min():+.2f} / {hh.gap.median():+.2f} / {hh.gap.max():+.2f}; "
                  f"hours with gap > 2 SE: {(hh.gap > 2 * hh.gap_se).sum()}/{len(hh)}", flush=True)
        if r:
            rr = pd.DataFrame(r).groupby("k").agg(gap=("gap", "mean"), holds=("type_holds_bps", "mean"), n=("gap", "size"))
            print(f"{c}: rolling typing, gap by k days ahead: " + ", ".join(f"k={k}: {g:+.2f} (holds {hh:+.1f} bps, n={n})" for k, (g, hh, n) in rr.iterrows()), flush=True)
        if mc:
            cc = pd.DataFrame(mc)
            prev = cc["rho_prev_day"].dropna() if "rho_prev_day" in cc else pd.Series(dtype=float)
            first = cc["rho_first_day"].dropna() if "rho_first_day" in cc else pd.Series(dtype=float)
            print(f"{c}: maker rank persistence, rho vs previous day: " + ", ".join(f"{x:+.2f}" for x in prev)
                  + "; vs first day: " + ", ".join(f"{x:+.2f}" for x in first), flush=True)
    for name, store in (("hour", H), ("day", D), ("rolling", R), ("makers", M), ("makers_corr", MC)):
        pd.DataFrame(store).to_csv(res / f"longwindow_{name}.csv", index=False)
    print("wrote", ", ".join(str(res / f"longwindow_{n}.csv") for n in ("hour", "day", "rolling", "makers", "makers_corr")))


if __name__ == "__main__":
    main()
