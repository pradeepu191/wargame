"""Calibration pipeline: recorder parquet -> the numbers the simulator's identity result turns on.

Reads data/raw/live/trades_<COIN>_<YYYYMMDDHH>.parquet (data/record.py output; columns time_ms,
coin, side, px, sz, tid, hash, buyer, seller) and writes, per coin, one row of statistics to
results/calibration.csv plus a per-wallet table results/wallets_<COIN>.csv.

What it reports and which simulator dial it calibrates:
    descriptives        hours, prints, orders, prints/min, USD volume, unique takers / makers
    markout_<d>_bps     mean maker markout (trade-price mid proxy) at d = 1, 10, 60 s     -> alpha level
    rho_order           P(next ORDER is from the same taker as the last) vs the random-matching
                        baseline sum_j p_j^2                                              -> rho
    rho_fill            the same at fill level (mechanical: one sweep = many fills; reported
                        only so nobody mistakes it for rho)
    type_rank_corr      split-half Spearman correlation of per-taker markouts (Zhai)      -> kappa (dispersion
    type_sd_bps         cross-sectional sd of per-taker mean markouts                        is a persistent type?)
    cond_diff_bps       maker markout on the fill after a toxic wallet's order minus after a
                        benign one, toxic classified on the first half                   -> the value-of-identity
                                                                                            signature itself
    n_mm, mm_volume_share, mm_avoidance_ratio, mm_leader_switch_1s / _10s                -> incumbents (RQ2/RQ3)

Everything uses prior observations only where a forecast is implied (classification on the first
half, measurement on the second).  Trade-price mid proxy until the L2 book is recorded.

Usage:
  python analysis/calibrate.py [--raw data/raw/live] [--coins BTC,ETH] [--out results]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # run from anywhere, no install needed
from analysis.l2 import book_summary, load_l2, sweep_decomposition, true_mid_markouts
from analysis.makers import maker_summary, maker_table
from analysis.wallets import (activity_herfindahl, arrival_persistence, collapse_orders,
                              conditional_markout_after_toxic, conditional_markout_two_signals, leader_switch,
                              mid_proxy, pair_matrix, split_half_type_persistence, wallet_features, with_roles)


def load_trades(raw: Path, coin: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / f"trades_{coin}_*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["buyer", "seller"]).drop_duplicates(["tid"] if "tid" in df.columns else None)
    return df.sort_values("time_ms", kind="stable").reset_index(drop=True)


def calibrate_coin(trades: pd.DataFrame, horizons=(1, 10, 60), n_min: int = 30,
                   book: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    t = with_roles(trades)
    orders = collapse_orders(t)
    hours = (t.time_ms.max() - t.time_ms.min()) / 3.6e6
    row = {
        "coin": str(t.coin.iloc[0]), "hours": hours, "prints": len(t), "orders": len(orders),
        "prints_per_min": len(t) / max(hours * 60, 1e-9), "orders_per_min": len(orders) / max(hours * 60, 1e-9),
        "usd_volume": float((t.px * t.sz).sum()), "median_order_usd": float(orders.notional.median()),
        "n_takers": int(t.taker.nunique()), "n_makers": int(t.maker.nunique()),
        "px_mean": float(t.px.mean()),
    }
    feats = wallet_features(t, horizons_s=horizons, n_min=n_min)
    # market-wide maker markouts straight from the fills (equal-weighted)
    for d in horizons:
        m_fut = mid_proxy(t, d * 1000)
        s = np.where(t.maker_sold, 1.0, -1.0)
        row[f"markout_{d}_bps"] = float((-s * (m_fut - t.px) / t.px * 1e4).mean())
    # arrival persistence at the order level (the simulator's rho) and its random-matching baseline
    tk = orders.taker.to_numpy()
    row["rho_order"] = float((tk[1:] == tk[:-1]).mean()) if len(tk) > 1 else np.nan
    p_ord = orders.taker.value_counts(normalize=True).to_numpy()
    row["rho_order_baseline"] = float((p_ord ** 2).sum())
    row["rho_fill"] = arrival_persistence(t, by="taker")
    row["rho_fill_baseline"] = activity_herfindahl(t, by="taker")
    tp = split_half_type_persistence(t, horizon_s=10, n_min=n_min)
    row["type_rank_corr"], row["type_rank_corr_p"], row["type_n_takers"] = tp["rank_corr"], tp["p"], tp["n_takers"]
    takers = t.groupby("taker").size()
    big = takers[takers >= n_min].index
    m_fut = mid_proxy(t, 10_000)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["mo10"] = -s * (m_fut - t.px) / t.px * 1e4
    per_taker = t[t.taker.isin(big)].groupby("taker").mo10.mean()
    row["type_sd_bps"] = float(per_taker.std()) if len(per_taker) > 1 else float("nan")
    row["type_q25_bps"], row["type_q75_bps"] = (float(per_taker.quantile(0.25)), float(per_taker.quantile(0.75))) if len(per_taker) > 3 else (np.nan, np.nan)
    cm = conditional_markout_after_toxic(t, horizon_s=10, n_min=n_min)
    row["cond_after_toxic_bps"], row["cond_after_benign_bps"], row["cond_diff_bps"] = cm["after_toxic"], cm["after_benign"], cm["diff_bps"]
    row["cond_n_after_toxic"] = cm["n_after_toxic"]
    two = conditional_markout_two_signals(t, horizon_s=10, n_min=n_min)
    for k in ("diff_id_bps", "diff_anon_bps", "diff_id_given_anon_bps", "p_adverse",
              "p_toxic_given_adverse", "p_toxic_given_not_adverse"):
        row[f"two_{k}"] = two.get(k, np.nan)
    if book is not None and len(book):
        # true-mid markouts (bid-ask bounce removed) and the book itself
        row.update(book_summary(book))
        tm = true_mid_markouts(t, book, horizons_s=horizons)
        row["l2_fills_marked"] = len(tm)
        row["effective_spread_bps"] = float(tm.effective_bps.mean())
        for d in horizons:
            row[f"mid_markout_{d}_bps"] = float(tm[f"markout_{d}_bps"].mean())
            row[f"mid_realized_{d}_bps"] = float(tm[f"realized_{d}_bps"].mean())
            row[f"proxy_bias_{d}_bps"] = row[f"markout_{d}_bps"] - row[f"mid_markout_{d}_bps"]
        row.update(sweep_decomposition(tm, horizon_s=10))
        # per-maker realized spread on marked fills (RQ3's empirical anchor); feats is the full-window
        # classifier, so the top set is chosen on activity, not on the outcome
        row.update(maker_summary(tm, feats, horizon_s=10, top_k=10, n_min=100, n_half=50))
        row["_makers"] = maker_table(tm, horizon_s=10, n_min=100)
        # the identity split against the true mid
        mm_tm = tm.rename(columns={"markout_10_bps": "maker_markout"})
        cut = mm_tm.time_ms.iloc[len(mm_tm) // 2]
        first = mm_tm[mm_tm.time_ms < cut].groupby("taker").maker_markout.agg(["mean", "size"])
        first = first[first["size"] >= n_min]
        if len(first) >= 8:
            toxic = set(first[first["mean"] <= first["mean"].quantile(0.25)].index)
            o = collapse_orders(mm_tm[mm_tm.time_ms >= cut].reset_index(drop=True), carry=("maker_markout",))
            prev = o.taker.shift(1)
            known = prev.notna().to_numpy(); tox = prev.isin(toxic).to_numpy(); mo = o.maker_markout.to_numpy()
            row["mid_cond_after_toxic_bps"] = float(mo[tox & known].mean()) if (tox & known).sum() > 50 else np.nan
            row["mid_cond_after_benign_bps"] = float(mo[~tox & known].mean()) if (~tox & known).sum() > 50 else np.nan
            row["mid_cond_diff_bps"] = row["mid_cond_after_toxic_bps"] - row["mid_cond_after_benign_bps"]
    if len(feats):
        mm = set(feats[feats.is_mm].wallet)
        row["n_mm"] = len(mm)
        row["mm_volume_share"] = float(feats[feats.is_mm].volume_share.sum())
        row["mm_maker_fill_share"] = float(t.maker.isin(mm).mean())
        if len(mm) >= 2:
            _, row["mm_avoidance_ratio"] = pair_matrix(t, mm)
            row["mm_leader_switch_1s"] = leader_switch(t, mm, window_ms=1000)
            row["mm_leader_switch_10s"] = leader_switch(t, mm, window_ms=10_000)
            # the turn-taking test needs a small, dominant set and a null: top-10 MM wallets by volume,
            # leader-switch vs the same statistic with maker labels shuffled (independence null)
            top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(10))
            row["mm_top10_volume_share"] = float(feats[feats.wallet.isin(top)].volume_share.sum())
            row["mm_top10_leader_switch_10s"] = leader_switch(t, set(top), window_ms=10_000)
            rng = np.random.default_rng(0)
            tt = t[t.maker.isin(top)]
            nulls = []
            for _ in range(10):
                sh = tt.copy(); sh["maker"] = rng.permutation(sh.maker.to_numpy())
                sh["buyer"] = np.where(sh.side == "buy", sh.taker, sh.maker); sh["seller"] = np.where(sh.side == "buy", sh.maker, sh.taker)
                nulls.append(leader_switch(sh, set(top), window_ms=10_000))
            row["mm_top10_leader_switch_10s_null"] = float(np.mean(nulls))
            _, row["mm_top10_avoidance_ratio"] = pair_matrix(t, set(top))
            for d in horizons:
                row[f"mm_markout_{d}_bps"] = float(feats[feats.is_mm][f"markout_{d}"].mean())
                row[f"nonmm_markout_{d}_bps"] = float(feats[~feats.is_mm][f"markout_{d}"].mean()) if (~feats.is_mm).any() else np.nan
    return row, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/live")
    ap.add_argument("--coins", default=None, help="comma list; default: every coin with files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-min", type=int, default=30)
    args = ap.parse_args()
    raw = Path(args.raw)
    coins = args.coins.split(",") if args.coins else sorted({Path(f).name.split("_")[1] for f in glob.glob(str(raw / "trades_*.parquet"))})
    if not coins:
        raise SystemExit(f"no trades_*.parquet under {raw}; run data/record.py first")
    rows = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in coins:
        tr = load_trades(raw, c)
        if tr.empty:
            print(f"{c}: no data"); continue
        book = load_l2(raw, c)
        row, feats = calibrate_coin(tr, n_min=args.n_min, book=book if len(book) else None)
        makers = row.pop("_makers", None)
        rows.append(row)
        feats.to_csv(out / f"wallets_{c}.csv", index=False)
        if makers is not None and len(makers):
            makers.to_csv(out / f"makers_{c}.csv", index=False)
        print(f"{c}: {row['hours']:.1f} h, {row['prints']} prints, {row['orders']} orders, "
              f"rho_order={row['rho_order']:.3f} (baseline {row['rho_order_baseline']:.3f}), "
              f"markout_10s={row['markout_10_bps']:+.2f} bps, type rank corr={row['type_rank_corr']:.2f}, "
              f"cond diff={row['cond_diff_bps']:+.2f} bps [id|anon-signal: {row['two_diff_id_given_anon_bps']:+.2f}, "
              f"anon alone: {row['two_diff_anon_bps']:+.2f}], MMs={row.get('n_mm', 0)}"
              + (f" | L2: {row['l2_coverage_h']:.1f} h in {row['l2_sessions']} session(s), gap {row['l2_median_gap_s']:.3f} s, "
                 f"{row['l2_fills_marked']} fills marked, quoted {row['spread_bps_median']:.2f} / effective "
                 f"{row['effective_spread_bps']:.2f} bps, true-mid markout_1/10/60 {row['mid_markout_1_bps']:+.2f} / "
                 f"{row['mid_markout_10_bps']:+.2f} / {row['mid_markout_60_bps']:+.2f} (proxy bias_10 {row['proxy_bias_10_bps']:+.2f}), "
                 f"cond diff {row.get('mid_cond_diff_bps', float('nan')):+.2f} | touch share {row['touch_share']:.2f}, "
                 f"effective touch/walk/order {row['effective_touch_bps']:.2f} / {row['effective_walk_bps']:.2f} / "
                 f"{row['effective_order_bps']:.2f}, markout_10 touch/walk {row['mid_markout_10_touch_bps']:+.2f} / "
                 f"{row['mid_markout_10_walk_bps']:+.2f}, {row['fills_per_order']:.2f} fills/order, "
                 f"{row['walk_order_share']:.2f} of orders walk"
                 if "spread_bps_median" in row else "")
              + (f" | MAKERS: all {row['mk_all_markout_bps']:+.2f}±{row['mk_all_se_bps']:.2f}; top-{row.get('mk_top_k', 0)} MM "
                 f"{row.get('mk_top_markout_bps', float('nan')):+.2f}±{row.get('mk_top_se_bps', float('nan')):.2f} "
                 f"({row.get('mk_top_fill_share', float('nan')):.2f} of fills, touch {row.get('mk_top_touch_share', float('nan')):.2f}); "
                 f"other MM {row.get('mk_othermm_markout_bps', float('nan')):+.2f}; non-MM {row.get('mk_nonmm_markout_bps', float('nan')):+.2f}; "
                 f"at the touch top/rest {row.get('mk_top_touch_markout_bps', float('nan')):+.2f} / {row.get('mk_rest_touch_markout_bps', float('nan')):+.2f}; "
                 f"{row.get('mk_n_makers', 0)} makers>=100 fills, {row.get('mk_positive_share', float('nan')):.2f} positive "
                 f"({row.get('mk_positive_fill_share', float('nan')):.2f} of fills), {row.get('mk_sig_positive_n', 0)} significantly; "
                 f"split-half rank corr {row.get('mk_split_rank_corr', float('nan')):+.2f} (p={row.get('mk_split_rank_corr_p', float('nan')):.3f}, "
                 f"n={row.get('mk_split_n_makers', 0)})"
                 if "mk_all_markout_bps" in row else ""), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.set_index("coin").T.to_string())
    print("wrote", out / "calibration.csv")


if __name__ == "__main__":
    main()
