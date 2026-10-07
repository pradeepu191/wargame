"""Hyperliquid wallet classification and trade-network analysis (RQ1/RQ2 empirical companion).

Input: a canonical trades frame with columns
    time_ms, coin, side ('B' = taker bought / 'A' = taker sold, or 'buy'/'sell'), px, sz,
    buyer, seller                                   (wallet addresses)
This is what data/record.py writes and what the Zenodo trades stream maps to.

Who is the maker?  On Hyperliquid `side` is the TAKER's side.  So:
    side == buy  -> taker is the buyer, maker is the seller
    side == sell -> taker is the seller, maker is the buyer

Per-wallet features (per coin):
    n_fills, maker_share            fraction of fills where the wallet was the resting side
    two_sided                       fraction of its maker fills on its minority side
                                    (0.5 = perfectly two-sided; 0 = one-directional)
    volume_share
    markout_<d>                     mean signed price move after its MAKER fills, d seconds later,
                                    in bps of px.  Negative = adversely selected.
    realized_spread_<d>             2 s (p - m_{t+d}) / p in bps where s = +1 if the wallet SOLD
                                    (maker on the ask side), using the trade-price mid proxy.
Market-maker flag:  maker_share >= 0.7 and two_sided >= 0.3 and n_fills >= n_min.
These thresholds are a first pass; the paper should report sensitivity.

Trade-pair matrix (Cartea-Chang-Graumans test):
    counts[i, j] = number of trades where wallet i was maker and j was taker, restricted to the
    MM set.  Under random matching, E[counts[i, j]] = maker_fills_i * taker_fills_j / total.
    `avoidance_ratio` = observed / expected over MM-MM pairs.  << 1 means MMs avoid each other.

Leader-switch (turn-taking) statistic:
    Within each window, the wallet with the most maker fills is the "leader".  leader_switch =
    fraction of consecutive windows where the leader changes.  The simulator's turn-taking
    collusion has leader_switch ~ 0.5-0.8 at the period scale; a comparable number on real
    data at the block scale would be the empirical signature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _norm_side(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.map({"b": "buy", "buy": "buy", "a": "sell", "s": "sell", "sell": "sell"})


def with_roles(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["side"] = _norm_side(t["side"])
    t["maker"] = np.where(t.side == "buy", t.seller, t.buyer)
    t["taker"] = np.where(t.side == "buy", t.buyer, t.seller)
    t["maker_sold"] = (t.side == "buy")           # maker was on the ask
    return t.sort_values("time_ms").reset_index(drop=True)


def mid_proxy(t: pd.DataFrame, horizon_ms: int) -> np.ndarray:
    """Trade price `horizon_ms` later (last trade at or before t + h) as a mid proxy.
    With an L2 feed, replace with the true mid; this proxy is what the trades-only data allows."""
    times = t.time_ms.to_numpy()
    px = t.px.to_numpy()
    idx = np.searchsorted(times, times + horizon_ms, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return px[idx]


def wallet_features(trades: pd.DataFrame, horizons_s=(1, 10, 60), n_min: int = 50) -> pd.DataFrame:
    t = with_roles(trades)
    for d in horizons_s:
        m_fut = mid_proxy(t, d * 1000)
        s = np.where(t.maker_sold, 1.0, -1.0)
        t[f"markout_{d}"] = -s * (m_fut - t.px) / t.px * 1e4       # bps, negative = maker lost
        t[f"realized_spread_{d}"] = 2 * s * (t.px - m_fut) / t.px * 1e4
    rows = []
    for w, g in t.groupby("maker"):
        n_maker = len(g)
        n_taker = int((t.taker == w).sum())
        n = n_maker + n_taker
        if n < n_min:
            continue
        sold = g.maker_sold.mean()
        row = {"wallet": w, "n_fills": n, "maker_share": n_maker / n,
               "two_sided": min(sold, 1 - sold),
               "volume": float((g.px * g.sz).sum()),
               }
        for d in horizons_s:
            row[f"markout_{d}"] = float(g[f"markout_{d}"].mean())
            row[f"realized_spread_{d}"] = float(g[f"realized_spread_{d}"].mean())
        rows.append(row)
    f = pd.DataFrame(rows)
    if f.empty:
        return f
    f["volume_share"] = f.volume / f.volume.sum()
    f["is_mm"] = (f.maker_share >= 0.7) & (f.two_sided >= 0.3)
    return f.sort_values("volume", ascending=False).reset_index(drop=True)


def pair_matrix(trades: pd.DataFrame, mm_wallets: set) -> tuple[pd.DataFrame, float]:
    """Observed maker x taker trade counts among MM wallets, and observed/expected ratio."""
    t = with_roles(trades)
    mm = t[t.maker.isin(mm_wallets) | t.taker.isin(mm_wallets)]
    obs = pd.crosstab(mm.maker, mm.taker)
    maker_tot = t.groupby("maker").size()
    taker_tot = t.groupby("taker").size()
    total = len(t)
    pairs = [(i, j) for i in mm_wallets for j in mm_wallets if i != j]
    o = sum(obs.loc[i, j] if (i in obs.index and j in obs.columns) else 0 for i, j in pairs)
    e = sum(maker_tot.get(i, 0) * taker_tot.get(j, 0) / total for i, j in pairs)
    return obs, (o / e if e > 0 else float("nan"))


def leader_switch(trades: pd.DataFrame, mm_wallets: set, window_ms: int = 1000) -> float:
    t = with_roles(trades)
    t = t[t.maker.isin(mm_wallets)]
    if t.empty:
        return float("nan")
    t["win"] = t.time_ms // window_ms
    leader = t.groupby(["win", "maker"]).size().reset_index(name="n")
    leader = leader.sort_values(["win", "n"], ascending=[True, False]).drop_duplicates("win")
    L = leader.maker.to_numpy()
    return float((L[1:] != L[:-1]).mean()) if len(L) > 1 else float("nan")


def arrival_persistence(trades: pd.DataFrame, by: str = "taker") -> float:
    """rho_hat: fraction of consecutive prints (time order) whose taker (or maker) wallet is the
    same.  This is the simulator's wallet_persistence, the dial on which the value of identity
    turns (results/REPLICATION_NOTES.md, identity experiment).  Under random matching the
    baseline is sum_j p_j^2 over wallet activity shares; report both."""
    t = with_roles(trades)
    w = t[by].to_numpy()
    if len(w) < 2:
        return float("nan")
    return float((w[1:] == w[:-1]).mean())


def activity_herfindahl(trades: pd.DataFrame, by: str = "taker") -> float:
    """sum_j p_j^2: the repeat rate random matching would produce (the rho = 0 baseline)."""
    t = with_roles(trades)
    p = t[by].value_counts(normalize=True).to_numpy()
    return float((p ** 2).sum())


def taker_type_dispersion(trades: pd.DataFrame, horizon_s: int = 10, n_min: int = 50) -> pd.DataFrame:
    """Per-taker mean markout (bps, from the MAKER's side: negative = the taker was informed) and
    its cross-sectional dispersion.  The spread of per-wallet markouts, relative to the
    market-wide mean, is the empirical counterpart of the simulator's type dispersion kappa."""
    t = with_roles(trades)
    m_fut = mid_proxy(t, horizon_s * 1000)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["maker_markout"] = -s * (m_fut - t.px) / t.px * 1e4
    g = t.groupby("taker").agg(n=("maker_markout", "size"), markout=("maker_markout", "mean"))
    return g[g.n >= n_min].sort_values("markout")


# ---------------------------------------------------------------- calibration helpers
def collapse_orders(trades: pd.DataFrame, carry: tuple = ()) -> pd.DataFrame:
    """One row per taker ORDER (an aggressive order sweeping several resting orders prints as
    several fills with the same taker, time and side).  Consecutive fills of one sweep are not
    'persistence' in the simulator's sense; the simulator's rho is the order-level repeat rate.
    Keeps the first fill's price, the summed size, the volume-weighted price, and n_fills."""
    t = with_roles(trades)
    # fills of one sweep share the taker, the side and the block timestamp.  The L1 hash is NOT
    # a usable key: it is all zeros for orders filled in their submission block (SCHEMA.md).
    key = ["time_ms", "taker", "side"]
    t["notional"] = t.px * t.sz
    g = t.groupby(key, sort=False)
    spec = dict(time_ms=("time_ms", "first"), coin=("coin", "first"), taker=("taker", "first"),
                side=("side", "first"), px=("px", "first"), sz=("sz", "sum"), n_fills=("px", "size"),
                notional=("notional", "sum"), **{c: (c, "first") for c in carry})
    out = g.agg(**spec).reset_index(drop=True)
    out["vwap"] = out.notional / out.sz
    return out.sort_values("time_ms", kind="stable").reset_index(drop=True)


def split_half_type_persistence(trades: pd.DataFrame, horizon_s: int = 10, n_min: int = 30) -> dict:
    """Zhai's statistic on our sample: per-taker mean markout in the first half of the sample vs the
    second half, Spearman rank correlation over takers with >= n_min fills in both halves.
    A high value means wallet toxicity is a persistent TYPE (the simulator's alpha_j), not noise."""
    from scipy.stats import spearmanr
    t = with_roles(trades)
    m_fut = mid_proxy(t, horizon_s * 1000)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["maker_markout"] = -s * (m_fut - t.px) / t.px * 1e4
    cut = t.time_ms.iloc[len(t) // 2]
    a = t[t.time_ms < cut].groupby("taker").maker_markout.agg(["mean", "size"])
    b = t[t.time_ms >= cut].groupby("taker").maker_markout.agg(["mean", "size"])
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j.size_1 >= n_min) & (j.size_2 >= n_min)]
    if len(j) < 5:
        return {"n_takers": int(len(j)), "rank_corr": float("nan"), "p": float("nan")}
    r, p = spearmanr(j.mean_1, j.mean_2)
    return {"n_takers": int(len(j)), "rank_corr": float(r), "p": float(p)}


def conditional_markout_after_toxic(trades: pd.DataFrame, horizon_s: int = 10, n_min: int = 30,
                                    quantile: float = 0.25) -> dict:
    """The identity-value test in its simplest empirical form.  Classify takers on the FIRST half
    of the sample (bottom `quantile` of per-taker maker-markout = 'toxic').  On the SECOND half,
    compare the maker's markout on a fill whose PREVIOUS order came from a toxic wallet with the
    markout when the previous order came from a non-toxic one.  A negative difference means
    'who printed last' forecasts the toxicity of the next fill -- exactly what WalletEntrant
    exploits, and zero if arrivals are not persistent (rho = 0) or types are not dispersed."""
    t = with_roles(trades)
    m_fut = mid_proxy(t, horizon_s * 1000)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["maker_markout"] = -s * (m_fut - t.px) / t.px * 1e4
    cut = t.time_ms.iloc[len(t) // 2]
    first = t[t.time_ms < cut].groupby("taker").maker_markout.agg(["mean", "size"])
    first = first[first["size"] >= n_min]
    if len(first) < 8:
        return {"n_classified": int(len(first)), "diff_bps": float("nan"), "after_toxic": float("nan"),
                "after_benign": float("nan"), "n_after_toxic": 0}
    toxic = set(first[first["mean"] <= first["mean"].quantile(quantile)].index)
    second = t[t.time_ms >= cut].reset_index(drop=True)
    orders = collapse_orders(second, carry=("maker_markout",))     # markout of each order's first fill
    prev_taker = orders.taker.shift(1)
    after_toxic = prev_taker.isin(toxic).to_numpy()
    mo = orders.maker_markout.to_numpy()
    known = prev_taker.notna().to_numpy()
    a, b = mo[after_toxic & known], mo[~after_toxic & known]
    return {"n_classified": int(len(first)), "n_toxic": len(toxic), "after_toxic": float(np.mean(a)) if len(a) else float("nan"),
            "after_benign": float(np.mean(b)) if len(b) else float("nan"),
            "diff_bps": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else float("nan"),
            "n_after_toxic": int(len(a)), "n_after_benign": int(len(b))}


def conditional_markout_two_signals(trades: pd.DataFrame, horizon_s: int = 10, n_min: int = 30,
                                    quantile: float = 0.25) -> dict:
    """Does wallet identity add regime information beyond what the anonymous tape already shows?

    At the moment order k arrives, an anonymous quoter knows the previous print's price and side
    and the current price, so it can see whether the market has since moved AGAINST the previous
    maker (anon signal: adverse = 1).  An identity-aware quoter additionally knows whether the
    previous taker is a toxic wallet (classified on the first half).  We measure the maker's
    markout on order k's fill under the 2 x 2 split and report
        diff_id     = E[mo | toxic] - E[mo | benign]                       (identity alone)
        diff_anon   = E[mo | adverse] - E[mo | not adverse]                 (anonymous tape alone)
        diff_id_given_anon = mean over anon strata of (E[mo | toxic, s] - E[mo | benign, s])
                                                                            (identity's increment)
    All on the second half of the sample.  If diff_id_given_anon ~ 0 the anonymous tape already
    carries the regime and identity is redundant for a quoter -- the simulator's regime result."""
    t = with_roles(trades)
    m_fut = mid_proxy(t, horizon_s * 1000)
    s = np.where(t.maker_sold, 1.0, -1.0)
    t["maker_markout"] = -s * (m_fut - t.px) / t.px * 1e4
    cut = t.time_ms.iloc[len(t) // 2]
    first = t[t.time_ms < cut].groupby("taker").maker_markout.agg(["mean", "size"])
    first = first[first["size"] >= n_min]
    out = {"n_classified": int(len(first))}
    if len(first) < 8:
        return {**out, "diff_id_bps": np.nan, "diff_anon_bps": np.nan, "diff_id_given_anon_bps": np.nan}
    toxic = set(first[first["mean"] <= first["mean"].quantile(quantile)].index)
    second = t[t.time_ms >= cut].reset_index(drop=True)
    o = collapse_orders(second, carry=("maker_markout",))
    prev_px, prev_side, prev_taker = o.px.shift(1), o.side.shift(1), o.taker.shift(1)
    known = prev_taker.notna().to_numpy()
    # previous maker sold if the previous taker bought; adverse to that maker if price rose since
    prev_maker_sold = (prev_side == "buy").to_numpy()
    moved = (o.px.to_numpy() - prev_px.to_numpy())
    adverse = np.where(prev_maker_sold, moved > 0, moved < 0)
    tox = prev_taker.isin(toxic).to_numpy()
    mo = o.maker_markout.to_numpy()
    def m(mask):
        mask = mask & known
        return float(mo[mask].mean()) if mask.sum() > 50 else np.nan
    out["mo_toxic"], out["mo_benign"] = m(tox), m(~tox)
    out["mo_adverse"], out["mo_not_adverse"] = m(adverse), m(~adverse)
    out["diff_id_bps"] = out["mo_toxic"] - out["mo_benign"]
    out["diff_anon_bps"] = out["mo_adverse"] - out["mo_not_adverse"]
    incr, wts = [], []
    for sgn in (True, False):
        a, b = m(tox & (adverse == sgn)), m(~tox & (adverse == sgn))
        if np.isfinite(a) and np.isfinite(b):
            w = int(((adverse == sgn) & known).sum())
            incr.append((a - b) * w); wts.append(w)
    out["diff_id_given_anon_bps"] = float(sum(incr) / sum(wts)) if wts else np.nan
    out["p_adverse"] = float(adverse[known].mean())
    out["p_toxic_given_adverse"] = float(tox[adverse & known].mean()) if (adverse & known).sum() else np.nan
    out["p_toxic_given_not_adverse"] = float(tox[~adverse & known].mean()) if (~adverse & known).sum() else np.nan
    return out
