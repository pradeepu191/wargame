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
