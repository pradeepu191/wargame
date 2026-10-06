"""Exact competitive and joint-monopoly benchmarks on the tick grid.

For a belief mu and symmetric half-spread h (ticks), expected one-period
profit of the *whole* market (all MMs quoting h) is, with x = h*tick,

  fixed edge:        Pi(h; mu) = mu * [ p_buy(V_H) (a - V_H) + p_sell(V_H) (V_H - b) ]
                               + (1-mu) * [ p_buy(V_L) (a - V_L) + p_sell(V_L) (V_L - b) ]
  exponential edge:  Pi(h)     = (1-alpha) * 2 x Phibar(x / sigma_L)  -  alpha * m * exp(-x / m)
                     (uninformed gain x on each side; an informed trader trades w.p. exp(-x/m)
                      and, by memorylessness, costs the MM m in expectation when it does)

with a = mid + x, b = mid - x.  Then

    h^C(mu) = min { h : Pi(h; mu) >= 0 }   (numerically: >= -1e-9)        (Glosten--Milgrom zero-profit quote,
                                                 rounded up to the grid; Bertrand
                                                 undercutting stops here)
    h^M(mu) = argmax_h Pi(h; mu)                 (joint monopoly)
    pi^C = Pi(h^C)/N,  pi^M = Pi(h^M)/N          (per-MM, symmetric split)

Collusion index (Calvano et al. 2020):  Delta = (pi_bar - pi^C) / (pi^M - pi^C).

Benchmarks are precomputed on a mu-grid and looked up by nearest neighbour,
which is exact enough for reporting and ~100x faster than recomputing.
"""
from __future__ import annotations

import numpy as np

from .env import GlostenMilgromEnv, MarketConfig


def expected_profit_by_half_spread(env: GlostenMilgromEnv, mu: float) -> np.ndarray:
    """Pi(h; mu) for h = 1..max_half_spread (market-wide, one period)."""
    from scipy.stats import norm
    c = env.cfg
    alpha = getattr(env, "alpha_eff", c.alpha)      # activity-weighted mean type under heterogeneous takers
    hs = np.arange(1, c.max_half_spread + 1)
    if c.edge_dist == "exponential":
        x = hs * c.tick
        return (1 - alpha) * 2 * x * norm.sf(x / c.sigma_L) - alpha * c.edge_mean * np.exp(-x / c.edge_mean)
    mid = mu * c.v_high + (1 - mu) * c.v_low
    out = np.empty(len(hs))
    for k, h in enumerate(hs):
        a, b = mid + h * c.tick, mid - h * c.tick
        pbh, psh = env._p_buy_sell_given_V(a, b, mid, c.v_high)
        pbl, psl = env._p_buy_sell_given_V(a, b, mid, c.v_low)
        out[k] = (mu * (pbh * (a - c.v_high) + psh * (c.v_high - b))
                  + (1 - mu) * (pbl * (a - c.v_low) + psl * (c.v_low - b)))
    return out


def competitive_and_monopoly(env: GlostenMilgromEnv, mu: float):
    """Returns (h_C, pi_C, h_M, pi_M) per MM at belief mu."""
    c = env.cfg
    Pi = expected_profit_by_half_spread(env, mu)
    nonneg = np.flatnonzero(Pi >= -1e-9)   # tolerance: zero-profit quote sits at Pi == 0
    hC = int(nonneg[0] + 1) if len(nonneg) else c.max_half_spread
    hM = int(np.argmax(Pi) + 1)
    return hC, Pi[hC - 1] / c.n_mm, hM, Pi[hM - 1] / c.n_mm


class BenchmarkTable:
    """Nearest-neighbour lookup of benchmarks on a mu grid."""

    def __init__(self, env: GlostenMilgromEnv, n_grid: int = 201):
        self.mu_grid = np.linspace(0.0, 1.0, n_grid)
        rows = [competitive_and_monopoly(env, m) for m in self.mu_grid]
        self.hC = np.array([r[0] for r in rows])
        self.piC = np.array([r[1] for r in rows])
        self.hM = np.array([r[2] for r in rows])
        self.piM = np.array([r[3] for r in rows])

    def _idx(self, mu: float) -> int:
        n = len(self.mu_grid) - 1
        i = int(round(mu * n))
        return 0 if i < 0 else n if i > n else i

    def lookup(self, mu: float):
        i = self._idx(mu)
        return int(self.hC[i]), float(self.piC[i]), int(self.hM[i]), float(self.piM[i])


def collusion_index(mean_profit_per_mm: float, piC: float, piM: float) -> float:
    denom = piM - piC
    return float("nan") if denom <= 1e-12 else (mean_profit_per_mm - piC) / denom


# ---------------------------------------------------------------- heterogeneous takers
def profit_curve(cfg: MarketConfig, alpha: float) -> np.ndarray:
    """Market-wide one-period expected profit Pi(h; alpha) for h = 1..K, closed form
    (exponential edge only).  Linear in alpha, so the curve for a mixture of types is the
    mixture of curves."""
    if cfg.edge_dist != "exponential":
        raise ValueError("profit_curve is closed-form for edge_dist='exponential' only")
    from scipy.stats import norm
    x = np.arange(1, cfg.max_half_spread + 1) * cfg.tick
    return (1 - alpha) * 2 * x * norm.sf(x / cfg.sigma_L) - alpha * cfg.edge_mean * np.exp(-x / cfg.edge_mean)


def type_benchmarks(cfg: MarketConfig, alpha: float):
    """(h^C, Pi^C, h^M, Pi^M) -- market-wide, against a taker of type alpha."""
    Pi = profit_curve(cfg, alpha)
    nonneg = np.flatnonzero(Pi >= -1e-9)
    hC = int(nonneg[0] + 1) if len(nonneg) else cfg.max_half_spread
    hM = int(np.argmax(Pi) + 1)
    return hC, float(Pi[hC - 1]), hM, float(Pi[hM - 1])


def identity_monopoly_profit(env: GlostenMilgromEnv, rho: float | None = None) -> float:
    """Upper bound for an identity-aware sole quoter: it knows every wallet's type and the
    persistence rho, forecasts the next arrival's type as rho*alpha_j + (1-rho)*alpha_eff when
    wallet j printed last period, and posts that type's monopoly quote.  Market-wide, per
    period, ignoring the (1-rho)-weighted periods after a no-trade (treated at alpha_eff).
    Equals Pi^M when rho = 0 or when types are homogeneous."""
    c = env.cfg
    if env.wallet_alpha is None:
        return type_benchmarks(c, c.alpha)[3]
    rho = c.wallet_persistence if rho is None else rho
    a_next = rho * env.wallet_alpha + (1 - rho) * env.alpha_eff
    best = np.array([type_benchmarks(c, float(a))[3] for a in a_next])
    return float(env.wallet_weights @ best)
