"""Exact competitive and joint-monopoly benchmarks on the tick grid.

For a belief mu and symmetric half-spread h (ticks), expected one-period
profit of the *whole* market (all MMs quoting h) is

    Pi(h; mu) = mu * [ p_buy(V_H) (a - V_H) + p_sell(V_H) (V_H - b) ]
              + (1-mu) * [ p_buy(V_L) (a - V_L) + p_sell(V_L) (V_L - b) ]

with a = mid + h*tick, b = mid - h*tick.  Then

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
    c = env.cfg
    mid = mu * c.v_high + (1 - mu) * c.v_low
    hs = np.arange(1, c.max_half_spread + 1)
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
