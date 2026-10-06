"""Strategic entrant policies (RQ1).

An entrant joins a market of frozen, trained incumbents and must decide whether the
prevailing spread is rent (undercut it) or the price of toxic flow (do not).  All
policies here see only PUBLIC information: the incumbents' posted quotes and the tape
of prints (price, side, and the later mark).  None reads incumbent Q-tables or inventory.

    CompetitiveEntrant     quotes the exact Glosten--Milgrom zero-profit half-spread h^C.
                           Ignores incumbents entirely.  The textbook benchmark.
    UndercutEntrant        quotes one tick inside the best incumbent quote (floored at h_min).
                           Myopic exploitation; an upper bound if incumbents do not punish.
    MarkoutEntrant         estimates incumbent rent from public prints and undercuts only
                           when the estimate is positive.  This is the RQ1 policy.

Markout rent estimate.  Each print at price p with taker side s (+1 buy) and the next
period's mid m' gives realized spread 2 s (p - m').  In this stationary market the mid
is constant so the markout horizon is one period and m' = m.  The realized spread of the
incumbents' fills is therefore 2 s (p - m) = the full quoted half-spread x 2, minus the
informed component which only shows up in expectation through V.  We cannot see V on the
public tape, so the entrant estimates toxicity from its OWN fills' PnL (which it does see)
and rent from the gap between the incumbents' quoted half-spread and the competitive
half-spread implied by its toxicity estimate.  Concretely, with alpha_hat the fraction of
its own fills that lost money, and h^C(alpha_hat) from the closed-form benchmark table,

    rent_hat = best incumbent half-spread - h^C(alpha_hat)

It undercuts by one tick when rent_hat >= margin, else quotes h^C(alpha_hat).  Before it
has k_min fills it quotes h^C at the prior alpha_0 (a conservative prior).

This is deliberately simple; the point of RQ1 is whether even this beats the competitive
benchmark, and by how much, as a function of true alpha.
"""
from __future__ import annotations

import numpy as np

from .agents import Agent
from .benchmarks import BenchmarkTable, competitive_and_monopoly
from .env import GlostenMilgromEnv, MarketConfig


class CompetitiveEntrant(Agent):
    def __init__(self, hC: int):
        self.hC = hC

    def act(self, obs, i):
        return self.hC


class UndercutEntrant(Agent):
    def __init__(self, h_min: int = 1):
        self.h_min = h_min

    def act(self, obs, i):
        hs = obs["last_half_spreads"]
        best = min(int(hs[j]) for j in range(len(hs)) if j != i)
        return max(self.h_min, best - 1)


class MarkoutEntrant(Agent):
    """Rent-or-toxicity inference from public prints and own fills."""

    def __init__(self, hC_by_alpha, alpha_prior: float = 0.3, k_min: int = 20,
                 margin: int = 2, window: int = 200, h_min: int = 1):
        self.hC_by_alpha = hC_by_alpha          # callable alpha -> h^C
        self.alpha_prior = alpha_prior
        self.k_min = k_min
        self.margin = margin
        self.window = window
        self.h_min = h_min
        self.reset_memory()

    def reset_memory(self):
        self.fills = []        # 1 if the fill lost money (informed), 0 otherwise
        self.last_decision = None

    def reset(self):
        pass                    # memory persists across episodes: the entrant keeps learning the market

    @property
    def alpha_hat(self) -> float:
        if len(self.fills) < self.k_min:
            return self.alpha_prior
        w = self.fills[-self.window:]
        return float(np.mean(w))

    def observe_fill(self, pnl: float):
        self.fills.append(1 if pnl < 0 else 0)

    def act(self, obs, i):
        hs = obs["last_half_spreads"]
        best = min(int(hs[j]) for j in range(len(hs)) if j != i)
        hC = int(self.hC_by_alpha(self.alpha_hat))
        rent_hat = best - hC
        if rent_hat >= self.margin:
            a = max(self.h_min, best - 1)
            self.last_decision = "undercut"
        else:
            a = hC
            self.last_decision = "competitive"
        return a


def hC_function(base_cfg: MarketConfig):
    """Returns alpha -> h^C for the given market, cached on a grid."""
    grid = np.linspace(0.0, 1.0, 101)
    table = []
    for a in grid:
        cfg = MarketConfig(**{**base_cfg.__dict__, "alpha": float(a)})
        env = GlostenMilgromEnv(cfg)
        hC, _, _, _ = competitive_and_monopoly(env, 0.5)
        table.append(hC)
    table = np.array(table)

    def f(alpha):
        return int(table[int(round(min(max(alpha, 0.0), 1.0) * 100))])
    return f
