"""One-tick book: the queue-position game (RQ3).

What the Hyperliquid book data showed (results/REPLICATION_NOTES.md): the quoted spread on
BTC/ETH/SOL is one tick essentially always, and a resting fill is under water within ten seconds
on average.  On such a book nobody can undercut, so rent -- if any -- can only come from WHICH
fills a maker gets.  This environment makes that the whole game.

  * The spread is fixed at `half_spread` ticks.  Default: one tick above the population break-even
    h^C(abar) -- a coarse tick leaves the inside slightly supra-competitive (the Cartea-Chang-Penalva
    point), so resting has positive average value and the queue shares it.  At h^C exactly, resting
    is a zero-mean gamble and identity-blind learners stay out.  Makers do not choose a price.
  * Each period every maker chooses REST (1) or OUT (0).  Resting makers form a queue in time
    priority: a maker that keeps resting keeps its place; one that steps out and returns joins at
    the BACK; the maker that gets filled has used its order and rejoins at the back if it rests
    again.  Fills therefore rotate among those who stay -- a round-robin -- and stepping out to
    dodge a toxic spell costs the place in line.
  * One taker per period, drawn exactly as in GlostenMilgromEnv (wallet types, toxicity regime,
    mark lag); it hits the FRONT of the queue for one unit.  If nobody is resting there is no
    trade and no print.
  * Public information is the same as in GlostenMilgromEnv (print now, mark later) plus the
    depth at the inside (how many are resting), which an L2 feed shows.  A maker also knows its
    own rank.  Nobody sees who else is in the queue.

Expected profit of a fill at half-spread x against a taker of type alpha is
    (1 - q) x - q mu,   q = alpha e^{-x/mu} / (alpha e^{-x/mu} + (1 - alpha) 2 Phibar(x/sigma))
which at the population break-even is ~0 on average, positive in the calm regime and negative in
the toxic one: all the value is in selection.  `selection_benchmarks` returns those numbers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .benchmarks import type_benchmarks
from .env import GlostenMilgromEnv, MarketConfig, _norm_sf


def fill_value(cfg: MarketConfig, half_spread: int, alpha: float) -> float:
    """E[maker PnL | fill] at half_spread ticks against type alpha (exponential edge)."""
    x = half_spread * cfg.tick
    e = float(np.exp(-x / cfg.edge_mean))
    u = float(2 * _norm_sf(x / cfg.sigma_L))
    q = alpha * e / (alpha * e + (1 - alpha) * u)
    return (1 - q) * x - q * cfg.edge_mean


def fill_prob(cfg: MarketConfig, half_spread: int, alpha: float) -> float:
    x = half_spread * cfg.tick
    return alpha * float(np.exp(-x / cfg.edge_mean)) + (1 - alpha) * float(2 * _norm_sf(x / cfg.sigma_L))


class QueueEnv(GlostenMilgromEnv):
    """Call reset(); then step(actions) with actions[i] in {0 = out, 1 = rest}."""

    def __init__(self, cfg: MarketConfig, half_spread: Optional[int] = None, ticks_above_breakeven: int = 1):
        super().__init__(cfg)
        if half_spread is None:
            hC, _, _, _ = type_benchmarks(cfg, self.alpha_eff)
            half_spread = hC + ticks_above_breakeven  # a coarse tick leaves the inside above break-even
        self.h = int(half_spread)
        self.queue: list[int] = []                   # maker ids, front first
        self.last_half_spreads = np.full(cfg.n_mm, self.h, dtype=np.int64)

    def reset(self, V=None) -> dict:
        obs = super().reset(V)
        self.queue = []
        self.last_half_spreads[:] = self.h
        return self.observation()

    def observation(self) -> dict:
        obs = super().observation()
        n = self.cfg.n_mm
        rank = np.full(n, -1, dtype=np.int64)
        for r, i in enumerate(self.queue):
            rank[i] = r
        obs["rank"] = rank                           # -1 = out; 0 = front
        obs["n_resting"] = len(self.queue)
        obs["last_half_spreads"] = np.full(n, self.h, dtype=np.int64)   # the fixed one-tick spread
        return obs

    def step(self, actions) -> tuple[dict, np.ndarray, bool, dict]:
        c = self.cfg
        n = c.n_mm
        a = [int(x) for x in actions]
        assert len(a) == n and all(x in (0, 1) for x in a)
        # ---- queue update: leavers drop out, newcomers join at the back, stayers keep their place
        self.queue = [i for i in self.queue if a[i] == 1]
        for i in range(n):
            if a[i] == 1 and i not in self.queue:
                self.queue.append(i)
        mid = self.mid
        best_ask = mid + self.h * c.tick
        best_bid = mid - self.h * c.tick

        # ---- taker arrival (identical draws to the spread game)
        wallet, informed, event = self._draw_taker(best_ask, best_bid, mid)
        if not self.queue:
            event = "noliq"                          # nobody at the inside: no trade, no print, and the
                                                     # absence of a print says nothing about the regime

        # ---- matching: front of the queue, then it rejoins at the back if it keeps resting
        reward = np.zeros(n)
        filled = -1
        price = float("nan")
        if event in ("buy", "sell"):
            filled = self.queue.pop(0)
            self.queue.append(filled)                # its order is consumed; a fresh one goes to the back
            if event == "buy":
                price = best_ask
                reward[filled] += (best_ask - self.V) + c.maker_rebate
                self.inventory[filled] -= 1
            else:
                price = best_bid
                reward[filled] += (self.V - best_bid) + c.maker_rebate
                self.inventory[filled] += 1

        V_t = self.V
        self._publish(event, self.h, price, wallet, V_t)
        if c.redraw_v_each_period:
            self.V = self._draw_v()
            self.mu = 0.5
        self.t += 1
        done = self.t >= c.horizon
        info = {"event": event, "price": price, "filled": filled, "informed": informed, "taker": wallet,
                "regime": self.regime, "mid": mid, "best_ask": best_ask, "best_bid": best_bid,
                "quoted_spread": best_ask - best_bid, "V": self.V, "V_t": V_t,
                "n_resting": len(self.queue), "mu_prev": 0.5}
        return self.observation(), reward, done, info


def selection_benchmarks(env: QueueEnv) -> dict:
    """Per-fill values at the fixed spread: population average, and by regime when there is one."""
    c, h = env.cfg, env.h
    out = {"half_spread": h, "fill_value_avg": fill_value(c, h, env.alpha_eff),
           "fill_prob_avg": fill_prob(c, h, env.alpha_eff)}
    if env.regime_alpha is not None:
        a0, a1 = float(env.regime_alpha[0]), float(env.regime_alpha[1])
        out.update({"alpha_calm": a0, "alpha_toxic": a1,
                    "fill_value_calm": fill_value(c, h, a0), "fill_value_toxic": fill_value(c, h, a1),
                    "fill_prob_calm": fill_prob(c, h, a0), "fill_prob_toxic": fill_prob(c, h, a1)})
    return out
