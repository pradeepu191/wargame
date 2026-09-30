"""Glosten--Milgrom dealer market with N quoting market makers on a tick grid.

Model (see paper/deliverable1.tex, Section 1):
  * Fundamental V in {V_L, V_H}, prior 1/2.  Two modes:
      redraw_v_each_period=True  : V redrawn every period, mu_t = 1/2 always
                                   (stationary repeated game; Colliard et al. baseline)
      redraw_v_each_period=False : V drawn once per episode, public belief
                                   mu_t = P(V=V_H | tape) updated by Bayes each period
  * Mid-price m_t = E[V | mu_t].
  * Each period every MM i posts symmetric quotes at half-spread h_i ticks:
        bid_i = m_t - h_i * tick,  ask_i = m_t + h_i * tick.
    (Skew and displayed size are v1 extensions; see TODO in Agent API.)
  * One taker per period:
        informed  w.p. alpha : buys if V=V_H, sells if V=V_L (only if the
                               quote is strictly better than V for them);
        uninformed otherwise : private valuation m_t + L, L ~ N(0, sigma_L);
                               buys if best ask < m_t + L, sells if best bid > m_t + L.
  * Taker trades one unit against the best quote; ties split uniformly.
  * Realised profit per fill: (ask - V) on a sale, (V - bid) on a purchase,
    plus maker rebate.  Inventory penalty phi * I^2 per period.

Everything is vectorised over MMs but the period loop is explicit; at T=100
periods per episode this runs ~1e4 episodes/minute in pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from math import erfc, sqrt

import numpy as np

_SQRT2 = sqrt(2.0)


def _norm_sf(x: float) -> float:
    return 0.5 * erfc(x / _SQRT2)


def _norm_cdf(x: float) -> float:
    return 0.5 * erfc(-x / _SQRT2)


@dataclass
class MarketConfig:
    n_mm: int = 2                 # number of incumbent market makers
    v_low: float = 90.0
    v_high: float = 110.0
    alpha: float = 0.3            # toxicity: P(taker is informed)
    sigma_L: float = 6.0          # dispersion of uninformed valuations
    tick: float = 1.0
    max_half_spread: int = 15     # action space is h in {1, ..., max_half_spread}
    horizon: int = 100            # periods per episode
    maker_rebate: float = 0.0
    inventory_penalty: float = 0.0
    inventory_cap: int = 50
    redraw_v_each_period: bool = False  # True: i.i.d. V (stationary game, Colliard et al. replication)
                                        # False: V persists for the episode, belief mu_t evolves
    seed: int = 0


class GlostenMilgromEnv:
    """Multi-agent environment. Call reset(); then step(half_spreads) per period."""

    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self.V: float = cfg.v_high
        self.mu: float = 0.5
        self.inventory = np.zeros(cfg.n_mm, dtype=np.int64)
        self.last_half_spreads = np.full(cfg.n_mm, cfg.max_half_spread, dtype=np.int64)

    # ------------------------------------------------------------------ helpers
    @property
    def mid(self) -> float:
        return self.mu * self.cfg.v_high + (1 - self.mu) * self.cfg.v_low

    def _p_buy_sell(self, ask: float, bid: float, mid: float, mu: float):
        """P(buy), P(sell) given best quotes and belief mu (marginal over V)."""
        c = self.cfg
        p_buy_h, p_sell_h = self._p_buy_sell_given_V(ask, bid, mid, c.v_high)
        p_buy_l, p_sell_l = self._p_buy_sell_given_V(ask, bid, mid, c.v_low)
        return (mu * p_buy_h + (1 - mu) * p_buy_l,
                mu * p_sell_h + (1 - mu) * p_sell_l)

    def _p_buy_sell_given_V(self, ask: float, bid: float, mid: float, V: float):
        c = self.cfg
        # informed component
        inf_buy = 1.0 if (V == c.v_high and ask < V) else 0.0
        inf_sell = 1.0 if (V == c.v_low and bid > V) else 0.0
        # uninformed component: valuation mid + L
        u_buy = _norm_sf((ask - mid) / c.sigma_L)
        u_sell = _norm_cdf((bid - mid) / c.sigma_L)
        return (c.alpha * inf_buy + (1 - c.alpha) * u_buy,
                c.alpha * inf_sell + (1 - c.alpha) * u_sell)

    def posterior(self, mu: float, ask: float, bid: float, mid: float, event: str) -> float:
        """Bayes update of P(V_H) after observing 'buy', 'sell' or 'none'."""
        c = self.cfg
        p_buy_h, p_sell_h = self._p_buy_sell_given_V(ask, bid, mid, c.v_high)
        p_buy_l, p_sell_l = self._p_buy_sell_given_V(ask, bid, mid, c.v_low)
        if event == "buy":
            lh, ll = p_buy_h, p_buy_l
        elif event == "sell":
            lh, ll = p_sell_h, p_sell_l
        else:
            lh, ll = 1 - p_buy_h - p_sell_h, 1 - p_buy_l - p_sell_l
        num = mu * lh
        den = num + (1 - mu) * ll
        return mu if den <= 0 else num / den

    # ------------------------------------------------------------------ API
    def reset(self, V: Optional[float] = None) -> dict:
        c = self.cfg
        self.t = 0
        self.V = float(self.rng.choice([c.v_low, c.v_high])) if V is None else V
        self.mu = 0.5
        self.inventory[:] = 0
        self.last_half_spreads[:] = c.max_half_spread
        return self.observation()

    def observation(self) -> dict:
        return {
            "t": self.t,
            "mu": self.mu,
            "mid": self.mid,
            "inventory": self.inventory.copy(),
            "last_half_spreads": self.last_half_spreads.copy(),
        }

    def step(self, half_spreads: np.ndarray) -> tuple[dict, np.ndarray, bool, dict]:
        """half_spreads: int array of shape (n_mm,), each in 1..max_half_spread."""
        c = self.cfg
        h = np.asarray(half_spreads, dtype=np.int64)
        assert h.shape == (c.n_mm,) and h.min() >= 1 and h.max() <= c.max_half_spread
        mid = self.mid
        asks = mid + h * c.tick
        bids = mid - h * c.tick
        best_ask, best_bid = asks.min(), bids.max()

        # ---- taker arrival
        informed = self.rng.random() < c.alpha
        event = "none"
        if informed:
            if self.V == c.v_high and best_ask < self.V:
                event = "buy"
            elif self.V == c.v_low and best_bid > self.V:
                event = "sell"
        else:
            val = mid + self.rng.normal(0.0, c.sigma_L)
            if val > best_ask:
                event = "buy"
            elif val < best_bid:
                event = "sell"

        # ---- matching: one unit to a random MM among those at the best quote
        reward = np.zeros(c.n_mm)
        filled = -1
        price = np.nan
        if event == "buy":
            cands = np.flatnonzero(asks == best_ask)
            filled = int(self.rng.choice(cands))
            price = best_ask
            reward[filled] += (best_ask - self.V) + c.maker_rebate
            self.inventory[filled] -= 1
        elif event == "sell":
            cands = np.flatnonzero(bids == best_bid)
            filled = int(self.rng.choice(cands))
            price = best_bid
            reward[filled] += (self.V - best_bid) + c.maker_rebate
            self.inventory[filled] += 1
        np.clip(self.inventory, -c.inventory_cap, c.inventory_cap, out=self.inventory)
        reward -= c.inventory_penalty * self.inventory.astype(float) ** 2

        # ---- public belief update and bookkeeping
        mu_prev = self.mu
        if c.redraw_v_each_period:
            self.V = float(self.rng.choice([c.v_low, c.v_high]))
            self.mu = 0.5
        else:
            self.mu = self.posterior(self.mu, best_ask, best_bid, mid, event)
        self.last_half_spreads = h.copy()
        self.t += 1
        done = self.t >= c.horizon
        info = {
            "event": event,
            "price": price,
            "filled": filled,
            "informed": informed,
            "mid": mid,
            "mid_next": self.mid,
            "best_ask": best_ask,
            "best_bid": best_bid,
            "quoted_spread": best_ask - best_bid,
            "mu_prev": mu_prev,
            "V": self.V,
        }
        return self.observation(), reward, done, info
