"""Glosten--Milgrom dealer market with N quoting market makers on a tick grid.

Model (see paper/deliverable1.tex, Section 1):
  * Fundamental V.  Two edge distributions (edge_dist):
      "fixed"       : V in {V_L, V_H} = mid -/+ v, prior 1/2 (binary Glosten--Milgrom).
                      Any half-spread >= v is immune to informed flow: an adverse-selection
                      CLIFF that tabular learners find and sit on (see results/REPLICATION_NOTES.md).
      "exponential" : V = mid + s*d, s = +/-1 w.p. 1/2, d ~ Exponential(mean = edge_mean).
                      An informed trader trades at half-spread x w.p. exp(-x/m) -- never
                      zero -- and by memorylessness the expected loss per informed fill is
                      exactly m at every spread.  No immune quote.  i.i.d. V only.
    Two timing modes (redraw_v_each_period):
      True  : V redrawn every period, mu_t = 1/2 always (stationary repeated game)
      False : V drawn once per episode, belief mu_t updated by Bayes (fixed edge only)
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
    edge_dist: str = "fixed"            # "fixed" (binary V) or "exponential" (continuous edge)
    edge_mean: float = 5.0              # mean informed edge for edge_dist="exponential"
    seed: int = 0

    def __post_init__(self):
        if self.edge_dist not in ("fixed", "exponential"):
            raise ValueError("edge_dist must be 'fixed' or 'exponential'")
        if self.edge_dist == "exponential" and not self.redraw_v_each_period:
            raise ValueError("edge_dist='exponential' requires redraw_v_each_period=True "
                             "(belief updating over a continuous V is not implemented)")


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
        return self.mu * self.cfg.v_high + (1 - self.mu) * self.cfg.v_low   # = prior mean when mu = 1/2

    def _draw_v(self) -> float:
        c = self.cfg
        if c.edge_dist == "exponential":
            d = self.rng.exponential(c.edge_mean)
            return self.mid + d if self.rng.random() < 0.5 else self.mid - d
        return c.v_high if self.rng.random() < 0.5 else c.v_low

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
        self.mu = 0.5
        self.V = self._draw_v() if V is None else V
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

    def step(self, half_spreads) -> tuple[dict, np.ndarray, bool, dict]:
        """half_spreads: int sequence of length n_mm, each in 1..max_half_spread."""
        c = self.cfg
        n = c.n_mm
        h = [int(x) for x in half_spreads]
        assert len(h) == n and min(h) >= 1 and max(h) <= c.max_half_spread
        mid = self.mid
        hmin = min(h)
        best_ask = mid + hmin * c.tick
        best_bid = mid - hmin * c.tick

        # ---- taker arrival
        informed = self.rng.random() < c.alpha
        event = "none"
        if informed:                       # trades only when the quote is strictly inside the true value
            if self.V > mid and best_ask < self.V:
                event = "buy"
            elif self.V < mid and best_bid > self.V:
                event = "sell"
        else:
            val = mid + self.rng.normal(0.0, c.sigma_L)
            if val > best_ask:
                event = "buy"
            elif val < best_bid:
                event = "sell"

        # ---- matching: one unit to a random MM among those at the best quote
        reward = np.zeros(n)
        filled = -1
        price = float("nan")
        if event != "none":
            cands = [j for j in range(n) if h[j] == hmin]
            filled = cands[0] if len(cands) == 1 else int(self.rng.choice(cands))
            if event == "buy":
                price = best_ask
                reward[filled] += (best_ask - self.V) + c.maker_rebate
                self.inventory[filled] -= 1
            else:
                price = best_bid
                reward[filled] += (self.V - best_bid) + c.maker_rebate
                self.inventory[filled] += 1
            if abs(self.inventory[filled]) > c.inventory_cap:
                self.inventory[filled] = c.inventory_cap if self.inventory[filled] > 0 else -c.inventory_cap
        if c.inventory_penalty:
            reward -= c.inventory_penalty * self.inventory.astype(float) ** 2

        # ---- public belief update and bookkeeping
        mu_prev = self.mu
        if c.redraw_v_each_period:
            self.V = self._draw_v()
            self.mu = 0.5
        else:
            self.mu = self.posterior(self.mu, best_ask, best_bid, mid, event)
        self.last_half_spreads = np.array(h, dtype=np.int64)
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
