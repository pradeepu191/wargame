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

Heterogeneous, persistent takers (n_wallets > 0; the Hyperliquid identity treatment):
  * A population of M wallets with persistent types alpha_j ~ Beta(abar*kappa, (1-abar)*kappa),
    drawn once from wallet_seed; kappa = wallet_concentration (kappa -> inf: homogeneous).
  * Activity weights w_j ∝ (j+1)^(-wallet_zipf) (0 = uniform).  The arriving wallet is the
    previous period's wallet w.p. rho = wallet_persistence, else a fresh draw from w: bursty,
    autocorrelated toxicity.  Stationary arrival distribution is w regardless of rho, so the
    market-wide benchmarks use alpha_eff = sum_j w_j alpha_j.
  * After commitment the print is public: the observation carries last period's event, price,
    wallet id and the ex-post mark V (the one-period markout).  Nobody ever sees the identity
    of the order about to hit them.  With n_wallets = 0 the RNG stream is untouched and every
    earlier result reproduces bit-for-bit (tests/test_env.py pins this).

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
    # heterogeneous persistent takers (0 = homogeneous, the original model)
    n_wallets: int = 0
    wallet_concentration: float = 10.0  # Beta concentration kappa of alpha_j around alpha
    wallet_persistence: float = 0.0     # rho: P(next arrival is the same wallet as the last)
    wallet_zipf: float = 0.0            # activity weights ∝ rank^-zipf (0 = uniform)
    wallet_seed: Optional[int] = None   # seed for the type draw (default: seed); keep it fixed
                                        # across training / entry / evaluation environments

    def __post_init__(self):
        if self.edge_dist not in ("fixed", "exponential"):
            raise ValueError("edge_dist must be 'fixed' or 'exponential'")
        if self.edge_dist == "exponential" and not self.redraw_v_each_period:
            raise ValueError("edge_dist='exponential' requires redraw_v_each_period=True "
                             "(belief updating over a continuous V is not implemented)")
        if self.n_wallets and not (0.0 <= self.wallet_persistence <= 1.0):
            raise ValueError("wallet_persistence must be in [0, 1]")
        if self.n_wallets and self.wallet_concentration <= 0:
            raise ValueError("wallet_concentration must be > 0")


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
        # public record of the previous period's print (post-commitment information)
        self.last_event: str = "none"
        self.last_price: float = float("nan")
        self.last_taker: int = -1          # wallet id of the last PRINT (-1: no trade last period)
        self.last_V: float = float("nan")  # ex-post mark of the previous period
        # heterogeneous takers
        self.alpha_eff: float = cfg.alpha
        self.wallet_alpha: Optional[np.ndarray] = None
        self.wallet_weights: Optional[np.ndarray] = None
        self._active_wallet: int = -1      # the wallet that arrived last period (traded or not)
        if cfg.n_wallets:
            from scipy.stats import beta as _beta
            wrng = np.random.default_rng(cfg.seed if cfg.wallet_seed is None else cfg.wallet_seed)
            k = cfg.wallet_concentration
            # stratified quantiles of Beta(abar*k, (1-abar)*k): the population mean is pinned at
            # alpha (up to discretisation) so kappa controls dispersion only; the random part is
            # which type lands on which activity rank
            q = (np.arange(cfg.n_wallets) + 0.5) / cfg.n_wallets
            self.wallet_alpha = _beta.ppf(q, cfg.alpha * k, (1 - cfg.alpha) * k)
            wrng.shuffle(self.wallet_alpha)
            w = (np.arange(cfg.n_wallets) + 1.0) ** (-cfg.wallet_zipf)
            self.wallet_weights = w / w.sum()
            self.alpha_eff = float(self.wallet_weights @ self.wallet_alpha)
            self._uniform_wallets = cfg.wallet_zipf == 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def mid(self) -> float:
        return self.mu * self.cfg.v_high + (1 - self.mu) * self.cfg.v_low   # = prior mean when mu = 1/2

    def _draw_wallet(self) -> int:
        c = self.cfg
        if self._active_wallet >= 0 and c.wallet_persistence > 0 and self.rng.random() < c.wallet_persistence:
            return self._active_wallet
        if self._uniform_wallets:
            return int(self.rng.integers(c.n_wallets))
        return int(self.rng.choice(c.n_wallets, p=self.wallet_weights))

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
        self.last_event, self.last_price, self.last_taker, self.last_V = "none", float("nan"), -1, float("nan")
        # the wallet population and the active-wallet chain persist across episodes on purpose:
        # episodes are a training device, the taker process is one continuing market
        return self.observation()

    def observation(self) -> dict:
        return {
            "t": self.t,
            "mu": self.mu,
            "mid": self.mid,
            "inventory": self.inventory.copy(),
            "last_half_spreads": self.last_half_spreads.copy(),
            # previous period's print, public after commitment
            "last_event": self.last_event,
            "last_price": self.last_price,
            "last_taker": self.last_taker,
            "last_V": self.last_V,
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
        wallet = -1
        if c.n_wallets:
            wallet = self._draw_wallet()
            self._active_wallet = wallet
            informed = self.rng.random() < self.wallet_alpha[wallet]
        else:
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
        V_t = self.V
        self.last_event, self.last_price = event, price
        self.last_taker = wallet if event != "none" else -1      # identity is public only via a print
        self.last_V = V_t                                          # the ex-post mark
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
            "taker": wallet,
            "mid": mid,
            "mid_next": self.mid,
            "best_ask": best_ask,
            "best_bid": best_bid,
            "quoted_spread": best_ask - best_bid,
            "mu_prev": mu_prev,
            "V": self.V,
            "V_t": V_t,
        }
        return self.observation(), reward, done, info
