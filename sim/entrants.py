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
from .env import GlostenMilgromEnv, MarketConfig, _norm_sf


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


# ---------------------------------------------------------------- identity treatment (RQ1)
class WalletEntrant(Agent):
    """Rent-or-toxicity inference from the PUBLIC post-commitment tape, under three nested
    information sets.  Requires the heterogeneous-taker environment (MarketConfig.n_wallets > 0);
    every print is public one period later as (event, price, wallet id, ex-post mark V).

        anon    sees prints and marks but no wallet ids: it estimates the population toxicity
                from every period (print or not) and can use the LAST print as a one-observation
                signal about whoever is active (flow autocorrelation), but cannot link a wallet's
                returns.
        id      additionally keys prints by wallet id, so each wallet's type is learned from its
                whole history (the Zhai signal).
        oracle  knows every wallet's type (upper bound).

    Estimation is exact Bayes on a 101-point grid over alpha (no plug-in bias).  Let
        e = exp(-x/m)   P(informed arrival trades at half-spread x)
        u = 2 Phibar(x/sigma)   P(uninformed arrival trades)
    Population toxicity abar, from every period (the best quote x is public even without a print):
        no print            : P = (1-a)(1-u) + a(1-e)
        print, maker PnL > 0: P = (1-a) u (1 - e/2)         (informed fills always lose)
        print, maker PnL <= 0: P = a e + (1-a) u e/2        (an uninformed fill loses w.p. e/2)
    Wallet type alpha_j, from wallet j's prints only (its non-arrivals are not attributable),
    conditional on the print:  q = a e / (a e + (1-a) u),  P(PnL>0) = (1-q)(1-e/2),
    P(PnL<=0) = q + (1-q) e/2.  Prior Beta(kappa abar_hat, kappa (1-abar_hat)).
    anon uses the same wallet posterior with exactly one print (the last one).
    Forecast of the next arrival, k periods after the last print by wallet j:
        alpha_next = rho^k alpha_j_hat + (1 - rho^k) abar_hat
    (rho, kappa, m, sigma are treated as known market structure; the types are not.)

    Regime filter (MarketConfig.regime_persistence > 0).  The entrant tracks b = P(z = toxic) for
    the coming arrival.  Every period is evidence: a print by wallet j has likelihood ratio
    w_j(1)/w_j(0) = alpha_j (1-abar) / ((1-alpha_j) abar) under the activity-tilt rule (id, oracle:
    identity is the evidence; the PnL sign adds nothing given j); without ids (anon) the PnL sign
    is the evidence through the regime's mean type, P(print & loss | z) = alpha(z) e + (1-alpha(z)) u e/2
    etc. (all linear in alpha, so only alpha(z) is needed); a no-print period is weak evidence
    either way.  Then b <- rho_z b + (1 - rho_z) p1 and the forecast is b alpha(1) + (1-b) alpha(0),
    combined with the wallet-persistence term when both channels are on.

    Quote rule, with (h^C, h^M) evaluated at alpha_next and best = best incumbent quote:
        if best - h^C >= margin:  h = min(best - 1, h^M)     undercut, never past that type's monopoly quote
        else:                     h = h^C                     do not pay to take this flow
    Against a toxic active wallet h^C is wide, so the entrant withdraws; against a benign one it
    undercuts.  Identity's value is fill SELECTION against identity-blind incumbents, which is why
    identity_monopoly_profit (a sole quoter) barely moves while an entrant's profit can.
    """

    GRID = np.linspace(0.005, 0.995, 100)

    def __init__(self, cfg: MarketConfig, info_set: str = "id", wallet_alpha=None, alpha_eff=None,
                 alpha_prior: float = 0.3, prior_strength: float = 20.0, margin: int = 2,
                 h_min: int = 1):
        if info_set not in ("anon", "id", "oracle"):
            raise ValueError(info_set)
        if cfg.n_wallets <= 0:
            raise ValueError("WalletEntrant needs MarketConfig.n_wallets > 0")
        if info_set == "oracle" and wallet_alpha is None:
            raise ValueError("oracle needs the wallet type vector")
        from .benchmarks import type_benchmarks
        self.cfg = cfg
        self.info_set = info_set
        self.wallet_alpha = None if wallet_alpha is None else np.asarray(wallet_alpha, dtype=float)
        self.alpha_eff = alpha_eff
        self.m, self.sigma, self.tick, self.K = cfg.edge_mean, cfg.sigma_L, cfg.tick, cfg.max_half_spread
        self.rho, self.kappa = cfg.wallet_persistence, cfg.wallet_concentration
        self.rho_z = cfg.regime_persistence
        self.alpha_prior, self.n0, self.margin, self.h_min = alpha_prior, prior_strength, margin, h_min
        A = self.GRID
        self._log_prior_pop = (self.n0 * alpha_prior) * np.log(A) + (self.n0 * (1 - alpha_prior)) * np.log(1 - A)
        # per-half-spread likelihood pieces, cached by x (ticks)
        self._e = {h: float(np.exp(-h * self.tick / self.m)) for h in range(1, self.K + 1)}
        self._u = {h: float(2 * _norm_sf(h * self.tick / self.sigma)) for h in range(1, self.K + 1)}
        self._hC = np.empty(101, dtype=int); self._hM = np.empty(101, dtype=int)
        for g, a in enumerate(np.linspace(0.0, 1.0, 101)):
            hC, _, hM, _ = type_benchmarks(cfg, float(a))
            self._hC[g], self._hM[g] = hC, hM
        self.reset_memory()

    def reset_memory(self):
        self._ll_pop = np.zeros_like(self.GRID)
        self._ll_w: dict[int, np.ndarray] = {}
        self.n_w: dict[int, int] = {}
        self._ll_last = None
        self.last_wallet = -1; self.k_since = 0
        self.last_decision = None; self.last_alpha_next = None
        self.n_prints_seen = 0
        self._abar_cache = None
        self.b = None                        # P(next arrival is in the toxic regime)
        self.last_b = None
        self._vec = None                     # cached per-wallet posterior means (id)
        self._vec_abar = None                # abar_hat the cache was built with

    def reset(self):
        pass                                 # memory persists across episodes

    # -- estimator -------------------------------------------------------------------------
    @staticmethod
    def _post_mean(logp: np.ndarray, A: np.ndarray) -> float:
        w = np.exp(logp - logp.max())
        return float((w * A).sum() / w.sum())

    @property
    def abar_hat(self) -> float:
        if self.info_set == "oracle":
            return float(self.alpha_eff if self.alpha_eff is not None else self.wallet_alpha.mean())
        if self._abar_cache is None:
            self._abar_cache = self._post_mean(self._log_prior_pop + self._ll_pop, self.GRID)
        return self._abar_cache

    def _print_loglik(self, h: int, pnl: float) -> np.ndarray:
        """log P(PnL sign | print at h ticks, alpha) over the grid (print-conditional)."""
        A, e, u = self.GRID, self._e[h], self._u[h]
        q = A * e / (A * e + (1 - A) * u)
        return np.log((1 - q) * (1 - e / 2)) if pnl > 0 else np.log(q + (1 - q) * e / 2)

    def _period_loglik(self, h: int, ev: str, pnl: float) -> np.ndarray:
        """log P(period outcome | alpha): arrival-level, uses no-print periods too."""
        A, e, u = self.GRID, self._e[h], self._u[h]
        if ev == "none":
            return np.log((1 - A) * (1 - u) + A * (1 - e))
        if pnl > 0:
            return np.log((1 - A) * u * (1 - e / 2))
        return np.log(A * e + (1 - A) * u * e / 2)

    def _wallet_logprior(self) -> np.ndarray:
        a = self.abar_hat
        return (self.kappa * a) * np.log(self.GRID) + (self.kappa * (1 - a)) * np.log(1 - self.GRID)

    def wallet_alpha_hat(self, j: int) -> float:
        if self.info_set == "oracle":
            return float(self.wallet_alpha[j])
        if self.info_set == "anon":
            ll = self._ll_last if self._ll_last is not None else 0.0
        else:
            ll = self._ll_w.get(j, 0.0)
        return self._post_mean(self._wallet_logprior() + ll, self.GRID)

    def _ingest(self, obs):
        hs = obs["last_half_spreads"]
        h = min(int(v) for v in hs)                                # the print (if any) was at the best quote
        if h < 1 or h > self.K:                                    # first observation of an episode: no period yet
            return
        ev = obs.get("last_event", "none")
        # ---- immediate: the print itself (identity, or the fact of a print)
        if ev == "noliq":                                          # one-tick book with nobody resting: no information
            if self.last_wallet >= 0:
                self.k_since += 1
        elif ev == "none":
            if self.last_wallet >= 0:
                self.k_since += 1
            self._regime_immediate(h, ev, -1)
        else:
            j = int(obs["last_taker"])
            self.n_prints_seen += 1
            self.n_w[j] = self.n_w.get(j, 0) + 1
            self.last_wallet, self.k_since = j, 1
            self._regime_immediate(h, ev, j)
        # ---- delayed: marks released this period (mark_lag periods after their print)
        for ago, mev, mh, mprice, mwallet, mV in obs.get("marks", ()):
            mh = int(mh)
            if mev == "noliq":
                continue
            if mev == "none":
                if self.info_set != "oracle":
                    self._ll_pop += self._period_loglik(mh, mev, 0.0); self._abar_cache = None
                continue
            pnl = (float(mprice) - float(mV)) if mev == "buy" else (float(mV) - float(mprice))
            if self.info_set != "oracle":
                self._ll_pop += self._period_loglik(mh, mev, pnl); self._abar_cache = None
                ll = self._print_loglik(mh, pnl)
                self._ll_last = ll
                if self.info_set == "id":
                    jj = int(mwallet)
                    self._ll_w[jj] = self._ll_w.get(jj, 0.0) + ll
                    if self._vec is not None:
                        self._vec[jj] = self.wallet_alpha_hat(jj)
            self._regime_delayed(mh, pnl, int(ago) - 1)
        self._regime_propagate()

    # -- regime filter ---------------------------------------------------------------------
    def _regime_params(self):
        """(alpha_calm, alpha_toxic, p1) under the entrant's current type estimates."""
        a = self.abar_hat
        if self.info_set == "anon":
            # types ~ Beta(kappa a, kappa (1-a)); tilting by alpha / (1-alpha) shifts one parameter by 1
            A, B = self.kappa * a, self.kappa * (1 - a)
            a1, a0 = (A + 1) / (A + B + 1), A / (A + B + 1)
        else:
            al = self._all_wallet_alpha()
            a1, a0 = float((al * al).sum() / al.sum()), float((al * (1 - al)).sum() / (1 - al).sum())
        p1 = (a - a0) / (a1 - a0) if a1 > a0 + 1e-12 else 0.5
        return a0, a1, float(min(max(p1, 0.0), 1.0))

    def _all_wallet_alpha(self) -> np.ndarray:
        if self.info_set == "oracle":
            return self.wallet_alpha
        a = self.abar_hat
        if self._vec is None or abs(a - self._vec_abar) > 0.005:          # prior moved: rebuild all
            self._vec = np.array([self.wallet_alpha_hat(j) for j in range(self.cfg.n_wallets)])
            self._vec_abar = a
        return self._vec

    def _regime_evidence(self, L1: float, L0: float, steps: int):
        """Bayes step on b (currently the belief about the regime of the last arrival) with evidence
        about the regime `steps` arrivals earlier.  The regime chain is 'keep w.p. rho_z, else redraw
        from the stationary law', which is reversible, so the evidence transfers with weight rho_z^steps
        and otherwise says only what the stationary law says."""
        if steps > 0:
            _, _, p1 = self._regime_params()
            w = self.rho_z ** steps
            mix = p1 * L1 + (1 - p1) * L0
            L1, L0 = w * L1 + (1 - w) * mix, w * L0 + (1 - w) * mix
        self.b = self.b * L1 / max(self.b * L1 + (1 - self.b) * L0, 1e-300)

    def _regime_immediate(self, h: int, ev: str, j: int):
        """Evidence available at the print itself: who printed (id, oracle) or that a print happened (anon)."""
        if self.rho_z <= 0:
            return
        a0, a1, p1 = self._regime_params()
        if self.b is None:
            self.b = p1
        e, u = self._e[h], self._u[h]
        if ev == "none":
            self._regime_evidence(1 - a1 * e - (1 - a1) * u, 1 - a0 * e - (1 - a0) * u, 0)
        elif self.info_set == "anon":
            self._regime_evidence(a1 * e + (1 - a1) * u, a0 * e + (1 - a0) * u, 0)
        else:
            aj, a = self.wallet_alpha_hat(j), self.abar_hat
            self._regime_evidence(aj / max(a, 1e-9), (1 - aj) / max(1 - a, 1e-9), 0)

    def _regime_delayed(self, h: int, pnl: float, steps: int):
        """Evidence that needs the mark (anon only; given identity the PnL sign says nothing about z)."""
        if self.rho_z <= 0 or self.info_set != "anon" or self.b is None:
            return
        a0, a1, _ = self._regime_params()
        e, u = self._e[h], self._u[h]
        def sign_lik(a):                                   # P(sign | print, z), the part not used at the print
            P = a * e + (1 - a) * u
            return ((1 - a) * u * (1 - e / 2) / P) if pnl > 0 else ((a * e + (1 - a) * u * e / 2) / P)
        self._regime_evidence(sign_lik(a1), sign_lik(a0), steps)

    def _regime_propagate(self):
        if self.rho_z > 0 and self.b is not None:
            _, _, p1 = self._regime_params()
            self.b = self.rho_z * self.b + (1 - self.rho_z) * p1
            self.last_b = self.b

    def alpha_next(self) -> float:
        a = self.abar_hat
        if self.rho_z > 0 and self.b is not None:
            a0, a1, _ = self._regime_params()
            a = self.b * a1 + (1 - self.b) * a0                     # regime forecast replaces the mean
        if self.last_wallet < 0 or self.rho <= 0:
            return a
        w = self.rho ** self.k_since
        return w * self.wallet_alpha_hat(self.last_wallet) + (1 - w) * a

    # -- policy ----------------------------------------------------------------------------
    def act(self, obs, i):
        self._ingest(obs)
        a_next = self.alpha_next()
        self.last_alpha_next = a_next
        g = int(round(min(max(a_next, 0.0), 1.0) * 100))
        hC, hM = int(self._hC[g]), int(self._hM[g])
        hs = obs["last_half_spreads"]
        best = min(int(hs[j]) for j in range(len(hs)) if j != i)
        if best - hC >= self.margin:
            self.last_decision = "undercut"
            return max(self.h_min, min(best - 1, hM))
        self.last_decision = "withdraw" if hC > best else "competitive"
        return min(self.K, hC)
