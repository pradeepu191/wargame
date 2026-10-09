"""Agents for the one-tick queue game (sim/queue_env.py).

QueueQAgent      identity-blind tabular Q-learner.  State = (own status: out / front / behind,
                 number of OTHER makers resting, anonymous tape signal: none / last revealed mark
                 was a gain / was a loss).  Action = rest (1) or out (0).  It sees exactly what an
                 L2 feed plus the public tape shows, with the mark arriving mark_lag periods late.
QueueWalletEntrant  the WalletEntrant's estimator and regime filter (anon / id / oracle), with the
                 decision "rest iff the expected value of a fill at the fixed spread against the
                 forecast type is positive".  Stepping out costs the place in line; the rule is
                 myopic about that on purpose -- it is the simplest identity-aware quoter.
AlwaysRest / NeverRest  the two trivial benchmarks (break-even round-robin; zero).
"""
from __future__ import annotations

from math import exp

import numpy as np

from .agents import Agent, QConfig
from .entrants import WalletEntrant
from .env import MarketConfig
from .queue_env import fill_value


class AlwaysRest(Agent):
    def act(self, obs, i):
        return 1


class NeverRest(Agent):
    def act(self, obs, i):
        return 0


class QueueQAgent(Agent):
    N_STATUS, N_SIGNAL = 3, 3

    def __init__(self, n_mm: int, qcfg: QConfig, seed: int = 0):
        self.n_mm = n_mm
        self.cfg = qcfg
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((self.N_STATUS, n_mm, self.N_SIGNAL, 2), qcfg.init_q)
        self.step_count = 0
        self.eval = False
        self._last = None
        self._signal = 0

    def set_eval(self, flag: bool):
        self.eval = flag

    def _state(self, obs, i):
        rank = int(obs["rank"][i])
        status = 0 if rank < 0 else (1 if rank == 0 else 2)
        others = int(obs["n_resting"]) - (1 if rank >= 0 else 0)
        # anonymous tape signal: the most recent mark released, if it was a print
        for ago, ev, h, price, wallet, V in obs.get("marks", ()):
            if ev != "none":
                pnl = (price - V) if ev == "buy" else (V - price)
                self._signal = 1 if pnl > 0 else 2
        return status, min(max(others, 0), self.n_mm - 1), self._signal

    @property
    def epsilon(self) -> float:
        if self.eval:
            return 0.0
        return max(self.cfg.eps_min, exp(-self.cfg.exploration_decay * self.step_count))

    def act(self, obs, i):
        s = self._state(obs, i)
        self._last = (s, i)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))
        q = self.Q[s]
        return int(q[1] > q[0]) if q[0] != q[1] else int(self.rng.integers(0, 2))

    def update(self, obs, action, reward, obs_next, done):
        if self.eval:
            return
        s, i = self._last
        s_next = self._state(obs_next, i)
        target = reward if done else reward + self.cfg.discount * self.Q[s_next].max()
        self.Q[s][action] += self.cfg.learning_rate * (target - self.Q[s][action])
        self.step_count += 1

    def reset(self):
        pass                                  # the tape signal persists across episodes: one continuing market


class QueueWalletEntrant(WalletEntrant):
    """WalletEntrant's inference with the rest/out decision at a fixed spread."""

    def __init__(self, cfg: MarketConfig, half_spread: int, info_set: str = "id", **kw):
        super().__init__(cfg, info_set, **kw)
        self.h = int(half_spread)
        self.threshold = 0.0

    def act(self, obs, i):
        self._ingest(obs)
        a_next = self.alpha_next()
        self.last_alpha_next = a_next
        v = fill_value(self.cfg, self.h, a_next)
        self.last_decision = "rest" if v > self.threshold else "out"
        return 1 if v > self.threshold else 0
