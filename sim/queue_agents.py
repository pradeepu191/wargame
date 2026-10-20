"""Agents for the one-tick queue game (sim/queue_env.py).

QueueQAgent      identity-blind tabular Q-learner.  State = (own status: out / front / behind,
                 number of OTHER makers resting, anonymous tape signal: none / last revealed mark
                 was a gain / was a loss).  Action = rest (1) or out (0).  It sees exactly what an
                 L2 feed plus the public tape shows, with the mark arriving mark_lag periods late.
QueueWalletEntrant  the WalletEntrant's estimator and regime filter (anon / id / oracle), with the
                 decision "rest iff the expected value of a fill at the fixed spread against the
                 forecast type is positive".  Stepping out costs the place in line; the rule is
                 myopic about that on purpose -- it is the simplest identity-aware quoter.
QueueHedgeAgent  a no-regret learner (EXP3) over a small set of fixed queue policies -- the
                 proposal's "no-regret / Hedge" incumbent.  Tabular Q-learning cannot learn to
                 queue: with time priority the reward of resting arrives only after several
                 consecutive periods in line, an epsilon-greedy learner breaks the chain, never
                 reaches the front, sees zero either way and flickers (results/REPLICATION_NOTES.md).
                 A learner over policies has no such problem: each expert is a complete way of
                 behaving in the queue and the learner only has to find out which one pays here.
AlwaysRest / NeverRest  the two trivial benchmarks (break-even round-robin; zero).
"""
from __future__ import annotations

from math import exp

import numpy as np

from .agents import Agent, QConfig
from .entrants import WalletEntrant
from .env import MarketConfig
from .queue_env import fill_prob, fill_value


class AlwaysRest(Agent):
    def act(self, obs, i):
        return 1


class NeverRest(Agent):
    def act(self, obs, i):
        return 0


class QueueQAgent(Agent):
    """State: own status (out / front / behind -- or, with rank_state, out / front / 2nd / 3rd /
    deeper), number of other makers resting, anonymous tape signal.  With rank_state the value of
    staying in line can propagate back along the queue (3rd -> 2nd -> front -> fill), which the
    three-status encoding cannot represent: a maker that is 'behind' with no idea how far behind
    sees the same zero reward whether it stays or flickers, and learns to flicker."""
    N_SIGNAL = 3

    def __init__(self, n_mm: int, qcfg: QConfig, seed: int = 0, rank_state: bool = False):
        self.n_mm = n_mm
        self.cfg = qcfg
        self.rank_state = rank_state
        self.N_STATUS = 5 if rank_state else 3
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
        if self.rank_state:
            status = 0 if rank < 0 else min(rank + 1, 4)      # out, front, 2nd, 3rd, 4th-or-deeper
        else:
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
    """WalletEntrant's inference with the rest/out decision at a fixed spread.

    With pay=True it also uses the venue's priority fee: when it would rest and is not already at
    the front in time priority, it pays f iff the expected value of being served this period,
    P(fill) * v(alpha_next), exceeds f.  A rule, not an optimum: it ignores that others may pay
    too and that a place in line has value beyond this period."""

    def __init__(self, cfg: MarketConfig, half_spread: int, info_set: str = "id", pay: bool = False, **kw):
        super().__init__(cfg, info_set, **kw)
        self.h = int(half_spread)
        self.threshold = 0.0
        self.pay = pay

    def act(self, obs, i):
        self._ingest(obs)
        a_next = self.alpha_next()
        self.last_alpha_next = a_next
        v = fill_value(self.cfg, self.h, a_next)
        if v <= self.threshold:
            self.last_decision = "out"
            return 0
        fee = float(obs.get("priority_fee", 0.0))
        if self.pay and fee > 0 and int(obs["fifo_rank"][i]) != 0:
            if fill_prob(self.cfg, self.h, a_next) * v > fee:
                self.last_decision = "pay"
                return 2
        self.last_decision = "rest"
        return 1


class QueueHedgeAgent(Agent):
    """EXP3 over fixed policies with per-episode (bandit) feedback.

    experts: list of agents (each with act(obs, i) and reset()).  At each episode start one expert
    is drawn from the current distribution and plays the whole episode; its scaled episode reward
    updates the weights with the importance-weighted EXP3 rule.  All experts' act() is called every
    period so that stateful experts (a regime filter) keep tracking the tape while not in charge.
    In eval mode the weights are frozen and the draw continues (a mixed strategy)."""

    def __init__(self, experts: list, eta: float = 0.1, explore: float = 0.05, reward_scale: float = 30.0,
                 seed: int = 0):
        self.experts = experts
        self.K = len(experts)
        self.eta, self.explore, self.scale = eta, explore, reward_scale
        self.rng = np.random.default_rng(seed)
        self.logw = np.zeros(self.K)
        self.eval = False
        self.choice = 0
        self.p = np.full(self.K, 1.0 / self.K)
        self.ep_reward = 0.0

    def set_eval(self, flag: bool):
        self.eval = flag

    def probs(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        p = w / w.sum()
        if not self.eval:
            p = (1 - self.explore) * p + self.explore / self.K
        return p

    def reset(self):
        for e in self.experts:
            e.reset()
        self.p = self.probs()
        self.choice = int(self.rng.choice(self.K, p=self.p))
        self.ep_reward = 0.0

    def act(self, obs, i):
        acts = [e.act(obs, i) for e in self.experts]      # keep every expert's state current
        return int(acts[self.choice])

    def update(self, obs, action, reward, obs_next, done):
        if self.eval:
            return
        self.ep_reward += reward
        if done:
            r = self.ep_reward / self.scale
            self.logw[self.choice] += self.eta * r / self.p[self.choice]
            self.logw -= self.logw.max()                      # keep the exponent bounded
