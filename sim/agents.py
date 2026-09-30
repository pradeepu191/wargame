"""Quoting agents. Every agent implements

    act(obs, i) -> int            half-spread in ticks for MM index i
    update(obs, a, r, obs_next, done)   (no-op for fixed policies)

Fixed / equilibrium policies (rubric: "fixed policy" baselines):
    RandomAgent, FixedSpreadAgent, CompetitiveGMAgent, GrimTriggerAgent
Learners:
    QLearningAgent (tabular, Calvano/Colliard-style)

TODO(v1): skew and displayed-size actions; Hedge (no-regret) agent; PPO agent;
          request-budget state B_t; entrant with markout inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .benchmarks import BenchmarkTable


class Agent:
    def act(self, obs: dict, i: int) -> int:
        raise NotImplementedError

    def update(self, obs, action, reward, obs_next, done):
        pass

    def set_eval(self, flag: bool):
        pass

    def reset(self):
        """Called at the start of every episode."""
        pass


class RandomAgent(Agent):
    def __init__(self, max_half_spread: int, seed: int = 0):
        self.K = max_half_spread
        self.rng = np.random.default_rng(seed)

    def act(self, obs, i):
        return int(self.rng.integers(1, self.K + 1))


class FixedSpreadAgent(Agent):
    def __init__(self, half_spread: int):
        self.h = half_spread

    def act(self, obs, i):
        return self.h


class CompetitiveGMAgent(Agent):
    """Quotes the zero-profit Glosten--Milgrom half-spread at the current belief."""

    def __init__(self, table: BenchmarkTable):
        self.table = table

    def act(self, obs, i):
        hC, _, _, _ = self.table.lookup(obs["mu"])
        return int(hC)


class GrimTriggerAgent(Agent):
    """Hand-coded collusion with finite punishment (a 'price-trigger' strategy).

    Phases:  coop   -> quote the monopoly half-spread h^M(mu).
             punish -> quote the competitive half-spread h^C(mu) for punish_len periods.
    Trigger: in coop phase, any rival quoting below h^M enters punish.
    A two-period grace after punishment stops rivals' own (offset-by-one)
    punishment quotes from re-triggering an endless price war.
    Used as a known-collusive reference for the impulse-response test.
    """

    def __init__(self, table: BenchmarkTable, punish_len: int = 5):
        self.table = table
        self.punish_len = punish_len
        self.reset()

    def reset(self):
        self.punish_left = 0
        self.grace = 0

    def act(self, obs, i):
        if obs["t"] == 0:
            self.reset()
        hC, _, hM, _ = self.table.lookup(obs["mu"])
        rivals = np.delete(obs["last_half_spreads"], i)
        if self.punish_left > 0:
            self.punish_left -= 1
            if self.punish_left == 0:
                self.grace = 2
            return int(hC)
        if self.grace > 0:
            self.grace -= 1
        elif obs["t"] > 0 and len(rivals) and rivals.min() < hM:
            self.punish_left = self.punish_len - 1
            return int(hC)
        return int(hM)


@dataclass
class QConfig:
    learning_rate: float = 0.15
    discount: float = 0.95
    exploration_decay: float = 2e-5      # eps_t = exp(-decay * global_step)
    eps_min: float = 0.0
    init_q: float = 0.0
    inventory_buckets: int = 3           # sign buckets: short / flat / long


class QLearningAgent(Agent):
    """Tabular Q-learning.

    State = (own last half-spread, min rival last half-spread, inventory bucket).
    Action = half-spread in 1..K.
    Exploration: eps-greedy with exponential decay (Calvano et al. 2020).
    """

    def __init__(self, max_half_spread: int, n_mm: int, qcfg: QConfig, seed: int = 0):
        self.K = max_half_spread
        self.cfg = qcfg
        self.rng = np.random.default_rng(seed)
        self.n_inv = qcfg.inventory_buckets
        self.Q = np.full((self.K + 1, self.K + 1, self.n_inv, self.K + 1), qcfg.init_q)
        self.step_count = 0
        self.eval = False
        self._last_state = None

    def set_eval(self, flag: bool):
        self.eval = flag

    def _state(self, obs, i):
        own = int(obs["last_half_spreads"][i])
        rivals = np.delete(obs["last_half_spreads"], i)
        riv = int(rivals.min()) if len(rivals) else own
        inv = int(np.sign(obs["inventory"][i])) + 1 if self.n_inv == 3 else 0
        return own, riv, inv

    @property
    def epsilon(self) -> float:
        if self.eval:
            return 0.0
        return max(self.cfg.eps_min, float(np.exp(-self.cfg.exploration_decay * self.step_count)))

    def act(self, obs, i):
        s = self._state(obs, i)
        self._last_state = (s, i)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(1, self.K + 1))
        q = self.Q[s][1:]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best)) + 1

    def update(self, obs, action, reward, obs_next, done):
        if self.eval:
            return
        s, i = self._last_state
        s_next = self._state(obs_next, i)
        target = reward if done else reward + self.cfg.discount * self.Q[s_next][1:].max()
        self.Q[s][action] += self.cfg.learning_rate * (target - self.Q[s][action])
        self.step_count += 1
