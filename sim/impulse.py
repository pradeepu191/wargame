"""Impulse-response test for collusion (after Calvano et al. 2020, Section VI), paired version.

After training, every agent plays greedily.  In each of `n_episodes` episodes we
run to period `t_shock`, snapshot the environment and the agents, and branch:

    baseline : continue greedy play untouched
    shocked  : FORCE agent `deviator` to quote the minimum half-spread (an
               undercut) at t_shock, then release it

and record every agent's half-spread from t_shock - pre to t_shock + post on
both branches.  The *paired difference* (shocked - baseline) is the response
to the deviation.  Pairing matters because converged Q-learning policies are
often limit cycles rather than fixed points; an unpaired average of the shocked
branch alone cannot separate punishment from the cycle's own motion.

With i.i.d. V and inventory outside the state, greedy play is deterministic
given (own, rival) last quotes up to tie-breaking, so the paired difference is
essentially noise-free per episode.

Reading the output:
    rivals_diff < 0 for several periods after 0, then -> 0     genuine punishment
    rivals_diff ~ 0                                             no reward-punishment scheme
                                                                (under-exploration / cycling)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .env import GlostenMilgromEnv


def _play(env, agents, t_end, force=None, record=None, t0=0):
    """Advance env with greedy agents until env.t == t_end. force=(t, i, h) overrides one action."""
    obs = env.observation()
    while env.t < t_end:
        actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
        if force is not None and obs["t"] == force[0]:
            actions[force[1]] = force[2]
        if record is not None and 0 <= obs["t"] - t0 < len(record):
            record[obs["t"] - t0] += actions
        obs, _, done, _ = env.step(actions)
        if done:
            break
    return obs


def impulse_response(env: GlostenMilgromEnv, agents, n_episodes: int = 200,
                     t_shock: int = 50, pre: int = 5, post: int = 15,
                     deviator: int = 0) -> pd.DataFrame:
    n = env.cfg.n_mm
    window = pre + post + 1
    t0 = t_shock - pre
    base = np.zeros((window, n))
    shock = np.zeros((window, n))
    for ag in agents:
        ag.set_eval(True)
    for _ in range(n_episodes):
        env.reset()
        for ag in agents:
            ag.reset()
        _play(env, agents, t0)                                  # warm up to window start
        env_b, ag_b = copy.deepcopy(env), copy.deepcopy(agents)
        env_s, ag_s = copy.deepcopy(env), copy.deepcopy(agents)
        _play(env_b, ag_b, t0 + window, force=None, record=base, t0=t0)
        _play(env_s, ag_s, t0 + window, force=(t_shock, deviator, 1), record=shock, t0=t0)
    for ag in agents:
        ag.set_eval(False)
    base /= n_episodes
    shock /= n_episodes
    rival_idx = [i for i in range(n) if i != deviator]
    df = pd.DataFrame({"rel_t": np.arange(-pre, post + 1)})
    for i in range(n):
        df[f"h{i}"] = shock[:, i]
        df[f"h{i}_baseline"] = base[:, i]
    df["deviator"] = shock[:, deviator]
    df["rivals"] = shock[:, rival_idx].mean(axis=1)
    df["rivals_baseline"] = base[:, rival_idx].mean(axis=1)
    df["rivals_diff"] = df["rivals"] - df["rivals_baseline"]
    return df
