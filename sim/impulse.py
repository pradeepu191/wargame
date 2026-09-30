"""Impulse-response test for collusion (Calvano et al. 2020, Section VI).

After training, put every agent in greedy (eval) mode.  Run `n_episodes`
episodes; in each, at period `t_shock`, FORCE agent `deviator` to quote the
minimum half-spread (an undercut) for one period, then release it.  Record
every agent's half-spread from `t_shock - pre` to `t_shock + post`.

If rivals respond by cutting their spreads for several periods and then
return to the pre-shock level, the learned policies carry a genuine
reward--punishment scheme (collusion in the legal sense).  If nothing moves,
wide spreads come from under-exploration rather than punishment
(Colliard, Foucault & Lovo's mechanism).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import GlostenMilgromEnv


def impulse_response(env: GlostenMilgromEnv, agents, n_episodes: int = 200,
                     t_shock: int = 50, pre: int = 5, post: int = 15,
                     deviator: int = 0) -> pd.DataFrame:
    n = env.cfg.n_mm
    window = pre + post + 1
    acc = np.zeros((window, n))
    for ag in agents:
        ag.set_eval(True)
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            if obs["t"] == t_shock:
                actions[deviator] = 1
            k = obs["t"] - (t_shock - pre)
            if 0 <= k < window:
                acc[k] += actions
            obs, _, done, _ = env.step(actions)
    for ag in agents:
        ag.set_eval(False)
    acc /= n_episodes
    df = pd.DataFrame(acc, columns=[f"h{i}" for i in range(n)])
    df.insert(0, "rel_t", np.arange(-pre, post + 1))
    return df
