"""The mechanism test must classify a myopic best-responder as unrestrained and a
forward-looking cooperator as restrained."""
import numpy as np

from analysis.mechanism import analyse_agent, immediate_reward_table
from sim import GlostenMilgromEnv, MarketConfig


def _env():
    return GlostenMilgromEnv(MarketConfig(n_mm=2, alpha=0.3, sigma_L=8.0, tick=0.5, max_half_spread=24,
                                          redraw_v_each_period=True, edge_dist="exponential", edge_mean=5.0))


def test_myopic_q_is_unrestrained():
    env = _env(); K = env.cfg.max_half_spread
    r = immediate_reward_table(env)
    Q = np.zeros((K + 1, K + 1, 1, K + 1))
    for own in range(1, K + 1):
        for riv in range(1, K + 1):
            Q[own, riv, 0, 1:] = r[riv, 1:]              # Q == immediate reward -> a* == b
    states = {(own, riv): 1 for own in range(1, K + 1) for riv in range(1, K + 1)}
    summ, _ = analyse_agent(Q, r, states, K)
    assert summ["restraint_share"] == 0.0


def test_cooperative_q_is_restrained():
    env = _env(); K = env.cfg.max_half_spread
    r = immediate_reward_table(env)
    hM = 14
    Q = np.zeros((K + 1, K + 1, 1, K + 1))
    for own in range(1, K + 1):
        for riv in range(1, K + 1):
            Q[own, riv, 0, 1:] = r[riv, 1:]
            Q[own, riv, 0, hM] += 10.0                    # continuation value for quoting monopoly
    # Restraint is only visible where undercutting would pay: rival at or below hM.
    # (If the rival sits above hM the myopic best response is already hM.)
    states = {(own, riv): 1 for own in range(1, K + 1) for riv in range(8, hM + 1)}
    summ, df = analyse_agent(Q, r, states, K)
    assert summ["restraint_share"] > 0.9
    assert summ["forgone_reward"] > 0
    assert (df[df.restrained].a_star == hM).all()
