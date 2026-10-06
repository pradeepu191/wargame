import numpy as np
import pytest

from sim import (BenchmarkTable, CompetitiveGMAgent, GlostenMilgromEnv, GrimTriggerAgent,
                 MarketConfig, QConfig, QLearningAgent, competitive_and_monopoly)
from sim.benchmarks import expected_profit_by_half_spread
from sim.runner import run_episode


def make_env(**kw):
    return GlostenMilgromEnv(MarketConfig(**kw))


def test_glosten_milgrom_textbook_numbers():
    """Textbook GM: uninformed trade w.p. 1/2 each side regardless of price (sigma_L -> inf);
    alpha=0.3 => E[V|buy] = 0.65*110 + 0.35*90 = 103 => zero-profit ask 103."""
    env = make_env(alpha=0.3, sigma_L=1e12, tick=0.5, max_half_spread=40)
    Pi = expected_profit_by_half_spread(env, mu=0.5)
    hC = int(np.flatnonzero(Pi >= -1e-9)[0] + 1)
    # ask = 100 + hC*0.5 must be the first grid point >= 103 -> hC = 6 (ask 103.0)
    assert hC == 6


def test_competitive_spread_increases_with_toxicity():
    hs = []
    for a in (0.1, 0.3, 0.5):
        env = make_env(alpha=a)
        hC, _, _, _ = competitive_and_monopoly(env, 0.5)
        hs.append(hC)
    assert hs == sorted(hs) and hs[0] < hs[-1]


def test_monopoly_profit_at_least_competitive():
    env = make_env()
    for mu in (0.2, 0.5, 0.8):
        _, piC, _, piM = competitive_and_monopoly(env, mu)
        assert piM >= piC - 1e-12


def test_posterior_moves_toward_truth():
    env = make_env(alpha=0.3)
    mid = env.mid
    mu = env.posterior(0.5, mid + 1.0, mid - 1.0, mid, "buy")
    assert mu > 0.5
    mu2 = env.posterior(mu, mid + 1.0, mid - 1.0, mid, "buy")
    assert mu2 > mu


def test_step_bookkeeping():
    env = make_env(n_mm=3, seed=1)
    obs = env.reset()
    n_fills = 0
    for _ in range(env.cfg.horizon):
        obs, r, done, info = env.step(np.array([2, 3, 4]))
        if info["event"] != "none":
            n_fills += 1
            assert info["filled"] == 0  # MM 0 has the best quotes on both sides
    assert done and env.t == env.cfg.horizon
    assert abs(env.inventory[0]) <= n_fills


def test_competitive_agents_have_near_zero_delta():
    env = make_env(seed=2)
    table = BenchmarkTable(env)
    agents = [CompetitiveGMAgent(table), CompetitiveGMAgent(table)]
    deltas = [run_episode(env, agents, table)[0]["delta"] for _ in range(200)]
    assert abs(np.nanmean(deltas)) < 0.1


def test_grim_agents_have_high_delta():
    env = make_env(seed=3)
    table = BenchmarkTable(env)
    agents = [GrimTriggerAgent(table), GrimTriggerAgent(table)]
    deltas = [run_episode(env, agents, table)[0]["delta"] for _ in range(200)]
    assert np.nanmean(deltas) > 0.6


def test_qlearning_monopolist_moves_toward_monopoly():
    """A single Q-learner (no rivals) should raise profit and Delta as it learns."""
    env = make_env(n_mm=1, seed=4)
    table = BenchmarkTable(env)
    agents = [QLearningAgent(env.cfg.max_half_spread, 1, QConfig(exploration_decay=1e-4), seed=0)]
    rows = [run_episode(env, agents, table)[0] for _ in range(300)]
    early = np.mean([r["mean_profit_per_mm"] for r in rows[:50]])
    late = np.mean([r["mean_profit_per_mm"] for r in rows[-50:]])
    assert late > early
    assert np.nanmean([r["delta"] for r in rows[-50:]]) > np.nanmean([r["delta"] for r in rows[:50]])
    assert agents[0].step_count == 300 * env.cfg.horizon


def test_exponential_edge_has_no_immune_quote():
    """With a continuous edge, informed traders trade at every half-spread with positive probability."""
    env = make_env(edge_dist="exponential", edge_mean=5.0, redraw_v_each_period=True, tick=0.5,
                   max_half_spread=24, alpha=1.0, sigma_L=8.0, seed=7)
    env.reset()
    fills = 0
    for _ in range(2000):
        _, _, done, info = env.step(np.array([24, 24]))     # widest possible quote
        fills += info["event"] != "none"
        if done:
            env.reset()
    assert fills > 0
    # closed-form benchmark: expected informed-trade probability at x = 12 is exp(-12/5) ~ 9%
    assert 0.05 < fills / 2000 < 0.14


def test_exponential_benchmarks_match_monte_carlo():
    env = make_env(edge_dist="exponential", edge_mean=5.0, redraw_v_each_period=True, tick=0.5,
                   max_half_spread=24, alpha=0.3, sigma_L=8.0, seed=8)
    Pi = expected_profit_by_half_spread(env, 0.5)
    h = 10
    env.reset(); tot = 0.0; n = 20000
    for _ in range(n):
        _, r, done, _ = env.step(np.array([h, h]))
        tot += r.sum()
        if done:
            env.reset()
    assert abs(tot / n - Pi[h - 1]) < 0.08      # MC noise at n=20k is ~0.03


def test_exponential_requires_iid_v():
    with pytest.raises(ValueError):
        make_env(edge_dist="exponential", redraw_v_each_period=False)
