"""The stateless EXP3 quoter and the no-regret experiment: a single learner finds the monopoly
quote, two learners compete, and the cycle entrant alternates its phases on a cross-episode clock."""
import numpy as np

from experiments.no_regret import LureUndercutEntrant, market, run_episodes, train_cartel
from sim import Exp3QuoteAgent, GlostenMilgromEnv
from sim.benchmarks import type_benchmarks


def test_exp3_update_moves_mass_to_the_rewarded_action():
    ag = Exp3QuoteAgent(6, eta=0.5, explore=0.0, reward_scale=1.0, seed=0)
    obs = {}
    for _ in range(30):
        a = ag.act(obs, 0)
        ag.update(obs, a, 1.0 if a == 3 else -1.0, obs, False)
    p = ag.probs()
    assert abs(p.sum() - 1) < 1e-9
    assert p[2] > 0.8, p


def test_single_exp3_learner_finds_the_monopoly_quote():
    alpha = 0.3
    inc = train_cartel(alpha, 1, 0.05, seed=0, n_episodes=1500)
    hC, PiC, hM, PiM = type_benchmarks(market(alpha, 1, 0), alpha)
    env = GlostenMilgromEnv(market(alpha, 1, 5))
    full, _ = run_episodes(env, inc, 300)
    C = (full.sum() - PiC) / (PiM - PiC)
    assert C > 0.6, C


def test_two_exp3_learners_compete_away_most_of_the_rent():
    alpha = 0.3
    inc = train_cartel(alpha, 2, 0.05, seed=0, n_episodes=1500)
    hC, PiC, hM, PiM = type_benchmarks(market(alpha, 2, 0), alpha)
    env = GlostenMilgromEnv(market(alpha, 2, 5))
    full, _ = run_episodes(env, inc, 300)
    C = (full.sum() - PiC) / (PiM - PiC)
    assert C < 0.4, C


def test_cycle_entrant_phases_run_across_episodes():
    e = LureUndercutEntrant(h_lure=24, T_lure=3, h_under=13, T_under=2)
    seq = [e.act({}, 0) for _ in range(10)]
    assert seq == [24, 24, 24, 13, 13, 24, 24, 24, 13, 13]
    e.reset()
    assert e.act({}, 0) == 24 and e.act({}, 0) == 24 and e.act({}, 0) == 24 and e.act({}, 0) == 13
