"""Exact best response by backward induction: indexing, the trivial cartel, DP-vs-simulation."""
import numpy as np

from experiments.best_response import CartelMDP, DPEntrant, response_tables
from experiments.entry import FrozenQ, play
from sim import GlostenMilgromEnv, MarketConfig
from sim.benchmarks import profit_curve


def q_tables_for(rule, K, N):
    """Q tables whose greedy action at (own, riv) is rule(own, riv); other actions sit at -1."""
    Q = []
    for _ in range(N):
        q = np.full((K + 1, K + 1, 1, K + 1), -1.0)
        for own in range(K + 1):
            for riv in range(K + 1):
                q[own, riv, 0, rule(own, riv)] = 0.0
        Q.append(q)
    return Q


def market(K=12, N=2, alpha=0.3):
    return MarketConfig(alpha=alpha, n_mm=N, max_half_spread=K, horizon=20, tick=0.5, sigma_L=8.0,
                        edge_dist="exponential", edge_mean=5.0, redraw_v_each_period=True, seed=0)


def test_next_state_index_matches_ravel():
    K, N = 6, 2
    Q = q_tables_for(lambda own, riv: max(1, min(own, riv)), K, N)
    mdp = CartelMDP(Q, market(K, N + 1), horizon=5)
    for s in (0, 17, 100, len(mdp.h) - 1):
        for a in (1, 3, K):
            expected = np.ravel_multi_index(list(mdp.g[s] - 1) + [a - 1], (K,) * (N + 1))
            assert mdp.base[s] + a - 1 == expected


def test_stubborn_cartel_is_undercut_by_one_tick_every_period():
    """Incumbents that always quote h_M: the best response quotes h_M - 1 from the first period
    and takes Pi(h_M - 1) every period; the cartel earns nothing."""
    K, N = 12, 2
    cfg = market(K, N)
    Pi = profit_curve(cfg, cfg.alpha)
    hM = int(Pi.argmax()) + 1
    Q = q_tables_for(lambda own, riv: hM, K, N)
    mdp = CartelMDP(Q, market(K, N + 1), horizon=20)
    policy, own, inc = mdp.solve(1.0)
    assert abs(own - Pi[hM - 2]) < 1e-9 and abs(inc) < 1e-9
    assert policy[0, mdp.s0] == hM - 1


def test_dp_value_matches_simulation_against_a_matching_cartel():
    """Incumbents that match the tightest quote they saw (min(own, riv), floored at h_C): the DP
    value from the reset state must agree with the simulated mean profit per period."""
    K, N = 12, 2
    cfg = market(K, N)
    Pi = profit_curve(cfg, cfg.alpha)
    hC = int(np.flatnonzero(Pi >= -1e-9)[0]) + 1
    Q = q_tables_for(lambda own, riv: max(hC, min(own, riv)), K, N)
    R, ties = response_tables(Q, K)
    assert ties == 0
    mdp = CartelMDP(Q, market(K, N + 1), horizon=20)
    for lam in (1.0, 0.0):
        policy, own_dp, inc_dp = mdp.solve(lam)
        env = GlostenMilgromEnv(market(K, N + 1))
        inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
        ent = DPEntrant(mdp, policy)
        pnl, _, _, _ = play(env, inc + [ent], 4000, entrant_idx=N)
        assert abs(pnl[N] - own_dp) < 0.05, (lam, pnl[N], own_dp)
        assert abs(pnl[:N].sum() - inc_dp) < 0.05, (lam, pnl[:N].sum(), inc_dp)
    # the damage-maximiser leaves the incumbents at most what the profit-maximiser leaves them
    _, _, inc_profit = mdp.solve(1.0)
    _, _, inc_damage = mdp.solve(0.0)
    assert inc_damage <= inc_profit + 1e-9


def test_stackelberg_value_sits_near_the_competitive_quote():
    """The entrant can only share a quote the incumbents have no incentive to undercut, so the
    Stackelberg value is a small positive number near Pi(h^C) and far below the rent."""
    from sim.benchmarks import stackelberg_value, type_benchmarks
    for alpha in (0.1, 0.3, 0.5):
        cfg = market(24, 2, alpha)
        hC, PiC, hM, PiM = type_benchmarks(cfg, alpha)
        sv = stackelberg_value(cfg, alpha, 2)
        for key in ("strong", "weak"):
            v, h_e, h_inc = sv[key]["value"], sv[key]["h_e"], sv[key]["h_inc"]
            assert 0.0 <= v <= 0.2 * (PiM - PiC), (alpha, key, v)
            assert hC <= h_e <= hC + 2 and h_inc == h_e, (alpha, key, h_e, h_inc)
        assert sv["strong"]["value"] >= sv["weak"]["value"]
