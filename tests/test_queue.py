"""One-tick queue game: priority rotation, no-liquidity handling, selection benchmarks."""
import numpy as np

from sim import MarketConfig
from sim.queue_agents import AlwaysRest, QueueWalletEntrant
from sim.queue_env import QueueEnv, fill_value, fill_prob, selection_benchmarks

EXP = dict(n_mm=3, alpha=0.3, sigma_L=8.0, tick=0.5, max_half_spread=24, redraw_v_each_period=True,
           edge_dist="exponential", edge_mean=5.0, seed=0, n_wallets=50, wallet_concentration=1.0, wallet_seed=0)


def test_default_spread_is_one_tick_above_breakeven_and_values_bracket_zero():
    env = QueueEnv(MarketConfig(**EXP, regime_persistence=0.9))
    bm = selection_benchmarks(env)
    assert env.h == 5 and bm["fill_value_avg"] > 0
    assert bm["fill_value_calm"] > 0 > bm["fill_value_toxic"]
    assert abs(fill_value(env.cfg, 4, 0.3)) < 0.2            # h^C is break-even by construction
    assert 0 < fill_prob(env.cfg, 5, 0.3) < 1


def test_fills_rotate_among_resting_makers_and_leavers_rejoin_at_the_back():
    env = QueueEnv(MarketConfig(**EXP))
    env.reset()
    fills = np.zeros(3)
    for _ in range(3000):
        _, _, d, info = env.step([1, 1, 1])
        if info["filled"] >= 0:
            fills[info["filled"]] += 1
        if d:
            env.reset()
    assert fills.min() > 0.9 * fills.max()                       # round-robin
    env.reset(); env.step([1, 1, 1])
    front = env.queue[0]
    a = [1, 1, 1]; a[front] = 0
    env.step(a)                                                  # the front steps out
    assert front not in env.queue
    env.step([1, 1, 1])                                          # and returns behind those who stayed
    assert front in env.queue and env.queue.index(front) >= 1    # (a fill in that step may rotate the filled maker behind it)


def test_no_liquidity_is_noliq_not_none():
    env = QueueEnv(MarketConfig(**EXP))
    obs = env.reset()
    obs, r, _, info = env.step([0, 0, 0])
    assert info["event"] == "noliq" and info["filled"] == -1 and obs["last_event"] == "noliq"
    assert obs["n_resting"] == 0 and (obs["rank"] == -1).all()


def test_regime_aware_entrant_beats_always_rest_and_is_less_adversely_selected():
    cfg = MarketConfig(**EXP, regime_persistence=0.9, mark_lag=1)
    res = {}
    for name in ("always", "oracle"):
        env = QueueEnv(cfg)
        ent = AlwaysRest() if name == "always" else QueueWalletEntrant(cfg, env.h, "oracle", wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff)
        agents = [AlwaysRest(), AlwaysRest(), ent]
        pnl = np.zeros(3); fills = np.zeros(3); inf = np.zeros(3); P = 0
        for _ in range(150):
            obs = env.reset(); done = False
            while not done:
                a = [ag.act(obs, i) for i, ag in enumerate(agents)]
                obs, r, done, info = env.step(a); pnl += r; P += 1
                if info["filled"] >= 0:
                    fills[info["filled"]] += 1; inf[info["filled"]] += info["informed"]
        res[name] = (pnl[2] / P, inf[2] / max(fills[2], 1), inf[:2].sum() / max(fills[:2].sum(), 1))
    assert res["oracle"][0] > res["always"][0]                   # selection pays
    assert res["oracle"][1] < res["always"][1]                   # its fills are cleaner
    assert res["oracle"][2] > res["always"][2]                   # the incumbents' fills are dirtier


def test_queue_game_cell_with_fringe_runs_and_reports_both_populations():
    from experiments.queue_game import one_cell, train_q, BASE
    from sim.queue_agents import QueueQAgent
    cfg = MarketConfig(**dict(BASE, alpha=0.3, regime_persistence=0.9, mark_lag=1, wallet_seed=0), n_mm=3, seed=0)
    env = QueueEnv(cfg, ticks_above_breakeven=1)
    agents = train_q(cfg, env.h, n_episodes=5, seed=0, n_learn=2)
    assert [type(a).__name__ for a in agents] == ["QueueQAgent", "QueueQAgent", "AlwaysRest"]
    assert all(a.eval for a in agents[:2])
    rows = one_cell((0.3, 0.9, 1, 1, 2, 0, "q", 2, 5, 1, 0.95))
    assert [r["policy"] for r in rows] == ["always", "anon", "id", "oracle"]
    r = rows[0]
    assert r["fringe"] == 1 and r["n_inc"] == 2 and np.isfinite(r["fringe_pnl"]) and 0 <= r["incumbent_rest_share"] <= 1
    assert abs(r["entrant_fill_share"] + r["incumbent_fill_share"] - 1) <= 1 + 1e-9   # plus the fringe's share
    rows0 = one_cell((0.3, 0.9, 1, 1, 2, 0, "always", 2, 5, 0, 0.95))
    assert np.isnan(rows0[0]["fringe_pnl"]) and np.isnan(rows0[0]["discount"])


def test_hedge_incumbents_learn_to_select_and_keep_expert_state():
    from experiments.queue_game import BASE, EXPERTS, train_hedge
    from sim.queue_agents import QueueHedgeAgent, NeverRest
    cfg = MarketConfig(**dict(BASE, alpha=0.3, regime_persistence=0.9, mark_lag=5, wallet_seed=0), n_mm=4, seed=0)
    env = QueueEnv(cfg, ticks_above_breakeven=0)
    inc = train_hedge(cfg, env.h, n_episodes=600, seed=0, n_learn=2)
    assert all(isinstance(a, QueueHedgeAgent) and a.eval for a in inc[:2])
    for a in inc[:2]:
        p = a.probs()
        assert abs(p.sum() - 1) < 1e-9 and p[EXPERTS.index("never")] < 0.05   # staying out never wins at break-even
    # a pure bandit check: two experts, one always paid, the other never -> weight goes to the paid one
    class Paid(AlwaysRest):
        pass
    h = QueueHedgeAgent([Paid(), NeverRest()], eta=0.5, explore=0.1, reward_scale=1.0, seed=1)
    for _ in range(200):
        h.reset()
        h.update(None, None, 1.0 if h.choice == 0 else 0.0, None, True)
    assert h.probs()[0] > 0.85
