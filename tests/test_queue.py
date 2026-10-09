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
