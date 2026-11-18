"""The ordinary trader (experiments/retail_trader.py)."""
import numpy as np

from experiments.retail_trader import evaluate_exact, hurdle, solve_speculator
from sim import GlostenMilgromEnv, MarketConfig


def test_directional_bet_against_the_simulator_pays_the_formula():
    """A market order that is right on the direction with probability p, at half-spread x, earns
    (2p - 1) mu - x against the simulator's own values and mid."""
    cfg = MarketConfig(alpha=0.1, sigma_L=8.0, tick=0.5, max_half_spread=24, n_mm=2, edge_dist="exponential",
                       edge_mean=5.0, redraw_v_each_period=True, seed=7)
    env = GlostenMilgromEnv(cfg)
    rng = np.random.default_rng(1)
    h, p = 2, 0.6
    x = h * cfg.tick
    pnl = []
    env.reset()
    for _ in range(40_000):
        _, _, done, info = env.step(np.array([h, h]))
        move = info["V_t"] - info["mid"]
        side = np.sign(move) if rng.random() < p else -np.sign(move)
        pnl.append(side * move - x)
        if done:
            env.reset()
    assert abs(np.mean(pnl) - ((2 * p - 1) * cfg.edge_mean - x)) < 0.15


def test_myopic_rule_trades_iff_belief_beats_the_hurdle():
    mu, x, a, b = 5.0, 0.5, 6.0, 4.0
    trade, _ = solve_speculator(a, b, x, mu, delta=0.0, N=60)
    for n in range(60):
        for k in range(n + 1):
            p_hat = (a + k) / (a + b + n)
            assert trade[n, k] == (p_hat > hurdle(x, mu) + 1e-9)


def test_patience_never_trades_less_and_values_are_consistent():
    mu, x = 5.0, 0.5
    t0, v0 = solve_speculator(6, 4, x, mu, 0.0, N=80)
    t1, v1 = solve_speculator(6, 4, x, mu, 0.99, N=80)
    assert v1 >= v0 >= 0
    assert np.all(t1 | ~t0)                                     # wherever the myopic trader trades, so does the patient one


def test_exact_propagation():
    mu, x, N = 5.0, 0.5, 50
    g = evaluate_exact(None, 0.5, x, mu, N=N)                  # the gambler trades every day
    assert np.isclose(g["exp_trades"], N) and np.isclose(g["exp_profit_mu"], -N * x / mu)
    trade, _ = solve_speculator(6, 4, x, mu, 0.0, N=N)
    e = evaluate_exact(trade, 0.5, x, mu, N=N)
    assert 0 < e["exp_trades"] < N and 0 <= e["p_still_trading_at_end"] <= 1
