"""Heterogeneous persistent takers: reproducibility of the homogeneous model, population
statistics, and the WalletEntrant estimator."""
import hashlib
import json

import numpy as np
import pytest

from sim import FixedSpreadAgent, GlostenMilgromEnv, MarketConfig
from sim.benchmarks import BenchmarkTable, identity_monopoly_profit, type_benchmarks
from sim.entrants import WalletEntrant

EXP = dict(n_mm=2, alpha=0.3, sigma_L=8.0, tick=0.5, max_half_spread=24, redraw_v_each_period=True,
           edge_dist="exponential", edge_mean=5.0)


def _trace(cfg, n_ep=3):
    env = GlostenMilgromEnv(cfg)
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n_ep):
        env.reset()
        done = False
        while not done:
            a = rng.integers(1, cfg.max_half_spread + 1, cfg.n_mm)
            _, r, done, info = env.step(a)
            out.append((tuple(np.round(r, 10)), info["event"], info["filled"]))
    return hashlib.sha256(json.dumps(out).encode()).hexdigest()


# traces recorded before the heterogeneous-taker extension (commit dd423f4): the homogeneous
# model must not consume a single extra random draw.
PINNED = {
    "exp": ("08d432fefef69079a6545d614401de7322f2ab5d618b0460e1dcc7c02ab518b5",
            MarketConfig(**EXP, seed=3)),
    "fixed": ("04f8ca2450b5d21e09939437106cd10c9d308f9e0fb2a65164b592a7fc14448f",
              MarketConfig(n_mm=3, alpha=0.3, seed=5)),
    "persist": ("aad93419a8d5272579ead84b59de9fc835002c105d517485adece68001eabbff",
                MarketConfig(n_mm=2, alpha=0.3, redraw_v_each_period=False, seed=7)),
}


@pytest.mark.parametrize("name", list(PINNED))
def test_homogeneous_model_reproduces_pre_extension_traces(name):
    h, cfg = PINNED[name]
    assert _trace(cfg) == h


def test_wallet_population_mean_pinned_and_dispersion_monotone():
    sds = []
    for kappa in (1e6, 20, 5, 1):
        env = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_concentration=kappa, wallet_seed=0))
        assert abs(env.alpha_eff - 0.3) < 2e-3
        sds.append(env.wallet_alpha.std())
    assert sds == sorted(sds)
    # types are a function of wallet_seed only, not of the simulation seed
    a = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_seed=0, seed=1)).wallet_alpha
    b = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_seed=0, seed=2)).wallet_alpha
    assert np.array_equal(a, b)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_arrival_persistence(rho):
    env = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_persistence=rho, wallet_seed=0))
    env.reset()
    prev, rep, n = -1, 0, 0
    for _ in range(20000):
        _, _, d, info = env.step([5, 5])
        if prev >= 0:
            n += 1
            rep += info["taker"] == prev
        prev = info["taker"]
        if d:
            env.reset()
    assert abs(rep / n - (rho + (1 - rho) / 50)) < 0.02


def test_identity_revealed_only_through_prints():
    env = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_seed=0))
    env.reset()
    for _ in range(300):
        obs, _, d, info = env.step([24, 24])        # widest quote: few prints
        assert (obs["last_taker"] >= 0) == (info["event"] != "none")
        assert obs["last_taker"] == (info["taker"] if info["event"] != "none" else -1)
        if d:
            env.reset()


def test_benchmarks_use_effective_alpha_and_identity_bound_dominates():
    env = GlostenMilgromEnv(MarketConfig(**EXP, n_wallets=50, wallet_concentration=1.0,
                                         wallet_persistence=0.9, wallet_zipf=1.0, wallet_seed=0))
    _, piC, _, piM = BenchmarkTable(env).lookup(0.5)
    hC_eff, PiC_eff, hM_eff, PiM_eff = type_benchmarks(env.cfg, env.alpha_eff)
    assert abs(2 * piM - PiM_eff) < 1e-9 and abs(2 * piC - PiC_eff) < 1e-9
    assert identity_monopoly_profit(env) >= PiM_eff - 1e-12
    assert abs(identity_monopoly_profit(env, rho=0.0) - PiM_eff) < 1e-12


def _run_entrant(info_set, n_ep=300, kappa=5.0, rho=0.7):
    cfg = MarketConfig(**{**EXP, "n_mm": 3}, n_wallets=50, wallet_concentration=kappa, wallet_persistence=rho,
                       wallet_seed=0, seed=1)
    env = GlostenMilgromEnv(cfg)
    ent = WalletEntrant(cfg, info_set, wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff)
    agents = [FixedSpreadAgent(12), FixedSpreadAgent(12), ent]
    for _ in range(n_ep):
        obs = env.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step([ag.act(obs, i) for i, ag in enumerate(agents)])
    return env, ent


def test_wallet_entrant_estimator_is_consistent():
    env, ent = _run_entrant("id")
    assert abs(ent.abar_hat - env.alpha_eff) < 0.03            # arrival-level toxicity, not print-level
    est = np.array([ent.wallet_alpha_hat(j) for j in range(50)])
    assert np.corrcoef(est, env.wallet_alpha)[0, 1] > 0.95
    assert np.abs(est - env.wallet_alpha).mean() < 0.05


def test_anon_cannot_rank_wallets_but_oracle_is_exact():
    env, anon = _run_entrant("anon")
    est = np.array([anon.wallet_alpha_hat(j) for j in range(50)])
    assert np.std(est) < 1e-9                                 # one number for everyone
    env, orc = _run_entrant("oracle")
    assert np.array_equal([orc.wallet_alpha_hat(j) for j in range(50)], env.wallet_alpha)
