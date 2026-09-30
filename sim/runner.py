"""Experiment runner: config -> agents -> episodes -> metrics on disk.

Every run writes to results/<run_name>/:
    config.yaml       the config actually used (with resolved seed)
    meta.json         git hash, timestamp, python/numpy versions
    episodes.csv      one row per episode (spread, profit, Delta, ...)
    tape_*.csv        per-period tape for the last `n_tape_episodes` episodes
    impulse.csv       impulse-response test (if enabled)
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .agents import (Agent, CompetitiveGMAgent, FixedSpreadAgent, GrimTriggerAgent,
                     QConfig, QLearningAgent, RandomAgent)
from .benchmarks import BenchmarkTable, collusion_index
from .env import GlostenMilgromEnv, MarketConfig
from .impulse import impulse_response


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_agents(spec: list[dict], mcfg: MarketConfig, table: BenchmarkTable, seed: int) -> list[Agent]:
    agents: list[Agent] = []
    for k, s in enumerate(spec):
        kind = s["type"]
        if kind == "q":
            agents.append(QLearningAgent(mcfg.max_half_spread, mcfg.n_mm,
                                         QConfig(**{k: (float(v) if isinstance(v, str) else v)
                                                    for k, v in s.get("params", {}).items()}), seed=seed * 1000 + k))
        elif kind == "random":
            agents.append(RandomAgent(mcfg.max_half_spread, seed=seed * 1000 + k))
        elif kind == "fixed":
            agents.append(FixedSpreadAgent(int(s["half_spread"])))
        elif kind == "competitive":
            agents.append(CompetitiveGMAgent(table))
        elif kind == "grim":
            agents.append(GrimTriggerAgent(table, punish_len=int(s.get("punish_len", 5))))
        else:
            raise ValueError(f"unknown agent type {kind}")
    assert len(agents) == mcfg.n_mm, "agents list must have n_mm entries"
    return agents


def run_episode(env: GlostenMilgromEnv, agents: list[Agent], table: BenchmarkTable,
                record_tape: bool = False):
    obs = env.reset()
    for ag in agents:
        ag.reset()
    n = env.cfg.n_mm
    profit = np.zeros(n)
    spreads, piC, piM, disc_err = [], [], [], []
    tape = []
    done = False
    while not done:
        actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
        obs_next, reward, done, info = env.step(actions)
        for i, ag in enumerate(agents):
            ag.update(obs, int(actions[i]), float(reward[i]), obs_next, done)
        profit += reward
        _, c_, _, m_ = table.lookup(info["mu_prev"])
        piC.append(c_); piM.append(m_)
        spreads.append(info["quoted_spread"])
        disc_err.append(abs(obs_next["mid"] - env.V))
        if record_tape:
            tape.append({"t": obs["t"], **{k: info[k] for k in
                         ("event", "price", "filled", "informed", "mid", "mid_next",
                          "best_ask", "best_bid", "quoted_spread", "V")},
                         **{f"h{i}": int(actions[i]) for i in range(n)}})
        obs = obs_next
    horizon = env.cfg.horizon
    mean_profit = profit.mean() / horizon        # per period, per MM (same units as pi_C, pi_M)
    row = {
        "mean_quoted_spread": float(np.mean(spreads)),
        "mean_profit_per_mm": float(mean_profit),          # per period
        "pi_C": float(np.mean(piC)),
        "pi_M": float(np.mean(piM)),
        "delta": collusion_index(mean_profit, float(np.mean(piC)), float(np.mean(piM))),
        "final_discovery_error": float(disc_err[-1]),
        "V": env.V,
    }
    for i in range(n):
        row[f"profit_total_{i}"] = float(profit[i])     # episode total
    return row, tape


def run(config_path: str, results_root: str = "results") -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg.get("seed", 0))
    mcfg = MarketConfig(**cfg["market"], seed=seed)
    env = GlostenMilgromEnv(mcfg)
    table = BenchmarkTable(env)
    agents = build_agents(cfg["agents"], mcfg, table, seed)

    n_episodes = int(cfg.get("n_episodes", 1000))
    n_tape = int(cfg.get("n_tape_episodes", 1))
    log_every = int(cfg.get("log_every", 100))

    out = Path(results_root) / f"{cfg['run_name']}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}))
    (out / "meta.json").write_text(json.dumps({
        "git_hash": git_hash(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(), "numpy": np.__version__}, indent=2))

    rows = []
    t0 = time.time()
    for ep in range(n_episodes):
        record = ep >= n_episodes - n_tape
        row, tape = run_episode(env, agents, table, record_tape=record)
        row["episode"] = ep
        rows.append(row)
        if record:
            pd.DataFrame(tape).to_csv(out / f"tape_ep{ep}.csv", index=False)
        if (ep + 1) % log_every == 0:
            recent = pd.DataFrame(rows[-log_every:])
            print(f"[{cfg['run_name']}] ep {ep+1}/{n_episodes}  "
                  f"spread={recent.mean_quoted_spread.mean():.3f}  "
                  f"profit={recent.mean_profit_per_mm.mean():.3f}  "
                  f"delta={recent.delta.mean():.3f}  ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)

    if cfg.get("impulse", {}).get("enabled", False):
        ir = impulse_response(env, agents, **{k: v for k, v in cfg["impulse"].items() if k != "enabled"})
        ir.to_csv(out / "impulse.csv", index=False)
    return out


if __name__ == "__main__":
    import sys
    print(run(sys.argv[1]))
