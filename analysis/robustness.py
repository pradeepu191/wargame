"""Summarise the robustness grid (robust_exp2): N x learning_rate x init_q at fixed alpha.

Reads results/robust_exp2_n_mm<N>_learning_rate<lr>_init_q<q>_seed<k>/
Writes results/robust_exp2_summary.csv and prints the table.

Usage: python analysis/robustness.py [--run robust_exp2] [--tail 2000]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig
from analysis.replication import tape_dynamics
from analysis.mechanism import analyse_agent, immediate_reward_table, visited_states

PAT = re.compile(r"_n_mm(?P<n>\d+)_learning_rate(?P<lr>[0-9.]+)_init_q(?P<q>[0-9.]+)_seed(?P<seed>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="robust_exp2")
    ap.add_argument("--tail", type=int, default=2000)
    args = ap.parse_args()
    results = Path(args.results)
    rows = []
    for d in sorted(results.glob(f"{args.run}_n_mm*_seed*")):
        m = PAT.search(d.name)
        if not m or not (d / "episodes.csv").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        mcfg = MarketConfig(**cfg["market"])
        env = GlostenMilgromEnv(mcfg)
        hC, piC, hM, piM = BenchmarkTable(env).lookup(0.5)
        ep = pd.read_csv(d / "episodes.csv").tail(args.tail)
        row = {"n_mm": int(m["n"]), "lr": float(m["lr"]), "init_q": float(m["q"]), "seed": int(m["seed"]),
               "half_spread": ep.mean_quoted_spread.mean() / (2 * mcfg.tick),
               "delta": ep.delta.mean(), "hC": hC, "hM": hM, **tape_dynamics(d, hM)}
        # mechanism test, averaged over agents
        if (d / "Q_agent0.npy").exists():
            r = immediate_reward_table(env)
            tapes = sorted(d.glob("tape_ep*.csv"))
            ms = []
            for i in range(mcfg.n_mm):
                Q = np.load(d / f"Q_agent{i}.npy")
                summ, _ = analyse_agent(Q, r, visited_states(tapes, i, mcfg.n_mm), mcfg.max_half_spread)
                ms.append(summ)
            for k in ("contested_share", "restraint_given_contested", "forgone_reward", "q_gap"):
                row[k] = float(np.mean([s[k] for s in ms]))
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(results / f"{args.run}_runs.csv", index=False)
    g = runs.groupby(["n_mm", "lr", "init_q"]).agg(
        seeds=("seed", "count"), half_spread=("half_spread", "mean"), hs_std=("half_spread", "std"),
        hC=("hC", "first"), hM=("hM", "first"),
        delta=("delta", "mean"), delta_std=("delta", "std"),
        leader_switch=("leader_switch", "mean"), near_monopoly=("near_monopoly", "mean"),
        contested=("contested_share", "mean"), restraint_c=("restraint_given_contested", "mean"),
        forgone=("forgone_reward", "mean"), q_gap=("q_gap", "mean")).reset_index()
    g.to_csv(results / f"{args.run}_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
