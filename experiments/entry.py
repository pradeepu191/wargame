"""Entry experiment (RQ1): a strategic entrant joins a market of frozen, trained incumbents.

For each trained run in results/<run>_alpha<a>_n_mm<N>_seed<k>/:
  1. rebuild the market with n_mm = N + 1 (the extra slot is the entrant);
  2. load the incumbents' Q-tables and play them greedily (frozen -- no further learning);
  3. play `n_episodes` episodes for each entrant policy from the same seed;
  4. record entrant PnL per period, fills won, and incumbents' PnL before/after entry.

A frozen incumbent trained at N still sees the entrant through its state (min rival
quote), so it WILL react according to its learned policy; it just does not update.
This isolates "how exploitable is the learned policy as it stands" (RQ2's exploitability
number for a one-tick undercutter) from "does the incumbent re-learn around the entrant"
(a later experiment).

Usage:
  python experiments/entry.py --run cfl_exp2 [--n-episodes 500] [--alphas 0.1,0.3,0.5] [--n-mm 2]
Writes results/<run>_entry.csv (one row per run x policy) and prints the summary.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig, QConfig, QLearningAgent
from sim.entrants import CompetitiveEntrant, MarkoutEntrant, UndercutEntrant, hC_function

PAT = re.compile(r"_alpha(?P<alpha>[0-9.]+)_n_mm(?P<n>\d+)_seed(?P<seed>\d+)$")


class FrozenQ(QLearningAgent):
    """A trained Q-learner that plays greedily and never updates."""

    def __init__(self, Q: np.ndarray, K: int, n_mm: int):
        super().__init__(K, n_mm, QConfig(inventory_buckets=Q.shape[2]), seed=0)
        self.Q = Q
        self.set_eval(True)

    def update(self, *a, **k):
        pass


def play(env, agents, n_episodes, entrant_idx, entrant=None):
    """Returns per-agent mean PnL per period, fill shares, and entrant decision mix."""
    n = env.cfg.n_mm
    pnl = np.zeros(n)
    fills = np.zeros(n)
    periods = 0
    decisions = {"undercut": 0, "competitive": 0}
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            if entrant is not None and getattr(entrant, "last_decision", None) in decisions:
                decisions[entrant.last_decision] += 1
            obs, r, done, info = env.step(actions)
            pnl += r
            periods += 1
            if info["filled"] >= 0:
                fills[info["filled"]] += 1
                if info["filled"] == entrant_idx and entrant is not None and hasattr(entrant, "observe_fill"):
                    entrant.observe_fill(float(r[entrant_idx]))
    return pnl / periods, fills / max(fills.sum(), 1), decisions, periods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--alphas", default="0.1,0.3,0.5")
    ap.add_argument("--n-mm", type=int, default=2)
    ap.add_argument("--seeds", default="0-9")
    args = ap.parse_args()
    results = Path(args.results)
    alphas = [float(a) for a in args.alphas.split(",")]
    a_, b_ = map(int, args.seeds.split("-"))
    seeds = range(a_, b_ + 1)

    rows = []
    for alpha in alphas:
        for seed in seeds:
            d = results / f"{args.run}_alpha{alpha}_n_mm{args.n_mm}_seed{seed}"
            if not (d / "Q_agent0.npy").exists():
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            m_inc = MarketConfig(**cfg["market"])                  # incumbents' market (for benchmarks)
            m_entry = MarketConfig(**{**cfg["market"], "n_mm": args.n_mm + 1, "seed": 10_000 + seed})
            hC, piC, hM, piM = BenchmarkTable(GlostenMilgromEnv(m_inc)).lookup(0.5)
            hC_fn = hC_function(m_inc)
            incumbents = [FrozenQ(np.load(d / f"Q_agent{i}.npy"), m_inc.max_half_spread, m_entry.n_mm)
                          for i in range(args.n_mm)]
            E = args.n_mm                                           # entrant index

            # --- baseline: incumbents alone (same number of episodes, same horizon)
            env0 = GlostenMilgromEnv(MarketConfig(**{**cfg["market"], "seed": 10_000 + seed}))
            inc0 = [FrozenQ(np.load(d / f"Q_agent{i}.npy"), m_inc.max_half_spread, args.n_mm)
                    for i in range(args.n_mm)]
            pnl0, share0, _, _ = play(env0, inc0, args.n_episodes, entrant_idx=-1)

            policies = {
                "competitive": CompetitiveEntrant(hC),
                "undercut": UndercutEntrant(),
                "markout": MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2),
            }
            for name, ent in policies.items():
                env = GlostenMilgromEnv(m_entry)
                agents = incumbents + [ent]
                pnl, share, dec, periods = play(env, agents, args.n_episodes, entrant_idx=E, entrant=ent)
                rows.append({
                    "alpha": alpha, "n_mm": args.n_mm, "seed": seed, "policy": name,
                    "hC": hC, "hM": hM, "piC": piC, "piM": piM,
                    "entrant_pnl": pnl[E], "entrant_fill_share": share[E],
                    "incumbent_pnl_before": pnl0.mean(), "incumbent_pnl_after": pnl[:E].mean(),
                    "undercut_share": dec["undercut"] / max(sum(dec.values()), 1) if name == "markout" else np.nan,
                    "alpha_hat_final": ent.alpha_hat if name == "markout" else np.nan,
                })
            print(f"alpha={alpha} seed={seed} done", flush=True)

    if not rows:
        raise SystemExit(f"no trained incumbents found under {results}/{args.run}_alpha*_n_mm{args.n_mm}_seed*/ "
                         f"-- run the replication grid first (experiments/configs/replicate_cfl_exp.yaml)")
    out = pd.DataFrame(rows)
    out.to_csv(results / f"{args.run}_entry.csv", index=False)
    pd.set_option("display.width", 220)
    g = out.groupby(["alpha", "policy"]).agg(
        seeds=("seed", "count"),
        entrant_pnl=("entrant_pnl", "mean"), entrant_pnl_std=("entrant_pnl", "std"),
        fill_share=("entrant_fill_share", "mean"),
        inc_before=("incumbent_pnl_before", "mean"), inc_after=("incumbent_pnl_after", "mean"),
        undercut_share=("undercut_share", "mean"), alpha_hat=("alpha_hat_final", "mean"),
        piC=("piC", "first"), piM=("piM", "first"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
