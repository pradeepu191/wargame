"""Strategizing against a no-regret market maker (Deng, Schneider and Sivan, NeurIPS 2019).

The learners of the paper are the stateless EXP3 quoters of sim.agents.Exp3QuoteAgent: mean-based
no-regret, bandit feedback, one update per period, no memory of the rivals' quotes.  Two
questions, both in rent units (1 = the monopoly rent Pi^M - Pi^C of the market):

1. Does a no-regret cartel earn a rent at all?  N EXP3 quoters in self-play, measured after
   training while they keep learning (they are never frozen).  A single learner (N = 1) should
   find the monopoly quote; two should compete, because without memory there is no threat.

2. How much can an optimizer take from a LIVE no-regret learner, against the Stackelberg
   benchmark?  The theorem: at least the Stackelberg value against any no-regret learner, no more
   than it against a no-swap-regret one, strictly more against a mean-based one with > 2 actions,
   and the optimal play is a control problem.  The attackers, all entering the trained market
   while the learners keep updating:
     undercut   -- one tick inside the incumbents' last best quote (the hand rule of entry.py);
                   undercut_floor is the same rule floored at h^C
     cycle      -- the control problem approximated by a two-phase cycle: quote h_lure for T_lure
                   periods (the learner's estimates of the wide quotes rise), then h_under for
                   T_under periods (it is undercut while they decay); the cycle is chosen by grid
                   search on selection seeds and evaluated on held-out seeds
     qlearn     -- a tabular Q-learning entrant, state (own last, min rival last), learning online
   Reported per attacker: the entrant's profit, the incumbents' loss relative to playing alone,
   and both over the whole attack window (the transient is part of the exploitation) and over its
   second half.  The Stackelberg value (sim.benchmarks.stackelberg_value, strong tie-break) is
   printed beside them.

Usage:
  python experiments/no_regret.py --alphas 0.3 --n-inc 1,2 --etas 0.05 --seeds 0-7 \
      --train-episodes 3000 --attack-episodes 2000 --jobs 8 --out results/no_regret.csv
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import Exp3QuoteAgent, GlostenMilgromEnv, MarketConfig, QConfig, QLearningAgent
from sim.agents import Agent
from sim.benchmarks import stackelberg_value, type_benchmarks
from sim.entrants import UndercutEntrant

K = 24


def market(alpha: float, n_mm: int, seed: int) -> MarketConfig:
    return MarketConfig(alpha=alpha, n_mm=n_mm, max_half_spread=K, horizon=100, tick=0.5, sigma_L=8.0,
                        edge_dist="exponential", edge_mean=5.0, redraw_v_each_period=True, seed=seed)


class LureUndercutEntrant(Agent):
    """Quote h_lure for T_lure periods, then h_under for T_under periods, and repeat; the clock
    runs across episodes (the learner's weights do too)."""

    def __init__(self, h_lure: int, T_lure: int, h_under: int, T_under: int):
        self.h_lure, self.T_lure, self.h_under, self.T_under = h_lure, T_lure, h_under, T_under
        self.t = 0

    def act(self, obs, i):
        phase = self.t % (self.T_lure + self.T_under)
        self.t += 1
        return self.h_lure if phase < self.T_lure else self.h_under


def run_episodes(env, agents, n_episodes, learn=True):
    """Mean profit per period per agent over the whole window and over its second half."""
    n = len(agents)
    pnl = np.zeros(n)
    pnl_late = np.zeros(n)
    periods = periods_late = 0
    for ep in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            acts = [ag.act(obs, i) for i, ag in enumerate(agents)]
            obs2, r, done, info = env.step(acts)
            if learn:
                for i, ag in enumerate(agents):
                    ag.update(obs, acts[i], r[i], obs2, done)
            obs = obs2
            pnl += r
            periods += 1
            if ep >= n_episodes // 2:
                pnl_late += r
                periods_late += 1
    return pnl / periods, pnl_late / max(periods_late, 1)


def train_cartel(alpha, n_inc, eta, seed, n_episodes):
    env = GlostenMilgromEnv(market(alpha, n_inc, seed))
    inc = [Exp3QuoteAgent(K, eta=eta, seed=1000 * seed + i) for i in range(n_inc)]
    run_episodes(env, inc, n_episodes)
    return inc


def attack(alpha, n_inc, seed, inc, entrant, n_episodes):
    """The trained incumbents (copied, still learning) plus the entrant at index n_inc, under the
    attack-window random numbers of this seed."""
    env = GlostenMilgromEnv(market(alpha, n_inc + 1, 10_000 + seed))
    agents = [copy.deepcopy(a) for a in inc] + [entrant]
    full, late = run_episodes(env, agents, n_episodes)
    return full, late


def one_cell(args):
    alpha, n_inc, eta, seed, n_train, n_attack, cycle_grid, cycle_choice = args
    hC, PiC, hM, PiM = type_benchmarks(market(alpha, n_inc, 0), alpha)
    rent = PiM - PiC
    sv = stackelberg_value(market(alpha, n_inc, 0), alpha, n_inc)["strong"]["value"]
    inc = train_cartel(alpha, n_inc, eta, seed, n_train)
    # alone, still learning, under the attack-window random numbers (common random numbers)
    env = GlostenMilgromEnv(market(alpha, n_inc, 10_000 + seed))
    alone_full, alone_late = run_episodes(env, [copy.deepcopy(a) for a in inc], n_attack)
    base = dict(alpha=alpha, n_inc=n_inc, eta=eta, seed=seed, hC=hC, hM=hM, PiC=PiC, PiM=PiM, rent=rent,
                stackelberg_norm=sv / rent, C_alone=(alone_full.sum() - PiC) / rent,
                C_alone_late=(alone_late.sum() - PiC) / rent)
    rows = []

    def record(name, full, late, **extra):
        rows.append({**base, "attacker": name,
                     "entrant_norm": full[n_inc] / rent, "entrant_norm_late": late[n_inc] / rent,
                     "inc_loss_norm": (alone_full.sum() - full[:n_inc].sum()) / rent,
                     "inc_loss_norm_late": (alone_late.sum() - late[:n_inc].sum()) / rent,
                     "inc_profit_norm": (full[:n_inc].sum() - PiC) / rent, **extra})

    full, late = attack(alpha, n_inc, seed, inc, UndercutEntrant(h_min=1), n_attack)
    record("undercut", full, late)
    full, late = attack(alpha, n_inc, seed, inc, UndercutEntrant(h_min=hC), n_attack)
    record("undercut_floor", full, late)          # the same rule, never inside break-even
    qe = QLearningAgent(K, n_inc + 1, QConfig(learning_rate=0.15, discount=0.95, exploration_decay=3e-5,
                                             inventory_buckets=1), seed=77 + seed)
    full, late = attack(alpha, n_inc, seed, inc, qe, n_attack)
    record("qlearn", full, late)
    # the cycle: every grid point on selection seeds; only the chosen cycle on evaluation seeds
    grid = cycle_grid if cycle_choice is None else [cycle_choice]
    for (h_lure, T_lure, h_under, T_under) in grid:
        full, late = attack(alpha, n_inc, seed, inc, LureUndercutEntrant(h_lure, T_lure, h_under, T_under), n_attack)
        record("cycle", full, late, h_lure=h_lure, T_lure=T_lure, h_under=h_under, T_under=T_under,
               selection=cycle_choice is None)
    return rows


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3")
    ap.add_argument("--n-inc", default="1,2")
    ap.add_argument("--etas", default="0.05")
    ap.add_argument("--seeds", default="0-7", help="first --n-select seeds choose the cycle; the rest evaluate it")
    ap.add_argument("--n-select", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--attack-episodes", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/no_regret.csv")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    n_incs = [int(x) for x in args.n_inc.split(",")]
    etas = [float(x) for x in args.etas.split(",")]
    seeds = parse_seeds(args.seeds)
    sel, ev = seeds[:args.n_select], seeds[args.n_select:]

    def grid_for(alpha, n_inc):
        hC, _, hM, _ = type_benchmarks(market(alpha, n_inc, 0), alpha)
        return list(itertools.product([K], [20, 100, 400], [max(hC, hM - 1), max(hC, hM - 4)], [20, 100, 400]))

    rows = []
    # --- selection
    jobs = [(a, n, e, s, args.train_episodes, args.attack_episodes, grid_for(a, n), None)
            for a in alphas for n in n_incs for e in etas for s in sel]
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(one_cell, jobs):
            rows.extend(r)
    d = pd.DataFrame(rows)
    choice = {}
    for (a, n, e), g in d[d.attacker == "cycle"].groupby(["alpha", "n_inc", "eta"]):
        m = g.groupby(["h_lure", "T_lure", "h_under", "T_under"]).entrant_norm.mean()
        choice[(a, n, e)] = tuple(int(x) for x in m.idxmax())
        print(f"alpha={a} n_inc={n} eta={e}: chosen cycle (h_lure, T_lure, h_under, T_under) = {choice[(a, n, e)]}, "
              f"entrant {m.max():.2f} rent units on selection seeds")
    # --- evaluation
    if ev:
        jobs = [(a, n, e, s, args.train_episodes, args.attack_episodes, None, choice[(a, n, e)])
                for a in alphas for n in n_incs for e in etas for s in ev]
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(one_cell, jobs):
                rows.extend(r)
        d = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    # --- summary on evaluation seeds (or selection seeds if there are none)
    e = d[d.seed.isin(ev)] if ev else d
    e = e[(e.attacker != "cycle") | (~e.selection.fillna(False).astype(bool))] if "selection" in e else e
    print("\nrent units; evaluation seeds" if ev else "\nrent units; selection seeds (no held-out seeds)")
    for (a, n, eta), g in e.groupby(["alpha", "n_inc", "eta"]):
        print(f"\nalpha={a}  n_inc={n}  eta={eta}  hC={int(g.hC.iloc[0])} hM={int(g.hM.iloc[0])}  "
              f"cartel alone C={g.C_alone.mean():.2f} (late {g.C_alone_late.mean():.2f})  "
              f"Stackelberg={g.stackelberg_norm.iloc[0]:.2f}")
        for name, h in g.groupby("attacker"):
            print(f"  {name:9s} entrant {h.entrant_norm.mean():+.2f} (late {h.entrant_norm_late.mean():+.2f})   "
                  f"incumbents lose {h.inc_loss_norm.mean():+.2f} (late {h.inc_loss_norm_late.mean():+.2f})   n={len(h)}")
    print("wrote", out)


if __name__ == "__main__":
    main()
