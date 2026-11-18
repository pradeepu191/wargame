"""The entrant's problem as a finite Markov decision problem, in the words of Lecture 15
("Planning with a Model", 6.S974, Nov 5 2026).

Against a FROZEN cartel the entrant faces a finite MDP (Lecture 15, slide 7):

  state  x   last period's quotes (h_1, ..., h_N, h_e): the cartel's Q-learners condition on their
             own last quote and the best rival quote, so this summary predicts the next cost and
             the next state (slide 12, "what must the state contain?")
  action a   the entrant's half-spread, 1..K
  P          deterministic: the cartel answers x with g(x), the next state is (g(x), a).  The frozen
             cartel is part of P, as the missiles are in the lecture's missile-defence example
             (slide 9); a cartel that keeps learning responds to the policy and is a player instead
  reward     the entrant's expected one-period profit (closed form, sim.benchmarks.profit_curve)
  horizon    a turn limit of T = 100 periods per episode (slide 11); backward induction over the
             periods left is value iteration with k = periods left (slides 19--20)

and compares, on every trained cartel, the profit per period of five policies, each evaluated
on the model from the reset state:

  undercut     the one-tick undercut rule: one tick inside the incumbents' LAST quote; no model
  myopic       knows P, maximises today's profit only (discount 0, slide 32)
  delta=0.5, 0.9, 0.99
               knows P, maximises the discounted sum (stationary policy by value iteration)
  optimal      knows P, maximises the total over the 100-period episode (backward induction;
               the exact best response of experiments/best_response.py)

The gap between `myopic` and `optimal` is slide 15's lesson in our model: the action that earns
most today can give the policy that earns less.  The script also prints the first periods of the
myopic and optimal paths on one cartel, as a table to read like slides 19--20.

    python experiments/planning_view.py --simulate 100 --jobs 6   # all two-maker cartels found locally
    python experiments/planning_view.py --dirs "results/cfl_exp2_alpha0.1_n_mm2_seed0" --show 8
"""
from __future__ import annotations

import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig
from experiments.best_response import CartelMDP, DPEntrant, evaluate
from sim.entrants import UndercutEntrant

DELTAS = (0.5, 0.9, 0.99)


def discounted_policy(mdp: CartelMDP, delta: float, tol: float = 1e-9, max_iter: int = 20_000) -> np.ndarray:
    """Stationary policy from value iteration on V(x) = max_a { r(x, a) + delta * V(next(x, a)) }."""
    idx = mdp.base[:, None] + np.arange(mdp.K)[None, :]
    V = np.zeros(len(mdp.h))
    for _ in range(max_iter):
        q = mdp.r_own + delta * V[idx]
        V_new = q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return (mdp.r_own + delta * V[idx]).argmax(axis=1)


def walk(mdp: CartelMDP, choose, T: int):
    """Follow a policy from the reset state for T periods.  choose(t, x) -> action index (0-based).
    Returns the entrant's total profit, the incumbents' mean best quote, and the visited path."""
    x, total, quotes, path = mdp.s0, 0.0, [], []
    for t in range(T):
        a = int(choose(t, x))
        r = float(mdp.r_own[x, a])
        path.append((t, tuple(int(v) for v in mdp.g[x]), a + 1, r))
        quotes.append(int(mdp.g[x].min()))
        total += r
        x = int(mdp.base[x] + a)
    return total, float(np.mean(quotes)), path


def one_cartel(d: Path, show: int = 0, n_sim: int = 0) -> list[dict]:
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N, alpha, seed = int(mk["n_mm"]), float(mk["alpha"]), int(cfg.get("seed", 0))
    m = MarketConfig(**mk)
    K, T = m.max_half_spread, m.horizon
    hC, piC, hM, piM = BenchmarkTable(GlostenMilgromEnv(m)).lookup(0.5)
    rent = (piM - piC) * N                                    # one rent unit: monopoly minus Nash profit
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T)

    pol_opt, _, _ = mdp.solve(1.0)
    pol_myopic = mdp.r_own.argmax(axis=1)
    last_inc = mdp.h[:, :N].min(axis=1)                       # incumbents' best quote LAST period
    pol_under = np.maximum(last_inc - 1, 1) - 1
    policies = {"undercut": lambda t, x: pol_under[x], "myopic": lambda t, x: pol_myopic[x]}
    for delta in DELTAS:
        p = discounted_policy(mdp, delta)
        policies[f"delta={delta:g}"] = (lambda p: (lambda t, x: p[x]))(p)
    policies["optimal"] = lambda t, x: pol_opt[t, x] - 1

    rows, paths = [], {}
    S = len(mdp.h)
    for name, choose in policies.items():
        total, h_inc, path = walk(mdp, choose, T)
        paths[name] = path
        row = {"run": d.name, "n_mm": N, "alpha": alpha, "seed": seed, "hC": hC, "hM": hM, "policy": name,
               "entrant_per_period_norm": total / T / rent, "h_inc_mean": h_inc}
        if n_sim:
            # the same policy in the real simulator (random tie-breaking in the cartel's argmax, real takers)
            if name == "undercut":
                factory = UndercutEntrant
            else:
                table = np.array([[choose(t, x) + 1 for x in range(S)] for t in range(T)], dtype=np.int8) \
                    if name == "optimal" else np.tile(np.array([choose(0, x) + 1 for x in range(S)], dtype=np.int8), (T, 1))
                factory = (lambda tb: (lambda: DPEntrant(mdp, tb)))(table)
            pnl, _, _ = evaluate(Q, mk, K, N, seed, n_sim, factory)
            row["entrant_sim_norm"] = float(pnl[N]) / rent
        rows.append(row)
    if show:
        print(f"\n{d.name}: first {show} periods (incumbents' quotes this period -> entrant's quote, its profit today in rent units)")
        for name in ("myopic", "optimal"):
            print(f"  {name}")
            for t, inc, a, r in paths[name][:show]:
                print(f"    t={t:<3d} incumbents {str(inc):<10s} entrant {a:>2d}   profit {r / rent:5.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_n_mm2_seed*"])
    ap.add_argument("--show", type=int, default=6, help="print the first periods of the paths on the first cartel")
    ap.add_argument("--simulate", type=int, default=0, help="also play each policy for this many episodes in the real simulator")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/planning_view.csv")
    args = ap.parse_args()
    dirs = sorted({Path(p) for g in args.dirs for p in glob.glob(g) if (Path(p) / "Q_agent0.npy").exists()})
    if not dirs:
        raise SystemExit("no trained cartels matched --dirs")
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for out in ex.map(one_cartel, dirs, [args.show] + [0] * (len(dirs) - 1), [args.simulate] * len(dirs)):
            rows.extend(out)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    order = ["undercut", "myopic"] + [f"delta={x:g}" for x in DELTAS] + ["optimal"]
    tab = df.pivot_table(index="policy", columns="alpha", values="entrant_per_period_norm", aggfunc="mean").reindex(order)
    q = df.pivot_table(index="policy", columns="alpha", values="h_inc_mean", aggfunc="mean").reindex(order)
    print(f"\n{df.run.nunique()} cartels.  Entrant's profit per period, rent units (mean over cartels; columns: alpha)")
    print(tab.round(2).to_string())
    print("\nIncumbents' mean best quote along the path, ticks")
    print(q.round(1).to_string())
    if "entrant_sim_norm" in df:
        s = df.pivot_table(index="policy", columns="alpha", values="entrant_sim_norm", aggfunc="mean").reindex(order)
        print(f"\nThe same policies in the real simulator ({args.simulate} episodes each), rent units")
        print(s.round(2).to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
