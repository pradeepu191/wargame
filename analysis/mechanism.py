"""Q-value mechanism test for learned collusion (replaces the impulse-response test for cycling policies).

For a converged tabular Q-learner we ask, state by state along the states its own greedy play
visits:

    a*(s) = argmax_a Q(s, a)              the action the agent actually takes
    b(s)  = argmax_a r(s, a)              the MYOPIC best response: maximises immediate expected
                                          reward given the rival's last quote (one-shot Bertrand)

If a*(s) = b(s) the agent is just best-responding period by period -- an Edgeworth cycle, no
forward-looking restraint.  If a*(s) != b(s) with r(s, b) > r(s, a*) but Q(s, a*) > Q(s, b), the
agent gives up immediate profit for continuation value: the learned value function encodes that
undercutting leads somewhere worse.  That is the reward-punishment structure that defines
collusion (Harrington 2018), read directly off the Q-table instead of inferred from a perturbation.

Only states where undercutting would pay can show restraint: if the rival quotes above the
monopoly level the myopic best response IS the monopoly quote and a* = b trivially.  We
therefore also report `contested_share`, the visit-weighted fraction of states where
b(s) < a*(s) is even possible (rival at or below a*), and `restraint_given_contested`.

We report, over visited states weighted by visit frequency:
    restraint_share     fraction of visited states where a* != b and r(s,b) > r(s,a*)
    forgone_reward      mean  r(s,b) - r(s,a*)   over restrained states (immediate profit given up)
    q_gap               mean  Q(s,a*) - Q(s,b)   over restrained states (how strongly Q disagrees)
    undercut_depth      mean  (b - a*) in ticks   over restrained states (how far below a* the
                                                   myopic response sits)

Immediate expected reward r(s, a) is computed exactly from the environment's expected-profit
machinery: with the rival at h_r and me at h_a, I earn the market-wide expected profit Pi(h_a)
if h_a < h_r, half of it if h_a == h_r, and zero if h_a > h_r (plus the inventory penalty, which
is zero in these configs).

Usage:  python analysis/mechanism.py --run cfl_exp [--results results]
Writes  results/<run>_mechanism.csv  (one row per run x agent) and prints a per-cell summary.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim import GlostenMilgromEnv, MarketConfig
from sim.benchmarks import expected_profit_by_half_spread

PAT = re.compile(r"_alpha(?P<alpha>[0-9.]+)_n_mm(?P<n>\d+)(?P<extra>.*)_seed(?P<seed>\d+)$")


def immediate_reward_table(env: GlostenMilgromEnv) -> np.ndarray:
    """r[h_rival, h_mine] = my expected one-period profit (ties split evenly among n_mm)."""
    c = env.cfg
    Pi = expected_profit_by_half_spread(env, 0.5)            # market-wide, index h-1
    K = c.max_half_spread
    r = np.zeros((K + 1, K + 1))
    for hr in range(1, K + 1):
        for ha in range(1, K + 1):
            if ha < hr:
                r[hr, ha] = Pi[ha - 1]
            elif ha == hr:
                r[hr, ha] = Pi[ha - 1] / c.n_mm                 # conservative: assume all tie
    return r


def visited_states(tapes: list[Path], agent: int, n_mm: int) -> dict[tuple[int, int], int]:
    """Count (own_last, rival_min_last) states the agent faced, from the saved tapes."""
    counts: dict[tuple[int, int], int] = {}
    for f in tapes:
        t = pd.read_csv(f)
        H = t[[f"h{i}" for i in range(n_mm)]].to_numpy()
        for row in range(1, len(H)):                            # state at t is last quotes at t-1
            own = int(H[row - 1, agent])
            riv = int(min(H[row - 1, j] for j in range(n_mm) if j != agent))
            counts[(own, riv)] = counts.get((own, riv), 0) + 1
    return counts


def analyse_agent(Q: np.ndarray, r: np.ndarray, states: dict, K: int) -> dict:
    rows = []
    for (own, riv), w in states.items():
        q = Q[own, riv, 0]
        a_star = int(np.argmax(q[1:]) + 1)
        b = int(np.argmax(r[riv, 1:]) + 1)
        restrained = (a_star != b) and (r[riv, b] > r[riv, a_star] + 1e-12)
        rows.append({"own": own, "riv": riv, "w": w, "a_star": a_star, "b": b,
                     "r_astar": r[riv, a_star], "r_b": r[riv, b],
                     "q_astar": q[a_star], "q_b": q[b], "restrained": restrained})
    df = pd.DataFrame(rows)
    W = df.w.sum()
    rs = df[df.restrained]
    wr = rs.w.sum()
    contested = df[df.riv <= df.a_star]          # rival at/below my greedy quote: undercutting is on the table
    wc = contested.w.sum()
    return {
        "n_states": len(df),
        "contested_share": float(wc / W),
        "restraint_given_contested": float(contested.restrained.mul(contested.w).sum() / wc) if wc else float("nan"),
        "restraint_share": float(wr / W),
        "forgone_reward": float(((rs.r_b - rs.r_astar) * rs.w).sum() / wr) if wr else 0.0,
        "q_gap": float(((rs.q_astar - rs.q_b) * rs.w).sum() / wr) if wr else 0.0,
        "undercut_depth": float(((rs.a_star - rs.b) * rs.w).sum() / wr) if wr else 0.0,
        "mean_astar": float((df.a_star * df.w).sum() / W),
        "mean_b": float((df.b * df.w).sum() / W),
    }, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp")
    ap.add_argument("--dump-states", action="store_true", help="also write per-state tables")
    args = ap.parse_args()
    results = Path(args.results)
    out_rows = []
    for d in sorted(results.glob(f"{args.run}_alpha*_n_mm*_seed*")):
        m = PAT.search(d.name)
        if not m or not (d / "Q_agent0.npy").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        mcfg = MarketConfig(**cfg["market"])
        env = GlostenMilgromEnv(mcfg)
        r = immediate_reward_table(env)
        tapes = sorted(d.glob("tape_ep*.csv"))
        for i in range(mcfg.n_mm):
            Q = np.load(d / f"Q_agent{i}.npy")
            states = visited_states(tapes, i, mcfg.n_mm)
            summ, df = analyse_agent(Q, r, states, mcfg.max_half_spread)
            out_rows.append({"alpha": float(m["alpha"]), "n_mm": int(m["n"]), "extra": m["extra"],
                             "seed": int(m["seed"]), "agent": i, **summ})
            if args.dump_states:
                df.sort_values("w", ascending=False).to_csv(d / f"mechanism_agent{i}.csv", index=False)
    out = pd.DataFrame(out_rows)
    out.to_csv(results / f"{args.run}_mechanism.csv", index=False)
    pd.set_option("display.width", 200)
    g = out.groupby(["alpha", "n_mm", "extra"]).agg(
        runs=("seed", "nunique"),
        contested=("contested_share", "mean"), restraint_c=("restraint_given_contested", "mean"),
        restraint_share=("restraint_share", "mean"), restraint_std=("restraint_share", "std"),
        forgone_reward=("forgone_reward", "mean"), q_gap=("q_gap", "mean"),
        undercut_depth=("undercut_depth", "mean"), mean_astar=("mean_astar", "mean"), mean_b=("mean_b", "mean"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
