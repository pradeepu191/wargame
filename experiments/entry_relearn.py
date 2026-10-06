"""Entry with RE-LEARNING incumbents (RQ1/RQ2 follow-up to experiments/entry.py).

The incumbents keep their trained Q-tables but are allowed to continue learning after the
entrant arrives, with a fresh exploration schedule starting at eps_0 (a "regime change"
restart; they explore again because the world changed).  The entrant plays a fixed policy
throughout.  We track, in windows of `window` episodes:

    entrant profit / period,  incumbents' profit / period,  market-wide Delta,
    best-quote mean,  entrant fill share.

Three outcomes are possible and the trajectories distinguish them:
    re-form   : incumbents' profit recovers while the entrant's stays high -> a 3-way convention
                (turn-taking that now includes the entrant's slot)
    collapse  : everyone converges toward pi^C -> the cartel is gone
    squeeze   : incumbents re-learn to undercut the entrant specifically; entrant profit falls
                while incumbents' recovers

Usage:
  python experiments/entry_relearn.py --run cfl_exp2 --alphas 0.1,0.3,0.5 --seeds 0-4 \
      --policy markout --n-episodes 5000 --eps0 0.3 --decay 0.00001
Writes results/<run>_entry_relearn_<policy>.csv (one row per window per run) and prints a summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig, QConfig, QLearningAgent
from sim.benchmarks import collusion_index
from sim.entrants import CompetitiveEntrant, MarkoutEntrant, UndercutEntrant, hC_function


class RelearningQ(QLearningAgent):
    """Trained Q-learner that resumes learning with a fresh exploration schedule."""

    def __init__(self, Q: np.ndarray, K: int, n_mm: int, lr: float, gamma: float,
                 eps0: float, decay: float):
        super().__init__(K, n_mm, QConfig(learning_rate=lr, discount=gamma,
                                          exploration_decay=decay, inventory_buckets=Q.shape[2]), seed=0)
        self.Q = Q.copy()
        self.eps0 = eps0

    @property
    def epsilon(self) -> float:
        if self.eval:
            return 0.0
        return self.eps0 * float(np.exp(-self.cfg.exploration_decay * self.step_count))


def make_entrant(policy: str, hC: int, hC_fn):
    if policy == "competitive":
        return CompetitiveEntrant(hC)
    if policy == "undercut":
        return UndercutEntrant()
    if policy == "markout":
        return MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2)
    raise ValueError(policy)


def run_one(d: Path, policy: str, n_episodes: int, window: int, eps0: float, decay: float, seed: int):
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    n_inc = cfg["market"]["n_mm"]
    m_inc = MarketConfig(**cfg["market"])
    m_entry = MarketConfig(**{**cfg["market"], "n_mm": n_inc + 1, "seed": 20_000 + seed})
    table = BenchmarkTable(GlostenMilgromEnv(m_inc))
    hC, piC, hM, piM = table.lookup(0.5)
    # per-MM benchmarks in the (N+1)-MM market: total market profit is the same, split N+1 ways
    piC_e, piM_e = piC * n_inc / (n_inc + 1), piM * n_inc / (n_inc + 1)
    lr = float(cfg["agents"][0]["params"].get("learning_rate", 0.15))
    gamma = float(cfg["agents"][0]["params"].get("discount", 0.95))
    inc = [RelearningQ(np.load(d / f"Q_agent{i}.npy"), m_inc.max_half_spread, m_entry.n_mm, lr, gamma, eps0, decay)
           for i in range(n_inc)]
    ent = make_entrant(policy, hC, hC_function(m_inc))
    agents = inc + [ent]
    E = n_inc
    env = GlostenMilgromEnv(m_entry)

    rows = []
    acc = {"ent": 0.0, "inc": 0.0, "best": 0.0, "fills": 0, "ent_fills": 0, "periods": 0}
    for ep in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            obs_next, r, done, info = env.step(actions)
            for i, ag in enumerate(agents):
                ag.update(obs, int(actions[i]), float(r[i]), obs_next, done)
            if info["filled"] >= 0:
                acc["fills"] += 1
                if info["filled"] == E:
                    acc["ent_fills"] += 1
                    if hasattr(ent, "observe_fill"):
                        ent.observe_fill(float(r[E]))
            acc["ent"] += r[E]
            acc["inc"] += r[:E].mean()
            acc["best"] += actions.min()
            acc["periods"] += 1
            obs = obs_next
        if (ep + 1) % window == 0:
            P = acc["periods"]
            market_pnl_per_mm = (acc["ent"] + acc["inc"] * n_inc) / P / (n_inc + 1)
            rows.append({
                "episode": ep + 1, "seed": seed,
                "entrant_pnl": acc["ent"] / P, "incumbent_pnl": acc["inc"] / P,
                "delta_market": collusion_index(market_pnl_per_mm, piC_e, piM_e),
                "best_h": acc["best"] / P, "entrant_fill_share": acc["ent_fills"] / max(acc["fills"], 1),
                "epsilon": inc[0].epsilon, "piC": piC, "piM": piM,
            })
            acc = {k: 0 for k in acc}
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    ap.add_argument("--policy", default="markout", choices=["competitive", "undercut", "markout"])
    ap.add_argument("--alphas", default="0.1,0.3,0.5")
    ap.add_argument("--n-mm", type=int, default=2)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--n-episodes", type=int, default=5000)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=1e-5)
    args = ap.parse_args()
    results = Path(args.results)
    a_, b_ = map(int, args.seeds.split("-"))
    out = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        for seed in range(a_, b_ + 1):
            d = results / f"{args.run}_alpha{alpha}_n_mm{args.n_mm}_seed{seed}"
            if not (d / "Q_agent0.npy").exists():
                continue
            df = run_one(d, args.policy, args.n_episodes, args.window, args.eps0, args.decay, seed)
            df["alpha"] = alpha
            out.append(df)
            print(f"alpha={alpha} seed={seed} done", flush=True)
    if not out:
        raise SystemExit(f"no trained incumbents found under {results}/{args.run}_alpha*_n_mm{args.n_mm}_seed*/ "
                         f"-- run the replication grid first (experiments/configs/replicate_cfl_exp.yaml)")
    res = pd.concat(out)
    res.to_csv(results / f"{args.run}_entry_relearn_{args.policy}.csv", index=False)
    pd.set_option("display.width", 220)
    # summary: first window vs last window, mean over seeds
    first = res[res.episode == res.episode.min()].groupby("alpha")[["entrant_pnl", "incumbent_pnl", "delta_market", "best_h", "entrant_fill_share"]].mean()
    last = res[res.episode == res.episode.max()].groupby("alpha")[["entrant_pnl", "incumbent_pnl", "delta_market", "best_h", "entrant_fill_share"]].mean()
    print("--- first window ---"); print(first.round(3).to_string())
    print("--- last window ---"); print(last.round(3).to_string())


if __name__ == "__main__":
    main()
