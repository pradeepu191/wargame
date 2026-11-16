"""The public-information attacker against a cartel that keeps learning.

experiments/public_attacker.py closed the oracle caveat for FROZEN cartels: the reaction function
is public and 1000 probing periods identify it.  A cartel that resumes Q-learning under attack is
a moving target in two ways: its greedy reaction function drifts as the Q-tables update, and
while it explores (epsilon > 0) a single observed answer may be an exploration move, not the
reaction.  This experiment measures what that costs the attacker, and whether forgetting helps.

The incumbents resume learning with a fresh exploration schedule (eps0, decay), exactly as in the
double oracle of experiments/best_response.py, for --n-episodes; then they are frozen and the
attack continues for --n-frozen episodes.  Attackers, each against the same cartel, seed and
schedule:

  stale        exact best response to the cartel's ORIGINAL tables, never re-solved
  fresh        oracle: re-solves from the cartel's current tables every --resolve-every periods
               (the upper reference: it needs the Q-tables)
  public       the attacker of public_attacker.py: probes at random for --probe periods, keeps the
               last observed answer of every cell for ever, re-solves every --resolve-every periods
  public_wW    the same with a forgetting window: only answers seen in the last W periods count,
               a cell's estimate is the answer seen most often in the window (so one exploration
               move does not overwrite it), and the attacker keeps probing at rate --probe-eps

Reported per cartel, attacker and block of --block episodes: the entrant's profit and the
incumbents' loss in rent units (loss against the frozen cartel's profit alone), the incumbents'
mean best quote, their exploration rate, and how much of the cartel's current reaction table the
attacker's estimate gets right (agreement over all cells and over cells it claims to know).

Usage:
  python experiments/public_attacker_relearn.py --dirs "results/cfl_exp2_alpha*_seed*" --jobs 10 \
      --out results/public_attacker_relearn.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import GlostenMilgromEnv, MarketConfig
from sim.agents import Agent
from sim.benchmarks import type_benchmarks
from experiments.best_response import CartelMDP, DPEntrant, response_tables
from experiments.entry import FrozenQ, play
from experiments.entry_relearn import RelearningQ
from experiments.public_attacker import IdentifyingEntrant


class WindowedIdentifyingEntrant(IdentifyingEntrant):
    """IdentifyingEntrant with a forgetting window: R_hat[i, own, riv] is the answer seen most
    often in the last `window` periods (ties: the most recent), unknown if not seen in the
    window.  Keeps probing at rate probe_eps after the initial budget."""

    def __init__(self, cfg, N, probe, resolve_every, default, hC, window: int, probe_eps: float = 0.0, seed: int = 0):
        super().__init__(cfg, N, probe, resolve_every, default, hC, seed=seed)
        self.window, self.probe_eps = window, probe_eps
        self.counts = np.zeros((N, self.K + 1, self.K + 1, self.K + 1), dtype=np.int32)
        self.log: deque = deque()

    def _recount(self, i, own, riv, last=None):
        c = self.counts[i, own, riv]
        m = int(c.max())
        if m == 0:
            self.R_hat[i, own, riv] = 0
        elif last is not None and c[last] == m:
            self.R_hat[i, own, riv] = last
        elif c[self.R_hat[i, own, riv]] < m:
            self.R_hat[i, own, riv] = int(c.argmax())

    def _record(self, hs):
        t = self.t_global
        while self.log and self.log[0][0] <= t - self.window:
            _, i, own, riv, a = self.log.popleft()
            self.counts[i, own, riv, a] -= 1
            self._recount(i, own, riv)
        if self.prev is None:
            return
        for i in range(self.N):
            own = int(self.prev[i])
            riv = min(int(self.prev[j]) for j in range(len(self.prev)) if j != i)
            a = int(hs[i])
            self.counts[i, own, riv, a] += 1
            self.log.append((t, i, own, riv, a))
            self._recount(i, own, riv, last=a)

    def act(self, obs, i):
        a = super().act(obs, i)
        if self.t_global > self.probe and self.probe_eps > 0 and self.rng.random() < self.probe_eps:
            return int(self.rng.integers(1, self.K + 1))
        return a


class FreshOracle(Agent):
    """Re-solves the exact best response from the incumbents' current Q-tables every `every` periods."""

    def __init__(self, incumbents, cfg: MarketConfig, every: int):
        self.inc, self.cfg, self.every = incumbents, cfg, every
        self.t_global = 0
        self.mdp = self.policy = None

    def act(self, obs, i):
        if self.policy is None or self.t_global % self.every == 0:
            self.mdp = CartelMDP([ag.Q for ag in self.inc], self.cfg, self.cfg.horizon)
            self.policy, _, _ = self.mdp.solve(1.0)
        self.t_global += 1
        return int(self.policy[min(int(obs["t"]), self.cfg.horizon - 1), self.mdp.state_index(obs["last_half_spreads"])])


def agreement(ent, incumbents, K):
    """Share of the cartel's current greedy reaction table the estimate gets right: over all cells,
    and over the cells the attacker claims to know."""
    if not hasattr(ent, "R_hat"):
        return np.nan, np.nan
    R, _ = response_tables([ag.Q for ag in incumbents], K)
    known = ent.R_hat[:, 1:, 1:] > 0
    right = ent.R_hat[:, 1:, 1:] == R[:, 1:, 1:]
    return float((right & known).mean()), float(right[known].mean()) if known.any() else np.nan


def one_cartel(args):
    d, n_episodes, n_frozen, block, probe, resolve_every, windows, probe_eps, eps0, decay = args
    d = Path(d)
    cfg_y = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg_y["market"])
    N, K = int(mk["n_mm"]), int(mk["max_half_spread"])
    seed, alpha = int(cfg_y.get("seed", 0)), float(mk["alpha"])
    p = cfg_y["agents"][0]["params"]
    lr, gamma = float(p.get("learning_rate", 0.15)), float(p.get("discount", 0.95))
    Q0 = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    hC, PiC, hM, PiM = type_benchmarks(MarketConfig(**mk), alpha)
    rent = PiM - PiC
    env0 = GlostenMilgromEnv(MarketConfig(**{**mk, "seed": 10_000 + seed}))
    pnl0, _, _, _ = play(env0, [FrozenQ(Q0[i], K, N) for i in range(N)], 300, entrant_idx=-1)
    Pi_nom = float(pnl0.sum())
    cfg_att = MarketConfig(**{**mk, "n_mm": N + 1, "seed": 20_000 + seed})
    names = ["stale", "fresh", "public"] + [f"public_w{w}" for w in windows]
    rows = []
    for name in names:
        env = GlostenMilgromEnv(cfg_att)
        inc = [RelearningQ(Q0[i], K, N + 1, lr, gamma, eps0, decay) for i in range(N)]
        for i, ag in enumerate(inc):
            ag.rng = np.random.default_rng(30_000 + 10 * seed + i)
        if name == "stale":
            mdp = CartelMDP(Q0, cfg_att, cfg_att.horizon)
            ent = DPEntrant(mdp, mdp.solve(1.0)[0])
        elif name == "fresh":
            ent = FreshOracle(inc, cfg_att, resolve_every)
        elif name == "public":
            ent = IdentifyingEntrant(cfg_att, N, probe, resolve_every, "competitive", hC, seed=seed)
        else:
            ent = WindowedIdentifyingEntrant(cfg_att, N, probe, resolve_every, "competitive", hC,
                                             window=int(name.split("_w")[1]), probe_eps=probe_eps, seed=seed)
        agents = inc + [ent]
        acc = np.zeros(N + 1); best = 0.0; periods = 0
        for ep in range(n_episodes + n_frozen):
            if ep == n_episodes:
                for ag in inc:
                    ag.set_eval(True)
            obs = env.reset()
            for ag in agents:
                ag.reset()
            done = False
            while not done:
                acts = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
                obs_next, r, done, _ = env.step(acts)
                for i, ag in enumerate(inc):
                    ag.update(obs, int(acts[i]), float(r[i]), obs_next, done)
                acc += r; best += acts[:N].min(); periods += 1
                obs = obs_next
            if (ep + 1) % block == 0 or ep + 1 == n_episodes + n_frozen:
                agree_all, agree_known = agreement(ent, inc, K)
                rows.append(dict(run=d.name, n_mm=N, alpha=alpha, seed=seed, hC=hC, hM=hM, rent=rent,
                                 C=(Pi_nom - PiC) / rent, attacker=name, episode=ep + 1,
                                 phase="relearning" if ep < n_episodes else "frozen_after",
                                 entrant_norm=acc[N] / periods / rent,
                                 inc_loss_norm=(Pi_nom - acc[:N].sum() / periods) / rent,
                                 inc_rent_norm=(acc[:N].sum() / periods - PiC) / rent,
                                 h_inc=best / periods, epsilon=inc[0].epsilon,
                                 agree_all=agree_all, agree_known=agree_known,
                                 probe=probe, resolve_every=resolve_every, probe_eps=probe_eps, eps0=eps0, decay=decay))
                acc[:] = 0; best = 0.0; periods = 0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_seed*"])
    ap.add_argument("--n-episodes", type=int, default=3000, help="episodes during which the cartel re-learns")
    ap.add_argument("--n-frozen", type=int, default=500, help="episodes after the cartel is frozen again")
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--probe", type=int, default=1000)
    ap.add_argument("--resolve-every", type=int, default=500)
    ap.add_argument("--windows", default="2000,20000")
    ap.add_argument("--probe-eps", type=float, default=0.02)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=1e-5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/public_attacker_relearn.csv")
    args = ap.parse_args()
    dirs = sorted(set(p for pat in args.dirs for p in glob.glob(pat)))
    windows = [int(w) for w in args.windows.split(",") if w]
    jobs = [(d, args.n_episodes, args.n_frozen, args.block, args.probe, args.resolve_every, windows,
             args.probe_eps, args.eps0, args.decay) for d in dirs]
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, r in enumerate(ex.map(one_cartel, jobs)):
            rows.extend(r)
            print(f"[{k + 1}/{len(jobs)}] {r[0]['run']}", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\nentrant's profit, rent units, mean over cartels (rows: attacker; columns: episode block end)")
    print(df.pivot_table(index="attacker", columns="episode", values="entrant_norm").round(2).to_string())
    print("\nincumbents' rent kept (their profit above Nash, rent units)")
    print(df.pivot_table(index="attacker", columns="episode", values="inc_rent_norm").round(2).to_string())
    print("\nshare of the cartel's current reaction table the estimate gets right (all cells / known cells)")
    print(df.pivot_table(index="attacker", columns="episode", values="agree_all").round(2).dropna().to_string())
    print(df.pivot_table(index="attacker", columns="episode", values="agree_known").round(2).dropna().to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
