"""The attacker with public information only: identify the cartel's reaction function from the
tape, then solve it.

The exact best response of experiments/best_response.py needs the cartel's Q tables.  But a frozen
cartel is a deterministic reaction function R_i(own, riv) -> next quote, and every one of its
values is public: each period the book shows every incumbent's new quote next to the (own, riv)
it answered, where riv = min over the others' last quotes -- and the entrant sets riv whenever its
own quote is the tightest.  So the entrant can fill in R_i cell by cell from what it sees, and
solve the same finite-horizon MDP on the estimate.  Cells it has never seen are filled with a
default rule (--default competitive: the incumbent goes to h^C there, so the planner never
counts on an unknown cell; repeat: it repeats its own last quote; wide: it goes to K).

Schedule: the entrant quotes uniformly at random for --probe periods (the probing budget, on a
clock that runs across episodes), recording transitions all the while; after that it re-solves
the DP on the current estimate every --resolve-every periods and plays the policy, still
recording.  Reported per cartel and budget: the entrant's profit and the incumbents' loss over
the whole attack window (probing included) and after probing, in rent units; the exact best
response from the true tables under the same random numbers; and coverage, the share of the
cells on the exact best response's own path that the estimate has identified.

Usage:
  python experiments/public_attacker.py --dirs "results/cfl_exp2_alpha*_seed*" --probes 0,500,2000,10000 \
      --n-episodes 1000 --jobs 8 --out results/public_attacker.csv
  python experiments/public_attacker.py --dirs results/demo_cartel --probes 0,1000 --n-episodes 300
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
from sim import GlostenMilgromEnv, MarketConfig
from sim.agents import Agent
from sim.benchmarks import type_benchmarks
from experiments.best_response import CartelMDP, DPEntrant, response_tables
from experiments.entry import FrozenQ


class IdentifyingEntrant(Agent):
    """Learns R_i(own, riv) from the public quotes and plans on the estimate.  Must be the LAST
    agent (index N)."""

    def __init__(self, cfg: MarketConfig, N: int, probe: int, resolve_every: int, default: str,
                 hC: int, seed: int = 0):
        self.cfg, self.N, self.K, self.T = cfg, N, cfg.max_half_spread, cfg.horizon
        self.probe, self.resolve_every, self.default, self.hC = probe, resolve_every, default, hC
        self.rng = np.random.default_rng(seed)
        self.R_hat = np.zeros((N, self.K + 1, self.K + 1), dtype=np.int64)   # 0 = unknown
        self.prev = None
        self.t_global = 0
        self.mdp = None
        self.policy = None
        self.n_solves = 0

    def reset(self):
        self.prev = None          # the reset state (everyone at K) is not a reaction

    def _record(self, hs):
        if self.prev is None:
            return
        for i in range(self.N):
            own = int(self.prev[i])
            riv = min(int(self.prev[j]) for j in range(len(self.prev)) if j != i)
            self.R_hat[i, own, riv] = int(hs[i])

    def _filled_table(self) -> np.ndarray:
        R = self.R_hat.copy()
        unknown = R == 0
        if self.default == "competitive":
            fill = np.full_like(R, self.hC)
        elif self.default == "wide":
            fill = np.full_like(R, self.K)
        else:                                     # repeat own last quote
            fill = np.broadcast_to(np.arange(self.K + 1)[None, :, None], R.shape)
        R[unknown] = fill[unknown]
        R[:, 0, :] = R[:, :, 0] = 1               # index 0 is never a quote
        return R

    def _resolve(self):
        self.mdp = CartelMDP(None, self.cfg, self.T, R=self._filled_table())
        self.policy, _, _ = self.mdp.solve(1.0)
        self.n_solves += 1

    def coverage(self, path_cells: set) -> float:
        known = sum(1 for (i, own, riv) in path_cells if self.R_hat[i, own, riv] > 0)
        return known / max(len(path_cells), 1)

    def act(self, obs, i):
        hs = obs["last_half_spreads"]
        self._record(hs)
        self.prev = np.array(hs, dtype=np.int64)
        t = self.t_global
        self.t_global += 1
        if t < self.probe:
            return int(self.rng.integers(1, self.K + 1))
        if self.policy is None or (t - self.probe) % self.resolve_every == 0:
            self._resolve()
        tt = min(int(obs["t"]), self.T - 1)
        return int(self.policy[tt, self.mdp.state_index(hs)])


def path_cells_of(mdp: CartelMDP, policy: np.ndarray) -> set:
    """The (i, own, riv) cells the exact best response's own trajectory asks the cartel."""
    cells = set()
    s = mdp.s0
    for t in range(mdp.T):
        h = mdp.h[s]
        for i in range(mdp.N):
            riv = int(np.delete(h, i).min())
            cells.add((i, int(h[i]), riv))
        a = int(policy[t, s])
        s = mdp.base[s] + a - 1
    return cells


def play(env, agents, n_episodes, probe, N):
    """Returns per-agent mean profit per period overall and after the probing periods."""
    n = len(agents)
    pnl, pnl_post = np.zeros(n), np.zeros(n)
    periods = periods_post = 0
    t_global = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            acts = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            obs, r, done, info = env.step(acts)
            pnl += r; periods += 1
            if t_global >= probe:
                pnl_post += r; periods_post += 1
            t_global += 1
    return pnl / periods, pnl_post / max(periods_post, 1)


def one_cartel(args):
    d, probes, n_episodes, resolve_every, default = args
    d = Path(d)
    cfg_y = yaml.safe_load(open(d / "config.yaml"))
    mk = cfg_y["market"]
    N, K = int(mk["n_mm"]), int(mk["max_half_spread"])
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    seed = int(cfg_y.get("seed", 0))
    alpha = float(mk["alpha"])
    hC, PiC, hM, PiM = type_benchmarks(MarketConfig(**mk), alpha)
    rent = PiM - PiC
    mk_att = {**mk, "n_mm": N + 1}
    cfg_att = MarketConfig(**{**mk_att, "seed": 10_000 + seed})
    # the oracle: exact best response and its path
    mdp = CartelMDP(Q, cfg_att, cfg_att.horizon)
    pol, own_dp, inc_dp = mdp.solve(1.0)
    path = path_cells_of(mdp, pol)
    env = GlostenMilgromEnv(MarketConfig(**{**mk_att, "seed": 10_000 + seed}))
    inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
    alone_env = GlostenMilgromEnv(MarketConfig(**{**mk, "seed": 10_000 + seed}))
    alone, _ = play(alone_env, [FrozenQ(Q[i], K, N) for i in range(N)], n_episodes, 0, N)
    exact, _ = play(env, inc + [DPEntrant(mdp, pol)], n_episodes, 0, N)
    rows = []
    for probe in probes:
        env = GlostenMilgromEnv(MarketConfig(**{**mk_att, "seed": 10_000 + seed}))
        ent = IdentifyingEntrant(cfg_att, N, probe, resolve_every, default, hC, seed=seed)
        full, post = play(env, inc + [ent], n_episodes, probe, N)
        rows.append(dict(run=d.name, n_mm=N, alpha=alpha, seed=seed, hC=hC, hM=hM, rent=rent,
                         C=(alone.sum() - PiC) / rent, probe=probe, default=default, resolve_every=resolve_every,
                         exact_entrant_norm=exact[N] / rent, exact_inc_loss_norm=(alone.sum() - exact[:N].sum()) / rent,
                         entrant_norm=full[N] / rent, inc_loss_norm=(alone.sum() - full[:N].sum()) / rent,
                         entrant_norm_post=post[N] / rent, inc_loss_norm_post=(alone.sum() - post[:N].sum()) / rent,
                         coverage=ent.coverage(path), cells_known=int((ent.R_hat > 0).sum()),
                         cells_total=N * K * K, n_solves=ent.n_solves, periods=n_episodes * cfg_att.horizon))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_seed*"])
    ap.add_argument("--probes", default="0,500,2000,10000")
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--resolve-every", type=int, default=500)
    ap.add_argument("--default", default="competitive", choices=["competitive", "repeat", "wide"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/public_attacker.csv")
    args = ap.parse_args()
    dirs = sorted(set(p for pat in args.dirs for p in glob.glob(pat)))
    probes = [int(x) for x in args.probes.split(",")]
    jobs = [(d, probes, args.n_episodes, args.resolve_every, args.default) for d in dirs]
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(one_cartel, jobs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("rent units, mean over cartels")
    for (a, n), g in df.groupby(["alpha", "n_mm"]):
        ex_e = g.exact_entrant_norm.mean(); ex_L = g.exact_inc_loss_norm.mean()
        print(f"\nalpha={a} N={n}  cartels={g.run.nunique()}  C={g.C.mean():.2f}  exact BR: entrant {ex_e:.2f}, cartel loses {ex_L:.2f}")
        for p, h in g.groupby("probe"):
            print(f"  probe={p:6d}  entrant {h.entrant_norm.mean():.2f} (after probing {h.entrant_norm_post.mean():.2f})  "
                  f"cartel loses {h.inc_loss_norm.mean():.2f} (after {h.inc_loss_norm_post.mean():.2f})  "
                  f"coverage of the exact path {h.coverage.mean():.2f}  cells known {h.cells_known.mean():.0f}/{h.cells_total.iloc[0]}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
