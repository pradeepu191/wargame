"""Both sides learning, run until it settles: a double oracle / PSRO loop between the cartel and
the entrant.

experiments/best_response.py ran three rounds of 'the cartel re-learns against the last best
response, the entrant re-solves'.  This runs the loop for --rounds rounds and in two modes:

  last      the cartel re-learns against the most recent best response only (iterated best
            response; what the three-round run did)
  uniform   the cartel re-learns against a uniform mixture over EVERY best response found so far,
            one drawn per episode (fictitious play / PSRO with a uniform meta-strategy): an
            adversarially trained cartel

After each round the cartel is frozen and measured:
  C          its rent index alone (0 = Nash, 1 = joint monopoly)
  fresh      the exact best response to it (re-solved): the entrant's profit and the cartel's
             loss, in rent units.  The cartel's loss is its exploitability L_BR.
  stale      the previous round's best response, not re-solved
  public     the public-information attacker (identifies the reaction table from the tape with
             --probe probing periods, then solves it): the learned, realistic entrant
  undercut   the one-tick undercut rule (held out: never trained against)
The question the Workshop 2 plan asks is answered by the path of (C, L_BR) across rounds: a
cartel that learns to defend its rent moves to lower L_BR at the same C; one that only becomes
less collusive moves down the diagonal L_BR ~ C.

Usage:
  python experiments/psro.py --dirs "results/cfl_exp2_alpha*_seed[0-4]" --rounds 8 --jobs 10 --out results/psro.csv
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
from sim.entrants import UndercutEntrant
from experiments.best_response import CartelMDP, DPEntrant, evaluate, relearn
from experiments.public_attacker import IdentifyingEntrant, play as play_probe
from experiments.entry import FrozenQ


class MixtureEntrant(Agent):
    """Plays one of the given entrants per episode, drawn uniformly at reset()."""

    def __init__(self, entrants: list, seed: int = 0):
        self.entrants, self.rng = entrants, np.random.default_rng(seed)
        self.cur = entrants[0]

    def reset(self):
        self.cur = self.entrants[int(self.rng.integers(len(self.entrants)))]
        self.cur.reset()

    def act(self, obs, i):
        return self.cur.act(obs, i)


def measure(Q, mk, K, N, seed, n_eval, probe, hC, PiC, rent, stale):
    """Frozen cartel Q against the four attackers; returns a dict of rent-unit numbers and the fresh best response."""
    alone, _, _ = evaluate(Q, mk, K, N, seed, n_eval)
    Pi = float(alone.sum())
    cfg_att = MarketConfig(**{**mk, "n_mm": N + 1})
    mdp = CartelMDP(Q, cfg_att, cfg_att.horizon)
    policy, _, _ = mdp.solve(1.0)
    fresh = DPEntrant(mdp, policy)
    out = {"C": (Pi - PiC) / rent, "q_ties": mdp.ties}

    def score(name, pnl):
        out[f"{name}_entrant"] = float(pnl[N]) / rent
        out[f"{name}_loss"] = (Pi - float(pnl[:N].sum())) / rent
    score("fresh", evaluate(Q, mk, K, N, seed, n_eval, lambda: fresh)[0])
    score("undercut", evaluate(Q, mk, K, N, seed, n_eval, lambda: UndercutEntrant())[0])
    if stale is not None:
        score("stale", evaluate(Q, mk, K, N, seed, n_eval, lambda: stale)[0])
    env = GlostenMilgromEnv(MarketConfig(**{**mk, "n_mm": N + 1, "seed": 10_000 + seed}))
    ent = IdentifyingEntrant(MarketConfig(**{**mk, "n_mm": N + 1, "seed": 10_000 + seed}), N, probe, 500, "competitive", hC, seed=seed)
    _, post = play_probe(env, [FrozenQ(Q[i], K, N + 1) for i in range(N)] + [ent], n_eval + probe // cfg_att.horizon, probe, N)
    score("public", post)
    return out, fresh


def one_cartel(args):
    d, mode, rounds, relearn_episodes, n_eval, probe, eps0, decay = args
    d = Path(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N, K, alpha, seed = int(mk["n_mm"]), int(mk["max_half_spread"]), float(mk["alpha"]), int(cfg.get("seed", 0))
    p = cfg["agents"][0]["params"]
    lr, gamma = float(p.get("learning_rate", 0.15)), float(p.get("discount", 0.95))
    hC, PiC, hM, PiM = type_benchmarks(MarketConfig(**mk), alpha)
    rent = PiM - PiC
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    base = dict(run=d.name, n_mm=N, alpha=alpha, seed=seed, mode=mode, hC=hC, hM=hM, rent=rent)
    rows, population, stale = [], [], None
    for rnd in range(rounds + 1):
        if rnd > 0:
            opp = population[-1] if mode == "last" else MixtureEntrant(population, seed=1000 * rnd + seed)
            Q, _ = relearn(Q, mk, K, N, seed, opp, relearn_episodes, lr, gamma, eps0, decay, rnd)
        m, fresh = measure(Q, mk, K, N, seed, n_eval, probe, hC, PiC, rent, stale)
        rows.append({**base, "round": rnd, **m})
        population.append(fresh)
        stale = fresh
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_seed*"])
    ap.add_argument("--modes", default="last,uniform")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--relearn-episodes", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--probe", type=int, default=1000)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=1e-5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/psro.csv")
    args = ap.parse_args()
    dirs = sorted(set(p for pat in args.dirs for p in glob.glob(pat)))
    jobs = [(d, mode, args.rounds, args.relearn_episodes, args.n_eval, args.probe, args.eps0, args.decay)
            for d in dirs for mode in args.modes.split(",")]
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, r in enumerate(ex.map(one_cartel, jobs)):
            rows.extend(r)
            print(f"[{k + 1}/{len(jobs)}] {r[0]['run']} {r[0]['mode']}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    cols = ["C", "fresh_loss", "fresh_entrant", "stale_entrant", "public_entrant", "undercut_entrant"]
    for mode, g in df.groupby("mode"):
        print(f"\nmode = {mode}: mean over {g.run.nunique()} cartels, rent units")
        print(g.groupby("round")[cols].mean().round(2).to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
