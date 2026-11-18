"""Attacks in the one-tick queue game: steering no-regret incumbents, and the request allowance.

Part A, steering (--part steer).  The queue incumbents are EXP3 learners over complete policies
{always rest, never rest, anonymous selector} (experiments/queue_game.py; tabular Q-learners
cannot learn to queue, see REPLICATION_NOTES).  A no-regret learner can be steered by an opponent
that commits to a policy while it learns (Deng, Schneider and Sivan 2019).  Here the entrant is
present DURING the incumbents' learning with a training-time policy T in {absent, always, anon,
id}, the incumbents are then frozen, and the entrant plays a test-time policy E in {always, anon,
id}.  Reported: the entrant's profit for every (T, E), the incumbents' learned weights and
profit, and the fringe's profit.  T = E is commitment; T != E = id is lure-then-switch; T =
absent is the entry experiment of queue_game.py.

Part B, the request allowance (--part budget).  Hyperliquid limits each address's order
placements by its traded volume: a maker that places and cancels without trading runs out.  The
review of the proposal noted that one user cannot drain another's allowance, only induce it to
exhaust its own.  In the queue game a selector is exactly a maker that cancels and re-places:
BudgetQueueEnv gives every maker `budget0` placements per episode, +`earn` per fill; joining the
queue (including re-joining after a fill) costs one; cancelling is free; a maker with no
allowance left cannot join.  Incumbents are two anonymous selectors and a fringe of two
always-resting makers.  Reported per allowance and entrant policy (always, anon, id, and id with
hysteresis: once in or out it stays for at least m periods): profit per period, placements per
period, the share of periods a maker wanted to rest and could not, and the uninformed taker's
fill probability in the toxic regime.

Usage:
  python experiments/queue_attacks.py --part steer --seeds 0-2 --jobs 6 --out results/queue_steer.csv
  python experiments/queue_attacks.py --part budget --seeds 0-4 --jobs 6 --out results/queue_budget.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import MarketConfig
from sim.agents import Agent
from sim.queue_agents import AlwaysRest, NeverRest, QueueHedgeAgent
from sim.queue_env import QueueEnv
from experiments.queue_game import BASE, EXPERTS, make_expert, play_all


# ----------------------------------------------------------------------------- part A: steering
def train_with_attacker(cfg, h, n_train, seed, n_inc, fringe, attacker: Agent, experts=EXPERTS, eta=0.1):
    env = QueueEnv(cfg, half_spread=h)
    inc = [QueueHedgeAgent([make_expert(e, cfg, h, env) for e in experts], eta=eta, seed=seed * 10 + k) for k in range(n_inc)]
    agents = inc + [AlwaysRest() for _ in range(fringe)] + [attacker]
    for _ in range(n_train):
        obs = env.reset()
        for a in agents:
            a.reset()
        done = False
        while not done:
            acts = [a.act(obs, i) for i, a in enumerate(agents)]
            obs_n, r, done, _ = env.step(acts)
            for i, a in enumerate(inc):
                a.update(obs, acts[i], float(r[i]), obs_n, done)
            obs = obs_n
    for a in inc:
        a.set_eval(True)
    return inc


def entrant_of(name, cfg, h, env):
    return NeverRest() if name == "absent" else make_expert(name, cfg, h, env)


def steer_cell(args):
    alpha, rho_z, L, above, seed, n_inc, fringe, n_train, n_eval = args
    mk = dict(BASE, alpha=alpha, regime_persistence=rho_z, mark_lag=L, wallet_seed=seed)
    n = n_inc + fringe + 1
    cfg_tr = MarketConfig(**mk, n_mm=n, seed=seed)
    h = QueueEnv(MarketConfig(**mk, n_mm=n - 1, seed=seed), ticks_above_breakeven=above).h
    cfg_te = MarketConfig(**mk, n_mm=n, seed=10_000 + seed)
    rows = []
    for T in ("absent", "always", "anon", "id"):
        env_tr = QueueEnv(cfg_tr, half_spread=h)
        inc = train_with_attacker(cfg_tr, h, n_train, seed, n_inc, fringe, entrant_of(T, cfg_tr, h, env_tr))
        w = np.mean([a.probs() for a in inc], axis=0)
        for E in ("always", "anon", "id"):
            env = QueueEnv(cfg_te, half_spread=h)
            ent = entrant_of(E, cfg_te, h, env)
            pnl, fills, inf, rest, periods = play_all(env, inc + [AlwaysRest() for _ in range(fringe)] + [ent], n_eval)
            rows.append(dict(alpha=alpha, rho_z=rho_z, mark_lag=L, above=above, h=h, seed=seed, train_policy=T, test_policy=E,
                             entrant_pnl=float(pnl[-1]), incumbent_pnl=float(pnl[:n_inc].mean()),
                             fringe_pnl=float(pnl[n_inc:n_inc + fringe].mean()),
                             entrant_informed_share=float(inf[-1] / max(fills[-1], 1)),
                             incumbent_informed_share=float(inf[:n_inc].sum() / max(fills[:n_inc].sum(), 1)),
                             incumbent_rest_share=float(rest[:n_inc].mean()),
                             **{f"w_{e}": float(w[k]) for k, e in enumerate(EXPERTS)}))
    return rows


# ------------------------------------------------------------------- part B: the request allowance
class BudgetQueueEnv(QueueEnv):
    """QueueEnv with a per-maker allowance of order placements (see the module docstring)."""

    def __init__(self, cfg, half_spread=None, budget0: float = np.inf, earn: float = 1.0, **kw):
        super().__init__(cfg, half_spread=half_spread, **kw)
        self.budget0, self.earn = budget0, earn
        self.budget = np.full(cfg.n_mm, budget0, dtype=float)
        self.placed = np.zeros(cfg.n_mm); self.blocked = np.zeros(cfg.n_mm)

    def reset(self, V=None):
        self.budget[:] = self.budget0
        return super().reset(V)

    def observation(self):
        obs = super().observation()
        obs["budget"] = self.budget.copy() if hasattr(self, "budget") else None
        return obs

    def step(self, actions):
        a = [int(x) for x in actions]
        for i in range(self.cfg.n_mm):
            if a[i] >= 1 and i not in self.queue:            # a new order
                if self.budget[i] >= 1:
                    self.budget[i] -= 1; self.placed[i] += 1
                else:
                    a[i] = 0; self.blocked[i] += 1
        obs, r, done, info = super().step(a)
        f = info["filled"]
        if f >= 0:                                           # the fill earns allowance; the fresh order at the back costs one
            self.budget[f] += self.earn
            if self.budget[f] >= 1:
                self.budget[f] -= 1; self.placed[f] += 1
            else:
                self.queue.remove(f)
            obs = self.observation()
        return obs, r, done, info


class Hysteresis(Agent):
    """Holds the base policy's decision for at least m periods (minimum quote lifetime, both ways)."""

    def __init__(self, base: Agent, m: int):
        self.base, self.m = base, m
        self.cur, self.age = 0, 10 ** 9

    def reset(self):
        self.base.reset()

    def act(self, obs, i):
        want = int(self.base.act(obs, i))                    # keeps the base's filter current
        if want != self.cur and self.age >= self.m:
            self.cur, self.age = want, 0
        self.age += 1
        return self.cur


def budget_cell(args):
    alpha, rho_z, L, above, seed, fringe, n_eval, budget0, earn, m = args
    mk = dict(BASE, alpha=alpha, regime_persistence=rho_z, mark_lag=L, wallet_seed=seed)
    n_inc = 2
    n = n_inc + fringe + 1
    h = QueueEnv(MarketConfig(**mk, n_mm=n - 1, seed=seed), ticks_above_breakeven=above).h
    cfg = MarketConfig(**mk, n_mm=n, seed=10_000 + seed)
    rows = []
    for E in ("always", "anon", "id", f"id_hyst{m}"):
        env = BudgetQueueEnv(cfg, half_spread=h, budget0=budget0, earn=earn)
        inc = [make_expert("anon", cfg, h, env) for _ in range(n_inc)]
        ent = Hysteresis(make_expert("id", cfg, h, env), m) if E.startswith("id_hyst") else make_expert(E, cfg, h, env)
        liq = {}
        pnl, fills, inf, rest, periods = play_all(env, inc + [AlwaysRest() for _ in range(fringe)] + [ent], n_eval, stats=liq)
        rows.append(dict(alpha=alpha, rho_z=rho_z, mark_lag=L, above=above, h=h, seed=seed, budget0=budget0, earn=earn,
                         policy=E, entrant_pnl=float(pnl[-1]), incumbent_pnl=float(pnl[:n_inc].mean()),
                         fringe_pnl=float(pnl[n_inc:n_inc + fringe].mean()),
                         entrant_placements=float(env.placed[-1] / periods), incumbent_placements=float(env.placed[:n_inc].mean() / periods),
                         entrant_blocked=float(env.blocked[-1] / periods), incumbent_blocked=float(env.blocked[:n_inc].mean() / periods),
                         entrant_rest_share=float(rest[-1]), incumbent_rest_share=float(rest[:n_inc].mean()),
                         entrant_informed_share=float(inf[-1] / max(fills[-1], 1)),
                         incumbent_informed_share=float(inf[:n_inc].sum() / max(fills[:n_inc].sum(), 1)),
                         uninformed_fill_toxic=liq["uninformed_fill_toxic"], uninformed_fill_calm=liq["uninformed_fill_calm"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["steer", "budget"], required=True)
    ap.add_argument("--alphas", default="0.3")
    ap.add_argument("--rhos-z", default="0.9,0.99")
    ap.add_argument("--mark-lags", default="5,20")
    ap.add_argument("--above", default="0")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--fringe", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--budgets", default="inf,30,15,8,4")
    ap.add_argument("--earn", type=float, default=1.0)
    ap.add_argument("--hysteresis", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s0, s1 = map(int, args.seeds.split("-"))
    grid = [(float(a), float(rz), int(L), int(ab), s) for a in args.alphas.split(",") for rz in args.rhos_z.split(",")
            for L in args.mark_lags.split(",") for ab in args.above.split(",") for s in range(s0, s1 + 1)]
    if args.part == "steer":
        jobs = [(a, rz, L, ab, s, 2, args.fringe, args.n_train, args.n_eval) for a, rz, L, ab, s in grid]
        fn, out = steer_cell, args.out or "results/queue_steer.csv"
    else:
        jobs = [(a, rz, L, ab, s, args.fringe, args.n_eval, float(b), args.earn, args.hysteresis)
                for a, rz, L, ab, s in grid for b in args.budgets.split(",")]
        fn, out = budget_cell, args.out or "results/queue_budget.csv"
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, r in enumerate(ex.map(fn, jobs)):
            rows.extend(r)
            print(f"[{k + 1}/{len(jobs)}]", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    if args.part == "steer":
        print("\nentrant's profit per period (rows: what it played while the incumbents learned; columns: what it plays after)")
        for (rz, L), g in df.groupby(["rho_z", "mark_lag"]):
            print(f"\nrho_z = {rz}, L = {L}")
            print(g.pivot_table(index="train_policy", columns="test_policy", values="entrant_pnl").round(3).to_string())
            print(g.groupby("train_policy")[["w_always", "w_never", "w_anon", "incumbent_pnl", "fringe_pnl"]].mean().round(3).to_string())
    else:
        cols = ["entrant_pnl", "incumbent_pnl", "fringe_pnl", "entrant_placements", "incumbent_placements",
                "incumbent_blocked", "entrant_blocked", "uninformed_fill_toxic"]
        print(df.groupby(["rho_z", "mark_lag", "budget0", "policy"])[cols].mean().round(3).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
