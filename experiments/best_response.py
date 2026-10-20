"""The exact best response to a frozen cartel: L_BR by dynamic programming, not by a learned
approximation.

A frozen cfl_exp2 cartel is a deterministic reaction function: incumbent i's next half-spread is
argmax_a Q_i[own last h, min rival last h, a] (no inventory state, i.i.d. periods).  Against it
the entrant faces a finite-horizon MDP whose state is last period's quote vector
s = (h_1, ..., h_N, h_e), whose transition is deterministic -- the incumbents answer s with
g(s), the entrant picks a, the next state is (g(s), a) -- and whose one-period rewards are the
closed-form expected profits of sim.benchmarks.profit_curve: the makers at the best quote split
Pi(h_min).  Backward induction over the 100-period episode from the reset state (everyone at K)
gives the entrant's optimal policy exactly, for any linear objective

    r_lambda = lambda * (own profit) - (1 - lambda) * (incumbents' profit)

lambda = 1 is the entrant's own best response (the true L_BR of the exploitability section,
against which the hand-crafted undercut / markout rules are lower bounds); lambda = 0 is the
damage-maximiser; in between is the attacker's profit--damage frontier.  The DP value is checked
by simulating the policy in the real environment with the real FrozenQ incumbents (ties in the
incumbents' argmax are broken by first index in the DP and at random in the simulator; the
check reports how far apart the two are).

Diagnostics of HOW the exact best response plays: the share of periods it quotes inside the
incumbents' best (undercut), at it (match) or outside it (lure), and the incumbents' mean quote
under attack vs alone -- an attacker that steers the reaction function quotes outside to make
the cartel re-widen and then undercuts.

Double oracle (--relearn-rounds R > 0).  The cartel is then allowed to answer: it re-learns for
--relearn-episodes against the fixed best-response entrant (fresh exploration schedule, as in
experiments/entry_relearn.py), is frozen again, and the entrant re-solves against the re-learned
cartel.  Per round: the re-learned cartel's rent alone C, what the STALE attacker still takes
from it (how much the cartel's answer bought), and what the FRESH best response takes.  A cartel
that can defend its rent shows L falling across rounds with C intact; one that cannot shows
either L staying at C or C itself collapsing.

Usage:
  python experiments/best_response.py --dirs "results/cfl_exp2_alpha*_seed*" --lambdas 1,0.5,0 \
      --n-episodes 500 --jobs 8 --out results/best_response.csv
  python experiments/best_response.py --dirs "results/cfl_exp2_alpha*_seed*" --lambdas 1 \
      --relearn-rounds 3 --relearn-episodes 3000 --out results/best_response_relearn.csv
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
from sim.agents import Agent
from sim.benchmarks import profit_curve
from sim.entrants import MarkoutEntrant, UndercutEntrant, hC_function
from experiments.entry import FrozenQ, play
from experiments.entry_relearn import RelearningQ


def response_tables(Q: list[np.ndarray], K: int) -> tuple[np.ndarray, int]:
    """R[i, own, riv] = incumbent i's greedy half-spread; and the number of (i, own, riv) cells
    with a tie at the top (where the simulator picks at random and the DP picks the first)."""
    N = len(Q)
    R = np.zeros((N, K + 1, K + 1), dtype=np.int64)
    ties = 0
    for i in range(N):
        q = Q[i][:, :, 0, 1:]                                  # own, riv, action-1
        R[i] = q.argmax(axis=2) + 1
        top = q.max(axis=2, keepdims=True)
        ties += int(((q == top).sum(axis=2) > 1)[1:, 1:].sum())
    return R, ties


class CartelMDP:
    """States are last period's quotes (h_1..h_N, h_e) in 1..K, flattened row-major with the
    entrant's quote as the fastest index, so that next_state(s, a) = base[s] + (a - 1)."""

    def __init__(self, Q: list[np.ndarray], cfg: MarketConfig, horizon: int):
        self.N, self.K, self.T = len(Q), cfg.max_half_spread, horizon
        N, K = self.N, self.K
        self.R, self.ties = response_tables(Q, K)
        self.Pi = profit_curve(cfg, cfg.alpha)                 # Pi[h-1], market-wide, one period
        grids = np.meshgrid(*[np.arange(1, K + 1)] * (N + 1), indexing="ij")
        self.h = np.stack([g.ravel() for g in grids], axis=1)  # (S, N+1): columns 0..N-1 incumbents, N entrant
        S = len(self.h)
        # the incumbents' answer to each state
        g = np.zeros((S, N), dtype=np.int64)
        for i in range(N):
            others = np.delete(self.h, i, axis=1)
            riv = others.min(axis=1)
            g[:, i] = self.R[i, self.h[:, i], riv]
        self.g = g
        self.base = np.ravel_multi_index([g[:, i] - 1 for i in range(N)] + [np.zeros(S, dtype=np.int64)], (K,) * (N + 1))
        # one-period expected rewards for every (state, entrant action): (S, K)
        a = np.arange(1, K + 1)[None, :]
        inc_min = g.min(axis=1)[:, None]
        hmin = np.minimum(inc_min, a)
        n_inc_best = (g[:, :, None] == hmin[:, None, :]).sum(axis=1)   # (S, K)
        e_best = (a == hmin)
        n_best = n_inc_best + e_best
        pi = self.Pi[hmin - 1]
        self.r_own = np.where(e_best, pi / n_best, 0.0)
        self.r_inc = pi * n_inc_best / n_best
        self.s0 = int(np.ravel_multi_index([K - 1] * (N + 1), (K,) * (N + 1)))

    def solve(self, lam: float) -> tuple[np.ndarray, float, float]:
        """Backward induction.  Returns the policy (T, S) and the entrant's and incumbents'
        expected mean profit per period from the reset state under it."""
        r = lam * self.r_own - (1 - lam) * self.r_inc
        S, K, T = len(self.h), self.K, self.T
        V = np.zeros(S)
        own = np.zeros(S)
        inc = np.zeros(S)
        policy = np.zeros((T, S), dtype=np.int8)
        idx = self.base[:, None] + np.arange(K)[None, :]       # next-state index for every (s, a)
        for t in range(T - 1, -1, -1):
            q = r + V[idx]
            a_star = q.argmax(axis=1)
            policy[t] = a_star + 1
            rows = np.arange(S)
            V = q[rows, a_star]
            own = self.r_own[rows, a_star] + own[idx[rows, a_star]]
            inc = self.r_inc[rows, a_star] + inc[idx[rows, a_star]]
        return policy, float(own[self.s0] / T), float(inc[self.s0] / T)

    def state_index(self, last_half_spreads) -> int:
        return int(np.ravel_multi_index([int(h) - 1 for h in last_half_spreads], (self.K,) * (self.N + 1)))


class DPEntrant(Agent):
    """Plays the backward-induction policy; the entrant must be the LAST agent."""

    def __init__(self, mdp: CartelMDP, policy: np.ndarray):
        self.mdp, self.policy = mdp, policy
        self.n_under = self.n_match = self.n_above = 0

    def act(self, obs, i):
        hs = obs["last_half_spreads"]
        t = int(obs["t"])
        a = int(self.policy[min(t, self.mdp.T - 1), self.mdp.state_index(hs)])
        inc_next = self.mdp.g[self.mdp.state_index(hs)].min()        # what the incumbents are about to quote
        if a < inc_next:
            self.n_under += 1
        elif a == inc_next:
            self.n_match += 1
        else:
            self.n_above += 1
        return a


def evaluate(Q, mk, K, N, seed, n_episodes, entrant_factory=None):
    """Frozen cartel alone (entrant_factory None) or with an entrant; returns (pnl per agent per
    period, fill shares, entrant object)."""
    n = N + (entrant_factory is not None)
    env = GlostenMilgromEnv(MarketConfig(**{**mk, "n_mm": n, "seed": 10_000 + seed}))
    inc = [FrozenQ(Q[i], K, n) for i in range(N)]
    ent = entrant_factory() if entrant_factory else None
    agents = inc + ([ent] if ent else [])
    pnl = np.zeros(n); fills = np.zeros(n); periods = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            obs, r, done, info = env.step(actions)
            pnl += r; periods += 1
            if info["filled"] >= 0:
                fills[info["filled"]] += 1
    return pnl / periods, fills / max(fills.sum(), 1), ent


def relearn(Q, mk, K, N, seed, ent, n_episodes, lr, gamma, eps0, decay, rnd):
    """The cartel resumes Q-learning against the fixed entrant; returns the new Q-tables and the
    incumbents' mean profit per period over the last fifth of the re-learning run."""
    n = N + 1
    env = GlostenMilgromEnv(MarketConfig(**{**mk, "n_mm": n, "seed": 20_000 + 1000 * rnd + seed}))
    inc = [RelearningQ(Q[i], K, n, lr, gamma, eps0, decay) for i in range(N)]
    for i, ag in enumerate(inc):
        ag.rng = np.random.default_rng(30_000 + 1000 * rnd + 10 * seed + i)
    agents = inc + [ent]
    tail_from = int(0.8 * n_episodes)
    acc, periods = 0.0, 0
    for ep in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
            obs_next, r, done, info = env.step(actions)
            for i, ag in enumerate(inc):
                ag.update(obs, int(actions[i]), float(r[i]), obs_next, done)
            if ep >= tail_from:
                acc += r[:N].sum(); periods += 1
            obs = obs_next
    return [ag.Q.copy() for ag in inc], acc / max(periods, 1)


def one_population(args):
    d, lambdas, n_episodes, relearn_rounds, relearn_episodes, eps0, decay = args
    d = Path(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N, alpha, seed = int(mk["n_mm"]), float(mk["alpha"]), int(cfg.get("seed", 0))
    p = cfg["agents"][0]["params"]
    m_inc = MarketConfig(**mk)
    K, T = m_inc.max_half_spread, m_inc.horizon
    hC, piC, hM, piM = BenchmarkTable(GlostenMilgromEnv(m_inc)).lookup(0.5)
    PiC, PiM = piC * N, piM * N
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    if Q[0].shape[2] != 1:
        raise ValueError(f"{d}: the DP needs incumbents without an inventory state")
    env0 = GlostenMilgromEnv(MarketConfig(**{**mk, "seed": 10_000 + seed}))
    inc0 = [FrozenQ(Q[i], K, N) for i in range(N)]
    pnl0, _, _, _ = play(env0, inc0, n_episodes, entrant_idx=-1)
    Pi_nom = float(pnl0.sum())
    base = {"run": d.name, "n_mm": N, "alpha": alpha, "seed": seed, "hC": hC, "hM": hM, "PiC": PiC, "PiM": PiM,
            "Pi_nominal": Pi_nom, "C": (Pi_nom - PiC) / (PiM - PiC)}
    mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T)
    base["q_ties"] = mdp.ties
    h_alone, cells = self_play_cells(Q, mk, K, N, seed)
    cells_by_inc = [{(c[1], c[2]) for c in cells if c[0] == i} for i in range(N)]
    m_entry = MarketConfig(**{**mk, "n_mm": N + 1, "seed": 10_000 + seed})
    rows = []
    # the hand-crafted rules on the same seed, for the gap to the exact best response
    hC_fn = hC_function(m_inc)
    for name, ent in (("undercut", UndercutEntrant()), ("markout", MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2))):
        env = GlostenMilgromEnv(m_entry)
        inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
        pnl, share, _, _ = play(env, inc + [ent], n_episodes, entrant_idx=N, entrant=ent)
        J = float(pnl[:N].sum())
        rows.append({**base, "attacker": name, "lambda": np.nan, "dp_entrant_norm": np.nan, "dp_L_BR_norm": np.nan,
                     "J_attack": J, "L_BR_norm": (Pi_nom - J) / (PiM - PiC), "entrant_pnl_norm": float(pnl[N]) / (PiM - PiC),
                     "entrant_fill_share": float(share[N]), "under_share": np.nan, "match_share": np.nan, "above_share": np.nan,
                     "h_inc_attack": np.nan, "off_path_share": np.nan})
    for lam in lambdas:
        policy, own_dp, inc_dp = mdp.solve(lam)
        env = GlostenMilgromEnv(m_entry)
        inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
        ent = DPEntrant(mdp, policy)
        quotes, off_path = [], 0
        # play, recording the incumbents' quotes under attack and whether the attack holds them
        # at Q-cells they never visit on their own equilibrium path
        n = N + 1
        pnl = np.zeros(n); fills = np.zeros(n); periods = 0
        for _ in range(n_episodes):
            obs = env.reset()
            for ag in inc + [ent]:
                ag.reset()
            done = False
            while not done:
                if any(inc[i]._state(obs, i)[:2] not in cells_by_inc[i] for i in range(N)):
                    off_path += 1
                actions = np.array([ag.act(obs, i) for i, ag in enumerate(inc + [ent])])
                quotes.append(actions[:N].min())
                obs, r, done, info = env.step(actions)
                pnl += r; periods += 1
                if info["filled"] >= 0:
                    fills[info["filled"]] += 1
        pnl /= periods; fills /= max(fills.sum(), 1)
        J = float(pnl[:N].sum())
        tot = ent.n_under + ent.n_match + ent.n_above
        rows.append({**base, "attacker": f"dp:{lam:g}", "lambda": lam,
                     "dp_entrant_norm": own_dp / (PiM - PiC), "dp_L_BR_norm": (Pi_nom - inc_dp) / (PiM - PiC),
                     "J_attack": J, "L_BR_norm": (Pi_nom - J) / (PiM - PiC), "entrant_pnl_norm": float(pnl[N]) / (PiM - PiC),
                     "entrant_fill_share": float(fills[N]), "under_share": ent.n_under / tot, "match_share": ent.n_match / tot,
                     "above_share": ent.n_above / tot, "h_inc_attack": float(np.mean(quotes)), "off_path_share": off_path / periods})
    for r in rows:
        r["h_inc_alone"] = h_alone
        r["round"] = 0
    # ---- double oracle: the cartel answers the best response by re-learning, the entrant re-solves
    lr = float(p.get("learning_rate", 0.15)); gamma = float(p.get("discount", 0.95))
    Qr = Q
    policy, _, _ = mdp.solve(1.0)
    ent_stale = DPEntrant(mdp, policy)
    for rnd in range(1, relearn_rounds + 1):
        Qr, inc_tail = relearn(Qr, mk, K, N, seed, ent_stale, relearn_episodes, lr, gamma, eps0, decay, rnd)
        pnl_alone, _, _ = evaluate(Qr, mk, K, N, seed, n_episodes)
        Pi_r = float(pnl_alone.sum())
        pnl_stale, share_stale, _ = evaluate(Qr, mk, K, N, seed, n_episodes, lambda: ent_stale)
        mdp_r = CartelMDP(Qr, MarketConfig(**{**mk, "n_mm": N + 1}), T)
        policy_r, own_dp, inc_dp = mdp_r.solve(1.0)
        ent_fresh = DPEntrant(mdp_r, policy_r)
        pnl_fresh, share_fresh, _ = evaluate(Qr, mk, K, N, seed, n_episodes, lambda: ent_fresh)
        for name, pnl_, share_, ent_ in (("dp:1@stale", pnl_stale, share_stale, ent_stale), ("dp:1@fresh", pnl_fresh, share_fresh, ent_fresh)):
            J = float(pnl_[:N].sum())
            tot = max(ent_.n_under + ent_.n_match + ent_.n_above, 1)
            rows.append({**base, "attacker": name, "lambda": 1.0, "round": rnd,
                         "Pi_nominal": Pi_r, "C": (Pi_r - PiC) / (PiM - PiC),
                         "inc_relearn_tail_norm": (inc_tail - PiC) / (PiM - PiC),
                         "dp_entrant_norm": own_dp / (PiM - PiC) if name == "dp:1@fresh" else np.nan,
                         "dp_L_BR_norm": (Pi_r - inc_dp) / (PiM - PiC) if name == "dp:1@fresh" else np.nan,
                         "J_attack": J, "L_BR_norm": (Pi_r - J) / (PiM - PiC), "entrant_pnl_norm": float(pnl_[N]) / (PiM - PiC),
                         "entrant_fill_share": float(share_[N]), "under_share": ent_.n_under / tot, "match_share": ent_.n_match / tot,
                         "above_share": ent_.n_above / tot, "h_inc_attack": np.nan, "off_path_share": np.nan,
                         "h_inc_alone": np.nan, "q_ties": mdp_r.ties})
        ent_stale = ent_fresh
    return rows


def self_play_cells(Q, mk, K, N, seed, n_episodes=100):
    """The incumbents' mean best quote alone and the set of (i, own, riv) Q-cells they visit on
    their own equilibrium path."""
    env0 = GlostenMilgromEnv(MarketConfig(**{**mk, "seed": 10_000 + seed}))
    inc0 = [FrozenQ(Q[i], K, N) for i in range(N)]
    q0, cells = [], set()
    for _ in range(n_episodes):
        obs = env0.reset()
        for ag in inc0:
            ag.reset()
        done = False
        while not done:
            for i, ag in enumerate(inc0):
                own, riv, _ = ag._state(obs, i)
                cells.add((i, own, riv))
            actions = np.array([ag.act(obs, i) for i, ag in enumerate(inc0)])
            q0.append(actions.min())
            obs, _, done, _ = env0.step(actions)
    return float(np.mean(q0)), cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_seed*"])
    ap.add_argument("--lambdas", default="1,0.75,0.5,0.25,0")
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--relearn-rounds", type=int, default=0, help="double-oracle rounds (cartel re-learns, entrant re-solves)")
    ap.add_argument("--relearn-episodes", type=int, default=3000)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--decay", type=float, default=1e-5)
    ap.add_argument("--out", default="results/best_response.csv")
    args = ap.parse_args()
    dirs = sorted({d for g in args.dirs for d in glob.glob(g) if (Path(d) / "Q_agent0.npy").exists()})
    if not dirs:
        raise SystemExit("no trained populations matched --dirs")
    lambdas = [float(x) for x in args.lambdas.split(",")]
    print(f"{len(dirs)} populations, lambdas {lambdas}, {args.n_episodes} episodes", flush=True)
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        jobs = [(d, lambdas, args.n_episodes, args.relearn_rounds, args.relearn_episodes, args.eps0, args.decay) for d in dirs]
        for k, out in enumerate(ex.map(one_population, jobs)):
            rows.extend(out)
            r1 = next(r for r in out if r["attacker"] == "dp:1")
            print(f"[{k + 1}/{len(dirs)}] {out[0]['run']} C={out[0]['C']:.2f} ties={out[0]['q_ties']}  "
                  + "  ".join(f"{r['attacker']}:L={r['L_BR_norm']:.2f},e={r['entrant_pnl_norm']:.2f}" for r in out)
                  + f"  | dp:1 predicted e={r1['dp_entrant_norm']:.2f} L={r1['dp_L_BR_norm']:.2f}; under/match/above "
                  f"{r1['under_share']:.2f}/{r1['match_share']:.2f}/{r1['above_share']:.2f}; h_inc alone {r1['h_inc_alone']:.1f} -> {r1['h_inc_attack']:.1f}; "
                  f"off-path {r1['off_path_share']:.2f}"
                  + "".join(f"\n    round {r['round']}: C={r['C']:.2f} relearn-tail={r['inc_relearn_tail_norm']:.2f} {r['attacker']}: L={r['L_BR_norm']:.2f} e={r['entrant_pnl_norm']:.2f}"
                            for r in out if r["round"] > 0),
                  flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
