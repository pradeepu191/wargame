"""The one-tick queue game: value of identity when stepping out costs your place in line (RQ3).

For each cell (regime persistence rho_z, mark lag L, ticks above break-even) and seed:
  1. incumbents: N makers that always rest (the tick-constrained competitive book) or N tabular
     Q-learners trained on the queue game (identity-blind; they see depth, own rank, and the
     anonymous tape signal with the mark lag);
  2. one entrant per policy, frozen incumbents, `n_episodes` episodes:
        always   rest every period (joins the round-robin)
        anon     WalletEntrant regime filter without ids
        id       with ids
        oracle   true types
  3. record profit per period for entrant and incumbents, informed share of each side's fills,
     the entrant's rest share, and its mean rank when filled.

Usage:
  python experiments/queue_game.py --alphas 0.3 --seeds 0-4 --rhos-z 0.9,0.99 --mark-lags 1,5,20 \
      --above 0,1 --incumbents always --n-episodes 300 --jobs 2 --out results/queue.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import MarketConfig, QConfig
from sim.queue_agents import AlwaysRest, QueueQAgent, QueueWalletEntrant
from sim.queue_env import QueueEnv, selection_benchmarks

BASE = dict(sigma_L=8.0, tick=0.5, max_half_spread=24, redraw_v_each_period=True, edge_dist="exponential",
            edge_mean=5.0, n_wallets=50, wallet_concentration=1.0)
POLICIES = ("always", "anon", "id", "oracle")


def play(env, agents, n_episodes, E):
    n = env.cfg.n_mm
    pnl = np.zeros(n); fills = np.zeros(n); inf = np.zeros(n); rest = 0; rank_sum = 0.0; periods = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            acts = [ag.act(obs, i) for i, ag in enumerate(agents)]
            rest += acts[E]
            obs, r, done, info = env.step(acts)
            pnl += r; periods += 1
            f = info["filled"]
            if f >= 0:
                fills[f] += 1; inf[f] += info["informed"]
    return pnl / periods, fills, inf, rest / periods, periods


def train_q(cfg: MarketConfig, h: int, n_episodes: int, seed: int):
    env = QueueEnv(cfg, half_spread=h)
    qc = QConfig(learning_rate=0.05, discount=0.95, exploration_decay=2e-6, inventory_buckets=1, init_q=1.0)
    agents = [QueueQAgent(cfg.n_mm, qc, seed=seed * 10 + k) for k in range(cfg.n_mm)]
    for _ in range(n_episodes):
        obs = env.reset(); done = False
        while not done:
            acts = [a.act(obs, i) for i, a in enumerate(agents)]
            obs_n, r, done, _ = env.step(acts)
            for i, a in enumerate(agents):
                a.update(obs, acts[i], float(r[i]), obs_n, done)
            obs = obs_n
    for a in agents:
        a.set_eval(True)
    return agents


def one_cell(args):
    alpha, rho_z, L, above, n_inc, seed, incumbents, n_episodes, n_train = args
    mk = dict(BASE, alpha=alpha, regime_persistence=rho_z, mark_lag=L, wallet_seed=seed)
    cfg_inc = MarketConfig(**mk, n_mm=n_inc, seed=seed)
    env_inc = QueueEnv(cfg_inc, ticks_above_breakeven=above)
    h = env_inc.h
    bm = selection_benchmarks(env_inc)
    if incumbents == "q":
        inc = train_q(cfg_inc, h, n_train, seed)
    else:
        inc = [AlwaysRest() for _ in range(n_inc)]
    rows = []
    cfg_e = MarketConfig(**mk, n_mm=n_inc + 1, seed=10_000 + seed)
    E = n_inc
    for name in POLICIES:
        env = QueueEnv(cfg_e, half_spread=h)
        if name == "always":
            ent = AlwaysRest()
        else:
            ent = QueueWalletEntrant(cfg_e, h, name, wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff)
        pnl, fills, inf, rest_share, periods = play(env, inc + [ent], n_episodes, E)
        rows.append({"alpha": alpha, "rho_z": rho_z, "mark_lag": L, "above": above, "h": h, "n_inc": n_inc,
                     "seed": seed, "incumbents": incumbents, "policy": name,
                     "fill_value_avg": bm["fill_value_avg"], "fill_value_calm": bm.get("fill_value_calm"),
                     "fill_value_toxic": bm.get("fill_value_toxic"),
                     "entrant_pnl": float(pnl[E]), "incumbent_pnl": float(pnl[:E].mean()),
                     "entrant_fill_share": float(fills[E] / max(fills.sum(), 1)),
                     "entrant_informed_share": float(inf[E] / max(fills[E], 1)),
                     "incumbent_informed_share": float(inf[:E].sum() / max(fills[:E].sum(), 1)),
                     "entrant_rest_share": rest_share, "fill_rate": float(fills.sum() / periods)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--rhos-z", default="0.9,0.99")
    ap.add_argument("--mark-lags", default="1,5,20")
    ap.add_argument("--above", default="0,1", help="ticks above the population break-even for the fixed spread")
    ap.add_argument("--n-inc", type=int, default=2)
    ap.add_argument("--incumbents", default="always", choices=["always", "q"])
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/queue.csv")
    args = ap.parse_args()
    a_, b_ = map(int, args.seeds.split("-"))
    jobs = [(float(a), float(rz), int(L), int(ab), args.n_inc, s, args.incumbents, args.n_episodes, args.n_train)
            for a in args.alphas.split(",") for rz in args.rhos_z.split(",") for L in args.mark_lags.split(",")
            for ab in args.above.split(",") for s in range(a_, b_ + 1)]
    print(f"{len(jobs)} cells x {len(POLICIES)} policies, incumbents={args.incumbents}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, out in enumerate(ex.map(one_cell, jobs)):
            rows.extend(out)
            r0 = out[0]
            print(f"[{k + 1}/{len(jobs)}] a={r0['alpha']} rho_z={r0['rho_z']} L={r0['mark_lag']} above={r0['above']} h={r0['h']} seed={r0['seed']}  "
                  + "  ".join(f"{r['policy']}:{r['entrant_pnl']:.3f}" for r in out), flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    g = df.groupby(["alpha", "above", "rho_z", "mark_lag", "policy"])[["entrant_pnl", "incumbent_pnl", "entrant_informed_share", "incumbent_informed_share", "entrant_rest_share"]].mean()
    print(g.round(3).to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
