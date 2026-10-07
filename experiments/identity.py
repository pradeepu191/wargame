"""RQ1 proper: the executable value of wallet identity to an entrant, pi^anon <= pi^id <= pi^oracle.

Frozen trained incumbents (identity-blind Q-learners) face a heterogeneous, persistent taker
population (MarketConfig.n_wallets > 0) and one entrant with a nested information set:

    competitive   h^C(abar): ignores incumbents and identity                (benchmark)
    markout       own-fill rule from the frozen-entry experiment             (continuity)
    anon          tape markouts, no wallet ids                               F^anon
    id            tape markouts keyed by wallet id                           F^id
    oracle        true wallet types                                          F^oracle

over a grid of type dispersion (wallet_concentration kappa; 1e6 = homogeneous) and arrival
persistence (wallet_persistence rho).  The incumbents were trained on the homogeneous market at
the same mean toxicity; their policies do not see identity, so this measures what identity is
worth to a newcomer against an identity-blind cartel.

Per (population, kappa, rho, policy): entrant and incumbent profit per period (also in rent
units), fill shares, the realized informed share of each side's fills (cream-skimming shows as
the entrant's informed share falling below the incumbents'), the entrant's decision mix, and
the market-wide Delta.

Usage:
  python experiments/identity.py --run cfl_exp2 --alphas 0.3,0.5 --seeds 0-4 \
      --kappas 1e6,5,1 --rhos 0,0.5,0.9 --n-wallets 50 --n-episodes 500 --jobs 2
Writes results/<run>_identity.csv.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig
from sim.benchmarks import collusion_index, identity_monopoly_profit
from sim.entrants import CompetitiveEntrant, MarkoutEntrant, WalletEntrant, hC_function
from experiments.entry import FrozenQ

POLICIES = ("competitive", "markout", "anon", "id", "oracle")


def play(env, agents, n_episodes, E):
    n = env.cfg.n_mm
    pnl = np.zeros(n); fills = np.zeros(n); informed_fills = np.zeros(n)
    dec = {"undercut": 0, "withdraw": 0, "competitive": 0}
    periods = 0
    ent = agents[E]
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done:
            actions = [ag.act(obs, i) for i, ag in enumerate(agents)]
            d = getattr(ent, "last_decision", None)
            if d in dec:
                dec[d] += 1
            obs, r, done, info = env.step(actions)
            pnl += r; periods += 1
            f = info["filled"]
            if f >= 0:
                fills[f] += 1
                informed_fills[f] += info["informed"]
                if f == E and hasattr(ent, "observe_fill"):
                    ent.observe_fill(float(r[E]))
    return pnl / periods, fills, informed_fills, dec, periods


def one_cell(args):
    d, kappa, rho, rho_z, mark_lag, n_wallets, n_episodes, margin = args
    d = Path(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = cfg["market"]
    N, alpha, seed = int(mk["n_mm"]), float(mk["alpha"]), int(cfg.get("seed", 0))
    het = dict(n_wallets=n_wallets, wallet_concentration=kappa, wallet_persistence=rho, wallet_seed=seed,
               regime_persistence=rho_z, mark_lag=mark_lag)
    m_inc = MarketConfig(**{**mk, **het, "seed": 10_000 + seed})
    m_entry = MarketConfig(**{**mk, **het, "n_mm": N + 1, "seed": 10_000 + seed})
    env_bm = GlostenMilgromEnv(m_inc)
    hC, piC, hM, piM = BenchmarkTable(env_bm).lookup(0.5)
    PiC, PiM = piC * N, piM * N
    PiM_id = identity_monopoly_profit(env_bm)
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    K = m_inc.max_half_spread
    hC_fn = hC_function(MarketConfig(**mk))

    # nominal: incumbents alone on the heterogeneous market
    env0 = GlostenMilgromEnv(m_inc)
    pnl0, fills0, inf0, _, _ = play(env0, [FrozenQ(Q[i], K, N) for i in range(N)], n_episodes, E=-1)
    rows = []
    for name in POLICIES:
        env = GlostenMilgromEnv(m_entry)
        if name == "competitive":
            ent = CompetitiveEntrant(hC)
        elif name == "markout":
            ent = MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2)
        else:
            ent = WalletEntrant(m_entry, name, wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff, margin=margin)
        pnl, fills, inf, dec, periods = play(env, [FrozenQ(Q[i], K, N + 1) for i in range(N)] + [ent], n_episodes, E=N)
        tot = fills.sum()
        rows.append({
            "run": d.name, "n_mm": N, "alpha": alpha, "seed": seed, "kappa": kappa, "rho": rho, "rho_z": rho_z, "mark_lag": mark_lag,
            "n_wallets": n_wallets, "alpha_eff": env.alpha_eff, "policy": name,
            "hC": hC, "hM": hM, "PiC": PiC, "PiM": PiM, "PiM_id": PiM_id,
            "Pi_nominal": float(pnl0.sum()), "informed_share_nominal": float(inf0.sum() / max(fills0.sum(), 1)),
            "entrant_pnl": float(pnl[N]), "entrant_pnl_norm": float(pnl[N]) / (PiM - PiC),
            "incumbent_pnl": float(pnl[:N].sum()), "incumbent_pnl_norm": float(pnl[:N].sum()) / (PiM - PiC),
            "delta_market": collusion_index(float(pnl.sum()) / (N + 1), PiC / (N + 1), PiM / (N + 1)),
            "entrant_fill_share": float(fills[N] / max(tot, 1)),
            "entrant_informed_share": float(inf[N] / max(fills[N], 1)),
            "incumbent_informed_share": float(inf[:N].sum() / max(fills[:N].sum(), 1)),
            "fill_rate": float(tot / periods),
            "dec_undercut": dec["undercut"] / periods, "dec_withdraw": dec["withdraw"] / periods,
            "dec_competitive": dec["competitive"] / periods,
            "abar_hat": getattr(ent, "abar_hat", np.nan) if name in ("anon", "id") else np.nan,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    ap.add_argument("--n-mm", type=int, default=2)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--kappas", default="1e6,5,1")
    ap.add_argument("--rhos", default="0,0.5,0.9", help="wallet persistence values")
    ap.add_argument("--rhos-z", default="0", help="regime persistence values (needs kappa < inf to matter)")
    ap.add_argument("--mark-lags", default="1", help="periods until a print's mark is public")
    ap.add_argument("--margin", type=int, default=1,
                    help="undercut when best - h^C(alpha_next) >= margin; 1 = whenever the undercut quote is at or "
                         "above break-even (the first identity grid used 2, which discards positive-EV fills when "
                         "the forecast is pessimistic and can make a better-informed entrant earn less)")
    ap.add_argument("--n-wallets", type=int, default=50)
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = Path(args.results)
    a_, b_ = map(int, args.seeds.split("-"))
    dirs = [results / f"{args.run}_alpha{float(a)}_n_mm{args.n_mm}_seed{s}"
            for a in args.alphas.split(",") for s in range(a_, b_ + 1)]
    dirs = [d for d in dirs if (d / "Q_agent0.npy").exists()]
    if not dirs:
        raise SystemExit("no trained incumbents found")
    jobs = [(str(d), float(k), float(r), float(rz), int(L), args.n_wallets, args.n_episodes, args.margin)
            for d in dirs for k in args.kappas.split(",") for r in args.rhos.split(",") for rz in args.rhos_z.split(",")
            for L in args.mark_lags.split(",")]
    print(f"{len(dirs)} populations x {len(jobs) // len(dirs)} (kappa, rho) cells x {len(POLICIES)} policies", flush=True)
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, out in enumerate(ex.map(one_cell, jobs)):
            rows.extend(out)
            r0 = out[0]
            print(f"[{k + 1}/{len(jobs)}] {r0['run']} kappa={r0['kappa']:g} rho={r0['rho']} rho_z={r0['rho_z']} lag={r0['mark_lag']} "
                  + "  ".join(f"{r['policy']}:{r['entrant_pnl_norm']:.2f}" for r in out), flush=True)
    df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else results / f"{args.run}_identity.csv"
    df.to_csv(out_path, index=False)
    pd.set_option("display.width", 240)
    g = df.groupby(["alpha", "kappa", "rho", "rho_z", "mark_lag", "policy"])[["entrant_pnl_norm", "entrant_informed_share", "incumbent_informed_share", "dec_withdraw", "entrant_fill_share"]].mean()
    print(g.round(3).to_string())
    print("wrote", out_path)


if __name__ == "__main__":
    main()
