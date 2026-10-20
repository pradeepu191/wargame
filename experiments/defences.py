"""RQ3, the makers' own levers: quoter-side defences and the profit--resilience frontier.

Frozen Q-learner cartels (results/<run>_seed<k>/) are wrapped in a defence and evaluated on the
heterogeneous market (wallet types, toxicity regime, mark lag -- the taker model the data support),
alone and against each adversary.  Defences:

    none            the trained policy as is
    random(eps,j)   with probability eps the quote is moved by U{-j..j} ticks: a noisy reaction
                    function (harder to learn against; may also unsettle the cartel itself)
    hold(m)         requote only every m periods (Brown-MacKay: speed is what makes coercion work;
                    a slow quoter is harder to steer and slower to defend)
    floor(anon|id)  identity-aware control: never quote inside the break-even for the forecast
                    type, h >= h^C(alpha_next), with alpha_next from the regime filter on the tape
                    (anon) or on wallet identities (id)

Adversaries: competitive (quotes h^C), undercut (one tick inside), markout (own-fill inference),
anon / id / oracle (the wallet entrant's regime filter with each information set).

Per (population, defence, adversary): the defenders' profit alone (Pi_nominal), under attack
(E and the 5% CVaR over episodes of the defenders' per-episode profit), L_BR and kappa_A, the
adversary's profit, and the taker-side quantities a frontier must show so that "robust" cannot
mean "withdrew": mean best half-spread quoted and the fill rate.

Usage:
  python experiments/defences.py --run cfl_exp2 --alphas 0.3,0.5 --seeds 0-9 --n-episodes 300 --jobs 2
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
from sim.entrants import CompetitiveEntrant, MarkoutEntrant, UndercutEntrant, WalletEntrant, hC_function
from experiments.entry import FrozenQ

ADVERSARIES = ("competitive", "undercut", "markout", "anon", "id", "oracle")
DEFENCES = ("none", "random:0.1", "random:0.3", "hold:2", "hold:5", "floor:anon", "floor:id")
HET = dict(n_wallets=50, wallet_concentration=1.0, wallet_persistence=0.0, regime_persistence=0.9, mark_lag=5)


class Defended(Agent):
    """A frozen incumbent plus one defence."""

    def __init__(self, base: FrozenQ, defence: str, K: int, hC_fn, cfg: MarketConfig, env, seed: int):
        self.base, self.K, self.hC_fn = base, K, hC_fn
        self.kind, _, p = defence.partition(":")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.filter = None
        if self.kind == "floor":
            self.filter = WalletEntrant(cfg, p, wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff)
        self.t = 0
        self.prev = None
        self.changes = 0

    def reset(self):
        self.base.reset()
        self.t = 0
        self.prev = None
        if self.filter is not None:
            self.filter.reset()

    def act(self, obs, i):
        a = int(self.base.act(obs, i))
        if self.kind == "random":
            if self.rng.random() < float(self.p):
                a = int(np.clip(a + self.rng.integers(-2, 3), 1, self.K))
        elif self.kind == "hold":
            if self.prev is not None and self.t % int(self.p) != 0:
                a = self.prev
        elif self.kind == "floor":
            self.filter._ingest(obs)
            a = max(a, self.hC_fn(self.filter.alpha_next()))
        if self.prev is not None and a != self.prev:
            self.changes += 1
        self.prev = a
        self.t += 1
        return a


def make_adversary(name, hC, hC_fn, alpha, cfg, env):
    if name == "competitive":
        return CompetitiveEntrant(hC)
    if name == "undercut":
        return UndercutEntrant()
    if name == "markout":
        return MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2)
    return WalletEntrant(cfg, name, wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff, margin=1)


def play(env, agents, n_episodes, N, E=None):
    """Per-agent mean profit per period; per-episode defenders' profit (for the CVaR); mean best
    half-spread quoted; fill rate; informed share of the defenders' and the adversary's fills."""
    n = env.cfg.n_mm
    pnl = np.zeros(n); fills = np.zeros(n); inf = np.zeros(n); periods = 0
    ep_inc = []; best_h = 0.0
    ent = agents[E] if E is not None else None
    for _ in range(n_episodes):
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False; ep = 0.0
        while not done:
            actions = [ag.act(obs, i) for i, ag in enumerate(agents)]
            best_h += min(actions)
            obs, r, done, info = env.step(actions)
            pnl += r; periods += 1; ep += float(r[:N].sum())
            f = info["filled"]
            if f >= 0:
                fills[f] += 1; inf[f] += info["informed"]
                if ent is not None and f == E and hasattr(ent, "observe_fill"):
                    ent.observe_fill(float(r[E]))
        ep_inc.append(ep / env.cfg.horizon)
    ep_inc = np.array(ep_inc)
    k = max(1, int(np.ceil(0.05 * len(ep_inc))))
    cvar = float(np.sort(ep_inc)[:k].mean())
    return dict(pnl=pnl / periods, fills=fills, inf=inf, periods=periods, cvar=cvar, ep_sd=float(ep_inc.std()),
                best_h=best_h / periods, fill_rate=float(fills.sum() / periods))


def one_population(args):
    d, n_episodes, defences, adversaries = args
    d = Path(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = cfg["market"]
    N, alpha, seed = int(mk["n_mm"]), float(mk["alpha"]), int(cfg.get("seed", 0))
    m_inc = MarketConfig(**{**mk, **HET, "wallet_seed": seed, "seed": 10_000 + seed})
    m_entry = MarketConfig(**{**mk, **HET, "wallet_seed": seed, "n_mm": N + 1, "seed": 10_000 + seed})
    hC, piC, hM, piM = BenchmarkTable(GlostenMilgromEnv(m_inc)).lookup(0.5)
    PiC, PiM = piC * N, piM * N
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    K = m_inc.max_half_spread
    hC_fn = hC_function(MarketConfig(**mk))
    rows = []
    for dname in defences:
        env0 = GlostenMilgromEnv(m_inc)
        inc0 = [Defended(FrozenQ(Q[i], K, N), dname, K, hC_fn, m_inc, env0, seed * 10 + i) for i in range(N)]
        r0 = play(env0, inc0, n_episodes, N)
        Pi_nom = float(r0["pnl"].sum())
        base = {"run": d.name, "n_mm": N, "alpha": alpha, "seed": seed, "defence": dname,
                "hC": hC, "hM": hM, "PiC": PiC, "PiM": PiM,
                "Pi_nominal": Pi_nom, "Pi_nominal_norm": Pi_nom / (PiM - PiC), "C": (Pi_nom - PiC) / (PiM - PiC),
                "nominal_cvar": r0["cvar"], "nominal_best_h": r0["best_h"], "nominal_fill_rate": r0["fill_rate"],
                "nominal_informed_share": float(r0["inf"].sum() / max(r0["fills"].sum(), 1)),
                "nominal_requotes": float(np.mean([a.changes for a in inc0]) / r0["periods"])}
        for aname in adversaries:
            env = GlostenMilgromEnv(m_entry)
            inc = [Defended(FrozenQ(Q[i], K, N + 1), dname, K, hC_fn, m_entry, env, seed * 10 + i) for i in range(N)]
            adv = make_adversary(aname, hC, hC_fn, alpha, m_entry, env)
            r = play(env, inc + [adv], n_episodes, N, E=N)
            J = float(r["pnl"][:N].sum())
            rows.append({**base, "adversary": aname, "J_attack": J, "J_attack_norm": J / (PiM - PiC),
                         "L_BR": Pi_nom - J, "L_BR_norm": (Pi_nom - J) / (PiM - PiC),
                         "kappa_A": (Pi_nom - J) / (Pi_nom - PiC) if Pi_nom - PiC > 1e-9 else np.nan,
                         "attack_cvar": r["cvar"], "attack_cvar_norm": r["cvar"] / (PiM - PiC), "attack_ep_sd": r["ep_sd"],
                         "adversary_pnl": float(r["pnl"][N]), "adversary_pnl_norm": float(r["pnl"][N]) / (PiM - PiC),
                         "adversary_fill_share": float(r["fills"][N] / max(r["fills"].sum(), 1)),
                         "attack_best_h": r["best_h"], "attack_fill_rate": r["fill_rate"],
                         "defender_informed_share": float(r["inf"][:N].sum() / max(r["fills"][:N].sum(), 1)),
                         "adversary_informed_share": float(r["inf"][N] / max(r["fills"][N], 1))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    ap.add_argument("--n-mm", type=int, default=2)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--defences", default=",".join(DEFENCES))
    ap.add_argument("--adversaries", default=",".join(ADVERSARIES))
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/defences.csv")
    args = ap.parse_args()
    a_, b_ = map(int, args.seeds.split("-"))
    dirs = []
    for a in args.alphas.split(","):
        for s in range(a_, b_ + 1):
            d = Path(args.results) / f"{args.run}_alpha{float(a):g}_n_mm{args.n_mm}_seed{s}"
            if (d / "Q_agent0.npy").exists():
                dirs.append(d)
    if not dirs:
        raise SystemExit("no trained populations found")
    defences = tuple(args.defences.split(",")); adversaries = tuple(args.adversaries.split(","))
    print(f"{len(dirs)} populations x {len(defences)} defences x {len(adversaries)} adversaries, {args.n_episodes} episodes", flush=True)
    jobs = [(str(d), args.n_episodes, defences, adversaries) for d in dirs]
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, out in enumerate(ex.map(one_population, jobs)):
            rows.extend(out)
            df = pd.DataFrame(out)
            summ = df.groupby("defence").agg(nom=("Pi_nominal_norm", "first"), worst=("J_attack_norm", "min"), cvar=("attack_cvar_norm", "min"))
            print(f"[{k + 1}/{len(dirs)}] {out[0]['run']}  " + "  ".join(f"{d}: nom {r.nom:.2f} worst {r.worst:.2f} cvar {r.cvar:.2f}" for d, r in summ.iterrows()), flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    g = df.groupby(["alpha", "defence", "adversary"])[["Pi_nominal_norm", "J_attack_norm", "attack_cvar_norm", "kappa_A", "adversary_pnl_norm", "attack_best_h", "attack_fill_rate"]].mean()
    print(g.round(3).to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
