"""A small instance of the wargame, end to end, in a few seconds.

    python experiments/demo.py            # uses the shipped cartel in results/demo_cartel/
    python experiments/demo.py --train    # trains a fresh two-maker cartel first (~2 min)

What it shows, in order:
  1. the market's closed-form benchmarks: the Nash (competitive) quote h^C, the joint-monopoly
     quote h^M, and the payoff curve Pi(h) between them;
  2. the frozen cartel playing alone -- its quotes and its rent index C;
  3. three adversaries entering the same market under common random numbers: the one-tick
     undercutter, the markout entrant (infers toxicity from its own fills), and the exact best
     response (backward induction over the cartel's reaction function); for each, the cartel's
     loss L_BR and the entrant's profit, in rent units, and the fraction of rent destroyed kappa_A;
  4. a ten-period trace of the exact best response against the cartel: who quoted what, who was
     hit, and what the fill paid -- the steering is visible by period 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import GlostenMilgromEnv, MarketConfig
from sim.benchmarks import profit_curve, type_benchmarks
from sim.entrants import MarkoutEntrant, UndercutEntrant, hC_function
from experiments.best_response import CartelMDP, DPEntrant
from experiments.entry import FrozenQ, play

DEMO = Path(__file__).resolve().parents[1] / "results" / "demo_cartel"


def load_cartel(d: Path):
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N = int(mk["n_mm"])
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    return mk, Q, int(cfg.get("seed", 0))


def trace(env, agents, n_periods, names):
    obs = env.reset()
    for ag in agents:
        ag.reset()
    print(f"\n  {'t':>2}  " + "  ".join(f"{n:>8}" for n in names) + "   taker      filled   payoff")
    for t in range(n_periods):
        actions = np.array([ag.act(obs, i) for i, ag in enumerate(agents)])
        obs, r, done, info = env.step(actions)
        who = names[info["filled"]] if info["filled"] >= 0 else "-"
        taker = ("informed " if info["informed"] else "noise    ") + info["event"] if info["event"] != "none" else "no trade"
        pay = f"{r[info['filled']]:+.2f}" if info["filled"] >= 0 else ""
        print(f"  {t:>2}  " + "  ".join(f"{a:>8d}" for a in actions) + f"   {taker:<12} {who:>8}   {pay}")
        if done:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartel", default=str(DEMO))
    ap.add_argument("--train", action="store_true", help="train a fresh cartel with experiments/configs/demo_cartel.yaml first")
    ap.add_argument("--n-episodes", type=int, default=200)
    args = ap.parse_args()
    d = Path(args.cartel)
    if args.train:
        from sim.runner import run
        d = run(str(Path(__file__).resolve().parents[1] / "experiments" / "configs" / "demo_cartel.yaml"))
    mk, Q, seed = load_cartel(d)
    N, K, T = int(mk["n_mm"]), int(mk["max_half_spread"]), int(mk["horizon"])
    cfg = MarketConfig(**mk)
    Pi = profit_curve(cfg, cfg.alpha)
    hC, PiC, hM, PiM = type_benchmarks(cfg, cfg.alpha)
    rent = PiM - PiC
    print(f"Market: alpha = {cfg.alpha} (informed share), tick = {cfg.tick}, K = {K} quotes, {T} periods per episode, N = {N} incumbents")
    print(f"Benchmarks: Nash quote h^C = {hC} (Pi = {PiC:.2f}), monopoly quote h^M = {hM} (Pi = {PiM:.2f}); rent unit Pi^M - Pi^C = {rent:.2f}")
    print("Pi(h) for h = 1..K: " + " ".join(f"{p:+.2f}" for p in Pi))

    # 2. the cartel alone
    env0 = GlostenMilgromEnv(MarketConfig(**{**mk, "seed": 10_000 + seed}))
    inc0 = [FrozenQ(Q[i], K, N) for i in range(N)]
    pnl0, _, _, _ = play(env0, inc0, args.n_episodes, entrant_idx=-1)
    Pi_nom = float(pnl0.sum())
    C = (Pi_nom - PiC) / rent
    print(f"\nCartel alone ({args.n_episodes} episodes): market profit {Pi_nom:.2f} per period, rent index C = {C:.2f}")
    print("  (C = 0 is the Nash point, C = 1 joint monopoly; the learners never communicated)")

    # 3. adversaries under common random numbers
    m_entry = MarketConfig(**{**mk, "n_mm": N + 1, "seed": 10_000 + seed})
    mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T)
    policy, own_dp, inc_dp = mdp.solve(1.0)
    hC_fn = hC_function(cfg)
    adversaries = {
        "undercut (one tick inside the best incumbent quote)": lambda: UndercutEntrant(),
        "markout (undercut only when inferred rent is there)": lambda: MarkoutEntrant(hC_fn, alpha_prior=0.3, k_min=20, margin=2),
        "exact best response (backward induction)": lambda: DPEntrant(mdp, policy),
    }
    print(f"\nEntry ({args.n_episodes} episodes each, same random numbers):")
    print(f"  {'adversary':<52} {'cartel loses':>12} {'entrant earns':>14} {'kappa_A':>8}")
    for name, make in adversaries.items():
        env = GlostenMilgromEnv(m_entry)
        inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
        ent = make()
        pnl, share, _, _ = play(env, inc + [ent], args.n_episodes, entrant_idx=N, entrant=ent)
        L = (Pi_nom - float(pnl[:N].sum())) / rent
        e = float(pnl[N]) / rent
        print(f"  {name:<52} {L:>12.2f} {e:>14.2f} {L / max(C, 1e-9):>8.2f}")
    print(f"  (rent units; the DP predicted the best response would earn {own_dp / rent:.2f} and leave the cartel {inc_dp / rent:+.2f})")

    # 4. trace
    print("\nTrace: the exact best response against the cartel (quotes in ticks; the taker hits the lowest):")
    env = GlostenMilgromEnv(m_entry)
    inc = [FrozenQ(Q[i], K, N + 1) for i in range(N)]
    trace(env, inc + [DPEntrant(mdp, policy)], 10, [f"inc{i}" for i in range(N)] + ["entrant"])
    print("\n  The incumbents answer the entrant's quote with their learned reaction; the entrant sits one tick inside\n"
          "  what they are about to quote, and holds them at states they never visited while learning.")


if __name__ == "__main__":
    main()
