"""Record a run of the simulator period by period, for the visual demonstration.

The browser player (demo/index.html) and the video renderer (analysis/trace_video.py) contain no
model logic: they replay what this script writes.  Every scenario runs the real environments and
agents with a fixed seed, so what is shown is what the experiments measure.

    python experiments/trace.py --all                       # every scenario -> demo/traces.js + demo/traces/*.json
    python experiments/trace.py --scenario best_response     # one scenario
    python experiments/trace.py --list

Scenarios (spread mode uses the shipped cartel in results/demo_cartel/; --cartel picks another.
The identity scenarios use results/demo_cartel_a05/, a cartel trained at alpha = 0.5, because identity
is worth nothing where even the toxic state leaves an undercut profitable; in the identity and queue
scenarios the entrant has watched the tape for 3000 periods before the recording starts):
  cartel_alone     the frozen cartel by itself: quotes settle near the monopoly quote h^M
  undercut         an entrant one tick inside the best incumbent quote
  markout          an entrant that undercuts only when its own fills say the spread is rent
  myopic           knows the cartel's reaction function but maximises only today's profit (Lecture 15's
                   myopic policy, discount 0)
  best_response    the exact best response (backward induction over the cartel's reaction function)
  public_attacker  the attacker with public information only: it probes at random, fills in the
                   cartel's reaction table from the public quotes, then plays the solved policy
  identity_anon    regime + mark lag market; entrant sees prints and marks but no wallet ids
  identity_id      same market and seed; entrant also sees which wallet printed
  queue_anon       one-tick book (spread fixed, makers choose rest / out); anonymous selector
  queue_id         same book and seed; identity-aware selector
  savings_*        the ordinary trader (experiments/retail_trader.py): someone betting savings on the direction
                   of the price with no information, on a one-tick book, one bet a day for a year.  Five runs
                   share one draw of outcomes (common random numbers): gambler, myopic and patient Bayesian
                   with no edge; myopic and patient Bayesian with a real edge

Trace format (JSON, columnar; one entry per period in every array of "p"):
  meta   scenario, title, text, mode ("spread" | "queue"), agents [{name, role}], K, tick, horizon,
         benchmarks (hC, hM, PiC, PiM, rent, Pi[h-1]) or the queue's fill values, config, seed, git,
         mark_lag, period_seconds (0.6: the median gap between orders on Hyperliquid), summary
  p      ep, t      episode and period within it
         a          actions per agent (half-spread in ticks; queue mode: 0 out / 1 rest / 2 rest and pay)
         ev         0 no trade, 1 taker buys, 2 taker sells, 3 nobody resting (queue mode)
         inf        1 if the taker was informed (never public; shown only after the fill)
         w, z       taker wallet id (-1: homogeneous takers), regime (0 calm, 1 toxic, -1: none)
         f          index of the maker filled (-1: none)
         r          realised profit per agent this period
         edge       V - mid for the period (the mark; public mark_lag periods later)
         x          expected market-wide profit of the period's best quote (spread mode)
         sv         serving order of the queue, front first (queue mode)
         b          entrant's belief that the next arrival is in the toxic regime (identity scenarios)
         cov        share of the exact best response's path the attacker has identified (public_attacker)
         ph         phase label for the entrant ("probe", "exploit", "undercut", "withdraw", ...)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import GlostenMilgromEnv, MarketConfig
from sim.benchmarks import profit_curve, type_benchmarks
from sim.entrants import MarkoutEntrant, UndercutEntrant, WalletEntrant, hC_function
from sim.queue_agents import AlwaysRest, QueueWalletEntrant
from sim.queue_env import QueueEnv, selection_benchmarks
from experiments.best_response import CartelMDP, DPEntrant
from experiments.entry import FrozenQ
from experiments.public_attacker import IdentifyingEntrant, path_cells_of
from experiments import retail_trader as rt

DEMO = ROOT / "results" / "demo_cartel"
DEMO_TOXIC = ROOT / "results" / "demo_cartel_a05"     # alpha = 0.5: where identity has value (REPLICATION_NOTES)
WARMUP = 3000                 # periods the identity / queue entrant watches the tape before the recording starts
OUT = ROOT / "demo"
EV = {"none": 0, "buy": 1, "sell": 2, "noliq": 3}
PERIOD_SECONDS = 0.6          # median gap between orders on Hyperliquid (results/REPLICATION_NOTES.md)

TEXT = {
    "cartel_alone": ("The cartel alone",
                     "Two Q-learners that never communicated. Their quotes settle far outside the Nash quote, "
                     "near the joint-monopoly quote, and the difference is the rent."),
    "undercut": ("Undercutting entrant",
                 "A third maker quotes one tick inside the best incumbent quote and takes the flow. "
                 "The incumbents answer with their learned reaction."),
    "markout": ("Markout entrant",
                "The entrant reads its own fills: if they are not toxic, the wide spread is rent, and it undercuts. "
                "Otherwise it quotes the competitive spread."),
    "myopic": ("Myopic entrant (knows the cartel, plans one period)",
               "The entrant knows how the cartel will answer and picks the quote that earns most today. "
               "The cartel answers by tightening, and the next periods earn less."),
    "best_response": ("Exact best response",
                      "The entrant knows the cartel's reaction function and solves the game by backward induction. "
                      "It steers the incumbents to states they never visited while learning, then sits one tick inside."),
    "public_attacker": ("Attacker with public information only",
                        "The entrant first quotes at random and records how each incumbent answers (the probing budget). "
                        "Then it solves the reaction table it has identified and plays the result."),
    "identity_anon": ("Anonymous entrant (no wallet ids)",
                      "Informed flow clusters in time (the toxic regime). The entrant sees prints, and their marks only "
                      "after the mark lag, so it learns the regime late."),
    "identity_id": ("Identity entrant (sees wallet ids)",
                    "Same market and seed. The entrant sees which wallet printed: a toxic wallet's print reveals the "
                    "regime before the price does, and the entrant steps back in time."),
    "queue_anon": ("One-tick book, anonymous selector",
                   "The spread cannot move, so makers only choose to rest or step out, and fills go to the front of the "
                   "queue. The selector without wallet ids reacts to marks, late."),
    "queue_id": ("One-tick book, identity-aware selector",
                 "Same book and seed. The selector steps out when toxic wallets print and returns when they stop: "
                 "the always-resting makers are left with the toxic fills."),
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_cartel(d: Path):
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N = int(mk["n_mm"])
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    return mk, Q, int(cfg.get("seed", 0))


def record(env, agents, n_periods: int, E: int = -1, extra=None, Pi=None, warmup: int = 0) -> dict:
    """Play n_periods (whole episodes; a final partial one is cut) and return the period columns.
    With warmup, that many periods are played first and not recorded (the entrant's memory of
    the tape persists across episodes, so it starts the recording already informed)."""
    if warmup:
        record(env, agents, warmup, E=E)
    cols = {k: [] for k in ("ep", "t", "a", "ev", "inf", "w", "z", "f", "r", "edge")}
    if Pi is not None:
        cols["x"] = []
    queue_mode = isinstance(env, QueueEnv)
    if queue_mode:
        cols["sv"] = []
    ext_cols: dict[str, list] = {}
    ent = agents[E] if E >= 0 else None
    ep = 0
    while len(cols["t"]) < n_periods:
        obs = env.reset()
        for ag in agents:
            ag.reset()
        done = False
        while not done and len(cols["t"]) < n_periods:
            acts = [int(ag.act(obs, i)) for i, ag in enumerate(agents)]
            ext = extra(ent) if extra else {}
            t = int(obs["t"])
            obs, r, done, info = env.step(acts)
            f = int(info["filled"])
            if f >= 0 and f == E and hasattr(ent, "observe_fill"):
                ent.observe_fill(float(r[E]))
            cols["ep"].append(ep); cols["t"].append(t); cols["a"].append(acts)
            cols["ev"].append(EV[info["event"]]); cols["inf"].append(int(bool(info["informed"])))
            cols["w"].append(int(info["taker"])); cols["z"].append(int(info["regime"]))
            cols["f"].append(f); cols["r"].append([round(float(v), 3) for v in r])
            cols["edge"].append(round(float(info["V_t"] - info["mid"]), 3))
            if Pi is not None:
                cols["x"].append(round(float(Pi[min(acts) - 1]), 4))
            if queue_mode:
                cols["sv"].append([int(i) for i in env.serving])
            for k, v in ext.items():
                ext_cols.setdefault(k, []).append(v)
        ep += 1
    cols.update(ext_cols)
    return cols


def summarise(cols: dict, names: list, unit: float, unit_name: str) -> dict:
    r = np.array(cols["r"])
    f, inf = np.array(cols["f"]), np.array(cols["inf"])
    fills = np.bincount(f[f >= 0], minlength=len(names))
    informed = np.bincount(f[(f >= 0) & (inf == 1)], minlength=len(names))
    return {"periods": len(r), "unit": unit_name,
            "informed_share_of_fills": {n: round(float(informed[i] / max(fills[i], 1)), 3) for i, n in enumerate(names)},
            "profit_per_period": {n: round(float(r[:, i].mean() / unit), 4) for i, n in enumerate(names)},
            "fills": {n: int(fills[i]) for i, n in enumerate(names)}}


# ------------------------------------------------------------------------------------ spread mode
def spread_scenario(name: str, cartel: Path, n_periods: int, seed_offset: int = 0) -> dict:
    mk, Q, seed = load_cartel(cartel)
    N, K, T = int(mk["n_mm"]), int(mk["max_half_spread"]), int(mk["horizon"])
    het = {}
    if name.startswith("identity"):
        het = dict(n_wallets=50, wallet_concentration=1.0, wallet_seed=seed, regime_persistence=0.97, mark_lag=20)
    base = MarketConfig(**{**mk, **het})
    run_seed = 10_000 + seed + seed_offset           # the evaluation seed of experiments/demo.py
    n = N if name == "cartel_alone" else N + 1
    cfg = MarketConfig(**{**mk, **het, "n_mm": n, "seed": run_seed})
    env = GlostenMilgromEnv(cfg)
    alpha = env.alpha_eff
    Pi = profit_curve(cfg, alpha)
    hC, PiC, hM, PiM = type_benchmarks(cfg, alpha)
    rent = PiM - PiC
    inc = [FrozenQ(Q[i], K, n) for i in range(N)]
    agents = [{"name": f"incumbent {i + 1}", "role": "incumbent"} for i in range(N)]
    extra, E, ent = None, -1, None
    if name == "undercut":
        ent = UndercutEntrant()
    elif name == "markout":
        ent = MarkoutEntrant(hC_function(MarketConfig(**mk)), alpha_prior=0.3, k_min=20, margin=2)
        extra = lambda e: {"ph": e.last_decision or ""}
    elif name == "myopic":
        mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": n}), T)
        today = (mdp.r_own.argmax(axis=1) + 1).astype(np.int8)
        ent = DPEntrant(mdp, np.tile(today, (T, 1)))
    elif name == "best_response":
        mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": n}), T)
        policy, _, _ = mdp.solve(1.0)
        ent = DPEntrant(mdp, policy)
    elif name == "public_attacker":
        mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": n}), T)
        policy, _, _ = mdp.solve(1.0)
        path = path_cells_of(mdp, policy)
        probe = min(300, n_periods // 2)
        ent = IdentifyingEntrant(cfg, N, probe, 100, "competitive", hC, seed=seed)
        extra = lambda e: {"ph": "probe" if e.t_global <= e.probe else "exploit", "cov": round(e.coverage(path), 3)}
    elif name in ("identity_anon", "identity_id"):
        ent = WalletEntrant(cfg, name.split("_")[1], wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff, margin=1)
        extra = lambda e: {"ph": e.last_decision or "", "b": round(float(e.last_b), 3) if e.last_b is not None else None}
    elif name != "cartel_alone":
        raise ValueError(name)
    if ent is not None:
        E = N
        agents.append({"name": "entrant", "role": "entrant"})
    cols = record(env, inc + ([ent] if ent is not None else []), n_periods, E=E, extra=extra, Pi=Pi,
                  warmup=WARMUP if het else 0)
    names = [a["name"] for a in agents]
    title, text = TEXT[name]
    meta = dict(scenario=name, title=title, text=text, mode="spread", agents=agents, K=K, tick=float(cfg.tick),
                horizon=T, mark_lag=int(cfg.mark_lag), period_seconds=PERIOD_SECONDS, seed=run_seed, git=git_hash(),
                cartel=cartel.name, config={k: v for k, v in cfg.__dict__.items()},
                benchmarks=dict(hC=int(hC), hM=int(hM), PiC=round(float(PiC), 4), PiM=round(float(PiM), 4),
                                rent=round(float(rent), 4), Pi=[round(float(p), 4) for p in Pi], alpha=round(float(alpha), 4)),
                summary=summarise(cols, names, rent, "rent (PiM - PiC)"))
    if env.wallet_alpha is not None:
        meta["wallet_alpha"] = [round(float(a), 3) for a in env.wallet_alpha]
    return {"meta": meta, "p": cols}


# ------------------------------------------------------------------------------------- queue mode
def queue_scenario(name: str, n_periods: int, seed: int = 0) -> dict:
    mk = dict(sigma_L=8.0, tick=0.5, max_half_spread=24, redraw_v_each_period=True, edge_dist="exponential",
              edge_mean=5.0, n_wallets=50, wallet_concentration=1.0, alpha=0.3, regime_persistence=0.97,
              mark_lag=20, wallet_seed=seed)
    n_inc = 3
    cfg = MarketConfig(**mk, n_mm=n_inc + 1, seed=10_000 + seed)
    env = QueueEnv(cfg, ticks_above_breakeven=1)
    h = env.h
    bm = selection_benchmarks(env)
    ent = QueueWalletEntrant(cfg, h, name.split("_")[1], wallet_alpha=env.wallet_alpha, alpha_eff=env.alpha_eff)
    agents = [{"name": f"resting maker {i + 1}", "role": "incumbent"} for i in range(n_inc)]
    agents.append({"name": "selector", "role": "entrant"})
    extra = lambda e: {"ph": e.last_decision or "", "b": round(float(e.last_b), 3) if e.last_b is not None else None}
    cols = record(env, [AlwaysRest() for _ in range(n_inc)] + [ent], n_periods, E=n_inc, extra=extra, warmup=WARMUP)
    names = [a["name"] for a in agents]
    unit = abs(bm["fill_value_calm"]) or 1.0
    title, text = TEXT[name]
    meta = dict(scenario=name, title=title, text=text, mode="queue", agents=agents, K=int(cfg.max_half_spread),
                tick=float(cfg.tick), horizon=int(cfg.horizon), mark_lag=int(cfg.mark_lag),
                period_seconds=PERIOD_SECONDS, seed=cfg.seed, git=git_hash(), config=dict(cfg.__dict__),
                benchmarks={k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in bm.items()},
                wallet_alpha=[round(float(a), 3) for a in env.wallet_alpha],
                summary=summarise(cols, names, unit, "value of one calm-regime fill"))
    return {"meta": meta, "p": cols}


# ---------------------------------------------------------------------------------- the ordinary trader
SPEC = {    # name: (policy discount, None = never updates; true probability of a right call; title; text)
    "savings_gambler": (None, 0.5, "Betting savings: never updates (no edge)",
                        "One bet a day on the direction of the price, with no information, and no change of mind however it goes."),
    "savings_myopic": (0.0, 0.5, "Betting savings: myopic Bayesian (no edge)",
                       "Same luck, same optimistic start. She updates her belief after every bet and stops for good once "
                       "today's bet is no longer worth it on her own belief."),
    "savings_patient": (0.99, 0.5, "Betting savings: patient Bayesian (no edge)",
                        "Same luck. She also values what each bet teaches her, so she keeps experimenting longer."),
    "savings_edge_myopic": (0.0, 0.6, "Real edge: myopic Bayesian",
                            "This time she really is right 60% of the time. The myopic rule can still give up after an early "
                            "run of bad luck: the trap of Lecture 15."),
    "savings_edge_patient": (0.99, 0.6, "Real edge: patient Bayesian",
                             "Same edge, same luck. Valuing information keeps her betting through early losses."),
}
TEXT.update({k: (v[2], v[3]) for k, v in SPEC.items()})


def speculator_scenario(name: str, n_periods: int | None = None, seed: int = 0) -> dict:
    """One year of the savings bettor of experiments/retail_trader.py on a one-tick book.  The day's
    outcome draws are shared by every policy (common random numbers), so two runs side by side see the
    same luck; the text quotes the many-run probabilities, because one year is one draw."""
    mk, _, _ = load_cartel(DEMO)
    cfg = MarketConfig(**mk)
    mu, tick, N = float(cfg.edge_mean), float(cfg.tick), rt.N_DAYS
    x = 1 * tick                                                # one-tick book: the half-spread is one tick
    a, b = rt.PRIORS["overconfident Beta(6,4)"]
    delta, p_true, title, text = SPEC[name]
    trade = None if delta is None else rt.solve_speculator(a, b, x, mu, delta)[0]
    rng = np.random.default_rng(20_000 + seed)
    U, D = rng.random(N), rng.exponential(mu, N)
    w, n, k, ended, why = rt.SAVINGS * mu, 0, 0, None, ""
    cols = {c: [] for c in ("t", "bet", "right", "move", "r", "w", "n", "k", "ph", "rec")}
    for t in range(n_periods or N):
        go = ended is None and w > 0 and (trade is None or bool(trade[n, k]))
        if not go and ended is None:
            ended, why = t, ("savings gone" if w <= 0 else "stopped for good")
        right, pnl = -1, 0.0
        if go:
            right = int(U[t] < p_true)
            pnl = (D[t] if right else -D[t]) - x
            w += pnl; n += 1; k += right
        cols["t"].append(t); cols["bet"].append(int(go)); cols["right"].append(right)
        cols["move"].append(round(float(D[t] / mu), 3) if go else 0.0)
        cols["r"].append(round(float(pnl / mu), 4)); cols["w"].append(round(float(w / mu), 4))
        rec = round((a + k) / (a + b + n), 4)                  # what her record says (the Bayesian posterior mean)
        cols["n"].append(n); cols["k"].append(k); cols["rec"].append(rec)
        cols["ph"].append(round(a / (a + b), 4) if delta is None else rec)   # her belief: the gambler never updates
    line = rt.quitting_line(trade) if trade is not None else None
    many = {**rt.evaluate_exact(trade, p_true, x, mu), **rt.evaluate_mc(trade, p_true, x, mu)}
    days = len(cols["t"])
    ruin = f"{many['p_ruin']:.0%}" if many["p_ruin"] >= 0.1 else f"{many['p_ruin']:.1%}"
    meta = dict(scenario=name, title=title, mode="speculator", agents=[{"name": "trader", "role": "entrant"}],
                text=text + f" Over many years with these rules: ruin {ruin}, expected result "
                            f"{many['exp_profit_mu']:+.1f} average price moves, {many['exp_trades']:.0f} bets.",
                horizon=N, period_seconds=None, unit_label="trading day", seed=seed, git=git_hash(),
                params=dict(mu=mu, tick=tick, x=x, hurdle=rt.hurdle(x, mu), prior=[a, b], p_true=p_true,
                            delta=delta, savings=rt.SAVINGS, policy="never updates" if delta is None else
                            ("myopic Bayesian" if delta == 0 else f"patient Bayesian (delta {delta:g})")),
                quitting_line=None if line is None else [int(v) for v in line],
                many_runs={k_: round(float(v), 4) for k_, v in many.items()},
                ended=ended, ended_why=why,
                summary={"periods": days, "unit": "average price move", "profit_per_period": {"trader": round(float(np.mean(cols["r"])), 4)},
                         "bets": n, "right": k, "final_savings": cols["w"][-1]})
    return {"meta": meta, "p": cols}


SCENARIOS = list(TEXT)


def make(name: str, cartel: Path = DEMO, n_periods: int | None = None) -> dict:
    if name in SPEC:
        return speculator_scenario(name, n_periods)
    if name.startswith("queue"):
        return queue_scenario(name, n_periods or 400)
    if name.startswith("identity") and cartel == DEMO and DEMO_TOXIC.exists():
        cartel = DEMO_TOXIC
    return spread_scenario(name, cartel, n_periods or (600 if name == "public_attacker" else 400))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cartel", default=str(DEMO))
    ap.add_argument("--periods", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.list:
        for s in SCENARIOS:
            print(f"{s:<16} {TEXT[s][0]}")
        return
    names = SCENARIOS if args.all else [args.scenario or "best_response"]
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    for s in names:
        tr = make(s, Path(args.cartel), args.periods)
        (out / "traces" / f"{s}.json").write_text(json.dumps(tr, separators=(",", ":"), default=_default))
        sm = tr["meta"]["summary"]
        print(f"{s:<16} {sm['periods']} periods  " + "  ".join(f"{k}: {v:+.2f}" for k, v in sm["profit_per_period"].items())
              + f"   ({sm['unit']})")
    # one script file with every trace on disk, so the player also works from file:// (no fetch)
    bundle = {p.stem: json.loads(p.read_text()) for p in sorted((out / "traces").glob("*.json"))}
    order = [s for s in SCENARIOS if s in bundle]
    (out / "traces.js").write_text("window.TRACES = " + json.dumps({s: bundle[s] for s in order}, separators=(",", ":")) + ";\n")
    print(f"wrote {out / 'traces.js'} ({len(order)} scenarios)")


if __name__ == "__main__":
    main()
