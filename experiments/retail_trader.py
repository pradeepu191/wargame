"""The ordinary trader: what is optimal for someone who has no information about the market.

The rest of the project takes the makers' and the attacker's side.  This script takes the side
of the participant with the least information, in the frame of Lecture 15 ("Planning with a
Model"), and answers three questions with the model we already have.

A. What does the market's structure charge an ordinary customer?  (needs local trained cartels)
   Along the model's path from the episode's starting state (experiments/planning_view.py), the
   best quote a customer faces when the cartel is alone, and with each entrant.  Two readings:
     * a customer who has to trade (moving savings in or out) keeps the gains from trade
       S(h) = sim.benchmarks.taker_surplus_curve; we report it as a share of S at the Nash quote;
     * a customer who bets on the direction of the price pays the half-spread x = h * tick on
       every trade, so he needs to be right with probability above the hurdle
           p* = 1/2 + x / (2 mu),           mu = mean size of a price move (edge_mean),
       because a directional market order earns (2p - 1) mu - x in expectation.

B. The speculator as a belief MDP (Lecture 15, part 2, with the trader in place of the ship).
   He does not know p, the probability that his call on the direction is right.  His belief is
   Beta(a + k, b + n - k) after n trades with k right: the state is (n, k).  Each period he
   either trades (one unit; expected reward (2 p_hat - 1) mu - x; he sees whether he was right)
   or stops for good (reward 0; stopping reveals nothing, so the belief never moves again, and
   stopping is optimal for ever once it is optimal, as slide 30 shows for the ship).  Turn limit:
   N trading days.  Bellman:
       V(n, k) = max{ 0,  (2 p_hat - 1) mu - x + delta [ p_hat V(n+1, k+1) + (1 - p_hat) V(n+1, k) ] }.
   Policies: the gambler (trades every day, never updates), the myopic Bayesian (delta = 0), the
   patient Bayesian (delta = 0.9, 0.99), and never trading.  Each is evaluated under the TRUE p
   by exact propagation of the probability over (n, k) (expected profit, trades, the probability
   of still trading at the end) and by Monte Carlo with real price moves (probability of losing
   the savings set aside).  p = 1/2 is a trader with no information; p above the hurdle is one
   with a real edge, for whom the danger is the opposite one, the trap of slide 37.

C. What if he posts a limit order instead?  On a one-tick book his resting order is the
   identity-blind "fringe" of the queue game (results/queue_budget.csv): it is filled by the
   flow the identity-aware makers step away from.  Read from the committed CSV.

This is a stylised model for research, not investment advice.

    python experiments/retail_trader.py                     # -> results/retail_*.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig
from sim.benchmarks import taker_surplus_curve
from experiments.best_response import CartelMDP

N_DAYS = 250                     # trading opportunities: one a day for a year
SAVINGS = 20.0                   # what he sets aside, in units of mu (twenty average price moves)
PRIORS = {"overconfident Beta(6,4)": (6.0, 4.0), "unsure Beta(1,1)": (1.0, 1.0)}
DELTAS = {"myopic Bayesian": 0.0, "patient Bayesian (delta 0.9)": 0.9, "patient Bayesian (delta 0.99)": 0.99}


def hurdle(x: float, mu: float) -> float:
    """Probability of being right that a directional market order needs to break even."""
    return 0.5 + x / (2.0 * mu)


# ------------------------------------------------------------------------------------- part B
def solve_speculator(a: float, b: float, x: float, mu: float, delta: float, N: int = N_DAYS):
    """Backward induction on the belief MDP.  Returns trade[n, k] (True: trade at belief (n, k)),
    for n = 0..N-1, k = 0..n, and V(0, 0).  Ties go to stopping (the lecture breaks ties toward L)."""
    trade = np.zeros((N, N + 1), dtype=bool)
    V_next = np.zeros(N + 2)                                   # V(N, k) = 0: no days left
    for n in range(N - 1, -1, -1):
        k = np.arange(n + 1)
        p = (a + k) / (a + b + n)
        cont = (2 * p - 1) * mu - x + delta * (p * V_next[k + 1] + (1 - p) * V_next[k])
        go = cont > 1e-12
        trade[n, : n + 1] = go
        V = np.zeros(N + 2)
        V[: n + 1] = np.where(go, cont, 0.0)
        V_next = V
    return trade, float(V_next[0])


def quitting_line(trade: np.ndarray) -> np.ndarray:
    """losses*(n): the smallest number of wrong calls after n trades at which he stops (n + 1 if never)."""
    N = trade.shape[0]
    out = np.full(N, np.nan)
    for n in range(N):
        stop_losses = [n - k for k in range(n + 1) if not trade[n, k]]
        out[n] = min(stop_losses) if stop_losses else n + 1
    return out


def evaluate_exact(trade: np.ndarray | None, p_true: float, x: float, mu: float, N: int = N_DAYS) -> dict:
    """Propagate the probability of each belief (n, k) while he is still trading, under the true p.
    trade None is the gambler (trades every day).  Profit in units of mu."""
    r = (2 * p_true - 1) * mu - x
    alive = np.zeros(N + 2); alive[0] = 1.0
    exp_trades, stopped_by = 0.0, np.zeros(N + 1)
    for n in range(N):
        go = np.ones(n + 1, bool) if trade is None else trade[n, : n + 1]
        mass = alive[: n + 1]
        stopped_by[n] = mass[~go].sum()
        m = np.where(go, mass, 0.0)
        exp_trades += m.sum()
        nxt = np.zeros(N + 2)
        nxt[1: n + 2] += p_true * m
        nxt[: n + 1] += (1 - p_true) * m
        alive = nxt
    still = alive[: N + 1].sum()
    return {"exp_profit_mu": exp_trades * r / mu, "exp_trades": exp_trades, "p_still_trading_at_end": still,
            "p_stopped_within_20": float(stopped_by[:20].sum())}


def evaluate_mc(trade: np.ndarray | None, p_true: float, x: float, mu: float, N: int = N_DAYS,
                savings: float = SAVINGS, n_paths: int = 20_000, seed: int = 0) -> dict:
    """Monte Carlo with real price moves (d ~ Exponential(mu)): the probability that he loses all
    the savings he set aside (he stops when they are gone), and the median outcome."""
    rng = np.random.default_rng(seed)
    w = np.full(n_paths, savings * mu)
    k = np.zeros(n_paths, dtype=np.int64)
    n = np.zeros(n_paths, dtype=np.int64)
    active = np.ones(n_paths, bool)
    for t in range(N):
        if trade is not None:
            active &= trade[np.minimum(n, N - 1), np.minimum(k, N)] & (n == t)
        active &= w > 0
        if not active.any():
            break
        right = rng.random(n_paths) < p_true
        d = rng.exponential(mu, n_paths)
        pnl = np.where(right, d, -d) - x
        w = np.where(active, w + pnl, w)
        k = np.where(active & right, k + 1, k)
        n = np.where(active, n + 1, n)
    final = (w - savings * mu) / mu
    return {"p_ruin": float(np.mean(w <= 0)), "median_profit_mu": float(np.median(final)),
            "p_loss_half": float(np.mean(final <= -savings / 2))}


def part_b(mu: float, tick: float, ticks=(1, 2), truths=(0.5, 0.55, 0.6)) -> pd.DataFrame:
    rows = []
    for h in ticks:
        x = h * tick
        for prior, (a, b) in PRIORS.items():
            policies = {"gambler (never updates)": None}
            for name, delta in DELTAS.items():
                policies[name] = solve_speculator(a, b, x, mu, delta)[0]
            for p in truths:
                for name, tr in policies.items():
                    if tr is None and prior != list(PRIORS)[0]:
                        continue                                # the gambler ignores his prior
                    rows.append({"half_spread_ticks": h, "x": x, "hurdle": hurdle(x, mu), "prior": prior if tr is not None else "-",
                                 "policy": name, "p_true": p, **evaluate_exact(tr, p, x, mu), **evaluate_mc(tr, p, x, mu)})
    return pd.DataFrame(rows)


def quitting_lines(mu: float, tick: float, h: int = 1, n_show: int = 60) -> pd.DataFrame:
    x = h * tick
    a, b = PRIORS["overconfident Beta(6,4)"]
    rows = []
    for name, delta in DELTAS.items():
        line = quitting_line(solve_speculator(a, b, x, mu, delta)[0])
        rows += [{"policy": name, "delta": delta, "n": n, "losses_to_stop": line[n]} for n in range(n_show)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- part A
def best_quote_paths(d: Path):
    """Best quote a customer faces, per period, along the model's path from the starting state:
    the cartel alone, and with the one-tick undercut rule, the myopic entrant, the exact best response."""
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    mk = dict(cfg["market"])
    N = int(mk["n_mm"])
    m = MarketConfig(**mk)
    K, T = m.max_half_spread, m.horizon
    Q = [np.load(d / f"Q_agent{i}.npy") for i in range(N)]
    mdp = CartelMDP(Q, MarketConfig(**{**mk, "n_mm": N + 1}), T)
    out = {}
    hs = np.full(N, K)                                          # the cartel alone, from everyone at K
    seq = []
    for _ in range(T):
        hs = np.array([mdp.R[i, hs[i], np.delete(hs, i).min()] for i in range(N)])
        seq.append(hs.min())
    out["cartel alone"] = np.array(seq)
    pol_opt, _, _ = mdp.solve(1.0)
    myopic = mdp.r_own.argmax(axis=1)
    last_inc = mdp.h[:, :N].min(axis=1)
    rules = {"one-tick undercut": lambda t, x: max(last_inc[x] - 1, 1) - 1,
             "myopic entrant": lambda t, x: myopic[x],
             "exact best response": lambda t, x: pol_opt[t, x] - 1}
    for name, choose in rules.items():
        x, seq = mdp.s0, []
        for t in range(T):
            a = int(choose(t, x))
            seq.append(min(int(mdp.g[x].min()), a + 1))
            x = int(mdp.base[x] + a)
        out[name] = np.array(seq)
    return mk, out


def part_a(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        mk, paths = best_quote_paths(d)
        m = MarketConfig(**mk)
        alpha = float(mk["alpha"])
        hC, _, hM, _ = BenchmarkTable(GlostenMilgromEnv(m)).lookup(0.5)
        S = taker_surplus_curve(m, alpha)
        mu = float(m.edge_mean)
        for name, seq in {"Nash quote": np.full(m.horizon, hC), **paths}.items():
            x = seq * m.tick
            rows.append({"run": d.name, "alpha": alpha, "structure": name, "best_quote": float(seq.mean()),
                         "customer_gain_share": float(S[seq - 1].mean() / S[hC - 1]),
                         "hurdle": float(np.mean(hurdle(x, mu))), "loss_per_bet_mu": float(x.mean() / mu)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- part C
def part_c(results: Path) -> pd.DataFrame | None:
    f = results / "queue_budget.csv"
    if not f.exists():
        return None
    q = pd.read_csv(f)
    q = q[np.isclose(q.rho_z, 0.9) & q.policy.isin(["always", "id"])]
    return q.groupby(["budget0", "policy"]).agg(resting_order_profit=("fringe_pnl", "mean"), n_seeds=("seed", "nunique")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/cfl_exp2_alpha*_n_mm2_seed*"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)
    cfg = MarketConfig(**yaml.safe_load((Path("results/demo_cartel") / "config.yaml").read_text())["market"])
    mu, tick = float(cfg.edge_mean), float(cfg.tick)
    pd.set_option("display.width", 200)

    dirs = sorted({Path(p) for g in args.dirs for p in glob.glob(g) if (Path(p) / "Q_agent0.npy").exists()})
    if dirs:
        a = part_a(dirs)
        a.to_csv(results / "retail_prices.csv", index=False)
        order = ["Nash quote", "cartel alone", "one-tick undercut", "myopic entrant", "exact best response"]
        print(f"A. What the market charges an ordinary customer ({a.run.nunique()} cartels; columns: alpha)")
        for col, label in (("best_quote", "best quote, ticks"), ("customer_gain_share", "gains from trade kept, share of the Nash quote's"),
                           ("hurdle", "share of calls he must get right to break even (above 1: impossible)")):
            print(f"\n  {label}")
            print(a.pivot_table(index="structure", columns="alpha", values=col).reindex(order).round(2).to_string())
    else:
        print("A skipped: no local trained cartels")

    b = part_b(mu, tick)
    b.to_csv(results / "retail_speculator.csv", index=False)
    print(f"\nB. The speculator, {N_DAYS} trading days, savings {SAVINGS:g} average price moves (profit in average price moves)")
    cols = ["half_spread_ticks", "hurdle", "prior", "policy", "p_true", "exp_profit_mu", "exp_trades",
            "p_still_trading_at_end", "p_ruin", "p_loss_half"]
    print(b[cols].round(3).to_string(index=False))
    ql = quitting_lines(mu, tick)
    ql.to_csv(results / "retail_quitting_lines.csv", index=False)

    c = part_c(results)
    if c is not None:
        c.to_csv(results / "retail_limit_orders.csv", index=False)
        print("\nC. A resting limit order on a one-tick book (the identity-blind fringe), profit per period;"
              " policy = what the professional selector does; budget0 = request allowance (inf: none)")
        print(c.round(3).to_string(index=False))
    print("\nwrote results/retail_*.csv")


if __name__ == "__main__":
    main()
