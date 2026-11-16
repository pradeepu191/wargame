"""One profit--resilience frontier for every lever (RQ3).

Each lever setting is one point: x = the incumbents' profit with no attack, y = their profit
under the worst attacker we have (minimum over adversaries, per cartel or seed, then averaged;
95% intervals over cartels or seeds).  The number printed next to a point is what an ordinary
taker gets under that setting, so that "robust" cannot mean "withdrew": the fill rate under
attack in the spread game; in the queue game (where an always-resting fringe keeps the book
filled, so the fill probability does not move) the profit of that fringe, the identity-blind
liquidity that pays for the selection rent.  Up and to the right is better.

  left    spread free (Glosten--Milgrom grid): the makers' own levers, from defences.csv, plus
          the exact best response and the public-information attacker from defences_strong.csv
          when present (same cartels, matched by alpha and seed)
  right   spread pinned at one tick (queue game): the venue's levers -- anonymity (may the
          incumbents condition on wallet ids), the priority fee, the request allowance

    python analysis/frontier_fig.py            # -> results/fig_frontier.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DEF_C = {"none": "#0b0b0b", "random:0.1": "#e67e22", "random:0.3": "#c0392b", "hold:2": "#7f8c8d", "hold:5": "#52514e",
         "floor:anon": "#1baf7a", "floor:id": "#8e44ad"}
DEF_L = {"none": "no defence", "random:0.1": "randomise 10%", "random:0.3": "randomise 30%", "hold:2": "requote every 2",
         "hold:5": "requote every 5", "floor:anon": "floor, tape forecast", "floor:id": "floor, identity forecast"}


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def point(ax, x, y, c, label, note=None, marker="o"):
    ax.errorbar(np.mean(x), np.mean(y), xerr=ci(x), yerr=ci(y), color=c, marker=marker, ms=5.5, capsize=2, lw=1.1, label=label)
    if note is not None:
        ax.annotate(note, (np.mean(x), np.mean(y)), xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=MUTED)


def spread_panel(ax, results: Path, alpha: float, y_col: str):
    d = pd.read_csv(results / "defences.csv")
    strong = results / "defences_strong.csv"
    n_strong = 0
    if strong.exists():
        s = pd.read_csv(strong)
        # the strong attackers were run on locally re-trained cartels: keep (alpha, seed, defence) cells whose
        # no-attack profit agrees with the cluster run, so that both files describe the same cartel
        key = ["alpha", "seed", "defence"]
        ref = d.groupby(key).Pi_nominal_norm.first().rename("ref")
        s = s.join(ref, on=key)
        s = s[(s.Pi_nominal_norm - s.ref).abs() < 0.05]
        n_strong = s[np.isclose(s.alpha, alpha)].seed.nunique()
        d = pd.concat([d, s[d.columns.intersection(s.columns)]], ignore_index=True)
    d = d[np.isclose(d.alpha, alpha) & (d.adversary != "competitive")]
    w = d.groupby(["defence", "seed"]).agg(nom=("Pi_nominal_norm", "first"), worst=(y_col, "min"),
                                           fill=("attack_fill_rate", "mean")).reset_index()
    for name, g in w.groupby("defence"):
        point(ax, g.nom, g.worst, DEF_C.get(name, MUTED), DEF_L.get(name, name), note=f"{g.fill.mean():.2f}")
    advs = sorted(d.adversary.unique())
    ax.set_title(rf"spread free: the makers' levers ($\bar\alpha$ = {alpha}, {w.seed.nunique()} cartels,"
                 f" {len(advs)} attackers" + (f"; strong attackers on {n_strong}" if n_strong else "") + ")", loc="left", fontsize=8.5)
    ax.set_xlabel(r"cartel's profit with no attack / $(\Pi^M-\Pi^C)$")
    return advs


def queue_panel(ax, results: Path, rho_z: float, L: int, above: int):
    def cell(df):
        return df[np.isclose(df.rho_z, rho_z) & (df.mark_lag == L) & (df.above == above)]

    def design(df, policies, nominal="always"):
        """Per seed: incumbents' profit facing an always-resting entrant, their worst over the entrant's policies, and
        the always-resting fringe's worst profit (the identity-blind liquidity an ordinary taker relies on)."""
        g = df[df.policy.isin(policies)]
        worst = g.groupby("seed").incumbent_pnl.min()
        nom = df[df.policy == nominal].groupby("seed").incumbent_pnl.mean()
        fill = g.groupby("seed").fringe_pnl.min()
        return nom.reindex(worst.index), worst, fill

    drew = []
    f_anon = results / "queue_hedge_always_never_anon_fringe2.csv"
    f_id = results / "queue_hedge_always_never_anon_id_fringe2.csv"
    if f_anon.exists():
        nom, worst, fill = design(cell(pd.read_csv(f_anon)), ("always", "anon"))
        point(ax, nom, worst, "#2f6fb3", "anonymous venue", note=f"{fill.mean():+.2f}", marker="s"); drew.append("anonymity")
    if f_id.exists():
        nom, worst, fill = design(cell(pd.read_csv(f_id)), ("always", "anon", "id"))
        point(ax, nom, worst, "#8e44ad", "public wallets (Hyperliquid)", note=f"{fill.mean():+.2f}", marker="s")
    f_fee = results / "queue_fee_id.csv"
    if f_fee.exists():
        q = cell(pd.read_csv(f_fee))
        fees = [f for f in sorted(q.fee.unique()) if f > 0]
        for k, fee in enumerate(fees):
            g = q[np.isclose(q.fee, fee)]
            nom, worst, fill = design(g, tuple(g.policy.unique()))
            point(ax, nom, worst, plt.cm.YlOrBr(0.35 + 0.6 * k / max(len(fees) - 1, 1)), f"priority fee {fee:g}", note=f"{fill.mean():+.2f}", marker="^")
        drew.append("fee")
    f_b = results / "queue_budget.csv"
    if f_b.exists():
        q = cell(pd.read_csv(f_b))
        bs = [b for b in sorted(q.budget0.unique(), reverse=True) if np.isfinite(b)]
        for k, b in enumerate(bs):
            g = q[np.isclose(q.budget0, b)]
            nom, worst, fill = design(g, tuple(g.policy.unique()))
            point(ax, nom, worst, plt.cm.Greens(0.35 + 0.6 * k / max(len(bs) - 1, 1)), f"request allowance {b:g}", note=f"{fill.mean():+.2f}", marker="D")
        drew.append("allowance")
    ax.set_title(rf"spread pinned at one tick: the venue's levers ($\rho_z$ = {rho_z}, L = {L})", loc="left", fontsize=8.5)
    ax.set_xlabel("incumbent's profit per period, entrant always rests")
    ax.set_ylabel("incumbent's profit against the worst entrant")
    return drew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--y", default="attack_cvar_norm", choices=["J_attack_norm", "attack_cvar_norm"])
    ap.add_argument("--rho-z", type=float, default=0.9)
    ap.add_argument("--mark-lag", type=int, default=5)
    ap.add_argument("--above", type=int, default=0)
    ap.add_argument("--out", default="fig_frontier")
    args = ap.parse_args()
    results = Path(args.results)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    advs = spread_panel(a1, results, args.alpha, args.y)
    a1.set_ylabel(("worst 5% of episodes" if args.y == "attack_cvar_norm" else "mean profit") + r" under the worst attacker / $(\Pi^M-\Pi^C)$")
    queue_panel(a2, results, args.rho_z, args.mark_lag, args.above)
    for ax in (a1, a2):
        ax.axhline(0, color=MUTED, lw=0.6); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.text(0.01, -0.03, "Numbers next to the points. Left: the fill rate under attack. Right: the profit per period of the always-resting "
             "fringe (identity-blind liquidity). Attackers, left: " + ", ".join(advs) + ".", fontsize=7, color=MUTED)
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
