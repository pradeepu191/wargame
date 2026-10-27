"""Strategizing against a live no-regret market maker.  Left: what each attacker earns and what
the incumbents lose (rent units, evaluation seeds, 95% intervals) against a single EXP3 learner
and against two, with the Stackelberg value as a line.  Right: the cartel's rent alone, N = 1 vs
N = 2 (a no-regret learner without memory finds the monopoly quote alone and competes when
there are two).
    python analysis/no_regret_fig.py [--results results] [--alpha 0.3] [--eta 0.05]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9890"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--out", default="fig_no_regret")
    args = ap.parse_args()
    res = Path(args.results)
    d = pd.read_csv(res / "no_regret.csv")
    d = d[np.isclose(d.alpha, args.alpha) & np.isclose(d.eta, args.eta)]
    if "selection" in d:
        d = d[(d.attacker != "cycle") | (~d.selection.fillna(False).astype(bool))]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    n_incs = sorted(d.n_inc.unique())
    fig, axes = plt.subplots(1, len(n_incs), figsize=(5.2 * len(n_incs), 3.6), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    order = [("undercut", "undercut rule\n(one tick inside)"), ("undercut_floor", "undercut rule,\nfloored at $h^C$"),
             ("cycle", "lure-then-undercut cycle\n(chosen on other seeds)"), ("qlearn", "Q-learning entrant\n(learns online)")]
    for ax, n in zip(axes, n_incs):
        g = d[d.n_inc == n]
        y = np.arange(len(order))[::-1]
        h = 0.36
        for yi, (key, lab) in zip(y, order):
            s = g[g.attacker == key]
            if s.empty:
                continue
            L, e = s.inc_loss_norm.mean(), s.entrant_norm.mean()
            ax.barh(yi + h / 2, L, h, color=BLUE, xerr=ci(s.inc_loss_norm), error_kw=dict(ecolor=INK, lw=0.8, capsize=2),
                    label="what the learners lose" if yi == y[0] else None)
            ax.barh(yi - h / 2, e, h, color=ORANGE, xerr=ci(s.entrant_norm), error_kw=dict(ecolor=INK, lw=0.8, capsize=2),
                    label="what the attacker earns" if yi == y[0] else None)
            ax.annotate(f"{L:.2f}", (max(L, 0) + ci(s.inc_loss_norm), yi + h / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
            ax.annotate(f"{e:.2f}", (max(e, 0) + ci(s.entrant_norm), yi - h / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
        sv = g.stackelberg_norm.iloc[0]
        C = g.C_alone.mean()
        ax.axvline(sv, color=AQUA, lw=1.2, ls=":")
        ax.axvline(C, color=MUTED, lw=1, ls="--")
        ax.annotate(f"Stackelberg value {sv:.2f}", (sv + 0.03, y[-1] - 0.55), fontsize=8, color=AQUA, va="center")
        ax.annotate(f"rent alone C = {C:.2f}", (C + 0.03, y[-1] - 0.85), fontsize=8, color=MUTED, va="center")
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in order], fontsize=8.5)
        ax.set_ylim(-1.2, len(order) + 0.3)
        n_seeds = g[g.attacker == "cycle"].seed.nunique() or g.seed.nunique()
        ax.set_title(f"{n} EXP3 learner{'s' if n > 1 else ''}, still learning under attack\n"
                     f"($\\bar\\alpha$ = {args.alpha}, $\\eta$ = {args.eta}; {n_seeds} held-out seeds)", fontsize=9.5, loc="left")
        ax.set_xlabel(r"rent units ($1$ = the monopoly rent $\Pi^M-\Pi^C$)")
        ax.grid(color=GRID, lw=0.8, axis="x"); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    out = res / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
