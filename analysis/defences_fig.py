"""The profit--resilience frontier of the makers' own levers.  One point per defence: x = the
defenders' profit alone (rent units), y = their worst-case profit under attack (minimum over the
profit-maximising adversaries, per cartel, then averaged); error bars 95% over cartels.  Panels by
population toxicity.  Up and to the right is better; a defence that only moves left is a cost.
    python analysis/defences_fig.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"none": "#0b0b0b", "random:0.1": "#e67e22", "random:0.3": "#c0392b", "hold:2": "#7f8c8d", "hold:5": "#52514e",
     "floor:anon": "#1baf7a", "floor:id": "#8e44ad"}
LABEL = {"none": "no defence", "random:0.1": "randomise quote (ε = 0.1)", "random:0.3": "randomise quote (ε = 0.3)",
         "hold:2": "requote every 2 periods", "hold:5": "requote every 5 periods",
         "floor:anon": "floor at break-even of the tape's forecast", "floor:id": "floor at break-even of the identity forecast"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", default="defences.csv")
    ap.add_argument("--y", default="J_attack_norm", choices=["J_attack_norm", "attack_cvar_norm"])
    ap.add_argument("--out", default="fig_defence_frontier")
    args = ap.parse_args()
    results = Path(args.results)
    d = pd.read_csv(results / args.csv)
    pm = d[d.adversary != "competitive"]
    w = pm.groupby(["alpha", "defence", "run"]).agg(nom=("Pi_nominal_norm", "first"), worst=(args.y, "min")).reset_index()
    alphas = sorted(w.alpha.unique())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.6 * len(alphas), 3.6), constrained_layout=True, squeeze=False)
    for ax, a in zip(axes[0], alphas):
        g = w[np.isclose(w.alpha, a)].groupby("defence").agg(x=("nom", "mean"), x_sd=("nom", "std"), y=("worst", "mean"), y_sd=("worst", "std"), cnt=("nom", "count"))
        for dname, r in g.iterrows():
            ax.errorbar(r.x, r.y, xerr=1.96 * r.x_sd / np.sqrt(r.cnt), yerr=1.96 * r.y_sd / np.sqrt(r.cnt), color=C.get(dname, MUTED),
                        marker="o", ms=6 if dname == "none" else 5, capsize=2, lw=1.2, label=LABEL.get(dname, dname))
        n = int(w[np.isclose(w.alpha, a)].run.nunique())
        ax.set_title(rf"$\bar\alpha$ = {a} ({n} cartels)", loc="left", fontsize=9)
        ax.set_xlabel(r"defenders' profit alone / $(\Pi^M-\Pi^C)$")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ylabel = "worst-case profit under attack" if args.y == "J_attack_norm" else "worst-case 5% CVaR under attack"
    axes[0, 0].set_ylabel(ylabel + r" / $(\Pi^M-\Pi^C)$")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower left")
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
