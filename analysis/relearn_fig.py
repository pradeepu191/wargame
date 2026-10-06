"""Figure: entry with re-learning incumbents. Trajectories of entrant profit and market Delta
over the incumbents' re-exploration schedule, for the three entrant policies."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C = {"competitive": "#2a78d6", "markout": "#eb6834", "undercut": "#1baf7a"}   # validated palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABEL = {"competitive": "competitive-GM", "markout": "markout-inference", "undercut": "one-tick undercut"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    args = ap.parse_args()
    results = Path(args.results)
    data = {p: pd.read_csv(results / f"{args.run}_entry_relearn_{p}.csv") for p in C}
    alphas = sorted(data["markout"].alpha.unique())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(alphas), figsize=(11, 5.6), constrained_layout=True, sharex=True)
    for j, a in enumerate(alphas):
        for p, df in data.items():
            d = df[df.alpha == a].groupby("episode")[["entrant_pnl", "delta_market"]].agg(["mean", "std"])
            x = d.index
            for i, key in enumerate(("entrant_pnl", "delta_market")):
                m, s = d[(key, "mean")], d[(key, "std")]
                ax = axes[i, j]
                ax.plot(x, m, color=C[p], lw=2, label=LABEL[p])
                ax.fill_between(x, m - s, m + s, color=C[p], alpha=0.15, lw=0)
        piM = data["markout"][data["markout"].alpha == a].piM.iloc[0]
        piC = data["markout"][data["markout"].alpha == a].piC.iloc[0]
        ax = axes[0, j]
        ax.axhline(piM, color=MUTED, lw=1, ls="--"); ax.axhline(piC, color=MUTED, lw=1, ls=":")
        ax.text(x[-1], piM, r" $\pi^M$", color=MUTED, fontsize=7, va="center")
        ax.text(x[-1], piC, r" $\pi^C$", color=MUTED, fontsize=7, va="center")
        ax.set_title(rf"$\alpha$ = {a}", loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax = axes[1, j]
        ax.axhline(0, color=MUTED, lw=1, ls=":"); ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.set_ylim(-0.3, 1.05); ax.set_xlabel("episodes since entry (incumbents re-learning)")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("entrant profit / period")
    axes[1, 0].set_ylabel(r"market-wide $\Delta$ (3 MMs)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="center right")
    # secondary axis annotation: epsilon schedule
    eps = data["markout"].groupby("episode").epsilon.first()
    for j in range(len(alphas)):
        axt = axes[1, j].twinx()
        axt.plot(eps.index, eps.values, color=MUTED, lw=1, ls="-.")
        axt.set_ylim(0, 0.35); axt.set_yticks([0, 0.1, 0.2, 0.3])
        axt.tick_params(axis="y", colors=MUTED, labelsize=7)
        axt.spines["top"].set_visible(False)
        if j == len(alphas) - 1:
            axt.set_ylabel(r"incumbents' $\epsilon$", color=MUTED, fontsize=8)
    out = results / f"fig_{args.run}_entry_relearn"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
