"""Figure for the entry experiment: entrant PnL by policy and toxicity, against exact benchmarks."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C = {"competitive": "#2a78d6", "markout": "#eb6834", "undercut": "#1baf7a"}   # validated palette, fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABEL = {"competitive": "competitive-GM entrant", "markout": "markout-inference entrant",
         "undercut": "one-tick undercutter"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    args = ap.parse_args()
    results = Path(args.results)
    e = pd.read_csv(results / f"{args.run}_entry.csv")
    g = e.groupby(["alpha", "policy"]).agg(pnl=("entrant_pnl", "mean"), std=("entrant_pnl", "std"),
                                            share=("entrant_fill_share", "mean"),
                                            inc_before=("incumbent_pnl_before", "mean"),
                                            inc_after=("incumbent_pnl_after", "mean"),
                                            piC=("piC", "first"), piM=("piM", "first")).reset_index()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    ref = g[g.policy == "competitive"].sort_values("alpha")

    ax = axes[0]
    ax.plot(ref.alpha, ref.piM, color=MUTED, lw=1.2, ls="--")
    ax.text(ref.alpha.iloc[-1], ref.piM.iloc[-1], " monopoly $\\pi^M$", color=MUTED, fontsize=7, va="center")
    ax.plot(ref.alpha, ref.piC, color=MUTED, lw=1.2, ls=":")
    ax.text(ref.alpha.iloc[-1], ref.piC.iloc[-1], " competitive $\\pi^C$", color=MUTED, fontsize=7, va="center")
    for p in ("competitive", "markout", "undercut"):
        s = g[g.policy == p].sort_values("alpha")
        ax.errorbar(s.alpha, s.pnl, yerr=s["std"], color=C[p], lw=2, marker="o", ms=5, capsize=3, label=LABEL[p])
    ax.set_xlabel(r"toxicity $\alpha$"); ax.set_ylabel("entrant profit per period")
    ax.set_title("(a) Entrant profit against frozen incumbents", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for p in ("competitive", "markout", "undercut"):
        s = g[g.policy == p].sort_values("alpha")
        ax.plot(s.alpha, s.share, color=C[p], lw=2, marker="o", ms=5, label=LABEL[p])
    ax.set_ylim(0, 1.05); ax.set_xlabel(r"toxicity $\alpha$"); ax.set_ylabel("share of fills won by entrant")
    ax.set_title("(b) Winning every fill is not the goal", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.plot(ref.alpha, ref.inc_before, color=MUTED, lw=2, marker="o", ms=5, label="incumbents alone")
    for p in ("competitive", "markout", "undercut"):
        s = g[g.policy == p].sort_values("alpha")
        ax.plot(s.alpha, s.inc_after, color=C[p], lw=2, marker="o", ms=5, label=f"after {LABEL[p].split()[0]} entry")
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlabel(r"toxicity $\alpha$"); ax.set_ylabel("incumbent profit per period (mean)")
    ax.set_title("(c) What entry does to the incumbents", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7)
    out = results / f"fig_{args.run}_entry"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
