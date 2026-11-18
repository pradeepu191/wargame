"""The ordinary trader (experiments/retail_trader.py), one figure with three panels.

  left    what the market's structure leaves an ordinary customer who has to trade: gains from
          trade as a share of what he keeps at the Nash quote (retail_prices.csv)
  middle  the speculator's quitting lines, Lecture 15 slide 35 style: after n trades, the number
          of wrong calls at which he stops, for three degrees of patience (retail_quitting_lines.csv)
  right   what each rule does to someone with no edge and to someone with a real one: expected
          profit over a year, in average price moves (retail_speculator.csv)

    python analysis/retail_fig.py            # -> results/fig_retail.{png,pdf}, fig_retail_customer.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, validated (light surface)
MARKERS = ["o", "s", "D"]


def panel_prices(ax, f: Path):
    if not f.exists():
        ax.text(0.5, 0.5, "needs local trained cartels", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    a = pd.read_csv(f)
    order = ["Nash quote", "one-tick undercut", "myopic entrant", "cartel alone", "exact best response"]
    labels = ["Nash quote", "entrant: one-tick\nundercut rule", "entrant: myopic", "cartel alone", "entrant: exact\nbest response"]
    tab = a.pivot_table(index="structure", columns="alpha", values="customer_gain_share").reindex(order)
    y = np.arange(len(order))[::-1]
    for j, alpha in enumerate(tab.columns):
        ax.plot(tab[alpha].values, y + (1 - j) * 0.13, ls="none", marker=MARKERS[j], ms=7, color=SERIES[j], mec=SURFACE, mew=1.5,
                label=rf"$\bar\alpha$ = {alpha:g}")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("gains from trade kept, share of the Nash quote's")
    ax.set_title("A customer who has to trade", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def panel_quitting(ax, f: Path, n_show: int = 40):
    q = pd.read_csv(f)
    for j, (name, g) in enumerate(q.groupby("delta", sort=True)):
        g = g[g.n < n_show]
        ax.step(g.n, g.losses_to_stop, where="post", color=SERIES[j], lw=2)
        last = g.iloc[-1]
        ax.annotate({0.0: "myopic", 0.9: r"$\delta$ = 0.9", 0.99: r"$\delta$ = 0.99"}[name], (last.n, last.losses_to_stop),
                    xytext=(4, {0.0: -7, 0.9: 4, 0.99: 0}[name]), textcoords="offset points", va="center", fontsize=7, color=INK)
    n = np.arange(n_show)
    ax.plot(n, n / 2, ls="--", color=MUTED, lw=1)
    ax.annotate("wrong calls if he has no edge, n/2", (n_show * 0.55, n_show * 0.55 / 2), xytext=(0, -12),
                textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, n_show + 6)
    ax.set_xlabel("trades made, n")
    ax.set_ylabel("wrong calls at which he stops for good")
    ax.set_title("Quitting lines: optimistic beginner, one-tick book", loc="left", fontsize=9)


def panel_outcomes(ax, f: Path):
    b = pd.read_csv(f)
    b = b[(b.half_spread_ticks == 1) & b.prior.isin(["-", "overconfident Beta(6,4)"])]
    pols = ["gambler (never updates)", "myopic Bayesian", "patient Bayesian (delta 0.9)", "patient Bayesian (delta 0.99)"]
    labels = ["gambler\n(never updates)", "myopic\nBayesian", r"patient" "\n" r"$\delta$ = 0.9", r"patient" "\n" r"$\delta$ = 0.99"]
    x = np.arange(len(pols))
    w = 0.36
    for j, (p, lab) in enumerate(((0.5, "no edge (p = 0.5)"), (0.6, "real edge (p = 0.6)"))):
        v = b[np.isclose(b.p_true, p)].set_index("policy").reindex(pols)
        bars = ax.bar(x + (j - 0.5) * (w + 0.02), v.exp_profit_mu, width=w, color=SERIES[j], label=lab)
        for xi, (val, ruin) in enumerate(zip(v.exp_profit_mu, v.p_ruin)):
            if p == 0.5:
                ax.annotate(f"ruin {ruin:.0%}" if ruin >= 0.1 else f"ruin {ruin:.1%}", (xi + (j - 0.5) * (w + 0.02), val), xytext=(0, -9), textcoords="offset points",
                            ha="center", fontsize=6.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("expected profit over a year, average price moves")
    ax.set_title("A speculator who may or may not have an edge", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    r = Path(args.results)
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1, 1.15]})
    panel_prices(axes[0], r / "retail_prices.csv")
    panel_quitting(axes[1], r / "retail_quitting_lines.csv")
    panel_outcomes(axes[2], r / "retail_speculator.csv")
    for ax in axes:
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.text(0.01, -0.04, "Stylised model, not investment advice. Speculator: 250 trading days, one unit per trade, savings of 20 "
             "average price moves; ruin = losing all of them. Left: 30 two-maker cartels, model paths.", fontsize=7, color=MUTED)
    out = r / "fig_retail"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("figure:", out.with_suffix(".png"))
    # the left panel alone, large enough for a slide
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    panel_prices(ax, r / "retail_prices.csv")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    out = r / "fig_retail_customer"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
