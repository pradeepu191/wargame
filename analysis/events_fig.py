"""The Binance hack, hour by hour, on ETH (the drained asset) and BTC.  Columns: the shock
(volatility and fills), the identity gap inside each hour (the top makers' share of benign vs
toxic wallets' fills), and who held the flow (the fringe wallet's share of the top-10 fills and
the top-10 markout).
    python analysis/events_fig.py [--results results/live_2026_10] [--event "Binance hot"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--event", default="Binance hot")
    ap.add_argument("--coins", default="ETH,BTC")
    ap.add_argument("--window", default="-8,12")
    ap.add_argument("--out", default="fig_event_hack")
    args = ap.parse_args()
    res = Path(args.results)
    e = pd.read_csv(res / "events.csv")
    e = e[e.event.str.startswith(args.event)]
    lo, hi = (int(x) for x in args.window.split(","))
    coins = args.coins.split(",")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(coins), 3, figsize=(12.5, 3.3 * len(coins)), constrained_layout=True, sharex=True)
    axes = np.atleast_2d(axes)
    for row, coin in enumerate(coins):
        g = e[(e.coin == coin) & e.k_hours.between(lo, hi)].sort_values("k_hours")
        ax, bx, cx = axes[row]
        ax.bar(g.k_hours, g.vol_bps, color=GREY, width=0.8, label="mid's 10 s move per fill (bps)")
        ax2 = ax.twinx(); ax2.plot(g.k_hours, g.toxic_fill_share, color=RED, marker="o", ms=3, lw=1.2, label="share of fills from toxic wallets")
        ax2.set_ylim(0, 0.6); ax2.tick_params(colors=RED); ax2.spines["right"].set_visible(True)
        ax.set_ylabel("volatility (bps)"); ax2.set_ylabel("toxic wallets' fill share", color=RED)
        ax.set_title(f"{coin}: the shock", fontsize=9.5, loc="left")
        if row == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5, frameon=False)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        bx.plot(g.k_hours, g.benign_top_share, color=BLUE, marker="o", ms=3, lw=1.3, label="of benign wallets' fills")
        bx.plot(g.k_hours, g.toxic_top_share, color=RED, marker="o", ms=3, lw=1.3, label="of toxic wallets' fills")
        bx.fill_between(g.k_hours, g.toxic_top_share, g.benign_top_share, where=g.benign_top_share >= g.toxic_top_share, color=AQUA, alpha=0.15)
        bx.fill_between(g.k_hours, g.toxic_top_share, g.benign_top_share, where=g.benign_top_share < g.toxic_top_share, color=RED, alpha=0.2)
        bx.set_ylabel("top-10 makers' share"); bx.set_ylim(0, 0.55)
        bx.set_title(f"{coin}: the identity gap inside each hour", fontsize=9.5, loc="left")
        if row == 0:
            bx.legend(loc="upper right", fontsize=7.5, frameon=False)
        cx.bar(g.k_hours, g.fringe_share, color=ORANGE, width=0.8, label="fringe wallet's share of the top-10 fills")
        cx2 = cx.twinx(); cx2.plot(g.k_hours, g.top_markout_bps, color=INK, marker="o", ms=3, lw=1.2, label="top-10 markout (bps)")
        cx2.axhline(0, color=MUTED, lw=0.6); cx2.spines["right"].set_visible(True)
        cx.set_ylabel("fringe wallet's share of top-10 fills"); cx2.set_ylabel("top-10 markout, bps"); cx.set_ylim(0, 0.5)
        cx.set_title(f"{coin}: who held it", fontsize=9.5, loc="left")
        if row == 0:
            h1, l1 = cx.get_legend_handles_labels(); h2, l2 = cx2.get_legend_handles_labels()
            cx2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7.5, frameon=False)
        for a in (ax, bx, cx):
            a.axvline(-0.5, color=RED, lw=1, ls="--")
            a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for a in axes[-1]:
        a.set_xlabel("hours from the drain (Oct 29, 20:24 UTC)")
    fig.suptitle("The Binance hack on Hyperliquid: on ETH, the asset being dumped, the identity gap fails for five hours and one wallet absorbs the flow; on BTC it holds",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    out = res / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
