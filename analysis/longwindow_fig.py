"""The identity-selection gap on a continuous recording.  Left: the gap (top makers' share of
benign takers' fills minus their share of toxic takers' fills) by calendar day, with 95%
intervals, wallets typed before the eight-day hole and scored after it.  Middle: rolling typing,
type on day d and score on day d + k, averaged over d.  Right: the rank correlation of makers'
daily markouts with the previous day's.
    python analysis/longwindow_fig.py [--results results/live_2026_10]
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9890"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"BTC": BLUE, "ETH": ORANGE, "SOL": AQUA, "HYPE": RED}


def day_label(d):
    return (date(1970, 1, 1) + timedelta(days=int(d))).strftime("%b %d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--out", default="fig_longwindow")
    args = ap.parse_args()
    res = Path(args.results)
    d = pd.read_csv(res / "longwindow_day.csv")
    r = pd.read_csv(res / "longwindow_rolling.csv")
    c = pd.read_csv(res / "longwindow_makers_corr.csv")
    d = d[d.n_benign + d.n_toxic >= 2000]                      # drop the partial last day
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for coin, g in d.groupby("coin"):
        g = g.sort_values("day")
        ax.errorbar(g.day, g.gap, yerr=1.96 * g.gap_se, color=COL[coin], marker="o", ms=3.5, lw=1.3, capsize=2, label=coin)
    ax.axhline(0, color=MUTED, lw=0.6)
    days = sorted(d.day.unique())
    ax.set_xticks(days); ax.set_xticklabels([day_label(x) for x in days], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("gap in the top makers' fill share\n(benign takers' fills minus toxic takers' fills)")
    ax.set_title("Every day, out of sample: wallets typed Oct 6 to 12,\nscored Oct 20 to 31 (95% intervals, 5-minute clusters)", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    rr = r.groupby(["coin", "k"]).agg(gap=("gap", "mean"), n=("gap", "size")).reset_index()
    for coin, g in rr.groupby("coin"):
        g = g[g.n >= 2]
        bx.plot(g.k, g.gap, color=COL[coin], marker="o", ms=3.5, lw=1.3, label=coin)
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.set_xlabel("k: days between typing a wallet and scoring it")
    bx.set_ylabel("gap, averaged over typing days")
    bx.set_title("The label does not decay: type on day d,\nscore on day d + k", fontsize=9.5, loc="left")
    bx.set_ylim(bottom=0)
    bx.grid(color=GRID, lw=0.8); bx.set_axisbelow(True)
    for coin, g in c.groupby("coin"):
        g = g.dropna(subset=["rho_prev_day"]).sort_values("day")
        g = g[g.n_common_prev >= 20]
        cx.plot(g.day, g.rho_prev_day, color=COL[coin], marker="o", ms=3.5, lw=1.3, label=coin)
    cx.axhline(0, color=MUTED, lw=0.6)
    cdays = sorted(c.dropna(subset=["rho_prev_day"]).day.unique())
    cx.set_xticks(cdays[::2]); cx.set_xticklabels([day_label(x) for x in cdays[::2]], rotation=45, ha="right", fontsize=8)
    cx.set_ylabel("Spearman rank correlation with the previous day")
    cx.set_title("Who earns persists: makers' daily markouts,\nranked, day against day (>= 100 fills each)", fontsize=9.5, loc="left")
    cx.set_ylim(-0.1, 1.0)
    cx.grid(color=GRID, lw=0.8); cx.set_axisbelow(True)
    out = res / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
