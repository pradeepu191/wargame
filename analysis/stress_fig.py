"""Who is at the touch after a stressful spell.  One panel per coin: each point is a 5-minute block,
x = the mid's absolute 10 s move per fill in the PREVIOUS block (bps, log scale), y = the top-10
market makers' share of fills at the touch in this block.  Tercile means with 95% intervals over
blocks are drawn on top; the title carries the Spearman rank correlation.
    python analysis/stress_fig.py --results results/live_2026_10
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INK, MUTED, GRID, ACCENT = "#0b0b0b", "#52514e", "#e6e5e1", "#c0392b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--coins", default="BTC,ETH,SOL,HYPE")
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--out", default="fig_stress_withdrawal")
    args = ap.parse_args()
    results = Path(args.results)
    coins = [c for c in args.coins.split(",") if (results / f"stress_{c}.csv").exists()]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(coins), figsize=(3.4 * len(coins), 3.3), constrained_layout=True, squeeze=False, sharey=True)
    for ax, c in zip(axes[0], coins):
        b = pd.read_csv(results / f"stress_{c}.csv")
        b = b[(b.n_touch >= args.n_min)].dropna(subset=["vol_lag"])
        ax.scatter(b.vol_lag, b.top_share, s=9, color=MUTED, alpha=0.45, lw=0)
        q = b.vol_lag.quantile([1 / 3, 2 / 3]).to_numpy()
        terc = np.digitize(b.vol_lag, q)
        xs, ys, es = [], [], []
        for k in range(3):
            g = b[terc == k]
            xs.append(g.vol_lag.median()); ys.append(g.top_share.mean()); es.append(1.96 * g.top_share.std() / np.sqrt(max(len(g), 1)))
        ax.errorbar(xs, ys, yerr=es, color=ACCENT, marker="o", ms=5, capsize=3, lw=1.4, zorder=3)
        r, p = spearmanr(b.vol_lag, b.top_share)
        ax.set_title(f"{c}: rank corr {r:+.2f} (p = {p:.3f}, {len(b)} blocks)", loc="left", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("|10 s mid move| per fill, previous 5 min (bps)")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("top-10 MM share of fills at the touch")
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
