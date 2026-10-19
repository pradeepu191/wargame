"""Per-maker realized spread on marked fills: each dot is a maker wallet with >= 100 marked fills,
its 10-s true-mid markout with a 5-minute-cluster SE, against the share of its fills that were at
the pre-trade touch (1 = rests only at the inside; 0 = rests behind it and gets swept).  Colour:
top-10 MM wallets by full-window volume / other classified MMs / makers the classifier does not
call MMs.  Horizontal line: the mean over all marked fills.
    python analysis/makers_fig.py --results results/live_2026_10
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"top": "#8e44ad", "mm": "#1baf7a", "non": "#52514e"}
LABEL = {"top": "top-10 MM wallets (by volume, full window)", "mm": "other MM wallets", "non": "not classified as MM"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--coins", default="BTC,ETH,HYPE,SOL")
    ap.add_argument("--out", default="results/fig_makers_live_2026_10")
    args = ap.parse_args()
    results = Path(args.results)
    coins = args.coins.split(",")
    cal = pd.read_csv(results / "calibration.csv").set_index("coin")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(coins), figsize=(3.6 * len(coins), 3.4), constrained_layout=True, squeeze=False)
    for ax, c in zip(axes[0], coins):
        m = pd.read_csv(results / f"makers_{c}.csv"); w = pd.read_csv(results / f"wallets_{c}.csv")
        top = set(w[w.is_mm].sort_values("volume", ascending=False).wallet.head(10)); mm = set(w[w.is_mm].wallet)
        m["cls"] = ["top" if x in top else ("mm" if x in mm else "non") for x in m.wallet]
        for cls in ("non", "mm", "top"):
            d = m[m.cls == cls]
            ax.errorbar(d.touch_share, d.markout_bps, yerr=1.96 * d.se_bps, fmt="none", ecolor=C[cls], alpha=0.35, lw=0.8)
            ax.scatter(d.touch_share, d.markout_bps, s=8 + 40 * np.sqrt(d.n_fills / 1000), color=C[cls],
                       alpha=0.9 if cls == "top" else 0.6, edgecolor="white", lw=0.4, label=LABEL[cls], zorder=3 if cls == "top" else 2)
        ax.axhline(float(cal.loc[c, "mk_all_markout_bps"]), color=INK, lw=1, ls="--")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{c}: {int(cal.loc[c, 'mk_n_makers'])} makers, top-10 {cal.loc[c, 'mk_top_markout_bps']:+.2f} vs rest at the touch "
                     f"{cal.loc[c, 'mk_rest_touch_markout_bps']:+.2f} bp", loc="left", fontsize=8)
        ax.set_xlabel("share of the maker's fills at the pre-trade touch")
        ax.set_xlim(-0.03, 1.03)
        lo, hi = np.nanpercentile(m.markout_bps, [2, 98])
        ax.set_ylim(min(lo, -1.5) - 0.3, max(hi, 1.0) + 0.3)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("10-s markout vs true mid (bp), mean ± 95% cluster CI")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
