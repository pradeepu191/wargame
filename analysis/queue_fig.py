"""Figure for the one-tick queue game: entrant profit per period by information set, against the
mark lag, for the two spread settings (break-even, one tick above) and two regime persistences."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"always": "#52514e", "anon": "#1baf7a", "id": "#8e44ad", "oracle": "#0b0b0b"}
LABEL = {"always": "always rest (round-robin)", "anon": r"selector, no ids ($\mathcal{F}^{anon}$)",
         "id": r"selector, wallet ids ($\mathcal{F}^{id}$)", "oracle": r"true types ($\mathcal{F}^{oracle}$)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", default="queue_always.csv")
    args = ap.parse_args()
    results = Path(args.results)
    df = pd.read_csv(results / args.csv)
    aboves = sorted(df.above.unique()); rhos = sorted(df.rho_z.unique()); lags = sorted(df.mark_lag.unique())
    x = np.arange(len(lags))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(aboves), len(rhos), figsize=(4.2 * len(rhos), 3.1 * len(aboves)),
                             constrained_layout=True, sharex=True, squeeze=False)
    for r, ab in enumerate(aboves):
        for c, rz in enumerate(rhos):
            ax = axes[r, c]
            for p in C:
                d = df[(df.above == ab) & (df.rho_z == rz) & (df.policy == p)]
                g = d.groupby("mark_lag").entrant_pnl.agg(["mean", "std", "count"]).reindex(lags)
                ax.errorbar(x, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]), color=C[p], marker="o", ms=3.5,
                            lw=2 if p != "always" else 1.2, ls="-" if p != "always" else "--", capsize=2, label=LABEL[p])
            h = int(d.h.iloc[0])
            ax.set_title(rf"spread = $h^C$ {'+ 1' if ab else ''} ({h} ticks), $\rho_z$ = {rz}", loc="left", fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels([str(L) for L in lags])
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
            if r == len(aboves) - 1:
                ax.set_xlabel("mark lag $L$ (arrivals until a print's mark is public)")
            if c == 0:
                ax.set_ylabel("entrant profit / period")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    out = results / "fig_queue_always"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
