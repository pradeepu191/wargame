"""Anonymity as a venue lever, from the defenders' side: liquidity in the toxic state when the
incumbents can (or cannot) see wallet identities.  Left: probability that an uninformed taker is
filled in the toxic state, against the mark lag, no fringe (whoever is at the inside is a selector).
Right: what an always-resting, identity-blind maker earns among those incumbents.  Lines: incumbents'
information (tape only vs wallet ids); solid rho_z = 0.9, dashed 0.99; 95% seed intervals.
    python analysis/liquidity_fig.py
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"anon": "#1baf7a", "id": "#8e44ad"}
LABEL = {"anon": r"incumbents see the tape only ($\mathcal{F}^{anon}$)", "id": r"incumbents see wallet ids ($\mathcal{F}^{id}$)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--above", type=int, default=0)
    ap.add_argument("--out", default="fig_liquidity_identity")
    args = ap.parse_args()
    results = Path(args.results)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(results / "queue_hedge_always_never_anon*_fringe*.csv")))])
    d["set"] = np.where(d.experts.str.contains("id"), "id", "anon")
    d = d[(d.above == args.above) & (d.fringe == 0)]
    lags = sorted(d.mark_lag.unique()); x = np.arange(len(lags)); rhos = sorted(d.rho_z.unique())
    STYLE = {rhos[0]: "-", rhos[-1]: "--"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    panels = [("uninformed_fill_toxic", "id", "P(uninformed taker is filled | toxic state)", "entrant policy: identity selector"),
              ("entrant_pnl", "always", "always-resting maker's profit / period", "entrant policy: always rest")]
    for ax, (col, pol, ylabel, sub) in zip(axes, panels):
        for s in ("anon", "id"):
            for rz in rhos:
                g = d[(d.set == s) & (d.rho_z == rz) & (d.policy == pol)].groupby("mark_lag")[col].agg(["mean", "std", "count"]).reindex(lags)
                ax.errorbar(x + (0.06 if rz == rhos[-1] else -0.06), g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]),
                            color=C[s], ls=STYLE[rz], marker="o", ms=3.5, lw=2, capsize=2,
                            label=f"{LABEL[s]}, " + rf"$\rho_z$ = {rz}")
        ax.set_xticks(x); ax.set_xticklabels([str(L) for L in lags])
        ax.set_xlabel("mark lag $L$ (arrivals until a print's mark is public)")
        ax.set_ylabel(ylabel); ax.set_title(sub, loc="left", fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylim(0, 0.9)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
