"""The priority fee: who gains and who pays as the fee rises.  Profit per period (net of fees) of
the identity entrant that uses the fee, the incumbents, and the fringe, plus the venue's revenue,
against the fee; left: incumbents that see the tape only, right: incumbents that see wallet ids.
Break-even, fringe 2, mean over rho_z and L, 95% seed intervals.
    python analysis/fee_fig.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"entrant": "#8e44ad", "incumbent": "#1baf7a", "fringe": "#52514e", "venue": "#c0392b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--above", type=int, default=0)
    ap.add_argument("--out", default="fig_priority_fee")
    args = ap.parse_args()
    results = Path(args.results)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True, sharey=True)
    for ax, info, title in zip(axes, ("anon", "id"), ("incumbents see the tape only", "incumbents see wallet ids")):
        d = pd.read_csv(results / f"queue_fee_{info}.csv")
        d = d[d.above == args.above]
        # the entrant that uses the fee is id_pay (id at fee 0, where paying does nothing)
        e = d[((d.fee > 0) & (d.policy == "id_pay")) | ((d.fee == 0) & (d.policy == "id"))]
        fees = sorted(e.fee.unique()); x = np.arange(len(fees))
        series = [("entrant", "entrant_pnl", "identity entrant (uses the fee)"), ("incumbent", "incumbent_pnl", "incumbents (no-regret, use the fee)"),
                  ("fringe", "fringe_pnl", "fringe (always rest, never pay)"), ("venue", "venue_rev", "venue revenue")]
        for key, col, label in series:
            # seed-level means over rho_z and L first, then the interval across seeds
            per_seed = e.groupby(["seed", "fee"])[col].mean().reset_index()
            g = per_seed.groupby("fee")[col].agg(["mean", "std", "count"]).reindex(fees)
            ax.errorbar(x, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]), color=C[key], marker="o", ms=3.5, lw=2,
                        ls="--" if key == "venue" else "-", capsize=2, label=label)
        ax.set_xticks(x); ax.set_xticklabels([f"{f:g}" for f in fees])
        ax.set_xlabel("priority fee $f$ (per period of priority)")
        ax.set_title(title, loc="left", fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("profit / period (net of fees)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
