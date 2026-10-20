"""Identity selection on the book.  One panel per coin: the top-10 market makers' share of fills
at the touch, by the taker's type as classified on the FIRST night (benign / unseen / toxic), on
the second night; the second bar in each pair keeps only fills after which the mid did not move
within one second (not pick-offs), so a gap that survives there is the wallet being avoided, not
a stale quote being hit.  The dashed line is the top makers' share of all touch fills.
    python analysis/identity_selection_fig.py --results results/live_2026_10
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_ALL, C_QUIET = "#8e44ad", "#d7bde2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/live_2026_10")
    ap.add_argument("--out", default="fig_identity_selection")
    args = ap.parse_args()
    results = Path(args.results)
    f = pd.read_csv(results / "stress_fill.csv").set_index("coin")
    coins = [c for c in ("BTC", "ETH", "SOL", "HYPE") if c in f.index]
    types = [("benign", "benign\nlast night"), ("new", "unseen"), ("toxic", "toxic\nlast night")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(coins), figsize=(3.1 * len(coins), 3.2), constrained_layout=True, squeeze=False, sharey=True)
    x = np.arange(len(types)); w = 0.38
    for ax, c in zip(axes[0], coins):
        r = f.loc[c]
        allv = [r[f"id_{t}_top_share"] for t, _ in types]
        quiet = [r[f"id_{t}_quiet_top_share"] for t, _ in types]
        ax.bar(x - w / 2, allv, w, color=C_ALL, label="all touch fills")
        ax.bar(x + w / 2, quiet, w, color=C_QUIET, label="mid unmoved 1 s later")
        overall = sum(r[f"id_{t}_top_share"] * r[f"id_{t}_fill_share"] for t in ("benign", "middle", "toxic", "new"))
        ax.axhline(overall, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in types])
        n_tox = int(r["id_toxic_fill_share"] * r["id_n_fills_later"])
        ax.set_title(f"{c}: {int(r['id_n_typed_takers'])} takers typed, toxic = {r['id_toxic_fill_share']:.0%} of fills",
                     loc="left", fontsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("top-10 MM share of fills at the touch")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    out = results / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
