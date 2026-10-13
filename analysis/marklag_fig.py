"""Regime channel: the value of identity (id - anon) and of the true regime (oracle - anon), in rent
units, against the mark lag L; paired per seed; panels by population toxicity; 95% seed intervals.
    python analysis/marklag_fig.py --csv cfl_exp2_identity_regime_10seed.csv --kappa 1 --suffix _10seed
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"anon": "#1baf7a", "id": "#8e44ad", "oracle": "#0b0b0b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", default="cfl_exp2_identity_regime_10seed.csv")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--suffix", default="_10seed")
    args = ap.parse_args()
    results = Path(args.results)
    df = pd.read_csv(results / args.csv)
    df = df[np.isclose(df.kappa, args.kappa)]
    alphas = [float(a) for a in args.alphas.split(",")]
    rhos = sorted(df.rho_z.unique()); lags = sorted(df.mark_lag.unique())
    x = np.arange(len(lags))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    # paired differences per seed (same incumbents, common random numbers): the level intervals are
    # dominated by which cartel the seed trained, the paired ones by the information set
    piv = df.pivot_table(index=["alpha", "rho_z", "mark_lag", "seed"], columns="policy", values="entrant_pnl_norm")
    piv["id-anon"] = piv["id"] - piv["anon"]; piv["oracle-anon"] = piv["oracle"] - piv["anon"]
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.4 * len(alphas), 3.2), constrained_layout=True, squeeze=False)
    STYLE = {0.9: "-", 0.99: "--"}
    for c, a in enumerate(alphas):
        ax = axes[0, c]
        for rz in rhos:
            for col, color in (("id-anon", C["id"]), ("oracle-anon", C["oracle"])):
                g = piv.xs((a, rz), level=("alpha", "rho_z"))[col].groupby("mark_lag").agg(["mean", "std", "count"]).reindex(lags)
                ax.errorbar(x + (0.06 if rz == rhos[-1] else -0.06), g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]),
                            color=color, ls=STYLE.get(rz, "-"), marker="o", ms=3.5, lw=2, capsize=2,
                            label=rf"{col}, $\rho_z$ = {rz}")
        n = int(df[np.isclose(df.alpha, a)].seed.nunique())
        ax.set_title(rf"$\bar\alpha$ = {a} ($\kappa_\alpha$ = {args.kappa:g}, {n} seeds, paired)", loc="left", fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(L) for L in lags])
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("mark lag $L$ (arrivals until a print's mark is public)")
        if c == 0:
            ax.set_ylabel(r"value of information / $(\Pi^M-\Pi^C)$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    out = results / f"fig_cfl_exp2_marklag{args.suffix}"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
