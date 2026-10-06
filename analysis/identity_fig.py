"""Figures for the identity experiment (experiments/identity.py).

fig_<run>_identity.png      entrant profit (rent units) vs arrival persistence rho, one line per
                            information set, rows = type dispersion kappa, columns = mean toxicity
fig_<run>_identity_as.png   the adverse-selection shift: informed share of the entrant's fills vs
                            the incumbents' fills, id entrant, by rho and kappa
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"competitive": "#2a78d6", "markout": "#eb6834", "anon": "#1baf7a", "id": "#8e44ad", "oracle": "#0b0b0b"}
LABEL = {"competitive": "competitive-GM", "markout": "own-fill markout", "anon": r"tape, no ids ($\mathcal{F}^{anon}$)",
         "id": r"tape + wallet ids ($\mathcal{F}^{id}$)", "oracle": r"true types ($\mathcal{F}^{oracle}$)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_exp2")
    args = ap.parse_args()
    results = Path(args.results)
    df = pd.read_csv(results / f"{args.run}_identity.csv")
    alphas = sorted(df.alpha.unique())
    kappas = [k for k in sorted(df.kappa.unique(), reverse=True) if k < 1e5]   # skip the homogeneous check
    rhos = sorted(df.rho.unique())
    x = np.arange(len(rhos))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})

    fig, axes = plt.subplots(len(kappas), len(alphas), figsize=(4.2 * len(alphas), 3.2 * len(kappas)),
                             constrained_layout=True, sharex=True, squeeze=False)
    for r, kappa in enumerate(kappas):
        for c, a in enumerate(alphas):
            ax = axes[r, c]
            for p in C:
                d = df[(df.alpha == a) & (df.kappa == kappa) & (df.policy == p)]
                g = d.groupby("rho").entrant_pnl_norm.agg(["mean", "std", "count"]).reindex(rhos)
                se = g["std"] / np.sqrt(g["count"])
                ax.errorbar(x, g["mean"], yerr=1.96 * se, color=C[p], lw=2 if p in ("anon", "id", "oracle") else 1.2,
                            ls="-" if p in ("anon", "id", "oracle") else "--", marker="o", ms=3.5, capsize=2, label=LABEL[p])
            sd = df[(df.alpha == a) & (df.kappa == kappa)].groupby("seed").alpha_eff.first()
            ax.set_title(rf"$\bar\alpha$ = {a}, type sd $\approx$ {_type_sd(a, kappa):.2f} ($\kappa$ = {kappa:g})", loc="left", fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels([f"{r_:g}" for r_ in rhos])
            ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
            if r == len(kappas) - 1:
                ax.set_xlabel(r"arrival persistence $\rho$")
            if c == 0:
                ax.set_ylabel(r"entrant profit / $(\Pi^M-\Pi^C)$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    out = results / f"fig_{args.run}_identity"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))

    # value of identity: paired (same seed, same market) differences id - anon and oracle - id
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.2), constrained_layout=True, squeeze=False)
    for c, a in enumerate(alphas):
        ax = axes[0, c]
        for kappa, ls in zip(kappas, ("-", "--", ":")):
            d = df[(df.alpha == a) & (df.kappa == kappa)].pivot_table(index=["rho", "seed"], columns="policy", values="entrant_pnl_norm")
            for key, col, lab in (("id-anon", C["id"], r"$\pi^{id}-\pi^{anon}$"), ("oracle-id", C["oracle"], r"$\pi^{oracle}-\pi^{id}$")):
                diff = (d["id"] - d["anon"]) if key == "id-anon" else (d["oracle"] - d["id"])
                g = diff.groupby("rho").agg(["mean", "std", "count"]).reindex(rhos)
                ax.errorbar(x, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]), color=col, ls=ls, marker="o", ms=3.5,
                            lw=1.8, capsize=2, label=rf"{lab}, $\kappa$ = {kappa:g}")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(rf"$\bar\alpha$ = {a}", loc="left", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"{r_:g}" for r_ in rhos]); ax.set_xlabel(r"arrival persistence $\rho$")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel(r"value of information / $(\Pi^M-\Pi^C)$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    out = results / f"fig_{args.run}_identity_value"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))

    # adverse-selection shift
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.2), constrained_layout=True, squeeze=False)
    for c, a in enumerate(alphas):
        ax = axes[0, c]
        for kappa, ls in zip(kappas, ("-", "--", ":")):
            d = df[(df.alpha == a) & (df.kappa == kappa) & (df.policy == "id")]
            g = d.groupby("rho")[["entrant_informed_share", "incumbent_informed_share", "informed_share_nominal"]].mean().reindex(rhos)
            ax.plot(x, g.entrant_informed_share, color=C["id"], ls=ls, marker="o", ms=3.5, lw=1.8,
                    label=rf"entrant ($\mathcal{{F}}^{{id}}$), $\kappa$ = {kappa:g}")
            ax.plot(x, g.incumbent_informed_share, color="#eb6834", ls=ls, marker="s", ms=3.5, lw=1.8,
                    label=rf"incumbents, $\kappa$ = {kappa:g}")
        nom = df[(df.alpha == a) & (df.policy == "id")].informed_share_nominal.mean()
        ax.axhline(nom, color=MUTED, lw=1, ls="-."); ax.text(x[-1], nom, " no entrant", color=MUTED, fontsize=7, va="center")
        ax.set_title(rf"$\bar\alpha$ = {a}", loc="left", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"{r_:g}" for r_ in rhos]); ax.set_xlabel(r"arrival persistence $\rho$")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel("informed share of own fills")
    axes[0, 0].legend(frameon=False, fontsize=7)
    out = results / f"fig_{args.run}_identity_as"
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


def _type_sd(a, kappa):
    return float(np.sqrt(a * (1 - a) / (kappa + 1)))


if __name__ == "__main__":
    main()
