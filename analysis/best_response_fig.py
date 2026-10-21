"""The exact best response.  Left: the attacker's profit--damage frontier per cartel (x = what the
cartel loses, y = what the attacker earns, rent units) for lambda from 1 (own profit) to 0
(damage), with the hand-crafted undercut and markout rules as hollow markers; the diagonal is
kappa_A = 1 (the attacker keeps everything it destroys).  Right: the double oracle -- the
cartel's rent alone after each round of re-learning against the best response, what the stale
best response still takes, and what the re-solved one takes; mean and 95% interval over seeds.
    python analysis/best_response_fig.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {0.1: "#1baf7a", 0.3: "#8e44ad", 0.5: "#c0392b"}


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="fig_best_response")
    args = ap.parse_args()
    res = Path(args.results)
    d = pd.read_csv(res / "best_response.csv")
    r = pd.read_csv(res / "best_response_relearn.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    # ---- frontier
    for a in sorted(d.alpha.unique()):
        g = d[np.isclose(d.alpha, a)]
        dp = g[g.attacker.str.startswith("dp:")]
        for run, h in dp.groupby("run"):
            h = h.sort_values("lambda", ascending=False)
            ax.plot(h.L_BR_norm, h.entrant_pnl_norm, color=COL[a], lw=0.6, alpha=0.35)
        top = dp[dp["lambda"] == 1.0]
        ax.scatter(top.L_BR_norm, top.entrant_pnl_norm, s=14, color=COL[a], zorder=3, label=rf"exact best response, $\bar\alpha$ = {a}")
        for name, mk in (("undercut", "s"), ("markout", "^")):
            h = g[g.attacker == name]
            ax.scatter(h.L_BR_norm, h.entrant_pnl_norm, s=18, facecolors="none", edgecolors=COL[a], marker=mk, lw=0.8, zorder=2,
                       label=f"{name} rule" if a == 0.3 else None)
    lim = max(d.L_BR_norm.max(), 1.3)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.6, ls="--")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"what the cartel loses, $L_{BR}\,/\,(\Pi^M-\Pi^C)$")
    ax.set_ylabel(r"what the attacker earns $/\,(\Pi^M-\Pi^C)$")
    ax.set_title(r"the attacker's frontier: $\lambda$ from 1 (own profit) to 0 (damage)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    # ---- double oracle
    rounds = sorted(r["round"].unique())
    for a in sorted(r.alpha.unique()):
        g = r[np.isclose(r.alpha, a)]
        C = [g[(g["round"] == k) & (g.attacker.isin(["dp:1", "dp:1@fresh"]))].groupby("run").C.first() for k in rounds]
        Lf = [g[(g["round"] == k) & (g.attacker.isin(["dp:1", "dp:1@fresh"]))].L_BR_norm for k in rounds]
        Ls = [g[(g["round"] == k) & (g.attacker == "dp:1@stale")].L_BR_norm for k in rounds[1:]]
        bx.errorbar(rounds, [c.mean() for c in C], yerr=[ci(c) for c in C], color=COL[a], marker="o", ms=4, lw=1.4, capsize=2,
                    label=rf"cartel's rent alone $C$, $\bar\alpha$ = {a}")
        bx.errorbar(rounds, [l.mean() for l in Lf], yerr=[ci(l) for l in Lf], color=COL[a], marker="v", ms=4, lw=1.0, ls=":", capsize=2,
                    label="re-solved best response takes" if a == 0.3 else None)
        bx.errorbar(rounds[1:], [l.mean() for l in Ls], yerr=[ci(l) for l in Ls], color=COL[a], marker="x", ms=5, lw=0.8, ls="--", capsize=2,
                    label="stale best response still takes" if a == 0.3 else None)
    bx.set_xticks(rounds); bx.set_xlabel("double-oracle round (cartel re-learns 3000 episodes; entrant re-solves)")
    bx.set_ylim(bottom=0.3)
    bx.set_ylabel(r"rent units $/\,(\Pi^M-\Pi^C)$")
    bx.set_title("re-learning does not defend the rent", loc="left", fontsize=9)
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)
    bx.grid(color=GRID, lw=0.8); bx.set_axisbelow(True)
    out = res / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
