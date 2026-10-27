"""The public-information attacker vs the oracle.  Left: what the attacker earns after probing,
by probing budget, against the exact best response from the true tables (dashed), per toxicity.
Right: coverage of the exact best response's path by the identified reaction table.
    python analysis/public_attacker_fig.py [--results results]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9890"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {0.1: AQUA, 0.3: BLUE, 0.5: ORANGE}


def ci(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="fig_public_attacker")
    args = ap.parse_args()
    res = Path(args.results)
    d = pd.read_csv(res / "public_attacker.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    probes = sorted(d.probe.unique())
    x = np.arange(len(probes))
    w = 0.26
    alphas = sorted(d.alpha.unique())
    for k, a in enumerate(alphas):
        g = d[np.isclose(d.alpha, a)]
        m = [g[g.probe == p].entrant_norm_post.mean() for p in probes]
        e = [ci(g[g.probe == p].entrant_norm_post) for p in probes]
        ax.bar(x + (k - 1) * w, m, w, color=COL[a], yerr=e, error_kw=dict(ecolor=INK, lw=0.8, capsize=2),
               label=rf"$\bar\alpha$ = {a} ({g.run.nunique()} cartels)")
        ex = g.exact_entrant_norm.mean()
        ax.hlines(ex, x[0] - 0.5, x[-1] + 0.5, color=COL[a], lw=1, ls="--")
        ax.annotate(f"oracle {ex:.2f}", (x[-1] + 0.5, ex), xytext=(2, 0), textcoords="offset points", fontsize=7.5, color=COL[a], va="center")
        c = [g[g.probe == p].coverage.mean() for p in probes]
        bx.plot(x, c, marker="o", ms=4, color=COL[a], lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"{p}" for p in probes])
    ax.set_xlabel("probing budget (periods of random quoting; one episode = 100 periods)")
    ax.set_ylabel("entrant's profit after probing, rent units")
    ax.set_title("Identify the reaction function from the tape, then solve it:\nthe public-information attacker reaches the oracle best response (dashed)", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, lw=0.8, axis="y"); ax.set_axisbelow(True)
    bx.set_xticks(x); bx.set_xticklabels([f"{p}" for p in probes])
    bx.set_ylim(0, 1.05)
    bx.set_xlabel("probing budget (periods)")
    bx.set_ylabel("share of the oracle path's cells identified")
    bx.set_title("Coverage of the exact best response's path", fontsize=9.5, loc="left")
    bx.grid(color=GRID, lw=0.8); bx.set_axisbelow(True)
    out = res / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("figure:", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
