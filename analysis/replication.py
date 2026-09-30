"""Aggregate the Colliard-Foucault-Lovo replication grid and make the figure.

Reads   results/<run>_alpha<a>_n_mm<N>_seed<k>/{episodes.csv, impulse.csv, config.yaml}
Writes  results/replication_summary.csv      one row per (alpha, N): mean +- std over seeds
        results/replication_impulse.csv      seed-averaged impulse responses per (alpha, N)
        results/fig_replication.png / .pdf   three panels

Usage:  python analysis/replication.py [--run cfl_2mm] [--tail 2000]

"tail" = number of final episodes per run used for the converged statistics
(exploration has decayed to ~0 there).  Report distributions over seeds, never a
single run.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim import BenchmarkTable, GlostenMilgromEnv, MarketConfig

PAT = re.compile(r"_alpha(?P<alpha>[0-9.]+)_n_mm(?P<n>\d+)_seed(?P<seed>\d+)$")


def load_runs(results: Path, run: str, tail: int):
    rows, impulses = [], []
    for d in sorted(results.glob(f"{run}_alpha*_n_mm*_seed*")):
        m = PAT.search(d.name)
        if not m or not (d / "episodes.csv").exists():
            continue
        alpha, n, seed = float(m["alpha"]), int(m["n"]), int(m["seed"])
        ep = pd.read_csv(d / "episodes.csv").tail(tail)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        env = GlostenMilgromEnv(MarketConfig(**cfg["market"]))
        hC, piC, hM, piM = BenchmarkTable(env).lookup(0.5)
        tick = cfg["market"]["tick"]
        rows.append({
            "alpha": alpha, "n_mm": n, "seed": seed,
            "half_spread": ep.mean_quoted_spread.mean() / (2 * tick),
            "profit": ep.mean_profit_per_mm.mean(),
            "delta": ep.delta.mean(),
            "hC": hC, "hM": hM, "piC": piC, "piM": piM,
            **tape_dynamics(d),
        })
        if (d / "impulse.csv").exists():
            ir = pd.read_csv(d / "impulse.csv")
            ir["alpha"], ir["n_mm"], ir["seed"] = alpha, n, seed
            impulses.append(ir)
    return pd.DataFrame(rows), (pd.concat(impulses) if impulses else pd.DataFrame())


def tape_dynamics(d: Path) -> dict:
    """Is the converged greedy joint policy a fixed point or a cycle?  From the saved tapes:
    change_rate = fraction of periods where at least one MM changed its quote;
    h_range     = max - min half-spread over the tape (0 for a fixed point)."""
    tapes = sorted(d.glob("tape_ep*.csv"))
    if not tapes:
        return {}
    t = pd.concat(pd.read_csv(f) for f in tapes)
    hcols = [c for c in t.columns if re.fullmatch(r"h\d+", c)]
    H = t[hcols].to_numpy()
    changed = (np.diff(H, axis=0) != 0).any(axis=1)
    return {"change_rate": float(changed.mean()), "h_range": int(H.max() - H.min()),
            "h_min_mean": float(H.min(axis=1).mean())}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["alpha", "n_mm"])
    out = g.agg(seeds=("seed", "count"),
                half_spread_mean=("half_spread", "mean"), half_spread_std=("half_spread", "std"),
                profit_mean=("profit", "mean"), profit_std=("profit", "std"),
                delta_mean=("delta", "mean"), delta_std=("delta", "std"),
                hC=("hC", "first"), hM=("hM", "first"), piC=("piC", "first"), piM=("piM", "first"),
                change_rate=("change_rate", "mean"), h_range=("h_range", "mean"))
    out["markup_vs_competitive"] = out.half_spread_mean - out.hC
    return out.reset_index()


def impulse_summary(imp: pd.DataFrame) -> pd.DataFrame:
    if imp.empty:
        return imp
    hcols = [c for c in imp.columns if re.fullmatch(r"h\d+", c)]
    imp = imp.copy()
    cols = [c for c in ("deviator", "rivals", "rivals_baseline", "rivals_diff") if c in imp.columns]
    return imp.groupby(["alpha", "n_mm", "rel_t"])[cols].mean().reset_index()


def make_figure(summary: pd.DataFrame, imp: pd.DataFrame, out: Path, imp_alpha=0.3, imp_n=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a"}          # validated categorical palette, fixed order
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    Ns = sorted(summary.n_mm.unique())

    # (a) collusion index vs alpha
    ax = axes[0]
    for n in Ns:
        s = summary[summary.n_mm == n].sort_values("alpha")
        ax.errorbar(s.alpha, s.delta_mean, yerr=s.delta_std, color=C[n], lw=2, marker="o", ms=5,
                    capsize=3, label=f"N = {n}")
        ax.annotate(f"N={n}", (s.alpha.iloc[-1], s.delta_mean.iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", color=INK, va="center", fontsize=8)
    ax.axhline(0, color=MUTED, lw=1, ls=":"); ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.text(summary.alpha.min(), 1.02, "monopoly", color=MUTED, fontsize=7, va="bottom")
    ax.text(summary.alpha.min(), 0.02, "competitive", color=MUTED, fontsize=7, va="bottom")
    ax.set_xlabel(r"toxicity $\alpha$"); ax.set_ylabel(r"collusion index $\Delta$")
    ax.set_title("(a) Learned rent falls with toxicity and N", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)

    # (b) learned half-spread vs benchmarks
    ax = axes[1]
    ref = summary[summary.n_mm == Ns[0]].sort_values("alpha")
    ax.plot(ref.alpha, ref.hM, color=MUTED, lw=1.2, ls="--"); ax.plot(ref.alpha, ref.hC, color=MUTED, lw=1.2, ls=":")
    ax.text(ref.alpha.iloc[-1], ref.hM.iloc[-1], " monopoly $h^M$", color=MUTED, fontsize=7, va="center")
    ax.text(ref.alpha.iloc[-1], ref.hC.iloc[-1], " competitive $h^C$", color=MUTED, fontsize=7, va="center")
    for n in Ns:
        s = summary[summary.n_mm == n].sort_values("alpha")
        ax.errorbar(s.alpha, s.half_spread_mean, yerr=s.half_spread_std, color=C[n], lw=2, marker="o",
                    ms=5, capsize=3, label=f"N = {n}")
    ax.set_xlabel(r"toxicity $\alpha$"); ax.set_ylabel("mean half-spread (ticks)")
    ax.set_title("(b) Learned quotes vs exact benchmarks", loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)

    # (c) impulse response
    ax = axes[2]
    if not imp.empty:
        if imp[(imp.alpha == imp_alpha) & (imp.n_mm == imp_n)].empty:   # fall back to first available combo
            imp_alpha, imp_n = imp.alpha.iloc[0], imp.n_mm.iloc[0]
        s = imp[(imp.alpha == imp_alpha) & (imp.n_mm == imp_n)].sort_values("rel_t")
        ax.plot(s.rel_t, s.rivals_diff, color=C[3], lw=2, marker="o", ms=4, label="rivals: shocked - baseline")
        ax.axhline(0, color=MUTED, lw=1, ls=":"); ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel("periods after forced undercut"); ax.set_ylabel("rivals' half-spread change (ticks)")
        ax.set_title(rf"(c) Paired impulse response, $\alpha$={imp_alpha}, N={imp_n}", loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="cfl_2mm")
    ap.add_argument("--tail", type=int, default=2000)
    ap.add_argument("--imp-alpha", type=float, default=0.3)
    ap.add_argument("--imp-n", type=int, default=2)
    args = ap.parse_args()
    results = Path(args.results)
    runs, imp = load_runs(results, args.run, args.tail)
    if runs.empty:
        raise SystemExit("no completed runs found")
    summary = summarize(runs)
    summary.to_csv(results / "replication_summary.csv", index=False)
    runs.to_csv(results / "replication_runs.csv", index=False)
    imps = impulse_summary(imp)
    if not imps.empty:
        imps.to_csv(results / "replication_impulse.csv", index=False)
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string(index=False))
    make_figure(summary, imps, results / "fig_replication", args.imp_alpha, args.imp_n)
    print("figure:", results / "fig_replication.png")


if __name__ == "__main__":
    main()
