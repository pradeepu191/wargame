"""Figures for the Workshop 2 experiments, from the committed CSVs (each is skipped if its CSV is missing).

    python analysis/workshop2_figs.py           # -> results/fig_relearn_attack, fig_psro, fig_queue_steer, fig_queue_budget

  fig_relearn_attack   public_attacker_relearn.csv: what each attacker earns while the cartel
                       re-learns, and how much of the cartel's current reaction table it knows
  fig_psro             psro.csv: the cartel's rent and its exploitability by round, both modes,
                       and the path of (rent, exploitability)
  fig_queue_steer      queue_steer.csv: the entrant's profit by what it played while the
                       incumbents learned and what it plays afterwards
  fig_queue_budget     queue_budget.csv: profit and placements against the request allowance
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
ATT_C = {"stale": "#7f8c8d", "fresh": "#0b0b0b", "public": "#e0701b", "public_w2000": "#2f6fb3", "public_w20000": "#1f9d8f"}
ATT_L = {"stale": "exact best response to the original cartel, never re-solved", "fresh": "oracle: re-solves from the cartel's current tables",
         "public": "public attacker, keeps every observation", "public_w2000": "public attacker, window 2000 periods",
         "public_w20000": "public attacker, window 20000 periods"}
POL_C = {"always": "#7f8c8d", "anon": "#2f6fb3", "id": "#e0701b", "absent": "#0b0b0b"}


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def band(ax, x, g, col, by, **kw):
    """Mean line with a 95% band over the units in column `by`."""
    piv = g.pivot_table(index=by, columns=x, values=col)
    m, n = piv.mean(), piv.count()
    h = 1.96 * piv.std(ddof=1) / np.sqrt(n)
    ax.plot(m.index, m.values, **kw)
    ax.fill_between(m.index, (m - h).values, (m + h).values, color=kw.get("color"), alpha=0.12, linewidth=0)


def finish(fig, axes, out):
    for ax in np.ravel(axes):
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figure:", out.with_suffix(".png"))


def relearn(results: Path):
    f = results / "public_attacker_relearn.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    n_rel = d[d.phase == "relearning"].episode.max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for name in [a for a in ATT_C if a in set(d.attacker)]:
        g = d[d.attacker == name]
        kw = dict(color=ATT_C[name], lw=1.8, marker="o", ms=3.5, label=ATT_L[name])
        band(axes[0], "episode", g, "entrant_norm", "run", **kw)
        band(axes[1], "episode", g, "inc_rent_norm", "run", **kw)
        if g.agree_all.notna().any():
            band(axes[2], "episode", g, "agree_all", "run", **kw)
    for ax in axes:
        ax.axvline(n_rel, color=MUTED, lw=0.8, ls=":"); ax.set_xlabel("episode (cartel re-learns until the dotted line, then is frozen)")
    axes[0].set_title(f"entrant's profit, rent units ({d.run.nunique()} cartels)", loc="left", fontsize=9)
    axes[1].set_title("cartel's profit above Nash, rent units", loc="left", fontsize=9); axes[1].axhline(0, color=MUTED, lw=0.6)
    axes[2].set_title("share of the cartel's current reaction table\nthe attacker's estimate gets right", loc="left", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    finish(fig, axes, results / "fig_relearn_attack")


def psro(results: Path):
    f = results / "psro.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    modes = [m for m in ("last", "uniform") if m in set(d["mode"])]
    lab = {"last": "cartel re-learns against the last best response", "uniform": "cartel re-learns against all past best responses"}
    fig, axes = plt.subplots(1, len(modes) + 1, figsize=(4.4 * (len(modes) + 1), 3.6), constrained_layout=True)
    series = [("C", "cartel's rent alone", "#0b0b0b"), ("fresh_loss", "its loss to the exact best response", "#c8372d"),
              ("fresh_entrant", "exact best response earns", "#e0701b"), ("public_entrant", "public attacker earns", "#2f6fb3"),
              ("stale_entrant", "last round's attacker earns", "#7f8c8d"), ("undercut_entrant", "undercut rule earns (held out)", "#1f9d8f")]
    for ax, mode in zip(axes, modes):
        g = d[d["mode"] == mode]
        for col, label, c in series:
            band(ax, "round", g[g[col].notna()], col, "run", color=c, lw=1.7, marker="o", ms=3.5, label=label)
        ax.set_title(f"{lab[mode]} ({g.run.nunique()} cartels)", loc="left", fontsize=8.5)
        ax.set_xlabel("round"); ax.axhline(0, color=MUTED, lw=0.6)
    axes[0].set_ylabel("rent units"); axes[0].legend(frameon=False, fontsize=7)
    ax = axes[-1]
    for mode, c in zip(modes, ("#e0701b", "#2f6fb3")):
        m = d[d["mode"] == mode].groupby("round")[["C", "fresh_loss"]].mean()
        ax.plot(m.C, m.fresh_loss, color=c, lw=1.5, marker="o", ms=4, label=lab[mode])
        for r, row in m.iterrows():
            ax.annotate(str(r), (row.C, row.fresh_loss), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--", label="loss = rent (rent fully exploitable)")
    ax.set_xlabel("cartel's rent alone"); ax.set_ylabel("loss to the exact best response")
    ax.set_title("path by round: defended, or only less collusive?", loc="left", fontsize=8.5); ax.legend(frameon=False, fontsize=7)
    finish(fig, axes, results / "fig_psro")


def queue_steer(results: Path):
    f = results / "queue_steer.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    cells = sorted(set(zip(d.rho_z, d.mark_lag)))
    fig, axes = plt.subplots(1, len(cells), figsize=(4.8 * len(cells), 3.6), constrained_layout=True, squeeze=False)
    T = ["absent", "always", "anon", "id"]; E = ["always", "anon", "id"]
    for ax, (rz, L) in zip(axes[0], cells):
        g = d[np.isclose(d.rho_z, rz) & (d.mark_lag == L)]
        for k, e in enumerate(E):
            s = g[g.test_policy == e].groupby("train_policy").entrant_pnl
            m, h = s.mean().reindex(T), (1.96 * s.std(ddof=1) / np.sqrt(s.count())).reindex(T)
            ax.bar(np.arange(len(T)) + (k - 1) * 0.26, m.values, 0.25, yerr=h.values, color=POL_C[e], capsize=2,
                   label=f"then plays: {e}", error_kw=dict(lw=0.8))
        w = g.groupby("train_policy")[["w_always", "w_never", "w_anon"]].mean().reindex(T)
        ax.set_xticks(range(len(T)))
        ax.set_xticklabels([f"{t}\n(incumbents end at\nalways {w.w_always[t]:.2f}, select {w.w_anon[t]:.2f})" for t in T], fontsize=7)
        ax.set_title(rf"$\rho_z$ = {rz}, L = {L} ({g.seed.nunique()} seeds)", loc="left", fontsize=9)
        ax.set_xlabel("what the entrant played while the incumbents were learning"); ax.axhline(0, color=MUTED, lw=0.6)
    axes[0, 0].set_ylabel("entrant's profit per period after the incumbents freeze"); axes[0, 0].legend(frameon=False, fontsize=7)
    finish(fig, axes, results / "fig_queue_steer")


def queue_budget(results: Path):
    f = results / "queue_budget.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    order = sorted(d.budget0.unique(), reverse=True)
    lab = ["none" if not np.isfinite(b) else f"{b:g}" for b in order]
    pols = [p for p in ("always", "anon", "id") if p in set(d.policy)] + sorted(p for p in set(d.policy) if p.startswith("id_hyst"))
    cols = {"always": "#7f8c8d", "anon": "#2f6fb3", "id": "#e0701b"}
    rzs = sorted(d.rho_z.unique())
    fig, axes = plt.subplots(len(rzs), 3, figsize=(13, 3.3 * len(rzs)), constrained_layout=True, squeeze=False)
    for row, rz in zip(axes, rzs):
        g = d[np.isclose(d.rho_z, rz)]
        for p in pols:
            s = g[g.policy == p].groupby("budget0")
            c = cols.get(p, "#8e44ad")
            for ax, col in zip(row, ("entrant_pnl", "entrant_placements", "incumbent_pnl")):
                m = s[col].mean().reindex(order); h = (1.96 * s[col].std(ddof=1) / np.sqrt(s[col].count())).reindex(order)
                ax.errorbar(range(len(order)), m.values, yerr=h.values, color=c, marker="o", ms=4, lw=1.5, capsize=2,
                            label=p.replace("id_hyst", "id, minimum lifetime "))
        for ax, t in zip(row, ("entrant's profit per period, by its policy", "entrant's order placements per period",
                               "anonymous-selector incumbents' profit per period")):
            ax.set_xticks(range(len(order))); ax.set_xticklabels(lab); ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_title(rf"{t} ($\rho_z$ = {rz}, {g.seed.nunique()} seeds)", loc="left", fontsize=8.5)
            ax.set_xlabel("request allowance per episode of 100 periods (+1 per fill)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    finish(fig, axes, results / "fig_queue_budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    style()
    r = Path(args.results)
    relearn(r); psro(r); queue_steer(r); queue_budget(r)


if __name__ == "__main__":
    main()
