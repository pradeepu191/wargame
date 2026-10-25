"""Figures for the Workshop 1 report and slides, designed to be read in one pass: one job per
panel, a title that states the takeaway, direct labels, a reference line where the number has
a benchmark, and the same three colours throughout.

    python analysis/report_figs.py            # writes results/fig_*.{pdf,png}

  fig_rent                 learned quotes sit between Nash and monopoly (dumbbell, by toxicity)
  fig_attackers            what each attacker takes from the cartel, and the double oracle
  fig_lag                  the anonymous entrant's profit decays with the mark lag, the identity entrant's does not
  fig_touch                on Hyperliquid the ordinary fill at the touch loses; the top makers earn
  fig_identity_selection   the top makers avoid last night's toxic wallets, also on fills that were not pick-offs
  fig_anonymity            with identities the incumbents withdraw in the toxic state; uninformed takers lose access
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9890"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COINS = ["BTC", "ETH", "SOL", "HYPE"]


def style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9})


def ci(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, out: Path):
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)
    print("figure:", out.with_suffix(".pdf"))


# ----------------------------------------------------------------------------- 1. rent
def fig_rent(res: Path):
    s = pd.read_csv(res / "cfl_exp2_summary.csv")
    r = pd.read_csv(res / "cfl_exp2_runs.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    rows = []
    for a in (0.1, 0.3, 0.5):
        g = s[np.isclose(s.alpha, a)]
        rr = r[np.isclose(r.alpha, a)]
        rows.append((a, int(g.hC.iloc[0]), float(rr.best_h_mean.mean()), int(g.hM.iloc[0]), float(rr.delta.mean())))
    y = np.arange(len(rows))[::-1]
    for yi, (a, hC, h, hM, C) in zip(y, rows):
        ax.plot([hC, hM], [yi, yi], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([hC], [yi], s=70, color=GREY, zorder=3)
        ax.scatter([hM], [yi], s=70, color=GREY, marker="s", zorder=3)
        ax.scatter([h], [yi], s=110, color=BLUE, zorder=4)
        ax.annotate(f"learned {h:.1f}", (h, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.annotate(f"rent index C = {C:.2f}", (19.4, yi), va="center", fontsize=9, color=INK)
    ax.annotate("Nash\n(competitive)", (rows[0][1], y[0]), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.annotate("joint\nmonopoly", (rows[0][3], y[0]), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([rf"informed share $\bar\alpha$ = {a}" for a, *_ in rows])
    ax.set_xlabel("half-spread the makers quote (ticks)")
    ax.set_xlim(0, 24.5); ax.set_ylim(-0.9, len(rows) - 0.3)
    ax.set_title("Independent Q-learners settle at the joint-monopoly quote,\nnot the Nash quote (10 seeds; N = 2 and 3 makers)", fontsize=10)
    grid(ax, "x")
    save(fig, res / "fig_rent")


# ----------------------------------------------------------------------------- 2. attackers
def fig_attackers(res: Path):
    d = pd.read_csv(res / "best_response.csv")
    r = pd.read_csv(res / "best_response_relearn.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    # --- left: what each attacker takes
    order = [("undercut", "undercut rule\n(one tick inside the last quote)"), ("markout", "markout rule\n(undercut only when rent is inferred)"),
             ("dp:1", "exact best response\n(backward induction, own profit)"), ("dp:0", "damage-maximiser\n(exact, incumbents' loss)")]
    per = d.groupby(["run", "attacker"]).agg(L=("L_BR_norm", "mean"), e=("entrant_pnl_norm", "mean"), C=("C", "first")).reset_index()
    C_mean = per.groupby("run").C.first().mean()
    y = np.arange(len(order))[::-1]
    h = 0.36
    for yi, (key, label) in zip(y, order):
        g = per[per.attacker == key]
        L, e = g.L.mean(), g.e.mean()
        Lci, eci = ci(g.L), ci(g.e)
        ax.barh(yi + h / 2, L, h, color=BLUE, xerr=Lci, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label="what the cartel loses" if yi == y[0] else None)
        ax.barh(yi - h / 2, e, h, color=ORANGE, xerr=eci, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label="what the attacker earns" if yi == y[0] else None)
        ax.annotate(f"{L:.2f}", (L + Lci, yi + h / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.annotate(f"{e:.2f}", (max(e, 0) + eci, yi - h / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axvline(C_mean, color=MUTED, lw=1, ls="--")
    ax.annotate(f"the cartel's whole rent\n(C = {C_mean:.2f} on average)", (C_mean + 0.04, y[-1] - 0.42), ha="left", va="top", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in order], fontsize=8.8)
    ax.set_xlabel(r"rent units ($1$ = the monopoly rent $\Pi^M-\Pi^C$); 60 cartels, 95% intervals")
    ax.set_xlim(-0.05, 2.25); ax.set_ylim(-1.0, len(order) + 0.25)
    ax.set_title("Rules take two thirds of the rent; the exact best response\ntakes all of it and more", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    grid(ax, "x")
    # --- right: double oracle, pooled over cartels
    rounds = sorted(r["round"].unique())
    def series(mask_fn):
        m, e = [], []
        for k in rounds:
            g = r[(r["round"] == k) & mask_fn(r)]
            m.append(g.L_BR_norm.mean() if "L" in mask_fn.__name__ else g.groupby("run").C.first().mean())
            e.append(ci(g.L_BR_norm) if "L" in mask_fn.__name__ else ci(g.groupby("run").C.first()))
        return np.array(m), np.array(e)
    def C_fn(x): return x.attacker.isin(["dp:1", "dp:1@fresh"])
    def L_fresh(x): return x.attacker.isin(["dp:1", "dp:1@fresh"])
    def L_stale(x): return x.attacker == "dp:1@stale"
    for fn, color, label, rr in ((C_fn, BLUE, "cartel's rent when it plays alone", rounds), (L_fresh, ORANGE, "what a re-solved best response takes", rounds), (L_stale, GREY, "what the previous round's attacker still takes", rounds[1:])):
        m, e = [], []
        for k in rr:
            g = r[(r["round"] == k) & fn(r)]
            if fn is C_fn:
                v = g.groupby("run").C.first()
            else:
                v = g.L_BR_norm
            m.append(v.mean()); e.append(ci(v))
        m, e = np.array(m), np.array(e)
        bx.fill_between(rr, m - e, m + e, color=color, alpha=0.18, lw=0)
        bx.plot(rr, m, color=color, lw=2, marker="o", ms=5, label=label)
        bx.annotate(f"{m[-1]:.2f}", (rr[-1], m[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    bx.set_xticks(rounds); bx.set_xticklabels(["before", "round 1", "round 2", "round 3"])
    bx.set_xlabel("each round: the cartel re-learns 3000 episodes\nagainst the attacker, then the attacker re-solves")
    bx.set_ylabel("rent units")
    bx.set_ylim(0.3, 1.12); bx.set_xlim(-0.2, 3.4)
    bx.set_title("Re-learning does not defend the rent\n(60 cartels, 95% intervals)", fontsize=10)
    bx.legend(loc="lower left", fontsize=8.5)
    grid(bx)
    save(fig, res / "fig_attackers")


# ----------------------------------------------------------------------------- 3. lag
def fig_lag(res: Path):
    d = pd.read_csv(res / "cfl_exp2_identity_regime_10seed.csv")
    d = d[(np.isclose(d.kappa, 1.0)) & (np.isclose(d.alpha, 0.5))]
    lags = sorted(d.mark_lag.unique()); x = np.arange(len(lags))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
    # left: levels at rho_z = 0.99 (means; the paired difference carries the uncertainty)
    g = d[np.isclose(d.rho_z, 0.99)]
    for pol, color, label in (("anon", GREY, "anonymous entrant\n(tape only)"), ("id", BLUE, "identity entrant\n(tape + wallets)"), ("oracle", ORANGE, "oracle entrant\n(knows the regime)")):
        m = [g[(g.policy == pol) & (g.mark_lag == L)].entrant_pnl_norm.mean() for L in lags]
        ax.plot(x, m, color=color, lw=2, marker="o", ms=5)
        ax.annotate(label, (x[-1], m[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([str(L) for L in lags]); ax.set_xlim(-0.3, len(lags) + 1.3)
    ax.set_xlabel("mark lag L: arrivals until a fill's outcome is public")
    ax.set_ylabel("entrant profit per period (rent units)")
    ax.set_title("The tape-reading entrant's profit decays as the mark\narrives later; the identity entrant's does not ($\\rho_z$ = 0.99)", fontsize=10)
    grid(ax)
    # right: identity's value, paired per cartel
    for rho, color, label in ((0.9, AQUA, r"short toxic spells ($\rho_z$ = 0.9, ~10 arrivals)"), (0.99, BLUE, r"long toxic spells ($\rho_z$ = 0.99, ~100 arrivals)")):
        g = d[np.isclose(d.rho_z, rho)]
        m, e = [], []
        for L in lags:
            a = g[(g.policy == "id") & (g.mark_lag == L)].set_index("run").entrant_pnl_norm
            b = g[(g.policy == "anon") & (g.mark_lag == L)].set_index("run").entrant_pnl_norm
            diff = (a - b).dropna(); m.append(diff.mean()); e.append(ci(diff))
        bx.errorbar(x, m, yerr=e, color=color, lw=2, marker="o", ms=5, capsize=3, label=label)
        for xi, mi in zip(x, m):
            bx.annotate(f"+{mi:.2f}", (xi, mi), xytext=(0, 8 if rho == 0.99 else -14), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    bx.axhline(0, color=MUTED, lw=0.8)
    bx.set_xticks(x); bx.set_xticklabels([str(L) for L in lags])
    bx.set_xlabel("mark lag L")
    bx.set_ylabel("identity's value: identity entrant minus\nanonymous entrant (rent units)")
    bx.set_ylim(-0.02, 0.26)
    bx.set_title("Identity's value grows with the lag,\nup to the lifetime of a toxic spell", fontsize=10)
    bx.legend(loc="upper left", fontsize=8.5)
    grid(bx)
    fig.suptitle(r"Where identity pays ($\bar\alpha$ = 0.5, $\kappa_\alpha$ = 1; 10 cartels, paired, 95% intervals)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, res / "fig_lag")


# ----------------------------------------------------------------------------- 4. touch
def fig_touch(res: Path):
    d = pd.read_csv(res / "live_2026_10" / "calibration.csv").set_index("coin").loc[COINS]
    fig, ax = plt.subplots(figsize=(6.8, 3.3), constrained_layout=True)
    x = np.arange(len(COINS)); w = 0.38
    top, rest = d.mk_top_touch_markout_bps, d.mk_rest_touch_markout_bps
    ax.bar(x - w / 2, rest, w, color=GREY, yerr=1.96 * d.mk_rest_touch_se_bps, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label="everyone else")
    ax.bar(x + w / 2, top, w, color=BLUE, yerr=1.96 * d.mk_top_touch_se_bps, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label="the 10 most active market-maker wallets")
    for xi, (a, b, ea, eb) in enumerate(zip(rest, top, 1.96 * d.mk_rest_touch_se_bps, 1.96 * d.mk_top_touch_se_bps)):
        ax.annotate(f"{a:+.2f}", (xi - w / 2, min(a, 0) - ea), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.annotate(f"{b:+.2f}", (xi + w / 2, max(b, 0) + eb), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(COINS)
    ax.set_ylabel("maker's profit per fill vs the mid\n10 s later (bps)")
    ax.set_ylim(-1.1, 0.85)
    ax.set_title("Same price, same tick: the ordinary fill at the touch loses,\nthe top makers' fills break even or earn (Hyperliquid, two nights)", fontsize=10)
    ax.legend(loc="upper left", ncol=1)
    grid(ax)
    save(fig, res / "fig_touch")


# ----------------------------------------------------------------------------- 5. identity selection
def fig_identity_selection(res: Path):
    f = pd.read_csv(res / "live_2026_10" / "stress_fill.csv").set_index("coin").loc[COINS]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True, sharey=True)
    types = [("benign", BLUE, "wallet was benign on night 1"), ("new", GREY, "wallet unseen on night 1"), ("toxic", RED, "wallet was toxic on night 1")]
    x = np.arange(len(COINS)); w = 0.26
    for ax, suffix, title in ((axes[0], "", "All fills at the touch:\n2.6\u20136.5\u00d7 fewer of the toxic wallets' fills"),
                              (axes[1], "quiet_", "Only fills with no mid move in the next 1 s (not pick-offs):\nstill 1.3\u20132.8\u00d7 fewer, so it is not speed alone")):
        for k, (t, color, label) in enumerate(types):
            v = f[f"id_{t}_{suffix}top_share"]
            ax.bar(x + (k - 1) * w, v, w, color=color, edgecolor="white", lw=1, label=label if ax is axes[0] else None)
            for xi, vi in zip(x, v):
                ax.annotate(f"{vi:.2f}", (xi + (k - 1) * w, vi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(COINS)
        ax.set_title(title, fontsize=9.8)
        grid(ax)
    axes[0].set_ylabel("top-10 makers' share of that\nwallet type's fills on night 2")
    axes[0].set_ylim(0, 0.62)
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=1, handlelength=1.2, borderaxespad=0.2)
    fig.suptitle("The top-10 makers are missing from the fills of wallets that were toxic the night before", x=0.01, y=0.995, ha="left", fontsize=10.5, fontweight="bold")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    save(fig, res / "fig_identity_selection")


# ----------------------------------------------------------------------------- 6. anonymity
def fig_anonymity(res: Path):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(res / "queue_hedge_always_never_anon*_fringe0.csv")))])
    d["set"] = np.where(d.experts.str.contains("id"), "id", "anon")
    d = d[(d.above == 0) & (d.fringe == 0) & np.isclose(d.rho_z, 0.9)]
    lags = sorted(d.mark_lag.unique()); x = np.arange(len(lags)); w = 0.38
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    for k, (st, color, label) in enumerate((("anon", GREY, "incumbents see the tape only"), ("id", BLUE, "incumbents see wallet identities"))):
        g = d[(d.set == st) & (d.policy == "id")]
        m = [g[g.mark_lag == L].uninformed_fill_toxic.mean() for L in lags]; e = [ci(g[g.mark_lag == L].uninformed_fill_toxic) for L in lags]
        ax.bar(x + (k - 0.5) * w, m, w, color=color, yerr=e, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label=label)
        for xi, mi, ei in zip(x, m, e):
            ax.annotate(f"{mi:.2f}", (xi + (k - 0.5) * w, mi + ei), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        g2 = d[(d.set == st) & (d.policy == "always")]
        m2 = [g2[g2.mark_lag == L].entrant_pnl.mean() for L in lags]; e2 = [ci(g2[g2.mark_lag == L].entrant_pnl) for L in lags]
        bx.bar(x + (k - 0.5) * w, m2, w, color=color, yerr=e2, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), label=label)
        for xi, mi, ei in zip(x, m2, e2):
            bx.annotate(f"{mi:+.2f}", (xi + (k - 0.5) * w, min(mi, 0) - ei), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    for a in (ax, bx):
        a.set_xticks(x); a.set_xticklabels([str(L) for L in lags]); a.set_xlabel("mark lag L"); grid(a)
    ax.set_ylim(0, 1.25); ax.set_ylabel("P(uninformed taker finds a standing order\n| toxic state)"); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("With identities the makers withdraw in toxic spells\nand uninformed takers lose access", fontsize=10)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, columnspacing=1.0)
    bx.axhline(0, color=MUTED, lw=0.8); bx.set_ylim(-0.3, 0.12)
    bx.set_ylabel("profit per period of a maker that\nalways rests among them")
    bx.set_title("…and whoever cannot see identities\nis left holding the toxic flow", fontsize=10)
    fig.suptitle(r"Anonymity as a venue lever (queue game at break-even, no fringe, $\rho_z$ = 0.9, 10 seeds)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, res / "fig_anonymity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    res = Path(args.results)
    style()
    for name, fn in (("rent", fig_rent), ("attackers", fig_attackers), ("lag", fig_lag), ("touch", fig_touch),
                     ("identity_selection", fig_identity_selection), ("anonymity", fig_anonymity)):
        if args.only and name not in args.only.split(","):
            continue
        fn(res)


if __name__ == "__main__":
    main()
