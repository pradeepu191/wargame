"""Render a recorded trace (experiments/trace.py) to a GIF or MP4, for slides and supplementary
material when a browser is not available.  No model logic: it draws what the trace holds.

    python analysis/trace_video.py --scenario best_response                 # -> results/demo/best_response.gif
    python analysis/trace_video.py --all --periods 200 --step 2 --fps 15
    python analysis/trace_video.py --scenario identity_id --out clip.mp4    # MP4 needs ffmpeg on the PATH

--fps frames per second and --step periods per frame give the playback rate (periods/s = fps * step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
COL = ["#2f6fb3", "#1f9d8f", "#7a5cc0"]
ENT, TOX, OK, REF = "#e0701b", "#c8372d", "#3d8b55", "#8a8f97"


def colour(meta, j):
    return ENT if meta["agents"][j]["role"] == "entrant" else COL[j % 3]


def render(trace: dict, out: Path, periods: int | None, step: int, fps: int, window: int = 120):
    m, p = trace["meta"], trace["p"]
    n = min(len(p["t"]), periods or len(p["t"]))
    A = len(m["agents"])
    a = np.array(p["a"][:n]); f = np.array(p["f"][:n]); inf = np.array(p["inf"][:n]); z = np.array(p["z"][:n])
    queue = m["mode"] == "queue"
    b = m["benchmarks"]
    unit = abs(b["fill_value_calm"]) if queue else b["rent"]
    cum = np.vstack([np.zeros(A), np.cumsum(np.array(p["r"][:n]), axis=0)]) / unit
    has_z = m["config"]["regime_persistence"] > 0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.suptitle(m["title"], x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, m["text"], fontsize=8.5, color="#555", wrap=True, va="top")
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.06, right=0.98, wspace=0.18)

    def frame(k):
        k = min(n, max(1, k))
        ax.clear(); ax2.clear()
        lo = max(0, k - window); t = np.arange(lo, k)
        if queue:
            for j in range(A):
                on = a[lo:k, j] >= 1
                ax.bar(t[on], 0.7, bottom=A - 1 - j - 0.35, width=1.0, color=colour(m, j), alpha=0.55, linewidth=0)
                hit = f[lo:k] == j
                ax.scatter(t[hit], np.full(hit.sum(), A - 1 - j), s=14, zorder=3,
                           c=[TOX if i else OK for i in inf[lo:k][hit]])
            ax.set_yticks(range(A)); ax.set_yticklabels([m["agents"][A - 1 - j]["name"] for j in range(A)], fontsize=8)
            ax.set_ylim(-0.8, A - 0.4)
            ax.set_title(f"resting (bar) and fills (dots); spread fixed at {b['half_spread']} ticks", fontsize=9, loc="left")
        else:
            for j in range(A):
                ax.step(t, a[lo:k, j] + (j - (A - 1) / 2) * 0.08, where="mid", color=colour(m, j),
                        lw=2.0 if m["agents"][j]["role"] == "entrant" else 1.4, label=m["agents"][j]["name"])
            hit = f[lo:k] >= 0
            ax.scatter(t[hit], a[lo:k][hit].min(axis=1), s=12, zorder=3, c=[TOX if i else OK for i in inf[lo:k][hit]])
            for v, lab in ((b["hC"], "Nash quote"), (b["hM"], "monopoly quote")):
                ax.axhline(v, color=REF, ls="--", lw=1); ax.text(lo + 1, v + 0.3, lab, color=REF, fontsize=8)
            ax.set_ylim(0, m["K"] + 1); ax.set_ylabel("half-spread (ticks)")
            ax.set_title("quotes over time; dots are fills (red: informed taker)", fontsize=9, loc="left")
        if has_z:
            y0, y1 = ax.get_ylim()
            tox = z[lo:k] == 1
            ax.bar(t[tox], (y1 - y0) * 0.03, bottom=y0, width=1.0, color=TOX, linewidth=0)
        ax.set_xlim(lo - 0.5, max(k, window) - 0.5 if lo == 0 else k - 0.5); ax.set_xlabel("period (one taker arrival)")
        ax.grid(alpha=0.25)
        for j in range(A):
            ax2.plot(np.arange(k + 1), cum[:k + 1, j], color=colour(m, j), lw=1.8, label=m["agents"][j]["name"])
        ax2.set_xlim(0, n); ax2.set_ylim(cum.min() - 1, cum.max() + 1)
        ax2.set_title("cumulative profit (" + ("calm-fill units" if queue else "rent units") + ")", fontsize=9, loc="left")
        ax2.set_xlabel(f"period {k} of {n}  (about {k * m['period_seconds']:.0f} s of market time)")
        ax2.legend(fontsize=8, loc="upper left", frameon=False); ax2.grid(alpha=0.25)

    frames = list(range(step, n + 1, step))
    anim = FuncAnimation(fig, frame, frames=frames, interval=1000 / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        anim.save(out, fps=fps, dpi=110)
    else:
        anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, {fps * step} periods/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--traces", default=str(ROOT / "demo" / "traces"))
    ap.add_argument("--out", default=None, help="output file (.gif or .mp4); default results/demo/<scenario>.gif")
    ap.add_argument("--periods", type=int, default=None, help="render only the first N periods")
    ap.add_argument("--step", type=int, default=2, help="periods per frame")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    files = sorted(Path(args.traces).glob("*.json")) if args.all else [Path(args.traces) / f"{args.scenario or 'best_response'}.json"]
    for fpath in files:
        out = Path(args.out) if args.out and not args.all else ROOT / "results" / "demo" / f"{fpath.stem}.gif"
        render(json.loads(fpath.read_text()), out, args.periods, args.step, args.fps)


if __name__ == "__main__":
    main()
