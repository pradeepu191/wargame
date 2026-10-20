"""Who is at the touch when the flow turns toxic?  The empirical counterpart of the anonymity lever.

The model (sim/queue_env.py, experiments/queue_game.py --identity) says that incumbents who can
read the regime -- from the tape with a lag, from identities without one -- withdraw from the
inside when toxic flow arrives, and the fringe of always-resting makers absorbs the toxic fills.
On the book this would look like: in the minutes after a stressful spell, the top market makers'
share of fills at the touch falls, fewer of them are present, and the fills that do get taken at
the touch lose more to the mid for the makers left behind than for the top makers.

Per 5-minute block (fills marked against the true mid, analysis.l2.true_mid_markouts):
  * top_share      -- the top-K MM wallets' share of fills at the pre-trade touch
  * n_top_active   -- how many of the top-K took at least one fill at the touch
  * tox_rest       -- adverse selection suffered by everyone else at the touch, -mean markout
                      (excludes the top makers' own fills so their share cannot move it by
                      construction)
  * vol            -- the mid's absolute 10 s move per fill, in bps, over all fills in the block
                      (the regime read off the price path, independent of who filled)
  * the same two in the previous block when that block is contiguous (lagged stress: what a maker
    reading the tape would have seen)

Reported
  * Spearman rank correlations of top_share and n_top_active with contemporaneous and lagged
    stress (blocks with >= n_min_block fills at the touch)
  * terciles of lagged vol (calm / middle / stress): top share, top makers active, top and rest
    markout at the touch in each
The model's sign is negative: skilled makers leave when it gets toxic.  A positive sign would say
the opposite -- the top makers are the ones who stay (or arrive) when it is dangerous, and the
fringe is what withdraws -- which would be just as much of a finding about who the fringe is.

Timescale.  The model's regime lives for ~1/(1-rho_z) periods, seconds on this book, so the block
test is also run at 30 s and 60 s, and a fill-level version conditions each fill at the touch on
the mid's absolute move over the preceding w seconds (what a maker reading the tape had just
seen when the order arrived):
    python analysis/stress.py --raw data/raw/live --results results/live_2026_10
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_MS = 300_000


def block_table(tm: pd.DataFrame, top: list[str], horizon_s: int = 10, block_ms: int = BLOCK_MS) -> pd.DataFrame:
    """One row per block with at least one fill at the touch; see the module docstring for columns."""
    mk = f"markout_{horizon_s}_bps"
    imp = f"impact_{horizon_s}_bps"
    t = tm.dropna(subset=[mk]).copy()
    if t.empty:
        return pd.DataFrame()
    t["block"] = (t.time_ms.to_numpy() // block_ms).astype(np.int64)
    t["is_top"] = t.maker.isin(set(top))
    at = t[t.at_touch.astype(bool)]
    rows = []
    for b, g in at.groupby("block", sort=True):
        gt, gr = g[g.is_top], g[~g.is_top]
        allb = t[t.block == b]
        rows.append({"block": int(b), "n_touch": int(len(g)), "n_fills": int(len(allb)),
                     "top_share": float(len(gt) / len(g)), "n_top_active": int(gt.maker.nunique()),
                     "tox_rest": float(-gr[mk].mean()) if len(gr) else np.nan,
                     "vol": float(allb[imp].abs().mean() / 2) if imp in allb else np.nan,
                     "top_markout": float(gt[mk].mean()) if len(gt) else np.nan,
                     "rest_markout": float(gr[mk].mean()) if len(gr) else np.nan})
    bt = pd.DataFrame(rows)
    contiguous = bt.block.diff() == 1
    for c in ("tox_rest", "vol", "top_share"):
        bt[f"{c}_lag"] = np.where(contiguous, bt[c].shift(1), np.nan)
    return bt


def _terciles(x: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 by rank thirds; ties are split by order so a mass of equal values (a mid that did
    not move in the window) cannot collapse two terciles into one."""
    return pd.qcut(pd.Series(x).rank(method="first"), 3, labels=False).to_numpy()


def _spearman(x, y) -> tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    from scipy.stats import spearmanr
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def stress_summary(tm: pd.DataFrame, feats: pd.DataFrame, horizon_s: int = 10, top_k: int = 10,
                   block_ms: int = BLOCK_MS, n_min_block: int = 20) -> dict:
    if feats is None or not len(feats) or "is_mm" not in feats or tm.empty:
        return {}
    top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(top_k))
    bt = block_table(tm, top, horizon_s=horizon_s, block_ms=block_ms)
    if bt.empty:
        return {}
    b = bt[bt.n_touch >= n_min_block]
    out = {"st_n_blocks": int(len(b))}
    for yname in ("top_share", "n_top_active"):
        for xname in ("tox_rest", "vol", "tox_rest_lag", "vol_lag"):
            r, p, n = _spearman(b[xname].to_numpy(dtype=float), b[yname].to_numpy(dtype=float))
            out[f"st_{yname}_vs_{xname}"], out[f"st_{yname}_vs_{xname}_p"] = r, p
    # does the top share persist block to block (a slow-moving presence) or churn?
    out["st_top_share_autocorr"], _, _ = _spearman(b.top_share_lag.to_numpy(dtype=float), b.top_share.to_numpy(dtype=float))
    # terciles of lagged stress
    lag = b.dropna(subset=["vol_lag"])
    if len(lag) >= 9:
        terc = _terciles(lag.vol_lag.to_numpy())  # 0 calm, 1 middle, 2 stress
        for k, name in enumerate(("calm", "mid", "stress")):
            g = lag[terc == k]
            out[f"st_{name}_n_blocks"] = int(len(g))
            out[f"st_{name}_top_share"] = float(g.top_share.mean())
            out[f"st_{name}_n_top_active"] = float(g.n_top_active.mean())
            # fill-weighted markouts inside the tercile
            out[f"st_{name}_top_markout_bps"] = float(np.nansum(g.top_markout * g.top_share * g.n_touch) / max(np.nansum(g.top_share * g.n_touch), 1e-12))
            out[f"st_{name}_rest_markout_bps"] = float(np.nansum(g.rest_markout * (1 - g.top_share) * g.n_touch) / max(np.nansum((1 - g.top_share) * g.n_touch), 1e-12))
            out[f"st_{name}_vol_bps"] = float(g.vol.mean())
    out["_stress_blocks"] = bt
    return out


def fill_level_stress(tm: pd.DataFrame, book: pd.DataFrame, top: list[str], horizon_s: int = 10,
                      windows_s=(10, 60), block_ms: int = BLOCK_MS) -> dict:
    """Condition each fill at the touch on the mid's absolute move over the preceding w seconds.
    Reports, per window: the Spearman correlation of 'filled by a top maker' with that move,
    the top makers' share of touch fills in calm / middle / stress terciles of the move, and the
    top and rest markouts in each tercile (cluster SE over 5-minute blocks)."""
    from analysis.l2 import mid_at
    from analysis.makers import cluster_se
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk])
    t = t[t.at_touch.astype(bool)].copy()
    if t.empty:
        return {}
    t["is_top"] = t.maker.isin(set(top)).astype(float)
    t["block"] = (t.time_ms.to_numpy() // block_ms).astype(np.int64)
    times = t.time_ms.to_numpy()
    out = {"fl_n_touch": int(len(t))}
    for w in windows_s:
        past = mid_at(book, times - w * 1000, strict=False, max_stale_ms=60_000)
        move = np.abs(t.mid_0.to_numpy() - past) / t.mid_0.to_numpy() * 1e4
        ok = np.isfinite(move)
        g = t[ok].assign(move=move[ok])
        r, p, n = _spearman(g.move.to_numpy(), g.is_top.to_numpy())
        out[f"fl{w}_top_vs_move"], out[f"fl{w}_top_vs_move_p"], out[f"fl{w}_n"] = r, p, n
        terc = _terciles(g.move.to_numpy())
        for k, name in enumerate(("calm", "mid", "stress")):
            h = g[terc == k]
            ht, hr = h[h.is_top > 0], h[h.is_top == 0]
            out[f"fl{w}_{name}_move_bps"] = float(h.move.mean())
            out[f"fl{w}_{name}_top_share"] = float(h.is_top.mean())
            out[f"fl{w}_{name}_top_markout_bps"] = float(ht[mk].mean()) if len(ht) else np.nan
            out[f"fl{w}_{name}_top_se_bps"] = cluster_se(ht[mk].to_numpy(), ht.block.to_numpy()) if len(ht) > 1 else np.nan
            out[f"fl{w}_{name}_rest_markout_bps"] = float(hr[mk].mean()) if len(hr) else np.nan
            out[f"fl{w}_{name}_rest_se_bps"] = cluster_se(hr[mk].to_numpy(), hr.block.to_numpy()) if len(hr) > 1 else np.nan
    return out


def identity_selection(tm: pd.DataFrame, book: pd.DataFrame | None, top: list[str], horizon_s: int = 10,
                       n_min_taker: int = 30, window_s: int = 10, block_ms: int = BLOCK_MS) -> dict:
    """Do the top makers avoid the takers who were toxic LAST night?  Takers are typed on the
    first session (cut at the largest gap > 1 h, else the median fill) by the mean markout they
    inflicted on whoever filled them; every touch fill of the later session is then tagged with
    its taker's type tercile (benign / middle / toxic) or 'new' (unseen, or < n_min_taker fills).
    Reported: the top makers' share of fills from each type, the top and rest markouts by type,
    the Spearman correlation of 'filled by a top maker' with the taker's past toxicity, and --
    when the book is given -- the same share split by the mid's move over the previous window_s
    seconds, so that identity selection can be told from tape-reading.  If identity is what the
    top makers condition on, their share of the toxic takers' fills is lower than their share of
    the benign takers' fills at every level of the recent move."""
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk])
    t = t[t.at_touch.astype(bool)].copy()
    if t.empty:
        return {}
    t["is_top"] = t.maker.isin(set(top)).astype(float)
    t["block"] = (t.time_ms.to_numpy() // block_ms).astype(np.int64)
    gaps = np.diff(t.time_ms.to_numpy())
    cut = t.time_ms.iloc[int(gaps.argmax()) + 1] if len(gaps) and gaps.max() > 3_600_000 else t.time_ms.iloc[len(t) // 2]
    a, b = t[t.time_ms < cut], t[t.time_ms >= cut].copy()
    typed = a.groupby("taker")[mk].agg(["mean", "size"])
    typed = typed[typed["size"] >= n_min_taker]
    out = {"id_n_typed_takers": int(len(typed)), "id_n_fills_later": int(len(b)), "id_cut_is_session": bool(len(gaps) and gaps.max() > 3_600_000)}
    if len(typed) < 9:
        return out
    tox = -typed["mean"]  # positive = toxic to the maker
    terc = pd.Series(_terciles(tox.to_numpy()), index=tox.index)
    b["taker_tox"] = b.taker.map(tox)
    b["type"] = b.taker.map(terc).map({0: "benign", 1: "middle", 2: "toxic"}).fillna("new")
    from analysis.makers import cluster_se
    for name in ("benign", "middle", "toxic", "new"):
        g = b[b.type == name]
        gt, gr = g[g.is_top > 0], g[g.is_top == 0]
        out[f"id_{name}_fill_share"] = float(len(g) / len(b))
        out[f"id_{name}_top_share"] = float(g.is_top.mean()) if len(g) else np.nan
        out[f"id_{name}_top_markout_bps"] = float(gt[mk].mean()) if len(gt) else np.nan
        out[f"id_{name}_top_se_bps"] = cluster_se(gt[mk].to_numpy(), gt.block.to_numpy()) if len(gt) > 1 else np.nan
        out[f"id_{name}_rest_markout_bps"] = float(gr[mk].mean()) if len(gr) else np.nan
        out[f"id_{name}_rest_se_bps"] = cluster_se(gr[mk].to_numpy(), gr.block.to_numpy()) if len(gr) > 1 else np.nan
        out[f"id_{name}_past_tox_bps"] = float(g.taker_tox.mean()) if name != "new" and len(g) else np.nan
    known = b.dropna(subset=["taker_tox"])
    r, p, n = _spearman(known.taker_tox.to_numpy(), known.is_top.to_numpy())
    out["id_top_vs_past_tox"], out["id_top_vs_past_tox_p"], out["id_n_known"] = r, p, n
    # the realized toxicity of the later session by type: did the typing hold up?
    out["id_type_holds_bps"] = float(known[known.type == "benign"][mk].mean() - known[known.type == "toxic"][mk].mean())
    if book is not None and len(book):
        from analysis.l2 import mid_at
        past = mid_at(book, known.time_ms.to_numpy() - window_s * 1000, strict=False, max_stale_ms=60_000)
        move = np.abs(known.mid_0.to_numpy() - past) / known.mid_0.to_numpy() * 1e4
        ok = np.isfinite(move)
        k = known[ok].assign(mterc=_terciles(move[ok]))
        for mi, mname in enumerate(("calm", "stress")):
            h = k[k.mterc == (0 if mi == 0 else 2)]
            for name in ("benign", "toxic"):
                g = h[h.type == name]
                out[f"id_{mname}_{name}_top_share"] = float(g.is_top.mean()) if len(g) else np.nan
                out[f"id_{mname}_{name}_n"] = int(len(g))
    return out


def main():
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # run from anywhere, no install needed
    from analysis.calibrate import load_trades
    from analysis.l2 import load_l2, true_mid_markouts
    from analysis.wallets import with_roles
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/live")
    ap.add_argument("--results", default="results/live_2026_10", help="holds wallets_<COIN>.csv; outputs go here")
    ap.add_argument("--coins", default="BTC,ETH,SOL,HYPE")
    ap.add_argument("--blocks", default="30,60,300", help="block lengths in seconds")
    ap.add_argument("--windows", default="10,60", help="fill-level look-back windows in seconds")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    raw, res = Path(args.raw), Path(args.results)
    blocks = [int(b) for b in args.blocks.split(",")]
    windows = tuple(int(w) for w in args.windows.split(","))
    scale_rows, fill_rows = [], []
    for c in args.coins.split(","):
        tr, book = load_trades(raw, c), load_l2(raw, c)
        if tr.empty or book.empty or not (res / f"wallets_{c}.csv").exists():
            print(f"{c}: missing trades, book or wallets_{c}.csv"); continue
        feats = pd.read_csv(res / f"wallets_{c}.csv")
        top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(args.top_k))
        tm = true_mid_markouts(with_roles(tr), book, horizons_s=(10,))
        for b in blocks:
            s = stress_summary(tm, feats, horizon_s=10, top_k=args.top_k, block_ms=b * 1000,
                               n_min_block=max(8, round(20 * b / 300)))
            s.pop("_stress_blocks", None)
            scale_rows.append({"coin": c, "block_s": b, **s})
            print(f"{c} {b:>4d}s blocks ({s.get('st_n_blocks', 0)}): top share vs lagged vol {s.get('st_top_share_vs_vol_lag', float('nan')):+.2f} "
                  f"(p={s.get('st_top_share_vs_vol_lag_p', float('nan')):.3f}), vs lagged toxicity {s.get('st_top_share_vs_tox_rest_lag', float('nan')):+.2f} "
                  f"(p={s.get('st_top_share_vs_tox_rest_lag_p', float('nan')):.3f}), contemporaneous vol {s.get('st_top_share_vs_vol', float('nan')):+.2f}; "
                  f"terciles of lagged vol: top share {s.get('st_calm_top_share', float('nan')):.3f} / {s.get('st_mid_top_share', float('nan')):.3f} / "
                  f"{s.get('st_stress_top_share', float('nan')):.3f}, top markout {s.get('st_calm_top_markout_bps', float('nan')):+.2f} / "
                  f"{s.get('st_mid_top_markout_bps', float('nan')):+.2f} / {s.get('st_stress_top_markout_bps', float('nan')):+.2f}, rest "
                  f"{s.get('st_calm_rest_markout_bps', float('nan')):+.2f} / {s.get('st_mid_rest_markout_bps', float('nan')):+.2f} / "
                  f"{s.get('st_stress_rest_markout_bps', float('nan')):+.2f}", flush=True)
        f = fill_level_stress(tm, book, top, horizon_s=10, windows_s=windows)
        f.update(identity_selection(tm, book, top, horizon_s=10, window_s=windows[0]))
        fill_rows.append({"coin": c, **f})
        for w in windows:
            print(f"{c} fill-level, move over previous {w:>3d}s ({f.get(f'fl{w}_n', 0)} touch fills): corr(top, move) {f.get(f'fl{w}_top_vs_move', float('nan')):+.3f} "
                  f"(p={f.get(f'fl{w}_top_vs_move_p', float('nan')):.3f}); calm/mid/stress move {f.get(f'fl{w}_calm_move_bps', float('nan')):.2f} / "
                  f"{f.get(f'fl{w}_mid_move_bps', float('nan')):.2f} / {f.get(f'fl{w}_stress_move_bps', float('nan')):.2f} bps, top share "
                  f"{f.get(f'fl{w}_calm_top_share', float('nan')):.3f} / {f.get(f'fl{w}_mid_top_share', float('nan')):.3f} / {f.get(f'fl{w}_stress_top_share', float('nan')):.3f}, "
                  f"top markout {f.get(f'fl{w}_calm_top_markout_bps', float('nan')):+.2f}±{f.get(f'fl{w}_calm_top_se_bps', float('nan')):.2f} / "
                  f"{f.get(f'fl{w}_mid_top_markout_bps', float('nan')):+.2f} / {f.get(f'fl{w}_stress_top_markout_bps', float('nan')):+.2f}±{f.get(f'fl{w}_stress_top_se_bps', float('nan')):.2f}, "
                  f"rest {f.get(f'fl{w}_calm_rest_markout_bps', float('nan')):+.2f}±{f.get(f'fl{w}_calm_rest_se_bps', float('nan')):.2f} / "
                  f"{f.get(f'fl{w}_mid_rest_markout_bps', float('nan')):+.2f} / {f.get(f'fl{w}_stress_rest_markout_bps', float('nan')):+.2f}±{f.get(f'fl{w}_stress_rest_se_bps', float('nan')):.2f}", flush=True)
        if "id_top_vs_past_tox" in f:
            print(f"{c} identity selection ({f['id_n_typed_takers']} takers typed on the {'first night' if f['id_cut_is_session'] else 'first half'}, "
                  f"{f['id_n_fills_later']} later touch fills): corr(top, taker's past toxicity) {f['id_top_vs_past_tox']:+.3f} (p={f['id_top_vs_past_tox_p']:.3f}); "
                  f"top share of benign/middle/toxic/new takers' fills {f['id_benign_top_share']:.3f} / {f['id_middle_top_share']:.3f} / "
                  f"{f['id_toxic_top_share']:.3f} / {f['id_new_top_share']:.3f} (fill shares {f['id_benign_fill_share']:.2f} / {f['id_middle_fill_share']:.2f} / "
                  f"{f['id_toxic_fill_share']:.2f} / {f['id_new_fill_share']:.2f}); top markout benign/toxic {f['id_benign_top_markout_bps']:+.2f}±{f['id_benign_top_se_bps']:.2f} / "
                  f"{f['id_toxic_top_markout_bps']:+.2f}±{f['id_toxic_top_se_bps']:.2f}, rest {f['id_benign_rest_markout_bps']:+.2f}±{f['id_benign_rest_se_bps']:.2f} / "
                  f"{f['id_toxic_rest_markout_bps']:+.2f}±{f['id_toxic_rest_se_bps']:.2f}; typing held up by {f['id_type_holds_bps']:+.2f} bps"
                  + (f"; within calm tape: benign/toxic {f['id_calm_benign_top_share']:.3f} / {f['id_calm_toxic_top_share']:.3f}, within stressed tape: "
                     f"{f['id_stress_benign_top_share']:.3f} / {f['id_stress_toxic_top_share']:.3f}" if "id_calm_benign_top_share" in f else ""), flush=True)
    pd.DataFrame(scale_rows).to_csv(res / "stress_scale.csv", index=False)
    pd.DataFrame(fill_rows).to_csv(res / "stress_fill.csv", index=False)
    print("wrote", res / "stress_scale.csv", "and", res / "stress_fill.csv")


if __name__ == "__main__":
    main()
