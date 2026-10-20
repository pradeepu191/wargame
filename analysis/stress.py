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
        q = lag.vol_lag.quantile([1 / 3, 2 / 3]).to_numpy()
        terc = np.digitize(lag.vol_lag.to_numpy(), q)  # 0 calm, 1 middle, 2 stress
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
