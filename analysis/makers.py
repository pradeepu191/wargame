"""Per-maker realized spread on fills marked against the true mid: RQ3's empirical anchor.

The question. On a one-tick book the average fill at the touch loses to the mid within ten
seconds (analysis/l2.py, results/REPLICATION_NOTES.md).  If rent lives in fill selection, the
makers who select -- by queue position, cancellation, depth, identity -- should show a realized
spread above the average, persistently.  If every maker's fills look like the average fill, there
is no selection to find on this book and the queue game's premise needs rethinking.

Inputs: the marked-fill frame from analysis.l2.true_mid_markouts (one row per print with the
maker wallet, markout_<d>_bps, at_touch, effective_bps) and the wallet feature table from
analysis.wallets.wallet_features (is_mm and volume over the FULL trade window, so the "top
makers" are chosen on activity over days, not on the six hours of marked fills whose outcome is
being measured).

Reported
  * per maker with >= n_min marked fills: fills, volume, mean markout at the horizon with a
    cluster-robust SE (5-minute blocks; fills inside a block are not independent), share of
    fills at the pre-trade touch (1 - share in which the maker was swept behind the inside).
  * the top-K MM wallets by full-window volume as one fixed set: fill share, markout, SE; the
    same for the other MM wallets and for makers the classifier does not call MMs.
  * dispersion: how many makers clear n_min, what share of them have a positive mean markout,
    and what share of all marked fills those makers took.
  * persistence: Spearman rank correlation of per-maker markout between the first and second
    half of the window (makers with >= n_half fills in each half).  Persistent maker skill is the
    signature of selection rather than luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_MS = 300_000


def cluster_se(x: np.ndarray, block: np.ndarray) -> float:
    """SE of the mean of x with arbitrary dependence inside blocks (one-way cluster-robust)."""
    n = len(x)
    if n < 2:
        return float("nan")
    d = x - x.mean()
    sums = pd.Series(d).groupby(block).sum().to_numpy()
    return float(np.sqrt((sums ** 2).sum()) / n)


def _block(tm: pd.DataFrame, block_ms: int) -> np.ndarray:
    return (tm.time_ms.to_numpy() // block_ms).astype(np.int64)


def maker_table(tm: pd.DataFrame, horizon_s: int = 10, n_min: int = 100, block_ms: int = BLOCK_MS) -> pd.DataFrame:
    """One row per maker wallet with >= n_min marked fills."""
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk]).copy()
    t["block"] = _block(t, block_ms)
    t["notional"] = t.px * t.sz
    rows = []
    for w, g in t.groupby("maker", sort=False):
        if len(g) < n_min:
            continue
        x = g[mk].to_numpy()
        rows.append({"wallet": w, "n_fills": int(len(g)), "volume": float(g.notional.sum()),
                     "markout_bps": float(x.mean()), "se_bps": cluster_se(x, g.block.to_numpy()),
                     "touch_share": float(g.at_touch.mean()), "effective_bps": float(g.effective_bps.mean())})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fill_share"] = out.n_fills / len(t)
    return out.sort_values("volume", ascending=False).reset_index(drop=True)


def _group_stats(t: pd.DataFrame, mask: np.ndarray, mk: str) -> dict:
    g = t[mask]
    if len(g) == 0:
        return {"n": 0, "fill_share": 0.0, "markout_bps": np.nan, "se_bps": np.nan, "touch_share": np.nan}
    x = g[mk].to_numpy()
    return {"n": int(len(g)), "fill_share": float(len(g) / len(t)), "markout_bps": float(x.mean()),
            "se_bps": cluster_se(x, g.block.to_numpy()), "touch_share": float(g.at_touch.mean())}


def maker_summary(tm: pd.DataFrame, feats: pd.DataFrame, horizon_s: int = 10, top_k: int = 10,
                  n_min: int = 100, n_half: int = 50, block_ms: int = BLOCK_MS) -> dict:
    mk = f"markout_{horizon_s}_bps"
    t = tm.dropna(subset=[mk]).copy()
    if t.empty:
        return {}
    t["block"] = _block(t, block_ms)
    out = {"mk_n_fills": int(len(t))}
    allx = t[mk].to_numpy()
    out["mk_all_markout_bps"] = float(allx.mean())
    out["mk_all_se_bps"] = cluster_se(allx, t.block.to_numpy())
    # fixed sets from the full-window classifier
    if feats is not None and len(feats) and "is_mm" in feats:
        mm_all = set(feats[feats.is_mm].wallet)
        top = list(feats[feats.is_mm].sort_values("volume", ascending=False).wallet.head(top_k))
        is_top = t.maker.isin(top).to_numpy()
        is_mm = t.maker.isin(mm_all).to_numpy()
        for name, mask in (("top", is_top), ("othermm", is_mm & ~is_top), ("nonmm", ~is_mm)):
            s = _group_stats(t, mask, mk)
            out.update({f"mk_{name}_n": s["n"], f"mk_{name}_fill_share": s["fill_share"],
                        f"mk_{name}_markout_bps": s["markout_bps"], f"mk_{name}_se_bps": s["se_bps"],
                        f"mk_{name}_touch_share": s["touch_share"]})
        out["mk_top_k"] = len(top)
        # top makers' fills at the touch only, against everyone else's fills at the touch:
        # removes the "they just show less depth behind the inside" explanation
        at = t.at_touch.to_numpy(dtype=bool)
        s_top = _group_stats(t, is_top & at, mk); s_rest = _group_stats(t, ~is_top & at, mk)
        out["mk_top_touch_markout_bps"] = s_top["markout_bps"]; out["mk_top_touch_se_bps"] = s_top["se_bps"]
        out["mk_rest_touch_markout_bps"] = s_rest["markout_bps"]; out["mk_rest_touch_se_bps"] = s_rest["se_bps"]
    # dispersion across makers
    table = maker_table(t, horizon_s=horizon_s, n_min=n_min, block_ms=block_ms)
    out["mk_n_makers"] = int(len(table))
    if len(table):
        pos = table[table.markout_bps > 0]
        sig = table[table.markout_bps > 1.96 * table.se_bps]
        out["mk_positive_share"] = float(len(pos) / len(table))
        out["mk_positive_fill_share"] = float(pos.fill_share.sum())
        out["mk_sig_positive_n"] = int(len(sig))
        out["mk_sig_positive_fill_share"] = float(sig.fill_share.sum())
        out["mk_best_markout_bps"] = float(table.markout_bps.max())
        out["mk_best_n_fills"] = int(table.loc[table.markout_bps.idxmax(), "n_fills"])
        out["mk_markout_sd_bps"] = float(table.markout_bps.std()) if len(table) > 1 else np.nan
    # persistence: first half vs second half (median cut), and, when the marked fills come from
    # more than one recording session, first session vs the rest (cut at the largest gap > 1 h) --
    # the across-nights version, which cannot be a single calm spell
    out.update(_persistence(t, mk, n_half, cut=t.time_ms.iloc[len(t) // 2], prefix="mk_split"))
    gaps = np.diff(t.time_ms.to_numpy())
    if len(gaps) and gaps.max() > 3_600_000:
        out.update(_persistence(t, mk, n_half, cut=t.time_ms.iloc[int(gaps.argmax()) + 1], prefix="mk_session"))
    return out


def _persistence(t: pd.DataFrame, mk: str, n_half: int, cut: int, prefix: str) -> dict:
    a = t[t.time_ms < cut].groupby("maker")[mk].agg(["mean", "size"])
    b = t[t.time_ms >= cut].groupby("maker")[mk].agg(["mean", "size"])
    both = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    both = both[(both.size_a >= n_half) & (both.size_b >= n_half)]
    out = {f"{prefix}_n_makers": int(len(both))}
    if len(both) >= 8:
        from scipy.stats import spearmanr
        r, p = spearmanr(both.mean_a, both.mean_b)
        out[f"{prefix}_rank_corr"], out[f"{prefix}_rank_corr_p"] = float(r), float(p)
    else:
        out[f"{prefix}_rank_corr"], out[f"{prefix}_rank_corr_p"] = np.nan, np.nan
    return out
