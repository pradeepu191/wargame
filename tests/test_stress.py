"""Withdrawal in stress: the block table and the sign of the top makers' presence against lagged stress."""
import numpy as np
import pandas as pd

from analysis.stress import block_table, stress_summary


def regime_fills(seed=0, withdraw=True, n_blocks=60, per_block=60):
    """Alternating calm / stress 5-minute blocks.  In a stress block the mid moves 4x as much and
    the rest of the market loses 1 bp at the touch.  With withdraw=True the top makers take 60% of
    touch fills in the block AFTER a calm one and 15% after a stressful one (they read the tape
    with a one-block lag); with withdraw=False their share is 40% throughout."""
    rng = np.random.default_rng(seed)
    t0 = 1_760_000_100_000 // 300_000 * 300_000  # block-aligned
    rows = []
    stress = np.array([(b // 3) % 2 == 1 for b in range(n_blocks)])  # runs of three
    for b in range(n_blocks):
        prev_stress = stress[b - 1] if b > 0 else False
        share = (0.15 if prev_stress else 0.6) if withdraw else 0.4
        vol = 4.0 if stress[b] else 1.0
        for _ in range(per_block):
            is_top = rng.random() < share
            maker = f"top{rng.integers(0, 10)}" if is_top else f"rest{rng.integers(0, 40)}"
            mu = 0.3 if is_top else (-1.0 if stress[b] else -0.2)
            move = rng.normal(0, vol)
            rows.append({"time_ms": t0 + b * 300_000 + int(rng.integers(0, 300_000)), "maker": maker, "taker": "tk",
                         "px": 100.0, "sz": 1.0, "maker_sold": True, "at_touch": True, "effective_bps": 1.0,
                         "markout_10_bps": mu + rng.normal(0, 0.3), "impact_10_bps": 2 * move})
    tm = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    feats = pd.DataFrame({"wallet": [f"top{i}" for i in range(10)] + [f"rest{i}" for i in range(40)],
                          "is_mm": [True] * 10 + [True] * 10 + [False] * 30,
                          "volume": [1e7 - i for i in range(10)] + [1e5] * 40})
    return tm, feats


def test_block_table_lags_only_contiguous_blocks():
    tm, feats = regime_fills()
    top = [f"top{i}" for i in range(10)]
    bt = block_table(tm, top)
    assert len(bt) == 60 and bt.n_touch.sum() == len(tm)
    assert np.isnan(bt.vol_lag.iloc[0]) and np.isfinite(bt.vol_lag.iloc[1:]).all()
    # a gap in the recording breaks the lag chain
    tm2 = tm.copy(); tm2.loc[tm2.time_ms >= tm.time_ms.iloc[0] + 30 * 300_000, "time_ms"] += 7_200_000
    bt2 = block_table(tm2, top)
    assert bt2.vol_lag.isna().sum() == 2


def test_withdrawal_has_a_negative_sign_against_lagged_stress():
    tm, feats = regime_fills(withdraw=True)
    s = stress_summary(tm, feats)
    assert s["st_n_blocks"] == 60
    assert s["st_top_share_vs_vol_lag"] < -0.5 and s["st_top_share_vs_vol_lag_p"] < 1e-3
    assert s["st_n_top_active_vs_vol_lag"] < -0.3
    assert s["st_stress_top_share"] < s["st_calm_top_share"]
    # the rest loses more in stress, the top makers do not
    assert s["st_stress_rest_markout_bps"] < s["st_calm_rest_markout_bps"] - 0.2
    assert abs(s["st_stress_top_markout_bps"] - s["st_calm_top_markout_bps"]) < 0.3


def test_no_withdrawal_is_flat():
    tm, feats = regime_fills(withdraw=False)
    s = stress_summary(tm, feats)
    assert abs(s["st_top_share_vs_vol_lag"]) < 0.35
    assert abs(s["st_stress_top_share"] - s["st_calm_top_share"]) < 0.12


def test_fill_level_stress_sees_a_reaction_to_the_last_ten_seconds():
    """A random-walk mid with a 20 s burst of volatility each minute.  Top makers take 10% of the
    touch fills that arrive when the mid moved more than 3 bps over the previous 10 s and 50%
    otherwise.  The 10 s look-back recovers the rule; a 60 s look-back, which sees a burst behind
    almost every fill, is much weaker."""
    from analysis.stress import fill_level_stress
    rng = np.random.default_rng(3)
    t0 = 1_760_000_100_000 // 300_000 * 300_000
    secs = np.arange(3600)
    sigma = np.where((secs % 60) < 20, 2.0, 0.3)  # bps per second
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma) * 1e-4))
    book = pd.DataFrame({"time_ms": t0 + secs * 1000, "bid": mid * (1 - 5e-5), "ask": mid * (1 + 5e-5), "mid": mid})
    rows = []
    for s in secs[10:]:
        move10 = abs(mid[s] - mid[s - 10]) / mid[s] * 1e4
        for _ in range(3):
            is_top = rng.random() < (0.1 if move10 > 3.0 else 0.5)
            rows.append({"time_ms": t0 + s * 1000 + 500, "maker": f"top{rng.integers(0, 10)}" if is_top else f"rest{rng.integers(0, 40)}",
                         "px": mid[s], "mid_0": mid[s], "at_touch": True, "markout_10_bps": (0.3 if is_top else -0.4) + rng.normal(0, 0.2)})
    tm = pd.DataFrame(rows)
    out = fill_level_stress(tm, book, [f"top{i}" for i in range(10)], windows_s=(10, 60))
    assert out["fl10_top_vs_move_p"] < 1e-6 and out["fl10_top_vs_move"] < -0.15
    assert out["fl10_stress_top_share"] < out["fl10_calm_top_share"] - 0.15
    assert abs(out["fl60_top_vs_move"]) < 0.5 * abs(out["fl10_top_vs_move"])
    # the rest loses in every tercile, the top makers earn in every tercile
    assert out["fl10_stress_rest_markout_bps"] < 0 < out["fl10_stress_top_markout_bps"]
