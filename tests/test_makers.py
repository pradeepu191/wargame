"""Per-maker realized spread on marked fills: fixed top set, cluster SE, dispersion, persistence."""
import numpy as np
import pandas as pd

from analysis.makers import cluster_se, maker_summary, maker_table


def marked_fills(n_per_maker=400, seed=0):
    """Three makers: 'sel' selects (markout +0.5 bps, persistent), 'avg' is the average fill
    (-0.4), 'swept' sits behind the inside and gets walked (-1.0, never at the touch)."""
    rng = np.random.default_rng(seed)
    t0 = 1_760_000_000_000
    rows = []
    for name, mu, touch in (("sel", 0.5, 1.0), ("avg", -0.4, 0.8), ("swept", -1.0, 0.0)):
        times = np.sort(rng.integers(0, 6 * 3600_000, n_per_maker)) + t0
        for tm in times:
            rows.append({"time_ms": int(tm), "maker": name, "taker": "tk", "px": 100.0, "sz": 1.0,
                         "maker_sold": bool(rng.integers(0, 2)), "at_touch": bool(rng.random() < touch),
                         "effective_bps": 1.0, "markout_10_bps": mu + rng.normal(0, 0.3)})
    tm = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    feats = pd.DataFrame({"wallet": ["sel", "avg", "swept", "retail"], "is_mm": [True, True, False, False],
                          "volume": [5e6, 9e6, 1e6, 1e5]})
    return tm, feats


def test_cluster_se_matches_iid_when_blocks_are_singletons():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    se_iid = x.std(ddof=0) / np.sqrt(len(x))
    assert abs(cluster_se(x, np.arange(len(x))) - se_iid) < 1e-9
    # perfectly dependent blocks of 10 identical values: SE grows by ~sqrt(10)
    y = np.repeat(rng.normal(0, 1, 200), 10)
    assert cluster_se(y, np.repeat(np.arange(200), 10)) > 2.5 * (y.std() / np.sqrt(len(y)))


def test_maker_summary_finds_the_selector():
    tm, feats = marked_fills()
    s = maker_summary(tm, feats, horizon_s=10, top_k=1, n_min=100, n_half=50)
    # top-1 MM by full-window volume is 'avg' (9e6), not the best-performing maker: the set is
    # chosen on activity, and its markout is the average fill's
    assert s["mk_top_k"] == 1 and abs(s["mk_top_markout_bps"] + 0.4) < 0.1
    assert abs(s["mk_othermm_markout_bps"] - 0.5) < 0.1        # 'sel'
    assert abs(s["mk_nonmm_markout_bps"] + 1.0) < 0.1          # 'swept'
    assert abs(s["mk_all_fill_share"] - 1.0) < 1e-9 if "mk_all_fill_share" in s else True
    assert s["mk_n_makers"] == 3 and abs(s["mk_positive_share"] - 1 / 3) < 1e-9
    assert s["mk_sig_positive_n"] == 1 and abs(s["mk_positive_fill_share"] - 1 / 3) < 1e-9
    assert s["mk_split_n_makers"] == 3 and np.isnan(s["mk_split_rank_corr"])   # < 8 makers: no test
    # persistence needs a population: ten makers with distinct persistent means rank identically in both halves
    rng = np.random.default_rng(2)
    t0 = 1_760_000_000_000
    rows = []
    for k in range(10):
        for tm_ in np.sort(rng.integers(0, 6 * 3600_000, 300)) + t0:
            rows.append({"time_ms": int(tm_), "maker": f"m{k}", "taker": "tk", "px": 100.0, "sz": 1.0,
                         "maker_sold": True, "at_touch": True, "effective_bps": 1.0,
                         "markout_10_bps": -1 + 0.2 * k + rng.normal(0, 0.05)})
    pop = pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)
    s2 = maker_summary(pop, feats.iloc[:0], horizon_s=10, n_min=100, n_half=50)
    assert s2["mk_split_n_makers"] == 10 and abs(s2["mk_split_rank_corr"] - 1.0) < 1e-9 and s2["mk_split_rank_corr_p"] < 1e-6
    assert "mk_top_markout_bps" not in s2                       # no classifier table, no fixed sets
    tab = maker_table(tm, horizon_s=10, n_min=100)
    assert set(tab.wallet) == {"avg", "sel", "swept"} and (tab.volume.diff().dropna() <= 0).all()   # volume order
    assert tab.set_index("wallet").loc["swept", "touch_share"] == 0.0
    assert (tab.se_bps > 0).all() and (tab.se_bps < 0.1).all()
