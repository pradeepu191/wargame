"""The long-window cuts: a market where the top makers avoid toxic takers on every day and in
every hour shows the gap in every cut, with labels that stay predictive day after day."""
import numpy as np
import pandas as pd

from analysis.longwindow import by_hour_and_day, maker_persistence, rolling_typing

DAY_MS = 86_400_000


def selecting_market(seed=0, n_days=5, per_hour=120, select=True):
    """n_days of fills, every UTC hour.  40 takers with fixed types (toxic ones inflict -1 bp,
    benign +0.3); with select=True the top makers take 50% of benign takers' fills and 10% of
    toxic ones, otherwise 30% of both.  The first day is separated by a 2 h gap so the
    first-session split lands there."""
    rng = np.random.default_rng(seed)
    t0 = 1_760_000_000_000 // DAY_MS * DAY_MS
    takers = [f"tk{i}" for i in range(40)]
    tox = {tk: (i % 3 == 2) for i, tk in enumerate(takers)}       # a third toxic
    ben = {tk: (i % 3 == 0) for i, tk in enumerate(takers)}
    rows = []
    for d in range(n_days):
        for h in range(24):
            if d == 0 and h in (22, 23):
                continue                                           # the gap after day 0
            for _ in range(per_hour):
                tk = takers[rng.integers(0, 40)]
                if select:
                    share = 0.1 if tox[tk] else 0.5 if ben[tk] else 0.3
                else:
                    share = 0.3
                is_top = rng.random() < share
                maker = f"top{rng.integers(0, 10)}" if is_top else f"rest{rng.integers(0, 30)}"
                skill = 0.4 if is_top else 0.0
                mu = (-1.0 if tox[tk] else 0.3 if ben[tk] else -0.2) + skill
                rows.append({"time_ms": t0 + d * DAY_MS + h * 3_600_000 + int(rng.integers(0, 3_600_000)),
                             "maker": maker, "taker": tk, "at_touch": True, "mid_0": 100.0,
                             "markout_10_bps": mu + rng.normal(0, 0.3), "impact_1_bps": 0.0})
    return pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)


TOP = [f"top{i}" for i in range(10)]


def test_gap_appears_in_every_hour_and_every_day_when_makers_select():
    tm = selecting_market()
    hours, days = by_hour_and_day(tm, TOP)
    h, d = pd.DataFrame(hours), pd.DataFrame(days)
    assert len(h) == 24 and (h.gap > 0.25).all(), h.gap.describe()
    assert len(d) == 4 and (d.gap > 2 * d.gap_se).all(), d[["gap", "gap_se"]]


def test_no_selection_means_no_gap():
    tm = selecting_market(select=False)
    hours, days = by_hour_and_day(tm, TOP)
    d = pd.DataFrame(days)
    assert (d.gap.abs() < 3 * d.gap_se).all(), d[["gap", "gap_se"]]


def test_rolling_typing_holds_at_every_horizon():
    tm = selecting_market()
    r = pd.DataFrame(rolling_typing(tm, TOP, max_k=4))
    assert set(r.k) == {1, 2, 3, 4}
    assert (r.gap > 0.25).all() and (r.type_holds_bps > 0.8).all(), r


def test_maker_ranking_persists_when_skill_is_fixed():
    rng = np.random.default_rng(1)
    tm = selecting_market()
    # give each maker a fixed skill so the ranking is real
    skill = {m: rng.normal(0, 0.5) for m in tm.maker.unique()}
    tm["markout_10_bps"] = tm["markout_10_bps"] + tm.maker.map(skill)
    per, corr = maker_persistence(tm, n_min=50)
    c = pd.DataFrame(corr)
    assert (c.rho_prev_day.dropna() > 0.6).all(), c
