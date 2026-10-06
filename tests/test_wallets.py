"""Ground-truth tests for the wallet classifier on a synthetic identity-tagged tape."""
import numpy as np
import pandas as pd

from analysis.wallets import leader_switch, pair_matrix, wallet_features


def synthetic_tape(n=6000, seed=0, avoid=True):
    """Two turn-taking MMs (mmA, mmB), 1 informed taker, 20 noise takers.
    Price is a random walk; informed trades move the price in their direction next step."""
    rng = np.random.default_rng(seed)
    px = 100.0
    rows = []
    t = 0
    for k in range(n):
        t += int(rng.exponential(200))                  # ms between trades
        leader = "mmA" if (k // 50) % 2 == 0 else "mmB"   # turn-taking in 50-trade blocks
        other = "mmB" if leader == "mmA" else "mmA"
        informed = rng.random() < 0.2
        if informed:
            taker, drift = "informed", rng.choice([-1, 1])
            side = "buy" if drift > 0 else "sell"
        else:
            taker, drift = f"noise{rng.integers(20)}", 0
            side = rng.choice(["buy", "sell"])
        # MMs also take sometimes (e.g. hedging).  With avoid=True they take only from
        # non-MM makers; with avoid=False they also hit each other.
        maker = leader
        if rng.random() < 0.1:
            taker = other
            if avoid:
                maker = f"noise{rng.integers(20)}"      # a resting retail order, not the other MM
        rows.append({"time_ms": t, "coin": "X", "side": side, "px": px, "sz": 1.0,
                     "buyer": taker if side == "buy" else maker,
                     "seller": maker if side == "buy" else taker})
        px += 0.05 * drift + rng.normal(0, 0.01)
    return pd.DataFrame(rows)


def test_classifier_finds_market_makers():
    f = wallet_features(synthetic_tape(), horizons_s=(1, 10), n_min=30)
    mms = set(f[f.is_mm].wallet)
    assert mms == {"mmA", "mmB"}
    # informed taker is never a maker; noise takers are not two-sided makers
    assert "informed" not in mms


def test_makers_are_adversely_selected_by_informed_flow():
    tape = synthetic_tape()
    f = wallet_features(tape, horizons_s=(10,), n_min=30).set_index("wallet")
    # markout after MM fills is negative on average (20% informed flow moves against them)
    assert f.loc["mmA", "markout_10"] < 0 and f.loc["mmB", "markout_10"] < 0


def test_pair_matrix_detects_avoidance():
    _, ratio_avoid = pair_matrix(synthetic_tape(avoid=True), {"mmA", "mmB"})
    _, ratio_mix = pair_matrix(synthetic_tape(avoid=False), {"mmA", "mmB"})
    assert ratio_avoid == 0.0
    assert ratio_mix > 0.0


def test_leader_switch_detects_turn_taking():
    tape = synthetic_tape()
    # 50-trade blocks at ~200ms spacing = ~10s per turn; a 10s window should switch often
    ls = leader_switch(tape, {"mmA", "mmB"}, window_ms=10_000)
    assert 0.3 < ls < 1.0


def test_arrival_persistence_and_baseline():
    from analysis.wallets import activity_herfindahl, arrival_persistence
    tape = synthetic_tape()
    rho = arrival_persistence(tape)
    base = activity_herfindahl(tape)
    assert 0 <= rho <= 1 and 0 <= base <= 1
    # informed taker is 20% of flow, 20 noise takers share the rest: random matching baseline
    # ~ 0.2^2 + 20 * (0.8/20 * 0.9)^2 ~ 0.07; the synthetic tape has no burstiness, so rho ~ base
    assert abs(rho - base) < 0.03
    # a bursty tape: repeat every taker 5 times in a row
    bursty = tape.loc[np.repeat(tape.index, 5)].reset_index(drop=True)
    bursty["time_ms"] = np.arange(len(bursty)) * 100
    assert arrival_persistence(bursty) > 0.75
