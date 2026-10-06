"""The calibration pipeline on a synthetic identity-tagged tape with known structure."""
import numpy as np
import pandas as pd

from analysis.calibrate import calibrate_coin
from analysis.wallets import collapse_orders, conditional_markout_after_toxic, split_half_type_persistence
from tests.test_wallets import synthetic_tape


def bursty_tape(n=8000, seed=0, rho=0.8, n_takers=20):
    """Takers with persistent continuous types (P(informed) from 0.02 to 0.6; informed orders
    move the price), bursty arrivals (same taker returns w.p. rho), two turn-taking MMs; each
    order prints as 1-3 fills at the same time."""
    rng = np.random.default_rng(seed)
    types = np.linspace(0.02, 0.6, n_takers)
    px, t, rows, cur = 100.0, 0, [], -1
    for k in range(n):
        t += int(rng.exponential(300))
        cur = cur if (cur >= 0 and rng.random() < rho) else int(rng.integers(n_takers))
        informed = rng.random() < types[cur]
        drift = rng.choice([-1, 1]) if informed else 0
        side = ("buy" if drift > 0 else "sell") if informed else rng.choice(["buy", "sell"])
        maker = "mmA" if (k // 50) % 2 == 0 else "mmB"
        for f in range(int(rng.integers(1, 4))):                 # one order, several fills
            rows.append({"time_ms": t, "coin": "X", "side": side, "px": px, "sz": 1.0, "tid": len(rows),
                         "hash": f"h{k}", "buyer": f"t{cur}" if side == "buy" else maker,
                         "seller": maker if side == "buy" else f"t{cur}"})
        px += 0.05 * drift + rng.normal(0, 0.005)
    return pd.DataFrame(rows), types


def test_collapse_orders_merges_fills_of_one_sweep():
    tape, _ = bursty_tape(n=500)
    orders = collapse_orders(tape)
    assert len(orders) == 500
    assert orders.n_fills.max() == 3 and orders.sz.sum() == len(tape)


def test_order_level_persistence_recovers_rho_and_fill_level_overstates_it():
    tape, _ = bursty_tape(rho=0.8)
    row, _ = calibrate_coin(tape, n_min=20)
    assert abs(row["rho_order"] - (0.8 + 0.2 / 20)) < 0.04
    assert row["rho_fill"] > row["rho_order"]                   # sweeps inflate the fill-level number
    assert row["rho_order_baseline"] < 0.1
    tape0, _ = bursty_tape(rho=0.0)
    row0, _ = calibrate_coin(tape0, n_min=20)
    assert abs(row0["rho_order"] - row0["rho_order_baseline"]) < 0.03


def test_type_persistence_and_conditional_markout_signature():
    # the synthetic impact is immediate, so the 1 s horizon is the clean one here
    tape, types = bursty_tape(rho=0.8)
    tp = split_half_type_persistence(tape, horizon_s=1, n_min=20)
    assert tp["rank_corr"] > 0.7                               # toxicity is a persistent type
    cm = conditional_markout_after_toxic(tape, horizon_s=1, n_min=20)
    assert cm["diff_bps"] < -1.0                               # after a toxic order the next fill is worse
    tape0, _ = bursty_tape(rho=0.0, seed=1)
    cm0 = conditional_markout_after_toxic(tape0, horizon_s=1, n_min=20)
    assert abs(cm0["diff_bps"]) < 0.5                          # and the signature vanishes without persistence


def test_pipeline_runs_on_the_mm_tape():
    row, feats = calibrate_coin(synthetic_tape(), n_min=30)
    assert row["n_mm"] == 2 and row["mm_avoidance_ratio"] == 0.0
    assert {"rho_order", "markout_10_bps", "type_rank_corr", "cond_diff_bps"} <= set(row)
