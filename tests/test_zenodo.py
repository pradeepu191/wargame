"""Converter for the Zenodo trades archive, on a synthetic tar in the documented layout."""
import gzip
import io
import json
import tarfile

import numpy as np
import pandas as pd

from data.zenodo_to_parquet import parse_hour
from analysis.calibrate import load_trades, calibrate_coin


def make_hour(rng, n=400, coins=("BTC", "ETH", "DOGE"), date="2026-01-05", hour=3, oid0=1000):
    """Synthetic hour of trades in the archive's JSON schema.  The resting order always has the
    smaller oid; side_info order is randomised so the converter has to use the oid rule."""
    lines, truth = [], []
    oid = oid0
    for k in range(n):
        coin = coins[k % len(coins)]
        side = "B" if rng.random() < 0.5 else "A"
        step = 3_600_000 // n
        t_ms = k * step + int(rng.integers(0, step))
        ts = f"{date}T{hour:02d}:{t_ms // 60000:02d}:{(t_ms // 1000) % 60:02d}.{(t_ms % 1000):03d}000000"
        maker = {"user": f"0xmaker{rng.integers(3)}", "start_pos": str(rng.normal()), "oid": oid, "twap_id": None, "cloid": None}
        taker = {"user": f"0xtaker{rng.integers(10)}", "start_pos": "0.0", "oid": oid + 1,
                 "twap_id": int(rng.integers(1, 9)) if rng.random() < 0.2 else None, "cloid": None}
        oid += 2
        si = [maker, taker] if rng.random() < 0.5 else [taker, maker]
        lines.append(json.dumps({"coin": coin, "side": side, "time": ts, "px": "100.5", "sz": "0.1",
                                 "hash": "0x" + "0" * 64 if rng.random() < 0.3 else "0xabc",
                                 "trade_dir_override": "Na", "side_info": si}))
        truth.append((coin, side, taker["user"], maker["user"], taker["twap_id"] is not None))
    return gzip.compress(("\n".join(lines) + "\n").encode()), truth


def test_parse_hour_assigns_taker_by_oid_and_filters_coins():
    rng = np.random.default_rng(0)
    raw, truth = make_hour(rng)
    df, diag = parse_hour(raw, {"BTC", "ETH"}, tid0=0)
    assert set(df.coin) == {"BTC", "ETH"} and diag["n_lines"] == 400
    kept = [t for t in truth if t[0] in ("BTC", "ETH")]
    assert len(df) == len(kept)
    for row, (coin, side, taker, maker, twap) in zip(df.itertuples(), kept):
        assert row.coin == coin and row.side == side
        buyer, seller = (taker, maker) if side == "B" else (maker, taker)
        assert row.buyer == buyer and row.seller == seller
        assert row.taker_twap == twap and row.taker_oid == row.maker_oid + 1
    assert 0.3 < diag["idx0_is_buyer_share"] < 0.7          # side_info order was random in the synthetic data
    assert df.time_ms.is_monotonic_increasing
    assert abs(df.same_block.mean() - 0.3) < 0.1
    assert pd.Timestamp(df.time_ms.iloc[0], unit="ms", tz="UTC").hour == 3


def test_converted_parquet_feeds_the_calibration_pipeline(tmp_path):
    rng = np.random.default_rng(1)
    raw, _ = make_hour(rng, n=3000, coins=("BTC",))
    df, _ = parse_hour(raw, {"BTC"}, tid0=0)
    df.to_parquet(tmp_path / "trades_BTC_2026010503.parquet", index=False)
    tr = load_trades(tmp_path, "BTC")
    row, feats = calibrate_coin(tr, n_min=20)
    assert row["prints"] == 3000 and row["orders"] == 3000
    assert "rho_order" in row and feats.is_mm.sum() >= 1   # the three resting wallets are two-sided makers
