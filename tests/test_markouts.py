import pandas as pd
from analysis.markouts import markouts


def test_decomposition_identity():
    tape = pd.DataFrame({
        "event": ["buy", "none", "sell", "none", "none"],
        "price": [100.1, float("nan"), 99.9, float("nan"), float("nan")],
        "mid":   [100.0, 100.4, 100.4, 100.2, 100.2],
    })
    f = markouts(tape, horizons=(1, 2))
    for d in (1, 2):
        assert ((f["effective_spread"] - f[f"realized_spread_{d}"] - f[f"price_impact_{d}"]).abs() < 1e-12).all()
    # first fill: MM sold at 100.1, mid then rose to 100.4 -> negative realized spread
    assert f.iloc[0]["realized_spread_1"] < 0
