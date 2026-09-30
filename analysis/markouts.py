"""Markout / spread decomposition (paper Section 1, 'Payoffs').

For a fill at price p with taker sign s (+1 taker bought, -1 taker sold),
mid m_t at the fill and m_{t+d} d periods later:

    effective_spread = 2 s (p - m_t)
    realized_spread  = 2 s (p - m_{t+d})       what the MM kept
    price_impact     = 2 s (m_{t+d} - m_t)     adverse selection cost
    effective = realized + impact

Works on simulator tapes (tape_ep*.csv) and, with a column rename, on
Hyperliquid fills.  Under competition E[realized] ~ 0; persistent
E[realized] > 0 is rent (RQ1 signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def markouts(tape: pd.DataFrame, horizons=(1, 10, 60), price_col="price",
             mid_col="mid", event_col="event") -> pd.DataFrame:
    df = tape.reset_index(drop=True).copy()
    fills = df[df[event_col].isin(["buy", "sell"])].copy()
    s = np.where(fills[event_col] == "buy", 1.0, -1.0)
    fills["effective_spread"] = 2 * s * (fills[price_col] - fills[mid_col])
    mids = df[mid_col].to_numpy()
    for d in horizons:
        idx = np.minimum(fills.index.to_numpy() + d, len(df) - 1)
        m_fut = mids[idx]
        fills[f"realized_spread_{d}"] = 2 * s * (fills[price_col] - m_fut)
        fills[f"price_impact_{d}"] = 2 * s * (m_fut - fills[mid_col])
    return fills


def summarize(fills: pd.DataFrame) -> pd.Series:
    cols = [c for c in fills.columns if c.startswith(("effective", "realized", "price_impact"))]
    return fills[cols].mean()
