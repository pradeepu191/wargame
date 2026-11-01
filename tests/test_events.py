"""Event study: an hour-aligned window around a dated event, with the toxic surge visible."""
import pandas as pd

from analysis.events import event_table
from tests.test_longwindow import TOP, selecting_market, DAY_MS


def test_event_table_is_hour_aligned_and_sees_a_toxic_surge():
    tm = selecting_market(n_days=4)
    # day 2, hours 10-12: the toxic takers triple their activity
    t0 = tm.time_ms.min() // DAY_MS * DAY_MS
    surge = tm[(tm.time_ms >= t0 + 2 * DAY_MS + 10 * 3_600_000) & (tm.time_ms < t0 + 2 * DAY_MS + 13 * 3_600_000)]
    surge = surge[surge.taker.str.replace("tk", "").astype(int) % 3 == 2]
    tm = pd.concat([tm, surge, surge]).sort_values("time_ms").reset_index(drop=True)
    ev = pd.DataFrame({"time_utc": [pd.Timestamp(t0 + 2 * DAY_MS + 10 * 3_600_000 + 1234, unit="ms", tz="UTC").isoformat()],
                       "label": ["surge"], "source": [""]})
    rows = pd.DataFrame(event_table(tm, TOP, ev, before_h=6, after_h=6))
    assert sorted(rows.k_hours.unique()) == list(range(-6, 7))
    pre, post = rows[rows.k_hours < 0], rows[rows.k_hours.between(0, 2)]
    assert post.toxic_fill_share.mean() > pre.toxic_fill_share.mean() + 0.15
    assert (rows.gap > 0.2).all()
