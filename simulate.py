"""Simulate a daily event table with realistic seasonality and optional incidents.

The table looks like a streaming service's playback events, split by platform.
Incidents you can inject (each mirrors a real-world failure mode):
  * platform_drop  - one platform loses most of its rows (a broken SDK release)
  * null_spike     - user_id goes null for one platform (an identity bug)
  * late_load      - a partition lands hours late (an upstream job stalled)

Usage:
    python simulate.py --db events.duckdb --days 90 --incident platform_drop
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

PLATFORMS = {"roku": 0.30, "ios": 0.22, "android": 0.20, "web": 0.16, "smart_tv": 0.12}
WEEKLY = np.array([0.92, 0.90, 0.93, 0.97, 1.05, 1.13, 1.10])  # Mon..Sun
BASE_ROWS = 2_000  # aggregated rows per day; each row stands for many raw events


def simulate(days: int = 90, end: date = date(2026, 10, 20), seed: int = 1,
             incident: str | None = None, incident_day: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    incident_day = incident_day or end
    rows = []
    for i in range(days):
        d = end - timedelta(days=days - 1 - i)
        trend = 1 + 0.002 * i
        load_ts = datetime.combine(d + timedelta(days=1), datetime.min.time()) + timedelta(hours=2, minutes=int(rng.integers(0, 30)))
        if incident == "late_load" and d == incident_day:
            load_ts += timedelta(hours=9)
        for platform, share in PLATFORMS.items():
            n = int(BASE_ROWS * share * WEEKLY[d.weekday()] * trend * rng.normal(1, 0.02))
            if incident == "platform_drop" and d == incident_day and platform == "roku":
                n = int(n * 0.3)
            null_rate = 0.004
            if incident == "null_spike" and d == incident_day and platform == "android":
                null_rate = 0.35
            user_null = rng.random(n) < null_rate
            rows.append(pd.DataFrame({
                "event_date": d,
                "platform": platform,
                "user_id": np.where(user_null, None, [f"u{x}" for x in rng.integers(0, 50_000, n)]),
                "watch_seconds": rng.gamma(2, 600, n).round(),
                "loaded_at": load_ts,
            }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="events.duckdb")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--incident", choices=["platform_drop", "null_spike", "late_load"])
    a = ap.parse_args()
    df = simulate(a.days, incident=a.incident)
    con = duckdb.connect(a.db)
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE playback_events AS SELECT * FROM df")
    print(f"playback_events: {len(df):,} rows over {a.days} days, incident={a.incident}")


if __name__ == "__main__":
    main()
