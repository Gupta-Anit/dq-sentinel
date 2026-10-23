from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

import sentinel as s
from simulate import simulate

DAY = date(2026, 10, 20)


def _con(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("df", df)
    con.execute("CREATE TABLE playback_events AS SELECT * FROM df")
    return con


def _check(con):
    return s.check(con, "playback_events", "event_date", ["user_id"], dim="platform", loaded_col="loaded_at")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_clean_data_raises_no_alerts(seed):
    assert _check(_con(simulate(days=60, seed=seed))) == []


def test_platform_drop_is_detected_and_attributed():
    anomalies = _check(_con(simulate(days=60, incident="platform_drop")))
    vol = [a for a in anomalies if a.metric == "row_count"]
    assert len(vol) == 1
    assert vol[0].pct_change < -0.15
    assert vol[0].causes[0]["value"] == "roku"
    assert vol[0].causes[0]["share_of_change"] > 0.8


def test_null_spike_is_detected_and_attributed():
    anomalies = _check(_con(simulate(days=60, incident="null_spike")))
    nulls = [a for a in anomalies if a.metric == "null_rate:user_id"]
    assert len(nulls) == 1
    assert nulls[0].causes[0]["value"] == "android"


def test_late_load_breaks_freshness_sla():
    anomalies = _check(_con(simulate(days=60, incident="late_load")))
    assert [a.metric for a in anomalies] == ["load_lag_hours"]
    assert anomalies[0].value > s.MAX_LOAD_LAG_HOURS


def test_baseline_ignores_a_past_incident():
    # An old incident four weeks back sits in the baseline window; the median keeps it from
    # dragging the baseline down, so today's normal volume does not look like a spike.
    df = simulate(days=60, incident="platform_drop", incident_day=date(2026, 9, 22))
    assert [a for a in _check(_con(df)) if a.metric == "row_count"] == []


def test_flat_history_does_not_divide_by_zero():
    series = pd.Series([100.0] * 5, index=pd.date_range("2026-09-01", periods=5, freq="7D").date)
    med, scale = s.robust_baseline(series, date(2026, 9, 29))
    assert med == 100 and scale > 0


def test_alert_payload_is_slack_shaped():
    anomalies = _check(_con(simulate(days=60, incident="platform_drop")))
    payload = s.alert_payload("playback_events", anomalies)
    assert payload["blocks"][0]["type"] == "header"
    assert "roku" in payload["blocks"][1]["text"]["text"]
