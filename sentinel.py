"""dq-sentinel: data quality anomaly detection with root-cause ranking.

For a daily-partitioned table it:
  1. Profiles each day: row count, null rate per column, and load lag (freshness).
  2. Compares each metric with a same-weekday baseline from the previous N weeks,
     using median and MAD (robust to past incidents still sitting in the history).
  3. When volume moves, ranks the dimension values (e.g. platform) that explain
     the change, so the alert says *where* to look, not just *that* something broke.
  4. Emits an alert payload ready to post to Slack.

Usage:
    python sentinel.py --db events.duckdb --table playback_events \
        --date-col event_date --dim platform --columns user_id --loaded-col loaded_at
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

Z_THRESHOLD = 4.0
MIN_RELATIVE_CHANGE = 0.10
BASELINE_WEEKS = 4
MAX_LOAD_LAG_HOURS = 6.0


@dataclass
class Anomaly:
    day: str
    metric: str
    value: float
    baseline: float
    z: float
    causes: list[dict] = field(default_factory=list)

    @property
    def pct_change(self) -> float:
        return (self.value - self.baseline) / self.baseline if self.baseline else float("inf")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def profile(con: duckdb.DuckDBPyConnection, table: str, date_col: str, columns: list[str],
            loaded_col: str | None = None) -> pd.DataFrame:
    nulls = ", ".join(f"AVG(CASE WHEN {_q(c)} IS NULL THEN 1.0 ELSE 0.0 END) AS {_q('null_rate:' + c)}"
                      for c in columns)
    lag = (f", DATE_DIFF('minute', CAST(MAX({_q(date_col)}) AS TIMESTAMP) + INTERVAL 1 DAY, "
           f"MAX({_q(loaded_col)})) / 60.0 AS load_lag_hours") if loaded_col else ""
    df = con.execute(f"""
        SELECT {_q(date_col)} AS day, COUNT(*) AS row_count, {nulls} {lag}
        FROM {_q(table)} GROUP BY 1 ORDER BY 1
    """).df()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df.set_index("day")


def robust_baseline(series: pd.Series, day, weeks: int = BASELINE_WEEKS) -> tuple[float, float] | None:
    """Median and MAD-based scale of the same weekday over the previous `weeks` weeks."""
    history = [day - pd.Timedelta(weeks=w) for w in range(1, weeks + 1)]
    vals = series.reindex(pd.to_datetime(history).date).dropna().to_numpy()
    if len(vals) < 3:
        return None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    # Floor the scale so a perfectly flat history does not turn tiny wobbles into alerts.
    scale = max(mad, abs(med) * 0.02, 1e-4)
    return med, scale


def detect(metrics: pd.DataFrame, day) -> list[Anomaly]:
    out = []
    for metric in metrics.columns:
        if metric == "load_lag_hours":
            v = metrics.at[day, metric]
            if v > MAX_LOAD_LAG_HOURS:  # freshness is an SLA, not a statistical question
                out.append(Anomaly(str(day), metric, round(float(v), 2), MAX_LOAD_LAG_HOURS, float("inf")))
            continue
        base = robust_baseline(metrics[metric], day)
        if base is None:
            continue
        med, scale = base
        if metric.startswith("null_rate:"):
            # A rate measured on n rows has sampling noise sqrt(p(1-p)/n) even when nothing is
            # wrong. With rare nulls that noise dwarfs the MAD of four points, so floor on it.
            n = float(metrics.at[day, "row_count"])
            scale = max(scale, 2 * np.sqrt(max(med, 1 / n) * (1 - med) / n))
        v = float(metrics.at[day, metric])
        z = (v - med) / scale
        rel = abs(v - med) / med if med else float("inf")
        if abs(z) >= Z_THRESHOLD and rel >= MIN_RELATIVE_CHANGE:
            out.append(Anomaly(str(day), metric, round(v, 4), round(med, 4), round(z, 1)))
    return out


def rank_causes(con: duckdb.DuckDBPyConnection, table: str, date_col: str, dim: str, day,
                column: str | None = None) -> list[dict]:
    """Which values of `dim` explain the change? Each value is compared with its own baseline."""
    measure = "COUNT(*)" if column is None else f"SUM(CASE WHEN {_q(column)} IS NULL THEN 1 ELSE 0 END)"
    per = con.execute(f"""
        SELECT {_q(date_col)} AS day, CAST({_q(dim)} AS VARCHAR) AS value, {measure} AS m
        FROM {_q(table)} GROUP BY 1, 2
    """).df()
    per["day"] = pd.to_datetime(per["day"]).dt.date
    per = per.pivot(index="day", columns="value", values="m").fillna(0)
    rows = []
    for value in per.columns:
        base = robust_baseline(per[value], day)
        if base is None:
            continue
        actual = float(per.at[day, value]) if day in per.index else 0.0
        rows.append({"value": value, "actual": actual, "expected": base[0], "delta": actual - base[0]})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    total = df["delta"].sum()
    # Share of the net change explained by each value (only values moving the same way as the total).
    same_dir = df[np.sign(df["delta"]) == np.sign(total)] if total else df
    denom = same_dir["delta"].sum() or 1.0
    df["share"] = np.where(df.index.isin(same_dir.index), df["delta"] / denom, 0.0)
    df = df.sort_values("share", ascending=False)
    return [{"value": r.value, "actual": round(r.actual), "expected": round(r.expected),
             "share_of_change": round(float(r.share), 3)} for r in df.itertuples() if r.share > 0.05]


def alert_payload(table: str, anomalies: list[Anomaly]) -> dict:
    blocks = [{"type": "header", "text": {"type": "plain_text",
                                          "text": f"Data quality alert: {table} ({len(anomalies)} issue(s))"}}]
    for a in anomalies:
        if a.metric == "load_lag_hours":
            line = f"*Freshness* on {a.day}: landed {a.value}h after the partition closed (SLA {a.baseline}h)"
        else:
            fmt = (lambda v: f"{v:.2%}") if a.metric.startswith("null_rate:") else (lambda v: f"{v:,.0f}")
            line = (f"*{a.metric}* on {a.day}: {fmt(a.value)} vs expected {fmt(a.baseline)} "
                    f"({a.pct_change:+.0%}, z={a.z})")
        if a.causes:
            top = ", ".join(f"{c['value']} ({c['share_of_change']:.0%})" for c in a.causes[:3])
            line += f"\nLikely source: {top}"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})
    return {"blocks": blocks}


def check(con: duckdb.DuckDBPyConnection, table: str, date_col: str, columns: list[str],
          dim: str | None = None, loaded_col: str | None = None, day=None) -> list[Anomaly]:
    metrics = profile(con, table, date_col, columns, loaded_col)
    day = day or metrics.index.max()
    anomalies = detect(metrics, day)
    if dim:
        for a in anomalies:
            if a.metric == "row_count":
                a.causes = rank_causes(con, table, date_col, dim, day)
            elif a.metric.startswith("null_rate:"):
                a.causes = rank_causes(con, table, date_col, dim, day, column=a.metric.split(":", 1)[1])
    return anomalies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="events.duckdb")
    ap.add_argument("--table", default="playback_events")
    ap.add_argument("--date-col", default="event_date")
    ap.add_argument("--columns", nargs="+", default=["user_id"])
    ap.add_argument("--dim", default="platform")
    ap.add_argument("--loaded-col", default="loaded_at")
    a = ap.parse_args()
    con = duckdb.connect(a.db)
    anomalies = check(con, a.table, a.date_col, a.columns, a.dim, a.loaded_col)
    if not anomalies:
        print(f"{a.table}: all checks passed")
        return
    print(json.dumps(alert_payload(a.table, anomalies), indent=2))


if __name__ == "__main__":
    main()
