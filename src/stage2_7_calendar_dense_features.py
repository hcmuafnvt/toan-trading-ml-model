#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2.7 — Calendar Dense Feature Engineering
Biến lịch kinh tế dạng event thành feature numeric dạng time-series M5.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

CAL_FILE = "data/econ_calendar_features.parquet"
GRID_FILE = "data/fx_timegrid.parquet"
OUT_FILE = "data/stage2_calendar_dense.parquet"

def log(msg):
    print(f"[2.7] {msg}")

def main():
    log("🚀 Stage 2.7 — Calendar Dense Feature Engineering")
    cal = pd.read_parquet(CAL_FILE)
    grid = pd.read_parquet(GRID_FILE)
    grid.index = pd.to_datetime(grid.index, utc=True)

    # --- Chuẩn hóa thời gian ---
    time_col = "timestamp_utc" if "timestamp_utc" in cal.columns else "time_utc"
    cal[time_col] = pd.to_datetime(cal[time_col], utc=True, errors="coerce")
    cal = cal.dropna(subset=[time_col]).sort_values(time_col)

    # --- Build dense grid (M5 UTC) ---
    grid["cal_minutes_since_event"] = np.nan
    grid["cal_minutes_to_next_event"] = np.nan
    grid["cal_active_impact_weight"] = 0.0
    grid["cal_high_impact_ahead_15m"] = 0
    grid["cal_med_impact_ahead_30m"] = 0
    grid["cal_event_count_1h"] = 0

    cal_events = cal[["timestamp_utc", "impact_weight", "currency"]].dropna(subset=["timestamp_utc"])
    cal_events = cal_events.set_index("timestamp_utc").sort_index()

    times = grid.index

    # Precompute event timestamps & impacts
    event_times = cal_events.index.to_numpy()
    event_impacts = cal_events["impact_weight"].to_numpy()

    for i, t in enumerate(times):
        # Gần nhất trước
        prev_idx = np.searchsorted(event_times, t) - 1
        next_idx = prev_idx + 1

        if prev_idx >= 0:
            delta = (t - event_times[prev_idx]).total_seconds() / 60.0
            grid.iloc[i, grid.columns.get_loc("cal_minutes_since_event")] = delta
            grid.iloc[i, grid.columns.get_loc("cal_active_impact_weight")] = event_impacts[prev_idx] * np.exp(-delta / 60)
        if next_idx < len(event_times):
            delta_next = (event_times[next_idx] - t).total_seconds() / 60.0
            grid.iloc[i, grid.columns.get_loc("cal_minutes_to_next_event")] = delta_next
            if delta_next <= 15 and event_impacts[next_idx] > 0.6:
                grid.iloc[i, grid.columns.get_loc("cal_high_impact_ahead_15m")] = 1
            if delta_next <= 30 and event_impacts[next_idx] > 0.3:
                grid.iloc[i, grid.columns.get_loc("cal_med_impact_ahead_30m")] = 1

    # --- Event count in past 1 hour ---
    # --- Event count in past 1 hour (fix duplicate timestamps) ---
    event_series = (
        pd.Series(1, index=cal_events.index)
        .groupby(level=0)
        .size()               # gộp trùng timestamp
        .rename("event_count")
    )
    event_series = event_series.sort_index()
    event_series = event_series[~event_series.index.duplicated(keep="last")]

    event_counts = (
        event_series.reindex(times.union(event_series.index))
        .sort_index()
        .ffill()
        .rolling("60min")
        .sum()
        .reindex(times)
    )
    grid["cal_event_count_1h"] = event_counts.fillna(0).to_numpy()

    # --- Fill missing numeric ---
    grid = grid.fillna(0)

    # --- Save ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(OUT_FILE)
    log(f"💾 Saved calendar dense features → {OUT_FILE} ({len(grid):,} rows)")
    log(f"📈 Columns: {list(grid.columns)}")
    log(f"🕒 Range: {grid.index.min()} → {grid.index.max()}")

if __name__ == "__main__":
    main()