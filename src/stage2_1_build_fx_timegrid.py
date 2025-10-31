#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2.1 — FX TimeGrid Builder
--------------------------------
Goal:
    - Build a unified 5-minute UTC time index from GBP/USD clean data.
    - This will serve as the master time backbone for all Stage 2 feature merges.

Input:
    data/GBP_USD_M5_clean.parquet   (from Stage 1)

Output:
    data/fx_timegrid.parquet

Notes:
    - Keeps only DatetimeIndex (UTC) as column 'timestamp'
    - Ensures continuous 5-minute frequency, forward-filled where needed.
"""

import pandas as pd
from pathlib import Path

IN_FILE  = "data/GBP_USD_M5_clean.parquet"
OUT_FILE = "data/fx_timegrid.parquet"

def main():
    print(f"📥 Loading {IN_FILE} ...")
    df = pd.read_parquet(IN_FILE)

    # ensure DateTimeIndex UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input file must have DateTimeIndex.")
    df.index = df.index.tz_convert("UTC")

    # build continuous 5-minute grid
    start, end = df.index.min(), df.index.max()
    grid = pd.date_range(start=start, end=end, freq="5min", tz="UTC")

    # create dataframe
    timegrid = pd.DataFrame(index=grid)
    timegrid.index.name = "timestamp"

    # sanity check continuity
    missing = grid.to_series().diff().dt.total_seconds().gt(300).sum()
    if missing:
        print(f"⚠️ Found {missing} irregular gaps >5 min")

    # save parquet
    Path("data").mkdir(parents=True, exist_ok=True)
    timegrid.to_parquet(OUT_FILE)

    print(f"✅ Saved {len(timegrid):,} rows → {OUT_FILE}")
    print("🕒 Range:", timegrid.index[0], "→", timegrid.index[-1])

if __name__ == "__main__":
    main()