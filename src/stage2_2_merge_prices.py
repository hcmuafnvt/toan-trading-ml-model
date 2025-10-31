#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2.2 — Merge multi-pair prices into the global M5 time grid

Input (baseline Stage 1):
- data/fx_timegrid.parquet                (UTC DatetimeIndex M5 master grid)
- data/GBP_USD_M5_std.parquet             (cols: open, high, low, close, volume)
- data/EUR_USD_M5_std.parquet
- data/USD_JPY_M5_std.parquet
- data/XAU_USD_M5_std.parquet

Output:
- data/stage2_prices_merged.parquet
  DatetimeIndex UTC at M5 frequency
  Columns like:
    gbpusd_open, gbpusd_high, gbpusd_low, gbpusd_close, gbpusd_volume,
    eurusd_close, ...,
    usdjpy_close, ...,
    xauusd_close, ...
"""

import pandas as pd
from pathlib import Path

PAIR_FILES = {
    "gbpusd": "data/GBP_USD_M5_std.parquet",
    "eurusd": "data/EUR_USD_M5_std.parquet",
    "usdjpy": "data/USD_JPY_M5_std.parquet",
    "xauusd": "data/XAU_USD_M5_std.parquet",
}

OUT_FILE = "data/stage2_prices_merged.parquet"

def load_timegrid():
    df = pd.read_parquet("data/fx_timegrid.parquet")
    # ensure DatetimeIndex UTC sorted unique
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("fx_timegrid.parquet must have DatetimeIndex")
    df = df.sort_index()
    # we only keep the index; we'll merge onto this
    base = pd.DataFrame(index=df.index.copy())
    base.index.name = "timestamp_utc"
    return base

def load_pair(path):
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have DatetimeIndex")

    # keep only canonical cols from Stage 1 baseline
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")

    df = df[needed].sort_index()

    # forward-fill gaps after reindex to master grid (we'll do align later)
    return df

def main():
    print("📥 Loading master time grid ...")
    master = load_timegrid()  # index = full M5 UTC timeline

    all_cols = []

    for prefix, fpath in PAIR_FILES.items():
        print(f"🔗 Loading {prefix.upper()} from {fpath} ...")
        px = load_pair(fpath)

        # reindex to master timeline
        aligned = master.join(px, how="left")
        aligned = aligned.ffill()

        # rename columns with prefix
        aligned = aligned.add_prefix(f"{prefix}_")

        # collect (but don't lose master index)
        all_cols.append(aligned)

    # concat all prefixed blocks column-wise on the same index
    merged = pd.concat(all_cols, axis=1)

    # optional sanity: drop duplicated columns if any collision (shouldn't happen now)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # final sort by index just in case
    merged = merged.sort_index()

    # save
    Path("data").mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)

    print(f"✅ Saved merged price feature frame → {OUT_FILE}")
    print(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    print(f"⬆ Shape: {merged.shape}")
    print("🔍 Sample columns:", merged.columns[:12].tolist())
    print(merged.head(5).to_string())

if __name__ == "__main__":
    main()