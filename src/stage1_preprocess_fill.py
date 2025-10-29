#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.2 — Normalize Timeline & Synthetic Gap Fill
Now includes EUR_USD
"""
import pandas as pd
from datetime import timedelta

PAIRS = ["GBP_USD", "USD_JPY", "XAU_USD", "EUR_USD"]
GRANULARITY = "M5"
FREQ = "5min"
FILL_LIMIT = 12  # ≤ 1 hour

for pair in PAIRS:
    in_path  = f"data/{pair}_{GRANULARITY}_all.parquet"
    out_path = f"data/{pair}_{GRANULARITY}_clean.parquet"

    print(f"\n🚀 Stage 1.2 — Normalize {pair}")
    df = pd.read_parquet(in_path)
    print(f"📦 Raw: {len(df):,} rows  {df.index[0]} → {df.index[-1]}")

    # UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    # regular 5-minute timeline
    full_index = pd.date_range(df.index[0], df.index[-1], freq=FREQ, tz="UTC")
    df = df.reindex(full_index)

    # fill small gaps with last close
    df = df.ffill(limit=FILL_LIMIT).bfill(limit=FILL_LIMIT)
    df["volume"] = df["volume"].fillna(0)
    df["is_synthetic"] = (df["volume"] == 0).astype(int)

    synth_ratio = df["is_synthetic"].mean() * 100
    print(f"✅ Synthetic candles: {df['is_synthetic'].sum():,} ({synth_ratio:.2f}%)")

    df.to_parquet(out_path)
    print(f"💾 Saved → {out_path}")

print("\n🎯 Stage 1.2 hoàn tất — tất cả 4 cặp đã được normalize timeline.")