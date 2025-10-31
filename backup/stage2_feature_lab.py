#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 — Feature Extraction Lab
--------------------------------
Trích xuất feature thống kê từ dữ liệu M5 clean (Stage 1.2)
dùng tsfresh.EfficientFCParameters (fund-grade version).
"""

import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from multiprocessing import cpu_count

# ========= CONFIG =========
PAIRS = ["GBP_USD", "USD_JPY", "EUR_USD", "XAU_USD"]
GRANULARITY = "M5"
DATA_DIR = "data"
OUT_PATH = "logs/stage2_features.csv"
N_JOBS = max(cpu_count() - 4, 4)

# ========= PREPARE =========
def load_pair(pair):
    path = f"{DATA_DIR}/{pair}_{GRANULARITY}_clean.parquet"
    df = pd.read_parquet(path)
    df = df.copy()

    # Remove synthetic candles (volume == 0)
    df = df[df["volume"] > 0].copy()

    # Add derived columns
    df["pair"] = pair
    df["returns"] = df["close"].pct_change().fillna(0)
    df["log_ret"] = np.log1p(df["returns"])
    df["spread"] = df["mid_h"] - df["mid_l"]
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek

    # TSFresh expects integer index per series_id
    df["time_id"] = np.arange(len(df))
    df["series_id"] = pair

    return df

# ========= FEATURE EXTRACTION =========
def extract_pair_features(df):
    settings = EfficientFCParameters()

    # chọn subset numeric columns
    feature_cols = ["returns", "log_ret", "spread", "volume"]
    df_feat = df[["series_id", "time_id"] + feature_cols]

    print(f"⚙️  Extracting {df['series_id'].iloc[0]} | {len(df_feat):,} rows")
    features = extract_features(
        df_feat,
        column_id="series_id",
        column_sort="time_id",
        default_fc_parameters=settings,
        n_jobs=N_JOBS,
        disable_progressbar=False,
    )

    # cleanup feature names
    features.columns = (
        features.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True).str.strip("_")
    )

    return features

# ========= MAIN =========
def main():
    os.makedirs("logs", exist_ok=True)
    all_features = []

    for pair in PAIRS:
        df = load_pair(pair)
        feats = extract_pair_features(df)
        feats["pair"] = pair
        all_features.append(feats.reset_index())

    merged = pd.concat(all_features, ignore_index=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved combined features → {OUT_PATH}")
    print(f"📊 Total feature columns: {len(merged.columns):,}")
    print(f"📦 Total rows: {len(merged):,}")

if __name__ == "__main__":
    main()