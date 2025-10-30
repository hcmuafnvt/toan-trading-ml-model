#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.9 — Normalize & Feature-Engineer Investing.com Calendar
Baseline: data/econ_calendar_investing.parquet
Output :  data/econ_calendar_features.parquet
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

IN_FILE  = "data/econ_calendar_investing.parquet"
OUT_FILE = "data/econ_calendar_features.parquet"

def clean_currency(x: str):
    if not isinstance(x, str): return x
    x = x.strip().upper()
    mapping = {"CNH":"CNY","RMB":"CNY","UK":"GBP","EU":"EUR"}
    return mapping.get(x, x)

def extract_event_key(event: str):
    if not isinstance(event, str) or event.strip()=="":
        return np.nan
    # remove month/quarter suffix, keep main noun phrase
    event = re.sub(r"\s*\([^)]*\)", "", event)
    event = re.sub(r"\s+", " ", event).strip()
    return event

def main():
    Path("data").mkdir(exist_ok=True)
    df = pd.read_parquet(IN_FILE)
    print(f"📥 Loaded {len(df):,} rows")

    # Normalize impact
    df["impact_weight"] = df["impact_level"].astype(float) / 3.0

    # Clean currency
    df["currency"] = df["currency"].apply(clean_currency)

    # Extract event key
    df["event_key"] = df["event"].apply(extract_event_key)

    # Temporal features
    df["event_hour_local"] = pd.to_datetime(df["time_local"], errors="coerce").dt.hour
    df["event_weekday"] = pd.to_datetime(df["timestamp_utc"]).dt.weekday
    df["event_month"] = pd.to_datetime(df["timestamp_utc"]).dt.month

    # Replace inf & clip extreme numeric
    for c in ["actual_num","forecast_num","previous_num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[np.isinf(df[c]), c] = np.nan
        # clip to avoid crazy 1e12 from parse mistakes
        df[c] = df[c].clip(lower=-1e11, upper=1e11)

    # Compute zscore by event_key (optional, for anomaly detection)
    df["actual_z"] = df.groupby("event_key")["actual_num"].transform(
        lambda s: (s - s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0
    )

    # Save
    df.to_parquet(OUT_FILE)
    print(f"💾 Saved normalized calendar → {OUT_FILE} ({len(df):,} rows)")

    print(df.head(10)[["timestamp_utc","currency","event_key","impact_weight","actual_num","actual_z"]])

if __name__ == "__main__":
    main()