#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.11 — Fetch Liquidity & Funding Drivers (FINAL baseline)
---------------------------------------------------------------
Fetch Fed balance sheet, RRP, SOFR, EFFR, T-bill 3M, compute derived features.

Enhancements:
- Forward-fill TBILL3M & TBILL3M_MINUS_FEDFUNDS across weekends/holidays.
- Add flag is_weekend_or_holiday = True when value filled (not original print).
- Compute 30d z-scores for core liquidity factors.
- Assert no NaN in critical fields before save.
"""

import os
import time
import pandas as pd
from datetime import datetime
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path

# ==============================
# CONFIG
# ==============================
load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing FRED_API_KEY in .env")

fred = Fred(api_key=API_KEY)

OUT_FILE_PARQUET = "data/liquidity_fxdrivers.parquet"
START_DATE = "2020-01-01"
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")

# ==============================
# FETCH HELPERS
# ==============================
def fetch_series(series_id, colname):
    """Fetch FRED series with retry."""
    for _ in range(3):
        try:
            s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            return s.rename(colname)
        except Exception as e:
            print(f"⚠️ Retry {series_id} due to {e}")
            time.sleep(2)
    print(f"❌ Failed {series_id}")
    return pd.Series(dtype="float64", name=colname)

def compute_features(df):
    """Compute deltas, spreads, composite index, z-scores."""
    df["Fed_BS_Delta_7d"] = df["Fed_BalanceSheet"].diff(7)
    df["RRP_Delta_7d"] = df["RRP_Usage"].diff(7)
    df["SOFR_EFFR_SPREAD"] = df["SOFR"] - df["EFFR"]
    df["TBILL3M_MINUS_FEDFUNDS"] = df["TBILL3M"] - df["EFFR"]

    # Simple liquidity composite (normalized mix)
    liq_parts = []
    for c in ["Fed_BS_Delta_7d", "RRP_Delta_7d", "SOFR_EFFR_SPREAD"]:
        if c in df:
            liq_parts.append((df[c] - df[c].mean()) / df[c].std(ddof=0))
    df["LIQ_COMPOSITE"] = sum(liq_parts) / len(liq_parts)

    # 30-day rolling mean for smoothing
    df["LIQ_COMPOSITE_30d_mean"] = df["LIQ_COMPOSITE"].rolling(30, min_periods=5).mean()

    # --- NEW: z-scores for main drivers ---
    for col in ["LIQ_COMPOSITE", "RRP_Usage", "Fed_BS_Delta_7d"]:
        if col in df:
            roll = df[col].rolling(30, min_periods=10)
            df[f"{col}_Z30"] = (df[col] - roll.mean()) / roll.std(ddof=0)

    return df

# ==============================
# MAIN
# ==============================
def main():
    Path("data").mkdir(exist_ok=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 🚀 Stage 1.11 — Fetching liquidity & funding drivers")

    # fetch core series
    data = {
        "Fed_BalanceSheet": fetch_series("WALCL", "Fed_BalanceSheet"),
        "RRP_Usage": fetch_series("RRPONTSYD", "RRP_Usage"),
        "TBILL3M": fetch_series("DTB3", "TBILL3M"),
        "SOFR": fetch_series("SOFR", "SOFR"),
        "EFFR": fetch_series("EFFR", "EFFR"),
    }

    df = pd.concat(data.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # --- NEW: fix Fed_BalanceSheet (weekly series) ---
    # Some early rows before the first Fed H.4.1 release are NaN → fill both sides
    df["Fed_BalanceSheet"] = df["Fed_BalanceSheet"].ffill().bfill()
    df["RRP_Usage"] = df["RRP_Usage"].ffill().bfill()

    # Forward-fill TBILL3M & spread-related columns
    for col in ["TBILL3M"]:
        df[f"{col}_orig_na"] = df[col].isna()
        df[col] = df[col].ffill()
    df["is_weekend_or_holiday"] = df["TBILL3M_orig_na"]
    df.drop(columns=["TBILL3M_orig_na"], inplace=True)

    # Compute derived fields
    df = compute_features(df)

    # --- Assert completeness ---
    core_cols = ["Fed_BalanceSheet", "RRP_Usage", "SOFR", "EFFR", "LIQ_COMPOSITE"]
    for c in core_cols:
        if df[c].isna().any():
            raise ValueError(f"❌ Missing values remain in {c}")

    # Save
    df.to_parquet(OUT_FILE_PARQUET)
    print(f"💾 Saved liquidity & funding → {OUT_FILE_PARQUET} ({len(df)} rows)")
    print(df.tail(10).to_string(index=False))
    print("\nColumns:", list(df.columns))


if __name__ == "__main__":
    main()