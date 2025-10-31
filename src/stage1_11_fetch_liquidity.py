#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.11 — Final: Fetch, clean, and compute Liquidity & Funding Drivers
Output: data/macro_liquidity.parquet
"""

import pandas as pd
from datetime import datetime, timezone
from fredapi import Fred
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")

if not API_KEY or len(API_KEY.strip()) != 32:
    raise ValueError(f"❌ Invalid or missing FRED_API_KEY: {API_KEY}")

# =============================
# CONFIG
# =============================
OUT_FILE_PARQUET = "data/macro_liquidity.parquet"
START_DATE = "2020-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

fred = Fred(api_key=API_KEY)

# =============================
# FETCH
# =============================
def fetch_series(series_id):
    s = fred.get_series(series_id)
    df = s.to_frame(name=series_id).reset_index()
    df.columns = ["date", series_id]
    return df


def main():
    print(f"[{datetime.now()}] 🚀 Stage 1.11 — Fetching liquidity & funding drivers")

    series_map = {
        "WALCL": "Fed_BalanceSheet",            # Fed total assets
        "RRPONTSYD": "RRP_Usage",               # Reverse repo usage
        "DTB3": "TBILL3M",                      # 3-month T-Bill
        "SOFR": "SOFR",
        "EFFR": "EFFR",
    }

    dfs = []
    for sid, alias in series_map.items():
        print(f"[{datetime.now()}] ↳ Fetch {sid} as {alias}")
        try:
            df = fetch_series(sid)
            df.rename(columns={sid: alias}, inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error fetching {sid}: {e}")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.loc[merged["date"] >= START_DATE].copy()

    # =============================
    # FEATURE ENGINEERING
    # =============================

    def compute_features(df):
        # --- Fill base columns ---
        for c in ["Fed_BalanceSheet", "RRP_Usage", "TBILL3M", "SOFR", "EFFR"]:
            if c in df.columns:
                df[c] = df[c].ffill().bfill()

        # --- Interpolate TBILL to fill weekend/holiday gaps ---
        df["TBILL3M"] = df["TBILL3M"].interpolate(limit_direction="both")

        # --- Compute derived features ---
        df["Fed_BS_Delta_7d"] = df["Fed_BalanceSheet"].diff(7)
        df["RRP_Delta_7d"] = df["RRP_Usage"].diff(7)
        df["SOFR_EFFR_SPREAD"] = df["SOFR"] - df["EFFR"]
        df["TBILL3M_MINUS_FEDFUNDS"] = df["TBILL3M"] - df["EFFR"]

        # --- Fill short NaN in deltas ---
        for c in ["Fed_BS_Delta_7d", "RRP_Delta_7d"]:
            df[c] = df[c].ffill().bfill()

        # --- Liquidity composite (rank weighted) ---
        df["LIQ_COMPOSITE"] = (
            df["RRP_Delta_7d"].rank(pct=True) * 0.4 +
            df["Fed_BS_Delta_7d"].rank(pct=True) * 0.4 +
            (df["SOFR_EFFR_SPREAD"] * -1).rank(pct=True) * 0.2
        )

        df["LIQ_COMPOSITE_30d_mean"] = df["LIQ_COMPOSITE"].rolling(30, min_periods=5).mean()

        # --- Normalize LIQ_COMPOSITE 0-1 for easier comparison ---
        liq_min, liq_max = df["LIQ_COMPOSITE"].min(), df["LIQ_COMPOSITE"].max()
        df["LIQ_COMPOSITE_NORM"] = (df["LIQ_COMPOSITE"] - liq_min) / (liq_max - liq_min)

        # --- Final fill ---
        df = df.ffill().bfill()

        # --- Validate no NaN left ---
        for c in df.columns:
            if df[c].isna().any():
                raise ValueError(f"❌ Missing values remain in {c}")

        return df

    merged = compute_features(merged)

    # =============================
    # SAVE
    # =============================
    merged.to_parquet(OUT_FILE_PARQUET)
    print(f"[{datetime.now()}] 💾 Saved liquidity & funding → {OUT_FILE_PARQUET} ({len(merged)} rows)")
    print("🔎 Last 10 rows:")
    print(merged.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()