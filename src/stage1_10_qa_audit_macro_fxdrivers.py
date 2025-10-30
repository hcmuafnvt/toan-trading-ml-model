#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.10-QA — Audit Macro FX Drivers Dataset

Purpose:
- Check completeness, missing values, continuity, and correlation sanity.
- Verify that spreads and real-yield differentials behave as expected.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FILE = "data/macro_fxdrivers.parquet"

def main():
    print("🚀 Stage 1.10-QA — Auditing macro FX drivers data")

    if not Path(FILE).exists():
        print(f"❌ File not found: {FILE}")
        return

    df = pd.read_parquet(FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n📥 Loaded {len(df):,} rows from {FILE}")
    print(f"🗓️ Range: {df['date'].min().date()} → {df['date'].max().date()}")

    # 1️⃣ Check missing %
    miss = df.isna().mean().sort_values(ascending=False)
    print("\n📊 Missing value percentage:")
    print(miss.to_string(float_format=lambda x: f"{x*100:5.2f}%"))

    # 2️⃣ Check continuity (missing dates)
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = set(all_days.date) - set(df["date"].dt.date)
    print(f"\n🧩 Missing calendar days: {len(missing_dates)}")
    if len(missing_dates) < 10:
        print(sorted(list(missing_dates)))

    # 3️⃣ Summary statistics
    print("\n📈 Basic statistics (first 5 cols):")
    print(df.describe(include=[np.number]).T.head(5))

    # 4️⃣ Correlation matrix (for yield & spreads)
    num_cols = [c for c in df.columns if c != "date" and df[c].dtype != "O"]
    corr = df[num_cols].corr()

    key_cols = ["UST2Y", "UST5Y", "UST10Y", "UST10Y_REAL",
                "UST2Y_10Y_SPREAD", "UST10Y_REAL_DIFF",
                "US_DE_SPREAD", "US_UK_SPREAD", "US_JP_SPREAD"]
    corr_sub = corr.loc[key_cols, key_cols].round(2)
    print("\n🔗 Correlation (key rates & spreads):")
    print(corr_sub)

    # 5️⃣ Constant detection
    constant = [c for c in num_cols if df[c].nunique() <= 2]
    if constant:
        print("\n⚠️ Columns nearly constant:", constant)
    else:
        print("\n✅ No columns constant.")

    # 6️⃣ Sanity: last 5 days preview
    print("\n📆 Last 5 days sample:")
    print(df.tail(5).to_string(index=False))

    print("\n✅ Audit complete — review correlations & missing stats.")

if __name__ == "__main__":
    main()