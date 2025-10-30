#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.8-QA — Audit Investing.com Calendar Data Quality

Reads: data/econ_calendar_investing_raw.parquet
Checks:
  - total rows, date coverage, per-day event counts
  - missing ratios for key fields
  - impact level distribution
  - sanity of numeric fields
  - quick sample of extreme or missing rows
"""

import pandas as pd
from datetime import datetime

FILE = "data/econ_calendar_investing_raw.parquet"

print("🚀 Stage 1.8-QA — Auditing Investing.com Calendar Data\n")

# ============================
# LOAD
# ============================
try:
    df = pd.read_parquet(FILE)
except Exception as e:
    raise SystemExit(f"❌ Cannot load {FILE}: {e}")

print(f"📥 Loaded {len(df):,} rows from {FILE}")
print("Columns:", list(df.columns), "\n")

# ============================
# DATE RANGE SUMMARY
# ============================
if "date_local" in df.columns:
    df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")
    print(f"🗓️  Range: {df['date_local'].min().date()} → {df['date_local'].max().date()}")

    per_day = df.groupby(df["date_local"].dt.date).size()
    print(f"📊 Days covered: {len(per_day)} | Avg events/day: {per_day.mean():.1f} | Min: {per_day.min()} | Max: {per_day.max()}\n")

    print("🔍 Sample of days with fewest events:")
    print(per_day.sort_values().head(5).to_string(), "\n")

# ============================
# IMPACT ANALYSIS
# ============================
impact_stats = df["impact_level"].value_counts(dropna=False).sort_index()
missing_impact = df["impact_level"].isna().mean() * 100

print("⚡ Impact Level distribution:")
print(impact_stats)
print(f"➡️  Missing impact_level: {missing_impact:.2f}%\n")

# ============================
# NUMERIC FIELDS AUDIT
# ============================
for col in ["actual_num", "forecast_num", "previous_num"]:
    if col in df.columns:
        ratio = df[col].notna().mean() * 100
        print(f"📈 {col}: {ratio:.2f}% non-null")

print("")

# ============================
# EVENT NAME / CURRENCY AUDIT
# ============================
missing_event = df["event"].isna().mean() * 100 if "event" in df else 0
missing_curr = df["currency"].isna().mean() * 100 if "currency" in df else 0
print(f"💬 Missing event: {missing_event:.2f}% | Missing currency: {missing_curr:.2f}%\n")

# ============================
# SANITY CHECK — SAMPLE OUTLIERS
# ============================
print("🔎 Example events with actual_num > 100 or < -100:")
outliers = df[(df["actual_num"].abs() > 100)]
print(outliers[["timestamp_utc", "currency", "event", "actual", "actual_num"]].head(5).to_string(index=False))

print("\n🔎 Example events missing impact_level but have actual/forecast:")
missing_impact_rows = df[df["impact_level"].isna() & df["actual"].notna()]
print(missing_impact_rows[["timestamp_utc", "currency", "event", "actual", "impact_text"]].head(5).to_string(index=False))

print("\n✅ Audit complete — review above stats before deciding full-range crawl.\n")