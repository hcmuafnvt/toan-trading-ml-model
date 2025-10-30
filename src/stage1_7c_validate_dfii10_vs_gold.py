#!/usr/bin/env python3
"""
Stage 1.7c — Validate DFII10 (TIPS 10Y Real Yield) vs XAUUSD
Goal: Check if replacing RealYield by DFII10 improves correlation with gold.
"""

import pandas as pd
import numpy as np

# =====================
# CONFIG
# =====================
MACRO_FILE = "data/macro_gold_fix.parquet"
XAU_FILE   = "data/XAU_USD_M5_clean.parquet"
ROLL_DAYS  = 7       # smoothing window
SHIFT_DAYS = 3       # lag shift (gold reacts after yields)

# =====================
# LOAD MACRO
# =====================
print("🚀 Stage 1.7c — Validate DFII10 (TIPS Real Yield) vs XAUUSD")

macro = pd.read_parquet(MACRO_FILE)

# --- Fix index ---
if "date" in macro.columns:
    macro["date"] = pd.to_datetime(macro["date"], utc=True, errors="coerce")
    macro = macro.set_index("date").sort_index()
else:
    macro.index = pd.to_datetime(macro.index, utc=True, errors="coerce")

# Smooth DFII10
macro["DFII10_smooth"] = macro["RealYield_DFII10"].rolling(ROLL_DAYS, min_periods=1).mean()
macro["DFII10_shifted"] = macro["DFII10_smooth"].shift(SHIFT_DAYS)

# =====================
# LOAD GOLD PRICE
# =====================
price = pd.read_parquet(XAU_FILE)
if not isinstance(price.index, pd.DatetimeIndex):
    price.index = pd.to_datetime(price.index, utc=True)
else:
    price = price.tz_convert("UTC")

price["close"] = price["mid_c"] if "mid_c" in price.columns else price["close"]
daily = price["close"].resample("1D").last().dropna()

# =====================
# MERGE & CORRELATION
# =====================
merged = pd.merge_asof(
    daily.sort_index(),
    macro[["DFII10_shifted"]].sort_index(),
    left_index=True, right_index=True, direction="backward"
).dropna()

corr = merged["close"].corr(merged["DFII10_shifted"])

print(f"\n📊 Correlation (XAUUSD close vs DFII10_shifted_{ROLL_DAYS}d, +{SHIFT_DAYS}d lag): {corr:.3f}")
merged.to_parquet("data/xauusd_dfii10_corr_audit.parquet")
print("💾 Saved → data/xauusd_dfii10_corr_audit.parquet")

if corr < -0.4:
    print("✅ Strong negative correlation — DFII10 works as expected.")
elif corr < -0.2:
    print("⚠️ Moderate negative correlation — partially correct, may refine lag.")
else:
    print("❌ Weak or wrong correlation — check lag or data alignment.")