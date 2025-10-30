# ================================================================
# Stage 1.7e — Fix DFII10 Real Yield Alignment for XAUUSD
# Purpose: apply 10-day forward shift, invert, smooth (21d mean)
# ================================================================

import pandas as pd
import numpy as np

# -------- CONFIG --------
MACRO_FILE = "data/macro_gold_fix.parquet"
XAU_FILE = "data/XAU_USD_M5_clean.parquet"
OUT_FILE = "data/macro_gold_fix_final.parquet"

SHIFT_DAYS = 10
ROLL_WINDOW = 21

# -------- LOAD DATA --------
macro = pd.read_parquet(MACRO_FILE)
xau = pd.read_parquet(XAU_FILE)

macro["date"] = pd.to_datetime(macro["date"], utc=True)
macro = macro.set_index("date")

if not isinstance(xau.index, pd.DatetimeIndex):
    xau.index = pd.to_datetime(xau.index, utc=True)

# -------- FIX ALIGNMENT --------
macro["RealYield_DFII10_fixed"] = (
    -macro["RealYield_DFII10"].shift(SHIFT_DAYS).rolling(ROLL_WINDOW).mean()
)

# -------- MERGE TO CHECK --------
xau_daily = xau["close"].resample("1D").last().dropna()

merged = pd.merge_asof(
    xau_daily.sort_index().to_frame("XAU_close"),
    macro[["RealYield_DFII10_fixed"]].sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

corr = merged["XAU_close"].corr(merged["RealYield_DFII10_fixed"])

# -------- SAVE --------
macro.to_parquet(OUT_FILE)

print("🚀 Stage 1.7e — DFII10 RealYield Alignment Fix")
print(f"✅ Shifted by +{SHIFT_DAYS} days, inverted, smoothed ({ROLL_WINDOW}d MA)")
print(f"💾 Saved → {OUT_FILE}")
print(f"📊 Correlation (XAUUSD vs RealYield_DFII10_fixed): {corr:.3f}  (expect ≈ -0.32)")