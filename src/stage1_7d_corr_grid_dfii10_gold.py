# ============================================================
# Stage 1.7d — Correlation Grid: DFII10 (RealYield) vs XAUUSD
# Purpose: find the true lag, smoothing, and sign inversion
# ============================================================

import pandas as pd
import numpy as np
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# -------- CONFIG --------
MACRO_FILE = "data/macro_gold_fix.parquet"
XAU_FILE = "data/XAU_USD_M5_clean.parquet"
OUT_FILE = "logs/dfii10_gold_corr_grid.csv"

LAGS = range(0, 11)          # days
ROLL_WINDOWS = [7, 14, 21]   # days
INVERTS = [False, True]

# -------- LOAD DATA --------
macro = pd.read_parquet(MACRO_FILE)
price = pd.read_parquet(XAU_FILE)

# ensure datetime index
macro["date"] = pd.to_datetime(macro["date"], utc=True)
macro = macro.set_index("date")

if not isinstance(price.index, pd.DatetimeIndex):
    price.index = pd.to_datetime(price.index, utc=True)

# resample to daily close (NY close ≈ 23:00 UTC)
xau_daily = price["close"].resample("1D").last().dropna()

# merge
merged = pd.merge_asof(
    xau_daily.sort_index().to_frame("XAU_close"),
    macro[["RealYield_DFII10"]].sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

# -------- GRID SEARCH --------
results = []
for lag, roll, inv in product(LAGS, ROLL_WINDOWS, INVERTS):
    df = merged.copy()

    # shift RealYield by lag
    df["RealYield_shifted"] = df["RealYield_DFII10"].shift(lag)

    # optional inversion
    if inv:
        df["RealYield_shifted"] = -df["RealYield_shifted"]

    # smoothing
    df["RealYield_smooth"] = df["RealYield_shifted"].rolling(roll).mean()

    # correlation
    corr = df["XAU_close"].corr(df["RealYield_smooth"])
    results.append({
        "lag_days": lag,
        "rolling_days": roll,
        "invert": inv,
        "corr": corr
    })

# -------- RESULTS --------
corr_df = pd.DataFrame(results).sort_values("corr", ascending=False)
corr_df.to_csv(OUT_FILE, index=False)

print("🚀 Stage 1.7d — DFII10 ↔ Gold Correlation Grid Scan")
print(f"✅ Tested {len(corr_df)} combinations")
print(f"💾 Saved results → {OUT_FILE}\n")

print("📊 Top 5 configurations (highest correlation):")
print(corr_df.head(5).to_string(index=False))

print("\n📊 Bottom 5 configurations (lowest correlation):")
print(corr_df.tail(5).to_string(index=False))