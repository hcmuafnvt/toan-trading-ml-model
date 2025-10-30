import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas_datareader import data as web

# ======================================================
# Stage 1.6i — Enhanced JPY Composite Driver
#
# Goal:
#   Build a stronger macro driver for USD/JPY using:
#   - Long-term rate spread (US10Y - JGB10Y)
#   - Real yield spread: (RealYield_US - RealYield_JP)
#   - Risk sentiment (SPX up / VIX down => carry on)
#
# Output:
#   - Print correlations of each component
#   - Print correlation of composite
#
# Requires:
#   data/macro_context_v6.parquet
#   data/USD_JPY_M5_clean.parquet
# Pulls:
#   Japan CPI YoY (proxy for JP inflation)
#   NOTE: We'll fetch from FRED: "JPNCPIALLMINMEI" (Japan CPI All Items, %YoY)
# ======================================================

DATA_DIR = "data"
MACRO_FILE = os.path.join(DATA_DIR, "macro_context_v6.parquet")
PAIR_FILE  = os.path.join(DATA_DIR, "USD_JPY_M5_clean.parquet")

print("🚀 Stage 1.6i — Enhanced USDJPY Macro Driver (composite)")

# 1️⃣ Load macro v6 (has DGS10, JGB10Y, RealYield_shifted, SPX/VIX, etc.)
macro = pd.read_parquet(MACRO_FILE)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

need_cols = [
    "DGS10",            # US 10Y
    "JGB10Y",           # JP 10Y
    "RealYield_shifted",# US real yield (already +3d adjusted logic)
    "('SPX', '^GSPC')", # SPX level
    "('VIX', '^VIX')"   # VIX level
]
missing_cols = [c for c in need_cols if c not in macro.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in macro_context_v6: {missing_cols}")
else:
    print(f"✅ Macro base columns OK: {need_cols}")

# We'll rename SPX/VIX to easy names
macro = macro.rename(columns={
    "('SPX', '^GSPC')": "SPX",
    "('VIX', '^VIX')" : "VIX"
})

# 2️⃣ Build core drivers that we are SURE we have (no Japan CPI)

# Long-term yield spread (US10Y - JGB10Y)
macro["Spr_long_US_JP"] = macro["DGS10"] - macro["JGB10Y"]

# Risk sentiment factor:
# SPX up (risk-on), VIX down (fear down) -> people short JPY to fund risk trades.
macro["SPX_ret_1d"] = macro["SPX"].pct_change()
macro["VIX_chg_1d"] = macro["VIX"].pct_change()
macro["RiskOnFactor"] = macro["SPX_ret_1d"] - macro["VIX_chg_1d"]

# 3️⃣ Resample to daily
daily_macro = (
    macro[[
        "Spr_long_US_JP",
        "RiskOnFactor"
    ]]
    .resample("1D")
    .last()
)

# Forward fill yield spread (bond markets can be closed some days)
daily_macro["Spr_long_US_JP"] = daily_macro["Spr_long_US_JP"].ffill()
# RiskOnFactor is daily flow signal; don't forward fill, keep NaN allowed

# 4️⃣ Load USDJPY daily
fx = pd.read_parquet(PAIR_FILE)
if "synthetic" in fx.columns:
    fx = fx[~fx["synthetic"].astype(bool)]

fx.index = pd.to_datetime(fx.index)
if fx.index.tz is None:
    fx.index = fx.index.tz_localize("UTC")
else:
    fx.index = fx.index.tz_convert("UTC")

fx_daily = fx["close"].resample("1D").mean().to_frame("USDJPY_close")

# 5️⃣ Align via merge_asof
merged = pd.merge_asof(
    fx_daily.sort_index(),
    daily_macro.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

# 6️⃣ Z-score components to combine them
def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-9)

merged["Z_long"]   = zscore(merged["Spr_long_US_JP"])
merged["Z_riskon"] = zscore(merged["RiskOnFactor"].fillna(0))

# Composite without JP CPI
merged["JPY_driver_composite_nocpi"] = (
    0.8 * merged["Z_long"] +
    0.2 * merged["Z_riskon"]
)

# 7️⃣ Correlation diagnostics
def corr_safe(a, b):
    if a.isna().all() or b.isna().all():
        return float("nan")
    return a.corr(b)

corr_long   = corr_safe(merged["USDJPY_close"], merged["Spr_long_US_JP"])
corr_risk   = corr_safe(merged["USDJPY_close"], merged["RiskOnFactor"])
corr_comp   = corr_safe(merged["USDJPY_close"], merged["JPY_driver_composite_nocpi"])

print("\n📊 Correlation vs USDJPY_close (daily)")
print(f"   LongTermYieldSpread     corr = {corr_long:.3f} ↑ (should be positive)")
print(f"   RiskOnFactor            corr = {corr_risk:.3f} ↑ (risk-on -> USDJPY up)")
print(f"   Composite_noCPI         corr = {corr_comp:.3f} ↑ (target >= 0.40)")

# 8️⃣ Save audit
OUT_AUDIT_FILE = os.path.join(DATA_DIR, "usdjpy_driver_audit.parquet")
merged.to_parquet(OUT_AUDIT_FILE)
print(f"\n💾 Audit snapshot saved → {OUT_AUDIT_FILE}")

if corr_comp >= 0.40:
    print("🎯 RESULT: PASS. composite_noCPI is strong enough for Stage 2 features.")
elif corr_long >= 0.40:
    print("🎯 RESULT: SEMI-PASS. use Spr_long_US_JP directly as JPY_driver_final.")
else:
    print("⚠️ RESULT: WEAK (<0.40). USDJPY will need special regime / intervention signals in Stage 1.8.")