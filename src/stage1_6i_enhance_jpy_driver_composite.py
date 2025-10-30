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

# 2️⃣ Fetch Japan CPI YoY (inflation proxy)
# We'll approximate Japan real yield = JGB10Y - CPI_JP_YoY
# FRED series: "JPNCPIALLMINMEI" (Consumer Price Index: All Items for Japan, % change YoY)
print("🌐 Fetching Japan CPI YoY from FRED...")
macro_start = macro.index.min().tz_convert("UTC").tz_localize(None)
macro_end   = macro.index.max().tz_convert("UTC").tz_localize(None)

try:
    jp_cpi = web.DataReader("JPNCPIALLMINMEI", "fred", macro_start, macro_end)
    jp_cpi.columns = ["JP_CPI_YoY"]
    jp_cpi.index = pd.to_datetime(jp_cpi.index).tz_localize("UTC")
    print(f"   ✅ JP CPI YoY: {len(jp_cpi)} rows ({jp_cpi.index.min().date()} → {jp_cpi.index.max().date()})")
except Exception as e:
    print(f"   ⚠️ Failed JP CPI fetch: {e}")
    jp_cpi = pd.DataFrame(columns=["JP_CPI_YoY"])

# merge JP CPI YoY into macro
macro = macro.merge(jp_cpi, left_index=True, right_index=True, how="left")
macro["JP_CPI_YoY"] = macro["JP_CPI_YoY"].ffill()

# 3️⃣ Build Japan real yield proxy
# RealYield_JP ≈ JGB10Y - JP_CPI_YoY
macro["RealYield_JP_proxy"] = macro["JGB10Y"] - macro["JP_CPI_YoY"]

# 4️⃣ Build individual candidate drivers

# 4a. Long-term rate spread (US10Y - JGB10Y)
macro["Spr_long_US_JP"] = macro["DGS10"] - macro["JGB10Y"]

# 4b. Real yield differential:
#     (US real yield shifted) - (JP real yield proxy)
macro["Spr_realDiff_US_JP"] = macro["RealYield_shifted"] - macro["RealYield_JP_proxy"]

# 4c. Risk sentiment carry factor
# Heuristic:
#   - when SPX daily return is strong and VIX is dropping,
#     market is risk-on -> funding JPY shorts -> USDJPY tends to go up.
#
# We'll create:
#   RiskOn = (+ΔSPX%) + (-ΔVIX%)
#   Use 1-day % change
macro["SPX_ret_1d"] = macro["SPX"].pct_change()  # ~ equity risk-on if positive
macro["VIX_chg_1d"] = macro["VIX"].pct_change()  # ~ fear up if positive
# invert VIX because risk-on = VIX down
macro["RiskOnFactor"] = macro["SPX_ret_1d"] - macro["VIX_chg_1d"]

# 5️⃣ Resample everything daily and dropna smart
daily_macro = (
    macro[[
        "Spr_long_US_JP",
        "Spr_realDiff_US_JP",
        "RiskOnFactor"
    ]]
    .resample("1D")
    .last()
)

# Some forward fill for slow series (yields don't update every single UTC day)
daily_macro["Spr_long_US_JP"] = daily_macro["Spr_long_US_JP"].ffill()
daily_macro["Spr_realDiff_US_JP"] = daily_macro["Spr_realDiff_US_JP"].ffill()
# RiskOnFactor is daily flow, don't ffill that too far (leave NaN OK)

# 6️⃣ Load USDJPY and resample to daily
fx = pd.read_parquet(PAIR_FILE)
if "synthetic" in fx.columns:
    fx = fx[~fx["synthetic"].astype(bool)]

fx.index = pd.to_datetime(fx.index)
if fx.index.tz is None:
    fx.index = fx.index.tz_localize("UTC")
else:
    fx.index = fx.index.tz_convert("UTC")

fx_daily = fx["close"].resample("1D").mean().to_frame("USDJPY_close")

# 7️⃣ Merge with asof (align last known macro to FX day)
merged = pd.merge_asof(
    fx_daily.sort_index(),
    daily_macro.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

# 8️⃣ Normalize components before composite
# We'll z-score each factor so they’re comparable scale.
def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-9)

merged["Z_long"]   = zscore(merged["Spr_long_US_JP"])
merged["Z_real"]   = zscore(merged["Spr_realDiff_US_JP"])
merged["Z_riskon"] = zscore(merged["RiskOnFactor"].fillna(0))

# 9️⃣ Build composite driver
# Weighted blend:
# - long-term yield spread is dominant (0.6)
# - real yield differential meaningful medium-term (0.3)
# - risk-on factor explains carry bursts (0.1)
merged["JPY_driver_composite"] = (
    0.6 * merged["Z_long"] +
    0.3 * merged["Z_real"] +
    0.1 * merged["Z_riskon"]
)

# 10️⃣ Measure correlations
components = [
    ("Spr_long_US_JP",         "LongTermYieldSpread"),
    ("Spr_realDiff_US_JP",     "RealYieldDiff"),
    ("RiskOnFactor",           "RiskOnFactor"),
    ("JPY_driver_composite",   "CompositeDriver")
]

print("\n📊 Correlation vs USDJPY_close (daily)")
corr_results = {}
for col, label in components:
    if col in merged.columns:
        c = merged["USDJPY_close"].corr(merged[col])
        corr_results[label] = c
        arrow = "↑ (should be positive)" if (c is not None and c > 0) else "↓"
        print(f"   {label:<20} corr = {c:.3f} {arrow}")
    else:
        corr_results[label] = None
        print(f"   {label:<20} missing")

# 11️⃣ Pick best explaining factor
valid = {k:v for k,v in corr_results.items() if v is not None}
if valid:
    best_key = max(valid, key=lambda k: abs(valid[k]))
    print(f"\n🏁 Best USDJPY macro explainer: {best_key} "
          f"with corr={valid[best_key]:.3f}")
else:
    print("\n❌ No usable drivers")

# 12️⃣ Save final merged daily driver snapshot for audit / future join if needed
OUT_AUDIT_FILE = os.path.join(DATA_DIR, "usdjpy_driver_audit.parquet")
merged.to_parquet(OUT_AUDIT_FILE)
print(f"\n💾 Audit snapshot saved → {OUT_AUDIT_FILE}")
print("🎯 Stage 1.6i complete — candidate JPY_driver_composite generated & evaluated.")