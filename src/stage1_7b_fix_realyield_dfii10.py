# ================================================================
# Stage 1.7b — Fix RealYield using DFII10 (TIPS 10Y Real Yield)
# ================================================================
import pandas as pd
from fredapi import Fred
from datetime import datetime
import os

# -------- CONFIG --------
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

START = "2023-01-01"
END   = datetime.utcnow().strftime("%Y-%m-%d")

MASTER_FILE = "data/macro_context_master.parquet"
OUT_FILE = "data/macro_gold_fix.parquet"

# -------- LOAD MACRO MASTER --------
print("🚀 Stage 1.7b — Fix RealYield using DFII10 (FRED TIPS 10Y)")
macro = pd.read_parquet(MASTER_FILE)
macro.index = pd.to_datetime(macro.index).tz_localize(None)

# -------- FETCH DFII10 --------
try:
    dfii10 = fred.get_series("DFII10", observation_start=START, observation_end=END)
    dfii10 = dfii10.to_frame("RealYield_DFII10")
    dfii10.index = pd.to_datetime(dfii10.index)
    print(f"✅ DFII10 fetched: {len(dfii10)} rows ({dfii10.index.min().date()} → {dfii10.index.max().date()})")
except Exception as e:
    print(f"❌ Failed to fetch DFII10: {e}")
    exit(1)

# -------- MERGE --------
macro = macro.merge(dfii10, left_on=macro.index.date, right_on=dfii10.index.date, how="left")
macro = macro.drop(columns=["key_0"], errors="ignore")
macro = macro.set_index(macro.index)
macro["RealYield_DFII10"] = macro["RealYield_DFII10"].ffill()

# -------- CORRELATION TEST (XAU/USD only) --------
try:
    xau = pd.read_parquet("data/XAU_USD_M5_clean.parquet")
    xau_daily = xau["mid_c"].resample("1D").last()
    merged = pd.merge_asof(
        xau_daily.sort_index(),
        macro[["RealYield", "RealYield_DFII10"]].sort_index(),
        left_index=True, right_index=True
    ).dropna()
    corr_old = merged["mid_c"].corr(merged["RealYield"])
    corr_new = merged["mid_c"].corr(merged["RealYield_DFII10"])
    print("\n📊 XAU/USD correlation check:")
    print(f"   Before fix (proxy RealYield): {corr_old:.3f}")
    print(f"   After  fix (DFII10):          {corr_new:.3f}")
except Exception as e:
    print(f"⚠️ Corr check skipped: {e}")

# -------- SAVE --------
macro.to_parquet(OUT_FILE)
print(f"\n💾 Saved fixed macro → {OUT_FILE}")
print("🎯 Stage 1.7b complete — RealYield replaced by DFII10.")