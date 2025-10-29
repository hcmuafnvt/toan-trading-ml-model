import pandas as pd
import os
from pandas_datareader import data as web
import warnings
warnings.filterwarnings("ignore")

# ================================
# Stage 1.6 — Macro Fix & Validation (v2)
# ================================
DATA_DIR = "data"
macro_path = os.path.join(DATA_DIR, "macro_context_v2.parquet")

print("🚀 Stage 1.6 — Resample macro daily, add BoJ rate & RealYieldTrend")

# -------------------------------------------------------------
# 1️⃣ Load enhanced macro context
# -------------------------------------------------------------
macro = pd.read_parquet(macro_path)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

print(f"✅ Loaded macro_context_v2.parquet: {len(macro):,} rows "
      f"({macro.index.min().date()} → {macro.index.max().date()})")

# -------------------------------------------------------------
# 2️⃣ Resample to DAILY (avoid fake intraday corr)
# -------------------------------------------------------------
macro_daily = macro.resample("1D").ffill()
print(f"📊 Resampled to daily: {len(macro_daily):,} rows")

# -------------------------------------------------------------
# 3️⃣ Add BoJ Policy Rate (FRED: INTGSTJPM193N)
# -------------------------------------------------------------
try:
    start = pd.Timestamp(macro_daily.index.min()).tz_localize(None)
    end = pd.Timestamp(macro_daily.index.max()).tz_localize(None)
    boj = web.DataReader("INTGSTJPM193N", "fred", start, end)
    boj.columns = ["BoJRate"]
    boj.index = pd.to_datetime(boj.index).tz_localize("UTC")
    macro_daily = macro_daily.merge(boj, left_index=True, right_index=True, how="left")
    print(f"✅ BoJ Policy Rate: {len(boj):,} rows ({boj.index.min().date()} → {boj.index.max().date()})")
except Exception as e:
    print(f"⚠️  Failed BoJ Policy Rate fetch: {e}")
    macro_daily["BoJRate"] = None

# -------------------------------------------------------------
# 4️⃣ Add RealYieldTrend (momentum of real yield)
# -------------------------------------------------------------
macro_daily["RealYieldTrend"] = macro_daily["RealYield"].diff(5).rolling(5).mean()

# -------------------------------------------------------------
# 5️⃣ Save cleaned macro
# -------------------------------------------------------------
macro_daily["date"] = macro_daily.index.date  # ✅ add here (for later merge)
out_path = os.path.join(DATA_DIR, "macro_context_v3.parquet")
macro_daily.to_parquet(out_path)
print(f"💾 Saved → {out_path} | cols={list(macro_daily.columns)}")

# -------------------------------------------------------------
# 6️⃣ Re-merge with FX closes for validation
# -------------------------------------------------------------
pairs = ["USD_JPY", "XAU_USD"]
for pair in pairs:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️ Missing FX file: {fx_path}")
        continue

    fx = pd.read_parquet(fx_path)
    fx = fx[~fx["synthetic"].astype(bool)] if "synthetic" in fx.columns else fx
    fx.index = pd.to_datetime(fx.index).tz_convert("UTC")
    fx["date"] = fx.index.date  # ✅ define before merge

    merged = pd.merge(
        fx[["close", "date"]],
        macro_daily[["date", "UST2Y", "JGB10Y", "BoJRate", "YieldSpread", "RealYield", "RealYieldTrend", "DXY"]],
        on="date", how="left"
    ).dropna()

    corr = merged[["close", "UST2Y", "JGB10Y", "BoJRate", "YieldSpread", "RealYield", "RealYieldTrend", "DXY"]].corr().round(2)
    print(f"\n📊 {pair} correlation matrix (validated):")
    print(corr)