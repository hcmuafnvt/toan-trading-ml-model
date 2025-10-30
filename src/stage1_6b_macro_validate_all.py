import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ================================
# Stage 1.6b — Extended Macro Validation (all pairs)
# ================================
DATA_DIR = "data"
MACRO_PATH = os.path.join(DATA_DIR, "macro_context_v3.parquet")

print("🚀 Stage 1.6b — Extended Macro Validation for 4 pairs")

# -------------------------------------------------------------
# 1️⃣ Load macro dataset
# -------------------------------------------------------------
macro = pd.read_parquet(MACRO_PATH)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

macro["date"] = pd.to_datetime(macro.index.date)
print(f"✅ Loaded macro_context_v3.parquet: {len(macro):,} rows | cols={list(macro.columns)}")

# -------------------------------------------------------------
# 2️⃣ Define pairs to test
# -------------------------------------------------------------
pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]

# -------------------------------------------------------------
# 3️⃣ Validate each pair
# -------------------------------------------------------------
for pair in pairs:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️ Missing FX file: {fx_path}")
        continue

    fx = pd.read_parquet(fx_path)
    fx = fx[~fx["synthetic"].astype(bool)] if "synthetic" in fx.columns else fx
    fx.index = pd.to_datetime(fx.index).tz_convert("UTC")
    fx["date"] = pd.to_datetime(fx.index.date)

    # merge_asof theo ngày gần nhất
    merged = pd.merge_asof(
        fx.sort_values("date"),
        macro.sort_values("date"),
        on="date",
        direction="backward"
    ).dropna()

    corr = merged[[
        "close", "DXY", "UST2Y", "JGB10Y", "BoJRate",
        "YieldSpread", "RealYield", "RealYieldTrend"
    ]].corr().round(2)

    print(f"\n📊 {pair} correlation matrix:")
    print(corr)

print("\n🎯 Stage 1.6b completed — macro consistency check finished.")