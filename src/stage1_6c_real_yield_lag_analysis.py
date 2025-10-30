import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Stage 1.6c — RealYield Lead-Lag Analysis
# ======================================================
DATA_DIR = "data"
MACRO_PATH = os.path.join(DATA_DIR, "macro_context_v3.parquet")

print("🚀 Stage 1.6c — RealYield Lead-Lag Correlation Analysis")

# 1️⃣ Load macro dataset
macro = pd.read_parquet(MACRO_PATH)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")
macro["date"] = pd.to_datetime(macro.index.date)
macro = macro.drop_duplicates("date").set_index("date")

pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]

# 2️⃣ Define lag window
LAGS = range(-3, 4)  # -3 to +3 days

results = {}

# 3️⃣ Compute correlation for each pair
for pair in pairs:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️ Missing FX file: {fx_path}")
        continue

    fx = pd.read_parquet(fx_path)
    fx = fx[~fx["synthetic"].astype(bool)] if "synthetic" in fx.columns else fx
    fx.index = pd.to_datetime(fx.index).tz_convert("UTC")
    fx["date"] = pd.to_datetime(fx.index.date)
    daily = fx.groupby("date")["close"].mean()  # daily mean close

    merged = pd.merge_asof(
        daily.sort_index().to_frame("close"),
        macro[["RealYield"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    ).dropna()

    lag_corr = {}
    for lag in LAGS:
        shifted = merged["RealYield"].shift(lag)
        corr = merged["close"].corr(shifted)
        lag_corr[lag] = round(corr, 3)

    results[pair] = lag_corr

# 4️⃣ Print table
table = pd.DataFrame(results)
print("\n📊 Lead-Lag Correlation Table (RealYield vs Close)")
print(table)

# 5️⃣ Highlight best lag
best_lags = {pair: max(vals, key=lambda k: abs(vals[k])) for pair, vals in results.items()}
print("\n🏁 Best lag by absolute correlation:")
for pair, lag in best_lags.items():
    sign = "↑" if results[pair][lag] > 0 else "↓"
    print(f"   {pair:<8} → lag {lag:+} days | corr={results[pair][lag]:+.3f} {sign}")