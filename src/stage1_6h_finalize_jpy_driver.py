import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Stage 1.6h — Finalize JPY Macro Driver
# Goal:
#   Add the confirmed JPY_driver (DGS10 - JGB10Y)
#   to macro dataset and validate correlation once more.
# ======================================================

DATA_DIR = "data"
IN_FILE = os.path.join(DATA_DIR, "macro_context_v5.parquet")
OUT_FILE = os.path.join(DATA_DIR, "macro_context_v6.parquet")
PAIR_FILE = os.path.join(DATA_DIR, "USD_JPY_M5_clean.parquet")

print("🚀 Stage 1.6h — Finalize JPY Macro Driver")

# 1️⃣ Load macro data
macro = pd.read_parquet(IN_FILE)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

for col in ["DGS10", "JGB10Y"]:
    if col not in macro.columns:
        raise ValueError(f"❌ Missing required column '{col}' in {IN_FILE}")

# 2️⃣ Create JPY driver
macro["JPY_driver"] = macro["DGS10"] - macro["JGB10Y"]
print("✅ Added JPY_driver = DGS10 - JGB10Y")

# 3️⃣ Merge with USDJPY close to validate
if not os.path.exists(PAIR_FILE):
    print(f"⚠️  Missing FX file {PAIR_FILE} — skipping correlation check")
else:
    fx = pd.read_parquet(PAIR_FILE)
    if "synthetic" in fx.columns:
        fx = fx[~fx["synthetic"].astype(bool)]

    fx.index = pd.to_datetime(fx.index)
    if fx.index.tz is None:
        fx.index = fx.index.tz_localize("UTC")
    else:
        fx.index = fx.index.tz_convert("UTC")

    # Downsample to daily mean
    fx_daily = fx.resample("1D")["close"].mean().dropna().to_frame("USDJPY_close")

    macro_daily = macro.resample("1D")["JPY_driver"].last().dropna().to_frame("JPY_driver")

    # Align indices for asof merge
    fx_daily = fx_daily.sort_index()
    macro_daily = macro_daily.sort_index()

    merged = pd.merge_asof(
        fx_daily,
        macro_daily,
        left_index=True,
        right_index=True,
        direction="backward"
    ).dropna()

    corr = merged["USDJPY_close"].corr(merged["JPY_driver"])
    arrow = "↑ (should be positive)" if corr > 0 else "↓"
    print(f"📊 Correlation (USDJPY vs JPY_driver): {corr:.3f} {arrow}")

# 4️⃣ Save output
macro.to_parquet(OUT_FILE)
print(f"💾 Saved → {OUT_FILE}")
print("🎯 Stage 1.6h complete — USD/JPY driver finalized and validated.")