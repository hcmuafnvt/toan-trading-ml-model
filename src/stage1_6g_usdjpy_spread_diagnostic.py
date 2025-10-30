import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Stage 1.6g — USDJPY Macro Driver Diagnostic
# Goal:
#   Find which macro spread best explains USD/JPY
#   and lock that into our dataset as the canonical JPY driver.
# ======================================================

DATA_DIR = "data"
MACRO_V5 = os.path.join(DATA_DIR, "macro_context_v5.parquet")  # after 1.6f
PAIR_FILE = os.path.join(DATA_DIR, "USD_JPY_M5_clean.parquet")

print("🚀 Stage 1.6g — USDJPY Macro Driver Diagnostic")

# 1️⃣ Load macro dataset
macro = pd.read_parquet(MACRO_V5)
macro.index = pd.to_datetime(macro.index)

# make sure tz-aware
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

# We expect these columns to exist from earlier stages:
# - UST2Y
# - BoJRate
# - DGS10 (US 10y)
# - JGB10Y (JP 10y)
# - RealYield_shifted (US real yield, lag-adjusted)

needed_cols = ["UST2Y", "BoJRate", "DGS10", "JGB10Y", "RealYield_shifted"]
missing = [c for c in needed_cols if c not in macro.columns]
if missing:
    print(f"❌ Missing columns in macro_context_v5: {missing}")
    print("   You may need to re-run previous stages or ensure v5 inherited all columns.")
    # we'll still continue as far as we can

# Create date index for merge_asof
macro["date"] = pd.to_datetime(macro.index.date)
macro_daily = (
    macro
    .groupby("date")
    .agg({
        "UST2Y": "last",
        "BoJRate": "last",
        "DGS10": "last",
        "JGB10Y": "last",
        "RealYield_shifted": "last"
    })
    .dropna(how="all")
)

# 2️⃣ Build candidate spreads
macro_daily["Spr_short_US_JP"] = macro_daily["UST2Y"] - macro_daily["BoJRate"]
macro_daily["Spr_long_US_JP"]  = macro_daily["DGS10"] - macro_daily["JGB10Y"]
macro_daily["Spr_real_US_JP"]  = macro_daily["RealYield_shifted"] - macro_daily["JGB10Y"]

print("✅ Built candidate spreads: Spr_short_US_JP, Spr_long_US_JP, Spr_real_US_JP")

# 3️⃣ Load USD/JPY and downsample to daily mean close
if not os.path.exists(PAIR_FILE):
    raise FileNotFoundError(f"⚠️ Missing FX file {PAIR_FILE}")

fx = pd.read_parquet(PAIR_FILE)
if "synthetic" in fx.columns:
    fx = fx[~fx["synthetic"].astype(bool)]

fx.index = pd.to_datetime(fx.index)
if fx.index.tz is None:
    fx.index = fx.index.tz_localize("UTC")
else:
    fx.index = fx.index.tz_convert("UTC")

fx["date"] = pd.to_datetime(fx.index.date)
fx_daily = fx.groupby("date")["close"].mean().to_frame("USDJPY_close")

# 4️⃣ Merge FX with macro spreads using asof
# first ensure both indices are tz-naive or both tz-aware with same
fx_daily.index = pd.to_datetime(fx_daily.index).tz_localize("UTC")
macro_daily.index = pd.to_datetime(macro_daily.index).tz_localize("UTC")

merged = pd.merge_asof(
    fx_daily.sort_index(),
    macro_daily.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

# 5️⃣ Compute correlations
cands = ["Spr_short_US_JP", "Spr_long_US_JP", "Spr_real_US_JP"]
corrs = {}
for c in cands:
    if c in merged.columns:
        corrs[c] = merged["USDJPY_close"].corr(merged[c])
    else:
        corrs[c] = None

print("\n📊 Correlation vs USDJPY_close")
for k, v in corrs.items():
    if v is None:
        print(f"   {k}: missing")
    else:
        arrow = "↑ (should be positive)" if v > 0 else "↓"
        print(f"   {k:<20} corr = {v:.3f} {arrow}")

# 6️⃣ Pick best spread
valid_corrs = {k: v for k, v in corrs.items() if v is not None}
if valid_corrs:
    best_key = max(valid_corrs, key=lambda k: abs(valid_corrs[k]))
    print(f"\n🏁 Best driver for USD/JPY appears to be: {best_key} "
          f"with corr={valid_corrs[best_key]:.3f}")
else:
    print("\n❌ No valid correlations computed.")

print("\n🎯 Stage 1.6g complete — Use this spread as USDJPY macro driver going forward.")