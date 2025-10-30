import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# ======================================================
# Stage 1.7 — Macro Finalization (master context)
#
# Goal:
#   - Consolidate all macro factors that are PROVEN useful
#     for EUR/USD, GBP/USD, XAU/USD.
#   - Keep placeholders for USD/JPY regime features.
#   - Save one master parquet for Stage 2 feature merge.
#
# Inputs we assume exist:
#   data/macro_context_v6.parquet
#   data/macro_context_v5.parquet  (for spreads US vs EU/UK/JP 2Y)
#   data/USD_JPY_M5_clean.parquet  etc. (used only for QC here)
#
# Output:
#   data/macro_context_master.parquet
#
# What goes in master:
#   - DXY
#   - UST2Y
#   - DGS10 (US10Y)
#   - JGB10Y
#   - BoJRate
#   - YieldSpread (US10Y - JP10Y)  [JPY_driver candidate, weak but keep for reference]
#   - Spread_US_EU_2Y, Spread_US_UK_2Y  (VERY strong for EUR, GBP)
#   - RealYield_shifted (with +3d lead logic we established)
#   - RealYieldTrend
#   - Risk sentiment: SPX, VIX
#   - placeholder column JPY_regime_flag (NaN for now, Stage 1.8 will fill)
#
# Also:
#   - We'll reindex to daily UTC end-of-day style.
#   - We'll print final sanity correlations for EUR/USD, GBP/USD, XAU/USD.
# ======================================================

DATA_DIR = "data"

MACRO_V6_FILE = os.path.join(DATA_DIR, "macro_context_v6.parquet")
MACRO_V5_FILE = os.path.join(DATA_DIR, "macro_context_v5.parquet")

PAIR_FILES = {
    "EUR_USD": os.path.join(DATA_DIR, "EUR_USD_M5_clean.parquet"),
    "GBP_USD": os.path.join(DATA_DIR, "GBP_USD_M5_clean.parquet"),
    "XAU_USD": os.path.join(DATA_DIR, "XAU_USD_M5_clean.parquet"),
    "USD_JPY": os.path.join(DATA_DIR, "USD_JPY_M5_clean.parquet"),
}

print("🚀 Stage 1.7 — Macro Finalization (master context)")

# 1️⃣ Load macro v6 (most up-to-date, has BoJRate, RealYield_shifted, etc.)
macro6 = pd.read_parquet(MACRO_V6_FILE)
macro6.index = pd.to_datetime(macro6.index)
if macro6.index.tz is None:
    macro6.index = macro6.index.tz_localize("UTC")
else:
    macro6.index = macro6.index.tz_convert("UTC")

# We expect these columns in v6:
must_cols_v6 = [
    "DXY",
    "UST2Y",
    "DGS10",
    "JGB10Y",
    "BoJRate",
    "YieldSpread",
    "RealYield",
    "RealYield_shifted",
    "RealYieldTrend",
    "('SPX', '^GSPC')",
    "('VIX', '^VIX')",
]
missing_v6 = [c for c in must_cols_v6 if c not in macro6.columns]
if missing_v6:
    print(f"⚠️  Warning: missing in macro_context_v6: {missing_v6}")

# rename SPX/VIX for cleaner downstream usage
macro6 = macro6.rename(columns={
    "('SPX', '^GSPC')": "SPX",
    "('VIX', '^VIX')" : "VIX"
})

# 2️⃣ Load macro v5 (this has country 2Y spreads we computed in 1.6f)
macro5 = pd.read_parquet(MACRO_V5_FILE)
macro5.index = pd.to_datetime(macro5.index)
if macro5.index.tz is None:
    macro5.index = macro5.index.tz_localize("UTC")
else:
    macro5.index = macro5.index.tz_convert("UTC")

# columns we need from v5:
need_from_v5 = [
    "Spread_US_EU_2Y",  # explains EUR/USD (corr ~ -0.70)
    "Spread_US_UK_2Y",  # explains GBP/USD (corr ~ -0.78)
    "Spread_US_JP_2Y",  # we keep it for audit, even though it's weak
    "DE2Y",
    "UK2Y",
    "JP2Y",
]
keep_v5 = [c for c in need_from_v5 if c in macro5.columns]
macro5_slim = macro5[keep_v5].copy()

# 3️⃣ Merge v6 + v5 (outer join then ffill to align)
macro_merged = pd.concat([macro6, macro5_slim], axis=1)

# forward fill slow-moving stuff like yields, central bank rates
macro_merged = macro_merged.sort_index().ffill()

# 4️⃣ Add JPY_regime_flag placeholder (Stage 1.8 will populate with intervention/risk regime)
macro_merged["JPY_regime_flag"] = pd.NA

# 5️⃣ Resample to daily end-of-day.
# For FX training later we won't inject per-minute macro — we'll join daily state.
macro_daily = macro_merged.resample("1D").last()

# 6️⃣ Save master macro for Stage 2+
MASTER_OUT = os.path.join(DATA_DIR, "macro_context_master.parquet")
macro_daily.to_parquet(MASTER_OUT)

print(f"💾 Saved macro master → {MASTER_OUT}")
print(f"   Columns in master ({len(macro_daily.columns)} total):")
print("   " + ", ".join(macro_daily.columns))

# 7️⃣ Quality check: for each pair (except USD/JPY),
#    confirm that macro drivers still correlate in correct direction.
def load_pair_daily(pair_name, pair_file):
    if not os.path.exists(pair_file):
        print(f"⚠️ Missing {pair_file}, skip QC for {pair_name}")
        return None
    df = pd.read_parquet(pair_file)
    if "synthetic" in df.columns:
        df = df[~df["synthetic"].astype(bool)]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    px_daily = df["close"].resample("1D").mean().to_frame(f"{pair_name}_close")
    return px_daily

pairs_daily = {}
for pair, fpath in PAIR_FILES.items():
    pairs_daily[pair] = load_pair_daily(pair, fpath)

def corr_driver(pair, price_col, macro_col, desc, expect_sign):
    if pairs_daily[pair] is None:
        return None
    merged = pd.merge_asof(
        pairs_daily[pair].sort_index(),
        macro_daily[[macro_col]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    ).dropna()
    if merged.empty:
        return None
    c = merged[price_col].corr(merged[macro_col])
    arrow = "↑" if c > 0 else "↓"
    ok_dir = (c < 0 and expect_sign == "neg") or (c > 0 and expect_sign == "pos")
    status = "✅" if ok_dir else "⚠️"
    print(f"   {status} {pair} vs {desc}: corr={c:.3f} {arrow} (expect {expect_sign})")
    return c

print("\n🔎 Final sanity correlation check (daily):")

# EUR/USD should move DOWN when US-EU spread widens -> negative corr
corr_driver("EUR_USD",
            "EUR_USD_close",
            "Spread_US_EU_2Y",
            "US vs EU 2Y spread (Spread_US_EU_2Y)",
            expect_sign="neg")

# GBP/USD should move DOWN when US-UK spread widens -> negative corr
corr_driver("GBP_USD",
            "GBP_USD_close",
            "Spread_US_UK_2Y",
            "US vs UK 2Y spread (Spread_US_UK_2Y)",
            expect_sign="neg")

# XAU/USD should move DOWN when RealYield_shifted is high -> negative corr
corr_driver("XAU_USD",
            "XAU_USD_close",
            "RealYield_shifted",
            "US real yield (+3d shifted)",
            expect_sign="neg")

# USD/JPY check only for log (we know it's weak, but we log anyway)
corr_driver("USD_JPY",
            "USD_JPY_close",
            "YieldSpread",  # from v6: US10Y - JP10Y basically
            "US vs JP 10Y spread (YieldSpread)",
            expect_sign="pos")

print("\n🎯 Stage 1.7 complete — macro_context_master.parquet is ready for Stage 2.")
print("   Note: USD_JPY marked as regime-driven; will get JPY_regime_flag in Stage 1.8 (event layer).")