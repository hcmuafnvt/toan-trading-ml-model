import pandas as pd
import os
from pandas_datareader import data as web
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Stage 1.6f — Add Relative Yield Spreads (US vs EU/UK/JP)
# ======================================================

DATA_DIR   = "data"
IN_FILE    = os.path.join(DATA_DIR, "macro_context_v4.parquet")
OUT_FILE   = os.path.join(DATA_DIR, "macro_context_v5.parquet")

print("🚀 Stage 1.6f — Building relative yield spreads (UST2Y vs foreign 2Y)")

# 1️⃣ Load macro_context_v4 (đã có UST2Y, BoJRate, RealYield_shifted,...)
macro = pd.read_parquet(IN_FILE)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

start_date = macro.index.min().tz_convert("UTC").tz_localize(None)
end_date   = macro.index.max().tz_convert("UTC").tz_localize(None)

print(f"✅ Loaded macro_context_v4.parquet: {len(macro):,} rows "
      f"({macro.index.min().date()} → {macro.index.max().date()})")

# 2️⃣ Fetch foreign 2Y yields from FRED (or closest proxy)
def fetch_fred(series_id, colname):
    try:
        df = web.DataReader(series_id, "fred", start_date, end_date)
        df.columns = [colname]
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        print(f"   ✅ FRED {colname}: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")
        return df
    except Exception as e:
        print(f"   ⚠️ FRED fetch failed for {colname} ({series_id}): {e}")
        return None

# These series IDs are our best guess for short-term sovereign yields.
# Note: if any fail we'll fallback later.
de2y = fetch_fred("IRLTLT01DEM156N", "DE2Y")   # Germany
uk2y = fetch_fred("IRLTLT01GBM156N", "UK2Y")   # United Kingdom
jp2y = fetch_fred("IRLTLT01JPM156N", "JP2Y")   # Japan

# 3️⃣ Merge those yields (left join daily, then ffill)
for df in [de2y, uk2y, jp2y]:
    if df is not None:
        macro = macro.merge(df, left_index=True, right_index=True, how="left")

macro[["DE2Y","UK2Y","JP2Y"]] = macro[["DE2Y","UK2Y","JP2Y"]].ffill()

# 4️⃣ Build spreads
#    Spread_US_EU_2Y = UST2Y - DE2Y
#    Spread_US_UK_2Y = UST2Y - UK2Y
#    Spread_US_JP_2Y = UST2Y - JP2Y (fallback BoJRate if JP2Y missing)

if "DE2Y" in macro.columns:
    macro["Spread_US_EU_2Y"] = macro["UST2Y"] - macro["DE2Y"]
else:
    macro["Spread_US_EU_2Y"] = None
    print("   ⚠️ Missing DE2Y, Spread_US_EU_2Y set to None")

if "UK2Y" in macro.columns:
    macro["Spread_US_UK_2Y"] = macro["UST2Y"] - macro["UK2Y"]
else:
    macro["Spread_US_UK_2Y"] = None
    print("   ⚠️ Missing UK2Y, Spread_US_UK_2Y set to None")

if "JP2Y" in macro.columns and macro["JP2Y"].notna().sum() > 0:
    macro["Spread_US_JP_2Y"] = macro["UST2Y"] - macro["JP2Y"]
elif "BoJRate" in macro.columns:
    # fallback: use BoJRate as Japan short end
    macro["Spread_US_JP_2Y"] = macro["UST2Y"] - macro["BoJRate"]
    print("   ℹ️ Using BoJRate as JP short-end proxy for Spread_US_JP_2Y")
else:
    macro["Spread_US_JP_2Y"] = None
    print("   ⚠️ Missing JP2Y & BoJRate, Spread_US_JP_2Y set to None")

# 5️⃣ Save macro_context_v5
macro.to_parquet(OUT_FILE)
print(f"\n💾 Saved → {OUT_FILE}")
print("    Columns now include:", [c for c in macro.columns if "Spread_" in c or c in ["DE2Y","UK2Y","JP2Y"]])

# 6️⃣ Validate correlation per pair using these spreads
pairs = [
    ("EUR_USD", "Spread_US_EU_2Y"),
    ("GBP_USD", "Spread_US_UK_2Y"),
    ("USD_JPY", "Spread_US_JP_2Y"),
]

for pair, spread_col in pairs:
    fx_file = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_file):
        print(f"\n⚠️ Skipping {pair}: missing {fx_file}")
        continue

    fx = pd.read_parquet(fx_file)
    if "synthetic" in fx.columns:
        fx = fx[~fx["synthetic"].astype(bool)]
    fx.index = pd.to_datetime(fx.index).tz_convert("UTC")
    fx["date"] = pd.to_datetime(fx.index.date)

    daily_close = fx.groupby("date")["close"].mean().to_frame("close")

    # align macro daily using asof (backward fill by date)
    merged = pd.merge_asof(
        daily_close.sort_index(),
        macro[[spread_col]].sort_index().rename(columns={spread_col: "spread"}),
        left_index=True,
        right_index=True,
        direction="backward"
    ).dropna()

    corr = merged["close"].corr(merged["spread"])
    print(f"\n📊 {pair} vs {spread_col}")
    print(f"   Corr = {corr:.3f} "
          f"{'↓ (should be negative)' if pair in ['EUR_USD','GBP_USD'] else '↑ (should be positive)'}")