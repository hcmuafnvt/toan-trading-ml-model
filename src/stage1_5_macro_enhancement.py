# =============================================================
# Stage 1.5 — Macro Enhancement (JGB10Y + Real Yield)
# =============================================================
import os, pandas as pd
from datetime import datetime
import pandas_datareader.data as web
import yfinance as yf

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "macro_context_v2.parquet")
os.makedirs(DATA_DIR, exist_ok=True)

START = "2023-01-01"
END   = datetime.utcnow().strftime("%Y-%m-%d")

print("🚀 Stage 1.5 — Enhancing macro dataset...")

# -------------------------------------------------------------
# 1️⃣ Load existing macro (from Stage 1.4)
# -------------------------------------------------------------
macro = pd.read_parquet(os.path.join(DATA_DIR, "macro_context.parquet"))
macro.columns = [c[1] if isinstance(c, tuple) else c for c in macro.columns]
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")
print(f"✅ Loaded base macro context: {list(macro.columns)}")

# -------------------------------------------------------------
# 2️⃣ Fetch JGB10Y (FRED Japan 10-year yields)
# -------------------------------------------------------------
try:
    jgb = web.DataReader("IRLTLT01JPM156N", "fred", START, END)
    jgb.columns = ["JGB10Y"]
    jgb.index = pd.to_datetime(jgb.index).tz_localize("UTC")
    print(f"✅ JGB10Y (FRED): {len(jgb):,} rows ({jgb.index.min().date()} → {jgb.index.max().date()})")
except Exception as e:
    print(f"⚠️  Failed JGB10Y (FRED): {e}")
    jgb = pd.DataFrame()

# -------------------------------------------------------------
# 3️⃣ Fetch DGS10 + CPIAUCSL (FRED)
# -------------------------------------------------------------
try:
    dgs10 = web.DataReader("DGS10", "fred", START, END)
    cpi = web.DataReader("CPIAUCSL", "fred", START, END)
    dgs10.columns = ["DGS10"]
    cpi.columns = ["CPIAUCSL"]
    dgs10.index = dgs10.index.tz_localize("UTC")
    cpi.index = cpi.index.tz_localize("UTC")
    cpi["CPI_YoY"] = cpi["CPIAUCSL"].pct_change(12) * 100
    real_yield = pd.concat([dgs10, cpi["CPI_YoY"]], axis=1)
    real_yield["RealYield"] = real_yield["DGS10"] - real_yield["CPI_YoY"]
    print(f"✅ RealYield: {len(real_yield):,} rows ({real_yield.index.min().date()} → {real_yield.index.max().date()})")
except Exception as e:
    print(f"⚠️  Failed RealYield: {e}")
    real_yield = pd.DataFrame()

# -------------------------------------------------------------
# 4️⃣ Merge all
# -------------------------------------------------------------
macro2 = macro.copy()
for add_df in [jgb, real_yield]:
    if not add_df.empty:
        macro2 = macro2.merge(add_df, left_index=True, right_index=True, how="left")
macro2 = macro2.ffill().astype("float32")

# Derived features
if "UST2Y" in macro2.columns and "JGB10Y" in macro2.columns:
    macro2["YieldSpread"] = macro2["UST2Y"] - macro2["JGB10Y"]

macro2.to_parquet(OUT_PATH)
print(f"\n💾 Saved enhanced macro context → {OUT_PATH} ({len(macro2):,} rows)")
print(macro2.tail())

# -------------------------------------------------------------
# 5️⃣ Quick QC correlation for USDJPY + XAUUSD
# -------------------------------------------------------------
for pair in ["USD_JPY", "XAU_USD"]:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️  Missing {fx_path}")
        continue
    fx = pd.read_parquet(os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet"))
    merged = fx.join(macro2, how="left").ffill()
    cols = [c for c in ["close", "UST2Y", "JGB10Y", "YieldSpread", "RealYield"] if c in merged.columns]
    corr = merged[cols].corr().round(2)
    print(f"\n📊 {pair} correlation matrix (enhanced):")
    print(corr)