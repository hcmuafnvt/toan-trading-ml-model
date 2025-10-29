# =============================================================
# Stage 1.4 — Macro ↔ FX Merge + QC  (AlphaForge)
# =============================================================
import os, pandas as pd
import pandas_datareader.data as web

DATA_DIR = "data"
OUT_DIR  = os.path.join(DATA_DIR, "merged_macro_FX")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1️⃣ Fetch thêm DXY_DEV (FRED DTWEXAFEGS)
# -------------------------------------------------------------
print("🚀 Stage 1.4 — Fetching DXY_DEV (Advanced FX economies)...")
try:
    dxy_dev = web.DataReader("DTWEXAFEGS", "fred", "2023-01-01")
    dxy_dev.columns = ["DXY_DEV"]
    dxy_dev.index = pd.to_datetime(dxy_dev.index).tz_localize("UTC")
    print(f"✅ DXY_DEV loaded: {len(dxy_dev):,} rows ({dxy_dev.index.min().date()} → {dxy_dev.index.max().date()})")
except Exception as e:
    print(f"⚠️  Failed to fetch DXY_DEV: {e}")
    dxy_dev = pd.DataFrame()

# -------------------------------------------------------------
# 2️⃣ Load macro_context.parquet  + combine DXY_DEV
# -------------------------------------------------------------
macro = pd.read_parquet(os.path.join(DATA_DIR, "macro_context.parquet"))
macro.columns = [c[1] if isinstance(c, tuple) else c for c in macro.columns]  # clean colnames
if not dxy_dev.empty:
    macro = macro.merge(dxy_dev, left_index=True, right_index=True, how="ffill")
macro = macro.ffill().astype("float32")

print(f"✅ Macro columns: {list(macro.columns)} | {len(macro):,} rows")

# -------------------------------------------------------------
# 3️⃣ Merge với từng cặp FX
# -------------------------------------------------------------
pairs = ["GBP_USD", "EUR_USD", "USD_JPY", "XAU_USD"]
for pair in pairs:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️  Missing {fx_path}")
        continue

    fx = pd.read_parquet(fx_path)
    fx.index = pd.to_datetime(fx.index)
    merged = fx.join(macro, how="left").ffill()

    out_path = os.path.join(OUT_DIR, f"{pair}_M5_macro.parquet")
    merged.to_parquet(out_path)

    # ---------------------------------------------------------
    # QC correlation logic
    # ---------------------------------------------------------
    corr = merged[["close", "DXY", "DXY_DEV", "UST2Y", "SPX", "VIX"]].corr().round(2)
    print(f"\n📊 {pair} correlation matrix:")
    print(corr)

    # save QC log
    qc_path = os.path.join(OUT_DIR, f"qc_{pair}.txt")
    with open(qc_path, "w") as f:
        f.write(str(corr))

print("\n🎯 Stage 1.4 completed — all pairs merged with macro context.")