# ================================================
# FETCH_REGIME_DATA.PY
# Tải DXY và VIX (daily close) từ Yahoo Finance
# Lưu file CSV để dùng cho Stage 3.2 Regime Adaptive
# ================================================

import yfinance as yf
import pandas as pd
from datetime import datetime

# -------- CONFIG --------
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"

OUT_DXY = "data/DXY_daily.csv"
OUT_VIX = "data/VIX_daily.csv"

# def fetch_symbol(symbol: str, name: str):
#     print(f"⏳ Downloading {name} ({symbol}) ...")
#     df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
#     if df.empty:
#         print(f"❌ No data for {symbol}")
#         return None
#     df = df[["Close"]].rename(columns={"Close": name})
#     df.index = pd.to_datetime(df.index)
#     print(f"✅ {name}: {len(df)} rows from {df.index.min().date()} → {df.index.max().date()}")
#     return df

# # --- Main ---
# if __name__ == "__main__":
#     dxy = fetch_symbol("DX-Y.NYB", "DXY")   # US Dollar Index
#     vix = fetch_symbol("^VIX", "VIX")       # Volatility Index (S&P500 implied vol)

#     if dxy is not None and vix is not None:
#         df = pd.concat([dxy, vix], axis=1)
#         df = df.ffill().dropna()

#         # Normalize DXY change %
#         df["DXY_Change"] = df["DXY"].pct_change() * 100

#         # Save separately and combined
#         dxy.to_csv(OUT_DXY)
#         vix.to_csv(OUT_VIX)
#         df.to_csv("data/regime_dxy_vix.csv")

#         print("\n✅ Saved:")
#         print(f" - {OUT_DXY}")
#         print(f" - {OUT_VIX}")
#         print(" - data/regime_dxy_vix.csv")
#         print(df.tail())
#     else:
#         print("⚠️ Failed to download one or more symbols.")
        
        

# df = pd.read_csv("data/regime_dxy_vix.csv", header=[0, 1], index_col=0)
# df.columns = [c[1] if c[1] else c[0] for c in df.columns]  # flatten columns
# df.index = pd.to_datetime(df.index)
# df = df.rename(columns={"DX-Y.NYB": "DXY", "^VIX": "VIX"})
# df.to_csv("data/regime_dxy_vix.csv")

# print("✅ Cleaned regime_dxy_vix.csv saved.")
# print(df.tail())


df = pd.read_csv("data/regime_dxy_vix.csv", index_col=0)
df = df.rename(columns={"Unnamed: 3_level_1": "DXY_Change"})
df.to_csv("data/regime_dxy_vix.csv")

print("✅ Final clean saved:")
print(df.tail())