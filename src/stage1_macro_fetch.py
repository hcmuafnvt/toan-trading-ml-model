# =============================================================
# Stage 1.3 — Macro Context Data Fetcher (AlphaForge)
# =============================================================
import pandas as pd, yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
import pandas_datareader.data as web
import os

START = "2023-01-01"
END   = datetime.utcnow().strftime("%Y-%m-%d")

OUT_PATH = "data/macro_context.parquet"
os.makedirs("data", exist_ok=True)

def fetch_fred(series):
    try:
        df = web.DataReader(series, "fred", START, END)
        df.columns = [series]
        return df
    except Exception as e:
        print(f"⚠️  FRED fetch failed for {series}: {e}")
        return pd.DataFrame()

def fetch_yahoo(symbol):
    try:
        df = yf.download(symbol, start=START, end=END, progress=False)[["Adj Close"]]
        df.columns = [symbol]
        return df
    except Exception as e:
        print(f"⚠️  Yahoo fetch failed for {symbol}: {e}")
        return pd.DataFrame()

print("🚀 Stage 1.3 — Fetching macro context data...")

# --- FRED series ---
fred_series = {
    "DXY": "DTWEXBGS",   # Broad USD index
    "UST2Y": "DGS2"      # 2-year yield
}
fred_df = pd.concat({k: fetch_fred(v) for k,v in fred_series.items()}, axis=1)

# --- Yahoo series ---
yahoo_symbols = ["^GSPC", "^VIX"]
yahoo_df = pd.concat({s.replace("^",""): fetch_yahoo(s) for s in yahoo_symbols}, axis=1)

# --- Merge & QC ---
macro = pd.concat([fred_df, yahoo_df], axis=1)
macro = macro.ffill().dropna(how="all")
macro.index = pd.to_datetime(macro.index).tz_localize("UTC")

# Normalize columns
macro.columns = ["DXY", "UST2Y", "SPX", "VIX"]
macro = macro.astype("float32")

# --- Expand to M5 timeline (align with price data) ---
full_index = pd.date_range(start=macro.index.min(), end=macro.index.max(), freq="5min", tz="UTC")
macro = macro.reindex(full_index, method="ffill")

# --- QC report ---
coverage = len(macro) / len(full_index) * 100
print(f"✅ Saved {len(macro):,} rows → {OUT_PATH} | coverage={coverage:.2f}%")
print(macro.tail())

macro.to_parquet(OUT_PATH)