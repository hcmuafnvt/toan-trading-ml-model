# =============================================================
# Stage 1.3 — Macro Context Data Fetcher (AlphaForge, fixed)
# =============================================================
import pandas as pd
from datetime import datetime
import pandas_datareader.data as web
import yfinance as yf
import os

START = "2023-01-01"
END = datetime.utcnow().strftime("%Y-%m-%d")
OUT_PATH = "data/macro_context.parquet"
os.makedirs("data", exist_ok=True)

def fetch_fred(series_id, name):
    """Fetch single series from FRED"""
    try:
        df = web.DataReader(series_id, "fred", START, END)
        df.columns = [name]
        print(f"✅ FRED {name}: {len(df):,} rows ({df.index.min().date()} → {df.index.max().date()})")
        return df
    except Exception as e:
        print(f"⚠️  FRED fetch failed for {name}: {e}")
        return pd.DataFrame()

def fetch_yahoo(symbol, name):
    """Fetch daily close from Yahoo Finance"""
    try:
        df = yf.download(symbol, start=START, end=END, progress=False)
        price_col = "Close" if "Close" in df.columns else df.columns[0]
        df = df[[price_col]].rename(columns={price_col: name})
        print(f"✅ Yahoo {name}: {len(df):,} rows ({df.index.min().date()} → {df.index.max().date()})")
        return df
    except Exception as e:
        print(f"⚠️  Yahoo fetch failed for {name}: {e}")
        return pd.DataFrame()

print("🚀 Stage 1.3 — Fetching macro context data...")

# --- FRED ---
fred_data = [
    ("DTWEXBGS", "DXY"),   # Broad USD index
    ("DGS2", "UST2Y"),     # 2-year Treasury yield
]
fred_df = pd.concat([fetch_fred(fid, name) for fid, name in fred_data], axis=1)

# --- Yahoo ---
yahoo_data = [
    ("^GSPC", "SPX"),      # S&P500
    ("^VIX", "VIX"),       # Volatility index
]
yahoo_df = pd.concat([fetch_yahoo(sym, name) for sym, name in yahoo_data], axis=1)

# --- Merge ---
macro = pd.concat([fred_df, yahoo_df], axis=1)
macro = macro.sort_index().ffill().dropna(how="all")
macro.index = pd.to_datetime(macro.index).tz_localize("UTC")
macro = macro.astype("float32")

# --- Expand to M5 timeline ---
full_index = pd.date_range(macro.index.min(), macro.index.max(), freq="5min", tz="UTC")
macro = macro.reindex(full_index, method="ffill")

coverage = len(macro) / len(full_index) * 100
macro.to_parquet(OUT_PATH)

print(f"\n✅ Saved {len(macro):,} rows → {OUT_PATH} | coverage={coverage:.2f}%")
print(macro.tail())