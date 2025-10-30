#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.10 — Fetch Macro FX Drivers (UST yields, risk sentiment, energy)

Goal:
- Pull key macro time series from FRED for FX / gold regime modeling
- Build yield curve spreads and cross-country rate differentials
- Persist as a clean daily feature set we can join later with prices

Output:
- data/macro_fxdrivers.parquet

Update policy:
- If macro_fxdrivers.parquet already exists, we merge on 'date'
  and overwrite any overlapping dates with fresh data, then save.

Range:
- Hard-coded 2020-01-01 → today (UTC date at runtime)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================

START_DATE = "2020-01-01"

# "today" in UTC (we freeze this at runtime for reproducibility)
TODAY_UTC = datetime.now(timezone.utc).date().isoformat()

OUT_FILE = "data/macro_fxdrivers.parquet"
LOG_FILE = "logs/stage1_10_fetch_macro_fxdrivers.log"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series list:
# U.S. nominal yields
SERIES_MAP = {
    # U.S. Treasuries
    "DGS2":      "UST2Y",      # 2-Year Treasury Yield (%)
    "DGS5":      "UST5Y",      # 5-Year Treasury Yield (%)
    "DGS10":     "UST10Y",     # 10-Year Treasury Yield (%)

    # U.S. REAL yield (10y TIPS)
    "DFII10":    "UST10Y_REAL",  # 10-Year Treasury Inflation-Indexed Security, Constant Maturity (%)

    # Risk sentiment / liquidity
    "VIXCLS":    "VIX",        # CBOE Volatility Index
    "SP500":     "SPX",        # S&P 500 index level
    "DCOILWTICO":"WTI",        # Crude Oil Prices: WTI (USD/barrel)

    # NOTE: Cross-country 2Y yields:
    # There is no single universal standard series naming.
    # We'll TRY some common FRED tickers below.
    # If some fail to fetch we'll just skip gracefully.
    #
    # Germany 2y Bund Yield (use 'IRLTLT01DEM156N' is long-term, but not perfect 2y.
    # We'll attempt some known tickers; if 404 we'll warn.
    #
    # We'll include placeholders so pipeline is ready. You can wire better tickers later.
    "IRLTLT01DEM156N": "DE_LONG_RATE",   # Germany long-term government bond yield, %
    "IRLTLT01GBM156N": "UK_LONG_RATE",   # U.K. long-term govt bond yield, %
    "IRLTLT01JPM156N": "JP_LONG_RATE",   # Japan long-term govt bond yield, %
    # These are not strictly 2Y, they're generic long-term yields.
    # We'll still pull them so we can at least compute some US-vs-Other spread proxy.
}

# after fetch we'll build derived cols:
# - UST2Y_10Y_SPREAD   = UST10Y - UST2Y
# - UST10Y_REAL_DIFF   = UST10Y - UST10Y_REAL  (rough proxy for inflation expectation)
# - US_DE_SPREAD       = UST2Y - DE_LONG_RATE  (proxy rate diff vs Germany)
# - US_UK_SPREAD       = UST2Y - UK_LONG_RATE
# - US_JP_SPREAD       = UST2Y - JP_LONG_RATE
DERIVED_COLS = [
    ("UST2Y_10Y_SPREAD",      lambda df: df["UST10Y"] - df["UST2Y"]),
    ("UST10Y_REAL_DIFF",      lambda df: df["UST10Y"] - df["UST10Y_REAL"]),
    ("US_DE_SPREAD",          lambda df: df["UST2Y"] - df["DE_LONG_RATE"]),
    ("US_UK_SPREAD",          lambda df: df["UST2Y"] - df["UK_LONG_RATE"]),
    ("US_JP_SPREAD",          lambda df: df["UST2Y"] - df["JP_LONG_RATE"]),
]


SLEEP_BETWEEN_CALLS = 0.3  # polite pacing


# =========================
# UTILS
# =========================

def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def fred_fetch_series(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Hit FRED API for a given series, return DataFrame with:
    date (as datetime64[ns, UTC? we'll keep naive date col), value (float or NaN)
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=10)
    except Exception as e:
        log(f"❌ Request error for {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])

    if r.status_code != 200:
        log(f"❌ HTTP {r.status_code} for {series_id}")
        return pd.DataFrame(columns=["date", "value"])

    try:
        data = r.json()
    except Exception as e:
        log(f"❌ JSON decode error for {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])

    obs = data.get("observations", [])
    rows = []
    for o in obs:
        d = o.get("date")  # YYYY-MM-DD
        v = o.get("value")
        # FRED sometimes gives "." for missing
        if v in [".", None, ""]:
            val = None
        else:
            try:
                val = float(v)
            except:
                val = None
        rows.append({"date": d, "value": val})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        # Keep "date" as date (no tz). We will treat it as market calendar daily key.
        df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()

    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        log("❌ FRED_API_KEY missing in .env")
        sys.exit(1)

    log("🚀 Stage 1.10 — Fetching macro FX drivers from FRED")
    log(f"   Range: {START_DATE} → {TODAY_UTC}")

    # step 1: fetch all series
    series_frames = []
    for fred_id, nice_name in SERIES_MAP.items():
        log(f"↳ Fetch {fred_id} as {nice_name} ...")
        df_s = fred_fetch_series(
            series_id=fred_id,
            api_key=fred_key,
            start_date=START_DATE,
            end_date=TODAY_UTC,
        )
        if df_s.empty:
            log(f"⚠️  No data for {fred_id}")
        else:
            df_s = df_s.rename(columns={"value": nice_name})
            series_frames.append(df_s[["date", nice_name]])
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not series_frames:
        log("❌ No series fetched. Abort.")
        sys.exit(1)

    # step 2: outer join on date
    macro_df = series_frames[0]
    for other in series_frames[1:]:
        macro_df = pd.merge(macro_df, other, on="date", how="outer")

    # stable sort by date
    macro_df = macro_df.sort_values("date").reset_index(drop=True)

    # step 3: build derived columns
    for col_name, fn in DERIVED_COLS:
        try:
            macro_df[col_name] = fn(macro_df)
        except Exception as e:
            log(f"⚠️ Could not compute {col_name}: {e}")
            macro_df[col_name] = None

    # step 4: forward-fill some macro stuff (typical for yields, VIX etc.)
    # We do ffill because some series (like weekend, holidays) are missing.
    macro_df = macro_df.set_index("date").sort_index()
    macro_df = macro_df.ffill()

    # step 5: merge with existing file if present
    if os.path.exists(OUT_FILE):
        log(f"📂 Existing file found: {OUT_FILE}, merging ...")
        old_df = pd.read_parquet(OUT_FILE)
        # ensure datetime types align
        if "date" in old_df.columns:
            old_df["date"] = pd.to_datetime(old_df["date"], errors="coerce")
            old_df = old_df.dropna(subset=["date"]).set_index("date").sort_index()
        else:
            log("⚠️ Existing file missing 'date' column, will ignore old file.")
            old_df = pd.DataFrame()

        # combine: new overwrites old on overlapping dates
        combined = old_df.combine_first(macro_df)
        # BUT combine_first keeps old values first. We want NEW to win.
        # So let's just concat and drop duplicates keeping last:
        stacked = pd.concat([old_df, macro_df])
        stacked = stacked[~stacked.index.duplicated(keep="last")]
        macro_df = stacked.sort_index()

    # step 6: final save
    macro_df = macro_df.reset_index()
    macro_df = macro_df.rename(columns={"index": "date"})
    macro_df = macro_df.sort_values("date").reset_index(drop=True)

    macro_df.to_parquet(OUT_FILE)

    log(f"💾 Saved macro drivers → {OUT_FILE} ({len(macro_df)} rows)")
    log("📊 Columns: " + ", ".join(macro_df.columns))

    # small preview
    print(macro_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()