#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.11 — USD Liquidity & Funding Conditions

What this does:
- Pull Fed/system liquidity + funding cost series from FRED:
    WALCL            (Fed balance sheet, $ billions)
    RRPONTSYD        (Reverse Repo Facility usage, $ billions)
    TB3MS            (3-Month T-Bill yield, %)
    SOFR             (Secured Overnight Financing Rate, %)
    EFFR             (Effective Fed Funds Rate, %)

- Compute derived stress / liquidity metrics:
    Fed_BS_Delta_7d
    RRP_Delta_7d
    SOFR_EFFR_SPREAD
    TBILL3M_MINUS_FEDFUNDS
    LIQ_COMPOSITE  (higher = tighter USD liquidity)

- Append/merge with any existing data/liquidity_funding.parquet

Output:
    data/liquidity_funding.parquet
    logs/stage1_11_fetch_liquidity.log
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

# ------------------
# CONFIG
# ------------------

START_DATE = "2020-01-01"   # earliest we care
END_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

OUT_FILE_PARQUET = "data/liquidity_funding.parquet"
LOG_FILE         = "logs/stage1_11_fetch_liquidity.log"

FRED_SERIES = {
    "WALCL": "Fed_BalanceSheet",      # Fed balance sheet (weekly)
    "RRPONTSYD": "RRP_Usage",         # Reverse Repo usage (daily)
    "DTB3": "TBILL3M",                # ⚡ 3M T-Bill secondary market rate (daily)
    "SOFR": "SOFR",                   # Secured Overnight Financing Rate
    "EFFR": "EFFR",                   # Effective Fed Funds Rate
}

def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def fetch_fred_series(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull one FRED series as a DataFrame with columns: date, <series_id>
    We'll coerce value to float.
    """
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&observation_start={start_date}"
        f"&observation_end={end_date}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED {series_id} failed HTTP {r.status_code}")
    js = r.json()
    obs = js.get("observations", [])
    rows = []
    for o in obs:
        d = o["date"]  # '2025-10-29'
        v = o["value"] # string, might be '.'
        try:
            val = float(v)
        except:
            val = None
        rows.append({"date": d, series_id: val})
    return pd.DataFrame(rows)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df with columns:
        date, Fed_BalanceSheet, RRP_Usage, TBILL3M, SOFR, EFFR
    Output same df + engineered cols:
        Fed_BS_Delta_7d
        RRP_Delta_7d
        SOFR_EFFR_SPREAD
        TBILL3M_MINUS_FEDFUNDS
        LIQ_COMPOSITE
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Convert billions-like series: WALCL, RRP are already billions in FRED.
    # We'll use simple diffs over 7 calendar days (not trading days).
    df["Fed_BS_Delta_7d"] = df["Fed_BalanceSheet"].diff(7)
    df["RRP_Delta_7d"]    = df["RRP_Usage"].diff(7)

    # Spreads
    df["SOFR_EFFR_SPREAD"] = df["SOFR"] - df["EFFR"]
    df["TBILL3M_MINUS_FEDFUNDS"] = df["TBILL3M"] - df["EFFR"]

    # Liquidity composite:
    # intuition:
    #   QT (Fed_BS_Delta_7d negative) = tighter USD, we want higher score
    #   RRP_Delta_7d negative (cash leaving RRP to market) = easier liquidity, so *tightness* is opposite
    #   SOFR_EFFR_SPREAD high = stress
    #
    # We'll z-score each subcomponent then average.
    def zscore_col(s):
        return (s - s.mean()) / s.std(ddof=0)

    # Tightening proxy 1: -Fed_BS_Delta_7d  (if Fed balance sheet shrinking => positive tightness)
    t1 = -df["Fed_BS_Delta_7d"]
    # Tightening proxy 2: -RRP_Delta_7d (if RRP draining => liquidity easier, so negative means looser,
    #                                    we invert so positive = tighter)
    t2 = -df["RRP_Delta_7d"]
    # Tightening proxy 3: SOFR_EFFR_SPREAD
    t3 = df["SOFR_EFFR_SPREAD"]

    # z-score with fallback if all NaN or std=0
    def safe_z(x):
        if x.isna().all() or x.std(ddof=0) == 0:
            return pd.Series([0.0]*len(x), index=x.index)
        return (x - x.mean()) / x.std(ddof=0)

    z1 = safe_z(t1)
    z2 = safe_z(t2)
    z3 = safe_z(t3)

    df["LIQ_COMPOSITE"] = (z1 + z2 + z3) / 3.0

    return df

def main():
    ensure_dirs()
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        log("❌ Missing FRED_API_KEY in env")
        sys.exit(1)

    log("🚀 Stage 1.11 — Fetching USD liquidity & funding data")
    log(f"   Range: {START_DATE} → {END_DATE}")

    # if we already have a parquet, we'll only refresh from the last saved date forward
    if os.path.exists(OUT_FILE_PARQUET):
        prev = pd.read_parquet(OUT_FILE_PARQUET)
        prev["date"] = pd.to_datetime(prev["date"], utc=True).dt.date
        last_date = prev["date"].max()
        new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        effective_start = min(new_start, START_DATE)  # just in case
        log(f"📂 Existing file found ({len(prev)} rows). Will fetch from {effective_start}.")
    else:
        prev = None
        effective_start = START_DATE
        log("📂 No existing file, full fetch.")

    # download all requested series
    dfs = []
    for series_id, nice_name in FRED_SERIES.items():
        log(f"↳ Fetch {series_id} as {nice_name} ...")
        dfi = fetch_fred_series(series_id, fred_key, effective_start, END_DATE)
        # rename column to nice_name
        dfi = dfi.rename(columns={series_id: nice_name})
        dfs.append(dfi)
        time.sleep(0.4)

    # outer merge on 'date'
    merged = None
    for dfi in dfs:
        if merged is None:
            merged = dfi
        else:
            merged = merged.merge(dfi, on="date", how="outer")

    # cast date -> datetime.date (UTC naive date ok)
    merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.date

    # if prev exists, concat and drop duplicates
    if prev is not None:
        big = pd.concat([prev, merged], ignore_index=True)
    else:
        big = merged.copy()

    # drop dups on date keeping last
    big = big.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # --- ✅ Normalize & forward-fill (Stage 1.11 v2 improvement) ---
    big["date"] = pd.to_datetime(big["date"])
    big = big.set_index("date").asfreq("B")  # resample to business days

    # forward-fill key economic series (Fed updates weekly / no weekend data)
    for col in ["Fed_BalanceSheet", "RRP_Usage", "TBILL3M", "SOFR", "EFFR"]:
        if col in big.columns:
            big[col] = big[col].ffill(limit=5)  # fill up to 1 week

    big = big.reset_index().rename(columns={"date": "date"})

    # compute engineered features on full history
    big = compute_features(big)

    # save parquet
    big.to_parquet(OUT_FILE_PARQUET)

    log(f"💾 Saved liquidity & funding → {OUT_FILE_PARQUET} ({len(big)} rows)")
    log("🔎 Last 10 rows:")
    log(big.tail(10).to_string(index=False))

    # also print to stdout (quick eyeball in terminal)
    print(big.tail(10).to_string(index=False))
    print("\nColumns:", list(big.columns))

if __name__ == "__main__":
    main()