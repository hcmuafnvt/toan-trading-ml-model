#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.8 — Investing.com Economic Calendar (FINAL, incremental-safe baseline)

Goal:
    - Crawl Investing.com economic calendar day by day
    - Extract: event text, numbers, impact (bull icons), timestamps
    - Save clean structured dataset for later stages (1.9+)
    - Auto-resume / incremental append if run multiple times

Output:
    - data/econ_calendar_investing_raw.parquet
    - logs/stage1_8_fetch_investing.log
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# =====================================================
# CONFIG
# =====================================================
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2025, 10, 30)

OUT_FILE_PARQUET = "data/econ_calendar_investing_raw.parquet"
LOG_FILE         = "logs/stage1_8_fetch_investing.log"

SITE_TZ_OFFSET_HOURS = -4   # assume Investing shows GMT-4
SITE_TZ = timezone(timedelta(hours=SITE_TZ_OFFSET_HOURS))
BASE_URL = "https://www.investing.com/economic-calendar/"
SLEEP_MIN, SLEEP_MAX = 0.5, 1.2

# =====================================================
# INIT ENV
# =====================================================
load_dotenv()
INV_COOKIE = os.getenv("INV_COOKIE")

if not INV_COOKIE:
    print("❌ Missing INV_COOKIE in .env")
    sys.exit(1)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/141.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Referer": "https://www.investing.com/economic-calendar/",
    "Cookie": INV_COOKIE,
}

# =====================================================
# UTILS
# =====================================================
def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def parse_floatish(x: str):
    if not x or not isinstance(x, str):
        return None
    x = x.strip().replace(",", "")
    if x in ("", "nan", "None"): 
        return None
    try:
        if x.endswith("%"):
            return float(x[:-1]) / 100.0
        if x.endswith("K"): return float(x[:-1]) * 1e3
        if x.endswith("M"): return float(x[:-1]) * 1e6
        if x.endswith("B"): return float(x[:-1]) * 1e9
        if x.endswith("T"): return float(x[:-1]) * 1e12
        return float(x)
    except:
        return None

def extract_impact(td):
    if not td:
        return None, None
    title = td.get("title")
    img_key = td.get("data-img_key")
    level = None
    if img_key and img_key.startswith("bull"):
        try:
            level = int(img_key.replace("bull", "").strip())
        except:
            pass
    return level, title

def to_utc(date_str, time_str):
    if not date_str: 
        return pd.NaT
    t = time_str or "00:00"
    parts = t.split(":")
    hh = int(parts[0]) if parts[0].isdigit() else 0
    mm = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    ss = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    try:
        dt_local = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=hh, minute=mm, second=ss, tzinfo=SITE_TZ
        )
        return pd.Timestamp(dt_local.astimezone(timezone.utc))
    except:
        return pd.NaT

# =====================================================
# CORE PARSER
# =====================================================
def fetch_day(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    url = f"{BASE_URL}?day={date_str}"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                break
            time.sleep(2)
        except Exception as e:
            log(f"⚠️ {date_str} attempt {attempt+1} failed: {e}")
            time.sleep(3)
    else:
        log(f"❌ {date_str} → all retries failed")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", id=lambda x: x and "economicCalendarData" in x)
    if not table:
        log(f"⚠️ {date_str} → No table found")
        return []

    rows_data = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        def td_text(i):
            return tds[i].get_text(strip=True) if i < len(tds) else None

        time_local = td_text(0)
        currency = td_text(1)
        impact_level, impact_text = extract_impact(tds[2] if len(tds) > 2 else None)
        event_td = tds[3] if len(tds) > 3 else None
        event_name = event_td.get_text(" ", strip=True) if event_td else None
        actual, forecast, previous = td_text(4), td_text(5), td_text(6)

        ts_utc = to_utc(date_str, time_local)

        rows_data.append({
            "date_local": date_str,
            "time_local": time_local,
            "timestamp_utc": ts_utc,
            "currency": currency,
            "event": event_name,
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
            "impact_level": impact_level,
            "impact_text": impact_text,
        })
    log(f"✅ {date_str} → {len(rows_data)} events")
    return rows_data

# =====================================================
# INCREMENTAL APPEND
# =====================================================
def load_existing():
    if not os.path.exists(OUT_FILE_PARQUET):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(OUT_FILE_PARQUET)
        log(f"📂 Loaded existing {len(df)} rows from {OUT_FILE_PARQUET}")
        return df
    except Exception as e:
        log(f"⚠️ Failed to read existing parquet: {e}")
        return pd.DataFrame()

# =====================================================
# MAIN
# =====================================================
def main():
    ensure_dirs()
    existing = load_existing()

    existing_dates = set(existing["date_local"].unique()) if not existing.empty else set()

    all_rows = []
    cur = START_DATE
    while cur <= END_DATE:
        if cur.strftime("%Y-%m-%d") in existing_dates:
            cur += timedelta(days=1)
            continue

        day_rows = fetch_day(cur)
        all_rows.extend(day_rows)

        # adaptive delay
        delay = random.uniform(SLEEP_MIN, SLEEP_MAX)
        if len(day_rows) < 50:
            delay += 1.0
        time.sleep(delay)

        cur += timedelta(days=1)

    if not all_rows:
        log("❌ No new data collected.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["actual_num"]   = new_df["actual"].apply(parse_floatish)
    new_df["forecast_num"] = new_df["forecast"].apply(parse_floatish)
    new_df["previous_num"] = new_df["previous"].apply(parse_floatish)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["timestamp_utc", "currency", "event"], inplace=True)
    combined.sort_values("timestamp_utc", inplace=True)

    combined.to_parquet(OUT_FILE_PARQUET)
    log(f"💾 Updated file saved: {OUT_FILE_PARQUET} (rows={len(combined)})")
    print(combined.tail(10).to_string(index=False))

# =====================================================
if __name__ == "__main__":
    main()