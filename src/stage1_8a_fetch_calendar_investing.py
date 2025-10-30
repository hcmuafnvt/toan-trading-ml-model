# src/stage1_8a_fetch_calendar_investing.py
# Robust Investing.com calendar crawler (test small window)
import time
import random
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper

# -------- CONFIG (test small range) --------
START = datetime(2024, 1, 1)
END = datetime(2024, 1, 5)
OUT_FILE = "data/econ_calendar_investing.parquet"

# Investing endpoint (used by their web UI)
POST_URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
PAGE_URL = "https://www.investing.com/economic-calendar/"

# Which countries to request (IDs used by Investing)
# 5=United States, 6=Japan, 72=Euro Area, 17=United Kingdom
COUNTRY_IDS = [5, 6, 72, 17]

# Create a cloudscraper session (handles Cloudflare)
scraper = cloudscraper.create_scraper()

# Common headers that look like a real browser
BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": PAGE_URL,
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_one_day(day_dt):
    day = day_dt.strftime("%Y-%m-%d")
    # step 1: GET main page to obtain cookies/session tokens
    try:
        r = scraper.get(PAGE_URL, headers=BASE_HEADERS, timeout=30)
    except Exception as e:
        return False, f"GET page failed: {e}"

    if r.status_code != 200:
        return False, f"GET page HTTP {r.status_code}"

    # it's helpful to have the response text (some sites embed tokens)
    # but mostly cookies (PHPSESSID) suffice
    # step 2: POST the calendar filtered request using same session
    payload = {
        "country[]": COUNTRY_IDS,
        "importance[]": [1, 2, 3],
        "dateFrom": day,
        "dateTo": day,
        "timeZone": 55,  # UTC
        "lang": "en"
    }

    # Build headers for POST — include Referer + same UA
    post_headers = BASE_HEADERS.copy()
    post_headers.update({
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    })

    # retry loop with jitter/backoff
    attempts = 0
    while attempts < 4:
        attempts += 1
        try:
            resp = scraper.post(POST_URL, data=payload, headers=post_headers, timeout=30)
        except Exception as e:
            err = f"POST exception: {e}"
            time.sleep(1.2 * attempts + random.random())
            continue

        if resp.status_code == 200 and resp.text.strip():
            # parse response
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("tr.js-event-item")
            if not rows:
                # sometimes investing returns JSON with 'data' key; try parse as json then fallback to HTML
                try:
                    js = resp.json()
                    # if it's JSON with 'data' pieces
                    if isinstance(js, dict) and js.get("data"):
                        # data is HTML string
                        soup2 = BeautifulSoup(js["data"], "html.parser")
                        rows = soup2.select("tr.js-event-item")
                except Exception:
                    pass

            if not rows:
                # empty (no events) is valid — return success with empty list
                return True, []

            parsed = []
            for row in rows:
                try:
                    t = row.get("data-event-datetime")  # format: YYYY/MM/DD HH:MM:SS
                    if not t:
                        continue
                    event_time = datetime.strptime(t, "%Y/%m/%d %H:%M:%S")
                    parsed.append({
                        "datetime": pd.Timestamp(event_time).tz_localize("UTC"),
                        "country": row.get("data-country"),
                        "currency": row.get("data-currency"),
                        "event": row.get("data-event-title"),
                        "impact": row.get("data-impact"),
                        "actual": row.get("data-event-actual"),
                        "forecast": row.get("data-event-forecast"),
                        "previous": row.get("data-event-previous")
                    })
                except Exception:
                    continue
            return True, parsed

        elif resp.status_code in (403, 429):
            # blocked or rate limited → backoff and retry
            err = f"POST HTTP {resp.status_code}"
            time.sleep(1.5 * attempts + random.random())
            continue
        else:
            err = f"POST HTTP {resp.status_code} resp_len={len(resp.text)}"
            time.sleep(1.2 * attempts + random.random())
            continue

    return False, err

def run_range(start_dt, end_dt):
    cur = start_dt
    all_rows = []
    day_count = 0
    while cur <= end_dt:
        day_count += 1
        ok, payload = fetch_one_day(cur)
        if ok:
            if payload:
                print(f"✅ {cur.date()} → {len(payload)} events")
                all_rows.extend(payload)
            else:
                print(f"⚠️ {cur.date()} → no data")
        else:
            print(f"❌ {cur.date()} → FAILED ({payload})")
        cur += timedelta(days=1)
        # small jitter to avoid automated detection
        time.sleep(1.0 + random.random()*1.5)
    return all_rows

if __name__ == "__main__":
    rows = run_range(START, END)
    if rows:
        df = pd.DataFrame(rows)
        # normalize columns and types
        df = df.drop_duplicates(subset=["datetime", "country", "event"])
        df = df.sort_values("datetime")
        df.to_parquet(OUT_FILE)
        print(f"\n💾 Saved {len(df):,} rows → {OUT_FILE}")
    else:
        print("\n❌ No events fetched for the test window.")