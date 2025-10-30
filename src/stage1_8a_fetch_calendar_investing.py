# src/stage1_8a_fetch_calendar_investing_v2.py
"""
Use your browser cookie (INV_COOKIE) + cloudscraper to POST to
/getCalendarFilteredData and save results to data/econ_calendar_investing.parquet
"""
import os
import time
import random
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper
from dotenv import load_dotenv
load_dotenv()

# CONFIG (test short range first)
START = datetime(2024, 1, 1)
END   = datetime(2024, 1, 5)
OUT_FILE = "data/econ_calendar_investing.parquet"

# read cookie from env
INV_COOKIE = os.getenv("INV_COOKIE")
if not INV_COOKIE:
    raise SystemExit("❌ INV_COOKIE not found in env. Put full cookie string into .env as INV_COOKIE='...'")


# create cloudscraper session
scraper = cloudscraper.create_scraper()

# inject cookie string into session.cookies
def cookie_str_to_dict(s: str):
    d = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

scraper.cookies.update(cookie_str_to_dict(INV_COOKIE))

# headers copied/normalized from your cURL (keep UA and Referer)
BASE_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.investing.com",
    "pragma": "no-cache",
    "referer": "https://www.investing.com/economic-calendar/",
    "sec-ch-ua": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
}

POST_URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
PAGE_URL = "https://www.investing.com/economic-calendar/"

# payload template based on your cURL -- this includes many country[] entries and timeZone=8
BASE_PAYLOAD = {
    # if you want smaller subset, modify country[] list here
    "country[]": ["25","32","6","37","72","22","17","39","14","10","35","43","56","36","110","11","26","12","4","5"],
    "timeZone": "8",
    "timeFilter": "timeRemain",
    "currentTab": "tomorrow",
    "limit_from": "0"
}

def fetch_day(day_dt):
    day = day_dt.strftime("%Y-%m-%d")
    # ensure we have fresh cookies: do a GET first to refresh session headers/cookies
    try:
        rget = scraper.get(PAGE_URL, headers={"User-Agent": BASE_HEADERS["user-agent"], "Referer": BASE_HEADERS["referer"]}, timeout=20)
    except Exception as e:
        return False, f"GET page exception: {e}"
    if rget.status_code != 200:
        return False, f"GET page HTTP {rget.status_code}"

    payload = {
        **BASE_PAYLOAD,
        "dateFrom": day,
        "dateTo": day,
    }

    # cloudscraper will send cookies in session
    attempts = 0
    last_err = None
    while attempts < 4:
        attempts += 1
        try:
            resp = scraper.post(POST_URL, data=payload, headers=BASE_HEADERS, timeout=30)
        except Exception as e:
            last_err = f"POST exception: {e}"
            time.sleep(1.2*attempts + random.random())
            continue

        if resp.status_code == 200 and resp.text:
            # try parse HTML rows
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                rows = soup.select("tr.js-event-item")
                if not rows:
                    # try JSON with 'data' key
                    try:
                        j = resp.json()
                        html = j.get("data") or j.get("html") or ""
                        if html:
                            soup2 = BeautifulSoup(html, "html.parser")
                            rows = soup2.select("tr.js-event-item")
                    except Exception:
                        pass

                if not rows:
                    # no events that day (valid)
                    return True, []
                out = []
                for r in rows:
                    try:
                        t = r.get("data-event-datetime")
                        if not t:
                            continue
                        dt = datetime.strptime(t, "%Y/%m/%d %H:%M:%S")
                        out.append({
                            "datetime": pd.Timestamp(dt).tz_localize("UTC"),
                            "country": r.get("data-country"),
                            "currency": r.get("data-currency"),
                            "event": r.get("data-event-title"),
                            "impact": r.get("data-impact"),
                            "actual": r.get("data-event-actual"),
                            "forecast": r.get("data-event-forecast"),
                            "previous": r.get("data-event-previous"),
                        })
                    except Exception:
                        continue
                return True, out
            except Exception as e:
                last_err = f"parse error: {e}"
                time.sleep(0.8 + random.random())
                continue

        elif resp.status_code in (403, 429):
            last_err = f"POST HTTP {resp.status_code}"
            time.sleep(1.5 * attempts + random.random())
            continue
        else:
            last_err = f"POST HTTP {resp.status_code} len={len(resp.text)}"
            time.sleep(1.2 * attempts + random.random())
            continue

    return False, last_err

def run_range(start, end):
    cur = start
    rows = []
    while cur <= end:
        ok, res = fetch_day(cur)
        if ok:
            if res:
                print(f"✅ {cur.date()} → {len(res)} events")
                rows.extend(res)
            else:
                print(f"⚠️ {cur.date()} → no data")
        else:
            print(f"❌ {cur.date()} → FAILED ({res})")
        cur += timedelta(days=1)
        time.sleep(1.0 + random.random()*1.2)
    return rows

if __name__ == "__main__":
    rows = run_range(START, END)
    if rows:
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["datetime", "country", "event"])
        df = df.sort_values("datetime")
        os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
        df.to_parquet(OUT_FILE)
        print(f"\n💾 Saved {len(df):,} rows → {OUT_FILE}")
    else:
        print("\n❌ No events fetched for the test window.")