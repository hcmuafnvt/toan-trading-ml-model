# src/stage1_8a_fetch_calendar_investing.py
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import cloudscraper

URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
scraper = cloudscraper.create_scraper()

START = datetime(2024, 1, 1)
END = datetime(2024, 1, 5)

def fetch_investing_page(day):
    payload = {
        "country[]": [5, 6, 72, 17],  # US, JP, EU, UK
        "importance[]": [1, 2, 3],
        "dateFrom": day.strftime("%Y-%m-%d"),
        "dateTo": day.strftime("%Y-%m-%d"),
        "timeZone": 55,
        "lang": "en"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": "https://www.investing.com/economic-calendar/"
    }
    r = scraper.post(URL, data=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ {day.date()} HTTP {r.status_code}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("tr.js-event-item")
    data = []
    for row in rows:
        try:
            t = row.get("data-event-datetime")
            if not t:
                continue
            event_time = datetime.strptime(t, "%Y/%m/%d %H:%M:%S")
            data.append({
                "datetime": event_time,
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
    return data

all_data = []
day = START
while day <= END:
    events = fetch_investing_page(day)
    if events:
        print(f"✅ {day.date()} → {len(events)} events")
        all_data.extend(events)
    else:
        print(f"⚠️ {day.date()} → no data")
    day += timedelta(days=1)
    time.sleep(1.5)

if all_data:
    df = pd.DataFrame(all_data)
    df.to_parquet("data/econ_calendar_investing.parquet")
    print(f"\n💾 Saved {len(df)} rows → data/econ_calendar_investing.parquet")
else:
    print("❌ No events fetched.")