# src/stage1_8a_fetch_calendar_investing.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"

# Giai đoạn 1: 1/11/2023 → 30/10/2025
START = datetime(2023, 11, 1)
END = datetime(2025, 10, 30)

def fetch_investing_page(day):
    payload = {
        "country[]": [5, 6, 72, 17],  # US, JP, EU, UK
        "importance[]": [1, 2, 3],
        "dateFrom": day.strftime("%Y-%m-%d"),
        "dateTo": day.strftime("%Y-%m-%d"),
        "timeZone": 55,  # UTC
        "lang": "en"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.post(URL, data=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ {day.date()} HTTP {r.status_code}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("tr.js-event-item")
    data = []
    for row in rows:
        try:
            time_str = row.get("data-event-datetime")
            event_time = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
            event = row.get("data-event-title", "")
            country = row.get("data-country", "")
            currency = row.get("data-currency", "")
            impact = row.get("data-impact", "")
            actual = row.get("data-event-actual", "")
            forecast = row.get("data-event-forecast", "")
            previous = row.get("data-event-previous", "")
            data.append({
                "datetime": event_time,
                "country": country,
                "currency": currency,
                "event": event,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "impact": impact
            })
        except Exception:
            continue
    return data

all_data = []
day = START
while day <= END:
    daily = fetch_investing_page(day)
    if daily:
        all_data.extend(daily)
        print(f"✅ {day.date()} → {len(daily)} events")
    day += timedelta(days=1)
    time.sleep(1.5)  # tránh bị block

if all_data:
    df = pd.DataFrame(all_data)
    df.to_parquet("data/econ_calendar_investing.parquet")
    print(f"\n💾 Saved {len(df)} rows → data/econ_calendar_investing.parquet")
    print(df["country"].value_counts())
else:
    print("❌ No events fetched.")