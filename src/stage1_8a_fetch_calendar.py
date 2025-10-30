# ================================================================
# Stage 1.8a — Economic Calendar Normalization
# Purpose: Fetch global macro events (US, JP, EU, UK)
# ================================================================

import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# -------- CONFIG --------
load_dotenv()
API_KEY = os.getenv("TE_API_KEY")  # e.g. c47b6f36da0348d:zq0wb4i0swy4fk6
OUT_FILE = "data/econ_calendar_master.parquet"

REGIONS = ["United States", "Japan", "Euro Area", "United Kingdom"]
START = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

BASE_URL = f"https://api.tradingeconomics.com/calendar?c={API_KEY}"

# -------- FETCH --------
print(f"🚀 Stage 1.8a — Fetching Economic Calendar ({START} → {END})")

frames = []
for region in REGIONS:
    url = f"{BASE_URL}&country={region}&start={START}&end={END}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"⚠️  Failed {region}: {r.status_code}")
        continue

    df = pd.DataFrame(r.json())
    if df.empty:
        print(f"⚠️  No data for {region}")
        continue

    df["country"] = region
    frames.append(df)

if not frames:
    raise SystemExit("❌ No calendar data fetched. Check API key or connection.")

raw = pd.concat(frames, ignore_index=True)

# -------- CLEANING --------
keep_cols = [
    "country", "event", "actual", "forecast", "previous",
    "importance", "date", "time", "unit"
]
df = raw[keep_cols].copy()

# Parse datetime
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", utc=True)
df = df.dropna(subset=["datetime"])
df = df.drop_duplicates(subset=["datetime", "event", "country"]).sort_values("datetime")

# Impact normalization
impact_map = {"low": 1, "medium": 2, "high": 3}
df["impact"] = df["importance"].str.lower().map(impact_map).fillna(1).astype(int)

# Surprise index
def compute_surprise(row):
    try:
        if row["forecast"] in (None, 0):
            return 0
        return (float(row["actual"]) - float(row["forecast"])) / abs(float(row["forecast"]))
    except Exception:
        return 0

df["surprise"] = df.apply(compute_surprise, axis=1)
df["surprise_sign"] = df["surprise"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Relevance weights
def region_weights(c):
    if "United States" in c: return (1, 0, 0, 0)
    if "Japan" in c: return (0, 1, 0, 0)
    if "Euro" in c: return (0, 0, 1, 0)
    if "United Kingdom" in c: return (0, 0, 0, 1)
    return (0, 0, 0, 0)

df[["usd_weight", "jpy_weight", "eur_weight", "gbp_weight"]] = df["country"].apply(
    lambda c: pd.Series(region_weights(c))
)

# Mark major events
MAJOR_KEYWORDS = ["CPI", "Inflation", "GDP", "Payroll", "Unemployment", "Rate", "PMI", "Retail"]
df["is_major"] = df["event"].apply(lambda e: any(k in str(e) for k in MAJOR_KEYWORDS))

# -------- SAVE --------
df = df[
    [
        "datetime", "country", "event", "actual", "forecast", "previous",
        "impact", "surprise", "surprise_sign",
        "usd_weight", "jpy_weight", "eur_weight", "gbp_weight", "is_major"
    ]
]

df.to_parquet(OUT_FILE)
print(f"✅ Saved {len(df):,} events → {OUT_FILE}")

# Basic stats
print("📊 Event distribution by country:")
print(df["country"].value_counts())
print("\n📊 Major events:", df["is_major"].sum())