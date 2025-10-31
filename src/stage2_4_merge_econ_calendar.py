import pandas as pd
from pathlib import Path
from datetime import datetime

TIMEGRID_FILE = "data/fx_timegrid.parquet"
CAL_FILE = "data/econ_calendar_features.parquet"
OUT_FILE = "data/stage2_calendar_merged.parquet"

def log(msg: str):
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] {msg}")

def main():
    log("🚀 Stage 2.4 — Merging economic calendar features onto M5 grid")

    # --- Load M5 time grid ---
    grid = pd.read_parquet(TIMEGRID_FILE).sort_index()
    log(f"📥 Loaded time grid: {len(grid):,} rows")

    # --- Load econ calendar (Stage 1 baseline) ---
    cal = pd.read_parquet(CAL_FILE)
    cal = cal.copy()
    # --- Detect time column automatically ---
    time_col = None
    for cand in ["time_utc", "datetime_utc", "timestamp", "time_local"]:
        if cand in cal.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"❌ No valid time column found in {CAL_FILE}")

    cal[time_col] = pd.to_datetime(cal[time_col], format="%H:%M", utc=True, errors="coerce")
    cal = cal.set_index(time_col).sort_index()
    log(f"🕒 Using time column: {time_col}")
    log(f"📊 Loaded econ_calendar_features: {len(cal):,} rows")

    # --- Remove duplicate timestamps before resampling ---
    cal = cal[~cal.index.duplicated(keep="last")]

    # --- Resample 5-min forward fill ---
    cal_resampled = cal.resample("5min").ffill()
    cal_resampled = cal_resampled.loc[
        (cal_resampled.index >= grid.index.min()) &
        (cal_resampled.index <= grid.index.max())
    ]

    # --- Align & merge ---
    merged = grid.join(cal_resampled, how="left").ffill()
    merged = merged.infer_objects(copy=False)

    # --- Save output ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)
    log(f"💾 Saved merged econ calendar → {OUT_FILE} ({len(merged):,} rows)")
    log(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    log(f"📈 Columns: {list(merged.columns)[:10]} ... total {len(merged.columns)} cols")

if __name__ == "__main__":
    main()