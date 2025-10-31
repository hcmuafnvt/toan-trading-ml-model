# =============================================
# Stage 2.5 — Merge Liquidity & Funding Data
# =============================================
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

GRID_FILE = "data/fx_timegrid.parquet"
LIQ_FILE = "data/liquidity_funding.parquet"
OUT_FILE = "data/stage2_liquidity_merged.parquet"

def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def main():
    log("🚀 Stage 2.5 — Merging liquidity & funding data onto M5 grid")

    # --- Load master time grid ---
    grid = pd.read_parquet(GRID_FILE)
    grid.index = pd.to_datetime(grid.index, utc=True)
    log(f"📥 Loaded time grid: {len(grid):,} rows")

    # --- Load liquidity dataset ---
    liq = pd.read_parquet(LIQ_FILE)
    liq["date"] = pd.to_datetime(liq["date"], utc=True)
    liq = liq.set_index("date").sort_index()
    log(f"📊 Loaded liquidity dataset: {len(liq):,} rows, columns={list(liq.columns)}")

    # --- Resample to 5-min frequency with forward-fill ---
    liq_resampled = liq.resample("5min").ffill()
    liq_resampled = liq_resampled.loc[
        (liq_resampled.index >= grid.index.min()) &
        (liq_resampled.index <= grid.index.max())
    ]
    liq_resampled = liq_resampled.infer_objects(copy=False)

    # --- Align and merge with grid ---
    merged = grid.join(liq_resampled, how="left").ffill()
    merged = merged.infer_objects(copy=False)

    # --- Save output ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)

    log(f"💾 Saved merged liquidity dataset → {OUT_FILE} ({len(merged):,} rows)")
    log(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    log(f"📈 Columns: {list(merged.columns)[:10]} ... total {len(merged.columns)} cols")

if __name__ == "__main__":
    main()