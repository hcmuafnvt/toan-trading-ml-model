import pandas as pd
from pathlib import Path
from datetime import datetime

TIMEGRID_FILE = "data/fx_timegrid.parquet"
MACRO_FILE = "data/macro_fxdrivers.parquet"
OUT_FILE = "data/stage2_macro_merged.parquet"


def log(msg: str):
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] {msg}")


def main():
    log("🚀 Stage 2.3 — Merging macro drivers onto M5 grid")

    # --- Load base time grid ---
    grid = pd.read_parquet(TIMEGRID_FILE)
    grid = grid.sort_index()
    log(f"📥 Loaded time grid: {len(grid):,} rows")

    # --- Load macro dataset ---
    macro = pd.read_parquet(MACRO_FILE)
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=True)
    macro = macro.set_index("date").sort_index()
    log(f"📊 Loaded macro_fxdrivers: {len(macro):,} rows")

    # --- Resample daily → M5 forward-fill ---
    macro_resampled = macro.resample("5min").ffill()
    macro_resampled = macro_resampled.loc[
        (macro_resampled.index >= grid.index.min()) &
        (macro_resampled.index <= grid.index.max())
    ]

    # --- Align to grid (full outer join then ffill) ---
    merged = grid.join(macro_resampled, how="left").ffill()

    # --- Save ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)
    log(f"💾 Saved merged macro drivers → {OUT_FILE} ({len(merged):,} rows)")
    log(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    log(f"📈 Columns: {list(merged.columns)[:10]} ... total {len(merged.columns)} cols")


if __name__ == "__main__":
    main()