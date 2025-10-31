# =====================================================
# Stage 2.6 — Unified Feature Fusion (Master Merge)
# =====================================================
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# --- Input sources ---
PRICE_FILE = "data/stage2_prices_merged.parquet"
MACRO_FILE = "data/stage2_macro_merged.parquet"
CAL_FILE   = "data/stage2_calendar_merged.parquet"
LIQ_FILE   = "data/stage2_liquidity_merged.parquet"
OUT_FILE   = "data/stage2_features_combined.parquet"

def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def prefix_columns(df, prefix):
    """Add prefix_ to all columns (except DatetimeIndex)."""
    df = df.copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def main():
    log("🚀 Stage 2.6 — Unified Feature Fusion started")

    # --- Load datasets ---
    price = pd.read_parquet(PRICE_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    cal   = pd.read_parquet(CAL_FILE)
    liq   = pd.read_parquet(LIQ_FILE)

    # Ensure all have UTC datetime index
    for name, df in {"price": price, "macro": macro, "cal": cal, "liq": liq}.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be DatetimeIndex")
        df.index = pd.to_datetime(df.index, utc=True)

    # --- Prefix columns for clarity ---
    price = prefix_columns(price, "px")
    macro = prefix_columns(macro, "macro")
    cal   = prefix_columns(cal, "cal")
    liq   = prefix_columns(liq, "liq")

    # --- Join all sequentially on index ---
    merged = price.join(macro, how="left")
    merged = merged.join(cal, how="left")
    merged = merged.join(liq, how="left")

    # --- Forward fill for any slow-moving data ---
    merged = merged.ffill().infer_objects(copy=False)

    # --- Save output ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)

    log(f"💾 Saved unified features → {OUT_FILE} ({len(merged):,} rows, {len(merged.columns)} cols)")
    log(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    log(f"📊 Sample columns: {list(merged.columns)[:10]} ...")

if __name__ == "__main__":
    main()