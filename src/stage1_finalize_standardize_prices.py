# src/stage1_finalize_standardize_prices.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

for file in DATA_DIR.glob("*_M5_clean.parquet"):
    df = pd.read_parquet(file)

    # --- Chuẩn hóa cột ---
    if all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"]):
        df = df.rename(columns={
            "mid_o": "open",
            "mid_h": "high",
            "mid_l": "low",
            "mid_c": "close"
        })
    df = df[["open", "high", "low", "close", "volume"]]

    # --- Chuẩn hóa index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{file} chưa có DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # --- Lưu lại ---
    out = file.with_name(file.stem.replace("_clean", "_std") + ".parquet")
    df.to_parquet(out)
    print(f"✅ Saved standardized → {out} ({len(df):,} rows)")