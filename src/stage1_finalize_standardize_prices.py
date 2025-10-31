# src/stage1_finalize_standardize_prices.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

for file in DATA_DIR.glob("*_M5_clean.parquet"):
    df = pd.read_parquet(file)

    # --- Xóa cột close cũ nếu đã tồn tại ---
    if "close" in df.columns and "mid_c" in df.columns:
        df = df.drop(columns=["close"])

    # --- Chuẩn hóa cột ---
    rename_map = {
        "mid_o": "open",
        "mid_h": "high",
        "mid_l": "low",
        "mid_c": "close",
    }
    df = df.rename(columns=rename_map)

    # --- Giữ đúng 5 cột cần thiết ---
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]

    # --- Chuẩn hóa index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{file} chưa có DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # --- Lưu lại ---
    out = file.with_name(file.stem.replace("_clean", "_std") + ".parquet")
    df.to_parquet(out)
    print(f"✅ Saved standardized → {out} ({len(df):,} rows)")