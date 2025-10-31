#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.1 — Tsfresh Feature Extraction (GBPUSD, windowed)

Mục tiêu:
- Chia chuỗi giá GBPUSD M5 thành nhiều cửa sổ thời gian (window) chồng lấp.
  + WINDOW_SIZE ~ 500 nến (~42 giờ)
  + STEP ~ 250 nến (50% overlap)
- Mỗi window đại diện cho state thị trường tại thời điểm KẾT THÚC window.
- Với mỗi window, trích xuất các đặc trưng thống kê/phức tạp (tsfresh).
- Kết quả: 1 dòng / 1 window_end_time.

Output chính:
- logs/stage4_tsfresh_features_gbpusd.csv
  (index = window_end_time, các cột = feature)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from datetime import datetime, timezone

INPUT_STACK = "data/stage3_feature_stack.parquet"
OUT_FEATURE_CSV = "logs/stage4_tsfresh_features_gbpusd.csv"

WINDOW_SIZE = 500   # số nến M5 trong 1 window (~42h)
STEP        = 250   # bước trượt giữa các window (~21h), overlap 50%

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}", flush=True)

def build_windowed_series(px: pd.Series,
                          window_size: int,
                          step: int) -> pd.DataFrame:
    """
    Tạo long-form dataframe cho tsfresh.

    Quan trọng:
    - Mỗi window có 500 cây nến.
    - Thay vì đặt id = số thứ tự 0,1,2,...,
      ta đặt id = thời điểm KẾT THÚC window (window_end_time, dạng Timestamp UTC).
      => đây sẽ là "anchor time" để join nhãn (label) sau này.

    Trả về DataFrame với cột:
      id    : timestamp cuối window (UTC)
      time  : timestamp từng cây nến trong window (để tsfresh biết thứ tự nội bộ)
      value : giá đóng cửa px_gbpusd_close
    """

    values = px.values
    times = px.index
    n = len(px)

    rows = []
    window_end_times = []  # để lưu timestamp cuối mỗi window

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        seg_vals = values[start:end]
        seg_times = times[start:end]

        # timestamp cuối cùng của window
        window_end_time = seg_times[-1]
        window_end_times.append(window_end_time)

        seg_df = pd.DataFrame({
            "id":   window_end_time,   # sử dụng timestamp làm id
            "time": seg_times,         # thời gian từng điểm trong window
            "value": seg_vals.astype(float),
        })
        rows.append(seg_df)

    if not rows:
        raise ValueError("Không tạo được window nào (chuỗi quá ngắn so với WINDOW_SIZE).")

    out = pd.concat(rows, ignore_index=True)

    # đảm bảo time đúng UTC
    out["time"] = pd.to_datetime(out["time"], utc=True)

    # ghi attribute window_end_times để align label sau này
    out.attrs["window_end_times"] = pd.Series(pd.to_datetime(window_end_times, utc=True))

    return out

def main():
    log("🚀 Stage 4.1 — Tsfresh Feature Extraction (GBPUSD, windowed)")
    log(f"📥 Loading {INPUT_STACK} ...")

    # 1. Load stage3_feature_stack (đã bao gồm giá, macro, calendar, liquidity, labels…)
    df = pd.read_parquet(INPUT_STACK)

    # 2. Validate index & cột giá
    if "px_gbpusd_close" not in df.columns:
        raise KeyError("px_gbpusd_close không tồn tại trong stage3_feature_stack.parquet")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index của stage3_feature_stack.parquet không phải DatetimeIndex")

    # ép index UTC (should already be UTC, but safe)
    df.index = pd.to_datetime(df.index, utc=True)

    # chỉ lấy chuỗi close của GBPUSD
    px = df["px_gbpusd_close"].astype(float).copy()

    log(f"📏 Total candles: {len(px)}")
    log(f"🪟 Window_SIZE={WINDOW_SIZE}, STEP={STEP}")

    # 3. Xây chuỗi long-form cho tsfresh
    log("🧱 Building windowed series for tsfresh ...")
    series_long = build_windowed_series(px, WINDOW_SIZE, STEP)
    # columns: ['id', 'time', 'value']

    log(f"📊 Windowed series shape: {series_long.shape}")
    log(f"📊 Unique windows (id=window_end_time): {series_long['id'].nunique()}")

    # 4. Gọi tsfresh
    log("⚙️ Extracting features via tsfresh (EfficientFCParameters) ...")
    extracted = extract_features(
        series_long,
        column_id="id",        # id = window_end_time (UTC Timestamp)
        column_sort="time",    # sort theo thời gian trong window
        column_value="value",  # giá trị close
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=28,
        disable_progressbar=False,
    )
    # `extracted` index = unique 'id' (window_end_time)
    # columns = các feature tsfresh tạo ra

    log(f"📈 Extracted feature matrix shape: {extracted.shape}")
    
    # ✅ Gán real UTC timestamp (window_end_time) làm index cho mỗi window
    window_map = series_long.attrs.get("window_end_times", None)
    if window_map is not None:
        extracted.index = pd.to_datetime(window_map.values, utc=True)
        extracted.index.name = "window_end_time"
        log(f"🕒 Assigned window_end_time index ({len(window_map)} windows)")

    # 5. Làm sạch tên cột cho LightGBM (rule fund-grade đã thống nhất)
    extracted.columns = (
        extracted.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )

    # 6. Đảm bảo index của extracted là DatetimeIndex UTC để join với label sau này
    extracted.index = pd.to_datetime(extracted.index, utc=True)
    extracted.index.name = "window_end_time"

    # 7. Lưu CSV
    Path("logs").mkdir(parents=True, exist_ok=True)
    extracted.to_csv(OUT_FEATURE_CSV, index=True)

    log(f"💾 Saved → {OUT_FEATURE_CSV} ({extracted.shape[1]} features, {extracted.shape[0]} windows)")
    log("✅ Stage 4.1 completed successfully")


if __name__ == "__main__":
    main()