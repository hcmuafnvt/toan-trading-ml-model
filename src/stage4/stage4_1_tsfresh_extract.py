#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.1 — Tsfresh Feature Extraction (GBPUSD, windowed)

Mục tiêu:
- Trích xuất feature thống kê/phức tạp từ chuỗi giá GBPUSD ở khung M5.
- Nhưng KHÔNG làm 1 feature cho cả lịch sử (1 row duy nhất) nữa.
- Thay vào đó, chia chuỗi thành nhiều cửa sổ (window) chồng lấp theo thời gian.
  Mỗi window ~500 cây M5 (~42h), step 250 (~21h overlap 50%).
- Mỗi window => 1 id => tsfresh trả ra 1 dòng feature.
- Kết quả: dataframe có hàng trăm dòng, mỗi dòng đại diện 1 trạng thái thị trường.
- Output: logs/stage4_tsfresh_features_gbpusd.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from datetime import datetime, timezone

INPUT_STACK = "data/stage3_feature_stack.parquet"
OUT_FEATURE_CSV = "logs/stage4_tsfresh_features_gbpusd.csv"

WINDOW = 500   # số nến trong một window
STEP   = 250   # bước trượt giữa các window (overlap 50%)

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}", flush=True)

def build_windowed_series(px: pd.Series, window: int, step: int) -> pd.DataFrame:
    """
    Tạo long-form dataframe cho tsfresh:
    - cột 'id'  : window_id (0,1,2,...)
    - cột 'time': timestamp trong window (giữ thứ tự nội bộ)
    - cột 'value': giá đóng cửa gbpusd_close tương ứng

    px: Series index=timestamp_utc (DatetimeIndex UTC), values=float close
    """
    values = px.values
    times = px.index

    rows = []
    wid = 0
    start = 0
    n = len(px)

    while start + window <= n:
        end = start + window
        # đoạn [start:end)
        seg_vals = values[start:end]
        seg_times = times[start:end]

        # gán cùng id cho cả segment
        seg_df = pd.DataFrame({
            "id": wid,
            "time": seg_times,      # giữ timestamp thật để tsfresh có trật tự
            "value": seg_vals
        })
        rows.append(seg_df)

        wid += 1
        start += step

    if not rows:
        raise ValueError("Không tạo được bất kỳ window nào (chuỗi quá ngắn?).")

    out = pd.concat(rows, ignore_index=True)
    return out


def main():
    log("🚀 Stage 4.1 — Tsfresh Feature Extraction (GBPUSD, windowed)")

    # 1. load feature stack
    log(f"📥 Loading {INPUT_STACK} ...")
    df = pd.read_parquet(INPUT_STACK)

    # 2. lấy chuỗi close của GBPUSD
    #    (px_gbpusd_close đã chuẩn hóa từ Stage 2)
    if "px_gbpusd_close" not in df.columns:
        raise KeyError("px_gbpusd_close không tồn tại trong stage3_feature_stack.parquet")

    # đảm bảo index là DatetimeIndex UTC (Stage 2 chuẩn hoá rồi, nhưng ta check lại)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index của stage3_feature_stack.parquet không phải DatetimeIndex")

    px = df["px_gbpusd_close"].astype(float).copy()

    log(f"📏 Total candles: {len(px)}")
    log(f"🪟 Window={WINDOW}, Step={STEP}")

    # 3. tạo long-form windowed series cho tsfresh
    log("🧱 Building windowed series for tsfresh ...")
    series_long = build_windowed_series(px, WINDOW, STEP)
    # series_long columns: id, time, value

    log(f"📊 Windowed series shape: {series_long.shape}")
    log(f"📊 Unique windows: {series_long['id'].nunique()}")

    # 4. chạy tsfresh
    log("⚙️ Extracting features via tsfresh (EfficientFCParameters) ...")
    extracted = extract_features(
        series_long,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=28,
        disable_progressbar=False,
    )
    # extracted index = window id
    # columns = tsfresh features

    log(f"📈 Extracted feature matrix shape: {extracted.shape}")

    # 5. cleanup columns theo rule Stage 2 (persisted in memory):
    #    sanitize colnames for downstream LightGBM
    extracted.columns = (
        extracted.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )

    # 6. save ra CSV
    Path("logs").mkdir(parents=True, exist_ok=True)
    extracted.to_csv(OUT_FEATURE_CSV, index=True)  # keep window id as index

    log(f"💾 Saved → {OUT_FEATURE_CSV} ({extracted.shape[1]} features, {extracted.shape[0]} windows)")
    log("✅ Stage 4.1 completed successfully")

if __name__ == "__main__":
    main()