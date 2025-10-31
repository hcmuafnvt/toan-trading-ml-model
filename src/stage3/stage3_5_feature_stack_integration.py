#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.5 — Feature Stack Integration

Mục tiêu:
- Hợp nhất full feature set (Stage 2) với nhãn & mask đã qua lọc (Stage 3.4).
- Tạo dataset duy nhất cho Stage 4 (feature engineering nâng cao + train model).

Input:
    data/stage2_features_combined.parquet
        - frame 296,996 rows
        - columns prefix:
            px_*    (giá multi-pair)
            macro_* (rates, curve, risk sentiment)
            liq_*   (Fed balance sheet, RRP, SOFR, funding stress)
            cal_*   (calendar snapshot kiểu impact tại thời điểm đó)
    data/stage3_train_ready.parquet
        - columns:
            label_mc_012   (0=down,1=chop,2=up AFTER remap)
            is_trainable   (True/False)
            drop_reason    (diagnostic)

Output:
    data/stage3_feature_stack.parquet
        - index: timestamp_utc (UTC M5)
        - all px_/macro_/liq_/cal_* features
        - target_label = label_mc_012
        - is_trainable flag
        - NO leakage columns like future_return_* (nếu có)
          (chúng ta giữ nguyên hiện tại; nếu phát hiện leakage ở Stage 4 thì loại sau)

Notes:
- Đây sẽ là input chuẩn cho:
    Stage 4.1 tsfresh extraction (dựa trên px_* series)
    Stage 4.x LightGBM training
- Sau bước này, Stage 3 coi như đóng băng.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


FEATURE_FILE = "data/stage2_features_combined.parquet"
TRAINREADY_FILE = "data/stage3_train_ready.parquet"
OUT_FILE = "data/stage3_feature_stack.parquet"
LOG_FILE = "logs/stage3_feature_stack.txt"


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    log("🚀 Stage 3.5 — Feature Stack Integration start")

    # -------------------------------------------------
    # 1. Load data
    # -------------------------------------------------
    feat = pd.read_parquet(FEATURE_FILE)
    train_ready = pd.read_parquet(TRAINREADY_FILE)

    # safety: ensure index is datetime+UTC
    if not isinstance(feat.index, pd.DatetimeIndex):
        raise ValueError("❌ features_combined index is not DatetimeIndex")
    if not isinstance(train_ready.index, pd.DatetimeIndex):
        raise ValueError("❌ train_ready index is not DatetimeIndex")

    if feat.index.tz is None or str(feat.index.tz) != "UTC":
        raise ValueError("❌ features_combined index is not tz-aware UTC")
    if train_ready.index.tz is None or str(train_ready.index.tz) != "UTC":
        raise ValueError("❌ train_ready index is not tz-aware UTC")

    # -------------------------------------------------
    # 2. Join on timestamp_utc index
    # left = feat (to keep full timeline); right = labels
    # -------------------------------------------------
    merged = feat.join(
        train_ready[["lbl_mc_012", "mask_train", "reason"]],
        how="left"
    ).rename(columns={
        "lbl_mc_012": "target_label",
        "mask_train": "target_is_trainable",
        "reason": "target_drop_reason",
    })

    # -------------------------------------------------
    # 3. Rename label column to a stable training name
    # -------------------------------------------------
    merged = merged.rename(
        columns={
            "label_mc_012": "target_label",
            "is_trainable": "target_is_trainable",
            "drop_reason": "target_drop_reason",
        }
    )

    # -------------------------------------------------
    # 4. Basic sanity stats
    # -------------------------------------------------
    total_rows = len(merged)
    usable_rows = merged["target_is_trainable"].fillna(False).sum()

    # class distribution on usable rows
    dist = (
        merged.loc[merged["target_is_trainable"] == True, "target_label"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    log(f"📏 Rows total: {total_rows:,}")
    log(f"🎯 Trainable rows: {usable_rows:,} ({usable_rows/total_rows*100:.2f}%)")

    for cls_val, ratio in dist.items():
        log(f"   Class {cls_val}: {ratio*100:.2f}%")

    # -------------------------------------------------
    # 5. Save to parquet
    # -------------------------------------------------
    Path("data").mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)

    log(f"💾 Saved feature stack → {OUT_FILE} ({len(merged):,} rows, {merged.shape[1]} cols)")
    log(f"🕒 Range: {merged.index.min()} → {merged.index.max()}")
    log("✅ Stage 3.5 completed (feature stack is now canonical input for Stage 4)")
    print(merged.head(5).to_string())


if __name__ == "__main__":
    main()