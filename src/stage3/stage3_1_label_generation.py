#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.1 — Label Definition & Generation
-----------------------------------------
Mục tiêu:
    • Tạo nhãn trading (labels) cho GBP/USD dựa trên hành vi giá tương lai.
    • Sử dụng logic TP/SL ảo trong horizon cố định.
    • Kết quả xuất thành bộ nhãn đa dạng (ternary, multiclass, regression...).

Input:
    data/stage2_features_combined.parquet
        - Index: timestamp_utc (UTC, M5)
        - Cột: px_gbpusd_close

Output:
    data/stage3_labels.parquet
        - Cột:
            lbl_raw_ternary     (-1, 0, +1)
            lbl_mc_012          (0, 1, 2)
            gbpusd_ret_fwd20
            gbpusd_ret_fwd20_pips
            future_max_move_pips
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==============================
# CONFIG
# ==============================
PAIR = "gbpusd"
CLOSE_COL = f"px_{PAIR}_close"
DATA_FILE = "data/stage2_features_combined.parquet"
OUT_FILE = "data/stage3_labels.parquet"

# Label params
TP_PIPS = 10
SL_PIPS = -10
HORIZON = 20  # nến M5 tương lai
PIP_SIZE = 0.0001  # GBP/USD

# ==============================
# FUNCTIONS
# ==============================

def log(msg):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def generate_labels(df):
    """Tạo nhãn dựa trên future high/low trong N nến tới."""
    close = df[CLOSE_COL].values
    n = len(close)

    lbl_raw = np.zeros(n)
    ret_fwd20 = np.full(n, np.nan)
    ret_pips = np.full(n, np.nan)
    future_max_move = np.full(n, np.nan)

    for i in range(n - HORIZON):
        window = close[i + 1 : i + HORIZON + 1]
        if len(window) < 2:
            continue

        future_high = np.max(window)
        future_low = np.min(window)

        up_pips = (future_high - close[i]) / PIP_SIZE
        dn_pips = (future_low - close[i]) / PIP_SIZE

        future_max_move[i] = max(abs(up_pips), abs(dn_pips))
        ret_fwd20[i] = (close[i + HORIZON] / close[i]) - 1
        ret_pips[i] = (close[i + HORIZON] - close[i]) / PIP_SIZE

        # Logic TP/SL (symmetrical)
        if up_pips >= TP_PIPS and dn_pips <= SL_PIPS:
            lbl_raw[i] = 0
        elif up_pips >= TP_PIPS:
            lbl_raw[i] = 1   # BUY wins
        elif dn_pips <= SL_PIPS:
            lbl_raw[i] = -1  # SELL wins
        else:
            lbl_raw[i] = 0

    df["lbl_raw_ternary"] = lbl_raw
    df["lbl_mc_012"] = df["lbl_raw_ternary"].map({-1:0, 0:1, 1:2})
    df["gbpusd_ret_fwd20"] = ret_fwd20
    df["gbpusd_ret_fwd20_pips"] = ret_pips
    df["future_max_move_pips"] = future_max_move

    return df


# ==============================
# MAIN
# ==============================
def main():
    log("🚀 Stage 3.1 — Label Generation started")
    df = pd.read_parquet(DATA_FILE)
    log(f"📥 Loaded {DATA_FILE} ({len(df):,} rows)")

    if CLOSE_COL not in df.columns:
        raise KeyError(f"❌ Missing {CLOSE_COL} in input file")

    df = generate_labels(df)

    # --- QA ---
    lbl_counts = df["lbl_mc_012"].value_counts(normalize=True).sort_index()
    log("🔍 Label distribution (ratio):")
    for k, v in lbl_counts.items():
        log(f"   Class {int(k)}: {v*100:.2f}%")

    if lbl_counts.min() < 0.05:
        log("⚠ Warning: one or more classes < 5% (potential imbalance).")

    # --- Save output ---
    cols = [
        "lbl_raw_ternary", "lbl_mc_012",
        "gbpusd_ret_fwd20", "gbpusd_ret_fwd20_pips", "future_max_move_pips"
    ]
    out_df = df[cols].copy()

    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_FILE)
    log(f"💾 Saved labels → {OUT_FILE} ({len(out_df):,} rows)")
    log(f"🕒 Range: {out_df.index.min()} → {out_df.index.max()}")
    log("✅ Stage 3.1 completed.")


if __name__ == "__main__":
    main()