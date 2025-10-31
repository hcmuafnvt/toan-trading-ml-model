#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.3 — Label Refinement & Masking
Purpose: filter out choppy / low-vol / afterhours / extreme-vol zones.
Output:
  - data/stage3_label_mask.parquet
  - logs/stage3_label_mask.txt
"""
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from pathlib import Path
from datetime import datetime

LABEL_FILE = "data/stage3_labels.parquet"
PRICE_FILE = "data/stage2_prices_merged.parquet"
OUT_FILE = "data/stage3_label_mask.parquet"
LOG_FILE = "logs/stage3_label_mask.txt"

def log(msg):
    print(f"[{datetime.utcnow()}] {msg}")

def main():
    log("🚀 Stage 3.3 — Label Refinement & Masking started")
    labels = pd.read_parquet(LABEL_FILE)
    prices = pd.read_parquet(PRICE_FILE)[["gbpusd_high", "gbpusd_low", "gbpusd_close"]]

    # --- Compute ATR (volatility proxy) ---
    atr = AverageTrueRange(
        high=prices["gbpusd_high"],
        low=prices["gbpusd_low"],
        close=prices["gbpusd_close"],
        window=48
    ).average_true_range()
    atr.name = "atr"
    df = pd.concat([labels, atr], axis=1)

    # --- Define conditions ---
    is_afterhours = df.index.hour.isin([22, 23, 0])
    is_choppy = df["lbl_mc_012"] == 1
    atr_q20, atr_q99 = df["atr"].quantile([0.20, 0.99])
    is_low_vol = df["atr"] < atr_q20
    is_extreme_vol = df["atr"] > atr_q99

    # --- Mask: 1 = keep, 0 = remove ---
    df["mask_train"] = 1
    df.loc[is_afterhours | is_choppy | is_low_vol | is_extreme_vol, "mask_train"] = 0

    # --- Reason tracking (for QA) ---
    df["reason"] = "ok"
    df.loc[is_afterhours, "reason"] = "afterhours"
    df.loc[is_choppy, "reason"] = "choppy"
    df.loc[is_low_vol, "reason"] = "low_vol"
    df.loc[is_extreme_vol, "reason"] = "extreme_vol"

    # --- Save results ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df[["mask_train", "reason"]].to_parquet(OUT_FILE)
    kept_ratio = df["mask_train"].mean() * 100

    log(f"💾 Saved mask → {OUT_FILE} ({len(df):,} rows)")
    log(f"✅ Kept for training: {kept_ratio:.2f}%")
    log(f"⚠️ Removed samples by reason:")
    reason_stats = df["reason"].value_counts(normalize=True) * 100
    for k, v in reason_stats.items():
        log(f"   {k:<12}: {v:.2f}%")

    # --- Write log file ---
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write(f"[Stage 3.3 — Mask Summary]\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Kept ratio: {kept_ratio:.2f}%\n\n")
        f.write(reason_stats.to_string())

    log("✅ Stage 3.3 completed successfully")

if __name__ == "__main__":
    main()