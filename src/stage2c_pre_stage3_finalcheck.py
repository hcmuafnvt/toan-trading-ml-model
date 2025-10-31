#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2C — Pre-Stage3 Final QA
Kiểm tra 3 điều kiện cuối cùng trước khi freeze Stage 2 và bắt đầu Stage 3:
1️⃣  Xác định timestamp nào toàn bộ các cặp FX đều NaN (market holiday)
2️⃣  Đảm bảo forward-fill đúng hướng (không leak backward)
3️⃣  Kiểm tra tính đơn điệu và hợp lý của lịch kinh tế (impact_level)
"""

import pandas as pd
from pathlib import Path

# --- FILE PATHS ---
PRICE_FILE = "data/stage2_prices_merged.parquet"
CAL_FILE = "data/stage2_calendar_merged.parquet"
LIQ_FILE = "data/stage2_liquidity_merged.parquet"
MACRO_FILE = "data/stage2_macro_merged.parquet"

def log(msg):
    print(f"[QA2C] {msg}")

def main():
    log("🚀 Stage 2C — Pre-Stage3 Final QA started")

    # 1️⃣ Check market-holiday (toàn bộ price NaN)
    dfp = pd.read_parquet(PRICE_FILE)
    price_cols = [c for c in dfp.columns if "_close" in c]
    nanmask = dfp[price_cols].isna().all(axis=1)
    dead_rows = nanmask.sum()
    pct = dead_rows / len(dfp) * 100
    log(f"📉 Market-holiday rows (all FX NaN): {dead_rows:,} ({pct:.4f}%)")
    if dead_rows > 0:
        log("→ ⚠ Recommend drop những rows này ở Stage 3 loader.")
        log("   Example timestamps:")
        log(dfp[nanmask].head(5).index)

    # 2️⃣ Check forward-fill direction (no backward leak)
    dfl = pd.read_parquet(LIQ_FILE)
    for col in ["Fed_BalanceSheet", "RRP_Usage", "SOFR"]:
        first_valid = dfl[col].first_valid_index()
        nan_before = dfl.loc[:first_valid, col].isna().sum()
        if nan_before > 0:
            log(f"⚠ {col}: {nan_before} NaN trước first_valid → leak backward")
        else:
            log(f"✅ {col}: no backward leak")

    # 3️⃣ Calendar monotonicity & sanity
    dfc = pd.read_parquet(CAL_FILE)
    if "impact_level" in dfc.columns:
        non_nan = dfc["impact_level"].dropna()
        changes = (non_nan != non_nan.shift()).sum()
        log(f"🕒 Calendar impact_level changes: {changes:,}")
        freq = non_nan.value_counts(dropna=False)
        log(f"   Top levels:\n{freq}")
        if changes == 0:
            log("⚠ Calendar có thể bị flat (không thay đổi giá trị).")
        else:
            log("✅ Calendar biến động bình thường.")
    else:
        log("⚠ Missing impact_level column!")

    # 4️⃣ Quick cross-check daily vs M5 alignment (macro/liquidity)
    dfm = pd.read_parquet(MACRO_FILE)
    if dfm.index.equals(dfp.index):
        log("✅ Macro index align perfectly with price grid")
    else:
        log("⚠ Macro misalignment detected (should re-ffill at Stage 3 loader)")

    log("✅ Stage 2C QA completed — ready for Stage 3 setup")

if __name__ == "__main__":
    main()