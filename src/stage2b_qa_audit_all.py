#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2B — QA Audit (All Baseline Datasets)
Purpose: sanity-check all Stage 2 datasets before Stage 3
"""

import pandas as pd
from pathlib import Path

FILES = {
    "timegrid": "data/fx_timegrid.parquet",
    "prices": "data/stage2_prices_merged.parquet",
    "macro": "data/stage2_macro_merged.parquet",
    "calendar": "data/stage2_calendar_merged.parquet",
    "calendar_dense": "data/stage2_calendar_dense.parquet",
    "liquidity": "data/stage2_liquidity_merged.parquet",
    "features_combined": "data/stage2_features_combined.parquet",
}

def log(msg):
    print(f"[QA] {msg}")

def audit_file(name, path, ref_index=None):
    df = pd.read_parquet(path)
    log(f"📂 {name}: {len(df):,} rows | {len(df.columns)} cols | {path}")
    log(f"   → Range: {df.index.min()} → {df.index.max()}")
    if not isinstance(df.index, pd.DatetimeIndex):
        log("   ⚠ Index not DatetimeIndex!")
    else:
        if not df.index.tz:
            log("   ⚠ Missing timezone info!")
        elif str(df.index.tz) != "UTC":
            log(f"   ⚠ Timezone ≠ UTC ({df.index.tz})")
    miss = df.isna().mean().sort_values(ascending=False)
    topmiss = miss[miss > 0].head(5)
    if len(topmiss):
        log(f"   ⚠ Missing values (top 5):\n{topmiss.round(4)}")
    else:
        log("   ✅ No missing values")
    if ref_index is not None:
        if not df.index.equals(ref_index):
            log("   ⚠ Index misalignment vs timegrid!")
        else:
            log("   ✅ Time alignment OK")
    if df.columns.duplicated().any():
        log("   ⚠ Duplicate columns detected!")
    
    # quick prefix check
    if len(df.columns) > 0:
        prefixes = df.columns.astype(str).str.split("_").str[0].value_counts().head(5)
        log(f"   🔤 Top column prefixes: {dict(prefixes)}")
    else:
        log("   ℹ No columns (empty structure, likely timegrid).")
    print("-" * 90)
    return df

def main():
    log("🚀 Stage 2B QA Audit started")
    Path("data").mkdir(exist_ok=True)
    ref = pd.read_parquet(FILES["timegrid"])
    ref_index = ref.index
    results = {}
    for name, path in FILES.items():
        if Path(path).exists():
            results[name] = audit_file(name, path, ref_index)
        else:
            log(f"❌ Missing file: {path}")
            
    # --- Extra QA: Calendar Dense Features ---
    try:
        dense = pd.read_parquet(FILES["calendar_dense"])
        if len(dense) == len(ref_index):
            print("[QA2C] ✅ calendar_dense aligned perfectly")
        else:
            print(f"[QA2C] ⚠ calendar_dense length mismatch: {len(dense)} vs {len(ref_index)}")
        if dense.isna().any().any():
            print("[QA2C] ⚠ calendar_dense contains NaN values")
        else:
            print("[QA2C] ✅ calendar_dense has no NaN values")
    except Exception as e:
        print(f"[QA2C] ⚠ calendar_dense check failed: {e}")

    log("✅ QA Audit completed")
    log("→ Verify that all datasets share same index & UTC timezone before Stage 3")

if __name__ == "__main__":
    main()