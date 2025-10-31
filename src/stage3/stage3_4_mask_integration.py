#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.4 — Mask QA & Integration
Purpose:
  - verify label–mask alignment
  - integrate both into one unified dataset ready for ML training
Outputs:
  - data/stage3_train_ready.parquet
  - logs/stage3_mask_integration.txt
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

LABEL_FILE = "data/stage3_labels.parquet"
MASK_FILE = "data/stage3_label_mask.parquet"
OUT_FILE = "data/stage3_train_ready.parquet"
LOG_FILE = "logs/stage3_mask_integration.txt"

def log(msg):
    print(f"[{datetime.utcnow()}] {msg}")

def main():
    log("🚀 Stage 3.4 — Mask QA & Integration started")

    # --- Load data ---
    lbl = pd.read_parquet(LABEL_FILE)
    mask = pd.read_parquet(MASK_FILE)

    log(f"📥 Loaded labels: {lbl.shape}, mask: {mask.shape}")

    # --- Alignment check ---
    if not lbl.index.equals(mask.index):
        log("⚠️ Index misaligned — performing join by index")
        merged = lbl.join(mask, how="inner")
    else:
        merged = pd.concat([lbl, mask], axis=1)

    # --- Sanity checks ---
    if merged["mask_train"].isna().any():
        n = merged["mask_train"].isna().sum()
        log(f"⚠️ {n:,} NaN masks detected → filled with 0 (excluded).")
        merged["mask_train"] = merged["mask_train"].fillna(0).astype(int)

    kept = merged["mask_train"].sum()
    total = len(merged)
    keep_ratio = kept / total * 100

    log(f"✅ Alignment OK — merged {len(merged):,} rows")
    log(f"📊 Trainable samples: {kept:,} / {total:,} ({keep_ratio:.2f}%)")

    # --- Save output ---
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE)
    log(f"💾 Saved integrated dataset → {OUT_FILE}")

    # --- Save summary log ---
    reason_stats = merged["reason"].value_counts(normalize=True) * 100
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("[Stage 3.4 — Mask Integration Summary]\n")
        f.write(f"Total rows: {total:,}\n")
        f.write(f"Trainable: {kept:,} ({keep_ratio:.2f}%)\n\n")
        f.write("Breakdown by reason:\n")
        f.write(reason_stats.to_string())

    log("✅ Stage 3.4 completed successfully")

if __name__ == "__main__":
    main()