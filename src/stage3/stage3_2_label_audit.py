#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.2 — Label Audit & Integrity Check
-----------------------------------------
Kiểm tra chất lượng nhãn Stage 3.1:
    • Phân phối class (balance)
    • Autocorrelation → xem có lặp pattern / bias không
    • Phân bố theo giờ phiên (Asia / London / NY)
    • Tỷ lệ choppy (class 1) trong các session

Input:
    data/stage3_labels.parquet
    data/stage2_features_combined.parquet (để lấy timestamp cho session mapping)

Output:
    logs/stage3_label_audit.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

LABEL_FILE = "data/stage3_labels.parquet"
FEATURE_FILE = "data/stage2_features_combined.parquet"
OUT_LOG = "logs/stage3_label_audit.txt"

def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")

def session_from_hour(h):
    """Trả về session tên dựa vào UTC hour."""
    if 0 <= h < 7:
        return "Asia"
    elif 7 <= h < 12:
        return "London"
    elif 12 <= h < 20:
        return "NY"
    else:
        return "AfterHours"

def main():
    Path("logs").mkdir(exist_ok=True)
    f = open(OUT_LOG, "w")
    log(f"[{datetime.utcnow()}] 🚀 Stage 3.2 — Label QA started", f)

    lbl = pd.read_parquet(LABEL_FILE)
    feat = pd.read_parquet(FEATURE_FILE)
    lbl = lbl.reindex(feat.index)  # đảm bảo index khớp
    log(f"📥 Loaded labels: {len(lbl):,} rows", f)

    # --- Distribution ---
    dist = lbl["lbl_mc_012"].value_counts(normalize=True).sort_index()
    log("\n📊 Class distribution:", f)
    for k, v in dist.items():
        log(f"   Class {int(k)} → {v*100:.2f}%", f)

    # --- Autocorrelation ---
    series = lbl["lbl_mc_012"].fillna(1).astype(float)
    autocorr1 = series.autocorr(lag=1)
    autocorr5 = series.autocorr(lag=5)
    autocorr20 = series.autocorr(lag=20)
    log("\n🔁 Autocorrelation (to detect label stickiness):", f)
    log(f"   lag=1  → {autocorr1:.4f}", f)
    log(f"   lag=5  → {autocorr5:.4f}", f)
    log(f"   lag=20 → {autocorr20:.4f}", f)

    # --- Session mapping ---
    hours = lbl.index.hour
    lbl["session"] = [session_from_hour(h) for h in hours]
    pivot = lbl.groupby("session")["lbl_mc_012"].value_counts(normalize=True).unstack(fill_value=0)
    log("\n🕒 Distribution by session (%):", f)
    log(str((pivot * 100).round(2)), f)

    # --- Summary of choppy periods ---
    choppy_ratio = (lbl["lbl_mc_012"] == 1).mean()
    log(f"\n🌊 Overall choppy ratio: {choppy_ratio*100:.2f}%", f)
    choppy_by_session = (lbl[lbl["lbl_mc_012"] == 1].groupby("session").size() / lbl.groupby("session").size()).round(3)
    log("🌍 Choppy ratio by session:", f)
    for s, v in choppy_by_session.items():
        log(f"   {s}: {v*100:.1f}%", f)

    # --- Save summary ---
    f.close()
    print(f"💾 Saved audit log → {OUT_LOG}")

if __name__ == "__main__":
    main()