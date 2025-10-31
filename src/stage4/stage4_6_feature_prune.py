#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.6 — Feature Prune (Top-Alpha Selection, GBPUSD)

Purpose:
    - Load feature diagnostic report from Stage 4.5.
    - Select the best-performing tsfresh_core features.
    - Extract their names from column `top5_features`.
    - Save as the alpha feature shortlist.

Input:
    logs/stage4_feature_diagnostics_gbpusd.csv
Output:
    logs/stage4_top_features_gbpusd.csv
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DIAG_FILE = "logs/stage4_feature_diagnostics_gbpusd.csv"
OUT_FILE  = "logs/stage4_top_features_gbpusd.csv"

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)

def main():
    log("🚀 Stage 4.6 — Feature Prune (Top-Alpha Selection GBPUSD)")

    df = pd.read_csv(DIAG_FILE)
    log(f"📥 Loaded diagnostic file: {df.shape}")

    # 1️⃣ Lấy nhóm tsfresh_core (nhóm có AUC cao nhất)
    tsf = df[df["group"] == "tsfresh_core"].copy()
    if tsf.empty:
        raise RuntimeError("❌ Không tìm thấy nhóm tsfresh_core trong diagnostic file.")

    log(f"🔎 Found tsfresh_core group (AUC={tsf['auc'].iloc[0]:.4f})")

    # 2️⃣ Parse danh sách top feature trong cột 'top5_features'
    feature_list = []
    for cell in tsf["top5_features"]:
        if isinstance(cell, str):
            feature_list.extend([f.strip() for f in cell.split(",") if f.strip()])

    feature_list = list(dict.fromkeys(feature_list))  # remove duplicates

    # 3️⃣ Save list
    Path("logs").mkdir(exist_ok=True)
    pd.Series(feature_list, name="feature_name").to_csv(OUT_FILE, index=False)

    log(f"💾 Saved {len(feature_list)} alpha features → {OUT_FILE}")
    log("✅ Stage 4.6 completed successfully")

if __name__ == "__main__":
    main()