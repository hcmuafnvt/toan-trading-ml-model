#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.6 — Feature Prune (Top-Alpha Selection, GBPUSD)

Purpose:
    - Load feature diagnostic report from Stage 4.5.
    - Select the most predictive tsfresh_core features (top N by AUC).
    - Save pruned feature list for Stage 4.7 model tuning.

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
TOP_N     = 100  # keep top N alpha features

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)

def main():
    log("🚀 Stage 4.6 — Feature Prune (Top-Alpha Selection GBPUSD)")

    df = pd.read_csv(DIAG_FILE)
    log(f"📥 Loaded diagnostic file: {df.shape}")

    # 1️⃣ Filter only tsfresh_core group
    if "group" not in df.columns:
        raise KeyError("Diagnostic file missing 'group' column.")
    tsf = df[df["group"] == "tsfresh_core"].copy()
    if tsf.empty:
        raise RuntimeError("No tsfresh_core features found in diagnostic file.")

    # 2️⃣ Sort by AUC descending
    if "auc" not in tsf.columns:
        raise KeyError("Diagnostic file missing 'auc' column.")
    tsf = tsf.sort_values("auc", ascending=False).head(TOP_N)

    # 3️⃣ Extract feature names
    # Most Stage 4.5 files contain `top_features` column with comma-separated names
    feature_list = []
    if "top_features" in tsf.columns:
        for row in tsf["top_features"]:
            if isinstance(row, str):
                feature_list.extend([f.strip() for f in row.split(",") if f.strip()])
    elif "feature" in tsf.columns:
        feature_list = tsf["feature"].astype(str).tolist()
    else:
        raise KeyError("Neither 'top_features' nor 'feature' column found in diagnostic file.")

    # Remove duplicates and keep top N
    feature_list = list(dict.fromkeys(feature_list))[:TOP_N]

    # 4️⃣ Save list
    Path("logs").mkdir(exist_ok=True)
    pd.Series(feature_list, name="feature_name").to_csv(OUT_FILE, index=False)

    log(f"💾 Saved top {len(feature_list)} features → {OUT_FILE}")
    log("✅ Stage 4.6 completed successfully")

if __name__ == "__main__":
    main()