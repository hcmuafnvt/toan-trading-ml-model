#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.6 — Feature Prune (Top-Alpha Selection, GBPUSD)

Purpose:
    - Load feature diagnostic report from Stage 4.5.
    - Select the most predictive tsfresh_core features (top N by AUC gain).
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
    if "group" not in df.columns:
        raise KeyError("File diagnostic missing 'group' column.")

    # 1️⃣ Filter only tsfresh_core group
    tsf = df[df["group"] == "tsfresh_core"].copy()
    if tsf.empty:
        raise RuntimeError("No tsfresh_core features found in diagnostic file.")

    # 2️⃣ Sort by AUC descending and keep top N
    tsf = tsf.sort_values("auc", ascending=False).head(TOP_N)

    # 3️⃣ Extract feature names only
    top_features = tsf["top_features"].iloc[0].split(",") if "top_features" in tsf.columns else tsf["feature"].tolist()
    top_features = [f.strip() for f in top_features if f.strip()]

    # 4️⃣ Save list
    Path("logs").mkdir(exist_ok=True)
    pd.Series(top_features, name="feature_name").to_csv(OUT_FILE, index=False)

    log(f"💾 Saved top {len(top_features)} features → {OUT_FILE}")
    log("✅ Stage 4.6 completed successfully")

if __name__ == "__main__":
    main()