#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.3.1 — Diagnostic Alignment Debug (GBPUSD)

Goal:
Check whether tsfresh window features align correctly with future labels.
We test if window END timestamp corresponds to label horizon (20 bars ahead).

Outputs:
- logs/debug_window_label_alignment.csv
- plots distribution of forward returns vs window end

"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FEATURE_FILE = "logs/stage4_tsfresh_features_gbpusd.csv"
TRAIN_READY_FILE = "data/stage3_train_ready.parquet"
OUT_CSV = "logs/debug_window_label_alignment.csv"


def log(msg):
    print(msg, flush=True)


def main():
    log("[4.3.1] 🚀 Stage 4.3.1 — Diagnostic Alignment Debug")

    # ----------------------------
    # Load features
    # ----------------------------
    feat = pd.read_csv(FEATURE_FILE)
    if "timestamp_utc" in feat.columns:
        ts_col = "timestamp_utc"
    elif "window_start" in feat.columns:
        ts_col = "window_start"
    else:
        ts_col = feat.columns[0]
    feat[ts_col] = pd.to_datetime(feat[ts_col], utc=True, errors="coerce")
    feat = feat.set_index(ts_col).sort_index()
    log(f"[4.3.1] 📥 features: {feat.shape}")

    # ----------------------------
    # Load labels (stage3)
    # ----------------------------
    lab = pd.read_parquet(TRAIN_READY_FILE)
    if not isinstance(lab.index, pd.DatetimeIndex):
        lab.index = pd.to_datetime(lab.index, utc=True, errors="coerce")

    cols_needed = ["lbl_mc_012", "mask_train", "gbpusd_ret_fwd20_pips"]
    for c in cols_needed:
        if c not in lab.columns:
            log(f"⚠️ Missing {c} in labels")

    lab = lab[cols_needed].copy()
    lab = lab.rename(columns={"lbl_mc_012": "label_mc"})
    log(f"[4.3.1] 📥 labels: {lab.shape}")

    # ----------------------------
    # Compute window_end (start + 500*5min)
    # ----------------------------
    WINDOW_SIZE = 500
    STEP = 250
    WINDOW_DURATION = pd.Timedelta(minutes=WINDOW_SIZE * 5)
    feat["window_start"] = feat.index
    feat["window_end"] = feat.index + WINDOW_DURATION

    # ----------------------------
    # Join labels at window_end (20 bars ahead ≈ 100 min)
    # ----------------------------
    joined = []
    for idx, row in feat.iterrows():
        ts_end = row["window_end"]
        # label about 20 bars ahead of window end
        target_ts = ts_end + pd.Timedelta(minutes=100)
        if target_ts in lab.index:
            fwd_move = lab.loc[target_ts, "gbpusd_ret_fwd20_pips"]
            joined.append((idx, ts_end, fwd_move))
    debug_df = pd.DataFrame(joined, columns=["window_start", "window_end", "fwd20_pips"])
    debug_df = debug_df.set_index("window_start")

    # ----------------------------
    # Diagnostics
    # ----------------------------
    mean_abs = debug_df["fwd20_pips"].abs().mean()
    pct_big = (debug_df["fwd20_pips"].abs() > 10).mean() * 100
    log(f"[4.3.1] 📊 Avg abs move after window: {mean_abs:.2f} pips")
    log(f"[4.3.1] 📊 % windows with >10 pips move: {pct_big:.2f}%")

    # ----------------------------
    # Save CSV
    # ----------------------------
    Path("logs").mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(OUT_CSV)
    log(f"[4.3.1] 💾 Saved debug table → {OUT_CSV} ({len(debug_df):,} rows)")

    # ----------------------------
    # Plot histogram
    # ----------------------------
    plt.figure(figsize=(7,4))
    debug_df["fwd20_pips"].hist(bins=50)
    plt.title("Distribution of forward 20-bar GBPUSD moves after window end")
    plt.xlabel("Forward 20-bar move (pips)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("logs/debug_window_label_alignment_hist.png")
    log("[4.3.1] 📈 Saved histogram → logs/debug_window_label_alignment_hist.png")

    log("[4.3.1] ✅ Diagnostic completed")


if __name__ == "__main__":
    main()