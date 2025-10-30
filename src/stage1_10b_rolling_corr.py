#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.10b — Rolling macro correlation QA / regime diagnostics

Purpose:
- Examine how stable key macro relationships are over time.
- Detect regime shifts (e.g. carry breakdown, yield-curve panic, risk-on vs risk-off).

Inputs:
- data/macro_fxdrivers.parquet  (output of Stage 1.10)

Outputs:
- logs/stage1_10b_rolling_corr.txt     (text summary of stats)
- logs/rollcorr_US2Y_USJP.png          (UST2Y vs US_JP_SPREAD + rolling corr)
- logs/rollcorr_curve_VIX.png          (UST2Y_10Y_SPREAD vs VIX + rolling corr)
- logs/rollcorr_VIX_SPX.png            (VIX vs SPX + rolling corr)

Notes:
- We are still in Stage 1, so NO FX spot pairs merged yet. That comes in Stage 2.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # allow saving plots headless (no GUI)
import matplotlib.pyplot as plt

DATA_FILE = "data/macro_fxdrivers.parquet"
LOG_TXT   = "logs/stage1_10b_rolling_corr.txt"

ROLL_SHORT = 60    # ~3 months trading days
ROLL_LONG  = 252   # ~1y trading days

def ensure_dirs():
    Path("logs").mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_parquet(DATA_FILE)
    # ensure datetime sorted
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def rolling_corr(df, col_x, col_y, window):
    """
    Return rolling correlation series between two columns.
    """
    return df[col_x].rolling(window).corr(df[col_y])

def plot_pair_with_corr(df, col_x, col_y, corr_short, corr_long, out_file, title):
    """
    Plot:
      top: the two raw series (normalized z-score so they can be on same axis)
      bottom: rolling corr short/long
    Save to out_file (PNG).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # z-score each series for comparability
    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    axes[0].plot(df["date"], zscore(df[col_x]), label=col_x)
    axes[0].plot(df["date"], zscore(df[col_y]), label=col_y)
    axes[0].set_ylabel("z-score")
    axes[0].set_title(f"{title} — normalized levels")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["date"], corr_short, label=f"rolling {ROLL_SHORT}d corr")
    axes[1].plot(df["date"], corr_long,  label=f"rolling {ROLL_LONG}d corr", alpha=0.6)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("corr")
    axes[1].set_title("Rolling correlation (short vs long window)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

def summarize_stats(corr_series, label, fh):
    """
    Write basic stats for a rolling corr series to log file.
    Also flag if corr ever flips sign or collapses.
    """
    valid = corr_series.dropna()
    if valid.empty:
        fh.write(f"[{label}] no valid window yet\n\n")
        return

    corr_min   = valid.min()
    corr_max   = valid.max()
    corr_med   = valid.median()
    corr_last  = valid.iloc[-1]

    flips_sign = ( (valid >  0).any() and (valid < 0).any() )

    fh.write(f"[{label}]\n")
    fh.write(f"  median: {corr_med:.2f}\n")
    fh.write(f"  min   : {corr_min:.2f}\n")
    fh.write(f"  max   : {corr_max:.2f}\n")
    fh.write(f"  last  : {corr_last:.2f}\n")
    fh.write(f"  flipped sign historically? {flips_sign}\n\n")

def main():
    ensure_dirs()
    df = load_data()

    # ---------------------------------
    # 1. UST2Y vs US_JP_SPREAD
    # carry / rate differential to Japan
    # ---------------------------------
    pair1_x = "UST2Y"
    pair1_y = "US_JP_SPREAD"

    c1_short = rolling_corr(df, pair1_x, pair1_y, ROLL_SHORT)
    c1_long  = rolling_corr(df, pair1_x, pair1_y, ROLL_LONG)

    plot_pair_with_corr(
        df, pair1_x, pair1_y,
        c1_short, c1_long,
        out_file="logs/rollcorr_US2Y_USJP.png",
        title="UST2Y vs US_JP_SPREAD (USD vs JPY rate advantage)"
    )

    # ---------------------------------
    # 2. UST2Y_10Y_SPREAD vs VIX
    # curve inversion vs risk stress
    # ---------------------------------
    pair2_x = "UST2Y_10Y_SPREAD"
    pair2_y = "VIX"

    c2_short = rolling_corr(df, pair2_x, pair2_y, ROLL_SHORT)
    c2_long  = rolling_corr(df, pair2_x, pair2_y, ROLL_LONG)

    plot_pair_with_corr(
        df, pair2_x, pair2_y,
        c2_short, c2_long,
        out_file="logs/rollcorr_curve_VIX.png",
        title="UST2Y_10Y_SPREAD vs VIX (yield curve stress vs fear)"
    )

    # ---------------------------------
    # 3. VIX vs SPX
    # equity risk sentiment
    # ---------------------------------
    pair3_x = "VIX"
    pair3_y = "SPX"

    c3_short = rolling_corr(df, pair3_x, pair3_y, ROLL_SHORT)
    c3_long  = rolling_corr(df, pair3_x, pair3_y, ROLL_LONG)

    plot_pair_with_corr(
        df, pair3_x, pair3_y,
        c3_short, c3_long,
        out_file="logs/rollcorr_VIX_SPX.png",
        title="VIX vs SPX (risk-off vs equities)"
    )

    # ---------------------------------
    # Write text diagnostics
    # ---------------------------------
    with open(LOG_TXT, "w") as fh:
        fh.write("Stage 1.10b Rolling Correlation QA\n")
        fh.write(f"Data rows: {len(df)}\n")
        fh.write(f"Date range: {df['date'].min()} → {df['date'].max()}\n\n")

        summarize_stats(c1_short, "UST2Y vs US_JP_SPREAD (60d)", fh)
        summarize_stats(c1_long,  "UST2Y vs US_JP_SPREAD (252d)", fh)

        summarize_stats(c2_short, "UST2Y_10Y_SPREAD vs VIX (60d)", fh)
        summarize_stats(c2_long,  "UST2Y_10Y_SPREAD vs VIX (252d)", fh)

        summarize_stats(c3_short, "VIX vs SPX (60d)", fh)
        summarize_stats(c3_long,  "VIX vs SPX (252d)", fh)

    print("✅ Stage 1.10b complete.")
    print("   → Saved logs/stage1_10b_rolling_corr.txt")
    print("   → Saved plots:")
    print("        logs/rollcorr_US2Y_USJP.png")
    print("        logs/rollcorr_curve_VIX.png")
    print("        logs/rollcorr_VIX_SPX.png")

if __name__ == "__main__":
    main()