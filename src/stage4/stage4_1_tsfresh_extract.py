# ============================================================
# Stage 4.1 — Tsfresh Feature Extraction (GBPUSD only)
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import re, time

# ------------------------------------------------------------
INPUT_FILE = "data/stage3_feature_stack.parquet"
OUT_FILE   = "logs/stage4_tsfresh_features_gbpusd.csv"
PAIR_COL   = "px_gbpusd_close"

# ------------------------------------------------------------
def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def sanitize_columns(df):
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )
    return df

# ------------------------------------------------------------
def main():
    log("🚀 Stage 4.1 — Tsfresh Feature Extraction (GBPUSD)")
    df = pd.read_parquet(INPUT_FILE)
    log(f"📥 Loaded {INPUT_FILE} ({len(df):,} rows)")

    # Filter only trainable rows
    df = df[df["target_is_trainable"] == 1].copy()

    # Select the close price
    series = df[[PAIR_COL]].copy().reset_index()
    series.rename(columns={PAIR_COL: "value", "timestamp_utc": "time"}, inplace=True)
    series["id"] = 0  # single series id

    log("⚙️ Extracting features via tsfresh (EfficientFCParameters)")
    settings = EfficientFCParameters()
    features = extract_features(
        series,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        n_jobs=28,
        disable_progressbar=False,
    )

    # Sanitize names
    features = sanitize_columns(features)

    # Save result
    Path("logs").mkdir(exist_ok=True)
    features.to_csv(OUT_FILE)
    log(f"💾 Saved → {OUT_FILE} ({features.shape[1]} features)")

    log("✅ Stage 4.1 completed successfully")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()