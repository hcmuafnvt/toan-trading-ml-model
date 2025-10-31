# ============================================================
# Stage 4.2 — Feature Selection & Sanitization (GBPUSD)
# ============================================================
import pandas as pd, numpy as np, time, re
from pathlib import Path

IN_FILE  = "logs/stage4_tsfresh_features_gbpusd.csv"
OUT_FILE = "logs/stage4_selected_features_gbpusd.csv"

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def sanitize_columns(df):
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )
    return df

def main():
    log("🚀 Stage 4.2 — Feature Selection & Sanitization (GBPUSD)")
    df = pd.read_csv(IN_FILE, index_col=0)
    log(f"📥 Loaded {IN_FILE} ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")

    # --- Step 1: Drop all-NaN or mostly NaN (>20%)
    nan_ratio = df.isna().mean()
    keep = nan_ratio[nan_ratio <= 0.2].index
    df = df[keep]
    log(f"🧹 Removed {len(nan_ratio) - len(keep)} cols with >20% NaN")

    # --- Step 2: Fill remaining NaN with median
    df = df.fillna(df.median(numeric_only=True))

    # --- Step 3: Drop near-zero variance
    var = df.var()
    keep = var[var > 1e-10].index
    df = df[keep]
    log(f"⚖️ Removed {len(var) - len(keep)} low-variance cols")

    # --- Step 4: Drop highly correlated features (|ρ| > 0.95)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)
    log(f"🔗 Removed {len(to_drop)} highly correlated cols")

    # --- Step 5: Sanitize & save
    df = sanitize_columns(df)
    Path("logs").mkdir(exist_ok=True)
    df.to_csv(OUT_FILE)
    log(f"💾 Saved clean feature set → {OUT_FILE} ({df.shape[1]} features)")
    log("✅ Stage 4.2 completed successfully")

if __name__ == "__main__":
    main()