# ============================================================
# STAGE 2 — RE-EXTRACT FEATURES (FAST MODE)
# ------------------------------------------------------------
# ✅ Rebuild tsfresh features from GBP_USD_M5_2024.parquet
# ✅ Save to logs/stage2_features.csv for later stages
# ✅ Multi-core extraction for EC2 (32 CPU)
# ============================================================

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from datetime import timedelta
import os

# ---------- CONFIG ----------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
OUT_CSV = "logs/stage2_features.csv"

# Số nến ahead và TP để định nhãn giống Stage 1
TP_PIPS = 10
N_AHEAD = 20
PIP_SIZE = 0.0001

# ---------- LOAD DATA ----------
print(f"⏳ Loading data from {DATA_FILE} ...")
df = pd.read_parquet(DATA_FILE)
df = df[["close"]].copy()
df.index = pd.to_datetime(df.index)
df = df.tz_localize(None)
print(f"✅ Loaded {len(df)} rows | {df.index.min()} → {df.index.max()}")

# ---------- LABEL CREATION (target = price tăng vượt TP) ----------
future_close = df["close"].shift(-N_AHEAD)
df["target"] = ((future_close - df["close"]) / PIP_SIZE >= TP_PIPS).astype(int)

# ---------- PREPARE TSFRESH INPUT ----------
# group_id = 1 vì chỉ 1 cặp
df_tsfresh = df.reset_index().rename(columns={"index": "time"})
df_tsfresh["id"] = 1

# chỉ lấy vùng có target hợp lệ
df_tsfresh = df_tsfresh.dropna(subset=["target"])

# ---------- FEATURE EXTRACTION ----------
print("⏳ Extracting tsfresh features (multi-core, efficient mode)...")
settings = EfficientFCParameters()

features = extract_features(
    df_tsfresh[["id", "time", "close"]],
    column_id="id",
    column_sort="time",
    default_fc_parameters=settings,
    n_jobs=32,                 # full parallel for EC2
    disable_progressbar=False
)

# ---------- POST-PROCESS ----------
features = features.ffill().bfill()
features["target"] = df_tsfresh["target"].values[:len(features)]
print(f"✅ Features ready: {features.shape}")

# ---------- SAVE ----------
os.makedirs("logs", exist_ok=True)
features.to_csv(OUT_CSV)
print(f"✅ Saved to {OUT_CSV}")
print(features.head(5))