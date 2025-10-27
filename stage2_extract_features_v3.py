# ============================================================
# STAGE 2 (v3.2 for OANDA mid schema) — Extract tsfresh features
# ------------------------------------------------------------
# Input : data/GBP_USD_M5_2024.parquet (DatetimeIndex, columns: mid_o, mid_h, mid_l, mid_c, volume)
# Output: logs/stage2_features.csv
# ============================================================

import os
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

# ---------------- CONFIG ----------------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
OUT_FILE = "logs/stage2_features.csv"

WINDOW = 200
STRIDE = 5
TARGET_SPECS = [
    ("target_10x40", 10, 40),
    ("target_15x60", 15, 60),
    ("target_20x80", 20, 80),
]
PIP_SIZE = 0.0001
N_JOBS = 28  # tận dụng 32 CPU EC2

os.makedirs("logs", exist_ok=True)

# ---------------- LOAD DATA ----------------
print(f"⏳ Loading price data {DATA_FILE} ...")
df = pd.read_parquet(DATA_FILE)

# Chuẩn hóa cột OHLC từ mid_*
if all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"]):
    df["open"], df["high"], df["low"], df["close"] = (
        df["mid_o"], df["mid_h"], df["mid_l"], df["mid_c"]
    )
elif all(c in df.columns for c in ["bid_o", "bid_h", "bid_l", "bid_c"]):
    df["open"], df["high"], df["low"], df["close"] = (
        df["bid_o"], df["bid_h"], df["bid_l"], df["bid_c"]
    )
elif all(c in df.columns for c in ["ask_o", "ask_h", "ask_l", "ask_c"]):
    df["open"], df["high"], df["low"], df["close"] = (
        df["ask_o"], df["ask_h"], df["ask_l"], df["ask_c"]
    )
else:
    raise ValueError("Không tìm thấy các cột mid_*, bid_* hoặc ask_* trong file!")

if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("Index phải là DatetimeIndex (thời gian).")

df = df.sort_index()
df.index = df.index.tz_localize(None)
print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

# ---------------- BUILD LONG-FORM ----------------
def build_long_form(df, window, stride):
    ids, times, values = [], [], []
    for i, t in enumerate(range(window, len(df), stride)):
        seg = df["close"].iloc[t - window : t].values
        ids.append(np.full(window, i))
        times.append(np.arange(window))
        values.append(seg)
    long_df = pd.DataFrame({
        "id": np.concatenate(ids),
        "time": np.concatenate(times),
        "close": np.concatenate(values),
    })
    return long_df

print("⏳ Building long-form for tsfresh ...")
long_df = build_long_form(df, WINDOW, STRIDE)
print(f"✅ Built long-form: {len(long_df):,} rows, {len(long_df['id'].unique()):,} samples")

# ---------------- EXTRACT FEATURES ----------------
print("⏳ Extracting tsfresh features ...")
fc = EfficientFCParameters()
X = extract_features(
    long_df,
    column_id="id",
    column_sort="time",
    default_fc_parameters=fc,
    n_jobs=N_JOBS,
    disable_progressbar=False,
)
impute(X)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X.columns = [c.replace(" ", "_").replace("(", "_").replace(")", "_") for c in X.columns]
print(f"✅ Features extracted: {X.shape}")

# ---------------- TARGET CREATION ----------------
def make_target(df, tp_pips, ahead):
    tp = tp_pips * PIP_SIZE
    y = []
    for i in range(len(df) - ahead):
        future = df["close"].iloc[i + 1 : i + ahead + 1]
        diff = (future.values - df["close"].iloc[i]) / PIP_SIZE
        up_hit = np.any(diff >= tp_pips)
        dn_hit = np.any(diff <= -tp_pips)
        if up_hit and not dn_hit:
            y.append(2)
        elif dn_hit and not up_hit:
            y.append(0)
        else:
            y.append(1)
    y += [1] * ahead
    return pd.Series(y, index=df.index, dtype=np.int8)

targets = {}
for name, tp, ahead in TARGET_SPECS:
    targets[name] = make_target(df, tp, ahead)
    vc = targets[name].value_counts().sort_index()
    print(f"✅ {name}: {dict(vc)}")

# Align theo stride
target_df = pd.DataFrame({k: v.iloc[WINDOW::STRIDE].reset_index(drop=True) for k, v in targets.items()})
final_df = pd.concat([X.reset_index(drop=True), target_df], axis=1)

final_df.to_csv(OUT_FILE, index=False)
print(f"✅ Saved features → {OUT_FILE} ({final_df.shape[0]:,} rows, {final_df.shape[1]:,} cols)")