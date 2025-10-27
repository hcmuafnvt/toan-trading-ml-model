# ============================================================
# STAGE 2 (v3.1 fixed for Toan) — Extract tsfresh features
# ------------------------------------------------------------
# Input : data/GBP_USD_M5_2024.parquet (DateTime index)
# Output: logs/stage2_features.csv
# ============================================================

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
import os

# ---------------- CONFIG ----------------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
OUT_FILE = "logs/stage2_features.csv"

WINDOW = 200      # số nến
STRIDE = 5        # mỗi 5 nến sample 1 lần
TP_PIPS = [10, 15, 20]
N_AHEAD = [40, 60, 80]
PIP_SIZE = 0.0001

os.makedirs("logs", exist_ok=True)

# ---------------- LOAD DATA ----------------
print(f"⏳ Loading price data {DATA_FILE} ...")
df = pd.read_parquet(DATA_FILE)

# Nếu index là datetime thì reset thành cột time
if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("❌ Index phải là DatetimeIndex (datetime).")
df = df.reset_index().rename(columns={"index": "time"})

# Nếu chỉ có 'close', tự tạo open/high/low ảo cho tsfresh (± small noise)
if "open" not in df.columns:
    df["open"] = df["close"].shift(1).fillna(df["close"])
if "high" not in df.columns:
    df["high"] = df["close"] + np.random.uniform(0, 0.0005, len(df))
if "low" not in df.columns:
    df["low"] = df["close"] - np.random.uniform(0, 0.0005, len(df))
if "volume" not in df.columns:
    df["volume"] = 1.0

df = df.dropna().reset_index(drop=True)
print(f"✅ Loaded {len(df):,} rows | Columns: {list(df.columns)}")

# ---------------- TARGET CREATION ----------------
def create_targets(df, tp_pips, n_ahead):
    tp = tp_pips * PIP_SIZE
    labels = []
    for i in range(len(df) - n_ahead):
        future = df["close"].iloc[i + 1 : i + n_ahead + 1]
        ret = (future.values - df["close"].iloc[i]) / PIP_SIZE
        hit_tp = np.any(ret >= tp_pips)
        hit_sl = np.any(ret <= -tp_pips)
        if hit_tp and not hit_sl:
            labels.append(2)
        elif hit_sl and not hit_tp:
            labels.append(0)
        else:
            labels.append(1)
    labels += [1] * n_ahead
    return pd.Series(labels, index=df.index, name=f"target_{tp_pips}x{n_ahead}")

for tp, na in zip(TP_PIPS, N_AHEAD):
    df[f"target_{tp}x{na}"] = create_targets(df, tp, na)

print("✅ Targets created:", [c for c in df.columns if "target_" in c])

# ---------------- TSFRESH FEATURES ----------------
features = []
ids, close_all, high_all, low_all, vol_all = [], [], [], [], []
for i in range(0, len(df) - WINDOW, STRIDE):
    win = df.iloc[i : i + WINDOW]
    ids.extend([i] * len(win))
    close_all.extend(win["close"])
    high_all.extend(win["high"])
    low_all.extend(win["low"])
    vol_all.extend(win["volume"])

long_df = pd.DataFrame({
    "id": ids,
    "close": close_all,
    "high": high_all,
    "low": low_all,
    "volume": vol_all,
})

print("⏳ Extracting tsfresh features (MinimalFCParameters)...")
settings = MinimalFCParameters()
X = extract_features(long_df, column_id="id", default_fc_parameters=settings, n_jobs=16)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"✅ Features extracted: {X.shape}")

valid_idx = np.arange(0, len(df) - WINDOW, STRIDE)
targets = df.iloc[WINDOW::STRIDE][["target_10x40", "target_15x60", "target_20x80"]].reset_index(drop=True)
final_df = pd.concat([X.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

final_df.to_csv(OUT_FILE, index=False)
print(f"✅ Saved features → {OUT_FILE} ({final_df.shape[0]:,} rows, {final_df.shape[1]:,} cols)")