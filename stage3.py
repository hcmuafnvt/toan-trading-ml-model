"""
STAGE 3 — Fusion + VectorBT Backtest (FINAL)
--------------------------------------------
✅ Load 3 LightGBM models (T1,T2,T3)
✅ Load đúng feature list (*.csv)
✅ Majority-vote fusion BUY/SELL/TIMEOUT
✅ Backtest bằng vectorbt
✅ EC2-optimized (n_jobs=28)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import vectorbt as vbt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import re, warnings
warnings.filterwarnings("ignore")

# ========= CONFIG =========
PAIR = "GBP_USD"
FILE = f"data/{PAIR}_M5_2024.parquet"
PIP_SIZE = 0.0001
LOOKBACK_N = 240
STRIDE = 10
FC_PARAMS = EfficientFCParameters()
N_JOBS = 28

MODELS = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}

# ========= LOAD PRICE =========
df = pd.read_parquet(FILE)
for base in ["mid_", "bid_", "ask_"]:
    if "close" not in df and f"{base}c" in df.columns:
        df["close"] = df[f"{base}c"]
df = df.dropna(subset=["close"]).copy()
print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

# ========= BUILD WINDOWS =========
def build_long(series, window, stride):
    vals, n = series.values, len(series)
    ids, times, values, idxs = [], [], [], []
    local_idx = np.arange(window)
    t = window
    while t < n:
        ids.append(np.full(window, t))
        times.append(local_idx)
        values.append(vals[t-window:t])
        idxs.append(t)
        t += stride
    long_df = pd.DataFrame({
        "id": np.concatenate(ids),
        "time": np.concatenate(times),
        "close": np.concatenate(values).astype("float32"),
    })
    return long_df, np.array(idxs, dtype=int)

print("⏳ Extracting tsfresh features (fast)...")
long_df, sample_idx = build_long(df["close"], LOOKBACK_N, STRIDE)
X = extract_features(
    long_df,
    column_id="id", column_sort="time",
    default_fc_parameters=FC_PARAMS,
    n_jobs=N_JOBS, disable_progressbar=False
)
impute(X)
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
X.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c) for c in X.columns]
print(f"✅ Features ready: {X.shape}")

# ========= PREDICT =========
pred_df = pd.DataFrame(index=X.index)
for name, path in MODELS.items():
    model = lgb.Booster(model_file=path)
    feat_file = f"logs/{name}_features.csv"
    feat_names = pd.read_csv(feat_file, header=None)[0].tolist()
    common = [c for c in feat_names if c in X.columns]
    X_pred = X[common].copy()
    probs = model.predict(X_pred, predict_disable_shape_check=True)
    pred_df[name] = np.argmax(probs, axis=1)   # 0=SELL,1=TIMEOUT,2=BUY
    print(f"✅ {name} prediction done | {len(common)} features used")

# ========= FUSION =========
vote_buy  = (pred_df == 2).sum(axis=1)
vote_sell = (pred_df == 0).sum(axis=1)
signal = np.where(vote_buy >= 2, 1, np.where(vote_sell >= 2, -1, 0))

df_bt = df.iloc[sample_idx].copy()
df_bt["signal"] = signal
print("\nSignal distribution:")
print(df_bt["signal"].value_counts())

# ========= BACKTEST =========
tp = 10 * PIP_SIZE
sl = 10 * PIP_SIZE
entries = df_bt["signal"] == 1
exits   = df_bt["signal"] == -1

pf = vbt.Portfolio.from_signals(
    df_bt["close"],
    entries=entries,
    exits=exits,
    sl_stop=sl,
    tp_stop=tp,
    size=1.0,
    fees=0.0002,
    direction="both"
)

print("\n========== FUSION BACKTEST ==========")
stats = pf.stats()
print(f"Total Trades : {stats['Total Trades']}")
print(f"Win Rate [%] : {stats['Win Rate [%]']:.2f}")
print(f"Total Return [%] : {stats['Total Return [%]']:.2f}")
print(f"Profit Factor : {stats['Profit Factor']:.2f}")
print(f"Max Drawdown [%] : {stats['Max Drawdown [%]']:.2f}")
print(f"Expectancy [%] : {stats['Expectancy [%]']:.2f}")
print("====================================")
