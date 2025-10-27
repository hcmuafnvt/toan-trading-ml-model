"""
STAGE 2 — Full Optimized for EC2 (save feature CSV)
---------------------------------------------------
✅ EfficientFCParameters (~800 feature)
✅ 16 core / 32 thread Xeon Platinum → n_jobs=28
✅ BUY=+1 | SELL=-1 | TIMEOUT=0
✅ Session bonus +5 pips (London/NY)
✅ Save LightGBM model (.txt) + feature list (.csv)
✅ Text-only summary
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# ========== CONFIG ==========
FILE = "data/GBP_USD_M5_2024.parquet"
PIP = 0.0001
SESSION_BONUS = 5
RR = 1.0
LOOKBACK_N = 240
STRIDE = 10
FC_PARAMS = EfficientFCParameters()
N_JOBS = 28

TARGETS = {
    "T1_10x40": {"tp": 10, "ahead": 40},
    "T2_15x60": {"tp": 15, "ahead": 60},
    "T3_20x80": {"tp": 20, "ahead": 80},
}

# ========== LOAD ==========
df = pd.read_parquet(FILE)
for base in ["mid_", "bid_", "ask_"]:
    if "close" not in df and f"{base}c" in df.columns:
        df["close"] = df[f"{base}c"]
    if "high" not in df and f"{base}h" in df.columns:
        df["high"] = df[f"{base}h"]
    if "low" not in df and f"{base}l" in df.columns:
        df["low"] = df[f"{base}l"]

df = df.dropna(subset=["close", "high", "low"]).copy()
df["hour"] = df.index.hour
def detect_session(h):
    if 7 <= h < 15: return "London"
    elif 12 <= h < 21: return "NewYork"
    else: return "Asia"
df["session"] = df["hour"].map(detect_session)

print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

# ========== LABEL ==========
def label_buy_sell(df, tp_pips=10, ahead=20, rr=1.0, pip=0.0001, bonus=0):
    n = len(df)
    res = np.zeros(n, dtype=np.int8)
    highs, lows, closes, sessions = df["high"].values, df["low"].values, df["close"].values, df["session"].values
    for i in range(n - ahead):
        tp_adj = tp_pips + (bonus if sessions[i] in ["London", "NewYork"] else 0)
        tp = tp_adj * pip; sl = tp * rr
        entry = closes[i]; tp_up = entry + tp; sl_down = entry - sl
        tp_down = entry - tp; sl_up = entry + sl
        sub_high, sub_low = highs[i+1:i+ahead+1], lows[i+1:i+ahead+1]
        hit_tp_up = np.where(sub_high >= tp_up)[0]
        hit_sl_down = np.where(sub_low <= sl_down)[0]
        hit_tp_down = np.where(sub_low <= tp_down)[0]
        hit_sl_up = np.where(sub_high >= sl_up)[0]
        buy_touch = np.inf if len(hit_tp_up)==0 else (hit_tp_up[0] if (len(hit_sl_down)==0 or hit_tp_up[0]<hit_sl_down[0]) else np.inf)
        sell_touch = np.inf if len(hit_tp_down)==0 else (hit_tp_down[0] if (len(hit_sl_up)==0 or hit_tp_down[0]<hit_sl_up[0]) else np.inf)
        if buy_touch==np.inf and sell_touch==np.inf: res[i]=0
        elif buy_touch<sell_touch: res[i]=1
        else: res[i]=-1
    return res

for name, cfg in TARGETS.items():
    df[name] = label_buy_sell(df, tp_pips=cfg["tp"], ahead=cfg["ahead"], rr=RR, pip=PIP, bonus=SESSION_BONUS)
    print(f"Label {name}: {pd.Series(df[name]).value_counts().to_dict()}")

# ========== BUILD LONG ==========
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

print("⏳ Extracting tsfresh features...")
long_df, sample_idx = build_long(df["close"], LOOKBACK_N, STRIDE)
X = extract_features(
    long_df,
    column_id="id",
    column_sort="time",
    default_fc_parameters=FC_PARAMS,
    n_jobs=N_JOBS,
    chunksize=50,
    disable_progressbar=False
)
impute(X)
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
print(f"✅ Features extracted: {X.shape}")

# ========== CLEAN COLUMN NAMES ==========
def clean_feature_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c) for c in df.columns]
    return df

# ========== TRAIN / EVAL ==========
def train_eval(X, y, name):
    y_num = pd.Series(y).replace({-1:0, 0:1, 1:2}).values
    X_buy = select_features(X, (y_num==2).astype(int))
    X_sell = select_features(X, (y_num==0).astype(int))
    cols_union = sorted(set(X_buy.columns).union(set(X_sell.columns)))
    X_sel = clean_feature_names(X[cols_union])

    # 💾 save feature names for inference
    pd.Series(X_sel.columns).to_csv(f"logs/{name}_features.csv", index=False)

    n = len(X_sel); split = int(n*0.8)
    X_train, X_test = X_sel.iloc[:split], X_sel.iloc[split:]
    y_train, y_test = y_num[:split], y_num[split:]

    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        n_jobs=N_JOBS,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n[{name}] report (0=SELL,1=TIMEOUT,2=BUY)")
    print(classification_report(y_test, y_pred, digits=3))

    imp = pd.Series(clf.feature_importances_, index=X_sel.columns)
    top5 = imp.sort_values(ascending=False).head(5)
    print(f"[{name}] top 5 features:")
    for k, v in top5.items():
        print(f"  {k}: {v}")

    # 💾 save model
    clf.booster_.save_model(f"logs/{name}_lightgbm.txt")
    print(f"[{name}] model + feature list saved ({len(cols_union)} features)")

def get_labels(col):
    return df[col].values[sample_idx]

# ========== RUN ==========
for name in TARGETS.keys():
    y = get_labels(name)
    train_eval(X, y, name)

print("\n✅ DONE — EC2 full optimized + saved feature lists.")