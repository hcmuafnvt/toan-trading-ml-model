#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3b — LightGBM with Stage 2 Features
-----------------------------------------
• Full-data (GBP_USD, EUR_USD, USD_JPY, XAU_USD)
• Merge meta-features from logs/stage2_features.csv
• Label = 3 classes (0 down, 1 flat, 2 up)
• Split: 2023-01-01→2024-12-31 (train), 2025-01-01→06-30 (valid), 2025-07-01→latest (test)
"""

import os
import gc
import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from multiprocessing import cpu_count

# ---------- CONFIG ----------
PAIRS = ["GBP_USD", "EUR_USD", "USD_JPY", "XAU_USD"]
GRAN = "M5"
DATA_DIR = "data"
LOG_DIR = "logs"
FEATURE_FILE = os.path.join(LOG_DIR, "stage2_features.csv")

TRAIN_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
VALID_END = pd.Timestamp("2025-06-30 23:59:59", tz="UTC")
N_AHEAD = 20

THRESH_PIPS = {"GBP_USD":4, "EUR_USD":4, "USD_JPY":4, "XAU_USD":20}
def pip_size(p): return 0.01 if "JPY" in p else 0.1 if "XAU" in p else 0.0001

LGB_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=127,
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    verbosity=-1,
    n_jobs=max(cpu_count()-4,4)
)

# ---------- UTILS ----------
def add_fast_features(df, pair):
    df["close"] = df["mid_c"].astype("float32")
    df["ret_1"] = df["close"].pct_change()
    df["ret_6"] = df["close"].pct_change(6)
    df["vol_48"] = df["ret_1"].rolling(48).std()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()
    df["ema_diff"] = df["ema_50"] - df["ema_200"]
    df["rsi_14"] = 100 - 100 / (1 + (df["close"].diff().clip(lower=0)
                                     .ewm(span=14).mean() /
                                     (-df["close"].diff().clip(upper=0))
                                     .ewm(span=14).mean().replace(0,np.nan)))
    df["hour"] = df.index.hour.astype("int8")
    df["dow"] = df.index.dayofweek.astype("int8")
    df["pair"] = pair
    df["is_syn"] = df.get("is_synthetic",0).astype("int8")
    df["vol"] = df["volume"].astype("float32")
    return df

def make_labels(df, pair, n_ahead, th_pips):
    ps = pip_size(pair)
    fut = df["close"].shift(-n_ahead)
    diff = fut - df["close"]
    th = th_pips * ps
    return np.where(diff > th, 2, np.where(diff < -th, 0, 1)).astype("int8")

def merge_stage2_features(df, pair):
    if not os.path.exists(FEATURE_FILE):
        return df
    feat = pd.read_csv(FEATURE_FILE)
    row = feat.loc[feat["pair"] == pair].drop(columns=["pair"], errors="ignore")
    for c in row.columns:
        df[f"sf_{c}"] = row.iloc[0, row.columns.get_loc(c)]
    return df

def load_pair_df(pair):
    print(f"🔹 Loading {pair} ...")
    df = pd.read_parquet(f"{DATA_DIR}/{pair}_{GRAN}_clean.parquet")
    if "is_synthetic" in df.columns:
        df = df[df["is_synthetic"] == 0]
    print(f"🔹 After removing synthetic: {len(df):,} rows")
    df = add_fast_features(df, pair)
    df["y"] = make_labels(df, pair, N_AHEAD, THRESH_PIPS[pair])
    df = merge_stage2_features(df, pair)
    df = df.dropna().astype("float32", errors="ignore")
    return df

def build_full_dataset(pairs): return pd.concat([load_pair_df(p) for p in pairs]).sort_index()

def time_split(df):
    tr = df.loc[df.index <= TRAIN_END]
    va = df.loc[(df.index > TRAIN_END) & (df.index <= VALID_END)]
    te = df.loc[df.index > VALID_END]
    return tr, va, te

def select_features(df):
    drop = ["mid_o","mid_h","mid_l","mid_c","volume","close","y"]
    return [c for c in df.columns if c not in drop]

# ---------- TRAIN ----------
def train_lgbm(Xtr, ytr, Xva, yva, cats):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr, free_raw_data=False)
    model = lgb.train(
        params=LGB_PARAMS,
        train_set=dtr,
        valid_sets=[dtr, dva],
        valid_names=["train","valid"],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return model

def evaluate(model, X, y, tag):
    p = model.predict(X)
    yhat = np.argmax(p, axis=1)
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, average="macro")
    print(f"[{tag}] ACC={acc:.4f} | F1-macro={f1:.4f}")
    print(classification_report(y, yhat, digits=3))

# ---------- MAIN ----------
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("📥 Loading full dataset with Stage 2 features…")
    df = build_full_dataset(PAIRS)
    print(f"✅ {len(df):,} rows  {df.index[0]} → {df.index[-1]}")
    feats = select_features(df)
    cats = [c for c in ["pair","hour","dow","is_syn"] if c in feats]
    for c in cats:
        df[c] = df[c].astype("category")
    tr, va, te = time_split(df)
    print(f"🧪 train={len(tr):,} | valid={len(va):,} | test={len(te):,}")

    model = train_lgbm(tr[feats], tr["y"], va[feats], va["y"], cats)
    evaluate(model, tr[feats], tr["y"], "TRAIN")
    evaluate(model, va[feats], va["y"], "VALID")
    if len(te)>0: evaluate(model, te[feats], te["y"], "TEST")

    mpath = os.path.join(LOG_DIR,"stage3b_lgbm_with_stage2.txt")
    model.save_model(mpath)
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(LOG_DIR,"stage3b_feature_importance.csv"),index=False)
    print(f"💾 Saved model → {mpath}")

if __name__ == "__main__":
    main()