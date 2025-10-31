#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3a — Full-Data Training (LightGBM, Quant-Grade)
- Dùng toàn bộ dữ liệu 4 cặp (GBP_USD, EUR_USD, USD_JPY, XAU_USD) từ 2023→nay (đã clean ở Stage 1.2)
- Tạo feature nhanh (price-based) để train ngay, không phụ thuộc Stage 2
- Nhãn 3 lớp: 0=down, 1=flat, 2=up (mapping từ -1/0/+1 → 0/1/2)
- Split theo thời gian: Train=2023-01-01→2024-12-31, Valid=2025-01-01→2025-06-30, Test=2025-07-01→latest
- Lưu model/text + feature importance vào logs/
"""

import os
import math
import gc
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG
# =========================
PAIRS = ["GBP_USD", "EUR_USD", "USD_JPY", "XAU_USD"]
GRAN = "M5"
DATA_DIR = "data"
LOG_DIR = "logs"

# Horizon để dự đoán (n_bars M5)
N_AHEAD = 20  # ~100 phút

# Ngưỡng để phân lớp up/flat/down (theo pip; flat nếu |return| < THRESH_PIPS)
THRESH_PIPS = {
    "GBP_USD": 4,     # 4 pips
    "EUR_USD": 4,
    "USD_JPY": 4,     # 4 pips với pip_size 0.01
    "XAU_USD": 20,    # 20 "pips" vàng (pip_size=0.1)
}

# Pip size theo chuẩn
def pip_size(pair: str) -> float:
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair:
        return 0.1
    return 0.0001

# Split mốc thời gian
TRAIN_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
VALID_END = pd.Timestamp("2025-06-30 23:59:59", tz="UTC")

# LightGBM params (multiclass, >= v4.0, dùng callbacks)
LGB_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=2.0,
    verbosity=-1,
    n_jobs=28,  # EC2 32 cores → dành 28 threads cho LGBM
)

# =========================
# FEATURE ENGINEERING (nhanh)
# =========================
def add_fast_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    df: columns >= [mid_o, mid_h, mid_l, mid_c, volume, close, is_synthetic], UTC index
    Tạo features rolling nhanh (EMA, returns, volatility, RSI-lite, spreads, session).
    """
    df = df.copy()

    # Giá đóng cửa alias (chuẩn pipeline)
    df["close"] = df["mid_c"].astype("float32")

    # Returns & Log-returns
    df["ret_1"]  = df["close"].pct_change().astype("float32")
    df["ret_3"]  = df["close"].pct_change(3).astype("float32")
    df["ret_6"]  = df["close"].pct_change(6).astype("float32")
    df["ret_12"] = df["close"].pct_change(12).astype("float32")

    # Volatility (rolling std of returns)
    df["vol_12"] = df["ret_1"].rolling(12).std().astype("float32")  # ~1h
    df["vol_48"] = df["ret_1"].rolling(48).std().astype("float32")  # ~4h

    # High-Low spread & true range lite
    df["hl_spread"] = (df["mid_h"] - df["mid_l"]).astype("float32")
    df["hl_ema_48"] = df["hl_spread"].ewm(span=48, adjust=False).mean().astype("float32")

    # EMA cross
    df["ema_50"]  = df["close"].ewm(span=50, adjust=False).mean().astype("float32")
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean().astype("float32")
    df["ema_diff"] = (df["ema_50"] - df["ema_200"]).astype("float32")

    # RSI-lite (EWMA-based)
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = up / (dn.replace(0, np.nan))
    df["rsi_14"] = (100 - 100 / (1 + rs)).astype("float32")

    # Session/time features
    df["hour"] = df.index.hour.astype("int8")
    df["dow"]  = df.index.dayofweek.astype("int8")
    df["is_syn"] = df.get("is_synthetic", 0).astype("int8")

    # Volume transforms
    df["vol"] = df["volume"].astype("float32")
    df["vol_ema_48"] = df["vol"].ewm(span=48, adjust=False).mean().astype("float32")

    # Pair categorical
    df["pair"] = pair

    return df

# =========================
# LABELS (3 lớp: 0/1/2)
# =========================
def make_labels(df: pd.DataFrame, pair: str, n_ahead: int, th_pips: int) -> pd.Series:
    """
    Up if future close - now close > +threshold
    Down if < -threshold
    Else flat
    Map: down→0, flat→1, up→2 (theo policy đã lưu trong memory)
    """
    ps = pip_size(pair)
    fut = df["close"].shift(-n_ahead)
    diff = (fut - df["close"]).astype("float32")
    th = th_pips * ps

    lab = pd.Series(np.where(diff > th, 2, np.where(diff < -th, 0, 1)), index=df.index, dtype="int8")
    return lab

# =========================
# LOAD & BUILD DATASET
# =========================
def load_pair_df(pair: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{pair}_{GRAN}_clean.parquet"
    df = pd.read_parquet(path)

    # Dùng dữ liệu thật (loại synthetic) để train
    if "is_synthetic" in df.columns:
        df = df[df["is_synthetic"] == 0].copy()

    df = add_fast_features(df, pair)
    df["y"] = make_labels(df, pair, N_AHEAD, THRESH_PIPS[pair])

    # Drop đầu/cuối do rolling & shift
    df = df.dropna().copy()

    # Giảm memory
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    return df

def build_full_dataset(pairs):
    frames = []
    for p in pairs:
        part = load_pair_df(p)
        frames.append(part)
    full = pd.concat(frames).sort_index()
    return full

# =========================
# TRAIN / VALID / TEST SPLIT (theo thời gian)
# =========================
def time_split(df: pd.DataFrame):
    tr = df.loc[df.index <= TRAIN_END]
    va = df.loc[(df.index > TRAIN_END) & (df.index <= VALID_END)]
    te = df.loc[df.index > VALID_END]
    return tr, va, te

def select_features(df: pd.DataFrame):
    drop_cols = ["mid_o","mid_h","mid_l","mid_c","volume","close","y"]  # raw columns
    # giữ feature cols + pair/time features
    feats = [c for c in df.columns if c not in drop_cols]
    return feats

# =========================
# TRAINER
# =========================
def train_lgbm(X_train, y_train, X_valid, y_valid, categorical_cols):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols, reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(30, verbose=True),  # ✅ theo policy LightGBM ≥4.0
        lgb.log_evaluation(50),
    ]
    model = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train","valid"],
        num_boost_round=5000,
        callbacks=callbacks,
    )
    return model

# =========================
# EVAL
# =========================
def evaluate(model, X, y, tag):
    pred_proba = model.predict(X)
    pred = np.argmax(pred_proba, axis=1)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred, average="macro")
    print(f"[{tag}] ACC={acc:.4f} | F1-macro={f1:.4f}")
    print(classification_report(y, pred, digits=3))

# =========================
# MAIN
# =========================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("📥 Loading full dataset (4 pairs, 2023→now)…")
    df = build_full_dataset(PAIRS)
    print(f"✅ Data: {len(df):,} rows | {df.index[0]} → {df.index[-1]}")

    feats = select_features(df)
    categorical_cols = ["pair", "hour", "dow", "is_syn"] if "is_syn" in df.columns else ["pair", "hour", "dow"]

    X = df[feats]
    # LightGBM handle categorical via pandas category
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    y = df["y"].astype("int8")

    # Split
    tr, va, te = time_split(df)
    X_tr, y_tr = tr[feats], tr["y"].astype("int8")
    X_va, y_va = va[feats], va["y"].astype("int8")
    X_te, y_te = te[feats], te["y"].astype("int8")

    for c in categorical_cols:
        for split_X in (X_tr, X_va, X_te):
            if c in split_X.columns:
                split_X[c] = split_X[c].astype("category")

    print(f"🧪 Split: train={len(X_tr):,} | valid={len(X_va):,} | test={len(X_te):,}")

    # Train
    model = train_lgbm(X_tr, y_tr, X_va, y_va, [c for c in categorical_cols if c in X_tr.columns])

    # Eval
    evaluate(model, X_tr, y_tr, "TRAIN")
    evaluate(model, X_va, y_va, "VALID")
    if len(X_te) > 0:
        evaluate(model, X_te, y_te, "TEST")

    # Save model + importance
    model_path = os.path.join(LOG_DIR, "stage3_lgbm_full.txt")
    model.save_model(model_path)
    print(f"💾 Saved model → {model_path}")

    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(LOG_DIR, "stage3_feature_importance.csv")
    imp.to_csv(imp_path, index=False)
    print(f"💾 Saved feature importance → {imp_path}")

if __name__ == "__main__":
    main()