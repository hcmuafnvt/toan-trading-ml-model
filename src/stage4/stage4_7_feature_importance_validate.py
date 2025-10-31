#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.7 — Feature Importance Validation (GBPUSD)

Mục tiêu:
1️⃣ Load top-α features từ Stage 4.6.
2️⃣ Lọc subset các features tương ứng trong stage4_refined_features.
3️⃣ Train lại LightGBM và kiểm tra feature importance (AUC + ranking).

Input:
    logs/stage4_top_features_gbpusd.csv
    logs/stage4_refined_features_gbpusd.csv
    data/stage3_train_ready.parquet
Output:
    logs/stage4_feature_importance_gbpusd.csv
    logs/stage4_feature_importance_model.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

TOP_FEATURE_FILE = "logs/stage4_top_features_gbpusd.csv"
FEATURE_FILE     = "logs/stage4_refined_features_gbpusd.csv"
LABEL_FILE       = "data/stage3_train_ready.parquet"
OUT_IMPORTANCE   = "logs/stage4_feature_importance_gbpusd.csv"
OUT_MODEL        = "logs/stage4_feature_importance_model.txt"

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)

def main():
    log("🚀 Stage 4.7 — Feature Importance Validation (GBPUSD)")

    # -----------------------------
    # 1️⃣ Load top features (support multiple schemas)
    # -----------------------------
    top_feats = pd.read_csv(TOP_FEATURE_FILE)

    if "feature" in top_feats.columns:
        selected = top_feats["feature"].dropna().tolist()
    elif "feature_name" in top_feats.columns:
        selected = top_feats["feature_name"].dropna().tolist()
    elif "top5_features" in top_feats.columns:
        selected = [f.strip() for f in top_feats["top5_features"].iloc[0].split(",")]
    else:
        raise KeyError("❌ Không tìm thấy cột feature_name / feature / top5_features trong file top features")

    log(f"📊 Loaded top features: {selected}")

    # -----------------------------
    # 2️⃣ Load refined features
    # -----------------------------
    features = pd.read_csv(FEATURE_FILE, index_col=0, parse_dates=True)
    log(f"📊 Loaded refined feature set: {features.shape}")

    # Chỉ giữ lại những cột có trong selected
    features = features[[c for c in selected if c in features.columns]]
    log(f"🎯 Using {len(features.columns)} selected features")

    # -----------------------------
    # 3️⃣ Load labels
    # -----------------------------
    labels = pd.read_parquet(LABEL_FILE)
    if "target_label" not in labels.columns:
        labels = labels.rename(columns={
            "lbl_mc_012": "target_label",
            "mask_train": "target_is_trainable"
        })

    labels.index = pd.to_datetime(labels.index, utc=True)
    labels = labels.sort_index()

    # Khớp label với index của features
    y = labels.loc[features.index, "target_label"].fillna(method="ffill").astype(int)
    log(f"📈 Matched labels: {y.shape}")

    # -----------------------------
    # 4️⃣ Train / validation split
    # -----------------------------
    split = int(len(features) * 0.8)
    X_train, X_valid = features.iloc[:split], features.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    # -----------------------------
    # 5️⃣ LightGBM training
    # -----------------------------
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=42,
        n_jobs=28,
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    log("🚀 Training LightGBM on top features...")
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # -----------------------------
    # 6️⃣ Evaluate
    # -----------------------------
    preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, preds)
    preds_bin = (preds > 0.5).astype(int)
    acc = accuracy_score(y_valid, preds_bin)

    log(f"📊 Validation AUC: {auc:.4f}")
    log(f"📊 Accuracy: {acc:.4f}")

    # -----------------------------
    # 7️⃣ Feature importance
    # -----------------------------
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)

    importance.to_csv(OUT_IMPORTANCE, index=False)
    model.save_model(OUT_MODEL)

    log(f"💾 Saved feature importance → {OUT_IMPORTANCE}")
    log(f"💾 Saved model → {OUT_MODEL}")
    log("✅ Stage 4.7 completed successfully")

if __name__ == "__main__":
    main()