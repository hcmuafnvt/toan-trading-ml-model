#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.7 — Feature Importance Validation (GBPUSD)

Mục tiêu:
- Kiểm tra xem 5 feature top-alpha có thực sự có tín hiệu ổn định không.
- Train nhanh LightGBM → in AUC + top feature importance.

Input:
  logs/stage4_refined_features_gbpusd.csv
  logs/stage4_top_features_gbpusd.csv
  data/stage3_train_ready.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

FEATURE_FILE = "logs/stage4_refined_features_gbpusd.csv"
TOP_FEATURE_FILE = "logs/stage4_top_features_gbpusd.csv"
LABEL_FILE = "data/stage3_train_ready.parquet"

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)

def main():
    log("🚀 Stage 4.7 — Feature Importance Validation (GBPUSD)")

    # 1️⃣ Load data
    features = pd.read_csv(FEATURE_FILE, index_col=0, parse_dates=True)
    # load top features với nhiều schema khác nhau
    top_feats = pd.read_csv(TOP_FEATURE_FILE)

    if "feature" in top_feats.columns:
        selected = top_feats["feature"].dropna().tolist()
    elif "feature_name" in top_feats.columns:
        selected = top_feats["feature_name"].dropna().tolist()
    elif "top5_features" in top_feats.columns:
        selected = [f.strip() for f in top_feats["top5_features"].iloc[0].split(",")]
    else:
        raise KeyError("❌ Không tìm thấy cột feature_name / feature / top5_features trong top feature file")

    log(f"📊 Loaded top features: {selected}")

    labels = pd.read_parquet(LABEL_FILE)

    # standardize label naming
    if "target_label" not in labels.columns:
        labels = labels.rename(columns={"lbl_mc_012": "target_label",
                                        "mask_train": "target_is_trainable"})
    labels = labels[labels["target_is_trainable"] == 1]
    labels.index = pd.to_datetime(labels.index, utc=True)

    log(f"📊 Loaded features: {features.shape}")
    log(f"📊 Loaded top features: {list(top_feats['feature'])}")

    # 2️⃣ align labels and select features
    X = features[top_feats["feature"]]
    y = labels["target_label"].reindex(X.index, method="ffill").fillna(method="bfill").astype(int)

    log(f"🔗 Aligned data shape: {X.shape}, labels: {y.shape}")

    # 3️⃣ train/test split
    split = int(len(X) * 0.8)
    X_train, X_valid = X.iloc[:split], X.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    # 4️⃣ LightGBM training
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        seed=42,
        n_jobs=8,
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, preds)
    log(f"📈 Validation AUC: {auc:.4f}")

    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)

    log("🏆 Top features by importance:")
    print(imp.head(10).to_string(index=False))

    imp.to_csv("logs/stage4_feature_importance_validate_gbpusd.csv", index=False)
    log("💾 Saved → logs/stage4_feature_importance_validate_gbpusd.csv")
    log("✅ Stage 4.7 completed successfully")

if __name__ == "__main__":
    main()