#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.4 — Feature Refinement & Quick Retrain (GBPUSD)

Purpose:
- Align tsfresh feature windows with labels from Stage 3.
- Enrich with directional meta-features (mean/std/slope/etc.).
- Scale features and run a quick LightGBM sanity test to check alpha.

Inputs:
    logs/stage4_tsfresh_features_gbpusd.csv
    data/stage3_train_ready.parquet
Outputs:
    logs/stage4_refined_features_gbpusd.csv
    logs/stage4_refined_model_gbpusd.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


FEATURE_CSV = "logs/stage4_tsfresh_features_gbpusd.csv"
LABEL_FILE  = "data/stage3_train_ready.parquet"
OUT_FEATURE = "logs/stage4_refined_features_gbpusd.csv"
OUT_MODEL   = "logs/stage4_refined_model_gbpusd.txt"


def log(msg: str):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)


def main():
    log("🚀 Stage 4.4 — Feature Refinement & Quick Retrain (GBPUSD)")

    # -----------------------------
    # 1️⃣ Load data
    # -----------------------------
    features = pd.read_csv(FEATURE_CSV, index_col=0, parse_dates=True)
    log(f"📥 Loaded features: {features.shape}")

    labels = pd.read_parquet(LABEL_FILE)
    labels.index = pd.to_datetime(labels.index, utc=True)

    # Standardize label column names
    if "target_label" not in labels.columns:
        labels = labels.rename(columns={
            "lbl_mc_012": "target_label",
            "mask_train": "target_is_trainable",
            "reason": "target_drop_reason"
        })

    # Keep only needed columns (but don't filter yet)
    labels = labels[["target_label", "target_is_trainable"]].copy()

    # Sort for merge_asof
    features = features.sort_index()
    labels = labels.sort_index()

    # -----------------------------
    # 2️⃣ Align features ↔ labels
    # -----------------------------
    log("🔗 Aligning features with labels via merge_asof...")
    # ✅ merge_asof: features → labels (forward)
    merged = pd.merge_asof(
        features.sort_index(),
        labels.sort_index(),
        left_index=True,
        right_index=True,
        direction="forward",            # lấy label ngay sau window_end_time
        tolerance=pd.Timedelta("48h"),  # cho phép lệch tối đa 2 ngày
    )

    log(f"[DEBUG] merged shape pre-dropna: {merged.shape}")
    log(f"[DEBUG] merged time range: {merged.index.min()} → {merged.index.max()}")
    log(f"[DEBUG] NaN ratio: {merged.isna().mean().mean():.3f}")    

    # Only now filter trainable
    if "target_is_trainable" in merged.columns:
        merged = merged[merged["target_is_trainable"] == 1]
        log(f"[DEBUG] kept trainable rows: {len(merged)}")

    if merged.empty:
        raise RuntimeError("❌ Merged dataset empty after alignment or filtering — check label timestamps.")

    X = merged.drop(columns=["target_label", "target_is_trainable"])
    y = merged["target_label"].astype(int)
    y = y.clip(upper=1)  # 0=SHORT, 1=LONG (gom class 2 vào 1)

    log(f"🔗 Final aligned samples: {X.shape}")

    # -----------------------------
    # 3️⃣ Directional meta-features
    # -----------------------------
    log("➕ Adding directional meta-features...")
    X["feat_mean"] = X.mean(axis=1)
    X["feat_std"] = X.std(axis=1)
    X["feat_skew"] = X.skew(axis=1)
    X["feat_kurt"] = X.kurtosis(axis=1)
    X["feat_slope"] = X["feat_mean"].diff().fillna(0)
    X["feat_volratio"] = (X["feat_std"] / (X["feat_mean"].abs() + 1e-6)).fillna(0)

    # -----------------------------
    # 4️⃣ Standardization
    # -----------------------------
    log("⚖️ Scaling features (z-score)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    
    # ✅ Fill NaN sau scale (LightGBM không nhận NaN)
    X_scaled = X_scaled.fillna(0)

    Path("logs").mkdir(exist_ok=True)
    X_scaled.to_csv(OUT_FEATURE)
    log(f"💾 Saved refined features → {OUT_FEATURE} ({X_scaled.shape})")

    # -----------------------------
    # 5️⃣ Quick LightGBM train/test
    # -----------------------------
    log("⚙️ Quick train/validation split...")
    split = int(len(X_scaled) * 0.8)
    X_train, X_valid = X_scaled.iloc[:split], X_scaled.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=42,
        n_jobs=28,
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    log("🚀 Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # -----------------------------
    # 6️⃣ Evaluation
    # -----------------------------
    preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, preds)
    preds_bin = (preds > 0.5).astype(int)
    acc = accuracy_score(y_valid, preds_bin)

    log(f"📊 Validation AUC: {auc:.4f}")
    log(f"📊 Accuracy: {acc:.4f}")
    log("🔎 Classification report:")
    print(classification_report(y_valid, preds_bin, target_names=["SHORT(0)", "LONG(1)"]))

    model.save_model(OUT_MODEL)
    log(f"💾 Model saved → {OUT_MODEL}")
    log("✅ Stage 4.4 completed successfully")


if __name__ == "__main__":
    main()