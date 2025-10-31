#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.3 — Model Training (GBPUSD)

Goal:
- Join engineered features (stage4_selected_features_gbpusd.csv)
  with supervised training targets (stage3_train_ready.parquet)
- Filter only rows allowed for training (target_is_trainable == 1)
- Binary classify directional move: SHORT(0) vs LONG(1)
  where LONG=label 2, SHORT=label 0, label 1 (choppy) is dropped
- Train LightGBM classifier with time-based split
- Export model + feature importance

Inputs:
- logs/stage4_selected_features_gbpusd.csv
    • columns: [window_id, feature1, feature2, ..., featureN]
    • index column: timestamp window start (we saved it as index)
- data/stage3_train_ready.parquet
    • columns include:
        target_label (0,1,2)
        target_is_trainable (0/1)
        target_drop_reason
    • index: timestamp_utc (M5)

Outputs:
- logs/stage4_lgbm_gbpusd.txt                (LightGBM model dump)
- logs/stage4_feature_importance_gbpusd.csv  (feature importances table)
- metrics printed to stdout
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ========== CONFIG ==========
FEATURE_FILE = "logs/stage4_selected_features_gbpusd.csv"
TRAIN_READY_FILE = "data/stage3_train_ready.parquet"

MODEL_TXT_OUT = "logs/stage4_lgbm_gbpusd.txt"
IMP_CSV_OUT   = "logs/stage4_feature_importance_gbpusd.csv"

MIN_SAMPLES_PER_CLASS = 200  # sanity floor
VALID_FRACTION = 0.2         # last 20% of timeline = validation


def log(msg: str):
    print(msg, flush=True)


def load_features(path: str) -> pd.DataFrame:
    """
    stage4_selected_features_gbpusd.csv was saved with index being window_start_timestamp,
    and 'window_id' column. We want a DatetimeIndex aligned to timestamp_utc so we
    can join with labels.

    We'll interpret the index as UTC timestamps (string -> datetime64[ns, UTC]).
    """
    df = pd.read_csv(path)
    # Expect columns: ['window_id', 'timestamp_utc', feat1, feat2, ...]
    # If timestamp_utc got saved as unnamed index column instead, handle both cases.

    # try common patterns
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    elif "index" in df.columns:
        ts_col = "index"
    elif "window_start" in df.columns:
        ts_col = "window_start"
    else:
        # fallback: assume first column is timestamp
        ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col)

    # drop obvious non-feature helper columns if present
    drop_like = ["window_id"]
    for c in drop_like:
        if c in df.columns:
            df = df.drop(columns=[c])

    # make sure columns are clean per AlphaForge rule #3 (sanitize names)
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+','_', regex=True)
        .str.strip('_')
    )

    return df.sort_index()


def load_labels(path: str) -> pd.DataFrame:
    """
    We load stage3_train_ready:
    - keep ['target_label','target_is_trainable']
    - drop choppy (label==1)
    - map {0:0, 2:1}
    """
    df = pd.read_parquet(path)
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # columns check
    needed_cols = ["target_label", "target_is_trainable"]
    for c in needed_cols:
        if c not in df.columns:
            raise RuntimeError(f"Column {c} missing in stage3_train_ready")

    # filter trainable rows
    df = df[df["target_is_trainable"] == 1].copy()

    # keep only classes 0 and 2
    df = df[df["target_label"].isin([0, 2])].copy()

    # map to binary 0/1
    df["y_binary"] = df["target_label"].map({0: 0, 2: 1}).astype(int)

    # now we only need y_binary
    return df[["y_binary"]].sort_index()


def align_features_labels(feat: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """
    We have:
    - feat indexed by timestamp_utc (window start)
    - lab indexed by timestamp_utc (candle time)
    BUT: labels are per candle; features are per window start (every 250 candles).
      We need to broadcast each feature window forward until next window boundary.

    Strategy:
    - Forward-fill features onto the candle timeline of lab.
    - After ffill, drop rows where we still have NaN (before first window).
    """
    # reindex feature df on label index using forward-fill
    aligned_feat = feat.reindex(lab.index, method="ffill")

    # concat
    merged = pd.concat([aligned_feat, lab], axis=1)

    # drop rows missing any feature OR label
    merged = merged.dropna(subset=["y_binary"])
    merged = merged.dropna(axis=0, how="any")

    return merged


def time_split(df: pd.DataFrame, valid_fraction: float):
    """
    Deterministic temporal split.
    """
    n = len(df)
    n_valid = int(n * valid_fraction)
    n_train = n - n_valid
    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:].copy()
    return train_df, valid_df


def train_lightgbm(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """
    Train LightGBM binary classifier.
    """
    feature_cols = [c for c in train_df.columns if c != "y_binary"]

    X_train = train_df[feature_cols]
    y_train = train_df["y_binary"].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["y_binary"].astype(int)

    # Make LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "verbose": -1,
    }

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(50),
        ],
    )

    # eval
    y_pred_prob = model.predict(X_valid, raw_score=False)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_valid, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_valid, y_pred, labels=[0,1], zero_division=0
    )

    report = classification_report(
        y_valid, y_pred, labels=[0,1],
        target_names=["SHORT(0)", "LONG(1)"],
        zero_division=0,
    )
    cm = confusion_matrix(y_valid, y_pred, labels=[0,1])

    return model, {
        "acc": acc,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "feature_cols": feature_cols,
        "y_valid_dist": y_valid.value_counts(normalize=True).to_dict(),
    }


def save_artifacts(model, feature_cols, metrics):
    Path("logs").mkdir(parents=True, exist_ok=True)

    # model text dump
    model_txt = model.model_to_string()
    with open(MODEL_TXT_OUT, "w") as f:
        f.write(model_txt)

    # feature importance
    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": imp_gain,
        "importance_split": imp_split,
    }).sort_values("importance_gain", ascending=False)

    imp_df.to_csv(IMP_CSV_OUT, index=False)

    return imp_df


def main():
    log("[4.3] 🚀 Stage 4.3 — Model Training (GBPUSD)")

    # 1) load
    feat_raw = load_features(FEATURE_FILE)
    log(f"[4.3] 📥 features loaded: {feat_raw.shape}")
    lab_raw = load_labels(TRAIN_READY_FILE)
    log(f"[4.3] 📥 labels loaded:   {lab_raw.shape}")

    # 2) align
    merged = align_features_labels(feat_raw, lab_raw)
    log(f"[4.3] 🔗 aligned rows after ffill/broadcast: {merged.shape}")

    # sanity class balance after mask
    class_counts = merged["y_binary"].value_counts(normalize=True) * 100
    for cls, pct in class_counts.items():
        log(f"[4.3]    class {cls}: {pct:.2f}%")

    # 3) temporal split
    train_df, valid_df = time_split(merged, VALID_FRACTION)
    log(f"[4.3] 🧪 split -> train {train_df.shape}, valid {valid_df.shape}")

    # check min samples
    bincount = train_df["y_binary"].value_counts()
    if (bincount < MIN_SAMPLES_PER_CLASS).any():
        log("[4.3] ❌ Not enough samples per class in train. This will hurt model quality.")

    # 4) train model
    model, metrics = train_lightgbm(train_df, valid_df)

    # 5) save artifacts
    imp_df = save_artifacts(model, metrics["feature_cols"], metrics)

    # 6) print summary
    log("[4.3] 📊 Validation metrics:")
    log(f"[4.3]    Accuracy: {metrics['acc']:.4f}")
    log("[4.3]    Class distribution in VALID:")
    for cls, pct in metrics["y_valid_dist"].items():
        log(f"[4.3]       class {cls}: {pct*100:.2f}%")

    log("[4.3] 🔎 Classification report:")
    log(metrics["report"])

    log("[4.3] 🔎 Confusion matrix [rows=true, cols=pred]:")
    log(str(metrics["confusion_matrix"]))

    log("[4.3] 💾 Saved:")
    log(f"[4.3]    Model dump  -> {MODEL_TXT_OUT}")
    log(f"[4.3]    Importance -> {IMP_CSV_OUT}")
    log(f"[4.3] ✅ Stage 4.3 completed successfully")


if __name__ == "__main__":
    main()