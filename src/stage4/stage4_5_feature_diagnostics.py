#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4.5 — Feature Diagnostics (GBPUSD)

Phân tích đóng góp alpha của từng nhóm feature.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score
from pathlib import Path

REFINED_FEATURES = "logs/stage4_refined_features_gbpusd.csv"
LABEL_FILE       = "data/stage3_train_ready.parquet"
OUT_REPORT       = "logs/stage4_feature_diagnostics_gbpusd.csv"

def log(msg):
    now = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S UTC]")
    print(f"{now} {msg}", flush=True)

def quick_auc(X, y):
    split = int(len(X) * 0.8)
    dtrain = lgb.Dataset(X.iloc[:split], label=y.iloc[:split])
    dvalid = lgb.Dataset(X.iloc[split:], label=y.iloc[split:])
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=31, feature_fraction=0.8, seed=42)
    model = lgb.train(params, dtrain, valid_sets=[dvalid],
                      num_boost_round=300,
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    preds = model.predict(X.iloc[split:])
    auc = roc_auc_score(y.iloc[split:], preds)
    imp = pd.Series(model.feature_importance(), index=X.columns).sort_values(ascending=False)
    return auc, imp

def main():
    log("🚀 Stage 4.5 — Feature Diagnostics (GBPUSD)")

    df = pd.read_csv(REFINED_FEATURES, index_col=0, parse_dates=True)
    labels = pd.read_parquet(LABEL_FILE)
    labels.index = pd.to_datetime(labels.index, utc=True)
    labels = labels[labels["target_is_trainable"] == 1]
    y = labels.loc[df.index, "target_label"].fillna(method="ffill").astype(int)

    log(f"📊 Loaded refined features: {df.shape}")
    log(f"📊 Matching labels: {y.shape}")

    # --- Group features ---
    groups = {
        "directional_meta": [c for c in df.columns if c.startswith("feat_")],
        "tsfresh_core":     [c for c in df.columns if c.startswith("value__")],
        "other":            [c for c in df.columns
                             if not (c.startswith("feat_") or c.startswith("value__"))],
    }

    results = []
    for name, cols in groups.items():
        if len(cols) < 5:
            continue
        log(f"🧪 Testing group: {name} ({len(cols)} features)")
        X = df[cols].copy()
        auc, imp = quick_auc(X, y)
        top_feats = ", ".join(imp.head(5).index)
        results.append([name, len(cols), round(auc, 4), top_feats])
        log(f"   → AUC={auc:.4f} | top: {top_feats}")

    out = pd.DataFrame(results,
                       columns=["group", "n_features", "auc", "top5_features"])
    Path("logs").mkdir(exist_ok=True)
    out.to_csv(OUT_REPORT, index=False)
    log(f"💾 Saved diagnostic report → {OUT_REPORT}")
    log("✅ Stage 4.5 completed successfully")

if __name__ == "__main__":
    main()