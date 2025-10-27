# ============================================================
# STAGE 2.1 — Train LightGBM models from extracted features
# ------------------------------------------------------------
# Input : logs/stage2_features.csv (from stage2_extract_features_v3.py)
# Output: logs/T*_lightgbm.txt + logs/T*_features.csv
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

# ---------------- CONFIG ----------------
FEATURE_FILE = "logs/stage2_features.csv"
OUT_DIR = "logs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    "T1_10x40": "target_10x40",
    "T2_15x60": "target_15x60",
    "T3_20x80": "target_20x80"
}

# ---------------- LOAD DATA ----------------
print(f"⏳ Loading features from {FEATURE_FILE} ...")
df = pd.read_csv(FEATURE_FILE)
print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} cols")

# Loại bỏ cột target để lấy feature matrix
feature_cols = [c for c in df.columns if not c.startswith("target_")]
X = df[feature_cols]

# ---------------- TRAIN FUNCTION ----------------
def train_model(df, target_col, name):
    print(f"\n========== TRAIN {name} ==========")
    X = df[feature_cols]
    y = df[target_col]

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # LightGBM params
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": 16
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, valid_set],
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # ---------------- EVALUATE ----------------
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class, average="macro")
    print(f"[{name}] Accuracy={acc:.4f} | F1={f1:.4f}")
    print(classification_report(y_test, y_pred_class, digits=3))

    # ---------------- FEATURE IMPORTANCE ----------------
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)

    model_path = f"{OUT_DIR}/{name}_lightgbm.txt"
    imp_path = f"{OUT_DIR}/{name}_features.csv"

    model.save_model(model_path)
    imp.to_csv(imp_path, index=False)
    print(f"✅ Model saved → {model_path}")
    print(f"✅ Feature importance → {imp_path}")

    return model, imp

# ---------------- TRAIN ALL TARGETS ----------------
models = {}
for name, col in TARGETS.items():
    models[name], _ = train_model(df, col, name)

print("\n✅ DONE. All models trained successfully and saved.")