#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding — Stage 3b: Clean Retrain Model
-----------------------------------------
Huấn luyện lại LightGBM với bộ features đã được Stage 4L chọn lọc
(đã loại bỏ noise, giữ lại các feature có alpha thực sự).

Input:
  - logs/stage2_features.csv       (X)
  - logs/stage3_y.csv              (y)
  - logs/lab_feature_selected.txt  (feature list từ Stage 4L)

Output:
  - logs/T4_clean_lightgbm.txt          (model)
  - logs/T4_clean_feature_importance.csv
  - logs/T4_clean_summary.txt
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ================= CONFIG =================
FEATURE_FILE = "logs/stage2_features.csv"
Y_FILE = "logs/stage3_y.csv"
SELECTED_TXT = "logs/lab_feature_selected.txt"

OUT_MODEL = "logs/T4_clean_lightgbm.txt"
OUT_IMP = "logs/T4_clean_feature_importance.csv"
OUT_SUMMARY = "logs/T4_clean_summary.txt"

SEED = 202
N_JOBS = 28
os.makedirs("logs", exist_ok=True)

# ================= LOAD DATA =================
print("⏳ Loading data ...")
X = pd.read_csv(FEATURE_FILE)
y = pd.read_csv(Y_FILE)["y"]

selected = [
    line.strip()
    for line in open(SELECTED_TXT)
    if line.strip() in X.columns
]
if len(selected) == 0:
    raise RuntimeError("Không có feature nào match giữa lab_feature_selected.txt và stage2_features.csv.")

X = X[selected]
print(f"✅ Using {len(selected)} selected features out of {X.shape[1]}")

# Align lengths
n = min(len(X), len(y))
X, y = X.iloc[:n].reset_index(drop=True), y.iloc[:n].reset_index(drop=True)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=SEED, shuffle=True
)

print(f"Train={len(X_train):,} | Valid={len(X_valid):,}")

# ================= LIGHTGBM PARAMS =================
params = dict(
    
    objective="multiclass",
    metric="multi_logloss",
    num_class=3,
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    max_depth=-1,
    n_jobs=N_JOBS,
    verbose=-1,
)

callbacks = [
    lgb.early_stopping(30),
    lgb.log_evaluation(50)
]

# ================= TRAIN MODEL =================
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

print("🚀 Training LightGBM clean model ...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=800,
    valid_sets=[valid_data],
    callbacks=callbacks
)

# ================= SAVE MODEL =================
model.save_model(OUT_MODEL)
print(f"✅ Saved model → {OUT_MODEL}")

# ================= FEATURE IMPORTANCE =================
imp = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
imp.to_csv(OUT_IMP, index=False)
print(f"✅ Feature importance saved → {OUT_IMP}")

# ================= VALIDATION METRICS =================
# LightGBM multiclass → output shape = [n_samples, 3]
preds_prob = model.predict(X_valid)
preds = np.argmax(preds_prob, axis=1)

# Chuẩn hoá y thành [0,1,2] tương ứng [-1,0,1]
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y_valid_enc = enc.fit_transform(y_valid)

acc = accuracy_score(y_valid_enc, preds)
f1 = f1_score(y_valid_enc, preds, average="macro")
report = classification_report(y_valid_enc, preds, digits=4)

print(f"\n✅ Validation ACC={acc:.4f} | F1={f1:.4f}")
print(report)

with open(OUT_SUMMARY, "w") as f:
    f.write(f"Validation ACC={acc:.4f}\nF1={f1:.4f}\n\n")
    f.write(report)

print(f"✅ Summary saved → {OUT_SUMMARY}")
print("🎯 Stage 3b retrain complete.")