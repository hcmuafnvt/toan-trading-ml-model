#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding — Stage 4B : Core Alpha Retrain & Comparison
--------------------------------------------------------
Huấn luyện LightGBM multiclass với 72 core features (được Stage 3c chọn ra)
và so sánh kết quả với T4_clean (59 features).

Input:
  - logs/stage2_features.csv
  - logs/stage3_y.csv
  - logs/stage3c_core_features.txt
  - logs/T4_clean_lightgbm.txt (để so sánh)
Output:
  - logs/T5_core_alpha_lightgbm.txt
  - logs/T5_core_alpha_feature_importance.csv
  - logs/T5_core_alpha_summary.txt
"""

import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========== CONFIG ==========
FEATURE_FILE = "logs/stage2_features.csv"
Y_FILE       = "logs/stage3_y.csv"
CORE_TXT     = "logs/stage3c_core_features.txt"
OUT_MODEL    = "logs/T5_core_alpha_lightgbm.txt"
OUT_IMP      = "logs/T5_core_alpha_feature_importance.csv"
OUT_SUMMARY  = "logs/T5_core_alpha_summary.txt"
COMPARE_MODEL= "logs/T4_clean_lightgbm.txt"

SEED, N_JOBS = 202, 28
os.makedirs("logs", exist_ok=True)

# ========== LOAD ==========
print("⏳ Loading data ...")
X = pd.read_csv(FEATURE_FILE)

# --- Load & sanitize target ---
y = pd.read_csv(Y_FILE)["y"].astype(int)

# Map labels to 0–2 if legacy file still has -1,0,1
if y.min() < 0:
    print("⚠️ Detected negative labels — remapping {-1,0,1} → {0,1,2}")
    y = y.map({-1: 0, 0: 1, 1: 2}).astype(int)
    
print("Unique labels:", sorted(y.unique()))

core_feats = [l.strip() for l in open(CORE_TXT) if l.strip() in X.columns]
if not core_feats:
    raise RuntimeError("Không có feature nào match giữa core_features.txt và stage2_features.csv.")

X = X[core_feats]
print(f"✅ Using {len(core_feats)} core features")

# Align
n = min(len(X), len(y))
X, y = X.iloc[:n].reset_index(drop=True), y.iloc[:n].reset_index(drop=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1,
                                                      random_state=SEED, shuffle=True)
print(f"Train={len(X_train):,} | Valid={len(X_valid):,}")

# ========== PARAMS ==========
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
callbacks = [lgb.early_stopping(30), lgb.log_evaluation(50)]

# ========== TRAIN ==========
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

print("🚀 Training T5_core_alpha model ...")
model = lgb.train(params, train_data, num_boost_round=800,
                  valid_sets=[valid_data], callbacks=callbacks)
model.save_model(OUT_MODEL)
print(f"✅ Saved model → {OUT_MODEL}")

# ========== IMPORTANCE ==========
imp = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
imp.to_csv(OUT_IMP, index=False)
print(f"✅ Feature importance saved → {OUT_IMP}")

# ========== VALIDATION ==========
preds_prob = model.predict(X_valid)
preds = np.argmax(preds_prob, axis=1)

acc = accuracy_score(y_valid, preds)
f1  = f1_score(y_valid, preds, average="macro")
report = classification_report(y_valid, preds, digits=4)

print(f"\n✅ Validation ACC={acc:.4f} | F1={f1:.4f}")
print(report)

with open(OUT_SUMMARY, "w") as f:
    f.write(f"Validation ACC={acc:.4f}\nF1={f1:.4f}\n\n")
    f.write(report)

print(f"✅ Summary saved → {OUT_SUMMARY}")

# ========== COMPARE WITH T4_clean ==========
if os.path.exists(COMPARE_MODEL):
    print("\n📊 Comparing with T4_clean model ...")
    old = lgb.Booster(model_file=COMPARE_MODEL)
    old_imp = pd.DataFrame({
        "feature": old.feature_name(),
        "importance": old.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    overlap = len(set(old_imp["feature"]) & set(core_feats))
    print(f"🔹 Feature overlap with T4_clean: {overlap}/{len(core_feats)} ({overlap/len(core_feats)*100:.1f}%)")
else:
    print("\n⚠️  T4_clean model not found → skip comparison.")

print("🎯 Stage 4B complete.")