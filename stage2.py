# ============================================================
# STAGE 2 — TRAIN 3 LIGHTGBM MODELS (T1/T2/T3)
# ------------------------------------------------------------
# Input : logs/stage2_features.csv (re-extracted features)
# Output: LightGBM models (T1, T2, T3)
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# ---------- CONFIG ----------
FEATURES_CSV = "logs/stage2_features.csv"
OUT_DIR = "logs"
os.makedirs(OUT_DIR, exist_ok=True)

# Mô hình và target config (TP_pips, N_ahead)
TARGETS = {
    "T1_10x40": {"tp_pips": 10, "ahead": 40},
    "T2_15x60": {"tp_pips": 15, "ahead": 60},
    "T3_20x80": {"tp_pips": 20, "ahead": 80},
}

# ---------- LOAD FEATURES ----------
print(f"⏳ Loading features from {FEATURES_CSV} ...")
df = pd.read_csv(FEATURES_CSV, index_col=0)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"✅ Loaded features: {df.shape}")

# ---------- TRAINING FUNCTION ----------
def train_model(df, name, params=None):
    print(f"\n========== TRAIN {name} ==========")
    
    # Lấy target (chúng ta tạm dùng target chung)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"], errors="ignore")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if params is None:
        params = dict(
            objective="multiclass",
            num_class=3,
            boosting_type="gbdt",
            metric="multi_logloss",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=-1,
            num_leaves=80,
            min_data_in_leaf=50,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=16
        )
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric="multi_logloss",
              verbose=False)
    
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    
    print(report)
    
    # Save model
    model_path = os.path.join(OUT_DIR, f"{name}_lightgbm.txt")
    model.booster_.save_model(model_path)
    print(f"✅ Model saved → {model_path}")
    
    # Feature importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.booster_.feature_importance()
    }).sort_values("importance", ascending=False)
    
    imp_path = os.path.join(OUT_DIR, f"{name}_features.csv")
    imp.to_csv(imp_path, index=False)
    print(f"✅ Feature importance saved → {imp_path}")
    
    return model, imp

# ---------- TRAIN LOOP ----------
models = {}
for name, cfg in TARGETS.items():
    model, imp = train_model(df.copy(), name)
    models[name] = model

print("\n✅ DONE. 3 models trained and saved:")
for name in models.keys():
    print(f" - {name}_lightgbm.txt")