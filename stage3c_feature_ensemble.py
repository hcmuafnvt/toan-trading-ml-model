#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding — Stage 3c: Feature Importance Ensemble Analysis
-----------------------------------------------------------
Phân tích importance của nhiều model LightGBM (T1/T2/T3/T4_clean)
để xác định các “core alpha features” lặp lại nhiều nhất.

Input:
  logs/T1_10x40_feature_importance.csv
  logs/T2_15x60_feature_importance.csv
  logs/T3_20x80_feature_importance.csv
  logs/T4_clean_feature_importance.csv

Output:
  logs/stage3c_feature_ensemble.csv      (bảng so sánh)
  logs/stage3c_core_features.txt         (feature chung quan trọng)
  logs/stage3c_overlap_heatmap.png       (heatmap visual)
"""

import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

# ========== CONFIG ==========
MODELS = {
    "T1_10x40": "logs/T1_10x40_feature_importance.csv",
    "T2_15x60": "logs/T2_15x60_feature_importance.csv",
    "T3_20x80": "logs/T3_20x80_feature_importance.csv",
    "T4_clean": "logs/T4_clean_feature_importance.csv",
}
OUT_CSV = "logs/stage3c_feature_ensemble.csv"
OUT_TXT = "logs/stage3c_core_features.txt"
OUT_PNG = "logs/stage3c_overlap_heatmap.png"
TOP_K = 100

os.makedirs("logs", exist_ok=True)

# ========== LOAD ==========
dfs = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️  Missing file: {path}")
        continue
    df = pd.read_csv(path)
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError(f"{path} missing columns.")
    df = df.sort_values("importance", ascending=False).head(TOP_K).copy()
    df["rank"] = np.arange(1, len(df) + 1)
    dfs[name] = df
    print(f"✅ Loaded {name} ({len(df)} top features)")

if not dfs:
    raise RuntimeError("No feature importance files found!")

# ========== MERGE & SCORE ==========
all_feats = sorted(set().union(*[df["feature"] for df in dfs.values()]))
ensemble = pd.DataFrame({"feature": all_feats})

for name, df in dfs.items():
    m = df.set_index("feature")["importance"]
    ensemble[name] = ensemble["feature"].map(m)

ensemble["mean_imp"] = ensemble[list(dfs.keys())].mean(axis=1, skipna=True)
ensemble["appear_count"] = ensemble[list(dfs.keys())].notna().sum(axis=1)

ensemble = ensemble.sort_values(
    ["appear_count", "mean_imp"], ascending=[False, False]
).reset_index(drop=True)

ensemble.to_csv(OUT_CSV, index=False)
print(f"✅ Saved ensemble table → {OUT_CSV}")

# ========== CORE FEATURES (EXACT OVERLAP) ==========
core = ensemble.query("appear_count >= 3").copy()
core["mean_imp_rank"] = core["mean_imp"].rank(ascending=False)
core_feats = core["feature"].tolist()

with open(OUT_TXT, "w") as f:
    f.write("\n".join(core_feats))
print(f"✅ Core features (overlap ≥ 3) saved → {OUT_TXT} ({len(core_feats)} features)")

# ========== VISUAL – HEATMAP OVERLAP ==========
print("📊 Generating heatmap of feature overlap ...")
features = sorted(set(core_feats))
model_names = list(dfs.keys())
heatmap = np.zeros((len(model_names), len(model_names)), dtype=int)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            heatmap[i, j] = len(set(dfs[m1]["feature"]))
        else:
            heatmap[i, j] = len(set(dfs[m1]["feature"]) & set(dfs[m2]["feature"]))

plt.figure(figsize=(6, 5))
plt.imshow(heatmap, cmap="YlGn", interpolation="nearest")
plt.xticks(range(len(model_names)), model_names, rotation=45)
plt.yticks(range(len(model_names)), model_names)
plt.title("Feature Overlap Matrix")
for i in range(len(model_names)):
    for j in range(len(model_names)):
        plt.text(j, i, heatmap[i, j], ha="center", va="center", color="black", fontsize=9)
plt.colorbar(label="Feature Overlap Count")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f"✅ Saved overlap heatmap → {OUT_PNG}")

print("\n🎯 Stage 3c complete — ensemble analysis done.")