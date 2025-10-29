import lightgbm as lgb, pandas as pd
for name in ["T1_10x40","T2_15x60","T3_20x80"]:
    model = lgb.Booster(model_file=f"logs/{name}_lightgbm.txt")
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    imp.to_csv(f"logs/{name}_feature_importance.csv", index=False)
    print(f"✅ Exported importance → logs/{name}_feature_importance.csv")