#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding - Stage 4L: Feature Lab & Selection
Mục tiêu:
- Load logs/stage2_features.csv và các model LightGBM (.txt) từ Stage 3
- Tính Pearson corr (|corr|), Mutual Information, LightGBM feature importance (mean)
- Group-ranking theo prefix (trước "__"), chọn feature có alpha + lọc trùng thông tin
- Xuất: logs/lab_feature_summary.csv, logs/lab_feature_selected.txt (+ quick JSON)

Tuân thủ Project Rules:
1) Data schema: Stage 2 đã chuẩn hóa và lưu features vào logs/stage2_features.csv (đã sanitization tên cột).
2) LightGBM ≥ 4.0 (không dùng early_stopping_rounds cũ).
3) Feature sanitization: đảm bảo tên cột sạch để đồng nhất (dù Stage 2 đã làm).
4) Feature persistence: Stage 4L chỉ đọc từ logs/stage2_features.csv (không re-extract).
5) Performance setup: hướng nhanh/gọn, chỉ numeric, pandas vectorized, seed cố định.
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer

# LightGBM ≥ 4.0
try:
    import lightgbm as lgb  # noqa
except Exception:
    lgb = None
    warnings.warn("LightGBM not available. Still running (importance from models may be skipped).")

# ========== CONFIG ==========
@dataclass
class CONFIG:
    # Stage 2 combined features (per Rule #4)
    FEATURES_PATH: str = "logs/stage2_features.csv"

    # Stage 3 model files (LightGBM text model, trained với tên feature đã sanitized)
    MODEL_PATHS: List[str] = (
        "logs/T1_10x40_lightgbm.txt",
        "logs/T2_15x60_lightgbm.txt",
        "logs/T3_20x80_lightgbm.txt",
    )

    # Outputs (per pipeline)
    OUTDIR: str = "logs"
    SUMMARY_CSV: str = "logs/lab_feature_summary.csv"
    SELECTED_TXT: str = "logs/lab_feature_selected.txt"
    QUICK_JSON: str = "logs/lab_feature_quick.json"

    # Target detection
    TARGET_CANDIDATES: Tuple[str, ...] = ("y", "target", "label", "ret_bin", "class", "cls", "y_cls")

    # Selection & quality thresholds
    DROP_NA_THRESHOLD: float = 0.20   # drop feature if >20% NaN
    TOP_PCT: float = 0.25             # union of top X% by (importance | MI | |corr|)
    REDUN_CORR: float = 0.95          # redundancy removal threshold on selected features

    # LightGBM importance type
    IMPORTANCE_TYPE: str = "gain"     # "gain" or "split"

    # Reproducibility / perf hints
    SEED: int = 202
    N_JOBS: int = 28


CFG = CONFIG()
np.random.seed(CFG.SEED)


# ========== UTILS ==========
def log(msg: str):
    print(f"[Stage4L] {msg}", flush=True)


def ensure_outdir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Feature sanitization (Rule #3)."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def detect_target_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    present = [c for c in candidates if c in df.columns]
    if present:
        return present[0]
    # fallback: common patterns
    for c in df.columns:
        lc = c.lower()
        if lc == "y" or lc.endswith("_y") or lc in ("label", "target", "class"):
            return c
    raise KeyError(f"Cannot detect target column. Tried: {candidates}. Head cols: {list(df.columns)[:20]}...")


def robust_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    # drop constant
    nunique = num.nunique()
    keep = nunique[nunique > 1].index
    return num[keep]


def compute_y_type(y: pd.Series) -> str:
    if y.dtype.kind in "ifu":
        nunique = y.nunique(dropna=True)
        try:
            uniq = pd.Series(y.dropna().unique()).astype(float).astype(int).unique()
        except Exception:
            uniq = []
        if nunique <= 20 and set(uniq).issubset(set(range(-5, 1000))):
            return "classification"
        return "regression"
    return "classification"


def compute_correlations(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce").astype(float)
    corrs = {}
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce").astype(float)
        mask = ~(x.isna() | y_num.isna())
        if mask.sum() < 3:
            corrs[col] = np.nan
            continue
        xv = x[mask]
        yv = y_num[mask]
        if xv.std() == 0 or yv.std() == 0:
            corrs[col] = np.nan
        else:
            corrs[col] = float(np.corrcoef(xv, yv)[0, 1])
    return pd.Series(corrs, name="pearson_corr")


def compute_mutual_info(X: pd.DataFrame, y: pd.Series, ytype: str, seed: int) -> pd.Series:
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    if ytype == "classification":
        y_enc = pd.Series(pd.Categorical(y).codes, index=y.index)
        mi = mutual_info_classif(X_imp, y_enc, random_state=seed, discrete_features=False)
    else:
        y_num = pd.to_numeric(y, errors="coerce").astype(float)
        mi = mutual_info_regression(X_imp, y_num, random_state=seed, discrete_features=False)
    return pd.Series(mi, index=X.columns, name="mutual_info")


def load_models(model_paths: List[str], importance_type: str) -> List[Tuple[List[str], np.ndarray]]:
    models = []
    if lgb is None:
        return models
    for p in model_paths:
        if not os.path.exists(p):
            log(f"WARNING: model file not found: {p}")
            continue
        try:
            booster = lgb.Booster(model_file=p)
            names = booster.feature_name()
            imp = booster.feature_importance(importance_type=importance_type)
            models.append((names, np.asarray(imp, dtype=float)))
            log(f"Loaded model: {p} | features={len(names)}")
        except Exception as e:
            log(f"WARNING: failed to load {p}: {e}")
    return models


def aggregate_importances(models: List[Tuple[List[str], np.ndarray]], all_features: List[str]) -> pd.DataFrame:
    data = {}
    for i, (names, imp) in enumerate(models, start=1):
        m = pd.Series(imp, index=names, dtype=float)
        aligned = m.reindex(all_features).fillna(0.0)
        data[f"lgb_imp_T{i}"] = aligned
    if not data:
        return pd.DataFrame(index=all_features)
    df_imp = pd.DataFrame(data)
    df_imp["lgb_imp_mean"] = df_imp.mean(axis=1)
    return df_imp


def group_from_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[0]
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def top_mask(s: pd.Series, pct: float) -> pd.Series:
    if s.notna().sum() == 0:
        return pd.Series(False, index=s.index)
    thr = s.quantile(1 - pct)
    return s >= thr


def redundancy_filter(selected: List[str], X: pd.DataFrame, corr_threshold: float) -> List[str]:
    if not selected:
        return []
    sub = X[selected].copy()
    sub = sub.fillna(sub.median())
    corr = sub.corr().abs()
    keep, dropped = [], set()
    order = sub.std().sort_values(ascending=False).index.tolist()
    for f in order:
        if f in dropped:
            continue
        keep.append(f)
        high_corr = corr.index[(corr[f] > corr_threshold)].tolist()
        for g in high_corr:
            if g == f:
                continue
            dropped.add(g)
    return [f for f in selected if f in keep]


# ========== MAIN PIPELINE ==========
def run():
    # 0) IO setup
    ensure_outdir(CFG.SUMMARY_CSV)
    ensure_outdir(CFG.SELECTED_TXT)
    ensure_outdir(CFG.QUICK_JSON)

    # --- LOAD FEATURE + TARGET MERGED ---
    log(f"Loading features: {CFG.FEATURES_PATH}")
    X = pd.read_csv(CFG.FEATURES_PATH)
    X = sanitize_columns(X)

    y_path = "logs/stage3_y.csv"
    if os.path.exists(y_path):
        y = pd.read_csv(y_path)["y"]
        log(f"✅ Loaded target vector from {y_path} (len={len(y)})")
    else:
        log("❌ Không tìm thấy logs/stage3_y.csv — cần chạy Stage 1 hoặc 3 trước!")
        raise FileNotFoundError("Missing logs/stage3_y.csv")

    # Cắt theo độ dài nhỏ nhất để tránh lệch
    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].reset_index(drop=True)

    df = pd.concat([X, y.rename("y")], axis=1)
    target_col = "y"    
    log(f"Detected target: {target_col}")

    y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # 2) numeric-only, drop-NaN-heavy
    X_num = robust_numeric_df(X_raw)
    na_frac = X_num.isna().mean()
    X_num = X_num.loc[:, na_frac <= CFG.DROP_NA_THRESHOLD]
    if X_num.shape[1] == 0:
        raise RuntimeError("No numeric features remaining after NaN filtering.")
    log(f"Numeric features after filtering: {X_num.shape[1]}")

    # 3) metrics
    ytype = compute_y_type(y)
    log(f"Detected problem type: {ytype}")

    corr_s = compute_correlations(X_num, y)
    mi_s = compute_mutual_info(X_num, y, ytype, CFG.SEED)

    # 4) LightGBM importances (across T1/T2/T3)
    models = load_models(list(CFG.MODEL_PATHS), CFG.IMPORTANCE_TYPE)
    imp_df = aggregate_importances(models, X_num.columns.tolist())

    # 5) summary table
    out = pd.DataFrame(index=X_num.columns)
    out["feature"] = out.index
    out["pearson_corr"] = corr_s.reindex(out.index)
    out["abs_corr"] = out["pearson_corr"].abs()
    out["mutual_info"] = mi_s.reindex(out.index)

    if not imp_df.empty:
        out = out.join(imp_df, how="left")
    else:
        out["lgb_imp_T1"] = 0.0
        out["lgb_imp_mean"] = 0.0

    # 6) grouping + ranks
    out["group"] = out["feature"].map(group_from_name)
    group_imp = out.groupby("group")["lgb_imp_mean"].mean().sort_values(ascending=False)
    out["group_imp_mean"] = out["group"].map(group_imp.to_dict())
    out["group_rank"] = out["group_imp_mean"].rank(ascending=False, method="dense").astype(int)

    out["rank_imp"] = out["lgb_imp_mean"].rank(ascending=False, method="dense")
    out["rank_mi"] = out["mutual_info"].rank(ascending=False, method="dense")
    out["rank_abs_corr"] = out["abs_corr"].rank(ascending=False, method="dense")

    # 7) selection (union of top pct) + redundancy filter
    m_imp = top_mask(out["lgb_imp_mean"].fillna(0.0), CFG.TOP_PCT)
    m_mi = top_mask(out["mutual_info"].fillna(0.0), CFG.TOP_PCT)
    m_corr = top_mask(out["abs_corr"].fillna(0.0), CFG.TOP_PCT)
    selected = out.index[m_imp | m_mi | m_corr].tolist()

    selected = redundancy_filter(selected, X_num, CFG.REDUN_CORR)

    # 8) save
    out_sorted = out.sort_values(["lgb_imp_mean", "mutual_info", "abs_corr"], ascending=[False, False, False])
    out_sorted.to_csv(CFG.SUMMARY_CSV, index=False)
    with open(CFG.SELECTED_TXT, "w", encoding="utf-8") as f:
        for name in selected:
            f.write(f"{name}\n")

    quick = out_sorted.head(30)[["feature", "lgb_imp_mean", "mutual_info", "abs_corr", "group", "group_rank"]]
    with open(CFG.QUICK_JSON, "w", encoding="utf-8") as f:
        json.dump(json.loads(quick.to_json(orient="records")), f, indent=2)

    log(f"Wrote summary CSV: {CFG.SUMMARY_CSV} (rows={len(out_sorted)})")
    log(f"Wrote selected TXT: {CFG.SELECTED_TXT} (n={len(selected)})")
    log(f"Wrote quick JSON : {CFG.QUICK_JSON}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()