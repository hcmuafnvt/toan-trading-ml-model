# stage2_tsfresh_lgbm.py
# ======================
# LightGBM + tsfresh + (BUY=+1 / TIMEOUT=0 / SELL=-1) cho 3 target:
#   T1: TP=10, AHEAD=40
#   T2: TP=15, AHEAD=60
#   T3: TP=20, AHEAD=80
# Session-aware: +5 pips cho London/NY khi label
#
# Yêu cầu: parquet có index datetime; cột mid_h, mid_l, mid_c (hoặc high/low/close)

import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ========= CONFIG =========
FILE = "data/GBP_USD_M5_2024.parquet"
PAIR = "GBP/USD"
PIP  = 0.0001

# 3 cấu hình target
TARGETS = {
    "T1_10x40": {"tp": 10, "ahead": 40},
    "T2_15x60": {"tp": 15, "ahead": 60},
    "T3_20x80": {"tp": 20, "ahead": 80},
}

SESSION_BONUS = 5     # +5 pips TP/SL cho London/NY
RR = 1.0              # TP:SL = 1:1
# Cửa sổ lookback để tạo feature (nến)
LOOKBACK_N = 240      # 240 nến M5 ≈ 20 giờ; chỉnh nhỏ nếu máy yếu
STRIDE = 5            # lấy mỗi 5 nến 1 mẫu để giảm tải
# tsfresh params nhanh & chất lượng
FC_PARAMS = EfficientFCParameters()

# ========= LOAD & NORMALIZE =========
df = pd.read_parquet(FILE)

# Chuẩn hoá cột
if "close" not in df.columns:
    if "mid_c" in df.columns:
        df["close"] = df["mid_c"]
    elif "bid_c" in df.columns:
        df["close"] = df["bid_c"]

if "high" not in df.columns:
    if "mid_h" in df.columns:
        df["high"] = df["mid_h"]
    elif "bid_h" in df.columns:
        df["high"] = df["bid_h"]

if "low" not in df.columns:
    if "mid_l" in df.columns:
        df["low"] = df["mid_l"]
    elif "bid_l" in df.columns:
        df["low"] = df["bid_l"]

df = df.dropna(subset=["close","high","low"]).copy()
df["hour"] = df.index.hour

def detect_session(h):
    if 7 <= h < 15:
        return "London"
    elif 12 <= h < 21:
        return "NewYork"
    else:
        return "Asia"

df["session"] = df["hour"].map(detect_session)

print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

# ========= LABEL ENGINE (BUY=+1, SELL=-1, TIMEOUT=0) =========
def label_buy_sell(df, tp_pips=10, ahead=20, rr=1.0, pip=0.0001, bonus=0):
    n = len(df)
    res = np.zeros(n, dtype=np.int8)
    highs = df["high"].values
    lows  = df["low"].values
    closes= df["close"].values
    sessions = df["session"].values

    for i in range(n - ahead):
        tp_adj = tp_pips + (bonus if sessions[i] in ["London", "NewYork"] else 0)
        tp = tp_adj * pip
        sl = tp * rr

        entry = closes[i]
        tp_up   = entry + tp
        sl_down = entry - sl
        tp_down = entry - tp
        sl_up   = entry + sl

        sub_high = highs[i+1:i+ahead+1]
        sub_low  = lows[i+1:i+ahead+1]

        hit_tp_up   = np.where(sub_high >= tp_up)[0]
        hit_sl_down = np.where(sub_low  <= sl_down)[0]
        hit_tp_down = np.where(sub_low  <= tp_down)[0]
        hit_sl_up   = np.where(sub_high >= sl_up)[0]

        # First-touch BUY
        if len(hit_tp_up)==0:
            buy_touch = np.inf
        elif len(hit_sl_down)==0:
            buy_touch = hit_tp_up[0]
        else:
            buy_touch = hit_tp_up[0] if hit_tp_up[0] < hit_sl_down[0] else np.inf

        # First-touch SELL
        if len(hit_tp_down)==0:
            sell_touch = np.inf
        elif len(hit_sl_up)==0:
            sell_touch = hit_tp_down[0]
        else:
            sell_touch = hit_tp_down[0] if hit_tp_down[0] < hit_sl_up[0] else np.inf

        if buy_touch == np.inf and sell_touch == np.inf:
            res[i] = 0
        elif buy_touch < sell_touch:
            res[i] = 1
        else:
            res[i] = -1
    return res

# Gán 3 label cột
for name, cfg in TARGETS.items():
    df[name] = label_buy_sell(df, tp_pips=cfg["tp"], ahead=cfg["ahead"],
                              rr=RR, pip=PIP, bonus=SESSION_BONUS)
    print(f"Label {name}: counts", pd.Series(df[name]).value_counts().to_dict())

# ========= MAKE ROLLING WINDOWS FOR TSFRESH =========
# Ta tạo long-format: mỗi id là 1 cửa sổ lookback (close series)
# id = chỉ số mẫu; time = thứ tự trong cửa sổ; value = close
def build_long_from_windows(series: pd.Series, window: int, stride: int):
    vals = series.values
    n = len(vals)
    ids = []
    times = []
    values = []
    sample_idx = []
    t = window
    local_idx = np.arange(window)

    # để đồng bộ label, id tương ứng với index trung tâm (t-1)
    while t < n:
        ids.append(np.full(window, t))     # id = vị trí "hiện tại"
        times.append(local_idx)            # 0..window-1
        values.append(vals[t-window:t])    # chuỗi quá khứ
        sample_idx.append(t)               # index ứng với nhãn ở vị trí t
        t += stride

    data = pd.DataFrame({
        "id": np.concatenate(ids),
        "time": np.concatenate(times),
        "close": np.concatenate(values),
    })
    return data, np.array(sample_idx, dtype=int)

# ========= FEATURE EXTRACTION (tsfresh) =========
def extract_tsfresh_features(series: pd.Series, window: int, stride: int):
    long_df, sample_idx = build_long_from_windows(series, window, stride)
    X = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=FC_PARAMS,
        disable_progressbar=False
    )
    impute(X)  # xử lý NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)  # vệ sinh lần cuối
    return X, sample_idx

print("⏳ Extracting tsfresh features...")
X, sample_idx = extract_tsfresh_features(df["close"], LOOKBACK_N, STRIDE)
print("✅ Features:", X.shape)

# ========= TRAIN / EVAL for each target =========
def train_eval_multiclass(X, y, name: str):
    # y in {-1,0,1} → map to {0: SELL, 1: TIMEOUT, 2: BUY} để LightGBM multiclass
    y_map = pd.Series(y, index=None)
    y_num = y_map.replace({-1:0, 0:1, 1:2}).values

    # One-vs-rest feature selection (union)
    #   - BUY vs others
    #   - SELL vs others
    # Timeout để lại implicit (không select trực tiếp để tránh bias)
    buy_mask = (y_num == 2).astype(int)
    sell_mask = (y_num == 0).astype(int)

    X_buy = select_features(X, buy_mask)
    X_sell = select_features(X, sell_mask)
    cols_union = sorted(set(X_buy.columns).union(set(X_sell.columns)))
    X_sel = X[cols_union].copy()

    print(f"[{name}] Selected features: {len(cols_union)}")

    # Time-aware split: train 80% đầu, test 20% cuối theo sample order
    n = len(X_sel)
    split = int(n * 0.8)
    X_train, X_test = X_sel.iloc[:split], X_sel.iloc[split:]
    y_train, y_test = y_num[:split], y_num[split:]

    # LightGBM multiclass
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n[{name}] Classification report (labels: 0=SELL, 1=TIMEOUT, 2=BUY)")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"[{name}] Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Top features (gain)
    imp = pd.Series(clf.feature_importances_, index=X_sel.columns)
    top20 = imp.sort_values(ascending=False).head(20)
    print(f"\n[{name}] Top-20 features:")
    for k, v in top20.items():
        print(f"  {k} : {int(v)}")

    return {
        "model": clf,
        "X_cols": X_sel.columns.tolist(),
        "top_features": top20
    }

# Chuẩn bị nhãn theo sample_idx (nhãn tại thời điểm t tương ứng id=t)
def get_labels_for_target(colname: str):
    y_all = df[colname].values
    # sample_idx trỏ tới thời điểm t; dùng nhãn tại t (không dùng tương lai)
    return y_all[sample_idx]

results = {}
for name in TARGETS.keys():
    y = get_labels_for_target(name)
    results[name] = train_eval_multiclass(X, y, name)

print("\n✅ DONE. Bạn có thể so sánh báo cáo 3 target (T1,T2,T3) ở trên.")
print("Gợi ý: chọn target có F1 tốt cho lớp 2 (BUY) và lớp 0 (SELL) đồng thời.")
