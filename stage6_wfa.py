#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import vectorbt as vbt

# ===================== CONFIG =====================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"      # parquet gốc (OANDA mid_*)
FEATURE_FILE = "logs/stage2_features.csv"       # features stage 2 đã lưu

# mapping schema parquet (OANDA mid_*)
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# sampling phải KHỚP Stage 2
WINDOW = 200
STRIDE = 5

# targets đã gán ở stage 2
TARGET_COLS = {
    "T1_10x40": "target_10x40",
    "T2_15x60": "target_15x60",
    "T3_20x80": "target_20x80",
}

# pip & USD & fees
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# backtest fixed stops cho WFA (match Stage 4 baseline tốt nhất)
TP_PIPS = 20.0
SL_PIPS = 20.0

# Walk-Forward: số fold và min train size
N_FOLDS = 6           # chia chuỗi feature theo thời gian thành 6 đoạn bằng nhau
MIN_TRAIN_RATIO = 0.3 # train tối thiểu 30% đầu chuỗi trước khi test fold 1

OUT_FOLDS_CSV = "logs/stage6_wfa_folds.csv"
OUT_SUMMARY   = "logs/stage6_wfa_summary.txt"
os.makedirs("logs", exist_ok=True)


# ===================== HELPERS =====================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def load_price(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # chuẩn hóa OHLC
    have_mids = all(c in df.columns for c in ["mid_o","mid_h","mid_l","mid_c"])
    if have_mids:
        if "close" in df.columns:
            df = df.drop(columns=["close"])
        for k,v in OHLC_MAP.items():
            df[v] = df[k].apply(safe_mean)
        keep = ["open","high","low","close"]
        if "volume" in df.columns:
            keep.append("volume")
        df = df[keep]
    else:
        req = ["open","high","low","close"]
        if not all(c in df.columns for c in req):
            raise ValueError("Parquet thiếu OHLC (mid_* hoặc open/high/low/close).")
        keep = req + (["volume"] if "volume" in df.columns else [])
        df = df[keep]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index phải là DatetimeIndex")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def fuse_weighted(prob_T1, prob_T2, prob_T3, index) -> pd.Series:
    # probs shape: [n,3] theo nhãn 0/1/2 (SELL/TIMEOUT/BUY)
    sum_probs = prob_T1 + prob_T2 + prob_T3  # [n,3]
    sig = sum_probs.argmax(axis=1)           # 0/1/2
    return pd.Series(sig.astype(int), index=index, name="signal")

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0=long, 1=short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def backtest_with_signal(price: pd.DataFrame, signal: pd.Series,
                         tp_pips=TP_PIPS, sl_pips=SL_PIPS):
    tp = pd.Series(tp_pips * PIP_SIZE, index=price.index)
    sl = pd.Series(sl_pips * PIP_SIZE, index=price.index)

    pf = vbt.Portfolio.from_signals(
        price["close"],
        entries=signal.eq(2),
        exits=~signal.eq(2),
        short_entries=signal.eq(0),
        short_exits=~signal.eq(0),
        tp_stop=tp,
        sl_stop=sl,
        size=1.0,
        fees=FEES,
        freq="5min"
    )
    stats = pf.stats()
    tr = pf.trades.records
    exp_pips = expectancy_from_trades(tr)
    return pf, stats, exp_pips

def _get_stat(stats, key):
    v = stats.get(key, np.nan)
    try:
        return float(v)
    except Exception:
        if hasattr(v, "values"):
            return float(np.asarray(v)[0])
        return float("nan")


# ===================== MAIN =====================
if __name__ == "__main__":
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price(DATA_FILE)
    print(f"✅ Loaded price: {len(price):,} rows | {price.index[0]} → {price.index[-1]}")

    print(f"⏳ Loading features: {FEATURE_FILE}")
    feat = pd.read_csv(FEATURE_FILE)
    feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"✅ Loaded features: {feat.shape}")

    # Xác định cột target
    for k, col in TARGET_COLS.items():
        if col not in feat.columns:
            raise ValueError(f"Missing target column in features: {col}")

    # Cố gắng lấy cột thời gian
    time_col = None
    for cand in ["time", "Time", "timestamp", "Timestamp", "index"]:
        if cand in feat.columns:
            time_col = cand
            break

    # Lấy ma trận X và y
    target_cols = list(TARGET_COLS.values())
    X_full = feat.drop(columns=target_cols, errors="ignore").copy()

    # Nếu không có time trong features → xây dựng từ price bằng WINDOW/STRIDE
    if time_col is None:
        sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
        # cắt để khớp số hàng feature (an toàn nếu lệch nhỏ)
        if len(sample_idx) < len(X_full):
            X_full = X_full.iloc[:len(sample_idx)].copy()
        elif len(sample_idx) > len(X_full):
            sample_idx = sample_idx[:len(X_full)]
        sample_times = price.index[sample_idx]
        # Gán time cho tiện split theo thời gian
        feat_times = pd.Series(sample_times, name="time")
    else:
        feat_times = pd.to_datetime(feat[time_col]).tz_localize(None)
        # đảm bảo cùng độ dài
        if len(feat_times) != len(X_full):
            # fallback: dùng lại WINDOW/STRIDE
            sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
            sample_idx = sample_idx[:len(X_full)]
            feat_times = pd.Series(price.index[sample_idx], name="time")

    # Bóc label từng target theo đúng chiều của X_full
    Y = {}
    for name, tcol in TARGET_COLS.items():
        Y[name] = feat.loc[X_full.index, tcol].astype(int).values

    # Xóa cột time ra khỏi X nếu nó hiện diện
    for c in ["time", "Time", "timestamp", "Timestamp", "index"]:
        if c in X_full.columns:
            X_full = X_full.drop(columns=[c])

    # Chuẩn bị folds theo thời gian (expanding window)
    n = len(X_full)
    min_train = int(max(MIN_TRAIN_RATIO * n, 500))  # ít nhất 500 mẫu để train ổn định
    fold_edges = np.linspace(min_train, n, N_FOLDS + 1, dtype=int)[1:]  # điểm kết thúc mỗi fold test

    rows = []
    fold_signals = []  # giữ signal theo index thời gian để gộp lại backtest tổng

    print("\n========== STAGE 6: Walk-Forward Validation ==========")
    print(f"Total samples: {n} | min_train={min_train} | folds={N_FOLDS}")

    for i, test_end in enumerate(fold_edges, start=1):
        # Chia test block đều
        test_start = fold_edges[i-2] if i > 1 else min_train
        # Train trên [0 : test_start), Test trên [test_start : test_end)
        tr_idx = np.arange(0, test_start)
        te_idx = np.arange(test_start, test_end)

        if len(te_idx) == 0 or len(tr_idx) < 100:
            continue

        X_tr = X_full.iloc[tr_idx].copy()
        X_te = X_full.iloc[te_idx].copy()
        time_te = pd.DatetimeIndex(feat_times.iloc[te_idx])

        # Train 3 targets
        probs_fold = []
        for tgt_name, tgt_col in TARGET_COLS.items():
            y_tr = Y[tgt_name][tr_idx]
            # class_weight balanced để kéo recall BUY/SELL
            clf = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                class_weight="balanced"
            )
            # Dùng 10% cuối train làm val theo thời gian
            val_cut = int(len(X_tr) * 0.9)
            X_tr_in, X_val = X_tr.iloc[:val_cut], X_tr.iloc[val_cut:]
            y_tr_in, y_val = y_tr[:val_cut], y_tr[val_cut:]

            clf.fit(
                X_tr_in, y_tr_in,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                verbose=False
            )

            # predict xác suất trên test
            prob = clf.predict_proba(X_te)  # shape [m, 3] theo (0,1,2)
            probs_fold.append(prob)

        # Fusion weighted
        prob_T1, prob_T2, prob_T3 = probs_fold
        sig_te = fuse_weighted(prob_T1, prob_T2, prob_T3, index=time_te)

        # Expand về full price index bằng ffill để backtest liên tục
        sig_full = sig_te.reindex(price.index, method="ffill").fillna(1).astype(int)

        # Backtest chỉ trên cửa sổ test-time của fold (công bằng)
        # Dải thời gian test thật sự
        start_t, end_t = time_te.min(), time_te.max()
        price_mask = (price.index >= start_t) & (price.index <= end_t)

        pf, stats, exp_pips = backtest_with_signal(price.loc[price_mask], sig_full.loc[price_mask],
                                                   tp_pips=TP_PIPS, sl_pips=SL_PIPS)

        fold_row = {
            "Fold": i,
            "Train_Samples": int(len(tr_idx)),
            "Test_Samples": int(len(te_idx)),
            "Start": str(start_t),
            "End": str(end_t),
            "Total Trades": int(_get_stat(stats, "Total Trades")),
            "Win Rate [%]": _get_stat(stats, "Win Rate [%]"),
            "Profit Factor": _get_stat(stats, "Profit Factor"),
            "Max Drawdown [%]": _get_stat(stats, "Max Drawdown [%]"),
            "Expectancy (pips)": float(exp_pips),
            "Expectancy (USD_1lot)": float(exp_pips * PIP_USD),
        }
        rows.append(fold_row)

        # Lưu signal test (để ghép backtest tổng sau)
        fold_signals.append(sig_te)

        print(
          f"[Fold {i}] "
          f"Trades={fold_row['Total Trades']} | "
          f"Win%={fold_row['Win Rate [%]']:.2f} | "
          f"PF={fold_row['Profit Factor']:.2f} | "
          f"Exp={fold_row['Expectancy (pips)']:.2f}p (${fold_row['Expectancy (USD_1lot)']:.2f}) | "
          f"{fold_row['Start']} → {fold_row['End']}"
        )

    # Lưu per-fold
    folds_df = pd.DataFrame(rows)
    folds_df.to_csv(OUT_FOLDS_CSV, index=False)

    # Backtest tổng hợp: ghép các signal test lại theo thời gian
    if len(fold_signals) > 0:
        sig_all = pd.concat(fold_signals, axis=0).sort_index()
        sig_all = sig_all.reindex(price.index, method="ffill").fillna(1).astype(int)

        pf_all, stats_all, exp_all = backtest_with_signal(price, sig_all,
                                                          tp_pips=TP_PIPS, sl_pips=SL_PIPS)

        with open(OUT_SUMMARY, "w") as f:
            f.write("========== STAGE 6: WALK-FORWARD VALIDATION ==========\n")
            f.write(f"Feature rows: {len(X_full)} | Price bars: {len(price)}\n")
            f.write(f"Folds: {len(rows)} | TP/SL fixed: {TP_PIPS}/{SL_PIPS} pips\n\n")

            f.write("---- Per-Fold Results ----\n")
            f.write(folds_df.to_string(index=False))
            f.write("\n\n---- Overall Backtest (concatenated test signals) ----\n")
            f.write(stats_all.to_string())
            f.write(f"\nExpectancy (pips): {exp_all:.4f} | Expectancy USD: {exp_all*PIP_USD:.2f}\n")

        print("\n========== OVERALL (concatenated test) ==========")
        print(f"Total Trades        : {int(_get_stat(stats_all,'Total Trades'))}")
        print(f"Win Rate [%]        : {_get_stat(stats_all,'Win Rate [%]'):.2f}")
        print(f"Profit Factor       : {_get_stat(stats_all,'Profit Factor'):.2f}")
        print(f"Max Drawdown [%]    : {_get_stat(stats_all,'Max Drawdown [%]'):.2f}")
        print(f"Expectancy (pips)   : {exp_all:.2f}")
        print(f"Expectancy (USD)    : ${exp_all*PIP_USD:.2f} /trade (1 lot)")

    print(f"\n✅ Saved folds → {OUT_FOLDS_CSV}")
    print(f"✅ Saved summary → {OUT_SUMMARY}")