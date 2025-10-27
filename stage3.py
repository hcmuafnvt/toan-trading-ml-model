"""
STAGE 4 — Fusion Optimizer (FINAL)
----------------------------------
✅ 3 fusion modes: 'majority', 'weighted', 'buy_only'
✅ TP/SL: 'fixed', 'atr', 'vol'
✅ Expectancy per trade + P/L USD với 1 lot = 100,000 (GBPUSD: 1 pip = 10 USD)
✅ Đọc data từ data/, model + features từ logs/
✅ Lưu kết quả grid vào logs/stage4_results.csv
"""

import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
DATA_FILE = "data/GBP_USD_M5_2024.parquet"
LOG_DIR   = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Feature extraction (phải khớp Stage 2/3)
LOOKBACK_N = 240
STRIDE     = 10
N_JOBS     = 28
FC_PARAMS  = EfficientFCParameters()

# 3 models + feature lists tương ứng (đã tạo ở Stage 2)
MODELS = {
    "T1_10x40": os.path.join(LOG_DIR, "T1_10x40_lightgbm.txt"),
    "T2_15x60": os.path.join(LOG_DIR, "T2_15x60_lightgbm.txt"),
    "T3_20x80": os.path.join(LOG_DIR, "T3_20x80_lightgbm.txt"),
}

# Pip & Lot
PIP_SIZE   = 0.0001
PIP_USD    = 10.0  # GBPUSD, 1 pip = $10 cho 1 lot = 100,000

# Grid tham số để thử
FUSION_MODES = ["majority", "weighted", "buy_only"]
STOP_MODES   = [
    {"name":"fixed", "tp_pips":10, "sl_pips":10},
    {"name":"atr",   "mult_tp":1.5, "mult_sl":1.5, "atr_window":14},
    {"name":"vol",   "mult_tp":2.0, "mult_sl":2.0, "vol_window":120}
]
WEIGHTED_THRESH = 0.5   # ngưỡng vote soft: sum(pBUY - pSELL) / n_models > 0.5 → BUY, < -0.5 → SELL
FEES = 0.0002           # phí giả định (theo tỉ lệ giá) cho vectorbt

# ===================== UTILS =====================
def clean_cols(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c) for c in df.columns]
    return df

def build_long(series, window, stride):
    vals, n = series.values, len(series)
    ids, times, values, idxs = [], [], [], []
    local_idx = np.arange(window)
    t = window
    while t < n:
        ids.append(np.full(window, t))
        times.append(local_idx)
        values.append(vals[t-window:t])
        idxs.append(t)
        t += stride
    long_df = pd.DataFrame({
        "id":   np.concatenate(ids),
        "time": np.concatenate(times),
        "close":np.concatenate(values).astype("float32"),
    })
    return long_df, np.array(idxs, dtype=int)

def ensure_ohlc(df):
    # Chuẩn hoá cột close/high/low
    for base in ["mid_", "bid_", "ask_"]:
        if "close" not in df and f"{base}c" in df.columns:
            df["close"] = df[f"{base}c"]
        if "high" not in df and f"{base}h" in df.columns:
            df["high"] = df[f"{base}h"]
        if "low" not in df and f"{base}l" in df.columns:
            df["low"] = df[f"{base}l"]
    if not {"close","high","low"}.issubset(df.columns):
        raise ValueError("Data cần có cột close/high/low. Hãy map từ mid_*/bid_*/ask_* nếu cần.")
    return df

def extract_tsfresh_features(close, lookback, stride, n_jobs):
    long_df, sample_idx = build_long(close, lookback, stride)
    X = extract_features(
        long_df,
        column_id="id", column_sort="time",
        default_fc_parameters=FC_PARAMS,
        n_jobs=n_jobs, disable_progressbar=False
    )
    impute(X)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X = clean_cols(X)
    return X, sample_idx

def load_models_and_predict(X):
    """Trả về pred hard-class (0/1/2) và soft probs cho từng model"""
    preds_hard = {}
    probs_soft = {}
    for name, mpath in MODELS.items():
        booster = lgb.Booster(model_file=mpath)
        feat_file = os.path.join(LOG_DIR, f"{name}_features.csv")
        feat_names = pd.read_csv(feat_file, header=None)[0].tolist()
        common = [c for c in feat_names if c in X.columns]
        X_pred = X[common].copy()
        probs = booster.predict(X_pred, predict_disable_shape_check=True)  # shape (n,3)
        probs_soft[name] = probs
        hard = np.argmax(probs, axis=1)  # 0=SELL, 1=TIMEOUT, 2=BUY
        preds_hard[name] = hard
        print(f"✅ {name} predict ok | features used: {len(common)}")
    return preds_hard, probs_soft

def fusion_signal(preds_hard, probs_soft, mode="majority", weighted_thresh=WEIGHTED_THRESH):
    """Trả về Series signal ∈ {-1,0,1} (SELL, FLAT, BUY) theo fusion mode"""
    model_names = list(preds_hard.keys())
    n = len(next(iter(preds_hard.values())))
    sig = np.zeros(n, dtype=np.int8)

    if mode == "majority":
        mat = np.column_stack([preds_hard[m] for m in model_names])
        vote_buy  = (mat == 2).sum(axis=1)
        vote_sell = (mat == 0).sum(axis=1)
        sig = np.where(vote_buy >= 2, 1, np.where(vote_sell >= 2, -1, 0))

    elif mode == "weighted":
        # score = avg(pBUY - pSELL)
        buy_scores  = []
        sell_scores = []
        for m in model_names:
            # probs_soft[m][:,2] = pBUY ; probs_soft[m][:,0] = pSELL
            buy_scores.append(probs_soft[m][:,2])
            sell_scores.append(probs_soft[m][:,0])
        buy_scores  = np.column_stack(buy_scores).mean(axis=1)
        sell_scores = np.column_stack(sell_scores).mean(axis=1)
        score = buy_scores - sell_scores  # range [-1,1]
        sig = np.where(score >  weighted_thresh,  1,
              np.where(score < -weighted_thresh, -1, 0))

    elif mode == "buy_only":
        # chỉ vào long khi >=2 model vote BUY, còn lại = 0 (không short)
        mat = np.column_stack([preds_hard[m] for m in model_names])
        vote_buy = (mat == 2).sum(axis=1)
        sig = np.where(vote_buy >= 2, 1, 0)

    else:
        raise ValueError("fusion mode không hợp lệ")

    return pd.Series(sig)

def calc_atr(df, window=14):
    # True Range (TR) cần high/low/close
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

def calc_vol(close, window=120):
    # Độ biến động theo std của returns (close-to-close)
    ret = close.pct_change()
    vol = ret.rolling(window, min_periods=window//2).std().fillna(method="bfill")
    # đổi vol (tỷ lệ) về "giá-tương-đương" bằng cách nhân close
    return (vol * close).abs()

def stops_by_mode(df_bt, mode_cfg):
    name = mode_cfg["name"]
    if name == "fixed":
        tp = np.full(len(df_bt), mode_cfg["tp_pips"] * PIP_SIZE)
        sl = np.full(len(df_bt), mode_cfg["sl_pips"] * PIP_SIZE)
        return pd.Series(tp, index=df_bt.index), pd.Series(sl, index=df_bt.index)

    elif name == "atr":
        atr = calc_atr(df_bt, window=mode_cfg.get("atr_window",14))
        tp = atr * mode_cfg.get("mult_tp",1.5)
        sl = atr * mode_cfg.get("mult_sl",1.5)
        return tp, sl

    elif name == "vol":
        vol = calc_vol(df_bt["close"], window=mode_cfg.get("vol_window",120))
        tp = vol * mode_cfg.get("mult_tp",2.0)
        sl = vol * mode_cfg.get("mult_sl",2.0)
        return tp, sl

    else:
        raise ValueError("stop mode không hợp lệ")

def backtest_and_metrics(df_bt, entries, exits, tp_series, sl_series, fees=FEES):
    pf = vbt.Portfolio.from_signals(
        df_bt["close"],
        entries=entries,
        exits=exits,
        tp_stop=tp_series,
        sl_stop=sl_series,
        size=1.0,            # notional placeholder (không dùng lot ở đây)
        fees=fees,
        direction="both"
    )
    stats = pf.stats()
    # Expectancy thủ công dựa trên pips và USD (1 lot)
    # Lấy trades
    trades = pf.trades.records_readable
    if len(trades) == 0:
        expectancy_pips = 0.0
        expectancy_usd  = 0.0
    else:
        # pips = (Exit - Entry) * sign * 10000
        # sign = +1 nếu long, -1 nếu short
        sign = np.where(trades["Direction"].str.lower().str.contains("long"), 1, -1)
        pips = (trades["Exit Price"].values - trades["Entry Price"].values) * sign * (1.0 / PIP_SIZE)
        # expectancy = mean(pips)
        expectancy_pips = float(np.nanmean(pips))
        expectancy_usd  = expectancy_pips * PIP_USD  # 1 lot

    # Đổi Total Return [%] sang USD với 1 lot thì cần quy ước thêm vốn ban đầu.
    # Ở đây ta vẫn báo cáo PF / Win Rate / Return % từ vectorbt
    return pf, stats, expectancy_pips, expectancy_usd

def run_one_config(df, X, sample_idx, fusion_mode, stop_cfg):
    # 1) Predict mỗi model
    preds_hard, probs_soft = load_models_and_predict(X)
    # 2) Fusion tín hiệu
    sig = fusion_signal(preds_hard, probs_soft, mode=fusion_mode)
    # 3) Mapping về chuỗi thời gian gốc
    df_bt = df.iloc[sample_idx].copy()
    df_bt["signal"] = sig.values

    # 4) Entries / exits
    entries = (df_bt["signal"] == 1)
    exits   = (df_bt["signal"] == -1)

    # 5) TP/SL theo chế độ
    tp_series, sl_series = stops_by_mode(df_bt, stop_cfg)

    # 6) Backtest + metrics
    pf, stats, exp_pips, exp_usd = backtest_and_metrics(df_bt, entries, exits, tp_series, sl_series, fees=FEES)

    # Chuẩn hoá metric output
    def get_stat(key, default=np.nan):
        return float(stats[key]) if key in stats.index else default

    out = {
        "fusion_mode": fusion_mode,
        "stop_mode": stop_cfg["name"],
        "tp_param": stop_cfg.get("tp_pips", stop_cfg.get("mult_tp", np.nan)),
        "sl_param": stop_cfg.get("sl_pips", stop_cfg.get("mult_sl", np.nan)),
        "Total Trades": get_stat("Total Trades", 0.0),
        "Win Rate [%]": get_stat("Win Rate [%]"),
        "Total Return [%]": get_stat("Total Return [%]"),
        "Profit Factor": get_stat("Profit Factor"),
        "Max Drawdown [%]": get_stat("Max Drawdown [%]"),
        "Expectancy (pips)": exp_pips,
        "Expectancy (USD_1lot)": exp_usd
    }
    return out

# ===================== MAIN =====================
if __name__ == "__main__":
    # 0) Load data
    df = pd.read_parquet(DATA_FILE)
    df = ensure_ohlc(df).copy()
    print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

    # 1) Extract features (giống Stage 3)
    print("⏳ Extracting tsfresh features...")
    X, sample_idx = extract_tsfresh_features(df["close"], LOOKBACK_N, STRIDE, N_JOBS)
    print(f"✅ Features ready: {X.shape}")

    # 2) Chạy grid fusion × stop
    results = []
    for fmode in FUSION_MODES:
        for scfg in STOP_MODES:
            print(f"\n>>> Running fusion='{fmode}' | stop='{scfg['name']}' ...")
            res = run_one_config(df, X, sample_idx, fmode, scfg)
            results.append(res)
            # in tóm tắt
            print(
                f"Trades={res['Total Trades']:.0f} | Win%={res['Win Rate [%]']:.2f} | "
                f"PF={res['Profit Factor']:.2f} | Ret%={res['Total Return [%]']:.2f} | "
                f"DD%={res['Max Drawdown [%]']:.2f} | Exp={res['Expectancy (pips)']:.2f} pips "
                f"({res['Expectancy (USD_1lot)']:.2f} USD)"
            )

    # 3) Save tổng hợp
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(LOG_DIR, "stage4_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved grid results → {out_csv}")

    # In top theo PF và theo Return
    def top_print(df_, key, k=5):
        print(f"\nTop {k} by {key}:")
        print(df_.sort_values(key, ascending=False).head(k).to_string(index=False))

    top_print(res_df, "Profit Factor", k=5)
    top_print(res_df, "Total Return [%]", k=5)
    top_print(res_df, "Win Rate [%]", k=5)
    top_print(res_df, "Expectancy (USD_1lot)", k=5)