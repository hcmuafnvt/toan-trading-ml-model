"""
STAGE 3.1 — Analyzer (stability & attribution)
----------------------------------------------
So sánh 3 cấu hình mạnh nhất:
  1) weighted + vol
  2) weighted + atr
  3) majority + vol

✅ Text-only
✅ No fees
✅ Recompute signals & backtest
✅ Breakdown: BUY/SELL, Session (Asia/London/NY), Monthly (YYYY-MM)
"""

import os, re, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore")

# ------------------- CONFIG -------------------
DATA_FILE = "data/GBP_USD_M5_2024.parquet"
LOG_DIR   = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Feature extraction (khớp Stage 2/3/4)
LOOKBACK_N = 240
STRIDE     = 10
N_JOBS     = 28
FC_PARAMS  = EfficientFCParameters()

# Pip & Lot (GBPUSD)
PIP_SIZE = 0.0001
PIP_USD  = 10.0  # 1 pip = $10 với 1 lot = 100,000

# 3 mô hình đã train (Stage 2)
MODELS = {
    "T1_10x40": os.path.join(LOG_DIR, "T1_10x40_lightgbm.txt"),
    "T2_15x60": os.path.join(LOG_DIR, "T2_15x60_lightgbm.txt"),
    "T3_20x80": os.path.join(LOG_DIR, "T3_20x80_lightgbm.txt"),
}

# 3 cấu hình cần phân tích sâu
CONFIGS = [
    {"fusion":"weighted", "stop":"vol", "params":{"mult_tp":2.0, "mult_sl":2.0, "vol_window":120}},
    {"fusion":"weighted", "stop":"atr", "params":{"mult_tp":1.5, "mult_sl":1.5, "atr_window":14}},
    {"fusion":"majority", "stop":"vol", "params":{"mult_tp":2.0, "mult_sl":2.0, "vol_window":120}},
]

# ------------------- HELPERS -------------------
def clean_cols(df):
    df = df.copy()
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c) for c in df.columns]
    return df

def ensure_ohlc(df):
    for base in ["mid_", "bid_", "ask_"]:
        if "close" not in df and f"{base}c" in df.columns:
            df["close"] = df[f"{base}c"]
        if "high" not in df and f"{base}h" in df.columns:
            df["high"] = df[f"{base}h"]
        if "low" not in df and f"{base}l" in df.columns:
            df["low"] = df[f"{base}l"]
    if not {"close","high","low"}.issubset(df.columns):
        raise ValueError("Data cần cột close/high/low.")
    return df

def detect_session_from_hour(h):
    if 7 <= h < 15: return "London"
    elif 12 <= h < 21: return "NewYork"
    else: return "Asia"

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

def extract_tsfresh_features(close, lookback, stride, n_jobs):
    long_df, sample_idx = build_long(close, lookback, stride)
    X = extract_features(
        long_df, column_id="id", column_sort="time",
        default_fc_parameters=FC_PARAMS,
        n_jobs=n_jobs, disable_progressbar=False
    )
    impute(X)
    X = X.replace([np.inf,-np.inf], np.nan).dropna(axis=1)
    X = clean_cols(X)
    return X, sample_idx

def load_models_and_predict(X):
    preds_hard, probs_soft = {}, {}
    for name, mpath in MODELS.items():
        booster = lgb.Booster(model_file=mpath)
        feat_file = os.path.join(LOG_DIR, f"{name}_features.csv")
        feat_names = pd.read_csv(feat_file, header=None)[0].tolist()
        common = [c for c in feat_names if c in X.columns]
        probs = booster.predict(X[common], predict_disable_shape_check=True)
        probs_soft[name] = probs
        preds_hard[name] = np.argmax(probs, axis=1)  # 0=SELL,1=TIMEOUT,2=BUY
        print(f"  - {name}: used {len(common)} features")
    return preds_hard, probs_soft

def fusion_signal(preds_hard, probs_soft, mode="weighted", weighted_thresh=0.5):
    model_names = list(preds_hard.keys())
    n = len(next(iter(preds_hard.values())))
    sig = np.zeros(n, dtype=np.int8)

    if mode == "majority":
        mat = np.column_stack([preds_hard[m] for m in model_names])
        vote_buy  = (mat == 2).sum(axis=1)
        vote_sell = (mat == 0).sum(axis=1)
        sig = np.where(vote_buy >= 2, 2, np.where(vote_sell >= 2, 0, 1))  # 2=BUY,0=SELL,1=TIMEOUT

    elif mode == "weighted":
        buy_scores, sell_scores = [], []
        for m in model_names:
            buy_scores.append(probs_soft[m][:,2])
            sell_scores.append(probs_soft[m][:,0])
        buy_scores  = np.column_stack(buy_scores).mean(axis=1)
        sell_scores = np.column_stack(sell_scores).mean(axis=1)
        score = buy_scores - sell_scores
        sig = np.where(score >  weighted_thresh, 2,
              np.where(score < -weighted_thresh, 0, 1))

    else:
        raise ValueError("fusion mode không hợp lệ")
    return pd.Series(sig)

def calc_atr(df, window=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

def calc_vol(close, window=120):
    ret = close.pct_change()
    vol = ret.rolling(window, min_periods=window//2).std().fillna(method="bfill")
    return (vol * close).abs()

def stops_by_mode(df_bt, stop_name, params):
    if stop_name == "vol":
        vol = calc_vol(df_bt["close"], window=params.get("vol_window",120))
        tp = vol * params.get("mult_tp",2.0)
        sl = vol * params.get("mult_sl",2.0)
        return tp, sl
    if stop_name == "atr":
        atr = calc_atr(df_bt, window=params.get("atr_window",14))
        tp = atr * params.get("mult_tp",1.5)
        sl = atr * params.get("mult_sl",1.5)
        return tp, sl
    if stop_name == "fixed":
        tp = np.full(len(df_bt), params.get("tp_pips",10) * PIP_SIZE)
        sl = np.full(len(df_bt), params.get("sl_pips",10) * PIP_SIZE)
        return pd.Series(tp, index=df_bt.index), pd.Series(sl, index=df_bt.index)
    raise ValueError("stop mode không hợp lệ")

def backtest(df_bt, entries, exits, tp_series, sl_series, fees=0.0):
    pf = vbt.Portfolio.from_signals(
        df_bt["close"],
        entries=entries,
        exits=exits,
        tp_stop=tp_series,
        sl_stop=sl_series,
        size=1.0,
        fees=fees,
        direction="both"
    )
    stats = pf.stats()
    trades = pf.trades.records
    return pf, stats, trades

def pf_from_trades(trades):
    if trades is None or len(trades)==0:
        return {"trades":0, "win_rate":0.0, "pf":0.0, "exp_pips":0.0, "exp_usd":0.0}
    entry_price = trades["entry_price"].values
    exit_price  = trades["exit_price"].values
    direction   = trades["direction"].values  # 0=long,1=short
    sign = np.where(direction==0, 1, -1)
    pips = (exit_price - entry_price) * sign * (1.0 / PIP_SIZE)
    pnl  = (exit_price - entry_price) * sign  # tính bằng "giá"
    wins = pnl > 0
    sum_win = pnl[wins].sum()
    sum_lose = pnl[~wins].sum()
    pf = (sum_win / abs(sum_lose)) if sum_lose < 0 else np.inf
    exp_pips = float(np.nanmean(pips)) if len(pips)>0 else 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()*100.0),
        "pf": float(pf),
        "exp_pips": exp_pips,
        "exp_usd": exp_pips * PIP_USD
    }

def attach_session(df_bt):
    # thêm session theo giờ
    hours = df_bt.index.hour
    sess = [detect_session_from_hour(h) for h in hours]
    df_bt = df_bt.copy()
    df_bt["session"] = sess
    return df_bt

def by_session(trades, df_bt):
    # map entry_idx -> timestamp -> session
    if trades is None or len(trades)==0:
        return {}
    entry_idx = trades["entry_idx"].values
    entry_ts  = df_bt.index.values[entry_idx]
    entry_sess = pd.Series(entry_ts).map(lambda ts: detect_session_from_hour(pd.Timestamp(ts).hour)).values
    out = {}
    for sess in ["Asia","London","NewYork"]:
        mask = (entry_sess == sess)
        sub = trades[mask]
        out[sess] = pf_from_trades(sub)
    return out

def by_month(trades, df_bt):
    if trades is None or len(trades)==0:
        return pd.DataFrame()
    entry_idx = trades["entry_idx"].values
    entry_ts  = pd.to_datetime(df_bt.index.values[entry_idx])
    months = pd.PeriodIndex(entry_ts, freq="M").astype(str)
    rows = []
    for m in sorted(pd.unique(months)):
        mask = (months == m)
        sub = trades[mask]
        row = pf_from_trades(sub)
        row["month"] = m
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    # 1) load data
    df = pd.read_parquet(DATA_FILE)
    df = ensure_ohlc(df)
    print(f"✅ Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

    # 2) features (nhanh)
    print("⏳ Extracting tsfresh features...")
    X, sample_idx = extract_tsfresh_features(df["close"], LOOKBACK_N, STRIDE, N_JOBS)
    print(f"✅ Features ready: {X.shape}")

    # 3) prepare base df for backtest alignment
    df_bt = df.iloc[sample_idx].copy()
    df_bt = attach_session(df_bt)

    lines = []
    for cfg in CONFIGS:
        fusion = cfg["fusion"]
        stop   = cfg["stop"]
        params = cfg["params"]

        print(f"\n>>> Running Analyzer for fusion='{fusion}' | stop='{stop}' ...")
        # predict all three target models
        preds_hard, probs_soft = load_models_and_predict(X)
        # fusion signal
        sig = fusion_signal(preds_hard, probs_soft, mode=fusion)
        df_bt["signal"] = sig.values

        # entries/exits theo quy ước 0=SELL,1=TIMEOUT,2=BUY
        entries = (df_bt["signal"] == 2)
        exits   = (df_bt["signal"] == 0)

        # dynamic stops
        tp_series, sl_series = stops_by_mode(df_bt, stop, params)

        # backtest & stats
        pf, stats, trades = backtest(df_bt, entries, exits, tp_series, sl_series, fees=0.0)
        overall = pf_from_trades(trades)

        # Session breakdown
        sess_stats = by_session(trades, df_bt)

        # Monthly breakdown
        month_df = by_month(trades, df_bt)

        # ---- print summary ----
        head = f"[{fusion.upper()} + {stop.upper()}]"
        print(head)
        print(f"Overall: Trades={overall['trades']} | Win%={overall['win_rate']:.2f} | PF={overall['pf']:.2f} | Exp={overall['exp_pips']:.2f} pips (${overall['exp_usd']:.2f})")

        for s in ["Asia","London","NewYork"]:
            if s in sess_stats:
                ss = sess_stats[s]
                print(f"  {s:<7}: Trades={ss['trades']:>4} | Win%={ss['win_rate']:.2f} | PF={ss['pf']:.2f} | Exp={ss['exp_pips']:.2f} pips (${ss['exp_usd']:.2f})")

        if len(month_df):
            # top 3 best & worst by PF
            md = month_df.copy()
            md["pf"] = md["pf"].replace(np.inf, np.nan)
            best = md.sort_values("pf", ascending=False).head(3).fillna(0)
            worst = md.sort_values("pf", ascending=True).head(3).fillna(0)
            print("  Top months by PF:")
            for _,r in best.iterrows():
                print(f"    {r['month']}: Trades={int(r['trades'])} | Win%={r['win_rate']:.2f} | PF={r['pf']:.2f} | Exp={r['exp_pips']:.2f} pips")
            print("  Worst months by PF:")
            for _,r in worst.iterrows():
                print(f"    {r['month']}: Trades={int(r['trades'])} | Win%={r['win_rate']:.2f} | PF={r['pf']:.2f} | Exp={r['exp_pips']:.2f} pips")

        # ---- collect for saving ----
        lines.append(f"{head}\nOverall: {overall}\nSessions: {sess_stats}\nMonthly:\n{month_df.to_string(index=False) if len(month_df) else '(no trades)'}\n")

    # 4) save report
    out_txt = os.path.join(LOG_DIR, "stage3_1_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n\n".join(lines))
    print(f"\n✅ Saved analyzer report → {out_txt}")