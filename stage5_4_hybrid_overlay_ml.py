#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding — Stage 5_4 : Hybrid Overlay + ML Confidence (Fixed)
---------------------------------------------------------------
Kết hợp model core alpha (T5_core) với overlay executor
(session filter + ATR TP/SL + probability threshold)
và đã fix lỗi ValueError: cannot reindex on axis with duplicate labels
(do duplicate 'close' columns).
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from ta.volatility import AverageTrueRange

# ================= CONFIG =================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILE = "logs/T5_core_alpha_lightgbm.txt"

WINDOW = 200
STRIDE = 5

PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

SESSIONS_TO_TRADE = ("London", "NewYork")
PROB_THRESH_GRID  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ATR_MULT_GRID     = [1.5, 2.0]

OUT_CSV = "logs/stage5_4_results.csv"
OUT_TXT = "logs/stage5_4_summary.txt"

os.makedirs("logs", exist_ok=True)

# ================= HELPERS =================
def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def load_price_oanda_bam(path: str) -> pd.DataFrame:
    """Load parquet OANDA BAM, drop duplicate OHLC, rename mid_* → open/high/low/close."""
    df = pd.read_parquet(path)

    # Drop old OHLC nếu tồn tại để tránh duplicate columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Flatten mid_* arrays (BAM)
    for c in ("mid_o","mid_h","mid_l","mid_c"):
        if c in df.columns:
            df[c] = df[c].apply(safe_mean)

    # Rename sang schema chuẩn
    df = df.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})

    # Remove duplicated columns nếu vẫn còn
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Chuẩn hoá DatetimeIndex
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    cols = ["open","high","low","close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols]

def detect_session_from_hour(h):
    if 7 <= h < 15:
        return "London"
    elif 12 <= h < 21:
        return "NewYork"
    else:
        return "Asia"

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0=long, 1=short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def atr_series(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Tính ATR an toàn (fix bug duplicate close / reindex)"""
    df = price_df[["high","low","close"]].copy()
    df = df.loc[~df.index.duplicated(keep="first")]

    # Ép sang Series RangeIndex để tránh align bug
    high_s  = pd.Series(df["high"].to_numpy(dtype=float))
    low_s   = pd.Series(df["low"].to_numpy(dtype=float))
    close_s = pd.Series(df["close"].to_numpy(dtype=float))

    atr = AverageTrueRange(high_s, low_s, close_s, window=window).average_true_range()
    atr.index = df.index  # restore datetime index
    return atr.ffill().bfill()

def run_backtest(price: pd.DataFrame, signal: pd.Series, atr_mult: float):
    """Backtest với TP/SL = ATR * atr_mult, RR 1:1. Signal: 0=SELL, 1=TIMEOUT, 2=BUY."""
    long_entries  = signal.eq(2)
    long_exits    = ~signal.eq(2)
    short_entries = signal.eq(0)
    short_exits   = ~signal.eq(0)

    atr = atr_series(price, 14)
    tp = (atr * atr_mult).clip(lower=1e-12)
    sl = (atr * atr_mult).clip(lower=1e-12)

    pf = vbt.Portfolio.from_signals(
        price["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        tp_stop=tp,
        sl_stop=sl,
        size=1.0,
        fees=FEES,
        freq="5min"
    )
    stats = pf.stats()
    exp_pips = expectancy_from_trades(pf.trades.records)
    return pf, stats, exp_pips

def get_stat(stats, key, default=np.nan):
    v = stats.get(key, default)
    try:
        return float(v)
    except Exception:
        if hasattr(v, "values"):
            arr = np.asarray(v)
            if arr.size > 0:
                return float(arr[0])
        return default

# ================= MAIN =================
if __name__ == "__main__":
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price_oanda_bam(DATA_FILE)
    price["hour"] = price.index.hour
    price["session"] = price["hour"].map(detect_session_from_hour)
    print(f"✅ Price loaded: {len(price):,} rows | cols={list(price.columns)}")

    print(f"⏳ Loading features: {FEATURE_FILE}")
    X_full = pd.read_csv(FEATURE_FILE)
    X_full.columns = (
        X_full.columns
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    print(f"✅ Features shape: {X_full.shape}")

    sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
    if len(sample_idx) < len(X_full):
        X_full = X_full.iloc[:len(sample_idx)].copy()
    sample_times = price.index[sample_idx[:len(X_full)]]

    print(f"⏳ Loading model: {MODEL_FILE}")
    booster = lgb.Booster(model_file=MODEL_FILE)
    feat_names = booster.feature_name()
    Xsub = X_full.reindex(columns=feat_names, fill_value=0.0)

    probs = np.asarray(booster.predict(Xsub.values))  # [n_samples, 3]
    pred_cls = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    signal_base = pd.Series(pred_cls, index=sample_times).reindex(price.index, method="ffill").fillna(1).astype(int)
    prob_base = pd.Series(max_prob, index=sample_times).reindex(price.index, method="ffill").fillna(0.0)
    session_mask = price["session"].isin(SESSIONS_TO_TRADE)

    rows = []
    print("\n========== Stage 5_4 Hybrid Overlay + ML Confidence ==========")
    for th in PROB_THRESH_GRID:
        tradable = session_mask & (prob_base >= th)
        sig = signal_base.where(tradable, 1)
        for k in ATR_MULT_GRID:
            print(f"\n→ THRESH={th:.2f} | ATRx={k}")
            pf, stats, exp_pips = run_backtest(price, sig, atr_mult=k)
            trades   = int(get_stat(stats, "Total Trades", 0))
            winrate  = get_stat(stats, "Win Rate [%]", np.nan)
            pfactor  = get_stat(stats, "Profit Factor", np.nan)
            ret      = get_stat(stats, "Total Return [%]", np.nan)
            mdd      = get_stat(stats, "Max Drawdown [%]", np.nan)
            exp_usd  = exp_pips * PIP_USD

            print(f"Trades={trades} | Win%={winrate:.2f} | PF={pfactor:.2f} | Ret%={ret:.2f} | DD%={mdd:.2f} | Exp={exp_pips:.2f} pips (${exp_usd:.2f})")

            rows.append({
                "prob_threshold": th,
                "atr_mult": k,
                "trades": trades,
                "winrate_%": winrate,
                "profit_factor": pfactor,
                "total_return_%": ret,
                "max_drawdown_%": mdd,
                "expectancy_pips": exp_pips,
                "expectancy_usd_1lot": exp_usd
            })

    res = pd.DataFrame(rows).sort_values(["profit_factor","expectancy_pips","winrate_%"], ascending=False)
    res.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w") as f:
        f.write("========== Stage 5_4 Hybrid Overlay + ML Confidence ==========\n\n")
        f.write(res.head(10).to_string(index=False))
        f.write("\n\nFiltered (PF>=1, trades>=50):\n")
        filt = res.query("profit_factor >= 1.0 and trades >= 50")
        if len(filt) > 0:
            f.write(filt.head(20).to_string(index=False))
        else:
            f.write("(none)\n")

    print(f"\n✅ Saved results → {OUT_CSV}")
    print(f"✅ Summary saved → {OUT_TXT}")
    print("🎯 Stage 5_4 complete.")