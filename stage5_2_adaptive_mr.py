# -*- coding: utf-8 -*-
# Stage 5.2 – Adaptive Mean-Reversion overlay for ML final signal
# - Inputs:
#     data/GBP_USD_M5_2024.parquet   (OHLC mid_* schema)
#     logs/stage4_final_signal.csv   (series int: 0/1/2 from Stage 4)
# - Logic:
#   * MR chỉ bật khi ML=1 (timeout)
#   * Filter: ATR regime Low + Bollinger bandwidth dưới ngưỡng + (optional) session
#   * Dùng cùng stop rule fixed 20/20 cho ML baseline và ML+MR để so sánh công bằng
# - Outputs:
#     logs/stage5_2_summary.txt
#     logs/stage5_2_overlay_signal.csv  (final combined 0/1/2)
#     console: bảng so sánh baseline vs overlay

import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

# ================== CONFIG ==================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
ML_FINAL_SIGNAL_FILE = "logs/stage4_final_signal.csv"  # từ stage 4 (series 0/1/2)

# Parquet (OANDA mid_*)
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# Pips & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# MR filters
ATR_WINDOW = 14
LOW_Q, HIGH_Q = 0.33, 0.66              # Low/Normal/High split
BB_WINDOW = 50
BB_NSTD   = 2.0
BW_PCTL   = 0.50                         # dùng bandwidth < 50th percentile
RSI_WINDOW = 14
RSI_LOW, RSI_HIGH = 30, 70               # optional, dùng mỗi nhánh tương ứng

# Session filter (None=all). Ví dụ: {"Asia", "London"} hoặc None
SESSION_WHITELIST = {"Asia", "London"}   # default: bật MR ở Asia + London

# Stops để so sánh công bằng
FIXED_TP_PIPS = 20.0
FIXED_SL_PIPS = 20.0

OUT_SUMMARY = "logs/stage5_2_summary.txt"
OUT_FINAL_SIG = "logs/stage5_2_overlay_signal.csv"

os.makedirs("logs", exist_ok=True)


# ================== HELPERS ==================
def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def load_price(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
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

def get_session(ts):
    h = ts.hour
    if 22 <= h or h < 6:
        return "Asia"
    elif 6 <= h < 14:
        return "London"
    else:
        return "NewYork"

def expectancy_from_trades(trades: pd.DataFrame) -> float:
    if trades is None or len(trades) == 0:
        return 0.0
    cols = {c.lower(): c for c in trades.columns}
    def get_col(*names):
        for n in names:
            if n.lower() in cols:
                return trades[cols[n.lower()]].values
        return None
    entry = get_col("entry_price", "avg entry price")
    exitp = get_col("exit_price", "avg exit price")
    direction = get_col("direction")
    if entry is None or exitp is None:
        return 0.0
    if direction is None:
        sign = np.ones_like(entry)
    else:
        ser = pd.Series(direction)
        if ser.dtype == object:
            sign = np.where(ser.astype(str).str.lower().str.contains("long"), 1.0, -1.0)
        else:
            sign = np.where(ser.values == 0, 1.0, -1.0)
    n = min(len(entry), len(exitp), len(sign))
    if n == 0:
        return 0.0
    pips = (exitp[:n] - entry[:n]) * sign[:n] / PIP_SIZE
    return float(np.nanmean(pips))

def _get_stat(stats: pd.Series, k: str) -> float:
    v = stats.get(k, np.nan)
    try:
        return float(v)
    except Exception:
        if hasattr(v, "values"):
            return float(np.asarray(v)[0])
        return float("nan")


# ================== MAIN ==================
if __name__ == "__main__":
    print("⏳ Loading price & ML final signal ...")
    price = load_price(DATA_FILE)
    close = price["close"]

    ml = pd.read_csv(ML_FINAL_SIGNAL_FILE, index_col=0, parse_dates=True).squeeze("columns")
    ml.index = ml.index.tz_localize(None)
    ml = ml.reindex(price.index, method="ffill").fillna(1).astype(int)

    # --- Derived series ---
    atr = AverageTrueRange(high=price["high"], low=price["low"], close=price["close"], window=ATR_WINDOW).average_true_range()
    atr = atr.reindex(price.index).ffill().bfill()

    # Regime (Low / Normal / High) theo ATR
    q_low, q_high = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
    regime = pd.Series(np.where(atr <= q_low, "Low",
                         np.where(atr >= q_high, "High", "Normal")), index=price.index)

    # Bollinger bandwidth
    ma = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    sd = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    upper = ma + BB_NSTD * sd
    lower = ma - BB_NSTD * sd
    bw = (upper - lower) / ma
    bw_thr = bw.quantile(BW_PCTL)

    # RSI (để an toàn thêm filter hướng)
    rsi = RSIIndicator(close=close, window=RSI_WINDOW).rsi()

    # Session
    sess = pd.Index(price.index.map(get_session), name="session")

    # ========== BASELINE (ML-only, fixed 20/20) ==========
    long_entries_ml  = ml.eq(2)
    long_exits_ml    = ~ml.eq(2)
    short_entries_ml = ml.eq(0)
    short_exits_ml   = ~ml.eq(0)

    tp_fixed = FIXED_TP_PIPS * PIP_SIZE
    sl_fixed = FIXED_SL_PIPS * PIP_SIZE

    pf_ml = vbt.Portfolio.from_signals(
        close,
        entries=long_entries_ml,
        exits=long_exits_ml,
        short_entries=short_entries_ml,
        short_exits=short_exits_ml,
        tp_stop=tp_fixed,
        sl_stop=sl_fixed,
        size=1.0,
        fees=FEES,
        freq="5min"
    )
    st_ml = pf_ml.stats()
    trades_ml = pf_ml.trades.records
    exp_ml = expectancy_from_trades(trades_ml)

    # ========== MR overlay (chỉ khi ML=1) ==========
    timeout_mask = ml.eq(1)

    # Filter “sideway thật”:
    #   - Low regime
    #   - bandwidth < median (hoặc BW_PCTL)
    #   - optional session whitelist
    low_regime = regime.eq("Low")
    tight_bw = bw.lt(bw_thr)
    session_ok = sess.isin(SESSION_WHITELIST) if SESSION_WHITELIST is not None else pd.Series(True, index=price.index)

    # Entry MR:
    #   Long khi close < lower & RSI < RSI_LOW
    #   Short khi close > upper & RSI > RSI_HIGH
    long_entry_mr  = (close < lower) & (rsi < RSI_LOW) & timeout_mask & low_regime & tight_bw & session_ok
    short_entry_mr = (close > upper) & (rsi > RSI_HIGH) & timeout_mask & low_regime & tight_bw & session_ok

    # Exit dùng chung fixed 20/20 (để so sánh công bằng) + exit khi chạm MA để hạn chế “kẹt lệnh”
    cross_to_ma = ( (close >= ma) & long_entry_mr.shift(1, fill_value=False) ) | \
                  ( (close <= ma) & short_entry_mr.shift(1, fill_value=False) )
    exit_sig = cross_to_ma.fillna(False)

    # Xây fused overlay signal: mặc định ML; override timeout bằng MR
    overlay = ml.copy()
    overlay[long_entry_mr]  = 2
    overlay[short_entry_mr] = 0

    # Backtest overlay (fixed 20/20 để công bằng)
    pf_ov = vbt.Portfolio.from_signals(
        close,
        entries=overlay.eq(2),
        exits=~overlay.eq(2),
        short_entries=overlay.eq(0),
        short_exits=~overlay.eq(0),
        tp_stop=tp_fixed,
        sl_stop=sl_fixed,
        size=1.0,
        fees=FEES,
        freq="5min"
    )
    st_ov = pf_ov.stats()
    trades_ov = pf_ov.trades.records
    exp_ov = expectancy_from_trades(trades_ov)

    # ========== REPORT ==========
    def fmt_stats(tag, st, exp):
        return (
            f"{tag}\n"
            f"Total Trades        : {int(_get_stat(st,'Total Trades'))}\n"
            f"Win Rate [%]        : {_get_stat(st,'Win Rate [%]'):.2f}\n"
            f"Profit Factor       : {_get_stat(st,'Profit Factor'):.2f}\n"
            f"Max Drawdown [%]    : {_get_stat(st,'Max Drawdown [%]'):.2f}\n"
            f"Expectancy (pips)   : {exp:.2f}\n"
            f"Expectancy (USD)    : ${exp * PIP_USD:.2f} /trade (1 lot)\n"
        )

    print("\n========== STAGE 5.2: ML baseline (fixed 20/20) ==========")
    print(fmt_stats("ML ONLY", st_ml, exp_ml))

    print("========== STAGE 5.2: ML + Adaptive MR overlay ==========")
    print(f"MR applied on timeout bars: {int(timeout_mask.sum())} total timeout | "
          f"entries long={int(long_entry_mr.sum())} short={int(short_entry_mr.sum())}")
    print(fmt_stats("ML + MR", st_ov, exp_ov))

    # Save summary + final overlay signal
    with open(OUT_SUMMARY, "w") as f:
        f.write("========== CONFIG ==========\n")
        f.write(f"ATR_WINDOW={ATR_WINDOW}, LOW_Q={LOW_Q}, HIGH_Q={HIGH_Q}\n")
        f.write(f"BB_WINDOW={BB_WINDOW}, BB_NSTD={BB_NSTD}, BW_PCTL={BW_PCTL}\n")
        f.write(f"RSI_WINDOW={RSI_WINDOW}, RSI_LOW={RSI_LOW}, RSI_HIGH={RSI_HIGH}\n")
        f.write(f"SESSION_WHITELIST={SESSION_WHITELIST}\n")
        f.write(f"Stops=FIXED {FIXED_TP_PIPS}/{FIXED_SL_PIPS} (pips)\n\n")

        f.write("========== ML ONLY ==========\n")
        f.write(fmt_stats("ML ONLY", st_ml, exp_ml))
        f.write("\n========== ML + MR ==========\n")
        f.write(fmt_stats("ML + MR", st_ov, exp_ov))

        # So sánh chênh lệch
        try:
            pf_ml_val = _get_stat(st_ml,"Profit Factor")
            pf_ov_val = _get_stat(st_ov,"Profit Factor")
            wr_ml = _get_stat(st_ml,"Win Rate [%]")
            wr_ov = _get_stat(st_ov,"Win Rate [%]")
            f.write("\n========== DELTA ==========\n")
            f.write(f"ΔPF      : {pf_ov_val - pf_ml_val:+.2f}\n")
            f.write(f"ΔWinRate : {wr_ov - wr_ml:+.2f} %\n")
            f.write(f"ΔExpect. : {exp_ov - exp_ml:+.2f} pips\n")
        except Exception:
            pass

    overlay.name = "final_signal"
    overlay.to_csv(OUT_FINAL_SIG, index=True)
    print(f"\n✅ Saved summary → {OUT_SUMMARY}")
    print(f"✅ Saved overlay final signal → {OUT_FINAL_SIG}")