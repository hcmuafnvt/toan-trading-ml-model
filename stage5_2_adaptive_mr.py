# stage5_2b_adaptive_mr_strict.py
import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
ML_FINAL_SIGNAL_FILE = "logs/stage4_final_signal.csv"

OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# ===== strict filters =====
ATR_WINDOW = 14
LOW_Q, HIGH_Q = 0.33, 0.66

BB_WINDOW = 50
BB_NSTD   = 2.0
BW_PCTL   = 0.30          # chặt hơn

RSI_WINDOW = 14
RSI_LOW, RSI_HIGH = 25, 75

# dist-to-MA bằng ATR
DIST_ATR_MIN = 0.7
DIST_ATR_MAX = 2.5

# Cooldown sau entry MR (bars)
COOLDOWN_BARS = 8

SESSION_WHITELIST = {"Asia", "London"}

FIXED_TP_PIPS = 20.0
FIXED_SL_PIPS = 20.0

OUT_SUMMARY = "logs/stage5_2b_summary.txt"
OUT_FINAL_SIG = "logs/stage5_2b_overlay_signal.csv"
os.makedirs("logs", exist_ok=True)

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

def expectancy_from_trades(trades: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if trades is None or len(trades) == 0:
        return 0.0
    cols = {c.lower(): c for c in trades.columns}
    def col(*names):
        for n in names:
            if n.lower() in cols:
                return trades[cols[n.lower()]].values
        return None
    entry = col("entry_price","avg entry price")
    exitp = col("exit_price","avg exit price")
    direction = col("direction")
    if entry is None or exitp is None:
        return 0.0
    if direction is None:
        sign = np.ones_like(entry)
    else:
        ser = pd.Series(direction)
        sign = np.where(ser.astype(str).str.lower().str.contains("long"), 1.0, -1.0) if ser.dtype==object else np.where(ser==0,1.0,-1.0)
    n = min(len(entry), len(exitp), len(sign))
    if n == 0:
        return 0.0
    pips = (exitp[:n]-entry[:n]) * sign[:n] / pip_size
    return float(np.nanmean(pips))

def _get_stat(st: pd.Series, k: str) -> float:
    v = st.get(k, np.nan)
    try:
        return float(v)
    except Exception:
        if hasattr(v, "values"):
            return float(np.asarray(v)[0])
        return float("nan")

# ---- MAIN ----
price = load_price(DATA_FILE)
close = price["close"]

ml = pd.read_csv(ML_FINAL_SIGNAL_FILE, index_col=0, parse_dates=True).squeeze("columns")
ml.index = ml.index.tz_localize(None)
ml = ml.reindex(price.index, method="ffill").fillna(1).astype(int)

# baseline ML (fixed 20/20) để so sánh
long_ml  = ml.eq(2); short_ml = ml.eq(0)
pf_ml = vbt.Portfolio.from_signals(
    close,
    entries=long_ml,
    exits=~long_ml,
    short_entries=short_ml,
    short_exits=~short_ml,
    tp_stop=FIXED_TP_PIPS*PIP_SIZE,
    sl_stop=FIXED_SL_PIPS*PIP_SIZE,
    size=1.0, fees=FEES, freq="5min"
)
st_ml = pf_ml.stats(); exp_ml = expectancy_from_trades(pf_ml.trades.records)

# indicators
atr = AverageTrueRange(price["high"], price["low"], price["close"], window=ATR_WINDOW).average_true_range().reindex(price.index).ffill().bfill()
ma  = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
sd  = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
upper = ma + BB_NSTD*sd; lower = ma - BB_NSTD*sd
bw = (upper - lower) / ma
bw_thr = bw.quantile(BW_PCTL)
rsi = RSIIndicator(close, window=RSI_WINDOW).rsi()
q_low, q_high = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
regime_low = atr.le(q_low)

sess = pd.Index(price.index.map(get_session), name="session")
session_ok = sess.isin(SESSION_WHITELIST) if SESSION_WHITELIST is not None else pd.Series(True, index=price.index)

# only when ML timeout
timeout_mask = ml.eq(1)

# distance to MA (in ATR units)
dist_atr = (close - ma).abs() / atr.replace(0, np.nan)
dist_ok = dist_atr.between(DIST_ATR_MIN, DIST_ATR_MAX)

# medium: chạm band HOẶC cross-back
touch_lo_prev = close.shift(1) <= lower.shift(1)
touch_up_prev = close.shift(1) >= upper.shift(1)
back_inside_up = (close <= lower) | ((close >= lower) & touch_lo_prev)
back_inside_dn = (close >= upper) | ((close <= upper) & touch_up_prev)

tight_bw = bw.lt(bw_thr)

long_mr  = back_inside_up  & (rsi < RSI_LOW)  & timeout_mask & regime_low & tight_bw & dist_ok & session_ok
short_mr = back_inside_dn  & (rsi > RSI_HIGH) & timeout_mask & regime_low & tight_bw & dist_ok & session_ok
print(f"MR candidates (after filters): long={int(long_mr.sum())}, short={int(short_mr.sum())}, timeout={int(timeout_mask.sum())}")

# cooldown: không mở MR mới trong COOLDOWN_BARS sau một entry
def apply_cooldown(sig: pd.Series, bars: int) -> pd.Series:
    sig = sig.astype(bool).copy()
    if bars <= 0: return sig
    open_flag = False; counter = 0
    out = np.zeros(len(sig), dtype=bool)
    vals = sig.values
    for i in range(len(vals)):
        if open_flag:
            counter -= 1
            if counter <= 0:
                open_flag = False
        if not open_flag and vals[i]:
            out[i] = True
            open_flag = True
            counter = bars
    return pd.Series(out, index=sig.index)

long_mr  = apply_cooldown(long_mr, COOLDOWN_BARS)
short_mr = apply_cooldown(short_mr, COOLDOWN_BARS)

# overlay: giữ ML, chỉ thay timeout bằng MR
overlay = ml.copy()
overlay[long_mr]  = 2
overlay[short_mr] = 0

pf_ov = vbt.Portfolio.from_signals(
    close,
    entries=overlay.eq(2),
    exits=~overlay.eq(2),
    short_entries=overlay.eq(0),
    short_exits=~overlay.eq(0),
    tp_stop=FIXED_TP_PIPS*PIP_SIZE,
    sl_stop=FIXED_SL_PIPS*PIP_SIZE,
    size=1.0, fees=FEES, freq="5min"
)
st_ov = pf_ov.stats(); exp_ov = expectancy_from_trades(pf_ov.trades.records)

def fmt(tag, st, exp):
    return (
        f"{tag}\n"
        f"Total Trades        : {int(_get_stat(st,'Total Trades'))}\n"
        f"Win Rate [%]        : {_get_stat(st,'Win Rate [%]'):.2f}\n"
        f"Profit Factor       : {_get_stat(st,'Profit Factor'):.2f}\n"
        f"Max Drawdown [%]    : {_get_stat(st,'Max Drawdown [%]'):.2f}\n"
        f"Expectancy (pips)   : {exp:.2f}\n"
        f"Expectancy (USD)    : ${exp*PIP_USD:.2f} /trade (1 lot)\n"
    )

print("\n========== STAGE 5.2b STRICT ==========")
print("---- ML ONLY (fixed 20/20) ----")
print(fmt("ML ONLY", st_ml, exp_ml))
print("---- ML + MR STRICT OVERLAY ----")
print(f"MR candidates: long={int(long_mr.sum())}, short={int(short_mr.sum())}, timeout bars={int(timeout_mask.sum())}")
print(fmt("ML + MR", st_ov, exp_ov))

with open(OUT_SUMMARY, "w") as f:
    f.write("========== CONFIG ==========\n")
    f.write(f"BW_PCTL={BW_PCTL}, RSI={RSI_LOW}/{RSI_HIGH}, DIST_ATR=[{DIST_ATR_MIN},{DIST_ATR_MAX}], COOLDOWN={COOLDOWN_BARS}\n")
    f.write("\n---- ML ONLY ----\n"); f.write(fmt("ML ONLY", st_ml, exp_ml))
    f.write("\n---- ML + MR STRICT ----\n"); f.write(fmt("ML + MR", st_ov, exp_ov))
    try:
        f.write("\n---- DELTA ----\n")
        f.write(f"ΔPF      : {_get_stat(st_ov,'Profit Factor') - _get_stat(st_ml,'Profit Factor'):+.2f}\n")
        f.write(f"ΔWinRate : {_get_stat(st_ov,'Win Rate [%]') - _get_stat(st_ml,'Win Rate [%]'):+.2f} %\n")
        f.write(f"ΔExpect. : {exp_ov - exp_ml:+.2f} pips\n")
    except Exception:
        pass

overlay.name = "final_signal"
overlay.to_csv(OUT_FINAL_SIG, index=True)
print(f"\n✅ Saved summary → {OUT_SUMMARY}")
print(f"✅ Saved overlay final signal → {OUT_FINAL_SIG}")