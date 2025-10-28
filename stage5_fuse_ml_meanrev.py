import os
import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.volatility import AverageTrueRange

# ===================== CONFIG =====================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"

ML_FINAL_SIGNAL_FILE   = "logs/stage4_final_signal.csv"        # đã lưu ở Stage 4
MR_BEST_SIGNAL_FILE    = "logs/stage4_1_meanrev_best_signal.csv" # đã lưu ở Stage 4.1/4.2
STAGE4_BESTCFG_JSON    = "logs/stage4_best_config.json"        # best per (session×regime)

OUT_SUMMARY = "logs/stage5_summary.txt"
OUT_SIGNAL  = "logs/stage5_final_signal.csv"
os.makedirs("logs", exist_ok=True)

# Schema OANDA mid_*
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# Pips & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# Regime theo ATR quantile (phải khớp Stage 4)
LOW_Q, HIGH_Q = 0.33, 0.66

# xung đột ML vs MR: 'ml' (ưu tiên ML), 'mr' (ưu tiên MR), 'skip' (đặt 1)
CONFLICT_POLICY = "ml"   # "ml" | "mr" | "skip"

# ===================== HELPERS =====================
def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def load_price(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Chuẩn hoá OHLC
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
    df.index = pd.to_datetime(df.index).tz_localize(None)  # đồng bộ tz-naive
    return df

def atr_from_ta(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    tmp = price_df[["high","low","close"]].copy()
    high_s  = pd.Series(tmp["high"].to_numpy(dtype=float))
    low_s   = pd.Series(tmp["low"].to_numpy(dtype=float))
    close_s = pd.Series(tmp["close"].to_numpy(dtype=float))
    atr = AverageTrueRange(high_s, low_s, close_s, window=window).average_true_range()
    atr.index = price_df.index
    return atr.ffill().bfill()

def get_session(ts):
    h = ts.hour
    if 22 <= h or h < 6:
        return "Asia"
    elif 6 <= h < 14:
        return "London"
    else:
        return "NewYork"

def expectancy_from_trades(tr_rec: pd.DataFrame) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    # vectorbt mới
    entry = tr_rec.get("entry_price", pd.Series(dtype=float)).values
    exitp = tr_rec.get("exit_price",  pd.Series(dtype=float)).values
    direction = tr_rec.get("direction", pd.Series(dtype=float)).values  # 0 long, 1 short
    if len(entry)==0 or len(exitp)==0:
        # vectorbt readable (cũ)
        entry = tr_rec.get("Avg Entry Price", pd.Series(dtype=float)).values
        exitp = tr_rec.get("Avg Exit Price",  pd.Series(dtype=float)).values
        direction = tr_rec.get("Direction", pd.Series(["long"]*len(entry))).values
        sign = np.where(pd.Series(direction).astype(str).str.lower().str.contains("long"), 1.0, -1.0)
    else:
        sign = np.where(direction == 0, 1.0, -1.0)
    n = min(len(entry), len(exitp), len(sign))
    if n == 0:
        return 0.0
    pips = (exitp[:n] - entry[:n]) * sign[:n] / PIP_SIZE
    return float(np.nanmean(pips))

def profit_usd_from_trades(tr_rec: pd.DataFrame) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec.get("entry_price", pd.Series(dtype=float)).values
    exitp = tr_rec.get("exit_price",  pd.Series(dtype=float)).values
    direction = tr_rec.get("direction", pd.Series(dtype=float)).values
    size = tr_rec.get("size", pd.Series([1.0]*len(entry))).values
    if len(entry)==0 or len(exitp)==0:
        entry = tr_rec.get("Avg Entry Price", pd.Series(dtype=float)).values
        exitp = tr_rec.get("Avg Exit Price",  pd.Series(dtype=float)).values
        direction = tr_rec.get("Direction", pd.Series(["long"]*len(entry))).values
        size = tr_rec.get("Size", pd.Series([1.0]*len(entry))).values
        sign = np.where(pd.Series(direction).astype(str).str.lower().str.contains("long"), 1.0, -1.0)
    else:
        sign = np.where(direction == 0, 1.0, -1.0)
    n = min(len(entry), len(exitp), len(sign), len(size))
    if n == 0:
        return 0.0
    pips = (exitp[:n] - entry[:n]) * sign[:n] / PIP_SIZE
    usd = pips * PIP_USD * size[:n]
    return float(np.nansum(usd))

def build_stops_from_bestcfg(price: pd.DataFrame, best_cfg: dict) -> (pd.Series, pd.Series):
    """Dựng TP/SL per (session×regime) từ stage4_best_config.json.
       Nếu thiếu segment -> fallback fixed 10p."""
    atr14 = atr_from_ta(price, window=14)
    q_low, q_high = atr14.quantile(LOW_Q), atr14.quantile(HIGH_Q)
    regime = pd.Series(np.where(atr14 <= q_low, "Low",
                         np.where(atr14 >= q_high, "High", "Normal")), index=price.index)
    session = price.index.map(get_session)

    tp = pd.Series(np.nan, index=price.index, dtype=float)
    sl = pd.Series(np.nan, index=price.index, dtype=float)

    for k, cfg in best_cfg.items():
        # key dạng "Asia__High"
        try:
            ss, rg = k.split("__")
        except:
            # phòng trường hợp key là tuple-like
            parts = str(k).replace("(", "").replace(")", "").replace("'", "").split(",")
            if len(parts) >= 2:
                ss = parts[0].strip()
                rg = parts[1].strip()
            else:
                continue
        mask = (session == ss) & (regime == rg)
        if mask.sum() == 0:
            continue
        sm = cfg.get("stop", "fixed")
        tpv = float(cfg.get("tp", 10.0))
        slv = float(cfg.get("sl", 10.0))
        if sm == "fixed":
            tp.loc[mask] = tpv * PIP_SIZE
            sl.loc[mask] = slv * PIP_SIZE
        else:
            tp.loc[mask] = (atr14.loc[mask] * tpv).clip(lower=1e-12)
            sl.loc[mask] = (atr14.loc[mask] * slv).clip(lower=1e-12)

    # Fallback cho NaN
    tp = tp.fillna(10 * PIP_SIZE)
    sl = sl.fillna(10 * PIP_SIZE)
    return tp, sl

# ===================== MAIN =====================
if __name__ == "__main__":
    print("⏳ Loading price & signals...")
    price = load_price(DATA_FILE)
    close = price["close"]

    # ML signal
    ml = pd.read_csv(ML_FINAL_SIGNAL_FILE, index_col=0, parse_dates=True, squeeze=True)
    ml.index = ml.index.tz_localize(None)
    ml = ml.reindex(price.index, method="ffill").fillna(1).astype(int)

    # MR signal
    mr = pd.read_csv(MR_BEST_SIGNAL_FILE, index_col=0, parse_dates=True, squeeze=True)
    mr.index = mr.index.tz_localize(None)
    mr = mr.reindex(price.index, method="ffill").fillna(1).astype(int)

    # Best config Stage 4 (TP/SL theo segment)
    if os.path.exists(STAGE4_BESTCFG_JSON):
        with open(STAGE4_BESTCFG_JSON, "r") as f:
            best_cfg = json.load(f)
    else:
        best_cfg = {}

    # --- Hợp nhất ---
    # base = ML; nếu ML=1 (timeout) và MR ∈ {0,2} -> dùng MR
    combined = ml.copy()
    use_mr_mask = (ml == 1) & (mr != 1)
    combined[use_mr_mask] = mr[use_mr_mask]

    # xung đột: cả 2 đều ∈ {0,2} nhưng ngược nhau
    conflict = (ml != 1) & (mr != 1) & (ml != mr)
    if conflict.any():
        if CONFLICT_POLICY == "ml":
            # giữ ML
            pass
        elif CONFLICT_POLICY == "mr":
            combined[conflict] = mr[conflict]
        elif CONFLICT_POLICY == "skip":
            combined[conflict] = 1

    # --- TP/SL theo best_cfg (session×regime); fallback fixed 10p
    tp_series, sl_series = build_stops_from_bestcfg(price, best_cfg)

    # --- Backtest danh mục hợp nhất ---
    pf = vbt.Portfolio.from_signals(
        close,
        entries=combined.eq(2),
        exits=~combined.eq(2),
        short_entries=combined.eq(0),
        short_exits=~combined.eq(0),
        tp_stop=tp_series,
        sl_stop=sl_series,
        size=1.0,         # đơn giản: 1 lot; có thể scale theo confidence nếu sau này ta lưu serie
        fees=FEES,
        freq="5min"
    )

    stats = pf.stats()
    trades = pf.trades.records
    exp_pips = expectancy_from_trades(trades)
    total_usd = profit_usd_from_trades(trades)

    def _get(s, k):
        v = s.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            if hasattr(v, "values"):
                return float(np.asarray(v)[0])
            return float("nan")

    print("\n========== STAGE 5: ML + MeanReversion (Adaptive) ==========")
    print(f"Total Trades        : {int(_get(stats,'Total Trades'))}")
    print(f"Win Rate [%]        : {_get(stats,'Win Rate [%]'):.2f}")
    print(f"Profit Factor       : {_get(stats,'Profit Factor'):.2f}")
    print(f"Max Drawdown [%]    : {_get(stats,'Max Drawdown [%]'):.2f}")
    print(f"Expectancy (pips)   : {exp_pips:.2f}")
    print(f"Expectancy (USD)    : ${exp_pips * PIP_USD:.2f} /trade (1 lot)")
    print(f"Total Profit (USD)  : ${total_usd:.0f} (1 lot)")

    # Lưu signal hợp nhất để dùng Stage sau
    combined.name = "stage5_signal"
    combined.to_csv(OUT_SIGNAL, index=True)

    with open(OUT_SUMMARY, "w") as f:
        f.write("========== STAGE 5: ML + MeanReversion (Adaptive) ==========\n")
        f.write(stats.to_string())
        f.write(f"\nExpectancy (pips): {exp_pips:.4f} | Expectancy USD: {exp_pips*PIP_USD:.2f} | Total Profit USD: {total_usd:.0f}\n")
        f.write(f"\nConflict policy: {CONFLICT_POLICY}\n")
        f.write(f"Signals:\n - ML: {ML_FINAL_SIGNAL_FILE}\n - MeanRev: {MR_BEST_SIGNAL_FILE}\n")
        f.write(f"TP/SL from: {STAGE4_BESTCFG_JSON if best_cfg else 'fallback fixed 10p'}\n")

    print(f"\n✅ Saved summary → {OUT_SUMMARY}")
    print(f"✅ Saved final fused signal → {OUT_SIGNAL}")