import pandas as pd
import numpy as np
import lightgbm as lgb
import vectorbt as vbt
from datetime import datetime
from ta.volatility import AverageTrueRange

# =========================================================
# CONFIG
# =========================================================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"

PIP_SIZE = 0.0001
PIP_USD = 10.0      # 1 lot = 100k
FEES = 0.0

TP_SL_FIXED = [10.0, 15.0, 20.0]  # pips
TP_SL_ATR = [1.5]                 # ATR multipliers
FUSION_MODES = ["majority", "weighted", "buy_only"]

# =========================================================
# CUSTOM ATR CALCULATION (WILDER)
# =========================================================
def atr_wilder(price: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR (Wilder) custom implementation to avoid ta.align bug."""
    px = price[["high", "low", "close"]].copy()
    px = px.sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)

    prev_close = px["close"].shift(1)
    tr1 = px["high"] - px["low"]
    tr2 = (px["high"] - prev_close).abs()
    tr3 = (px["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

# =========================================================
# BACKTEST FUNCTION
# =========================================================
def run_backtest(price: pd.DataFrame, signal: pd.Series, stop_mode="fixed", tp_param=10.0, sl_param=10.0):
    """VectorBT backtest logic."""
    long_entries = signal == 2
    long_exits   = signal == 0
    short_entries = signal == 0
    short_exits   = signal == 2

    if stop_mode == "fixed":
        tp = tp_param * PIP_SIZE
        sl = sl_param * PIP_SIZE
    elif stop_mode == "atr":
        atr = atr_wilder(price, window=14).ffill().bfill()
        tp = (atr * tp_param).clip(lower=1e-7)
        sl = (atr * sl_param).clip(lower=1e-7)
    else:
        raise ValueError(f"Unknown stop_mode: {stop_mode}")

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
    trades = pf.trades.records_readable

    if trades.empty:
        exp_pips = 0.0
    else:
        sign = np.where(trades["Direction"] == "Long", 1, -1)
        pips = (trades["Exit Price"] - trades["Entry Price"]) * sign / PIP_SIZE
        exp_pips = np.nanmean(pips)

    return pf, stats, exp_pips

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print(f"⏳ Loading price data {DATA_FILE} ...")
    price = pd.read_parquet(DATA_FILE)
    price["close"] = price["mid_c"]
    price["high"] = price["mid_h"]
    price["low"]  = price["mid_l"]
    price.index = pd.to_datetime(price.index).tz_localize(None)
    print(f"✅ Loaded {len(price):,} rows | {price.index[0]} → {price.index[-1]}")

    # Load models
    models = {}
    for name in ["T1_10x40", "T2_15x60", "T3_20x80"]:
        models[name] = lgb.Booster(model_file=f"logs/{name}_lightgbm.txt")
        print(f"✅ Loaded model {name}")

    # Load features
    print(f"⏳ Loading features from {FEATURE_FILE} ...")
    df_feat = pd.read_csv(FEATURE_FILE)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"✅ Features loaded: {df_feat.shape}")

    X = df_feat.drop(columns=["target_10x40", "target_15x60", "target_20x80"], errors="ignore")

    # Predict for each model
    probs = {k: models[k].predict(X) for k in models}
    preds = {k: np.argmax(probs[k], axis=1) for k in models}

    # Fusion
    print("\n========== FUSION BACKTEST ==========")
    fusion_modes = FUSION_MODES

    for fmode in fusion_modes:
        if fmode == "majority":
            combined = pd.DataFrame(preds).mode(axis=1)[0]
        elif fmode == "weighted":
            weights = [1.2, 1.0, 0.8]
            weighted_sum = sum(preds[k] * w for k, w in zip(preds.keys(), weights))
            combined = np.round(weighted_sum / np.mean(weights))
        elif fmode == "buy_only":
            combined = np.where(preds["T1_10x40"] == 2, 2, 1)
        else:
            raise ValueError("Unknown fusion mode")

        signal = pd.Series(combined, index=price.index[-len(combined):])

        # Fixed stops
        for tp_sl in TP_SL_FIXED:
            tp, sl = tp_sl, tp_sl
            print(f"\n>>> Running fusion='{fmode}' | stop='fixed' | tp={tp} sl={sl}")
            pf, stats, exp_pips = run_backtest(price, signal, "fixed", tp, sl)
            winrate = stats["Win Rate [%]"]
            pf_val  = stats["Profit Factor"]
            ret_val = stats["Total Return [%]"]
            dd_val  = stats["Max Drawdown [%]"]
            print(f"Trades={len(pf.trades.records)} | Win%={winrate:.2f} | PF={pf_val:.2f} | Ret%={ret_val:.2f} | DD%={dd_val:.2f} | Exp={exp_pips:.2f} pips (${exp_pips*PIP_USD:.2f})")

        # ATR-based stops
        for mult in TP_SL_ATR:
            print(f"\n>>> Running fusion='{fmode}' | stop='atr' | tp={mult} sl={mult}")
            pf, stats, exp_pips = run_backtest(price, signal, "atr", mult, mult)
            winrate = stats["Win Rate [%]"]
            pf_val  = stats["Profit Factor"]
            ret_val = stats["Total Return [%]"]
            dd_val  = stats["Max Drawdown [%]"]
            print(f"Trades={len(pf.trades.records)} | Win%={winrate:.2f} | PF={pf_val:.2f} | Ret%={ret_val:.2f} | DD%={dd_val:.2f} | Exp={exp_pips:.2f} pips (${exp_pips*PIP_USD:.2f})")