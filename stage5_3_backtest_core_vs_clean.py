#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding — Stage 5 : Backtest & Forward Simulation
----------------------------------------------------
So sánh hiệu quả thực tế giữa 2 mô hình:
  - T4_clean_lightgbm.txt  (59 features)
  - T5_core_alpha_lightgbm.txt (34 features)

Đánh giá trên dữ liệu giá M5 2024:
  Profit Factor, Winrate, Max Drawdown, Expectancy (pips/USD)
"""

import os, numpy as np, pandas as pd, lightgbm as lgb, vectorbt as vbt
from ta.volatility import AverageTrueRange
from datetime import datetime

# ========== CONFIG ==========
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"
Y_FILE = "logs/stage3_y.csv"

MODELS = {
    "T4_clean": "logs/T4_clean_lightgbm.txt",
    "T5_core":  "logs/T5_core_alpha_lightgbm.txt"
}

PIP_SIZE = 0.0001
PIP_USD = 10.0
SL_PIPS, TP_PIPS = 10, 10
FEES = 0.0
OUT_CSV = "logs/stage5_backtest_results.csv"
OUT_TXT = "logs/stage5_summary.txt"
os.makedirs("logs", exist_ok=True)

# ========== HELPERS ==========
def load_price(path):
    df = pd.read_parquet(path)
    df["close"] = df["mid_c"]
    df = df.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def atr_from_ta(df, window=14):
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=window).average_true_range()
    return atr.ffill().bfill()

def expectancy_from_trades(rec, pip_size=PIP_SIZE):
    if len(rec) == 0: return 0
    entry, exitp, direction = rec["entry_price"], rec["exit_price"], rec["direction"]
    sign = np.where(direction == 0, 1, -1)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def run_backtest(price, signal):
    long_e  = signal.eq(2)
    long_x  = ~signal.eq(2)
    short_e = signal.eq(0)
    short_x = ~signal.eq(0)

    tp = pd.Series(TP_PIPS * PIP_SIZE, index=price.index)
    sl = pd.Series(SL_PIPS * PIP_SIZE, index=price.index)

    pf = vbt.Portfolio.from_signals(
        price["close"], entries=long_e, exits=long_x,
        short_entries=short_e, short_exits=short_x,
        tp_stop=tp, sl_stop=sl, size=1.0, fees=FEES, freq="5min"
    )
    stats = pf.stats()
    exp_pips = expectancy_from_trades(pf.trades.records)
    return pf, stats, exp_pips

# ========== MAIN ==========
print("⏳ Loading data ...")
price = load_price(DATA_FILE)
X = pd.read_csv(FEATURE_FILE)
y = pd.read_csv(Y_FILE)["y"].astype(int)

print(f"✅ Loaded price={len(price):,} | features={X.shape} | y={len(y):,}")

sample_idx = np.arange(200, len(price), 5, dtype=int)
X = X.iloc[:len(sample_idx)]
sample_times = price.index[sample_idx[:len(X)]]

rows = []
for name, path in MODELS.items():
    print(f"\n🚀 Predicting with {name} ...")
    model = lgb.Booster(model_file=path)
    feat_names = model.feature_name()
    Xsub = X.reindex(columns=feat_names, fill_value=0)
    probs = model.predict(Xsub.values)
    preds = np.argmax(probs, axis=1)
    signal = pd.Series(preds, index=sample_times).reindex(price.index, method="ffill").fillna(1).astype(int)

    pf, stats, exp_pips = run_backtest(price, signal)

    def _get(s, k):
        v = s.get(k, np.nan)
        try: return float(v)
        except: return float(np.asarray(v)[0]) if hasattr(v, "values") else np.nan

    trades = int(_get(stats, "Total Trades"))
    winrate = _get(stats, "Win Rate [%]")
    pfactor = _get(stats, "Profit Factor")
    ret = _get(stats, "Total Return [%]")
    dd = _get(stats, "Max Drawdown [%]")
    exp_usd = exp_pips * PIP_USD

    print(f"{name}: Trades={trades} | Win%={winrate:.2f} | PF={pfactor:.2f} | Ret%={ret:.2f} | DD%={dd:.2f} | Exp={exp_pips:.2f} pips (${exp_usd:.2f})")

    rows.append(dict(
        model=name, trades=trades, winrate=winrate,
        profit_factor=pfactor, total_return=ret,
        max_dd=dd, expectancy_pips=exp_pips,
        expectancy_usd=exp_usd
    ))

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w") as f:
    f.write("========== Stage 5 Backtest Summary ==========\n")
    f.write(res.to_string(index=False))
print(f"\n✅ Saved results → {OUT_CSV}")
print(f"✅ Summary saved → {OUT_TXT}")
print("🎯 Stage 5 complete.")