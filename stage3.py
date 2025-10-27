# =============================================================
# STAGE 3.2 — REGIME ADAPTIVE FUSION
#  - Uses DXY, VIX, ATR-based volatility regime detection
#  - Adaptive TP/SL by regime
#  - Baseline: Weighted + Vol stop (Stage 3.1)
# =============================================================

import pandas as pd
import numpy as np
import vectorbt as vbt
import lightgbm as lgb
from ta.volatility import AverageTrueRange
from datetime import timedelta

# ---------- CONFIG ----------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
REGIME_FILE = "data/regime_dxy_vix.csv"
MODEL_PATHS = {
    "T1": "logs/T1_10x40_lightgbm.txt",
    "T2": "logs/T2_15x60_lightgbm.txt",
    "T3": "logs/T3_20x80_lightgbm.txt"
}
FEATURES_CSV = "logs/stage2_features.csv"

PIP_SIZE = 0.0001
PIP_USD = 10
BASE_TP = 20  # pips
BASE_SL = 20  # pips

# ---------- LOAD PRICE ----------
print(f"⏳ Loading price data {DATA_FILE} ...")
price = pd.read_parquet(DATA_FILE)
if "close" not in price.columns:
    raise ValueError("File must contain 'close' column.")

price = price.copy()
price["datetime"] = price.index
price = price.tz_localize(None)
print(f"✅ Loaded {len(price)} rows | {price.index.min()} → {price.index.max()}")

# ---------- ATR CALC ----------
atr = AverageTrueRange(high=price["close"], low=price["close"], close=price["close"], window=14).average_true_range()
atr_daily = atr.resample("1D").mean()
atr_norm = (atr_daily - atr_daily.min()) / (atr_daily.max() - atr_daily.min())

# ---------- LOAD DXY + VIX ----------
regime = pd.read_csv(REGIME_FILE, parse_dates=["Date"], index_col="Date")
regime = regime.rename(columns={"DXY_Change": "DXY_Change_pct"})
regime["DXY_norm"] = (regime["DXY_Change_pct"] - regime["DXY_Change_pct"].min()) / (regime["DXY_Change_pct"].max() - regime["DXY_Change_pct"].min())
regime["VIX_norm"] = (regime["VIX"] - regime["VIX"].min()) / (regime["VIX"].max() - regime["VIX"].min())

# ---------- MERGE REGIME + ATR ----------
regime_all = pd.concat([regime, atr_norm.rename("ATR_norm")], axis=1).ffill().dropna()
regime_all["regime_score"] = 0.5 * regime_all["ATR_norm"] + 0.3 * regime_all["VIX_norm"] + 0.2 * regime_all["DXY_norm"]

high_thresh = regime_all["regime_score"].quantile(0.66)
low_thresh = regime_all["regime_score"].quantile(0.33)
regime_all["regime_label"] = np.select(
    [regime_all["regime_score"] >= high_thresh,
     regime_all["regime_score"] <= low_thresh],
    ["HighVol", "LowVol"],
    default="Normal"
)

print(f"✅ Regime split done → HighVol>={high_thresh:.3f}, LowVol<={low_thresh:.3f}")
print(regime_all[["regime_score", "regime_label"]].tail())

# ---------- LOAD FEATURES + MODELS ----------
features = pd.read_csv(FEATURES_CSV, index_col=0)
X = features.drop(columns=["target"], errors="ignore")

models = {k: lgb.Booster(model_file=v) for k, v in MODEL_PATHS.items()}

# ---------- PREDICT SIGNALS ----------
probs = {k: models[k].predict(X) for k in models}
pred_df = pd.DataFrame(probs, index=X.index)
pred_df["fusion"] = (pred_df.mean(axis=1) * 2).round().clip(-1, 2)

# Align predictions to price
pred_df = pred_df.reindex(price.index, method="nearest").ffill()

# ---------- JOIN WITH REGIME ----------
merged = price.join(regime_all, how="left")
merged = merged.ffill()

merged["signal"] = pred_df["fusion"]

# ---------- DEFINE TP/SL BY REGIME ----------
def get_tp_sl(row):
    if row["regime_label"] == "HighVol":
        return 20, 20  # more room to run
    elif row["regime_label"] == "LowVol":
        return 10, 15  # tighter range
    else:
        return BASE_TP, BASE_SL

tp_sl = merged.apply(get_tp_sl, axis=1, result_type="expand")
merged["TP_pips"], merged["SL_pips"] = tp_sl[0], tp_sl[1]

# ---------- BACKTEST ----------
print("⏳ Running backtest (Weighted + Vol stop + Regime adaptive)...")

entries = merged["signal"] == 2  # BUY
exits   = merged["signal"] == 0  # SELL
tp_series = merged["TP_pips"] * PIP_SIZE
sl_series = merged["SL_pips"] * PIP_SIZE

pf = vbt.Portfolio.from_signals(
    merged["close"],
    entries=entries,
    exits=exits,
    tp_stop=tp_series,
    sl_stop=sl_series,
    direction="both",
    size=1.0
)

stats = pf.stats()
trades = pf.trades.records_readable

# ---------- SUMMARY BY REGIME ----------
trades["date"] = merged.index[trades["entry_idx"]]
trades["regime"] = trades["date"].dt.floor("D").map(regime_all["regime_label"])
summary = trades.groupby("regime")["pnl"].agg(["count", "mean", "sum"])
summary["winrate"] = (trades.groupby("regime")["pnl"].apply(lambda x: (x > 0).mean()) * 100)
summary["expectancy_pips"] = summary["mean"] / PIP_SIZE
summary["expectancy_usd"] = summary["expectancy_pips"] * PIP_USD

print("\n========== REGIME SUMMARY ==========")
print(summary)
print("\n========== OVERALL STATS ==========")
print(stats)

# ---------- SAVE ----------
summary.to_csv("logs/stage3_2_regime_summary.csv")
with open("logs/stage3_2_regime_summary.txt", "w") as f:
    f.write("========== REGIME SUMMARY ==========\n")
    f.write(summary.to_string())
    f.write("\n\n========== OVERALL STATS ==========\n")
    f.write(stats.to_string())

print("\n✅ Saved → logs/stage3_2_regime_summary.txt")