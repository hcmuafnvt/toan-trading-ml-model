# stage4_1_mean_reversion.py
#  ├─ load price + final_signal.csv (từ stage4)
#  ├─ detect sideway zones (signal == 1)
#  ├─ tính EMA50 + ATR
#  ├─ rule: dist_from_ema > ±1.5 × ATR → vào lệnh ngược
#  ├─ TP = 1 × ATR, SL = 1 × ATR
#  ├─ backtest nhanh bằng vectorbt
#  └─ log PF, Win%, Expectancy (pips, USD)

# ==========================================
# STAGE 4.1 — MEAN REVERSION FOR TIMEOUTS
# ==========================================
import pandas as pd
import numpy as np
import vectorbt as vbt
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

PAIR = "GBP_USD"
PRICE_FILE = f"data/{PAIR}_M5_2024.parquet"
SIGNAL_FILE = "logs/stage4_final_signal.csv"   # nếu Stage 4 lưu signal
PIP_SIZE = 0.0001
PIP_USD  = 10.0

print("⏳ Loading data...")
price = pd.read_parquet(PRICE_FILE)
price["close"] = price["mid_c"]
price = price[["close","mid_h","mid_l","mid_o","volume"]].rename(
    columns={"mid_h":"high","mid_l":"low","mid_o":"open"}
)

signal = pd.read_csv(SIGNAL_FILE, index_col=0, parse_dates=True)

# --- Fix timezone mismatch ---
price.index = price.index.tz_localize(None)
signal.index = signal.index.tz_localize(None)

signal = signal.reindex(price.index, method="ffill")

# --- Mean Reversion only when timeout (1) ---
mask = signal["final_signal"] == 1
df = price.loc[mask].copy()
print(f"Timeout samples: {len(df)}")

# --- Indicators ---
df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
df["atr"]   = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()

# --- Reversion signals ---
dist = (df["close"] - df["ema50"]) / df["atr"]
df["long_entry"]  = (dist < -1.5)
df["short_entry"] = (dist >  1.5)
df["exit"] = (dist.abs() < 0.3)

print("✅ Entries generated:")
print(df[["long_entry","short_entry","exit"]].sum())

# --- Backtest ---
pf = vbt.Portfolio.from_signals(
    close=df["close"],
    entries=df["long_entry"],
    exits=df["exit"],
    short_entries=df["short_entry"],
    short_exits=df["exit"],
    size=1.0,
    fees=0.0,
    direction="both",
    freq="5min"
)

stats = pf.stats()
trades = pf.trades.records_readable

# --- Extract trade records safely (handle both old/new vectorbt schemas) ---
cols = [c.lower() for c in trades.columns]
colmap = {c.lower(): c for c in trades.columns}

def get_col(*names):
    for n in names:
        if n.lower() in colmap:
            return trades[colmap[n.lower()]].values
    return None

entry = get_col("entry_price", "avg entry price")
exitp = get_col("exit_price", "avg exit price")

if entry is None or exitp is None:
    print("⚠️ entry_price / exit_price columns not found, skipping pips calc")
    entry = np.zeros(len(trades))
    exitp = np.zeros(len(trades))

direction = get_col("direction")
if direction is None:
    sign = np.ones(len(entry))
else:
    if trades[colmap["direction"]].dtype == object:
        direction_series = pd.Series(direction.astype(str))
        sign = np.where(direction_series.str.lower().str.contains("long"), 1.0, -1.0)
    else:
        sign = np.where(direction == 0, 1.0, -1.0)

min_len = min(len(entry), len(exitp), len(sign))
entry, exitp, sign = entry[:min_len], exitp[:min_len], sign[:min_len]

pips = (exitp - entry) * sign / PIP_SIZE

expect_pip = float(np.mean(pips))
expect_usd = expect_pip * PIP_USD
pf_ratio = stats["Profit Factor"]
winrate = stats["Win Rate [%]"]

summary = f"""========== MEAN REVERSION RESULT ==========
Trades={len(trades)}
Win%={winrate:.2f}
PF={pf_ratio:.2f}
Expectancy={expect_pip:.2f} pips (${expect_usd:.2f})
============================================
"""
print(summary)

with open("logs/stage4_1_meanreversion_summary.txt","w") as f:
    f.write(summary)
trades.to_csv("logs/stage4_1_meanreversion_results.csv",index=False)
print("✅ Saved results to logs/")