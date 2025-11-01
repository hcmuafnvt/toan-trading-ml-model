# ==========================================
# TP/SL BACKTEST — dùng pandas_ta
# ==========================================
# ✅ Chạy ổn trên macOS / EC2 (Python 3.12)
# ✅ Output: text PF, Winrate, Pips, Trades
# ==========================================

import pandas as pd
import numpy as np
import pandas_ta as ta

# ---------- CONFIG ----------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_std.parquet"
PIP_SIZE = 0.0001  # non-JPY
TP_LIST = [5, 10, 15, 20, 25, 30]
SL_LIST = [5, 10, 15, 20, 25, 30]

# ---------- LOAD DATA ----------
df = pd.read_parquet(DATA_FILE).copy()
df = df.sort_index()

# ---------- ENTRY RULE ----------
df["ema50"] = ta.ema(df["close"], 50)
df["long_signal"] = df["close"] > df["ema50"]
df["short_signal"] = df["close"] < df["ema50"]

# ---------- CORE BACKTEST ----------
def run_backtest(df, tp_pips, sl_pips):
    tp = tp_pips * PIP_SIZE
    sl = sl_pips * PIP_SIZE
    trades = []
    in_pos = False
    direction = None
    entry_price = entry_time = tp_level = sl_level = None

    for t, row in df.iterrows():
        high, low, close = row["high"], row["low"], row["close"]

        if not in_pos:
            if row["long_signal"]:
                in_pos = True
                direction = "long"
                entry_price = close
                entry_time = t
                tp_level = entry_price + tp
                sl_level = entry_price - sl
            elif row["short_signal"]:
                in_pos = True
                direction = "short"
                entry_price = close
                entry_time = t
                tp_level = entry_price - tp
                sl_level = entry_price + sl
            continue

        if direction == "long":
            if high >= tp_level:
                pnl = (tp_level - entry_price) / PIP_SIZE
                trades.append((entry_time, t, direction, pnl))
                in_pos = False
            elif low <= sl_level:
                pnl = (sl_level - entry_price) / PIP_SIZE
                trades.append((entry_time, t, direction, pnl))
                in_pos = False
        elif direction == "short":
            if low <= tp_level:
                pnl = (entry_price - tp_level) / PIP_SIZE
                trades.append((entry_time, t, direction, pnl))
                in_pos = False
            elif high >= sl_level:
                pnl = (entry_price - sl_level) / PIP_SIZE
                trades.append((entry_time, t, direction, pnl))
                in_pos = False

    return pd.DataFrame(trades, columns=["entry", "exit", "dir", "pips"])

# ---------- STATISTICS ----------
def calc_stats(trades_df):
    if trades_df.empty:
        return {"PF": np.nan, "Winrate%": np.nan, "Total pips": 0, "Trades": 0}
    win = trades_df["pips"] > 0
    gross_profit = trades_df.loc[win, "pips"].sum()
    gross_loss = -trades_df.loc[~win, "pips"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
    winrate = win.mean() * 100
    return {"PF": pf, "Winrate%": winrate,
            "Total pips": trades_df["pips"].sum(),
            "Trades": len(trades_df)}

# ---------- MAIN LOOP ----------
results = []
for tp_pips in TP_LIST:
    for sl_pips in SL_LIST:
        trades = run_backtest(df, tp_pips, sl_pips)
        long_trades = trades[trades["dir"] == "long"]
        short_trades = trades[trades["dir"] == "short"]
        results.append({"TP": tp_pips, "SL": sl_pips,
                        "Direction": "long", **calc_stats(long_trades)})
        results.append({"TP": tp_pips, "SL": sl_pips,
                        "Direction": "short", **calc_stats(short_trades)})

res_df = pd.DataFrame(results)

# ---------- OUTPUT ----------
print("\n==================== TP/SL Backtest Summary ====================")
print(res_df.to_string(index=False))
print("\n==================== Top 10 by Profit Factor ====================")
print(res_df.sort_values("PF", ascending=False).head(10).to_string(index=False))