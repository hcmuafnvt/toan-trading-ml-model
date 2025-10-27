"""
STAGE 1.5 — REALISTIC TARGET DISCOVERY ENGINE
----------------------------------------------
Mục tiêu:
 - Label dữ liệu theo 3 trạng thái: BUY (+1), SELL (-1), TIMEOUT (0)
 - TP/SL 1:1  (mặc định)
 - Session-aware: London + NewYork cộng thêm 5 pips vào TP/SL
 - Không tính PnL timeout (vì ta không vào lệnh)
 - Kết quả: thống kê BUY/SELL/TIMEOUT, total pips và heatmap BUY vs SELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
FILE = "GBP_USD_M5_2024.parquet"
PAIR = "GBP/USD"
PIP = 0.0001
TP_LIST = [5, 10, 15, 20, 25, 30]
AHEAD_LIST = [5, 10, 20, 40, 60, 80]
SESSION_BONUS = 5  # +5 pips TP/SL cho London & NewYork
RR = 1.0            # TP : SL = 1:1

# ========== LOAD DATA ==========
df = pd.read_parquet(FILE)
if "mid_c" in df.columns:
    df["close"] = df["mid_c"]
if "mid_h" in df.columns:
    df["high"] = df["mid_h"]
if "mid_l" in df.columns:
    df["low"] = df["mid_l"]
df = df.dropna(subset=["high","low","close"]).copy()
df["hour"] = df.index.hour

# Detect session (UTC)
def detect_session(h):
    if 7 <= h < 15:
        return "London"
    elif 12 <= h < 21:
        return "NewYork"
    else:
        return "Asia"
df["session"] = df["hour"].map(detect_session)

print(f"✅ Loaded {len(df):,} rows from {FILE}")
print(df[["close","session"]].head(3))

# ========== LABEL FUNCTION ==========
def label_buy_sell(df, tp_pips=10, ahead=20, rr=1.0, pip=0.0001, bonus=0):
    n = len(df)
    res = np.zeros(n)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    sessions = df["session"].values

    for i in range(n - ahead):
        # Session adjustment
        tp_adj = tp_pips + (bonus if sessions[i] in ["London","NewYork"] else 0)
        tp = tp_adj * pip
        sl = tp * rr

        entry = closes[i]
        tp_up = entry + tp
        sl_down = entry - sl
        tp_down = entry - tp
        sl_up = entry + sl

        sub_high = highs[i+1:i+ahead+1]
        sub_low  = lows[i+1:i+ahead+1]

        hit_tp_up = np.where(sub_high >= tp_up)[0]
        hit_sl_down = np.where(sub_low <= sl_down)[0]
        hit_tp_down = np.where(sub_low <= tp_down)[0]
        hit_sl_up = np.where(sub_high >= sl_up)[0]

        # First-touch BUY
        if len(hit_tp_up)==0 and len(hit_sl_down)==0:
            buy_touch = np.inf
        elif len(hit_tp_up)==0:
            buy_touch = np.inf
        elif len(hit_sl_down)==0:
            buy_touch = hit_tp_up[0]
        else:
            buy_touch = hit_tp_up[0] if hit_tp_up[0] < hit_sl_down[0] else np.inf

        # First-touch SELL
        if len(hit_tp_down)==0 and len(hit_sl_up)==0:
            sell_touch = np.inf
        elif len(hit_tp_down)==0:
            sell_touch = np.inf
        elif len(hit_sl_up)==0:
            sell_touch = hit_tp_down[0]
        else:
            sell_touch = hit_tp_down[0] if hit_tp_down[0] < hit_sl_up[0] else np.inf

        if buy_touch == np.inf and sell_touch == np.inf:
            res[i] = 0          # timeout
        elif buy_touch < sell_touch:
            res[i] = 1          # BUY
        elif sell_touch < buy_touch:
            res[i] = -1         # SELL
        else:
            res[i] = 0          # tie fallback

    return res

# ========== GRID SEARCH ==========
results = []
for tp in TP_LIST:
    for ahead in AHEAD_LIST:
        label = label_buy_sell(df, tp_pips=tp, ahead=ahead,
                               rr=RR, pip=PIP, bonus=SESSION_BONUS)
        buy_n = np.sum(label == 1)
        sell_n = np.sum(label == -1)
        timeout_n = np.sum(label == 0)

        buy_pips = buy_n * tp
        sell_pips = sell_n * tp
        total_pips = buy_pips + sell_pips

        results.append({
            "tp_pips": tp,
            "ahead": ahead,
            "buy_count": buy_n,
            "sell_count": sell_n,
            "timeout_count": timeout_n,
            "buy_total_pips": buy_pips,
            "sell_total_pips": sell_pips,
            "total_pips": total_pips
        })

res_df = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(res_df.sort_values("total_pips", ascending=False).head(10))

# ========== HEATMAPS ==========
def plot_heatmap(metric, title):
    pivot = res_df.pivot(index="tp_pips", columns="ahead", values=metric)
    plt.figure(figsize=(8,5))
    plt.title(f"{PAIR} — {title}")
    plt.xlabel("AHEAD (candles)")
    plt.ylabel("TP target (pips)")
    plt.imshow(pivot, cmap="YlGn", aspect="auto", origin="lower")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.show()

plot_heatmap("buy_total_pips", "BUY total pips by TP/AHEAD")
plot_heatmap("sell_total_pips", "SELL total pips by TP/AHEAD")
