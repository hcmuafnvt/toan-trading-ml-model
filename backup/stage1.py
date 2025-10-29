"""
STAGE 1.5 — REALISTIC TARGET DISCOVERY ENGINE (UPDATED)
-------------------------------------------------------
Mục tiêu:
 - Label dữ liệu theo 3 trạng thái: BUY (+1), SELL (-1), TIMEOUT (0)
 - TP/SL 1:1 (mặc định)
 - Session-aware: London + NewYork cộng thêm 5 pips vào TP/SL
 - Không tính PnL timeout (vì ta không vào lệnh)
 - Kết quả: thống kê BUY/SELL/TIMEOUT, total pips và heatmap BUY vs SELL
 - ✅ Tự động export logs/stage3_y.csv (cho Stage 3 & 4L)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
PAIR = "GBP_USD"
FILE = f"data/{PAIR}_M5_2024.parquet"
OUT_Y = "logs/stage3_y.csv"
os.makedirs("logs", exist_ok=True)

PIP = 0.0001
TP_LIST = [5, 10, 15, 20, 25, 30]
AHEAD_LIST = [5, 10, 20, 40, 60, 80]
SESSION_BONUS = 5
RR = 1.0  # TP:SL = 1:1

# ========== LOAD DATA ==========
df = pd.read_parquet(FILE)
if "mid_c" in df.columns:
    df["close"] = df["mid_c"]
if "mid_h" in df.columns:
    df["high"] = df["mid_h"]
if "mid_l" in df.columns:
    df["low"] = df["mid_l"]
df = df.dropna(subset=["high", "low", "close"]).copy()
df["hour"] = df.index.hour

def detect_session(h):
    if 7 <= h < 15:
        return "London"
    elif 12 <= h < 21:
        return "NewYork"
    else:
        return "Asia"
df["session"] = df["hour"].map(detect_session)

print(f"✅ Loaded {len(df):,} rows from {FILE}")

# ========== LABEL FUNCTION ==========
def label_buy_sell(df, tp_pips=10, ahead=20, rr=1.0, pip=0.0001, bonus=0):
    n = len(df)
    res = np.zeros(n)
    highs, lows, closes, sessions = (
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["session"].values,
    )

    for i in range(n - ahead):
        tp_adj = tp_pips + (bonus if sessions[i] in ["London", "NewYork"] else 0)
        tp = tp_adj * pip
        sl = tp * rr

        entry = closes[i]
        tp_up, sl_down = entry + tp, entry - sl
        tp_down, sl_up = entry - tp, entry + sl

        sub_high, sub_low = highs[i + 1 : i + ahead + 1], lows[i + 1 : i + ahead + 1]

        hit_tp_up = np.where(sub_high >= tp_up)[0]
        hit_sl_down = np.where(sub_low <= sl_down)[0]
        hit_tp_down = np.where(sub_low <= tp_down)[0]
        hit_sl_up = np.where(sub_high >= sl_up)[0]

        buy_touch = np.inf
        if len(hit_tp_up) > 0 and (len(hit_sl_down) == 0 or hit_tp_up[0] < hit_sl_down[0]):
            buy_touch = hit_tp_up[0]

        sell_touch = np.inf
        if len(hit_tp_down) > 0 and (len(hit_sl_up) == 0 or hit_tp_down[0] < hit_sl_up[0]):
            sell_touch = hit_tp_down[0]

        if buy_touch == np.inf and sell_touch == np.inf:
            res[i] = 1
        elif buy_touch < sell_touch:
            res[i] = 2
        elif sell_touch < buy_touch:
            res[i] = 0
        else:
            res[i] = 1
            
        if i == n - ahead - 1:
            print("Unique labels generated:", np.unique(res))
    
    return res

# ========== GRID SEARCH ==========
results = []
for tp in TP_LIST:
    for ahead in AHEAD_LIST:
        label = label_buy_sell(df, tp_pips=tp, ahead=ahead, rr=RR, pip=PIP, bonus=SESSION_BONUS)
        buy_n, sell_n, timeout_n = (np.sum(label == 1), np.sum(label == -1), np.sum(label == 0))
        total_pips = (buy_n + sell_n) * tp
        results.append(
            dict(tp_pips=tp, ahead=ahead,
                 buy_count=buy_n, sell_count=sell_n, timeout_count=timeout_n,
                 total_pips=total_pips)
        )

res_df = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(res_df.sort_values("total_pips", ascending=False).head(10))

# ========== EXPORT TARGET ==========
best_row = res_df.sort_values("total_pips", ascending=False).iloc[0]
best_tp, best_ahead = int(best_row.tp_pips), int(best_row.ahead)
print(f"\n✅ Best combination → TP={best_tp} pips | AHEAD={best_ahead}")

# Tạo vector y theo best TP/AHEAD
y = label_buy_sell(df, tp_pips=best_tp, ahead=best_ahead,
                   rr=RR, pip=PIP, bonus=SESSION_BONUS)
pd.Series(y, name="y").to_csv(OUT_Y, index=False)
print(f"✅ Exported target vector to {OUT_Y} | len={len(y)} | unique={np.unique(y)}")

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

plot_heatmap("buy_count", "BUY count by TP/AHEAD")
plot_heatmap("sell_count", "SELL count by TP/AHEAD")