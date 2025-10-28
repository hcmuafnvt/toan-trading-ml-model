# stage4_1_mean_reversion_v2.py
# Grid-search mean reversion cho vùng TIMEOUT (signal==1) từ Stage 4
# - Entry rule: Bollinger(50, 2.0)
#     Long  khi close < lower band  (quá bán)  → mean reversion lên SMA
#     Short khi close > upper band  (quá mua)  → mean reversion xuống SMA
#     Exit  khi chạm SMA hoặc sau max_bars
# - TP/SL cố định theo pips (không ATR) để test R:R
# - Kết quả: Win%, PF, Expectancy (pips & USD), Trades
# - Lưu: logs/stage4_1_meanrev_grid.csv, logs/stage4_1_meanrev_summary.txt
#        và logs/stage4_1_meanrev_best_signal.csv (series 0/1/2)

import os
import numpy as np
import pandas as pd
import vectorbt as vbt

# ================== CONFIG ==================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FINAL_SIGNAL_FILE = "logs/stage4_final_signal.csv"  # từ Stage 4

# Mapping schema parquet (OANDA mid_*)
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# Pip & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# Mean reversion params
BB_WINDOW = 50
BB_NSTD   = 2.0
MAX_BARS  = 24   # thoát tối đa sau 24 nến (~2h) nếu chưa về SMA

# Grid R:R: SL cố định 10 pips, TP thay đổi
SL_PIPS = 10.0
TP_LIST = [5.0, 10.0, 15.0, 20.0]  # → R:R = 0.5, 1.0, 1.5, 2.0

OUT_GRID_CSV = "logs/stage4_1_meanrev_grid.csv"
OUT_SUMMARY  = "logs/stage4_1_meanrev_summary.txt"
OUT_BEST_SIG = "logs/stage4_1_meanrev_best_signal.csv"

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
    # đồng bộ tz-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def bollinger_bands(close: pd.Series, window=20, nstd=2.0):
    close = pd.Series(close).astype(float)   # 🔒 luôn ép về 1D Series
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    return ma, upper, lower

def expectancy_from_trades(trades: pd.DataFrame) -> float:
    """Tính kỳ vọng theo pips, tương thích schema cũ/mới của vectorbt."""
    if trades is None or len(trades) == 0:
        return 0.0

    cols = [c.lower() for c in trades.columns]
    colmap = {c.lower(): c for c in trades.columns}

    def get_col(*names):
        for n in names:
            if n.lower() in colmap:
                return trades[colmap[n.lower()]].values
        return None

    entry = get_col("entry_price", "avg entry price")
    exitp = get_col("exit_price", "avg exit price")
    direction = get_col("direction")

    if entry is None or exitp is None:
        # không đủ cột -> không tính expectancy
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

    entry, exitp, sign = entry[:n], exitp[:n], sign[:n]
    pips = (exitp - entry) * sign / PIP_SIZE
    return float(np.nanmean(pips))

def profit_usd_from_trades(trades: pd.DataFrame) -> float:
    if trades is None or len(trades) == 0:
        return 0.0

    cols = [c.lower() for c in trades.columns]
    colmap = {c.lower(): c for c in trades.columns}

    def get_col(*names):
        for n in names:
            if n.lower() in colmap:
                return trades[colmap[n.lower()]].values
        return None

    entry = get_col("entry_price", "avg entry price")
    exitp = get_col("exit_price", "avg exit price")
    direction = get_col("direction")
    size = get_col("size")
    if size is None:
        size = np.ones(len(trades))

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

    n = min(len(entry), len(exitp), len(sign), len(size))
    if n == 0:
        return 0.0

    entry, exitp, sign, size = entry[:n], exitp[:n], sign[:n], size[:n]
    pips = (exitp - entry) * sign / PIP_SIZE
    usd = pips * PIP_USD * size
    return float(np.nansum(usd))


# ================== MAIN ==================
if __name__ == "__main__":
    print("⏳ Loading data...")
    price = load_price(DATA_FILE)
    close = price["close"].squeeze()       # ép thành Series 1D

    # === Load final fusion signal (stage 4)
    signal = pd.read_csv(FINAL_SIGNAL_FILE, index_col=0, parse_dates=True)
    # đồng bộ tz-naive
    signal.index = signal.index.tz_localize(None)
    # align theo price
    signal = signal.reindex(price.index, method="ffill").fillna(1).astype(int)

    # === Tạo timeout mask sau khi signal đã load xong
    timeout_mask = signal.eq(1).squeeze()  # ✅ tạo mask & ép Series 1D

    print(f"Timeout samples: {int(timeout_mask.sum())}")

    # === Mean reversion entries based on Bollinger
    sma, up, lo = bollinger_bands(close, BB_WINDOW, BB_NSTD)

    # Entry:
    long_entry  = (close < lo) & timeout_mask
    short_entry = (close > up) & timeout_mask

    # Exit khi chạm SMA hoặc sau MAX_BARS
    cross_up   = (close >= sma)
    base_exit  = cross_up.fillna(False)
    long_exit_time  = long_entry.shift(MAX_BARS, fill_value=False)
    short_exit_time = short_entry.shift(MAX_BARS, fill_value=False)
    exit_sig = (base_exit | long_exit_time | short_exit_time).fillna(False)

    print("✅ Entries generated:")
    print(pd.Series({
        "long_entry": int(long_entry.sum()),
        "short_entry": int(short_entry.sum()),
        "exit": int(exit_sig.sum())
    }))

    # === Grid-search các TP
    rows = []
    best = None
    best_series = None

    for TP_PIPS in TP_LIST:
        tp_val = TP_PIPS * PIP_SIZE
        sl_val = SL_PIPS * PIP_SIZE

        # Backtest
        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entry,
            exits=exit_sig,
            short_entries=short_entry,
            short_exits=exit_sig,
            tp_stop=tp_val,
            sl_stop=sl_val,
            size=1.0,
            fees=FEES,
            freq="5min"
        )
        stats = pf.stats()
        trades = pf.trades.records

        # Tránh crash nếu thiếu metric trong version vectorbt
        def _get(s, k):
            v = s.get(k, np.nan)
            try:
                return float(v)
            except Exception:
                if hasattr(v, "values"):
                    return float(np.asarray(v)[0])
                return float("nan")

        trades_n   = int(_get(stats, "Total Trades"))
        win_rate   = _get(stats, "Win Rate [%]")
        pfactor    = _get(stats, "Profit Factor")
        exp_pips   = expectancy_from_trades(trades)
        exp_usd    = exp_pips * PIP_USD
        rr_label   = f"{TP_PIPS:.1f}:{SL_PIPS:.1f}"  # TP:SL in pips
        rr_ratio   = TP_PIPS / SL_PIPS if SL_PIPS > 0 else np.nan

        rows.append({
            "TP_pips": TP_PIPS,
            "SL_pips": SL_PIPS,
            "R:R": rr_ratio,
            "Trades": trades_n,
            "Win Rate [%]": win_rate,
            "Profit Factor": pfactor,
            "Expectancy (pips)": exp_pips,
            "Expectancy (USD_1lot)": exp_usd
        })

        print(f"[RR={rr_label}] Trades={trades_n} | Win%={win_rate:.2f} | PF={pfactor:.2f} "
              f"| Exp={exp_pips:.2f}p (${exp_usd:.2f})")

        # chọn best theo PF trước, tie-break bằng Expectancy
        key = (pfactor, exp_pips)
        if (best is None) or (key > best[0]):
            best = (key, TP_PIPS, trades_n, win_rate, pfactor, exp_pips, exp_usd)
            # lưu series 0/1/2: 2=buy, 0=sell, 1=timeout
            sig_series = pd.Series(1, index=price.index, dtype=int)
            sig_series[long_entry] = 2
            sig_series[short_entry] = 0
            best_series = sig_series.copy()

    # === Save grid & summary
    grid_df = pd.DataFrame(rows).sort_values(
        ["Profit Factor","Expectancy (pips)","Trades"], ascending=[False, False, False]
    )
    grid_df.to_csv(OUT_GRID_CSV, index=False)

    if best is not None:
        (_, TP_BEST, ntr, wr, pfac, expp, expu) = best
        with open(OUT_SUMMARY, "w") as f:
            f.write("========== MEAN REVERSION GRID (TIMEOUT) ==========\n")
            f.write(grid_df.to_string(index=False))
            f.write("\n\n========== BEST CONFIG ==========\n")
            f.write(f"TP={TP_BEST:.1f} pips | SL={SL_PIPS:.1f} pips | R:R={TP_BEST/SL_PIPS:.2f}\n")
            f.write(f"Trades={ntr} | Win%={wr:.2f} | PF={pfac:.2f} | "
                    f"Expectancy={expp:.2f} pips (${expu:.2f})\n")

        # Lưu signal của best cấu hình (để merge portfolio sau này)
        if best_series is not None:
            best_series.name = "meanrev_signal"
            best_series.to_csv(OUT_BEST_SIG, index=True)

        print("\n✅ Saved grid →", OUT_GRID_CSV)
        print("✅ Saved summary →", OUT_SUMMARY)
        print("✅ Saved best signal →", OUT_BEST_SIG)
    else:
        print("\n⚠️ No trades generated in grid. Nothing saved.")