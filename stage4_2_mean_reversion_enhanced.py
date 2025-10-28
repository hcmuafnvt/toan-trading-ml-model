# stage4_2_mean_reversion_enhanced.py
# Mean Reversion + RSI + ATR regime filter (Low-vol only)

import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ========== CONFIG ==========
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FINAL_SIGNAL_FILE = "logs/stage4_final_signal.csv"

OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

BB_WINDOW = 30
BB_NSTD   = 1.6
RSI_WINDOW = 14
RSI_LOW, RSI_HIGH = 35, 65
ATR_WINDOW = 14
LOW_Q = 0.33  # chỉ lấy regime thấp

RR_GRID = [
    (1.5, 1.0),
    (2.0, 1.0),
    (2.0, 1.5),
    (2.5, 1.5)
]

OUT_GRID = "logs/stage4_2_meanrev_enhanced_grid.csv"
OUT_SUMMARY = "logs/stage4_2_meanrev_enhanced_summary.txt"
OUT_BEST_SIG = "logs/stage4_2_meanrev_enhanced_best_signal.csv"
os.makedirs("logs", exist_ok=True)


# ========== HELPERS ==========
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
        keep = req + (["volume"] if "volume" in df.columns else [])
        df = df[keep]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def bollinger_bands(close, window, nstd):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return ma, ma + nstd*sd, ma - nstd*sd

def expectancy_from_trades(trades):
    if trades is None or len(trades)==0: return 0.0
    entry = trades.get("Avg Entry Price") or trades.get("entry_price")
    exitp = trades.get("Avg Exit Price") or trades.get("exit_price")
    direction = trades.get("Direction") or trades.get("direction")
    if entry is None or exitp is None: return 0.0
    sign = np.where(pd.Series(direction).astype(str).str.lower().str.contains("long"),1,-1)
    pips = (exitp-entry)*sign/PIP_SIZE
    return float(np.nanmean(pips))

# ========== MAIN ==========
if __name__ == "__main__":
    print("⏳ Loading price & signal...")
    price = load_price(DATA_FILE)
    close = price["close"]

    signal = pd.read_csv(FINAL_SIGNAL_FILE, index_col=0, parse_dates=True)
    signal.index = signal.index.tz_localize(None)
    signal = signal.reindex(price.index, method="ffill").fillna(1).astype(int)
    timeout_mask = signal.eq(1)

    # Indicators
    sma, up, lo = bollinger_bands(close, BB_WINDOW, BB_NSTD)
    rsi = RSIIndicator(close, window=RSI_WINDOW).rsi()
    atr = AverageTrueRange(price["high"], price["low"], close, window=ATR_WINDOW).average_true_range()
    atr_q = atr.quantile(LOW_Q)
    low_regime = atr < atr_q

    # Entries (only timeout + low regime)
    long_entry  = (close < lo) & (rsi < RSI_LOW) & timeout_mask & low_regime
    short_entry = (close > up) & (rsi > RSI_HIGH) & timeout_mask & low_regime
    exit_sig = ((close >= sma) | (close <= sma)).fillna(False)

    print("✅ Entries generated:")
    print(pd.Series({
        "long_entry": int(long_entry.sum()),
        "short_entry": int(short_entry.sum()),
        "exit": int(exit_sig.sum())
    }))

    rows, best, best_series = [], None, None
    for tp_mult, sl_mult in RR_GRID:
        tp_stop = atr * tp_mult
        sl_stop = atr * sl_mult
        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entry,
            exits=exit_sig,
            short_entries=short_entry,
            short_exits=exit_sig,
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            size=1.0,
            fees=FEES,
            freq="5min"
        )
        stats = pf.stats()
        trades = pf.trades.records

        trades_n = int(stats.get("Total Trades", 0))
        win = float(stats.get("Win Rate [%]", 0))
        pfac = float(stats.get("Profit Factor", 0))
        exp_pips = expectancy_from_trades(trades)
        exp_usd = exp_pips * PIP_USD

        rows.append({
            "TPxATR": tp_mult, "SLxATR": sl_mult, "Trades": trades_n,
            "Win Rate [%]": win, "Profit Factor": pfac,
            "Expectancy (pips)": exp_pips, "Expectancy (USD_1lot)": exp_usd
        })

        print(f"[TP={tp_mult}×ATR | SL={sl_mult}×ATR] Trades={trades_n} | "
              f"Win={win:.2f}% | PF={pfac:.2f} | Exp={exp_pips:.2f}p (${exp_usd:.2f})")

        key = (pfac, exp_pips)
        if best is None or key > best[0]:
            best = (key, tp_mult, sl_mult, trades_n, win, pfac, exp_pips, exp_usd)
            sig = pd.Series(1, index=price.index)
            sig[long_entry] = 2
            sig[short_entry] = 0
            best_series = sig.copy()

    grid = pd.DataFrame(rows).sort_values(["Profit Factor","Expectancy (pips)"], ascending=[False,False])
    grid.to_csv(OUT_GRID, index=False)

    if best:
        (_, tp, sl, ntr, wr, pfac, expp, expu) = best
        with open(OUT_SUMMARY,"w") as f:
            f.write("========== MEAN REV ENHANCED (LOW ATR) ==========\n")
            f.write(grid.to_string(index=False))
            f.write(f"\n\nBest: TP={tp}×ATR SL={sl}×ATR | Trades={ntr} | Win={wr:.2f}% "
                    f"| PF={pfac:.2f} | Exp={expp:.2f}p (${expu:.2f})\n")
        best_series.name="meanrev_enhanced_signal"
        best_series.to_csv(OUT_BEST_SIG)
        print(f"\n✅ Saved grid → {OUT_GRID}")
        print(f"✅ Saved summary → {OUT_SUMMARY}")
        print(f"✅ Saved best signal → {OUT_BEST_SIG}")
    else:
        print("\n⚠️ No valid trades.")