import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from ta.volatility import AverageTrueRange

# ================= CONFIG =================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILES = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}

WINDOW = 200
STRIDE = 5
PIP_SIZE = 0.0001
PIP_USD = 10
FEES = 0.0

FUSION_MODES = ["majority", "weighted", "buy_only"]
STOP_GRID = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("atr", 1.5, 1.5),
    ("atr", 2.0, 2.0),
]

OUT_TXT = "logs/stage3_1_summary.txt"
OUT_CSV = "logs/stage3_1_results.csv"
os.makedirs("logs", exist_ok=True)


# ================= HELPERS =================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return df


def load_price(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"]):
        df = df.rename(
            columns={
                "mid_o": "open",
                "mid_h": "high",
                "mid_l": "low",
                "mid_c": "close",
            }
        )
    elif all(c in df.columns for c in ["bid_o", "bid_h", "bid_l", "bid_c"]):
        df = df.rename(
            columns={
                "bid_o": "open",
                "bid_h": "high",
                "bid_l": "low",
                "bid_c": "close",
            }
        )
    else:
        raise ValueError("Không tìm thấy OHLC columns trong file parquet.")
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close"]]


def expectancy_from_trades(trades_rec, pip_size=PIP_SIZE) -> float:
    if trades_rec is None or len(trades_rec) == 0:
        return 0.0
    entry = trades_rec["entry_price"].values
    exitp = trades_rec["exit_price"].values
    direction = trades_rec["direction"].values  # 0 long, 1 short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))


def atr_from_ta(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Dùng ATR của thư viện ta, đảm bảo không lỗi duplicate index."""
    tmp = df[["high", "low", "close"]].copy()
    tmp = tmp.reset_index(drop=True)  # bỏ DatetimeIndex để tránh align bug
    atr = AverageTrueRange(tmp["high"], tmp["low"], tmp["close"], window=window).average_true_range()
    atr.index = df.index  # trả lại index gốc
    return atr.ffill().bfill()


def run_backtest(price, signal, stop_mode, tp_param, sl_param):
    long_entries = signal.eq(2)
    long_exits = ~signal.eq(2)
    short_entries = signal.eq(0)
    short_exits = ~signal.eq(0)

    if stop_mode == "fixed":
        tp = pd.Series(tp_param * PIP_SIZE, index=price.index)
        sl = pd.Series(sl_param * PIP_SIZE, index=price.index)
    elif stop_mode == "atr":
        atr = atr_from_ta(price, window=14)
        tp = atr * tp_param
        sl = atr * sl_param
    else:
        raise ValueError("Unknown stop mode")

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
    trades = pf.trades.records
    exp_pips = expectancy_from_trades(trades)
    return pf, stats, exp_pips


# ================= MAIN =================
print(f"⏳ Loading price: {DATA_FILE}")
price = load_price(DATA_FILE)
print(f"✅ Price loaded: {len(price):,} rows")

print(f"⏳ Loading features: {FEATURE_FILE}")
feat = pd.read_csv(FEATURE_FILE)
feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
print(f"✅ Features loaded: {feat.shape}")

target_cols = [c for c in feat.columns if c.startswith("target_")]
X_full = feat.drop(columns=target_cols, errors="ignore")
n_samples = len(X_full)
sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
sample_times = price.index[sample_idx[:n_samples]]

# Load model predictions
probs_dict, pred_dict = {}, {}
for name, path in MODEL_FILES.items():
    print(f"⏳ Loading model {name}")
    booster = lgb.Booster(model_file=path)
    feat_names = booster.feature_name()
    X = X_full.reindex(columns=feat_names, fill_value=0.0)
    probs = booster.predict(X.values, num_iteration=booster.best_iteration)
    preds = probs.argmax(axis=1)
    probs_dict[name] = probs
    pred_dict[name] = preds
    print(f"✅ {name} predicted {len(preds)} samples")

probs_stack = np.stack(list(probs_dict.values()), axis=0)
preds_stack = np.stack(list(pred_dict.values()), axis=0)

def fuse(mode: str) -> pd.Series:
    if mode == "majority":
        from scipy import stats
        sig = stats.mode(preds_stack, axis=0, keepdims=False).mode
        sig = pd.Series(sig, index=sample_times)
    elif mode == "weighted":
        sum_probs = probs_stack.sum(axis=0)
        sig = pd.Series(sum_probs.argmax(axis=1), index=sample_times)
    elif mode == "buy_only":
        sum_probs = probs_stack.sum(axis=0)
        hard = np.where(sum_probs.argmax(axis=1) == 2, 2, 1)
        sig = pd.Series(hard, index=sample_times)
    else:
        raise ValueError("Unknown fusion mode")

    # align full index
    sig = sig.reindex(price.index, method="ffill").fillna(0).astype(int)
    return sig


rows = []
for fmode in FUSION_MODES:
    sig = fuse(fmode)
    print(f"\n>>> FUSION MODE: {fmode}")
    for stop_mode, tp, sl in STOP_GRID:
        print(f"→ {stop_mode.upper()} | TP={tp} | SL={sl}")
        pf, stats, exp_pips = run_backtest(price, sig, stop_mode, tp, sl)
        exp_usd = exp_pips * PIP_USD
        print(f"Trades={int(stats['Total Trades'])} | Win%={stats['Win Rate [%]']:.2f} | PF={stats['Profit Factor']:.2f} | Exp={exp_pips:.2f}p ({exp_usd:.1f}$)")