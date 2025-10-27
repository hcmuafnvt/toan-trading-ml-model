# ============================================================
# STAGE 3.1 — Fusion Backtest (Baseline, fast)
#   - Load features (no re-extract), load 3 LightGBM models
#   - Fusion: majority / weighted / buy_only
#   - Stops: fixed / atr / vol
#   - Output: logs/stage3_1_results.csv + logs/stage3_1_summary.txt
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from ta.volatility import AverageTrueRange

# ---------------- CONFIG ----------------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"   # has DatetimeIndex, mid_* cols
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILES = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}
OUT_TXT = "logs/stage3_1_summary.txt"
OUT_CSV = "logs/stage3_1_results.csv"
os.makedirs("logs", exist_ok=True)

# must match Stage 2 extract constants
WINDOW = 200
STRIDE = 5

PIP_SIZE = 0.0001
PIP_USD = 10
FEES = 0.0  # bỏ qua phí như đã thống nhất

# test grid
STOP_GRID = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("fixed", 20.0, 20.0),
    ("atr",   1.5,  1.5),
    ("atr",   2.0,  2.0),
    ("vol",   1.5,  1.5),
    ("vol",   2.0,  2.0),
]
FUSION_MODES = ["majority", "weighted", "buy_only"]

# ---------------- HELPERS ----------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )
    return df

def load_price(path):
    df = pd.read_parquet(path)
    # map mid_* -> ohlc
    if all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"]):
        df = df.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})
    elif all(c in df.columns for c in ["bid_o","bid_h","bid_l","bid_c"]):
        df = df.rename(columns={"bid_o":"open","bid_h":"high","bid_l":"low","bid_c":"close"})
    elif all(c in df.columns for c in ["ask_o","ask_h","ask_l","ask_c"]):
        df = df.rename(columns={"ask_o":"open","ask_h":"high","ask_l":"low","ask_c":"close"})
    else:
        raise ValueError("Không tìm thấy mid_*, bid_* hoặc ask_* trong file giá.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index phải là DatetimeIndex.")
    df = df.sort_index()
    df.index = df.index.tz_localize(None)
    return df[["open","high","low","close"]]

def expectancy_from_trades(trades_rec, pip_size=PIP_SIZE) -> float:
    if trades_rec is None or len(trades_rec)==0:
        return 0.0
    entry = trades_rec["entry_price"].values
    exitp = trades_rec["exit_price"].values
    direction = trades_rec["direction"].values  # 0 long, 1 short
    sign = np.where(direction==0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def run_backtest(price, signal_series, stop_mode, tp_param, sl_param):
    # build TP/SL series (in price distance)
    if stop_mode == "fixed":
        tp = pd.Series(tp_param * PIP_SIZE, index=price.index)
        sl = pd.Series(sl_param * PIP_SIZE, index=price.index)
    elif stop_mode == "atr":
        atr = AverageTrueRange(
            high=price["high"], low=price["low"], close=price["close"], window=14
        ).average_true_range().ffill().bfill()
        # atr is in price units; convert to pips multiplier
        tp = (atr * tp_param).clip(lower=1e-7)
        sl = (atr * sl_param).clip(lower=1e-7)
    elif stop_mode == "vol":
        # volatility by rolling std of returns * close
        ret = price["close"].pct_change().rolling(96, min_periods=32).std().ffill().bfill()
        vol_px = (ret * price["close"]).clip(lower=1e-12)
        tp = (vol_px * tp_param).clip(lower=1e-7)
        sl = (vol_px * sl_param).clip(lower=1e-7)
    else:
        raise ValueError("Unknown stop_mode")

    # long/short entries & exits
    long_entries  = signal_series.eq(2)
    long_exits    = ~signal_series.eq(2)  # close long khi không còn BUY
    short_entries = signal_series.eq(0)
    short_exits   = ~signal_series.eq(0)  # close short khi không còn SELL

    pf = vbt.Portfolio.from_signals(
        price["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        tp_stop=tp,
        sl_stop=sl,
        direction="both",
        size=1.0,
        fees=FEES
    )
    stats = pf.stats()
    trades = pf.trades.records  # raw records
    exp_pips = expectancy_from_trades(trades)
    return pf, stats, exp_pips

# ---------------- LOAD EVERYTHING ----------------
print(f"⏳ Loading price: {DATA_FILE}")
price = load_price(DATA_FILE)
print(f"✅ Price loaded: {len(price):,} rows | {price.index.min()} → {price.index.max()}")

print(f"⏳ Loading features: {FEATURE_FILE}")
feat = pd.read_csv(FEATURE_FILE)
feat = clean_cols(feat)
print(f"✅ Features loaded: {feat.shape}")

# remove targets to get X_full
target_cols = [c for c in feat.columns if c.startswith("target_")]
X_full = feat.drop(columns=target_cols, errors="ignore")

# reconstruct sample timestamps to align predictions
sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
if len(sample_idx) != len(X_full):
    m = min(len(sample_idx), len(X_full))
    print(f"⚠️ Length mismatch X({len(X_full)}) vs sample_idx({len(sample_idx)}), truncating to {m}")
    sample_idx = sample_idx[:m]
    X_full = X_full.iloc[:m].copy()
sample_times = price.index[sample_idx]

# load models and predict
probs_dict = {}
pred_dict = {}
for name, path in MODEL_FILES.items():
    print(f"⏳ Loading model {name}: {path}")
    booster = lgb.Booster(model_file=path)
    feat_names = booster.feature_name()
    # align X to model feature order; fill missing with 0
    X = X_full.reindex(columns=feat_names, fill_value=0.0)
    # predict class probabilities
    probs = booster.predict(X.values, num_iteration=booster.best_iteration)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    probs_dict[name] = probs
    pred_dict[name]  = preds
    print(f"✅ {name} predicted: shape {probs.shape}")

# build fusion signals at sample times
probs_stack = np.stack(list(probs_dict.values()), axis=0)  # [models, samples, classes]
preds_stack = np.stack(list(pred_dict.values()), axis=0)   # [models, samples]

def fuse(mode):
    if mode == "majority":
        # vote across hard classes
        from scipy import stats as sstats
        vote = sstats.mode(preds_stack, axis=0, keepdims=False).mode  # [samples]
        signal_s = pd.Series(vote, index=sample_times)
    elif mode == "weighted":
        # sum probabilities across models, choose argmax
        sum_probs = probs_stack.sum(axis=0)  # [samples, 3]
        signal_s = pd.Series(sum_probs.argmax(axis=1), index=sample_times)
    elif mode == "buy_only":
        sum_probs = probs_stack.sum(axis=0)
        hard = sum_probs.argmax(axis=1)
        # only execute BUY; otherwise TIMEOUT
        hard = np.where(hard==2, 2, 1)
        signal_s = pd.Series(hard, index=sample_times)
    else:
        raise ValueError("Unknown fusion mode")
    # expand to full index by forward-fill
    full = pd.Series(1, index=price.index, dtype=int)  # default TIMEOUT=1
    full.loc[signal_s.index] = signal_s.values
    full = full.ffill().astype(int)
    return full

# ---------------- RUN GRID ----------------
rows = []
for fmode in FUSION_MODES:
    sig = fuse(fmode)
    for stop_mode, tp_param, sl_param in STOP_GRID:
        print(f"\n>>> Running fusion='{fmode}' | stop='{stop_mode}' | tp={tp_param} sl={sl_param}")
        pf, stats, exp_pips = run_backtest(price, sig, stop_mode, tp_param, sl_param)

        # collect stats
        total_trades = int(stats.get("Total Trades", 0))
        win_rate     = float(stats.get("Win Rate [%]", 0.0))
        pfactor      = float(stats.get("Profit Factor", 0.0))
        ret          = float(stats.get("Total Return [%]", 0.0))
        mdd          = float(stats.get("Max Drawdown [%]", 0.0))
        exp_usd      = exp_pips * PIP_USD

        print(f"Trades={total_trades} | Win%={win_rate:.2f} | PF={pfactor:.2f} | Ret%={ret:.2f} | DD%={mdd:.2f} | Exp={exp_pips:.2f} pips (${exp_usd:.2f})")

        rows.append({
            "fusion_mode": fmode,
            "stop_mode": stop_mode,
            "tp_param": tp_param,
            "sl_param": sl_param,
            "Total Trades": total_trades,
            "Win Rate [%]": win_rate,
            "Total Return [%]": ret,
            "Profit Factor": pfactor,
            "Max Drawdown [%]": mdd,
            "Expectancy (pips)": exp_pips,
            "Expectancy (USD_1lot)": exp_usd
        })

# save results
res = pd.DataFrame(rows)
res.sort_values(["Profit Factor","Total Return [%]","Win Rate [%]"], ascending=False, inplace=True)
res.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w") as f:
    def top(df, by, k=5):
        return df.sort_values(by, ascending=False).head(k)

    f.write("========== TOP by Profit Factor ==========\n")
    f.write(top(res, "Profit Factor").to_string(index=False))
    f.write("\n\n========== TOP by Total Return [%] ==========\n")
    f.write(top(res, "Total Return [%]").to_string(index=False))
    f.write("\n\n========== TOP by Win Rate [%] ==========\n")
    f.write(top(res, "Win Rate [%]").to_string(index=False))
    f.write("\n")

print(f"\n✅ Saved grid results → {OUT_CSV}")
print(f"✅ Summary → {OUT_TXT}")