#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FX Coding – Stage 3_1 Fusion Backtest + Export Target
-----------------------------------------------------
Kết hợp 3 mô hình LightGBM (T1/T2/T3), backtest portfolio,
và lưu lại target vector (y) + Xy dataset cho Stage 4L.

Tuân theo project rules 1–5 (EC2 runtime friendly, deterministic).
"""

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
PIP_USD  = 10.0           # 1 lot = $10/pip với GBPUSD
FEES     = 0.0

FUSION_MODES = ["majority", "weighted", "buy_only"]
STOP_GRID = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("fixed", 20.0, 20.0),
    ("atr",   1.5,  1.5),
    ("atr",   2.0,  2.0),
]

OUT_TXT = "logs/stage3_1_summary.txt"
OUT_CSV = "logs/stage3_1_results.csv"
os.makedirs("logs", exist_ok=True)


# ================= HELPERS =================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def load_price_oanda_bam(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "close" in df.columns:
        df = df.drop(columns=["close"])
    for c in ["mid_o", "mid_h", "mid_l", "mid_c"]:
        df[c] = df[c].apply(safe_mean)
    df = df.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cols = ["open","high","low","close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols]

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0=long  1=short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def atr_from_ta(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    tmp = price_df[["high","low","close"]].copy()
    high_s, low_s, close_s = (
        pd.Series(tmp["high"].to_numpy(float)),
        pd.Series(tmp["low"].to_numpy(float)),
        pd.Series(tmp["close"].to_numpy(float)),
    )
    atr = AverageTrueRange(high_s, low_s, close_s, window=window).average_true_range()
    atr.index = price_df.index
    return atr.ffill().bfill()

def run_backtest(price: pd.DataFrame, signal: pd.Series, stop_mode: str, tp_param: float, sl_param: float):
    long_entries  = signal.eq(2)
    long_exits    = ~signal.eq(2)
    short_entries = signal.eq(0)
    short_exits   = ~signal.eq(0)
    if stop_mode == "fixed":
        tp = pd.Series(tp_param * PIP_SIZE, index=price.index)
        sl = pd.Series(sl_param * PIP_SIZE, index=price.index)
    elif stop_mode == "atr":
        atr = atr_from_ta(price, 14)
        tp, sl = (atr * tp_param).clip(lower=1e-12), (atr * sl_param).clip(lower=1e-12)
    else:
        raise ValueError("Unknown stop_mode")

    pf = vbt.Portfolio.from_signals(
        price["close"], long_entries, long_exits,
        short_entries, short_exits, tp_stop=tp, sl_stop=sl,
        size=1.0, fees=FEES, freq="5min"
    )
    stats = pf.stats()
    exp_pips = expectancy_from_trades(pf.trades.records)
    return pf, stats, exp_pips


# ================= MAIN =================
if __name__ == "__main__":
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price_oanda_bam(DATA_FILE)
    print(f"✅ Loaded: {len(price):,} rows")

    print(f"\n⏳ Loading features: {FEATURE_FILE}")
    feat = pd.read_csv(FEATURE_FILE)
    feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"✅ Features loaded: {feat.shape}")

    target_cols = [c for c in feat.columns if c.startswith("target_")]
    X_full = feat.drop(columns=target_cols, errors="ignore")
    n_samples = len(X_full)

    # Map sample index → timestamp
    sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
    if len(sample_idx) < n_samples:
        X_full = X_full.iloc[:len(sample_idx)].copy()
        n_samples = len(X_full)
    sample_times = price.index[sample_idx[:n_samples]]

    # ---------- SAVE / LOAD y VECTOR (Stage 3 Export) ----------
    y_candidates = [c for c in feat.columns if c.lower() in ("y","target","target_class","label")]

    if y_candidates:
        # Nếu stage2_features.csv có target sẵn
        y = feat[y_candidates[0]].iloc[:n_samples].reset_index(drop=True)
        print(f"✅ Found target column in features: {y_candidates[0]} (len={len(y)})")
    else:
        # Nếu không có, tự load từ logs/stage3_y.csv (đã export từ Stage 1)
        y_path = "logs/stage3_y.csv"
        if os.path.exists(y_path):
            y = pd.read_csv(y_path)["y"].iloc[:n_samples].reset_index(drop=True)
            print(f"✅ Loaded y from {y_path} (len={len(y)})")
        else:
            print("❌ Không tìm thấy target column và cũng không có logs/stage3_y.csv → STOP.")
            raise FileNotFoundError("Cần chạy Stage 1 trước để tạo logs/stage3_y.csv")

    # --- Save y & Xy để Stage 4L dùng ---
    y.to_csv("logs/stage3_y.csv", index=False)
    Xy = pd.concat([X_full.reset_index(drop=True), y.rename('y')], axis=1)
    Xy.to_csv("logs/stage3_Xy.csv", index=False)
    print(f"✅ Saved target vector: logs/stage3_y.csv (len={len(y)})")
    print(f"✅ Saved combined Xy: logs/stage3_Xy.csv (shape={Xy.shape})")

    # ---------- LOAD MODELS & PREDICT ----------
    probs_dict, pred_dict = {}, {}
    for name, path in MODEL_FILES.items():
        print(f"⏳ Loading model {name}: {path}")
        booster = lgb.Booster(model_file=path)
        feat_names = booster.feature_name()
        X = X_full.reindex(columns=feat_names, fill_value=0.0)
        probs = booster.predict(X.values, num_iteration=booster.best_iteration)
        preds = np.asarray(probs).argmax(axis=1)
        probs_dict[name], pred_dict[name] = np.asarray(probs), preds
        print(f"✅ {name} predicted: {len(preds)} samples")

    probs_stack = np.stack(list(probs_dict.values()), axis=0)  # [n_models, n_samples, 3]
    preds_stack = np.stack(list(pred_dict.values()), axis=0)   # [n_models, n_samples]

    def fuse(mode: str) -> pd.Series:
        if mode == "majority":
            vote0 = (preds_stack == 0).sum(axis=0)
            vote1 = (preds_stack == 1).sum(axis=0)
            vote2 = (preds_stack == 2).sum(axis=0)
            vote = np.select([vote0>=vote1, vote2>=vote1], [0,2], default=1)
            sig = pd.Series(vote, index=sample_times)
        elif mode == "weighted":
            sum_probs = probs_stack.sum(axis=0)
            sig = pd.Series(sum_probs.argmax(axis=1), index=sample_times)
        elif mode == "buy_only":
            sum_probs = probs_stack.sum(axis=0)
            hard = np.where(sum_probs.argmax(axis=1)==2,2,1)
            sig = pd.Series(hard, index=sample_times)
        else:
            raise ValueError("Unknown fusion mode")
        return sig.reindex(price.index, method="ffill").fillna(0).astype(int)

    # ---------- BACKTEST ----------
    rows = []
    print("\n========== FUSION BACKTEST ==========")
    for fmode in FUSION_MODES:
        sig = fuse(fmode)
        print(f"[Align] fusion={fmode} | price={len(price):,} | signal={len(sig):,}")

        for stop_mode, tp_param, sl_param in STOP_GRID:
            print(f"\n→ {stop_mode.upper()} | TP={tp_param} | SL={sl_param}")
            pf, stats, exp_pips = run_backtest(price, sig, stop_mode, tp_param, sl_param)

            def _get(s, k):
                v = s.get(k, np.nan)
                if hasattr(v, "values"):
                    v = np.asarray(v)[0]
                try:
                    return float(v)
                except Exception:
                    return np.nan

            total_trades = int(_get(stats, "Total Trades"))
            win_rate     = _get(stats, "Win Rate [%]")
            pfactor      = _get(stats, "Profit Factor")
            ret          = _get(stats, "Total Return [%]")
            mdd          = _get(stats, "Max Drawdown [%]")
            exp_usd      = exp_pips * PIP_USD

            print(f"Trades={total_trades} | Win%={win_rate:.2f} | PF={pfactor:.2f} | "
                  f"Ret%={ret:.2f} | DD%={mdd:.2f} | Exp={exp_pips:.2f} pips (${exp_usd:.2f})")

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

    res = pd.DataFrame(rows).sort_values(
        ["Profit Factor","Total Return [%]","Win Rate [%]"],
        ascending=False
    )
    res.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT,"w") as f:
        def top(df, by, k=5): return df.sort_values(by, ascending=False).head(k)
        f.write("========== TOP by Profit Factor ==========\n")
        f.write(top(res,"Profit Factor").to_string(index=False))
        f.write("\n\n========== TOP by Total Return [%] ==========\n")
        f.write(top(res,"Total Return [%]").to_string(index=False))
        f.write("\n\n========== TOP by Win Rate [%] ==========\n")
        f.write(top(res,"Win Rate [%]").to_string(index=False))
        f.write("\n")

    print(f"\n✅ Saved grid results → {OUT_CSV}")
    print(f"✅ Summary → {OUT_TXT}")