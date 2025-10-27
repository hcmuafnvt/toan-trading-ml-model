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
PIP_USD  = 10.0
FEES     = 0.0

# Regime thresholds by ATR quantiles
LOW_Q, HIGH_Q = 0.33, 0.66

# Small grid to search per-regime
FUSION_CANDIDATES = ["weighted", "majority", "buy_only"]
STOP_CANDIDATES = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("fixed", 20.0, 20.0),
    ("atr",   1.5,  1.5),
    ("atr",   2.0,  2.0),
]

# Confidence → position sizing
SIZE_MIN = 0.5
SIZE_MAX = 2.0

OUT_CSV = "logs/stage3_2_adaptive_results.csv"
OUT_TXT = "logs/stage3_2_adaptive_summary.txt"
os.makedirs("logs", exist_ok=True)


# ================= HELPERS =================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    return df

def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return float(arr.mean())
    return float(x)

def load_price_oanda_bam(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # drop legacy 'close' if present to avoid duplication
    if "close" in df.columns:
        df = df.drop(columns=["close"])

    req = ["mid_o", "mid_h", "mid_l", "mid_c"]
    if not all(c in df.columns for c in req):
        raise ValueError("Thiếu mid_o/mid_h/mid_l/mid_c trong parquet.")

    for col in req:
        df[col] = df[col].apply(safe_mean)

    df = df.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index phải là DatetimeIndex.")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    cols = ["open","high","low","close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols]

def atr_from_ta(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    tmp = price_df[["high","low","close"]].copy()
    high_s  = pd.Series(tmp["high"].to_numpy(dtype=float))
    low_s   = pd.Series(tmp["low"].to_numpy(dtype=float))
    close_s = pd.Series(tmp["close"].to_numpy(dtype=float))
    atr = AverageTrueRange(high_s, low_s, close_s, window=window).average_true_range()
    atr.index = price_df.index
    return atr.ffill().bfill()

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0 = long, 1 = short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def run_backtest(price: pd.DataFrame, signal: pd.Series, stop_mode: str, tp_param: float, sl_param: float, size_series: pd.Series = None):
    long_entries  = signal.eq(2)
    long_exits    = ~signal.eq(2)
    short_entries = signal.eq(0)
    short_exits   = ~signal.eq(0)

    if stop_mode == "fixed":
        tp = pd.Series(tp_param * PIP_SIZE, index=price.index)
        sl = pd.Series(sl_param * PIP_SIZE, index=price.index)
    elif stop_mode == "atr":
        atr = atr_from_ta(price, window=14)
        tp = (atr * tp_param).clip(lower=1e-12)
        sl = (atr * sl_param).clip(lower=1e-12)
    else:
        raise ValueError("Unknown stop_mode")

    kw = dict(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        tp_stop=tp,
        sl_stop=sl,
        fees=FEES,
        freq="5min"
    )
    if size_series is not None:
        kw["size"] = size_series

    pf = vbt.Portfolio.from_signals(price["close"], **kw)
    stats = pf.stats()
    exp_pips = expectancy_from_trades(pf.trades.records)
    return pf, stats, exp_pips

def get_confidence_from_probs(sum_probs: np.ndarray) -> np.ndarray:
    # sum_probs: [n_samples, 3]
    sorted_probs = np.sort(sum_probs, axis=1)  # ascending
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margin

def scale_to_range(x: np.ndarray, lo=0.0, hi=1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a, b = np.nanmin(x), np.nanmax(x)
    if b - a < 1e-12:
        return np.full_like(x, (lo+hi)/2.0)
    return lo + (x - a) * (hi - lo) / (b - a)


# ================= MAIN =================
if __name__ == "__main__":
    # 1) Price + ATR + Regime
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price_oanda_bam(DATA_FILE)
    print(f"✅ Loaded price: {len(price):,} rows | Example:")
    print(price.head(3))

    atr = atr_from_ta(price, window=14)
    q_low, q_high = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
    regime = pd.Series(np.where(atr <= q_low, "Low",
                        np.where(atr >= q_high, "High", "Normal")), index=price.index)
    print(f"\n✅ Regime by ATR quantiles: Low≤{q_low:.6f}, High≥{q_high:.6f}")
    print(regime.value_counts())

    # 2) Features + map samples to timestamps
    print(f"\n⏳ Loading features: {FEATURE_FILE}")
    feat = pd.read_csv(FEATURE_FILE)
    feat = clean_cols(feat).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    target_cols = [c for c in feat.columns if c.startswith("target_")]
    X_full = feat.drop(columns=target_cols, errors="ignore")
    n_samples = len(X_full)
    print(f"✅ Features loaded: {X_full.shape}")

    sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
    if len(sample_idx) < n_samples:
        X_full = X_full.iloc[:len(sample_idx)].copy()
        n_samples = len(X_full)
    sample_times = price.index[sample_idx[:n_samples]]

    # 3) Predict with all models
    probs_dict, pred_dict = {}, {}
    for name, path in MODEL_FILES.items():
        print(f"⏳ Loading model {name}: {path}")
        booster = lgb.Booster(model_file=path)
        feat_names = booster.feature_name()
        X = X_full.reindex(columns=feat_names, fill_value=0.0)
        probs = booster.predict(X.values, num_iteration=booster.best_iteration)  # [n_samples, 3]
        preds = np.asarray(probs).argmax(axis=1)
        probs_dict[name] = np.asarray(probs)
        pred_dict[name]  = preds
        print(f"✅ {name} predicted: {len(preds)} samples")

    probs_stack = np.stack(list(probs_dict.values()), axis=0)  # [3, n_samples, 3]
    preds_stack = np.stack(list(pred_dict.values()), axis=0)   # [3, n_samples]

    # 4) Build fusion signals on sample times, then expand to full index
    def fuse(mode: str) -> pd.Series:
        if mode == "majority":
            vote0 = (preds_stack == 0).sum(axis=0)
            vote1 = (preds_stack == 1).sum(axis=0)
            vote2 = (preds_stack == 2).sum(axis=0)
            vote = np.select([vote0>=vote1, vote2>=vote1], [0, 2], default=1)
            sig = pd.Series(vote, index=sample_times)
        elif mode == "weighted":
            sum_probs = probs_stack.sum(axis=0)  # [n_samples, 3]
            sig = pd.Series(sum_probs.argmax(axis=1), index=sample_times)
        elif mode == "buy_only":
            sum_probs = probs_stack.sum(axis=0)
            hard = np.where(sum_probs.argmax(axis=1) == 2, 2, 1)
            sig = pd.Series(hard, index=sample_times)
        else:
            raise ValueError("Unknown fusion mode")
        return sig.reindex(price.index, method="ffill").fillna(0).astype(int)

    fusion_series = {fm: fuse(fm) for fm in FUSION_CANDIDATES}

    # Confidence (use weighted probs)
    sum_probs = probs_stack.sum(axis=0)  # [n_samples, 3]
    conf_sample = get_confidence_from_probs(sum_probs)  # length n_samples
    conf_series = pd.Series(conf_sample, index=sample_times).reindex(price.index, method="ffill").fillna(0.0)
    size_series = pd.Series(scale_to_range(conf_series.values, SIZE_MIN, SIZE_MAX), index=price.index)

    # 5) Evaluate each (fusion, stop) per regime and pick best by PF
    def eval_one(sig: pd.Series, stop_mode: str, tp: float, sl: float, mask: pd.Series):
        # mask timestamps of given regime; set TIMEOUT=1 outside regime to "do nothing"
        sig_masked = sig.copy()
        sig_masked[~mask] = 1  # TIMEOUT
        pf, stats, exp_pips = run_backtest(price, sig_masked, stop_mode, tp, sl, size_series=size_series)
        # extract floats robustly
        def _get(s, k):
            v = s.get(k, np.nan)
            try:
                return float(v)
            except Exception:
                if hasattr(v, "values"):
                    return float(np.asarray(v)[0])
                return float("nan")
        return {
            "Total Trades": int(_get(stats, "Total Trades")),
            "Win Rate [%]": _get(stats, "Win Rate [%]"),
            "Total Return [%]": _get(stats, "Total Return [%]"),
            "Profit Factor": _get(stats, "Profit Factor"),
            "Max Drawdown [%]": _get(stats, "Max Drawdown [%]"),
            "Expectancy (pips)": exp_pips,
            "Expectancy (USD_1lot)": exp_pips * PIP_USD
        }

    masks = {
        "Low": regime.eq("Low"),
        "Normal": regime.eq("Normal"),
        "High": regime.eq("High")
    }

    rows = []
    best_cfg = {}  # regime -> (fusion, stop, tp, sl, metrics)

    print("\n========== STAGE 3.2: Regime-wise Search ==========")
    for reg_name, reg_mask in masks.items():
        print(f"\n--- Regime: {reg_name} (samples={int(reg_mask.sum())}) ---")
        best = None
        for fm in FUSION_CANDIDATES:
            for stop_mode, tp, sl in STOP_CANDIDATES:
                metrics = eval_one(fusion_series[fm], stop_mode, tp, sl, reg_mask)
                row = {"Regime": reg_name, "Fusion": fm, "Stop": stop_mode, "TP": tp, "SL": sl}
                row.update(metrics)
                rows.append(row)
                if (best is None) or (metrics["Profit Factor"] > best[4]["Profit Factor"]):
                    best = (fm, stop_mode, tp, sl, metrics)
                print(f"[{reg_name}] {fm:9s} + {stop_mode:5s} "
                      f"→ PF={metrics['Profit Factor']:.2f} | Win%={metrics['Win Rate [%]']:.2f} "
                      f"| Exp={metrics['Expectancy (pips)']:.2f}p | Trades={metrics['Total Trades']}")
        best_cfg[reg_name] = best
        fm, sm, tp, sl, m = best
        print(f"✅ Best {reg_name}: {fm} + {sm} (TP={tp}, SL={sl}) | PF={m['Profit Factor']:.2f}, "
              f"Win%={m['Win Rate [%]']:.2f}, Exp={m['Expectancy (pips)']:.2f}p")

    # 6) Build final adaptive signal + per-bar TP/SL series
    final_signal = pd.Series(1, index=price.index, dtype=int)  # default TIMEOUT
    tp_series = pd.Series(np.nan, index=price.index, dtype=float)
    sl_series = pd.Series(np.nan, index=price.index, dtype=float)
    atr_full = atr_from_ta(price, window=14)  # for ATR modes

    for reg_name, reg_mask in masks.items():
        fm, sm, tpv, slv, _ = best_cfg[reg_name]
        sig = fusion_series[fm]
        # apply only on regime timestamps
        final_signal.loc[reg_mask] = sig.loc[reg_mask]

        if sm == "fixed":
            tp_series.loc[reg_mask] = tpv * PIP_SIZE
            sl_series.loc[reg_mask] = slv * PIP_SIZE
        else:  # atr
            tp_series.loc[reg_mask] = (atr_full.loc[reg_mask] * tpv).clip(lower=1e-12)
            sl_series.loc[reg_mask] = (atr_full.loc[reg_mask] * slv).clip(lower=1e-12)

    # Any remaining NaN (shouldn't) → fill with fixed small stops to be safe
    tp_series = tp_series.fillna(10 * PIP_SIZE)
    sl_series = sl_series.fillna(10 * PIP_SIZE)

    # 7) Final adaptive backtest with confidence sizing
    long_entries  = final_signal.eq(2)
    long_exits    = ~final_signal.eq(2)
    short_entries = final_signal.eq(0)
    short_exits   = ~final_signal.eq(0)

    pf_final = vbt.Portfolio.from_signals(
        price["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        tp_stop=tp_series,
        sl_stop=sl_series,
        size=size_series,   # confidence-weighted sizing
        fees=FEES,
        freq="5min"
    )
    stats_final = pf_final.stats()
    exp_final = expectancy_from_trades(pf_final.trades.records)

    def _get(s, k):
        v = s.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            if hasattr(v, "values"):
                return float(np.asarray(v)[0])
            return float("nan")

    print("\n========== FINAL ADAPTIVE PORTFOLIO ==========")
    print(f"Total Trades     : {int(_get(stats_final,'Total Trades'))}")
    print(f"Win Rate [%]     : {_get(stats_final,'Win Rate [%]'):.2f}")
    print(f"Profit Factor    : {_get(stats_final,'Profit Factor'):.2f}")
    print(f"Total Return [%] : {_get(stats_final,'Total Return [%]'):.2f}")
    print(f"Max Drawdown [%] : {_get(stats_final,'Max Drawdown [%]'):.2f}")
    print(f"Expectancy       : {exp_final:.2f} pips (${exp_final*PIP_USD:.2f})")

    # 8) Save grids & summary
    res = pd.DataFrame(rows).sort_values(["Regime","Profit Factor","Win Rate [%]","Total Return [%]"], ascending=[True,False,False,False])
    res.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w") as f:
        f.write("========== BEST CONFIG PER REGIME ==========\n")
        for reg in ["Low","Normal","High"]:
            fm, sm, tpv, slv, m = best_cfg[reg]
            f.write(f"{reg}: {fm} + {sm} (TP={tpv}, SL={slv}) | PF={m['Profit Factor']:.2f}, Win%={m['Win Rate [%]']:.2f}, Exp={m['Expectancy (pips)']:.2f}p, Trades={m['Total Trades']}\n")
        f.write("\n========== FINAL ADAPTIVE STATS ==========\n")
        f.write(stats_final.to_string())
        f.write(f"\nExpectancy (pips): {exp_final:.4f} | USD_1lot: {exp_final*PIP_USD:.2f}\n")

    print(f"\n✅ Saved per-regime grid → {OUT_CSV}")
    print(f"✅ Summary (best configs + final stats) → {OUT_TXT}")