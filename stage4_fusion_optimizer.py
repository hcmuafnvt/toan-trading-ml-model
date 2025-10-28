import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from ta.volatility import AverageTrueRange

# ===================== CONFIG =====================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILES = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}

# mapping schema parquet (OANDA mid_*)
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# sampling map: phải khớp Stage 2
WINDOW = 200
STRIDE = 5

# pips & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# ATR regime quantiles
LOW_Q, HIGH_Q = 0.33, 0.66

# Fusion & Stops search space
FUSION_CANDIDATES = ["weighted", "majority", "buy_only"]
STOP_CANDIDATES = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("fixed", 20.0, 20.0),
    ("atr",   1.5,  1.5),
    ("atr",   2.0,  2.0),
]

# Confidence -> position sizing
CONF_TH = 0.0      # có thể nâng lên 0.2~0.3 nếu muốn lọc tín hiệu yếu
SIZE_MIN = 0.5
SIZE_MAX = 2.0

OUT_GRID_CSV = "logs/stage4_grid_results.csv"
OUT_BEST_JSON = "logs/stage4_best_config.json"
OUT_SUMMARY = "logs/stage4_summary.txt"
os.makedirs("logs", exist_ok=True)


# ===================== HELPERS =====================
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

def load_price(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # chuẩn hóa OHLC
    have_mids = all(c in df.columns for c in ["mid_o","mid_h","mid_l","mid_c"])
    if have_mids:
        # xoá 'close' legacy nếu có để tránh trùng cột
        if "close" in df.columns:
            df = df.drop(columns=["close"])
        for k,v in OHLC_MAP.items():
            df[v] = df[k].apply(safe_mean)
        keep = ["open","high","low","close"]
        if "volume" in df.columns:
            keep.append("volume")
        df = df[keep]
    else:
        # giả định đã có open/high/low/close
        req = ["open","high","low","close"]
        if not all(c in df.columns for c in req):
            raise ValueError("Parquet thiếu OHLC (mid_* hoặc open/high/low/close).")
        keep = req + (["volume"] if "volume" in df.columns else [])
        df = df[keep]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index phải là DatetimeIndex")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def atr_from_ta(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    tmp = price_df[["high","low","close"]].copy()
    high_s  = pd.Series(tmp["high"].to_numpy(dtype=float))
    low_s   = pd.Series(tmp["low"].to_numpy(dtype=float))
    close_s = pd.Series(tmp["close"].to_numpy(dtype=float))
    atr = AverageTrueRange(high_s, low_s, close_s, window=window).average_true_range()
    atr.index = price_df.index
    return atr.ffill().bfill()

def get_session(ts):
    h = ts.hour
    if 22 <= h or h < 6:
        return "Asia"
    elif 6 <= h < 14:
        return "London"
    else:
        return "NewYork"

def get_confidence_from_probs(sum_probs: np.ndarray) -> np.ndarray:
    s = np.sort(sum_probs, axis=1)
    return s[:,-1] - s[:,-2]

def scale_to_range(x: np.ndarray, lo=0.5, hi=2.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a, b = np.nanmin(x), np.nanmax(x)
    if b - a < 1e-12:
        return np.full_like(x, (lo + hi) / 2.0)
    return lo + (x - a) * (hi - lo) / (b - a)

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0=long, 1=short
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    return float(np.nanmean(pips))

def profit_usd_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE, pip_usd=PIP_USD) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    size  = tr_rec["size"].values if "size" in tr_rec.columns else np.ones_like(entry)
    direction = tr_rec["direction"].values
    sign = np.where(direction == 0, 1.0, -1.0)
    pips = (exitp - entry) * sign / pip_size
    usd = pips * pip_usd * size
    return float(np.nansum(usd))


# ===================== MAIN =====================
if __name__ == "__main__":
    # 1) Load price & regimes
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price(DATA_FILE)
    print(f"✅ Loaded price: {len(price):,} rows | Example:\n{price.head(3)}")

    atr = atr_from_ta(price, window=14)
    q_low, q_high = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
    regime = pd.Series(np.where(atr <= q_low, "Low",
                          np.where(atr >= q_high, "High", "Normal")), index=price.index)
    session = price.index.map(get_session)
    print("\n✅ Regime/session matrix:")
    print(pd.crosstab(session, regime).to_string())

    # 2) Load features (Stage 2) & align to sample times
    print(f"\n⏳ Loading features: {FEATURE_FILE}")
    feat = pd.read_csv(FEATURE_FILE)
    feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_cols = [c for c in feat.columns if c.startswith("target_")]
    X_full = feat.drop(columns=target_cols, errors="ignore")
    n_samples = len(X_full)
    print(f"✅ Features loaded: {X_full.shape}")

    sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
    if len(sample_idx) < n_samples:
        # thu hẹp features cho khớp
        X_full = X_full.iloc[:len(sample_idx)].copy()
        n_samples = len(X_full)
    sample_times = price.index[sample_idx[:n_samples]]

    # 3) Predict với 3 model
    probs_dict, pred_dict = {}, {}
    for name, path in MODEL_FILES.items():
        print(f"⏳ Loading model {name}: {path}")
        booster = lgb.Booster(model_file=path)
        feature_names = booster.feature_name()
        X = X_full.reindex(columns=feature_names, fill_value=0.0)
        probs = booster.predict(X.values, num_iteration=booster.best_iteration)
        preds = np.asarray(probs).argmax(axis=1)
        probs_dict[name] = np.asarray(probs)
        pred_dict[name]  = preds
        print(f"✅ {name} predicted: {len(preds)} samples")

    probs_stack = np.stack(list(probs_dict.values()), axis=0)   # [3, n, 3]
    preds_stack = np.stack(list(pred_dict.values()), axis=0)    # [3, n]

    # 4) Fusion series (expand từ sample_times -> full index)
    def fuse(mode: str) -> pd.Series:
        if mode == "majority":
            vote0 = (preds_stack == 0).sum(axis=0)
            vote1 = (preds_stack == 1).sum(axis=0)
            vote2 = (preds_stack == 2).sum(axis=0)
            vote = np.select([vote0 >= vote1, vote2 >= vote1], [0, 2], default=1)
            sig = pd.Series(vote, index=sample_times)
        elif mode == "weighted":
            sum_probs = probs_stack.sum(axis=0)  # [n, 3]
            sig = pd.Series(sum_probs.argmax(axis=1), index=sample_times)
        elif mode == "buy_only":
            sum_probs = probs_stack.sum(axis=0)
            hard = np.where(sum_probs.argmax(axis=1) == 2, 2, 1)
            sig = pd.Series(hard, index=sample_times)
        else:
            raise ValueError("Unknown fusion mode")
        return sig.reindex(price.index, method="ffill").fillna(1).astype(int)

    fusion_series = {fm: fuse(fm) for fm in FUSION_CANDIDATES}

    # Confidence & size
    sum_probs = probs_stack.sum(axis=0)  # [n,3]
    conf_sample = get_confidence_from_probs(sum_probs)
    conf_series = pd.Series(conf_sample, index=sample_times).reindex(price.index, method="ffill").fillna(0.0)
    # filter tín hiệu yếu (timeout)
    for k in fusion_series:
        s = fusion_series[k].copy()
        s[conf_series < CONF_TH] = 1
        fusion_series[k] = s

    size_series = pd.Series(scale_to_range(conf_series.values, SIZE_MIN, SIZE_MAX), index=price.index)

    # 5) Backtest helpers
    def run_backtest(signal: pd.Series, stop_mode: str, tp_param: float, sl_param: float):
        long_entries  = signal.eq(2)
        long_exits    = ~signal.eq(2)
        short_entries = signal.eq(0)
        short_exits   = ~signal.eq(0)

        if stop_mode == "fixed":
            tp = pd.Series(tp_param * PIP_SIZE, index=price.index)
            sl = pd.Series(sl_param * PIP_SIZE, index=price.index)
        elif stop_mode == "atr":
            atr14 = atr_from_ta(price, window=14)
            tp = (atr14 * tp_param).clip(lower=1e-12)
            sl = (atr14 * sl_param).clip(lower=1e-12)
        else:
            raise ValueError("Unknown stop_mode")

        pf = vbt.Portfolio.from_signals(
            price["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            tp_stop=tp,
            sl_stop=sl,
            size=size_series,
            fees=FEES,
            freq="5min"
        )
        stats = pf.stats()
        tr = pf.trades.records
        exp_pips = expectancy_from_trades(tr)
        profit_usd = profit_usd_from_trades(tr)
        return pf, stats, exp_pips, profit_usd

    # 6) Evaluate per (session x regime)
    masks = {}
    sess_vals = ["Asia","London","NewYork"]
    reg_vals  = ["Low","Normal","High"]
    for ss in sess_vals:
        for rg in reg_vals:
            masks[(ss,rg)] = (session == ss) & (regime == rg)

    rows = []
    best_cfg = {}  # key=(session,regime) -> (fusion, stop, tp, sl, metrics)

    print("\n========== STAGE 4: FUSION OPTIMIZER (USD-based) ==========")
    for ss in sess_vals:
        for rg in reg_vals:
            mask = masks[(ss,rg)]
            n_mask = int(mask.sum())
            if n_mask < 1000:
                # quá ít nến/segment -> bỏ qua tối ưu phần này
                continue
            print(f"\n--- Segment: {ss} × {rg} (bars={n_mask}) ---")
            best = None
            for fm in FUSION_CANDIDATES:
                sig = fusion_series[fm].copy()
                # enforce timeout ngoài segment để không rớt shape
                sig[~mask] = 1

                for stop_mode, tp, sl in STOP_CANDIDATES:
                    _, st, exp_p, usd = run_backtest(sig, stop_mode, tp, sl)

                    def _get(s, k):
                        v = s.get(k, np.nan)
                        try:
                            return float(v)
                        except Exception:
                            if hasattr(v, "values"):
                                return float(np.asarray(v)[0])
                            return float("nan")

                    row = {
                        "Session": ss,
                        "Regime": rg,
                        "Fusion": fm,
                        "Stop": stop_mode,
                        "TP": tp,
                        "SL": sl,
                        "Total Trades": int(_get(st, "Total Trades")),
                        "Win Rate [%]": _get(st, "Win Rate [%]"),
                        "Profit Factor": _get(st, "Profit Factor"),
                        "Total Return [%]": _get(st, "Total Return [%]"),
                        "Expectancy (pips)": exp_p,
                        "Expectancy (USD_1lot)": exp_p * PIP_USD,
                        "Total Profit (USD_1lot)": usd,
                        "Max Drawdown [%]": _get(st, "Max Drawdown [%]"),
                    }
                    rows.append(row)

                    if (best is None) or (usd > best[4]["Total Profit (USD_1lot)"]):
                        best = (fm, stop_mode, tp, sl, row)

                    print(f"[{ss}/{rg}] {fm:9s}+{stop_mode:5s} "
                          f"→ PF={row['Profit Factor']:.2f} | Win={row['Win Rate [%]']:.2f}% "
                          f"| Exp={row['Expectancy (pips)']:.2f}p (${row['Expectancy (USD_1lot)']:.2f}) "
                          f"| Trades={row['Total Trades']} | Profit=${row['Total Profit (USD_1lot)']:.0f}")
            if best is not None:
                best_cfg[(ss,rg)] = best
                fm, sm, tpv, slv, m = best
                print(f"✅ Best {ss}×{rg}: {fm}+{sm} TP={tpv} SL={slv} "
                      f"| Profit=${m['Total Profit (USD_1lot)']:.0f} | PF={m['Profit Factor']:.2f} "
                      f"| Exp={m['Expectancy (pips)']:.2f}p | Win={m['Win Rate [%]']:.2f}%")

    # 7) Compose final adaptive signal & TP/SL theo best_cfg
    # default: TIMEOUT
    final_signal = pd.Series(1, index=price.index, dtype=int)
    final_tp = pd.Series(np.nan, index=price.index, dtype=float)
    final_sl = pd.Series(np.nan, index=price.index, dtype=float)

    # weighted fusion (global) để dùng khi segment không có best
    default_fused = fusion_series["weighted"]

    # fill theo segment
    for ss in sess_vals:
        for rg in reg_vals:
            mask = masks[(ss,rg)]
            if mask.sum() == 0:
                continue
            if (ss,rg) in best_cfg:
                fm, sm, tpv, slv, _ = best_cfg[(ss,rg)]
                seg_sig = fusion_series[fm]
                final_signal.loc[mask] = seg_sig.loc[mask]
                if sm == "fixed":
                    final_tp.loc[mask] = tpv * PIP_SIZE
                    final_sl.loc[mask] = slv * PIP_SIZE
                else:
                    atr14 = atr_from_ta(price, window=14)
                    final_tp.loc[mask] = (atr14.loc[mask] * tpv).clip(lower=1e-12)
                    final_sl.loc[mask] = (atr14.loc[mask] * slv).clip(lower=1e-12)
            else:
                # không có best -> dùng default
                final_signal.loc[mask] = default_fused.loc[mask]

    # fill NaN stops an toàn
    final_tp = final_tp.fillna(10 * PIP_SIZE)
    final_sl = final_sl.fillna(10 * PIP_SIZE)

    # 8) Final backtest trên toàn chuỗi (size theo confidence)
    pf_final = vbt.Portfolio.from_signals(
        price["close"],
        entries=final_signal.eq(2),
        exits=~final_signal.eq(2),
        short_entries=final_signal.eq(0),
        short_exits=~final_signal.eq(0),
        tp_stop=final_tp,
        sl_stop=final_sl,
        size=size_series,
        fees=FEES,
        freq="5min"
    )
    stats_final = pf_final.stats()
    tr_final = pf_final.trades.records
    exp_final_pips = expectancy_from_trades(tr_final)
    profit_final_usd = profit_usd_from_trades(tr_final)

    def _get(s, k):
        v = s.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            if hasattr(v, "values"):
                return float(np.asarray(v)[0])
            return float("nan")

    print("\n========== FINAL OPTIMIZED PORTFOLIO ==========")
    print(f"Total Trades        : {int(_get(stats_final,'Total Trades'))}")
    print(f"Win Rate [%]        : {_get(stats_final,'Win Rate [%]'):.2f}")
    print(f"Profit Factor       : {_get(stats_final,'Profit Factor'):.2f}")
    print(f"Total Return [%]    : {_get(stats_final,'Total Return [%]'):.2f}")
    print(f"Max Drawdown [%]    : {_get(stats_final,'Max Drawdown [%]'):.2f}")
    print(f"Expectancy (pips)   : {exp_final_pips:.2f} pips")
    print(f"Expectancy (USD)    : ${exp_final_pips * PIP_USD:.2f} /trade (1 lot)")
    print(f"Total Profit (USD)  : ${profit_final_usd:.0f} (1 lot)")

    # 9) Save grid & best config & summary
    grid_df = pd.DataFrame(rows)
    if len(grid_df) > 0:
        grid_df = grid_df.sort_values(
            ["Session","Regime","Total Profit (USD_1lot)","Profit Factor","Win Rate [%]"],
            ascending=[True, True, False, False, False]
        )
        grid_df.to_csv(OUT_GRID_CSV, index=False)

    best_json = {}
    for k,v in best_cfg.items():
        ss,rg = k
        fm, sm, tpv, slv, m = v
        best_json[f"{ss}__{rg}"] = {
            "fusion": fm, "stop": sm, "tp": tpv, "sl": slv,
            "profit_usd": m["Total Profit (USD_1lot)"],
            "pf": m["Profit Factor"],
            "win_rate": m["Win Rate [%]"],
            "exp_pips": m["Expectancy (pips)"],
            "exp_usd": m["Expectancy (USD_1lot)"]
        }
    with open(OUT_BEST_JSON, "w") as f:
        json.dump(best_json, f, indent=2)

    with open(OUT_SUMMARY, "w") as f:
        f.write("========== BEST CONFIG PER (SESSION × REGIME) ==========\n")
        for k in sorted(best_cfg.keys()):
            fm, sm, tpv, slv, m = best_cfg[k]
            f.write(f"{k[0]} × {k[1]}: {fm}+{sm} TP={tpv} SL={slv} "
                    f"| Profit=${m['Total Profit (USD_1lot)']:.0f} "
                    f"| PF={m['Profit Factor']:.2f} | Win={m['Win Rate [%]']:.2f}% "
                    f"| Exp={m['Expectancy (pips)']:.2f}p (${m['Expectancy (USD_1lot)']:.2f})\n")
        f.write("\n========== FINAL PORTFOLIO ==========\n")
        f.write(stats_final.to_string())
        f.write(f"\nExpectancy (pips): {exp_final_pips:.4f} | Expectancy USD: {exp_final_pips*PIP_USD:.2f} | Total Profit USD: {profit_final_usd:.0f}\n")

    print(f"\n✅ Saved grid → {OUT_GRID_CSV}")

    # --- Save best fusion signal for later stages ---
    try:
        best_row = grid_df.sort_values("Profit Factor", ascending=False).iloc[0]
        best_fusion = best_row["Fusion"]      # ✅ đổi tên cột
        best_stop   = best_row["Stop"]        # ✅ đổi tên cột

        # lấy signal tương ứng trong dict đã dùng ở trên
        best_signal = fusion_series[best_fusion].copy()  # ✅ chỉ cần fusion key thôi
        best_signal.name = "final_signal"
        best_signal.to_csv("logs/stage4_final_signal.csv", index=True)
        print(f"✅ Saved final fusion signal ({best_fusion}+{best_stop}) → logs/stage4_final_signal.csv")
    except Exception as e:
        print(f"⚠️ Could not save final fusion signal: {e}")