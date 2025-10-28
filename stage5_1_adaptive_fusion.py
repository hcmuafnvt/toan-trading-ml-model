# stage5_1_adaptive_fusion.py
# Gộp thông minh ML (Stage4) + MeanReversion (Stage4.2):
# - Chỉ thay ML bằng MR khi: ML==1 (timeout) và ((regime=='Low' & session=='Asia') hoặc (ML_confidence < CONF_TH))
# - TP/SL: lấy theo logs/stage4_best_config.json cho từng (session×regime); fallback fixed 10p nếu thiếu
# - Kết quả: Win%, PF, Expectancy (pips & USD), Total Profit USD
# - Lưu: logs/stage5_final_signal.csv, logs/stage5_summary.txt

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

ML_FINAL_SIGNAL_FILE = "logs/stage4_final_signal.csv"
MR_BEST_SIGNAL_FILE  = "logs/stage4_1_meanrev_best_signal.csv"
BEST_CFG_JSON        = "logs/stage4_best_config.json"

# (tùy chọn) để tính confidence nếu cần
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILES = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}
WINDOW = 200
STRIDE = 5

# OANDA mid_* mapping
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# Pips & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# ATR regime split
LOW_Q, HIGH_Q = 0.33, 0.66

# Khi confidence < ngưỡng này và ML==1 → cho phép MR override
CONF_TH = 0.30   # có thể nâng lên 0.35/0.40 nếu muốn lọc mạnh hơn

OUT_FINAL_SIG = "logs/stage5_final_signal.csv"
OUT_SUMMARY   = "logs/stage5_summary.txt"
os.makedirs("logs", exist_ok=True)

# ===================== HELPERS =====================
def safe_mean(x):
    if isinstance(x, (list, np.ndarray)):
        return float(np.asarray(x, dtype=float).mean())
    return float(x)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True).str.strip("_")
    return df

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
    df.index = pd.to_datetime(df.index).tz_localize(None)  # tz-naive
    return df

def atr_from_ta(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    tmp = price_df[["high","low","close"]].astype(float)
    atr = AverageTrueRange(tmp["high"], tmp["low"], tmp["close"], window=window).average_true_range()
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

def load_series_csv(path: str, name: str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.index = s.index.tz_localize(None)
    s.name = name
    return s

def compute_confidence_if_needed(price_idx: pd.DatetimeIndex) -> pd.Series:
    """
    Tính ML-confidence dựa trên 3 model + features (Stage 2).
    Nếu thiếu file/ model → trả về 0 để vô hiệu hóa nhánh confidence.
    """
    try:
        feat = pd.read_csv(FEATURE_FILE)
        feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target_cols = [c for c in feat.columns if c.startswith("target_")]
        X_full = feat.drop(columns=target_cols, errors="ignore")
        n_samples = len(X_full)

        sample_idx = np.arange(WINDOW, len(price_idx), STRIDE, dtype=int)
        if len(sample_idx) < n_samples:
            X_full = X_full.iloc[:len(sample_idx)].copy()
            n_samples = len(X_full)
        sample_times = pd.DatetimeIndex(price_idx)[sample_idx[:n_samples]]

        probs_list = []
        for name, path in MODEL_FILES.items():
            booster = lgb.Booster(model_file=path)
            feature_names = booster.feature_name()
            X = X_full.reindex(columns=feature_names, fill_value=0.0)
            probs = booster.predict(X.values, num_iteration=booster.best_iteration)
            probs_list.append(np.asarray(probs))

        probs_stack = np.stack(probs_list, axis=0)  # [3, n, 3]
        s = np.sort(probs_stack.sum(axis=0), axis=1)
        conf_sample = s[:, -1] - s[:, -2]
        conf = pd.Series(conf_sample, index=sample_times).reindex(price_idx, method="ffill").fillna(0.0)
        conf.name = "ml_conf"
        return conf.astype(float)
    except Exception:
        # fallback: không dùng confidence
        return pd.Series(0.0, index=price_idx, name="ml_conf")

def tp_sl_from_bestcfg(price: pd.DataFrame, session: pd.Series, regime: pd.Series, best_cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Tạo TP/SL series theo best config Stage 4 từng (session×regime)."""
    tp = pd.Series(np.nan, index=price.index, dtype=float)
    sl = pd.Series(np.nan, index=price.index, dtype=float)
    atr14 = None  # tính lazy khi cần

    for ss in ["Asia","London","NewYork"]:
        for rg in ["Low","Normal","High"]:
            key = f"{ss}__{rg}"
            mask = (session == ss) & (regime == rg)
            if key in best_cfg:
                cfg = best_cfg[key]
                stop = cfg.get("stop", "fixed")
                tpv  = float(cfg.get("tp", 10.0))
                slv  = float(cfg.get("sl", 10.0))

                if stop == "fixed":
                    tp.loc[mask] = tpv * PIP_SIZE
                    sl.loc[mask] = slv * PIP_SIZE
                else:  # atr
                    if atr14 is None:
                        atr14 = atr_from_ta(price, window=14)
                    tp.loc[mask] = (atr14.loc[mask] * tpv).clip(lower=1e-12)
                    sl.loc[mask] = (atr14.loc[mask] * slv).clip(lower=1e-12)
            # else: để NaN, sẽ fill default sau

    tp = tp.fillna(10 * PIP_SIZE)
    sl = sl.fillna(10 * PIP_SIZE)
    return tp, sl

# ===================== MAIN =====================
if __name__ == "__main__":
    # 1) Load price + regime/session
    price = load_price(DATA_FILE)
    close = price["close"]
    atr = atr_from_ta(price, window=14)
    q_low, q_high = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
    regime = pd.Series(np.where(atr <= q_low, "Low", np.where(atr >= q_high, "High", "Normal")), index=price.index)
    session = price.index.map(get_session)

    # 2) Load ML & MR signals
    ml_sig = load_series_csv(ML_FINAL_SIGNAL_FILE, "ml_signal").reindex(price.index, method="ffill").fillna(1).astype(int)
    mr_sig = load_series_csv(MR_BEST_SIGNAL_FILE, "mr_signal").reindex(price.index, method="ffill").fillna(1).astype(int)

    # 3) ML-confidence (optional)
    conf = compute_confidence_if_needed(price.index)

    # 4) Adaptive merge rule
    #    Chỉ override khi ML==1 và ((regime=='Low' & session=='Asia') hoặc (conf < CONF_TH))
    low_asia_mask = (regime == "Low") & (session == "Asia")
    weak_ml_mask  = conf < CONF_TH
    override_mask = ml_sig.eq(1) & (low_asia_mask | weak_ml_mask)

    final_sig = ml_sig.copy()
    # MR chỉ có 0/2 (vẫn giữ 1 ở nơi MR==1):
    final_sig.loc[override_mask] = mr_sig.loc[override_mask]

    # 5) TP/SL theo best config Stage 4 (per segment)
    best_cfg = {}
    if os.path.exists(BEST_CFG_JSON):
        with open(BEST_CFG_JSON, "r") as f:
            best_cfg = json.load(f)
    tp_series, sl_series = tp_sl_from_bestcfg(price, session, regime, best_cfg)

    # 6) Backtest (vectorbt)
    pf = vbt.Portfolio.from_signals(
        close,
        entries=final_sig.eq(2),
        exits=~final_sig.eq(2),
        short_entries=final_sig.eq(0),
        short_exits=~final_sig.eq(0),
        tp_stop=tp_series,
        sl_stop=sl_series,
        size=1.0,     # 1 lot chuẩn, không scale theo confidence để tách hiệu ứng
        fees=FEES,
        freq="5min"
    )
    st = pf.stats()
    tr = pf.trades.records
    exp_pips = expectancy_from_trades(tr)
    profit_usd = profit_usd_from_trades(tr)

    def _get(s, k):
        v = s.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            if hasattr(v, "values"):
                return float(np.asarray(v)[0])
            return float("nan")

    # 7) Báo cáo
    used_mr = int(override_mask.sum())
    used_mr_pct = 100.0 * used_mr / len(override_mask)
    print("\n========== STAGE 5.1: Adaptive Fusion (ML + MR) ==========")
    print(f"MR overrides on timeout: {used_mr} bars ({used_mr_pct:.2f}%)")
    print(f"Total Trades        : {int(_get(st,'Total Trades'))}")
    print(f"Win Rate [%]        : {_get(st,'Win Rate [%]'):.2f}")
    print(f"Profit Factor       : {_get(st,'Profit Factor'):.2f}")
    print(f"Max Drawdown [%]    : {_get(st,'Max Drawdown [%]'):.2f}")
    print(f"Expectancy (pips)   : {exp_pips:.2f}")
    print(f"Expectancy (USD)    : ${exp_pips * PIP_USD:.2f} /trade (1 lot)")
    print(f"Total Profit (USD)  : ${profit_usd:.0f} (1 lot)")

    # 8) Save
    final_sig.rename("final_signal_adaptive").to_csv(OUT_FINAL_SIG, index=True)
    with open(OUT_SUMMARY, "w") as f:
        f.write("========== STAGE 5.1: Adaptive Fusion (ML + MR) ==========\n")
        f.write(f"Overrides (bars): {used_mr} / {len(override_mask)} ({used_mr_pct:.2f}%)\n\n")
        f.write(st.to_string())
        f.write(f"\n\nExpectancy (pips): {exp_pips:.4f}\n")
        f.write(f"Expectancy (USD): {exp_pips * PIP_USD:.2f}\n")
        f.write(f"Total Profit (USD): {profit_usd:.0f}\n")

    print(f"✅ Saved final signal → {OUT_FINAL_SIG}")
    print(f"✅ Saved summary → {OUT_SUMMARY}")