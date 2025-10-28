# stage6_wfa_hybrid.py
# Walk-Forward Validation (Hybrid):
# - Dùng features Stage 2 (CSV) + 3 models (T1/T2/T3) được train lại theo từng fold
# - Tối ưu fusion + stop theo (session × regime) trên train-fold (giống Stage 4)
# - Áp cấu hình sang test-fold -> backtest & tổng hợp

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import vectorbt as vbt
from ta.volatility import AverageTrueRange
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ===================== CONFIG =====================
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"  # từ stage2_extract_features_v3.py

# schema parquet (OANDA mid_*)
OHLC_MAP = {"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"}

# sampling như Stage 2/3/4
WINDOW = 200
STRIDE = 5

# pips & USD
PIP_SIZE = 0.0001
PIP_USD  = 10.0
FEES     = 0.0

# Fusion & Stops
FUSION_CANDIDATES = ["weighted", "majority", "buy_only"]
STOP_CANDIDATES = [
    ("fixed", 10.0, 10.0),
    ("fixed", 15.0, 15.0),
    ("fixed", 20.0, 20.0),
    ("atr",   1.5,  1.5),
    ("atr",   2.0,  2.0),
]

# Confidence filter
CONF_TH = 0.15
SIZE_MIN = 1.0
SIZE_MAX = 1.0  # giữ 1 lot cố định trong WFA

# WFA folds
N_FOLDS = 6
START_FRAC = 0.3         # train ban đầu = 30% samples đầu
TOPK_FEATURES = 250      # chọn theo importance mỗi model

# Regime theo ATR quantile
LOW_Q, HIGH_Q = 0.33, 0.66

OUT_FOLDS_CSV = "logs/stage6_wfa_folds.csv"
OUT_SUMMARY   = "logs/stage6_wfa_summary.txt"
os.makedirs("logs", exist_ok=True)

# ===================== HELPERS =====================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                  .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
                  .str.strip("_"))
    return df

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
    df.index = pd.to_datetime(df.index).tz_localize(None)
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

def make_folds(n_samples: int, n_folds: int, start_frac: float):
    start = int(n_samples * start_frac)
    # chia phần còn lại thành n_folds phần test gần bằng nhau (expanding train)
    remain = n_samples - start
    fold_sizes = [remain // n_folds] * n_folds
    for i in range(remain % n_folds):
        fold_sizes[i] += 1
    folds = []
    train_end = start
    for fs in fold_sizes:
        test_start = train_end
        test_end = test_start + fs
        folds.append((slice(0, train_end), slice(test_start, test_end)))
        train_end = test_end
    return folds

def get_confidence_from_probs(sum_probs: np.ndarray) -> np.ndarray:
    s = np.sort(sum_probs, axis=1)
    return s[:,-1] - s[:,-2]

def scale_to_range(x: np.ndarray, lo=1.0, hi=1.0) -> np.ndarray:
    # giữ 1 lot cố định
    return np.full_like(x, lo, dtype=float)

def expectancy_from_trades(tr_rec: pd.DataFrame, pip_size=PIP_SIZE) -> float:
    if tr_rec is None or len(tr_rec) == 0:
        return 0.0
    entry = tr_rec["entry_price"].values
    exitp = tr_rec["exit_price"].values
    direction = tr_rec["direction"].values  # 0=long,1=short
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

def backtest_from_signal(price: pd.DataFrame,
                         signal: pd.Series,
                         stop_mode: str, tp_param: float, sl_param: float,
                         fees=FEES, freq="5min"):
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

    size_series = pd.Series(1.0, index=price.index)
    pf = vbt.Portfolio.from_signals(
        price["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        tp_stop=tp,
        sl_stop=sl,
        size=size_series,
        fees=fees,
        freq=freq
    )
    stats = pf.stats()
    tr = pf.trades.records
    exp_pips = expectancy_from_trades(tr)
    profit_usd = profit_usd_from_trades(tr)
    return pf, stats, exp_pips, profit_usd

def fuse_preds(preds_stack, probs_stack, times_index, price_index, mode, conf_th=CONF_TH):
    # preds_stack: [3, n] ; probs_stack: [3, n, 3]
    if mode == "majority":
        vote0 = (preds_stack == 0).sum(axis=0)
        vote1 = (preds_stack == 1).sum(axis=0)
        vote2 = (preds_stack == 2).sum(axis=0)
        vote = np.select([vote0 >= vote1, vote2 >= vote1], [0, 2], default=1)
        sig = pd.Series(vote, index=times_index)
    elif mode == "weighted":
        sum_probs = probs_stack.sum(axis=0)  # [n,3]
        sig = pd.Series(sum_probs.argmax(axis=1), index=times_index)
    elif mode == "buy_only":
        sum_probs = probs_stack.sum(axis=0)
        hard = np.where(sum_probs.argmax(axis=1) == 2, 2, 1)  # 2 else 1
        sig = pd.Series(hard, index=times_index)
    else:
        raise ValueError("Unknown fusion mode")

    # Confidence filter
    sum_probs = probs_stack.sum(axis=0)
    conf = get_confidence_from_probs(sum_probs)
    conf_series = pd.Series(conf, index=times_index).reindex(price_index, method="ffill").fillna(0.0)

    out = sig.reindex(price_index, method="ffill").fillna(1).astype(int)
    out[conf_series < conf_th] = 1
    return out

def pick_best_config(price, signal_train, session_train, regime_train):
    # grid search trên TRAIN bằng (session × regime)
    masks = {}
    sess_vals = ["Asia","London","NewYork"]
    reg_vals  = ["Low","Normal","High"]
    for ss in sess_vals:
        for rg in reg_vals:
            masks[(ss,rg)] = (session_train == ss) & (regime_train == rg)

    best_cfg = {}
    for ss in sess_vals:
        for rg in reg_vals:
            mask = masks[(ss,rg)]
            if mask.sum() < 1000:
                continue
            # giữ tín hiệu ngoài segment là timeout để không lẫn shape
            sig_seg = signal_train.copy()
            sig_seg[~mask] = 1
            best = None
            for (stop_mode, tp, sl) in STOP_CANDIDATES:
                _, st, _, usd = backtest_from_signal(price, sig_seg, stop_mode, tp, sl)
                # lấy Profit USD làm tiêu chí
                v = st.get("Total Return [%]", np.nan)  # chỉ để tránh crash nếu thiếu keys
                profit = usd
                pf_val = float(st.get("Profit Factor", np.nan))
                wr_val = float(st.get("Win Rate [%]", np.nan))
                key = (profit, pf_val, wr_val)
                if (best is None) or (key > best[0]):
                    best = (key, stop_mode, tp, sl)
            if best is not None:
                best_cfg[(ss,rg)] = (best[1], best[2], best[3])
    return best_cfg

def apply_cfg_on_test(price, signal_full, session_full, regime_full, best_cfg):
    # lắp stop theo (session × regime) chọn được
    final_signal = pd.Series(1, index=price.index, dtype=int)
    final_tp = pd.Series(np.nan, index=price.index, dtype=float)
    final_sl = pd.Series(np.nan, index=price.index, dtype=float)

    for (ss,rg), (stop_mode, tpv, slv) in best_cfg.items():
        mask = (session_full == ss) & (regime_full == rg)
        if mask.sum() == 0:
            continue
        final_signal.loc[mask] = signal_full.loc[mask]
        if stop_mode == "fixed":
            final_tp.loc[mask] = tpv * PIP_SIZE
            final_sl.loc[mask] = slv * PIP_SIZE
        else:
            atr14 = atr_from_ta(price, window=14)
            final_tp.loc[mask] = (atr14.loc[mask] * tpv).clip(lower=1e-12)
            final_sl.loc[mask] = (atr14.loc[mask] * slv).clip(lower=1e-12)

    # fill NaN
    final_tp = final_tp.fillna(10 * PIP_SIZE)
    final_sl = final_sl.fillna(10 * PIP_SIZE)
    return final_signal, final_tp, final_sl

# ===================== MAIN =====================
if __name__ == "__main__":
    # 1) Load price
    print(f"⏳ Loading price: {DATA_FILE}")
    price = load_price(DATA_FILE)
    print(f"✅ Loaded price: {len(price):,} rows | {price.index[0]} → {price.index[-1]}")

    # 2) Regime & session series (full)
    atr14_full = atr_from_ta(price, window=14)
    q_low, q_high = atr14_full.quantile(LOW_Q), atr14_full.quantile(HIGH_Q)
    regime_full = pd.Series(np.where(atr14_full <= q_low, "Low",
                               np.where(atr14_full >= q_high, "High", "Normal")), index=price.index)
    session_full = price.index.map(get_session)

    # 3) Load features CSV (Stage 2)
    print(f"⏳ Loading features: {FEATURE_FILE}")
    feat = pd.read_csv(FEATURE_FILE)
    feat = clean_cols(feat).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_cols = [c for c in feat.columns if c.startswith("target_")]
    assert len(target_cols) == 3, "Stage2 features phải có target_10x40/15x60/20x80"

    X_all = feat.drop(columns=target_cols, errors="ignore")
    y_T1 = feat["target_10x40"].astype(int).values
    y_T2 = feat["target_15x60"].astype(int).values
    y_T3 = feat["target_20x80"].astype(int).values
    n_samples = len(X_all)
    print(f"✅ Features: {X_all.shape} | targets len={n_samples}")

    # 4) Rebuild sample_times để align với price
    sample_idx = np.arange(WINDOW, len(price), STRIDE, dtype=int)
    sample_idx = sample_idx[:n_samples]
    sample_times = price.index[sample_idx]

    # Đảm bảo session_full và regime_full là Series, không phải Index
    if not isinstance(session_full, pd.Series):
        session_full = pd.Series(session_full, index=price.index)
    if not isinstance(regime_full, pd.Series):
        regime_full = pd.Series(regime_full, index=price.index)

    # Dùng sample_times để tạo session/regime sample-level
    sess_samp = session_full.reindex(sample_times)
    reg_samp  = regime_full.reindex(sample_times)

    # 5) Create WFA folds theo sample order
    folds = make_folds(n_samples, N_FOLDS, START_FRAC)
    print(f"========== STAGE 6.2: Hybrid WFA ==========")
    print(f"Total samples: {n_samples} | folds={len(folds)}")

    rows = []
    all_test_stats = []

    for fi, (tr_slice, te_slice) in enumerate(folds, start=1):
        tr_idx = np.arange(n_samples)[tr_slice]
        te_idx = np.arange(n_samples)[te_slice]
        if len(tr_idx) < 2000 or len(te_idx) < 500:
            continue

        print(f"\n--- Fold {fi} ---")
        print(f"Train samples: {len(tr_idx)} | Test samples: {len(te_idx)}")

        # 5.1 Train 3 models với class_weight balanced
        def train_one(X, y, name):
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=63,
                min_data_in_leaf=96,
                feature_fraction=0.7,
                bagging_fraction=0.8,
                bagging_freq=1,
                class_weight="balanced",
                n_jobs=-1
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            # chọn top-k features
            imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            top_cols = imp.head(min(TOPK_FEATURES, len(imp))).index.tolist()
            # Lưu top_cols theo đúng thứ tự gốc trong X
            top_cols = [c for c in X.columns if c in top_cols]
            return clf, top_cols

        # top-k theo từng model (tách cột)
        X_train_full = X_all.iloc[tr_idx].copy()
        X_test_full  = X_all.iloc[te_idx].copy()

        m1, cols1 = train_one(X_train_full, y_T1[tr_idx], "T1")
        m2, cols2 = train_one(X_train_full, y_T2[tr_idx], "T2")
        m3, cols3 = train_one(X_train_full, y_T3[tr_idx], "T3")

        # 5.2 Predict train-fold để optimize fusion+stop theo (session×regime)
        def predict_stack(model, cols, X_train, X_test):
            # Chỉ dùng đúng top feature columns đã train
            Xtr = X_train[cols].copy()
            Xte = X_test[cols].copy()

            # Đảm bảo cùng số cột
            assert Xtr.shape[1] == len(cols) and Xte.shape[1] == len(cols), \
                f"Mismatch feature count: train={Xtr.shape}, test={Xte.shape}"

            p_tr = model.predict_proba(Xtr, num_iteration=model.best_iteration_)
            y_tr = p_tr.argmax(axis=1)

            p_te = model.predict_proba(Xte, num_iteration=model.best_iteration_)
            y_te = p_te.argmax(axis=1)

            return (y_tr, p_tr), (y_te, p_te)

        (y1_tr, p1_tr), (y1_te, p1_te) = predict_stack(m1, cols1, X_all.iloc[tr_idx], X_all.iloc[te_idx])
        (y2_tr, p2_tr), (y2_te, p2_te) = predict_stack(m2, cols2, X_all.iloc[tr_idx], X_all.iloc[te_idx])
        (y3_tr, p3_tr), (y3_te, p3_te) = predict_stack(m3, cols3, X_all.iloc[tr_idx], X_all.iloc[te_idx])

        preds_tr_stack = np.stack([y1_tr, y2_tr, y3_tr], axis=0)
        probs_tr_stack = np.stack([p1_tr, p2_tr, p3_tr], axis=0)

        preds_te_stack = np.stack([y1_te, y2_te, y3_te], axis=0)
        probs_te_stack = np.stack([p1_te, p2_te, p3_te], axis=0)

        # 5.3 Fusion per mode trên TRAIN để chọn signal base
        sig_train_dict = {}
        for fm in FUSION_CANDIDATES:
            sig_train_dict[fm] = fuse_preds(
                preds_tr_stack, probs_tr_stack,
                times_index=sample_times[tr_idx],
                price_index=price.index,
                mode=fm, conf_th=CONF_TH
            )

        # 5.4 Build session/regime vector trên TRAIN
        session_train = pd.Series(session_full.loc[price.index].values, index=price.index)
        regime_train  = pd.Series(regime_full.loc[price.index].values, index=price.index)

        # 5.5 Chọn best stop theo (session×regime) cho từng fusion -> pick fusion tốt nhất theo Profit USD
        fm_best = None
        cfg_best = None
        usd_best = -1e18

        for fm in FUSION_CANDIDATES:
            sig_train = sig_train_dict[fm]
            cfg = pick_best_config(price, sig_train, session_train, regime_train)
            # áp thử trên TRAIN để chấm điểm (USD)
            fin_sig, fin_tp, fin_sl = apply_cfg_on_test(price, sig_train, session_train, regime_train, cfg)
            pf, st, _, usd = backtest_from_signal(price, fin_sig, "fixed", 10, 10)  # tp/sl dummy vì đã set series
            usd = profit_usd_from_trades(pf.trades.records)
            if usd > usd_best:
                usd_best = usd
                fm_best = fm
                cfg_best = cfg

        print(f"Fold {fi}: chosen fusion='{fm_best}' on TRAIN (Profit USD={usd_best:.0f})")

        # 5.6 Áp cấu hình sang TEST
        #   - tạo signal test theo fusion đã chọn
        sig_test_base = fuse_preds(
            preds_te_stack, probs_te_stack,
            times_index=sample_times[te_idx],
            price_index=price.index,
            mode=fm_best, conf_th=CONF_TH
        )

        #   - sessions/regimes full
        session_series = pd.Series(session_full.loc[price.index].values, index=price.index)
        regime_series  = pd.Series(regime_full.loc[price.index].values, index=price.index)
        #   - lắp stop theo cfg_best
        fin_sig_test, fin_tp_test, fin_sl_test = apply_cfg_on_test(
            price, sig_test_base, session_series, regime_series, cfg_best
        )

        #   - mask chỉ test-window
        test_time_mask = (price.index >= sample_times[te_idx][0]) & (price.index <= sample_times[te_idx][-1])
        fin_sig_test = fin_sig_test.where(test_time_mask, 1)
        fin_tp_test  = fin_tp_test.where(test_time_mask, 10*PIP_SIZE)
        fin_sl_test  = fin_sl_test.where(test_time_mask, 10*PIP_SIZE)

        pf = vbt.Portfolio.from_signals(
            price["close"],
            entries=fin_sig_test.eq(2),
            exits=~fin_sig_test.eq(2),
            short_entries=fin_sig_test.eq(0),
            short_exits=~fin_sig_test.eq(0),
            tp_stop=fin_tp_test,
            sl_stop=fin_sl_test,
            size=1.0,
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

        fold_row = {
            "Fold": fi,
            "Train_Samples": len(tr_idx),
            "Test_Samples": len(te_idx),
            "Fusion": fm_best,
            "Total Trades": int(_get(st,"Total Trades")),
            "Win Rate [%]": _get(st,"Win Rate [%]"),
            "Profit Factor": _get(st,"Profit Factor"),
            "Max Drawdown [%]": _get(st,"Max Drawdown [%]"),
            "Expectancy (pips)": exp_pips,
            "Expectancy (USD_1lot)": exp_pips * PIP_USD,
            "Total Profit (USD_1lot)": profit_usd
        }
        rows.append(fold_row)
        all_test_stats.append((pf, st, exp_pips, profit_usd))
        print(f"Fold {fi} → Trades={fold_row['Total Trades']} | Win%={fold_row['Win Rate [%]']:.2f} "
              f"| PF={fold_row['Profit Factor']:.2f} | Exp={exp_pips:.2f}p (${exp_pips*PIP_USD:.2f}) "
              f"| Profit=${profit_usd:.0f}")

    # 6) Tổng hợp kết quả
    folds_df = pd.DataFrame(rows)
    if len(folds_df) > 0:
        folds_df.to_csv(OUT_FOLDS_CSV, index=False)

        # gộp metric theo tổng trade
        total_trades = folds_df["Total Trades"].sum()
        total_profit = folds_df["Total Profit (USD_1lot)"].sum()
        # PF tổng xấp xỉ: sum(gain)/sum(loss) (không hoàn hảo nếu thiếu breakdown)
        # nên dùng vectorbt concat thì nặng; tạm báo cáo trung bình có trọng số theo trades
        wr = np.average(folds_df["Win Rate [%]"], weights=folds_df["Total Trades"])
        exp_pips = np.average(folds_df["Expectancy (pips)"], weights=folds_df["Total Trades"])
        dd = folds_df["Max Drawdown [%]"].max()  # bảo thủ

        with open(OUT_SUMMARY, "w") as f:
            f.write("========== STAGE 6.2: Hybrid WFA ==========\n")
            f.write(folds_df.to_string(index=False))
            f.write("\n\n----- OVERALL (weighted by trades) -----\n")
            f.write(f"Total Trades        : {int(total_trades)}\n")
            f.write(f"Win Rate [%]        : {wr:.2f}\n")
            f.write(f"Expectancy (pips)   : {exp_pips:.2f}\n")
            f.write(f"Expectancy (USD)    : ${exp_pips*PIP_USD:.2f} /trade (1 lot)\n")
            f.write(f"Total Profit (USD)  : ${total_profit:.0f} (1 lot)\n")
            f.write(f"Max Drawdown [%]    : {dd:.2f}\n")

        print("\n========== OVERALL (weighted by trades) ==========")
        print(f"Total Trades        : {int(total_trades)}")
        print(f"Win Rate [%]        : {wr:.2f}")
        print(f"Expectancy (pips)   : {exp_pips:.2f}")
        print(f"Expectancy (USD)    : ${exp_pips*PIP_USD:.2f} /trade (1 lot)")
        print(f"Total Profit (USD)  : ${total_profit:.0f} (1 lot)")
        print(f"Max Drawdown [%]    : {dd:.2f}")

        print(f"\n✅ Saved folds → {OUT_FOLDS_CSV}")
        print(f"✅ Saved summary → {OUT_SUMMARY}")
    else:
        print("⚠️ No folds produced results. Check configuration.")