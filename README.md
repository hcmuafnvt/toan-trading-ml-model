───────────────────────────────────────────────────────────────
             PHASE 1 — DATA FOUNDATION (What is the world?)
───────────────────────────────────────────────────────────────
│
├── 📊 FX Price Layer (OANDA)
│     ├─ GBP_USD, EUR_USD, USD_JPY, XAU_USD
│     └─ Cleaned, resampled M5 data (2023–2025+)
│
├── 🌐 Macro Layer (FRED + Yahoo)
│     ├─ DXY, UST2Y, DGS10, CPI_YoY, DFII10 (TIPS real yield)
│     ├─ VIX, SPX, JGB10Y, BoJRate
│     ├─ Yield spreads (US–EU, US–UK, US–JP)
│     └─ RealYield_shifted (+3d), macro_context_master.parquet
│
├── ⚙️ Regime & Event Layer (Stage 1.8 → soon)
│     ├─ Risk-On/Off flags (VIX > 20)
│     ├─ Event windows: FOMC, CPI, NFP, BoJ interventions
│     └─ Output → regime_context.parquet
│
└── ✅ Unified Macro Context
      ↓  (daily→M5 alignment)
      data/macro_context_master.parquet
───────────────────────────────────────────────────────────────
             PHASE 2 — FEATURE ENGINEERING (How it behaves?)
───────────────────────────────────────────────────────────────
│
├── 🧩 Technical Features (tsfresh + custom)
│     ├─ Momentum, volatility, EMA, ATR, RSI
│     └─ Intraday session / overlap metrics
│
├── 🧠 Macro Features
│     ├─ RealYield_zscore, ΔTIPS_real, YieldSpread_zscore
│     ├─ RiskOffFlag, DXY_zscore, SPX_momentum
│     └─ regime_interaction = f(macro × price)
│
└── 💾 Output → logs/stage2_features.csv
───────────────────────────────────────────────────────────────
             PHASE 3 — MODEL TRAINING (What’s likely next?)
───────────────────────────────────────────────────────────────
│
├── 🎯 Per-pair classifiers (LightGBM)
│     ├─ y ∈ {0: down, 1: flat, 2: up}
│     ├─ train/valid/test, macro-aware features
│     └─ metrics: accuracy, F1, regime-split performance
│
├── 📈 Model explainability
│     ├─ SHAP importance by macro group
│     ├─ Feature ranking vs alpha stability
│     └─ Output → logs/stage3_feature_importance.csv
│
└── 💾 Output → logs/stage3_model_lgbm.txt
───────────────────────────────────────────────────────────────
             PHASE 4 — ENSEMBLE & META-MODEL (When to act?)
───────────────────────────────────────────────────────────────
│
├── 🧮 Meta-Model = f(α_EUR, α_GBP, α_JPY, α_XAU)
│     ├─ Learns global USD regime
│     ├─ Chooses which pairs to trade / skip
│
├── ⚔️ Risk Allocation Engine
│     ├─ Vol-scaled sizing
│     ├─ Correlation-adjusted exposure
│
└── 📊 Output → daily “Alpha Dashboard” (bias + size)
───────────────────────────────────────────────────────────────
             PHASE 5 — BACKTEST & LIVE DEPLOYMENT
───────────────────────────────────────────────────────────────
│
├── ⏱  Walk-forward evaluation (WF-DUAL-OPT)
│     ├─ Rolling 90-day retrain / 14-day test
│     └─ Equity, drawdown, PF, Sharpe, Stability
│
└── 💰 Live trading engine
      ├─ Auto-update macro + event feed
      └─ OANDA REST execution layer
───────────────────────────────────────────────────────────────