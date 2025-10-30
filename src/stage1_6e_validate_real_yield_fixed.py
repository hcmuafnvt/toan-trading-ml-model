import pandas as pd
import os

# ======================================================
# Stage 1.6e — Invert RealYield (non-USD) & Validate Correlation
# ======================================================
DATA_DIR = "data"
MACRO_PATH = os.path.join(DATA_DIR, "macro_context_v4.parquet")

pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
non_usd = {"EUR_USD", "GBP_USD", "XAU_USD"}

print("🚀 Stage 1.6e — Validate RealYield_shifted (with inverse for non-USD pairs)")

# 1️⃣ Load macro
macro = pd.read_parquet(MACRO_PATH)
macro.index = pd.to_datetime(macro.index)
macro["date"] = pd.to_datetime(macro["date"])
macro = macro.dropna(subset=["RealYield_shifted"])
macro = macro.set_index("date")

results = {}

# 2️⃣ For each pair → merge with macro and compute correlation
for pair in pairs:
    fx_path = os.path.join(DATA_DIR, f"{pair}_M5_clean.parquet")
    if not os.path.exists(fx_path):
        print(f"⚠️ Missing FX data: {fx_path}")
        continue

    fx = pd.read_parquet(fx_path)
    if "synthetic" in fx.columns:
        fx = fx[~fx["synthetic"].astype(bool)]
    fx["date"] = pd.to_datetime(fx.index.date)

    daily = fx.groupby("date")["close"].mean().to_frame("close")

    merged = pd.merge_asof(
        daily.sort_index(),
        macro[["RealYield_shifted"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    ).dropna()

    # apply inverse for non-USD base pairs
    merged["RealYield_fixed"] = (
        -merged["RealYield_shifted"] if pair in non_usd else merged["RealYield_shifted"]
    )

    corr = merged["close"].corr(merged["RealYield_fixed"])
    results[pair] = round(corr, 3)

print("\n📊 Correlation (close vs RealYield_fixed after +3d shift):")
for k, v in results.items():
    arrow = "↓" if v < 0 else "↑"
    print(f"   {k:<8}  corr={v:+.3f} {arrow}")