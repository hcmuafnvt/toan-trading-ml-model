import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# Stage 1.6h-fix — Compare interpolation vs timezone shift
# Goal:
#   See which method restores correlation with USDJPY best.
# ======================================================

DATA_DIR = "data"
MACRO_FILE = os.path.join(DATA_DIR, "macro_context_v6.parquet")
PAIR_FILE = os.path.join(DATA_DIR, "USD_JPY_M5_clean.parquet")

print("🚀 Stage 1.6h-fix — Testing correlation recovery methods")

# --- Load macro ---
macro = pd.read_parquet(MACRO_FILE)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

# --- Load USDJPY data ---
fx = pd.read_parquet(PAIR_FILE)
if "synthetic" in fx.columns:
    fx = fx[~fx["synthetic"].astype(bool)]
fx.index = pd.to_datetime(fx.index)
if fx.index.tz is None:
    fx.index = fx.index.tz_localize("UTC")
else:
    fx.index = fx.index.tz_convert("UTC")

fx_daily = fx.resample("1D")["close"].mean().dropna().to_frame("USDJPY_close")

# ------------------------------------------------------
# Option A — Interpolated Macro
# ------------------------------------------------------
macroA = macro.copy()
for col in ["DGS10", "JGB10Y"]:
    if col in macroA.columns:
        macroA[col] = macroA[col].interpolate(limit_direction="both")
macroA["JPY_driver_interp"] = macroA["DGS10"] - macroA["JGB10Y"]

macroA_daily = macroA.resample("1D")["JPY_driver_interp"].last().dropna().to_frame("JPY_driver_interp")

mergedA = pd.merge_asof(
    fx_daily.sort_index(),
    macroA_daily.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

corrA = mergedA["USDJPY_close"].corr(mergedA["JPY_driver_interp"])

# ------------------------------------------------------
# Option B — Timezone +9h shift (Tokyo align)
# ------------------------------------------------------
macroB = macro.copy()
macroB.index = macroB.index + pd.Timedelta(hours=9)
macroB["JPY_driver_shift"] = macroB["DGS10"] - macroB["JGB10Y"]
macroB_daily = macroB.resample("1D")["JPY_driver_shift"].last().dropna().to_frame("JPY_driver_shift")

mergedB = pd.merge_asof(
    fx_daily.sort_index(),
    macroB_daily.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
).dropna()

corrB = mergedB["USDJPY_close"].corr(mergedB["JPY_driver_shift"])

# ------------------------------------------------------
# Compare results
# ------------------------------------------------------
def mark(v): return f"{v:+.3f} ↑" if v > 0 else f"{v:+.3f} ↓"

print(f"\n📊 Correlation comparison (USDJPY vs JPY_driver)")
print(f"   Option A — Interpolated  : {mark(corrA)}")
print(f"   Option B — +9h shift      : {mark(corrB)}")

if abs(corrA) > abs(corrB):
    best = "A (Interpolation)"
    best_val = corrA
else:
    best = "B (Tokyo +9h Shift)"
    best_val = corrB

print(f"\n🏁 Best correlation method → {best}  | corr={best_val:.3f}")