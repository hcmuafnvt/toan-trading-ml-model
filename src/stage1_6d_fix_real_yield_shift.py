import pandas as pd
import numpy as np
import os

# ======================================================
# Stage 1.6d — Fix RealYield Timing & Sign
# ======================================================
DATA_DIR = "data"
IN_FILE = os.path.join(DATA_DIR, "macro_context_v3.parquet")
OUT_FILE = os.path.join(DATA_DIR, "macro_context_v4.parquet")

print("🚀 Stage 1.6d — Fix RealYield Timing & Sign (+3d shift, inverse for non-USD pairs)")

# 1️⃣ Load macro dataset
macro = pd.read_parquet(IN_FILE)
macro.index = pd.to_datetime(macro.index)
if macro.index.tz is None:
    macro.index = macro.index.tz_localize("UTC")
else:
    macro.index = macro.index.tz_convert("UTC")

# 2️⃣ Shift RealYield forward +3 days (FX reacts 3 days later)
macro["RealYield_shifted"] = macro["RealYield"].shift(-3)

# 3️⃣ Save shifted macro context
macro.to_parquet(OUT_FILE)
print(f"✅ Saved shifted macro dataset → {OUT_FILE}")
print(f"   Columns: {list(macro.columns)}")
print(f"   Rows: {len(macro):,} ({macro.index[0]} → {macro.index[-1]})")

# 4️⃣ Preview sanity check
print("\n🔍 Sample preview (last 5 rows):")
print(macro[["RealYield", "RealYield_shifted"]].tail())