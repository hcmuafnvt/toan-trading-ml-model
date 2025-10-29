import pandas as pd
from datetime import timedelta

PAIR = "GBP_USD"
GRANULARITY = "M5"  # nếu bạn đổi, nhớ sửa
FILE = f"data/{PAIR}_{GRANULARITY}_all.parquet"
GRAN_SECONDS = {"M1":60,"M5":300,"M15":900,"H1":3600}

print(f"🔍 Testing data quality: {FILE}")
df = pd.read_parquet(FILE)

# --- 1️⃣ Index ---
assert df.index.is_monotonic_increasing, "❌ Time index không tăng đều"
assert df.index.tz is not None, "❌ Index chưa có timezone (UTC)"
print(f"✅ Index OK ({len(df):,} rows from {df.index[0]} → {df.index[-1]})")

# --- 2️⃣ Columns ---
expected_cols = ["mid_o","mid_h","mid_l","mid_c","volume","close"]
for c in expected_cols:
    assert c in df.columns, f"❌ Thiếu cột {c}"
print("✅ Schema columns OK")

# --- 3️⃣ Numeric sanity ---
for c in ["mid_o","mid_h","mid_l","mid_c"]:
    assert (df[c] > 0).all(), f"❌ Giá âm hoặc 0 ở cột {c}"
assert (df["volume"] >= 0).all(), "❌ Volume âm"
print("✅ Value sanity OK")

# --- 4️⃣ Gaps ---
sec = GRAN_SECONDS[GRANULARITY]
diffs = df.index.to_series().diff().dt.total_seconds().dropna()
gaps = diffs[diffs > sec * 1.5]
if len(gaps):
    print(f"⚠️ Có {len(gaps)} gap > {sec}s (max gap = {gaps.max()}s)")
else:
    print("✅ Không có gap lớn")

# --- 5️⃣ NaN check ---
nan_ratio = df.isna().mean()
bad = nan_ratio[nan_ratio > 0.01]
if len(bad):
    print("⚠️ NaN >1%:", bad)
else:
    print("✅ Không có NaN đáng kể")

print("🎯 QC test hoàn tất – data đủ chuẩn cho Stage 2 (feature extraction).")