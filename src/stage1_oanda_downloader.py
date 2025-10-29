#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.1 — OANDA Data Downloader (Multi-instrument, Config-driven, Dotenv, Fund-grade QA)

Role (Introduce Page): Quant Research Lead
- Chọn frameworks, định nghĩa stage
- Kiểm tra data quality (fund-level clarity)
- Mặc định chính sách dữ liệu hợp lý (không cần user đoán)

Highlights
- Nhiều instrument trong 1 lần chạy (GBP_USD, USD_JPY, XAU_USD ...).
- Config ngay trong file (không dùng argparse).
- Tự load .env (OANDA_API_KEY, optional PROXY).
- Daily incremental update (last_ts + 1s → now()-1min).
- Fund-grade QC: duplicates, monotonic, NaN ratio, expected vs actual bars, gap detection.
- Chuẩn schema: DateTimeIndex(UTC), columns: mid_o, mid_h, mid_l, mid_c, volume, close (+ bid_*, ask_* nếu chọn BAM).

Usage
1) Tạo .env (cùng thư mục project gốc):
    OANDA_API_KEY=your_real_oanda_token
    # PROXY=http://user:pass@host:port  (nếu cần)

2) Cài deps:
    pip install pandas requests pyarrow python-dotenv

3) Chạy:
    python3 src/oanda_downloader.py
"""

import os
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ---- Load environment (.env) ----
load_dotenv()
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
if not OANDA_API_KEY:
    raise SystemExit("❌ Missing OANDA_API_KEY in environment (.env).")

PROXY = os.getenv("PROXY")  # optional, e.g. http://user:pass@host:port

# ---- API hosts ----
OANDA_HOSTS = {
    "fxtrade": "https://api-fxtrade.oanda.com",
    "fxpractice": "https://api-fxpractice.oanda.com",
}

# ---- Granularity seconds map ----
GRAN_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600,
    "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
    "D": 86400,
}

# =========================
# CONFIG (no CLI needed)
# =========================
CONFIG = {
    # Danh sách instrument cần tải (thêm/bớt tuỳ ý)
    "instruments": ["EUR_USD"],

    # Tham số chung
    "granularity": "M5",           # S5..D
    "price": "BAM",                # B, A, M, BA, BM, AM, BAM
    "env": "fxtrade",              # fxtrade | fxpractice
    "max_per_request": 5000,       # safety chunk per API call
    "daily_update": True,          # incremental nếu file đã có

    # Data horizon policy (Quant default ≈ 3 năm gần nhất)
    "start_date": "2023-01-01T00:00:00Z",  # nếu chưa có file thì dùng mốc này
    "end_date": None,                      # None → now(UTC) - 1 phút

    # Đường dẫn output (mặc định theo pattern dưới)
    # Nếu muốn tuỳ chỉnh từng instrument, có thể tạo dict map riêng (không bắt buộc)
    "output_pattern": "data/{instrument}_{gran}_all.parquet",

    # Logs
    "qc_report_pattern": "logs/oanda_qc_report_{instrument}.txt",
    "qc_summary_path": "logs/oanda_qc_summary.txt",
}

# =========================
# HTTP session + helpers
# =========================
SESSION = requests.Session()
if PROXY:
    SESSION.proxies.update({"http": PROXY, "https": PROXY})

def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}

def _host(env: str) -> str:
    if env not in OANDA_HOSTS:
        raise ValueError(f"Invalid env '{env}'. Use one of {list(OANDA_HOSTS.keys())}")
    return OANDA_HOSTS[env]

def _oanda_get(url: str, params: Dict[str, str]) -> Dict:
    for attempt in range(6):
        try:
            r = SESSION.get(url, headers=_headers(), params=params, timeout=60)
            if r.status_code == 429:
                # Rate-limited → backoff
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 5:
                raise
            print(f"⚠️ Retry {attempt+1} after error: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Unreachable")

# =========================
# Time helpers
# =========================
def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _compute_chunk_seconds(granularity: str, max_per_request: int) -> int:
    sec = GRAN_SECONDS[granularity]
    return math.floor(sec * max_per_request * 0.98)  # margin tránh off-by-one

# =========================
# IO helpers
# =========================
def read_existing_parquet(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
    except FileNotFoundError:
        return None

def _normalize_candles(candles: List[Dict], price_type: str) -> pd.DataFrame:
    want_b = "B" in price_type
    want_a = "A" in price_type
    want_m = "M" in price_type

    recs = []
    for c in candles:
        t = parse_iso(c["time"])
        r = {"time": t, "volume": c.get("volume")}
        if want_m and "mid" in c:
            r.update({f"mid_{k}": float(c["mid"][k]) for k in "ohlc"})
        if want_b and "bid" in c:
            r.update({f"bid_{k}": float(c["bid"][k]) for k in "ohlc"})
        if want_a and "ask" in c:
            r.update({f"ask_{k}": float(c["ask"][k]) for k in "ohlc"})
        recs.append(r)

    df = pd.DataFrame.from_records(recs).set_index("time").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def _postprocess_schema(df: pd.DataFrame) -> pd.DataFrame:
    # enforce numeric
    for p in ("mid", "bid", "ask"):
        for k in "ohlc":
            col = f"{p}_{k}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # alias
    df["close"] = df.get("mid_c", pd.NA)

    # strict order
    ordered = ["mid_o","mid_h","mid_l","mid_c","volume","close",
               "bid_o","bid_h","bid_l","bid_c","ask_o","ask_h","ask_l","ask_c"]
    for col in ordered:
        if col not in df.columns:
            df[col] = pd.NA

    # dedup + sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[ordered]

def _merge_dedup(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    out = pd.concat([old, new]).sort_index()
    return out[~out.index.duplicated(keep="last")]

# =========================
# QC — Fund-grade checks
# =========================
def _qc_report(df: pd.DataFrame, instrument: str, granularity: str,
               start_req: datetime, end_req: datetime) -> str:
    lines = []
    lines.append("=== OANDA QC REPORT (Stage 1.1) ===")
    lines.append(f"Instrument     : {instrument}")
    lines.append(f"Granularity    : {granularity}")
    lines.append(f"Range requested: {start_req.isoformat()} → {end_req.isoformat()}")
    lines.append(f"Rows total     : {len(df):,}")

    if len(df) > 0:
        lines.append(f"First ts       : {df.index[0].isoformat()}")
        lines.append(f"Last ts        : {df.index[-1].isoformat()}")

    dup_count = df.index.duplicated(keep="last").sum()
    lines.append(f"Duplicates     : {dup_count}")
    lines.append(f"Monotonic time : {df.index.is_monotonic_increasing}")

    # NaN ratios (top 6)
    nan_stats = df.isna().mean().sort_values(ascending=False)
    top_nan = nan_stats.head(6)
    lines.append("NaN ratio top6 : " + ", ".join([f"{c}={v:.2%}" for c, v in top_nan.items()]))

    # Expected vs actual
    sec = GRAN_SECONDS[granularity]
    expected = int((end_req - start_req).total_seconds() // sec) + 1
    coverage = len(df)/max(expected, 1)
    lines.append(f"Expected bars  : ~{expected:,} (calendar-based)")
    lines.append(f"Coverage ratio : {coverage:.2%}")

    # Gaps (>1 step)
    gaps = []
    if len(df) > 1:
        diffs = (df.index[1:] - df.index[:-1]).to_series(index=df.index[1:])
        step = timedelta(seconds=sec)
        big_gaps = diffs[diffs > step]
        for ts, delta in big_gaps.items():
            prev = ts - delta
            gaps.append((prev, ts, int(delta.total_seconds())))
    lines.append(f"Gaps (>1 step) : {len(gaps)}")
    for g in gaps[:5]:
        lines.append(f"  - {g[0].isoformat()} → {g[1].isoformat()} ({g[2]}s)")

    return "\n".join(lines)

# =========================
# Core downloader (per instrument)
# =========================
def fetch_for_instrument(instrument: str, cfg: Dict) -> Dict:
    gran = cfg["granularity"]
    price = cfg["price"]
    env = cfg["env"]
    out_path = cfg["output_pattern"].format(instrument=instrument, gran=gran)
    base_url = _host(env) + f"/v3/instruments/{instrument}/candles"

    # Range decision
    existing = read_existing_parquet(out_path)
    if existing is not None and not existing.empty and cfg["daily_update"]:
        start = existing.index[-1] + timedelta(seconds=1)
        print(f"🔁 [{instrument}] Incremental from {start}")
    else:
        start = parse_iso(cfg["start_date"])
        print(f"🆕 [{instrument}] Fresh from {start}")

    end = parse_iso(cfg["end_date"]) if cfg.get("end_date") else (datetime.now(timezone.utc) - timedelta(minutes=1))
    if start >= end:
        print(f"✅ [{instrument}] Up to date.")
        merged = existing if existing is not None else pd.DataFrame()
        s_qc = merged.index[0] if len(merged) else start
        e_qc = merged.index[-1] if len(merged) else end
        report = _qc_report(merged, instrument, gran, s_qc, e_qc)
        _write_text(cfg["qc_report_pattern"].format(instrument=instrument), report)
        return {"instrument": instrument, "rows": len(merged), "coverage": None, "report_path": cfg["qc_report_pattern"].format(instrument=instrument)}

    # Chunk loop
    chunk_seconds = _compute_chunk_seconds(gran, cfg["max_per_request"])
    frames: List[pd.DataFrame] = []
    cursor = start
    last_progress = None
    while cursor < end:
        chunk_end = min(cursor + timedelta(seconds=chunk_seconds), end)
        params = {
            "granularity": gran,
            "price": price,
            "from": to_iso(cursor),
            "to": to_iso(chunk_end),
            "includeFirst": "true",
        }
        data = _oanda_get(base_url, params)
        candles = data.get("candles", [])

        if not candles:
            # Không có candle nào mới → dừng nếu cursor không tiến
            if last_progress == cursor:
                print(f"⚠️  [{instrument}] No new data after {cursor}, stopping to avoid infinite loop.")
                break
            last_progress = cursor
            cursor = chunk_end
            continue

        df = _normalize_candles(candles, price)
        frames.append(df)
        new_cursor = df.index.max() + timedelta(seconds=1)

        if new_cursor <= cursor:
            print(f"⚠️  [{instrument}] No forward progress (cursor stuck at {cursor}), breaking.")
            break

        cursor = new_cursor
        last_progress = cursor
        print(f"⏱  [{instrument}] up to {cursor}")

    # Merge + save
    if frames:
        new_df = _postprocess_schema(pd.concat(frames).sort_index())
        merged = _postprocess_schema(_merge_dedup(existing, new_df))
    else:
        merged = existing if existing is not None else pd.DataFrame()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path)
    print(f"✅ [{instrument}] Saved → {out_path} | rows={len(merged):,}")

    # QC report
    s_qc = merged.index[0] if len(merged) else start
    e_qc = merged.index[-1] if len(merged) else end
    report = _qc_report(merged, instrument, gran, s_qc, e_qc)
    rep_path = cfg["qc_report_pattern"].format(instrument=instrument)
    _write_text(rep_path, report)

    # parse coverage from report quickly (optional)
    coverage_line = [ln for ln in report.splitlines() if ln.startswith("Coverage ratio")]
    coverage = float(coverage_line[0].split(":")[1].strip().rstrip("%"))/100 if coverage_line else None
    return {"instrument": instrument, "rows": len(merged), "coverage": coverage, "report_path": rep_path}

def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content + "\n")

# =========================
# Entry: loop all instruments + summary
# =========================
def main():
    cfg = CONFIG
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    results = []
    for inst in cfg["instruments"]:
        print(f"\n🚀 Stage 1.1 — Downloading [{inst}] gran={cfg['granularity']} price={cfg['price']} env={cfg['env']}")
        res = fetch_for_instrument(inst, cfg)
        results.append(res)

     # Summary
    lines = ["=== OANDA QC SUMMARY (Stage 1.1) ==="]
    for r in results:
        cov_text = "N/A"
        if r["coverage"] is not None:
            cov_text = f"{r['coverage']:.2%}"
        lines.append(f"- {r['instrument']:8s} | rows={r['rows']:,} | coverage={cov_text} | report={r['report_path']}")
    summary = "\n".join(lines)
    _write_text(cfg["qc_summary_path"], summary)
    print("\n" + summary)

if __name__ == "__main__":
    main()