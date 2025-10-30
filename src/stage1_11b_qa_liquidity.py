#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.11b — QA liquidity & funding dataset

- sanity check missing %
- rolling mean / zscore of LIQ_COMPOSITE
- last rows preview
- write text summary to logs/stage1_11b_liquidity_QA.txt
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

IN_FILE  = "data/liquidity_funding.parquet"
OUT_FILE = "logs/stage1_11b_liquidity_QA.txt"

def main():
    df = pd.read_parquet(IN_FILE)
    # ensure sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # missing %
    miss = df.isna().mean().sort_values(ascending=False) * 100

    # rolling mean of composite (30d)
    df["LIQ_COMPOSITE_30d_mean"] = df["LIQ_COMPOSITE"].rolling(30, min_periods=5).mean()

    # summary text
    lines = []
    lines.append("Stage 1.11b Liquidity QA")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Date range: {df['date'].min()} → {df['date'].max()}")
    lines.append("")
    lines.append("Missing % by column:")
    for col, pct in miss.items():
        lines.append(f"  {col:25s} {pct:5.2f}%")
    lines.append("")
    lines.append("Last 5 rows:")
    lines.append(df.tail(5).to_string(index=False))
    lines.append("")
    lines.append("Recent LIQ_COMPOSITE levels (last 5):")
    lines.append(df[["date","LIQ_COMPOSITE","LIQ_COMPOSITE_30d_mean"]].tail(5).to_string(index=False))

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path(OUT_FILE).write_text("\n".join(lines), encoding="utf-8")

    # print to console too
    print("\n".join(lines))

if __name__ == "__main__":
    main()