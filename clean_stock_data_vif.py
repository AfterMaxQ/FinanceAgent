"""
Filter cleaned stock data to keep only the VIF筛选后的特征集合，并输出精简版本。

默认读取 datas/stock_datas.csv（clean_stock_data.py 的输出），输出 datas/stock_datas_vif.csv。
可通过 --output 指定输出路径；若需要覆盖原文件请显式传入。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "datas" / "stock_datas.csv"
DEFAULT_OUTPUT = ROOT / "datas" / "stock_datas_vif.csv"

# VIF保留特征（来自 selected_features.md）
VIF_FEATURES: List[str] = [
    "Volume",
    "GSPC_Close",
    "GSPC_LogReturn",
    "LogReturn",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "Intraday_Range",
    "Trend_Strength",
    "Candle_Body",
    "Month_Sin",
    "Sentiment_Score",
    "BBB_20_2.0_2.0",
    "BBP_20_2.0_2.0",
    "ATR_14",
    "OBV",
    "Beta_60",
    "Alpha_60",
    "Sharpe_60",
]

# 识别用的基础列（保留日期/代码及新闻文本）
ID_COLS = ["Date", "Ticker", "NewsTitles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep only VIF-selected features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV (default datas/stock_datas.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV (default datas/stock_datas_vif.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    cols_available = [c for c in ID_COLS + VIF_FEATURES if c in df.columns]
    missing = [c for c in ID_COLS + VIF_FEATURES if c not in df.columns]

    if not cols_available:
        raise ValueError("No expected columns found; please check input file.")

    filtered = df[cols_available].copy()
    filtered.to_csv(args.output, index=False)

    print(f"Written filtered data to {args.output} (rows={len(filtered)}, cols={len(filtered.columns)})")
    if missing:
        print(f"Warning: missing columns not found in input: {missing}")


if __name__ == "__main__":
    main()

