"""
Clean and cap outliers for datas/stock_datas.csv produced by feature_engineering.py.

Operations:
- Fill Sentiment_Score NaN with 0; NewsTitles NaN with "".
- Per Ticker: drop leading rows whose rolling indicators are NaN.
- Per Ticker: cap ratio/return-like columns at 1%/99% quantiles.
- Overwrite the input CSV by default and create a .bak backup.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DEFAULT_STOCK_PATH = ROOT / "datas" / "stock_datas.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean stock datas CSV and cap outliers.")
    parser.add_argument(
        "--stock-file",
        type=Path,
        default=DEFAULT_STOCK_PATH,
        help="Input CSV path (default datas/stock_datas.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default overwrite input).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing .bak when overwriting input.",
    )
    return parser.parse_args()


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "Sentiment_Score" in df.columns:
        df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0)
    else:
        df["Sentiment_Score"] = 0

    if "NewsTitles" in df.columns:
        df["NewsTitles"] = df["NewsTitles"].fillna("")
    else:
        df["NewsTitles"] = ""
    return df


def drop_leading_rolling_nans(group: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in required_cols if c in group.columns]
    if not cols:
        return group
    mask = group[cols].notnull().all(axis=1)
    if mask.any():
        first_idx = mask.idxmax()
        return group.loc[first_idx:]
    return group


def pick_ratio_cols(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"(return|volatility|sharpe|beta|alpha|trend|candle|intraday|macd|rsi)", re.IGNORECASE)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if pattern.search(c)]


def cap_outliers(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in group.columns:
            continue
        series = group[col]
        if series.notnull().sum() == 0:
            continue
        q_low = series.quantile(0.01)
        q_high = series.quantile(0.99)
        if pd.isna(q_low) or pd.isna(q_high) or q_low == q_high:
            continue
        group[col] = series.clip(q_low, q_high)
    return group


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing(df)

    rolling_cols = [
        "LogReturn",
        "Volatility_20",
        "RSI_14",
        "MACD_12_26_9",
        "MACDh_12_26_9",
        "MACDs_12_26_9",
        "SMA_5",
        "SMA_20",
        "BBL_20_2.0_2.0",
        "BBM_20_2.0_2.0",
        "BBU_20_2.0_2.0",
        "BBB_20_2.0_2.0",
        "BBP_20_2.0_2.0",
        "ATR_14",
        "EMA_12",
        "EMA_26",
        "OBV",
        "Intraday_Range",
        "Trend_Strength",
        "Candle_Body",
        "Beta_60",
        "Alpha_60",
        "Sharpe_60",
        "GSPC_LogReturn",
    ]
    ratio_cols = pick_ratio_cols(df)

    groups = []
    for _, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        g = drop_leading_rolling_nans(g, rolling_cols)
        g = cap_outliers(g, ratio_cols)
        groups.append(g)

    cleaned = pd.concat(groups, ignore_index=True)
    cleaned = cleaned.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return cleaned


def main() -> None:
    args = parse_args()
    stock_path = args.stock_file
    output_path = args.output or stock_path

    df = pd.read_csv(stock_path)
    cleaned = clean_dataframe(df)

    if output_path == stock_path and not args.no_backup:
        backup_path = stock_path.with_suffix(stock_path.suffix + ".bak")
        shutil.copy2(stock_path, backup_path)
        print(f"Backup written to {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Wrote cleaned data to {output_path} (rows={len(cleaned)}, cols={len(cleaned.columns)})")


if __name__ == "__main__":
    main()

