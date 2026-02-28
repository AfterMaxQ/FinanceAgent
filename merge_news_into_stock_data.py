"""
Merge news titles from datas/analyst_ratings_processed.csv into datas/stock_datas.csv.

The news file must contain columns: title, date, stock
The stock data must contain columns: Date, Ticker, ...

Behavior:
- Groups news by (Ticker, Date) and concatenates titles with " | ".
- Adds a new column `NewsTitles` to the stock data.
- By default writes back to the stock file; creates a .bak backup first.

Usage:
    python merge_news_into_stock_data.py \
        --stock-file datas/stock_datas.csv \
        --news-file datas/analyst_ratings_processed.csv \
        --output datas/stock_datas.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge news titles into stock data CSV.")
    parser.add_argument(
        "--stock-file",
        type=Path,
        default=Path("datas") / "stock_datas.csv",
        help="Path to stock data CSV.",
    )
    parser.add_argument(
        "--news-file",
        type=Path,
        default=Path("datas") / "analyst_ratings_processed.csv",
        help="Path to news CSV with columns: title,date,stock.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: overwrite stock-file).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup when overwriting stock-file.",
    )
    return parser.parse_args()


def normalize_dates(series: pd.Series) -> pd.Series:
    """
    Convert to pandas datetime (handles mixed time zones) and normalize to midnight.
    Invalid parses become NaT.
    """
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    # Remove timezone then normalize; guard dt access for safety
    dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()


def load_stock(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("stock file must contain columns Date and Ticker")
    df["Date"] = normalize_dates(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def load_news(path: Path) -> pd.DataFrame:
    cols = ["title", "date", "stock"]
    df = pd.read_csv(path, usecols=cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"news file missing columns: {missing}")
    df["Date"] = normalize_dates(df["date"])
    df["Ticker"] = df["stock"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker", "title"])
    return df


def aggregate_news(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Ticker", "Date"])["title"]
        .apply(lambda s: " | ".join(s))
        .reset_index()
        .rename(columns={"title": "NewsTitles"})
    )
    return grouped


def main() -> None:
    args = parse_args()
    stock_path = args.stock_file
    news_path = args.news_file
    output_path = args.output or stock_path

    stock_df = load_stock(stock_path)
    news_df = load_news(news_path)
    news_grouped = aggregate_news(news_df)

    merged = stock_df.merge(news_grouped, how="left", on=["Ticker", "Date"])

    if output_path == stock_path and not args.no_backup:
        backup_path = stock_path.with_suffix(stock_path.suffix + ".bak")
        shutil.copy2(stock_path, backup_path)
        print(f"Backup written to {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset written to {output_path}")
    print(f"Rows: {len(merged)}, columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()

