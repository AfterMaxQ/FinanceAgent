"""
Fetches stock and macro market data using yfinance and writes a merged CSV.

Requirements from user:
- Stock data: Ticker, Date, Open, High, Low, Close, Volume
- Macro data: ^VIX, ^TNX, ^GSPC (same fields)
- Only include tickers whose Percentage >= 0.1 in datas/stock_statistics.csv
- Output saved to datas/stock_datas.csv

Run (defaults to 2009-01-01 through 2019-12-31):
    python get_data.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--period 1y]
If --start/--end are omitted, yfinance period is used (default 1y).
"""

import os
proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy


import argparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional
import warnings

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
STATS_PATH = ROOT / "datas" / "stock_statistics.csv"
OUTPUT_PATH = ROOT / "datas" / "stock_datas.csv"
MACRO_TICKERS = ["^VIX", "^TNX", "^GSPC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download stock and macro data via yfinance.")
    parser.add_argument(
        "--start", type=str, default="2009-01-01", help="Start date YYYY-MM-DD (default 2009-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2019-12-31", help="End date YYYY-MM-DD (default 2019-12-31)"
    )
    # --period 参数在 yf.Ticker().history() 中不常用，保留但逻辑上主要使用 start/end
    parser.add_argument(
        "--period",
        type=str,
        default=None, # 默认不使用 period
        help="yfinance period when start/end not provided (e.g., 1mo, 6mo, 1y, max)",
    )
    parser.add_argument("--threshold", type=float, default=0.1, help="Percentage filter threshold")
    parser.add_argument(
        "--chunk-size", type=int, default=1, help="Batch size (kept for compatibility; forced to 1)"
    )
    parser.add_argument("--retries", type=int, default=4, help="Retry count when rate limited/errors")
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=0.1,
        help="Base seconds to wait between retries (multiplied by attempt number)",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent batch workers") # 适当增加线程数
    return parser.parse_args()


def load_tickers(stats_path: Path, threshold: float) -> List[str]:
    df = pd.read_csv(stats_path)
    if "Percentage" not in df.columns:
        raise ValueError("stock_statistics.csv missing 'Percentage' column")
    tickers = df.loc[df["Percentage"] >= threshold, "Stock"].dropna().astype(str).unique()
    return sorted(tickers.tolist())

# --- 修改开始 2: 替换核心数据获取逻辑 ---
# 使用更稳定的 yf.Ticker().history() 代替 yf.download()
def fetch_single_ticker_history(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
) -> pd.DataFrame:
    """获取单只股票的历史数据，模仿第二份成功代码的逻辑。"""
    
    # 忽略 yfinance 可能产生的 UserWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # yf.Ticker().history() 更喜欢 start/end 参数
        # 如果没有提供 start/end，才考虑使用 period
        if start or end:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        elif period:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        else:
            # 如果都没有提供，则获取最大历史数据（可以根据需要调整）
            df = yf.Ticker(ticker).history(period="max", auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

    # 数据格式化，使其与原脚本输出兼容
    df.index = pd.to_datetime(df.index).tz_localize(None) # 移除时区信息
    df = df.reset_index()
    df["Ticker"] = ticker # 添加 Ticker 列

    # 重命名列以匹配原脚本
    df = df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj Close",
            "Volume": "Volume",
        }
    )
    
    # 保留所需列
    keep_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # 删除价格和交易量都为空的行
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=price_cols, how="all")
    
    return df
# --- 修改结束 2 ---


def download_all(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    period: str,
    retries: int,
    retry_wait: float,
    max_workers: int,
) -> pd.DataFrame:
    frames = []

    def fetch_with_retry(ticker: str) -> pd.DataFrame:
        for attempt in range(1, retries + 1):
            try:
                # --- 修改开始 3: 调用新的获取函数 ---
                df = fetch_single_ticker_history(ticker, start=start, end=end, period=period)
                # --- 修改结束 3 ---
                if not df.empty:
                    return df
            except Exception as exc:
                msg = str(exc).lower()
                is_rate = "rate limit" in msg or "too many request" in msg
                no_data = "no price data found" in msg or "possibly delisted" in msg
                if no_data:
                    print(f"Ticker {ticker} has no price data; skipping. ({exc})")
                    return pd.DataFrame()
                wait = retry_wait * attempt
                print(
                    f"Ticker {ticker} attempt {attempt}/{retries} failed "
                    f"({'rate limited' if is_rate else 'error'}: {exc}); sleeping {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            
            # 如果返回空 DataFrame，也进行重试
            wait = retry_wait * attempt
            print(f"Ticker {ticker} received empty data, attempt {attempt}/{retries}; sleeping {wait:.1f}s")
            time.sleep(wait)

        print(f"Ticker {ticker} failed after {retries} retries.")
        return pd.DataFrame()

    # 并行拉取，每个线程负责 1 只股票
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_with_retry, ticker): ticker for ticker in tickers}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"Ticker {ticker} failed with unexpected error: {exc}")
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[进度] 已完成 {done}/{total} 支股票/指数")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    tickers = load_tickers(STATS_PATH, args.threshold)
    print("=" * 60)
    print("股票数据下载")
    print("=" * 60)
    print(f"数量: {len(tickers)}")
    print(f"日期范围: {args.start} -> {args.end}")
    print(f"线程数: {args.max_workers}")
    print("=" * 60)

    stock_df = download_all(
        tickers,
        args.start,
        args.end,
        args.period,
        args.retries,
        args.retry_wait,
        args.max_workers,
    )
    if not stock_df.empty:
        stock_df["Type"] = "stock"

    print("\n" + "=" * 60)
    print("宏观数据下载")
    print("=" * 60)
    macro_df = download_all(
        MACRO_TICKERS,
        args.start,
        args.end,
        args.period,
        retries=args.retries,
        retry_wait=args.retry_wait,
        max_workers=3, # 宏观指数少，3个线程足矣
    )
    if not macro_df.empty:
        macro_df["Type"] = "macro"

    combined = pd.concat([stock_df, macro_df], ignore_index=True)
    if combined.empty:
        print("No data fetched; nothing to write.")
        return

    combined.sort_values(["Ticker", "Date"], inplace=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()