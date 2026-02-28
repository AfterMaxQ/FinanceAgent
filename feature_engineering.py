"""
Add engineered features to datas/stock_datas.csv:
- Technical: GSPC_LogReturn, RSI_14, MACD_12_26_9 trio, SMA_5, SMA_20, Volatility_20, Intraday_Range, Trend_Strength, Candle_Body
- Calendar: Month_Sin
- Sentiment: Sentiment_Score via local FinBERT (Score = Positive - Negative)

Defaults: read datas/stock_datas.csv, write back with .bak backup. Offline FinBERT path:
    /g:/Python/FinanceAgentV2.0/models/yiyanghkust_finbert-tone
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from pandas_ta import atr, bbands, ema, macd, obv, rsi, sma
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math


ROOT = Path(__file__).parent
DEFAULT_STOCK_PATH = ROOT / "datas" / "stock_datas.csv"
DEFAULT_FINBERT_PATH = ROOT / "models" / "yiyanghkust_finbert-tone"
FACTOR_WINDOW = 60  # rolling window for Alpha/Beta/Sharpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer features for stock data CSV.")
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
        help="Output CSV (default overwrite input).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip .bak backup when overwriting input.",
    )
    parser.add_argument(
        "--finbert-path",
        type=Path,
        default=DEFAULT_FINBERT_PATH,
        help="Local FinBERT directory (must contain config/tokenizer/model files).",
    )
    parser.add_argument(
        "--sentiment-batch",
        type=int,
        default=16,
        help="Batch size for FinBERT inference.",
    )
    parser.add_argument(
        "--sentiment-maxlen",
        type=int,
        default=256,
        help="Max tokens for FinBERT truncation.",
    )
    return parser.parse_args()


def normalize_dates(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("CSV must contain Date and Ticker columns.")
    df["Date"] = normalize_dates(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def add_gspc_features(df: pd.DataFrame) -> pd.DataFrame:
    gspc = (
        df.loc[df["Ticker"] == "^GSPC", ["Date", "Close"]]
        .dropna(subset=["Close"])
        .sort_values("Date")
        .rename(columns={"Close": "GSPC_Close"})
    )
    gspc["GSPC_LogReturn"] = np.log(gspc["GSPC_Close"] / gspc["GSPC_Close"].shift(1))
    gspc_map_close = gspc.set_index("Date")["GSPC_Close"]
    gspc_map_lr = gspc.set_index("Date")["GSPC_LogReturn"]
    df["GSPC_Close"] = df["Date"].map(gspc_map_close)
    df["GSPC_LogReturn"] = df["Date"].map(gspc_map_lr)
    return df


def _compute_group_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    g["LogReturn"] = np.log(g["Close"] / g["Close"].shift(1))
    g["Volatility_20"] = g["LogReturn"].rolling(window=20, min_periods=5).std()

    # Bollinger Bands (20, 2)
    bb = bbands(close=g["Close"], length=20, std=2)
    if bb is not None:
        for col in bb.columns:
            g[col] = bb[col].values

    # ATR (14)
    g["ATR_14"] = atr(high=g["High"], low=g["Low"], close=g["Close"], length=14)

    # EMA (12, 26)
    g["EMA_12"] = ema(close=g["Close"], length=12)
    g["EMA_26"] = ema(close=g["Close"], length=26)

    # OBV
    g["OBV"] = obv(close=g["Close"], volume=g["Volume"])

    g["RSI_14"] = rsi(close=g["Close"], length=14)
    macd_df = macd(close=g["Close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        for col in macd_df.columns:
            g[col] = macd_df[col].values

    g["SMA_5"] = sma(close=g["Close"], length=5)
    g["SMA_20"] = sma(close=g["Close"], length=20)

    g["Intraday_Range"] = (g["High"] - g["Low"]) / (g["Close"] + 1e-9)
    g["Trend_Strength"] = g["Close"] / g["SMA_20"]
    g["Candle_Body"] = (g["Close"] - g["Open"]) / (g["Open"] + 1e-9)

    g["Month_Sin"] = np.sin(2 * np.pi * g["Date"].dt.month / 12)
    return g


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for _, g in df.groupby("Ticker", group_keys=False):
        grouped.append(_compute_group_features(g))
    return pd.concat(grouped, ignore_index=True)


def add_factor_metrics(df: pd.DataFrame, window: int = FACTOR_WINDOW) -> pd.DataFrame:
    grouped = []
    sqrt_252 = math.sqrt(252)
    for _, g in df.groupby("Ticker", group_keys=False):
        g = g.sort_values("Date").copy()
        # Require market returns present
        ret = g["LogReturn"]
        mret = g["GSPC_LogReturn"]

        cov = ret.rolling(window).cov(mret)
        var_m = mret.rolling(window).var()
        beta = cov / var_m.replace(0, np.nan)
        mean_ret = ret.rolling(window).mean()
        mean_m = mret.rolling(window).mean()
        alpha = mean_ret - beta * mean_m
        std_ret = ret.rolling(window).std()
        sharpe = (mean_ret / std_ret.replace(0, np.nan)) * sqrt_252

        g[f"Beta_{window}"] = beta
        g[f"Alpha_{window}"] = alpha
        g[f"Sharpe_{window}"] = sharpe
        grouped.append(g)
    return pd.concat(grouped, ignore_index=True)


def load_finbert(finbert_path: Path) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    tokenizer = AutoTokenizer.from_pretrained(finbert_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(finbert_path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def batch(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    bucket = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def compute_sentiment(df: pd.DataFrame, finbert_path: Path, batch_size: int, max_len: int) -> pd.Series:
    valid_mask = df["NewsTitles"].astype(str).str.strip().ne("") & df["NewsTitles"].notna()
    if not valid_mask.any():
        return pd.Series(np.nan, index=df.index)

    tokenizer, model, device = load_finbert(finbert_path)
    # 显式读取标签映射，避免不同权重文件的顺序差异
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    pos_id = label2id.get("positive", 0)
    neg_id = label2id.get("negative", 1)
    texts = df.loc[valid_mask, "NewsTitles"]
    scores = pd.Series(np.nan, index=texts.index)

    for idx_batch in batch(list(texts.index), size=batch_size):
        batch_texts = texts.loc[idx_batch].tolist()
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            positive = probs[:, pos_id]
            negative = probs[:, neg_id]
            score = positive - negative
        scores.loc[idx_batch] = score.cpu().numpy()

    out = pd.Series(np.nan, index=df.index)
    out.loc[scores.index] = scores
    return out


def engineer_features(
    stock_path: Path,
    output_path: Path,
    finbert_path: Path,
    sentiment_batch: int,
    sentiment_maxlen: int,
    make_backup: bool,
) -> Path:
    df = load_df(stock_path)

    df = add_gspc_features(df)
    df = add_technical_features(df)
    df = add_factor_metrics(df)

    if "NewsTitles" not in df.columns:
        df["NewsTitles"] = ""
    df["Sentiment_Score"] = compute_sentiment(
        df, finbert_path=finbert_path, batch_size=sentiment_batch, max_len=sentiment_maxlen
    )

    if output_path == stock_path and make_backup:
        backup_path = stock_path.with_suffix(stock_path.suffix + ".bak")
        backup_path.write_bytes(stock_path.read_bytes())
        print(f"Backup written to {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features written to {output_path} (rows={len(df)}, cols={len(df.columns)})")
    return output_path


def main() -> None:
    args = parse_args()
    stock_path = args.stock_file
    output_path = args.output or stock_path
    engineer_features(
        stock_path=stock_path,
        output_path=output_path,
        finbert_path=args.finbert_path,
        sentiment_batch=args.sentiment_batch,
        sentiment_maxlen=args.sentiment_maxlen,
        make_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()

