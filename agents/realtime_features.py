"""实时特征构建器 (RealtimeFeatureBuilder)。

把 DataProvider 拉取到的实时 OHLCV + 宏观指数数据，通过复用
feature_engineering.py 里**与训练时完全相同**的特征函数，现算出 Hybrid 模型
需要的 18 个 VIF 特征 + Sentiment_Score，从而避免训练-推理特征不一致
(train/serving skew)。

返回的 DataFrame 可直接传给 HybridPredictor.predict(simulation_data=df, ...)。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 复用训练时的特征函数，保证列名/计算口径完全一致
from feature_engineering import (  # noqa: E402
    add_factor_metrics,
    add_gspc_features,
    add_technical_features,
)
from agents.data_provider import DataProvider, MACRO_TICKERS  # noqa: E402

logger = logging.getLogger(__name__)

# 滚动指标列（参考 clean_stock_data.py），用于丢弃开头的 NaN 行
_ROLLING_COLS = [
    "LogReturn", "Volatility_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9",
    "MACDs_12_26_9", "SMA_5", "SMA_20", "ATR_14", "OBV", "Intraday_Range",
    "Trend_Strength", "Candle_Body", "Beta_60", "Alpha_60", "Sharpe_60",
    "GSPC_LogReturn",
]


class RealtimeFeatureBuilder:
    """基于 DataProvider 现算特征，供实时模式下的模型推理与图表使用。"""

    def __init__(self, provider: Optional[DataProvider] = None) -> None:
        self.provider = provider or DataProvider()

    def build_features(
        self,
        ticker: str,
        lookback_days: int = 400,
        sentiment_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """为单只标的现算特征。

        Args:
            ticker: 股票代码。
            lookback_days: 回溯天数（默认 400，足够算 Beta_60 等长窗口指标）。
            sentiment_score: 可选，外部传入的最新情感分（用于填充 Sentiment_Score）。

        Returns:
            字典：
            - data: 含工程特征的 DataFrame（按 Date 升序，仅该 ticker）。
            - current_price: 最新收盘价。
            - as_of: 数据截止日期。
            - source: 数据来源（live / fallback / none）。
            - error: 错误信息（成功则为 None）。
        """
        ticker = ticker.upper().strip()
        result: Dict[str, Any] = {
            "data": pd.DataFrame(),
            "current_price": None,
            "as_of": None,
            "source": "none",
            "error": None,
        }

        try:
            stock = self.provider.get_ohlcv(ticker, lookback_days=lookback_days)
            if stock is None or stock.empty:
                result["error"] = f"无法获取 {ticker} 的行情数据（实时与本地均失败）"
                return result

            macro = self.provider.get_macro(lookback_days=lookback_days)

            # 拼接为长表（与原始 stock_datas.csv 结构一致）
            base_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            frames = [stock]
            if macro is not None and not macro.empty:
                frames.append(macro)
            raw = pd.concat(frames, ignore_index=True)
            raw = raw[[c for c in base_cols if c in raw.columns]].copy()
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.tz_localize(None)
            raw["Ticker"] = raw["Ticker"].astype(str).str.upper().str.strip()
            raw = raw.dropna(subset=["Date", "Close"])
            if "NewsTitles" not in raw.columns:
                raw["NewsTitles"] = ""

            # 复用训练时的特征管线（顺序与 engineer_features 一致）
            df = add_gspc_features(raw)
            df = add_technical_features(df)
            df = add_factor_metrics(df)

            # Sentiment_Score：实时无逐日新闻，默认 0；若外部传入最新情感分则填充
            df["Sentiment_Score"] = float(sentiment_score) if sentiment_score is not None else 0.0

            # 只保留目标 ticker，丢弃宏观行
            df = df[df["Ticker"] == ticker].copy()
            df = df.sort_values("Date").reset_index(drop=True)

            # 处理 NaN：先丢弃开头滚动窗口未填满的行，再前向/后向填充残余
            present_rolling = [c for c in _ROLLING_COLS if c in df.columns]
            if present_rolling:
                mask = df[present_rolling].notnull().all(axis=1)
                if mask.any():
                    first_valid = mask.idxmax()
                    df = df.loc[first_valid:].reset_index(drop=True)
            df = df.ffill().bfill()

            if df.empty:
                result["error"] = f"{ticker} 特征计算后无有效数据（历史长度不足）"
                return result

            last = df.iloc[-1]
            result["data"] = df
            result["current_price"] = float(last["Close"])
            result["as_of"] = str(pd.to_datetime(last["Date"]).date())
            result["source"] = self.provider.last_source.get(f"ohlcv:{ticker}", "live")
            return result

        except Exception as exc:  # noqa: BLE001
            logger.error("实时特征构建失败 (%s): %s", ticker, exc, exc_info=True)
            result["error"] = str(exc)
            return result
