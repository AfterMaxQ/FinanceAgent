"""轻量级工具集合（适合快速理解与讲解）。

包含两个本科实习级别、逻辑简单的工具：

1. TechnicalSnapshotTool.get_technical_snapshot
   - 用纯 pandas 计算 RSI / MACD / SMA / 布林带的「当前读数」，并给出大白话状态
     （超买 / 超卖 / 金叉 / 偏多空），不依赖 pandas_ta，便于面试现场解释。

2. RecentNewsTool.fetch_recent_news
   - 通过 DataProvider 拉取最近新闻（yfinance，免费免 key），用 FinBERT 逐条打分，
     闭环「多模态」能力。FinBERT 不可用时优雅降级为仅返回新闻标题。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class TechnicalSnapshotTool:
    """计算并解读当前技术指标快照（自包含，纯 pandas 实现）。"""

    def get_technical_snapshot(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """根据 OHLCV 数据计算最新的 RSI/MACD/SMA/布林带读数并给出状态解读。

        Args:
            df: 含 Close 列的 DataFrame（至少约 30 行才有意义）。
            ticker: 股票代码（仅用于回显）。

        Returns:
            含各指标数值与大白话解读的字典；数据不足时返回 status=error。
        """
        if df is None or df.empty or "Close" not in df.columns:
            return {"status": "error", "reason": "缺少 Close 价格数据，无法计算技术快照"}

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 20:
            return {"status": "error", "reason": f"数据量不足（{len(close)} 行，至少需 20 行）"}

        # SMA
        sma5 = close.rolling(5).mean()
        sma20 = close.rolling(20).mean()

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        # MACD(12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        # 布林带(20,2) + %B
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        rng = (upper - lower).replace(0, np.nan)
        pct_b = (close - lower) / rng

        last_close = _safe_float(close.iloc[-1])
        rsi_v = _safe_float(rsi.iloc[-1])
        macd_v = _safe_float(macd.iloc[-1])
        signal_v = _safe_float(signal.iloc[-1])
        hist_v = _safe_float(hist.iloc[-1])
        sma5_v = _safe_float(sma5.iloc[-1])
        sma20_v = _safe_float(sma20.iloc[-1])
        pctb_v = _safe_float(pct_b.iloc[-1])

        # 大白话解读
        notes: List[str] = []
        if rsi_v is not None:
            if rsi_v > 70:
                notes.append(f"RSI={rsi_v:.1f}，处于超买区域（>70），短线有回调压力")
            elif rsi_v < 30:
                notes.append(f"RSI={rsi_v:.1f}，处于超卖区域（<30），短线有反弹可能")
            else:
                notes.append(f"RSI={rsi_v:.1f}，处于中性区域")
        if macd_v is not None and signal_v is not None:
            if macd_v > signal_v:
                notes.append("MACD 在信号线上方（金叉态势），动能偏多")
            else:
                notes.append("MACD 在信号线下方（死叉态势），动能偏空")
        if last_close is not None and sma20_v is not None:
            if last_close > sma20_v:
                notes.append("价格位于 20 日均线上方，中短期趋势偏强")
            else:
                notes.append("价格位于 20 日均线下方，中短期趋势偏弱")
        if pctb_v is not None:
            if pctb_v > 1:
                notes.append("价格突破布林带上轨，波动偏强")
            elif pctb_v < 0:
                notes.append("价格跌破布林带下轨，超跌信号")

        return {
            "status": "ok",
            "ticker": (ticker or "").upper(),
            "close": last_close,
            "rsi_14": rsi_v,
            "macd": macd_v,
            "macd_signal": signal_v,
            "macd_hist": hist_v,
            "sma_5": sma5_v,
            "sma_20": sma20_v,
            "boll_upper": _safe_float(upper.iloc[-1]),
            "boll_lower": _safe_float(lower.iloc[-1]),
            "percent_b": pctb_v,
            "interpretation": "；".join(notes) if notes else "指标计算完成",
        }


class RecentNewsTool:
    """拉取最近新闻并用 FinBERT 打分（多模态情感闭环）。"""

    def __init__(self, provider: Optional[Any] = None) -> None:
        # 延迟创建 provider，避免无谓开销
        self._provider = provider

    def _get_provider(self):
        if self._provider is None:
            from agents.data_provider import DataProvider
            self._provider = DataProvider()
        return self._provider

    def fetch_recent_news(self, ticker: str, limit: int = 8) -> Dict[str, Any]:
        """获取最近新闻并逐条情感打分。

        Args:
            ticker: 股票代码。
            limit: 最多返回的新闻条数。

        Returns:
            含新闻列表、平均情感分、整体情感倾向的字典。
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return {"status": "error", "reason": "未提供股票代码"}

        try:
            provider = self._get_provider()
            news = provider.get_news(ticker, limit=limit)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": f"获取新闻失败: {exc}"}

        if not news:
            return {
                "status": "ok",
                "ticker": ticker,
                "count": 0,
                "items": [],
                "avg_sentiment": 0.0,
                "overall": "无新闻",
            }

        titles = [n.get("title", "") for n in news]

        # 尝试用 FinBERT 打分；不可用时优雅降级
        sentiments: List[Optional[float]] = [None] * len(titles)
        finbert_ok = False
        try:
            from agents.Tools.finbert_analyzer import FinBERTAnalyzer

            analyzer = FinBERTAnalyzer()
            classifications = analyzer.classify_texts(titles)
            for i, res in enumerate(classifications):
                if isinstance(res, dict) and "scores" in res:
                    scores = res["scores"]
                    pos = float(scores.get("Positive", scores.get("positive", 0.0)))
                    neg = float(scores.get("Negative", scores.get("negative", 0.0)))
                    sentiments[i] = pos - neg
            finbert_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT 不可用，新闻情感降级为空: %s", exc)

        items = []
        valid_scores = []
        for n, s in zip(news, sentiments):
            if s is not None:
                valid_scores.append(s)
            items.append(
                {
                    "title": n.get("title", ""),
                    "publish_time": n.get("publish_time", ""),
                    "source": n.get("source", ""),
                    "sentiment": round(s, 4) if s is not None else None,
                }
            )

        avg = float(np.mean(valid_scores)) if valid_scores else 0.0
        overall = "正面" if avg > 0.15 else "负面" if avg < -0.15 else "中性"

        return {
            "status": "ok",
            "ticker": ticker,
            "count": len(items),
            "items": items,
            "avg_sentiment": round(avg, 4),
            "overall": overall,
            "finbert_available": finbert_ok,
        }
