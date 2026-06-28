"""统一数据提供层 (DataProvider)。

为「实时模式」提供行情 / 宏观 / 新闻数据，底层使用 yfinance；
当网络不可用或拉取失败时，自动回退到本地 datas/stock_datas.csv（回测数据），
保证演示在任何网络环境下都不会中断。

设计要点：
1. 统一接口：get_ohlcv / get_quote / get_macro / get_news。
2. TTL 缓存：日线数据缓存 1 天、最新价缓存 60s、新闻缓存 5 分钟，避免频繁请求。
3. 优雅降级：每个方法都有 yfinance -> 本地 CSV 的回退路径，并记录数据来源（live / fallback）。
4. 代理可配置：通过环境变量 FINAGENT_PROXY / HTTP_PROXY 配置，默认不启用。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = _PROJECT_ROOT / "datas" / "stock_datas.csv"
MACRO_TICKERS = ["^GSPC", "^VIX", "^TNX"]

# OHLCV 标准列
_OHLCV_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def configure_proxy() -> Optional[str]:
    """根据环境变量配置代理；默认不启用。

    优先读取 FINAGENT_PROXY，其次 HTTP_PROXY。若都未设置则不做任何事，
    这样在没有代理的演示环境（如面试现场）下也能直连。
    """
    proxy = os.environ.get("FINAGENT_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        logger.info("DataProvider 使用代理: %s", proxy)
    return proxy


class _TTLCache:
    """极简 TTL 缓存（进程内）。"""

    def __init__(self) -> None:
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str, ttl: float) -> Optional[Any]:
        item = self._store.get(key)
        if item is None:
            return None
        ts, value = item
        if time.time() - ts > ttl:
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


class DataProvider:
    """实时/回测统一数据提供者。"""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        quote_ttl: float = 60.0,
        daily_ttl: float = 86400.0,
        news_ttl: float = 300.0,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
        self.quote_ttl = quote_ttl
        self.daily_ttl = daily_ttl
        self.news_ttl = news_ttl
        self._cache = _TTLCache()
        self._yf = None
        self._csv_cache: Optional[pd.DataFrame] = None
        self._csv_mtime: Optional[float] = None
        # 记录最近一次取数来源，供前端展示（"live" / "fallback" / "none"）
        self.last_source: Dict[str, str] = {}
        configure_proxy()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _yf_module(self):
        if self._yf is None:
            import yfinance as yf  # 延迟导入，避免无网络环境下的导入开销
            self._yf = yf
        return self._yf

    @staticmethod
    def _period_for(lookback_days: int) -> str:
        """把回溯天数映射为 yfinance 的 period 字符串。"""
        if lookback_days <= 5:
            return "5d"
        if lookback_days <= 30:
            return "1mo"
        if lookback_days <= 90:
            return "3mo"
        if lookback_days <= 180:
            return "6mo"
        if lookback_days <= 370:
            return "1y"
        if lookback_days <= 740:
            return "2y"
        if lookback_days <= 1850:
            return "5y"
        return "max"

    def _load_csv(self) -> pd.DataFrame:
        """带 mtime 缓存地加载本地 CSV。"""
        try:
            mtime = self.data_path.stat().st_mtime
        except FileNotFoundError:
            logger.warning("本地数据文件不存在: %s", self.data_path)
            return pd.DataFrame(columns=_OHLCV_COLS)

        if self._csv_cache is not None and self._csv_mtime == mtime:
            return self._csv_cache

        df = pd.read_csv(self.data_path, low_memory=False)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        if "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        self._csv_cache = df
        self._csv_mtime = mtime
        return df

    def _fallback_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        df = self._load_csv()
        if df.empty or "Ticker" not in df.columns:
            return pd.DataFrame(columns=_OHLCV_COLS)
        sub = df[df["Ticker"] == ticker.upper()].copy()
        if sub.empty:
            return pd.DataFrame(columns=_OHLCV_COLS)
        sub = sub.sort_values("Date")
        keep = [c for c in _OHLCV_COLS if c in sub.columns]
        return sub[keep].tail(lookback_days).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def get_ohlcv(self, ticker: str, lookback_days: int = 400) -> pd.DataFrame:
        """获取单只标的的日线 OHLCV（优先 yfinance，失败回退 CSV）。"""
        ticker = ticker.upper().strip()
        key = f"ohlcv:{ticker}:{lookback_days}"
        cached = self._cache.get(key, self.daily_ttl)
        if cached is not None:
            return cached.copy()

        df = self._fetch_ohlcv_yf(ticker, lookback_days)
        if df is None or df.empty:
            logger.warning("yfinance 获取 %s OHLCV 失败，回退本地 CSV", ticker)
            df = self._fallback_ohlcv(ticker, lookback_days)
            self.last_source[f"ohlcv:{ticker}"] = "fallback" if not df.empty else "none"
        else:
            self.last_source[f"ohlcv:{ticker}"] = "live"

        if df is not None and not df.empty:
            self._cache.set(key, df)
        return df.copy() if df is not None else pd.DataFrame(columns=_OHLCV_COLS)

    def _fetch_ohlcv_yf(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        try:
            yf = self._yf_module()
            period = self._period_for(lookback_days)
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if hist is None or hist.empty:
                return pd.DataFrame()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.reset_index()
            hist["Ticker"] = ticker.upper()
            keep = [c for c in _OHLCV_COLS if c in hist.columns]
            hist = hist[keep]
            price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in hist.columns]
            hist = hist.dropna(subset=price_cols, how="all")
            return hist.tail(lookback_days).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - 任何异常都回退
            logger.warning("yfinance OHLCV 异常 (%s): %s", ticker, exc)
            return pd.DataFrame()

    def get_macro(self, lookback_days: int = 400) -> pd.DataFrame:
        """获取宏观指数（^GSPC/^VIX/^TNX）的长表 OHLCV，便于与个股拼接做特征工程。"""
        key = f"macro:{lookback_days}"
        cached = self._cache.get(key, self.daily_ttl)
        if cached is not None:
            return cached.copy()

        frames: List[pd.DataFrame] = []
        live_count = 0
        for mt in MACRO_TICKERS:
            df = self._fetch_ohlcv_yf(mt, lookback_days)
            if df is not None and not df.empty:
                live_count += 1
                frames.append(df)
            else:
                fb = self._fallback_ohlcv(mt, lookback_days)
                if not fb.empty:
                    frames.append(fb)

        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=_OHLCV_COLS)

        self.last_source["macro"] = (
            "live" if live_count == len(MACRO_TICKERS) else ("fallback" if not out.empty else "none")
        )
        if not out.empty:
            self._cache.set(key, out)
        return out.copy()

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """获取最新价（优先 yfinance fast_info，失败回退最近一根 K 线收盘价）。"""
        ticker = ticker.upper().strip()
        key = f"quote:{ticker}"
        cached = self._cache.get(key, self.quote_ttl)
        if cached is not None:
            return dict(cached)

        quote = self._fetch_quote_yf(ticker)
        if quote is None:
            df = self.get_ohlcv(ticker, lookback_days=5)
            if not df.empty:
                last = df.iloc[-1]
                quote = {
                    "ticker": ticker,
                    "price": float(last["Close"]),
                    "as_of": str(last.get("Date", "")),
                    "source": "fallback",
                }
            else:
                quote = {"ticker": ticker, "price": None, "as_of": "", "source": "none"}
        self._cache.set(key, quote)
        return dict(quote)

    def _fetch_quote_yf(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            yf = self._yf_module()
            tk = yf.Ticker(ticker)
            price = None
            try:
                fast = tk.fast_info
                price = fast.get("last_price") if hasattr(fast, "get") else getattr(fast, "last_price", None)
            except Exception:  # noqa: BLE001
                price = None
            if price is None:
                hist = tk.history(period="1d", auto_adjust=False)
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is None:
                return None
            return {
                "ticker": ticker,
                "price": float(price),
                "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "live",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance 最新价异常 (%s): %s", ticker, exc)
            return None

    def get_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近新闻标题（yfinance 自带 news，免费免 key）。

        失败时回退到本地 CSV 的 NewsTitles 列（历史新闻）。
        返回列表，每项含 title / publish_time / source。
        """
        ticker = ticker.upper().strip()
        key = f"news:{ticker}:{limit}"
        cached = self._cache.get(key, self.news_ttl)
        if cached is not None:
            return list(cached)

        news = self._fetch_news_yf(ticker, limit)
        if news:
            self.last_source[f"news:{ticker}"] = "live"
        else:
            news = self._fallback_news(ticker, limit)
            self.last_source[f"news:{ticker}"] = "fallback" if news else "none"

        self._cache.set(key, news)
        return list(news)

    def _fetch_news_yf(self, ticker: str, limit: int) -> List[Dict[str, Any]]:
        try:
            yf = self._yf_module()
            raw = yf.Ticker(ticker).news or []
            out: List[Dict[str, Any]] = []
            for item in raw[:limit]:
                # yfinance 新旧版本结构不同，做防御式解析
                content = item.get("content", item) if isinstance(item, dict) else {}
                title = (
                    content.get("title")
                    or item.get("title")
                    if isinstance(item, dict)
                    else None
                )
                if not title:
                    continue
                ts = (
                    item.get("providerPublishTime")
                    if isinstance(item, dict)
                    else None
                )
                if ts:
                    publish_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                else:
                    pub = content.get("pubDate") if isinstance(content, dict) else None
                    publish_time = str(pub) if pub else ""
                publisher = ""
                if isinstance(item, dict):
                    publisher = item.get("publisher", "")
                if not publisher and isinstance(content, dict):
                    prov = content.get("provider") or {}
                    if isinstance(prov, dict):
                        publisher = prov.get("displayName", "")
                out.append(
                    {
                        "title": str(title),
                        "publish_time": publish_time,
                        "source": publisher or "Yahoo Finance",
                    }
                )
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance 新闻异常 (%s): %s", ticker, exc)
            return []

    def _fallback_news(self, ticker: str, limit: int) -> List[Dict[str, Any]]:
        df = self._load_csv()
        if df.empty or "Ticker" not in df.columns or "NewsTitles" not in df.columns:
            return []
        sub = df[df["Ticker"] == ticker.upper()].copy()
        if sub.empty:
            return []
        sub = sub.sort_values("Date", ascending=False)
        out: List[Dict[str, Any]] = []
        for _, row in sub.iterrows():
            titles = row.get("NewsTitles")
            if pd.isna(titles) or not str(titles).strip():
                continue
            for headline in str(titles).split(" | "):
                headline = headline.strip()
                if headline and headline != "No significant news":
                    out.append(
                        {
                            "title": headline,
                            "publish_time": str(row.get("Date", "")),
                            "source": "History",
                        }
                    )
                    if len(out) >= limit:
                        return out
        return out

    def get_available_tickers(self) -> List[str]:
        """从本地 CSV 读取可用股票列表（用于降级模式的下拉提示）。"""
        df = self._load_csv()
        if df.empty or "Ticker" not in df.columns:
            return []
        tickers = [t for t in df["Ticker"].unique() if not str(t).startswith("^")]
        return sorted(tickers)
