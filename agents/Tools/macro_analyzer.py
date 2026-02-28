"""
MacroAnalyzer - Macro Regime Scanner for ^GSPC / ^VIX / ^TNX.

Reads datas/stock_datas.csv once (with simple mtime-based cache) and returns
rich quantitative metrics so the LLM can reason about systemic risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
DEFAULT_DATA_PATH = ROOT / "datas" / "stock_datas.csv"


class MacroAnalyzer:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
        self._cache_df: Optional[pd.DataFrame] = None
        self._cache_mtime: Optional[float] = None

    # ---------------------- internal helpers ---------------------- #
    def _load_data_cached(self) -> pd.DataFrame:
        try:
            mtime = self.data_path.stat().st_mtime
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        if self._cache_df is not None and self._cache_mtime == mtime:
            return self._cache_df

        df = pd.read_csv(self.data_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.dropna(subset=["Date"])

        # cache
        self._cache_df = df
        self._cache_mtime = mtime
        return df

    def _latest_row_on_or_before(self, frame: pd.DataFrame, target: pd.Timestamp) -> Optional[pd.Series]:
        frame = frame[frame["Date"] <= target].sort_values("Date")
        if frame.empty:
            return None
        return frame.iloc[-1]

    def _with_sma(self, frame: pd.DataFrame, lengths: tuple[int, ...]) -> pd.DataFrame:
        out = frame.sort_values("Date").copy()
        for l in lengths:
            out[f"SMA{l}"] = out["Close"].rolling(l, min_periods=max(5, l // 4)).mean()
        return out

    # ---------------------- metric calculators ---------------------- #
    def _gspc_metrics(self, df: pd.DataFrame, target_ts: pd.Timestamp) -> Dict:
        gspc = df[df["Ticker"] == "^GSPC"][["Date", "Close"]]
        if gspc.empty:
            return {"error": "缺少 ^GSPC 数据"}

        gspc = self._with_sma(gspc, (20, 50, 200))
        row = self._latest_row_on_or_before(gspc, target_ts)
        if row is None:
            return {"error": "目标日前无 ^GSPC 数据"}

        close = float(row["Close"])
        sma20 = float(row["SMA20"]) if not np.isnan(row["SMA20"]) else None
        sma50 = float(row["SMA50"]) if not np.isnan(row["SMA50"]) else None
        sma200 = float(row["SMA200"]) if not np.isnan(row["SMA200"]) else None

        def pct_diff(val: Optional[float]) -> Optional[float]:
            if val is None or val == 0:
                return None
            return (close - val) / val * 100

        dist20 = pct_diff(sma20)
        dist200 = pct_diff(sma200)

        phase = "数据不足"
        if sma200:
            if close > sma200:
                phase = "Bull Market"
            elif sma50 and close > sma50:
                phase = "Recovery/Neutral"
            else:
                phase = "Bear Market"
        elif sma50:
            phase = "Neutral" if close >= sma50 else "Pressure"

        return {
            "date_used": str(row["Date"].date()),
            "close": close,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "dist_to_sma20_pct": dist20,
            "dist_to_sma200_pct": dist200,
            "market_phase": phase,
        }

    def _vix_metrics(self, df: pd.DataFrame, target_ts: pd.Timestamp) -> Dict:
        vix = df[df["Ticker"] == "^VIX"][["Date", "Close"]].sort_values("Date")
        if vix.empty:
            return {"error": "缺少 ^VIX 数据"}
        row = self._latest_row_on_or_before(vix, target_ts)
        if row is None:
            return {"error": "目标日前无 ^VIX 数据"}
        idx = vix.index.get_loc(row.name)
        prev_row = vix.iloc[idx - 1] if idx - 1 >= 0 else None
        prev_close = float(prev_row["Close"]) if prev_row is not None else None
        close = float(row["Close"])
        pct_change = None
        if prev_close and prev_close != 0:
            pct_change = (close - prev_close) / prev_close * 100

        regime = "未知"
        if close >= 30:
            regime = "High Fear"
        elif close >= 20:
            regime = "Elevated"
        else:
            regime = "Calm"

        return {
            "date_used": str(row["Date"].date()),
            "close": close,
            "pct_change_1d": pct_change,
            "regime": regime,
        }

    def _tnx_metrics(self, df: pd.DataFrame, target_ts: pd.Timestamp) -> Dict:
        tnx = df[df["Ticker"] == "^TNX"][["Date", "Close"]]
        if tnx.empty:
            return {"error": "缺少 ^TNX 数据"}
        tnx = self._with_sma(tnx, (50,))
        row = self._latest_row_on_or_before(tnx, target_ts)
        if row is None:
            return {"error": "目标日前无 ^TNX 数据"}
        close = float(row["Close"])
        sma50 = float(row["SMA50"]) if not np.isnan(row["SMA50"]) else None
        trend = "数据不足"
        if sma50:
            trend = "Uptrend (above SMA50)" if close >= sma50 else "Downtrend (below SMA50)"
        return {
            "date_used": str(row["Date"].date()),
            "close": close,
            "sma50": sma50,
            "trend": trend,
        }

    # ---------------------- public API ---------------------- #
    def analyze_market_regime(self, current_date: str) -> Dict:
        try:
            target_ts = pd.to_datetime(current_date, errors="coerce")
            if pd.isna(target_ts):
                return {"error": f"无法解析日期: {current_date}"}

            df = self._load_data_cached()

            gspc_info = self._gspc_metrics(df, target_ts)
            vix_info = self._vix_metrics(df, target_ts)
            tnx_info = self._tnx_metrics(df, target_ts)

            # Determine overall risk regime
            risk_regime = "Neutral"
            rationale_parts = []

            g_phase = gspc_info.get("market_phase")
            vix_level = vix_info.get("close")

            if isinstance(vix_level, (int, float)):
                rationale_parts.append(f"VIX={vix_level:.2f}")
            if g_phase:
                rationale_parts.append(f"GSPC={g_phase}")

            if g_phase == "Bull Market" and isinstance(vix_level, (int, float)) and vix_level < 20:
                risk_regime = "Risk-On"
            elif g_phase == "Bear Market" or (isinstance(vix_level, (int, float)) and vix_level >= 25):
                risk_regime = "Risk-Off"

            return {
                "date_requested": current_date,
                "gspc": gspc_info,
                "vix": vix_info,
                "tnx": tnx_info,
                "risk_regime": risk_regime,
                "rationale": "; ".join(rationale_parts),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}


# Backward compatibility alias (if older code imports MacroRegimeAnalyzer)
MacroRegimeAnalyzer = MacroAnalyzer

