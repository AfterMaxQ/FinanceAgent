"""图表渲染器：把 Agent 工具的结构化输出「画回」到 K 线图上。

设计为「工具名 -> 渲染函数」的注册表，前端在收到某个工具的 tool_result 事件时，
调用 render_tool_chart(...) 即可得到对应的 Plotly 图（没有合适图表则返回 None）。

依赖：plotly、pandas。不依赖任何重型模型库，便于独立测试。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 基础 K 线图
# ---------------------------------------------------------------------------
def base_candlestick(data: pd.DataFrame, title: str = "K线 + 均线") -> go.Figure:
    """根据 OHLC 数据绘制专业 K 线图，叠加 SMA5/SMA20（若存在）。"""
    fig = go.Figure()
    if data is None or data.empty:
        return fig

    df = data.copy()
    if "Date" in df.columns:
        x = pd.to_datetime(df["Date"])
    else:
        x = df.index

    have_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
    if have_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="K线",
                increasing_line_color="#ef4444",  # 红涨（A股习惯）
                decreasing_line_color="#22c55e",  # 绿跌
            )
        )
    elif "Close" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], name="收盘价", line=dict(color="#2563eb")))

    for col, color in (("SMA_5", "#f59e0b"), ("SMA_20", "#3b82f6")):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=x, y=df[col], name=col, line=dict(color=color, width=1.5, dash="dash")))

    fig.update_layout(
        title=dict(text=title, y=0.97, x=0, xanchor="left", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=44, b=48),
        height=360,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="价格 ($)", showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.12,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 各工具的叠加 / 专用图
# ---------------------------------------------------------------------------
def overlay_support_resistance(fig: go.Figure, result: Dict[str, Any]) -> go.Figure:
    """在 K 线图上叠加支撑/阻力水平线。"""
    if not isinstance(result, dict):
        return fig
    supports: List[Dict] = result.get("supports", []) or []
    resistances: List[Dict] = result.get("resistances", []) or []

    for lvl in supports[:4]:
        price = lvl.get("price")
        if price is not None:
            fig.add_hline(
                y=price, line_dash="dot", line_color="#22c55e", opacity=0.6,
                annotation_text=f"支撑 {price:.2f}", annotation_position="right",
            )
    for lvl in resistances[:4]:
        price = lvl.get("price")
        if price is not None:
            fig.add_hline(
                y=price, line_dash="dot", line_color="#ef4444", opacity=0.6,
                annotation_text=f"阻力 {price:.2f}", annotation_position="right",
            )

    near_s = result.get("nearest_support") or {}
    near_r = result.get("nearest_resistance") or {}
    if near_s.get("price") is not None:
        fig.add_hline(y=near_s["price"], line_width=2, line_color="#16a34a",
                      annotation_text=f"最近支撑 {near_s['price']:.2f}", annotation_position="left")
    if near_r.get("price") is not None:
        fig.add_hline(y=near_r["price"], line_width=2, line_color="#dc2626",
                      annotation_text=f"最近阻力 {near_r['price']:.2f}", annotation_position="left")
    fig.update_layout(title=dict(text="支撑 / 阻力位（KDE 筹码峰）", y=0.97, x=0, xanchor="left", font=dict(size=13)))
    return fig


def overlay_prediction(
    fig: go.Figure,
    result: Dict[str, Any],
    data: pd.DataFrame,
    current_price: Optional[float] = None,
) -> go.Figure:
    """在 K 线图上标注 T+1 预测点位与方向概率区间。"""
    if not isinstance(result, dict) or result.get("status") == "error":
        return fig

    predicted = result.get("predicted_price")
    if predicted is None:
        return fig
    change_pct = result.get("predicted_change_pct")
    direction_prob = result.get("direction_prob")

    # 计算 T+1 的横坐标（最后一个交易日 + 1 天）
    last_x = None
    cur_price = current_price
    if data is not None and not data.empty and "Date" in data.columns:
        last_date = pd.to_datetime(data["Date"]).max()
        last_x = last_date + pd.Timedelta(days=1)
        if cur_price is None and "Close" in data.columns:
            cur_price = float(data["Close"].iloc[-1])

    color = "#ef4444" if (change_pct or 0) >= 0 else "#22c55e"
    if last_x is not None and cur_price is not None:
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(data["Date"]).max(), last_x],
            y=[cur_price, predicted],
            mode="lines+markers",
            name="T+1 预测",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=10, symbol="star"),
        ))
    label = f"T+1 预测: {predicted:.2f}"
    if change_pct is not None:
        label += f" ({change_pct:+.2f}%)"
    if direction_prob is not None:
        label += f"\n上涨概率 {direction_prob*100:.0f}%"
    fig.add_hline(y=predicted, line_dash="dash", line_color=color, opacity=0.5,
                  annotation_text=label, annotation_position="right")
    fig.update_layout(title=dict(text="T+1 价格预测（Hybrid 模型）", y=0.97, x=0, xanchor="left", font=dict(size=13)))
    return fig


def render_similarity(result: Dict[str, Any]) -> Optional[go.Figure]:
    """绘制「当前形态 vs 历史相似形态」的标准化叠加图。"""
    matches = result.get("matches") if isinstance(result, dict) else None
    if not matches:
        return None
    top = matches[0]
    viz = top.get("visualization_data") or {}
    target = viz.get("target")
    matched = viz.get("matched")
    if not target or not matched:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=target, mode="lines", name="当前形态",
                             line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(y=matched, mode="lines", name="历史最相似形态",
                             line=dict(color="#f59e0b", width=2, dash="dash")))
    sim = top.get("similarity_score")
    ret = top.get("subsequent_return")
    date = top.get("date", "")
    subtitle = []
    if sim is not None:
        subtitle.append(f"相似度 {sim:.1f}")
    if ret is not None:
        subtitle.append(f"后续收益 {ret:+.2f}%")
    if date:
        subtitle.append(str(date)[:10])
    fig.update_layout(
        title=dict(
            text="历史形态匹配（Z-Score 标准化）" + ("　|　" + " · ".join(subtitle) if subtitle else ""),
            y=0.97, x=0, xanchor="left", font=dict(size=13),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=44, b=52), height=320,
        yaxis=dict(title="标准化价格"), xaxis=dict(title="窗口内交易日"),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
    )
    return fig


def render_tool_chart(
    tool_name: str,
    result: Any,
    data: Optional[pd.DataFrame] = None,
    current_price: Optional[float] = None,
) -> Optional[go.Figure]:
    """根据工具名把工具结果渲染为图表；无合适图表时返回 None。"""
    if result is None:
        return None
    if isinstance(result, dict) and result.get("status") == "error":
        return None

    if tool_name in ("calculate_levels", "get_support_resistance"):
        fig = base_candlestick(data) if data is not None else go.Figure()
        return overlay_support_resistance(fig, result)

    if tool_name in ("predict", "get_stock_prediction"):
        fig = base_candlestick(data) if data is not None else go.Figure()
        return overlay_prediction(fig, result, data, current_price)

    if tool_name in ("search_similar_periods", "find_similar_patterns"):
        return render_similarity(result)

    return None
