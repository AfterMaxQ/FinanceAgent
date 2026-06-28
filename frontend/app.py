import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import warnings
import traceback
import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# 添加项目根目录到路径，以便导入统一接口
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
# 将项目根目录添加到路径，这样可以明确导入 frontend 和 src 模块
sys.path.insert(0, str(_PROJECT_ROOT))

# 修复 Windows 证书库损坏导致的 SSL 报错（覆盖运行时 yfinance/DeepSeek/SEC 的 HTTPS）
try:
    import ssl_fix  # noqa: F401
except Exception:
    pass

# 创建简单的配置模块（如果不存在）
try:
    from src.config.settings import finbert_settings, deepseek_settings
except ImportError:
    # 创建简单的配置对象
    class SimpleSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # 创建 finbert_settings
    finbert_settings = SimpleSettings(
        embedding_model_name="yiyanghkust/finbert-tone",
        embedding_max_length=128,
        model_name="yiyanghkust/finbert-tone",
        max_length=256,
        device=None,
        model_cache_dir=_PROJECT_ROOT / "models"
    )
    
    # 创建 deepseek_settings
    deepseek_settings = SimpleSettings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        # 升级到 DeepSeek V4 Pro（deepseek-chat 别名将于 2026-07-24 停用）
        deepseek_model="deepseek-v4-pro",
        deepseek_timeout=60
    )
    
    # 将配置对象添加到 sys.modules，以便 agents 模块可以导入
    import types
    config_module = types.ModuleType('src.config.settings')
    config_module.finbert_settings = finbert_settings
    config_module.deepseek_settings = deepseek_settings
    sys.modules['src.config.settings'] = config_module

# Hybrid 模型推理 (新的多任务模型)
from agents.Tools.finbert_analyzer import FinBERTAnalyzer
from agents.Tools.predict_t1 import HybridPredictor
from agents.Tools.macro_analyzer import MacroAnalyzer
from agents.Tools.support_resistance import SupportResistanceScanner

# 导入金融智能体模块
from agents.decision_agent import FinancialAgent, AGENT_INTERFACE_VERSION

# 多会话存储（历史对话列表与重载）
from agents.conversation_store import ConversationStore, Conversation

# 统一数据提供层（实时/回测，自带降级）
from agents.data_provider import DataProvider

# 图表渲染器（把工具输出画回 K 线图）
try:
    from frontend import chart_renderers as cr
except Exception:  # pragma: no cover - 直接以脚本方式运行时的回退导入
    import chart_renderers as cr

# 可选：定时自动刷新组件（实时模式用），缺失时优雅降级
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except Exception:  # pragma: no cover
    _AUTOREFRESH_AVAILABLE = False

# ==========================================
# 数据加载函数和常量定义
# ==========================================
# 数据文件路径
_DATA_DIR = _PROJECT_ROOT / "datas"
_STOCK_DATA_PATH = _DATA_DIR / "stock_datas.csv"
_STOCK_STATS_PATH = _DATA_DIR / "stock_statistics.csv"

# 从数据文件读取日期范围
@st.cache_data
def _get_data_date_range():
    """从CSV文件读取日期范围"""
    try:
        df = pd.read_csv(_STOCK_DATA_PATH, usecols=['Date'], nrows=1)
        df_full = pd.read_csv(_STOCK_DATA_PATH, usecols=['Date'])
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        return df_full['Date'].min().date(), df_full['Date'].max().date()
    except Exception:
        # 默认日期范围（根据 stock_datas_explained.md）
        return date(2008, 3, 31), date(2016, 12, 30)

KAGGLE_START_DATE, KAGGLE_END_DATE = _get_data_date_range()

# 从 stock_statistics.csv 读取支持的股票代码，并过滤出实际存在于数据文件中的股票
@st.cache_data
def _get_supported_tickers():
    """从 stock_statistics.csv 读取支持的股票代码，并验证它们是否存在于数据文件中"""
    try:
        # 首先读取实际数据文件中的可用股票
        try:
            data_df = pd.read_csv(_STOCK_DATA_PATH, usecols=['Ticker'], low_memory=False)
            available_tickers = set(data_df['Ticker'].str.upper().unique())
        except Exception:
            available_tickers = set()
        
        # 从统计文件读取股票列表
        stats_df = pd.read_csv(_STOCK_STATS_PATH)
        tickers = stats_df['Stock'].str.upper().unique().tolist()
        
        # 只返回在数据文件中实际存在的股票
        if available_tickers:
            tickers = [t for t in tickers if t in available_tickers]
        
        return sorted(tickers) if tickers else ["EBAY", "ORCL", "KO", "JNJ", "MS", "HD", "MA", "PEP", "QCOM"]
    except Exception:
        # 默认股票列表（确保这些股票在数据中存在）
        return ["EBAY", "ORCL", "KO", "JNJ", "MS", "HD", "MA", "PEP", "QCOM"]

SUPPORTED_TICKERS = _get_supported_tickers()

def load_stock_data_dynamic(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """从 CSV 文件加载指定股票和日期范围的数据"""
    try:
        df = pd.read_csv(_STOCK_DATA_PATH, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # 首先检查该股票是否存在
        available_tickers = df['Ticker'].unique()
        if ticker.upper() not in available_tickers:
            st.warning(f"⚠️ 股票代码 {ticker.upper()} 在数据集中不存在。可用股票: {', '.join(sorted(available_tickers)[:20])}...")
            return pd.DataFrame()
        
        # 过滤 ticker 和日期范围
        mask = (df['Ticker'] == ticker.upper()) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
        filtered_df = df[mask].copy()
        
        # 按日期排序
        filtered_df = filtered_df.sort_values('Date').reset_index(drop=True)
        
        # 确保 NewsTitles 列存在（如果不存在则创建）
        if 'NewsTitles' not in filtered_df.columns:
            filtered_df['NewsTitles'] = ''
        
        return filtered_df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return pd.DataFrame()

def calculate_aggregated_sentiment(
    data: pd.DataFrame, 
    window: int = 7,
    use_market_adjustment: bool = True
) -> pd.Series:
    """计算市场情绪指数（聚合情感分数）
    
    使用滚动窗口平滑单个股票的情感分数，并可选择结合市场整体表现进行调整。
    
    Args:
        data: 包含 Sentiment_Score 和 Date 列的 DataFrame
        window: 滚动窗口大小（天数），默认7天
        use_market_adjustment: 是否使用市场指标（GSPC）进行调整
    
    Returns:
        市场情绪指数 Series
    """
    if len(data) == 0 or 'Sentiment_Score' not in data.columns:
        return pd.Series([0.0] * len(data), index=data.index)
    
    # 确保数据按日期排序（保留原始索引）
    data_sorted = data.sort_values('Date').copy()
    original_index = data.index
    
    # 1. 基础滚动窗口平滑（使用EMA以获得更平滑的曲线）
    sentiment_series = data_sorted['Sentiment_Score'].fillna(0.0)
    
    # 使用指数移动平均（EMA）而不是简单移动平均，对近期数据给予更高权重
    aggregated = sentiment_series.ewm(span=window, adjust=False).mean()
    
    # 2. 可选：结合市场整体表现进行调整
    if use_market_adjustment and 'GSPC_LogReturn' in data_sorted.columns:
        # 获取市场收益率（GSPC对数收益）
        market_returns = data_sorted['GSPC_LogReturn'].fillna(0.0)
        
        # 计算市场收益率的滚动平均（用于判断市场整体趋势）
        market_trend = market_returns.rolling(window=window, min_periods=1).mean()
        
        # 将市场趋势转换为调整因子（范围约在-0.2到+0.2之间）
        # 市场上涨时增强正面情绪，市场下跌时增强负面情绪
        market_adjustment = np.tanh(market_trend * 10) * 0.2
        
        # 结合市场调整因子
        aggregated = aggregated + market_adjustment
        
        # 确保结果在合理范围内（-1到1）
        aggregated = np.clip(aggregated, -1.0, 1.0)
    
    # 创建结果Series，使用原始索引
    result = pd.Series(0.0, index=original_index)
    
    # 将计算结果映射回原始索引
    for orig_idx in original_index:
        date_val = data.loc[orig_idx, 'Date']
        # 在排序后的数据中找到对应日期的值
        matching_rows = data_sorted[data_sorted['Date'] == date_val]
        if len(matching_rows) > 0:
            # 取第一个匹配的值（如果同一天有多条记录）
            sorted_idx = matching_rows.index[0]
            result.loc[orig_idx] = aggregated.loc[sorted_idx]
    
    return result

def extract_news_before_date(ticker: str, target_date: date, days_before: int = 5, max_news_per_day: int = 3) -> List[Dict]:
    """从数据中提取指定日期前几天的新闻"""
    try:
        end_date = target_date
        start_date = target_date - timedelta(days=days_before)
        
        df = pd.read_csv(_STOCK_DATA_PATH, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # 过滤 ticker 和日期范围
        mask = (df['Ticker'] == ticker.upper()) & (df['Date'] >= start_date) & (df['Date'] < end_date)
        filtered_df = df[mask].copy()
        
        news_list = []
        for _, row in filtered_df.iterrows():
            if pd.notna(row.get('NewsTitles')) and str(row['NewsTitles']).strip():
                # 分割多条新闻（用 | 分隔）
                headlines = str(row['NewsTitles']).split(" | ")
                for headline in headlines[:max_news_per_day]:
                    headline = headline.strip()
                    if headline and headline != "No significant news":
                        # 计算情感分数（如果有 Sentiment_Score 列）
                        sentiment = row.get('Sentiment_Score', 0.0)
                        if pd.isna(sentiment):
                            sentiment = 0.0
                        
                        news_list.append({
                            'Date': str(row['Date']),
                            'Title': headline,
                            'Sentiment': float(sentiment),
                            'Source': 'News'
                        })
        
        return news_list
    except Exception as e:
        warnings.warn(f"提取新闻失败: {e}")
        return []

# 技术指标计算（简化版，因为数据中已包含技术指标）
def generate_tech_indicators(df: pd.DataFrame, normalize: bool = False, return_scaler: bool = False) -> pd.DataFrame:
    """技术指标计算（如果数据中已有则直接返回）"""
    # 数据中已包含技术指标，直接返回
    return df.copy()

# ==========================================
# 1. 页面配置与 UI 设计系统
# ==========================================
#
# 设计语言 (Design Tokens)
# -------------------------
# 主色:        #2563eb (royal blue) — 信任 / 数据
# 强调色:      #0f172a (slate-900)  — 文字主色
# 背景层级:    #f5f7fb (页面) / #ffffff (卡片) / #f8fafc (区段)
# 辅助色:      #22c55e 涨 / #ef4444 跌 / #f59e0b 警戒
# 字体:        Inter / -apple-system 系统字体栈
# 圆角:        10px (卡片) / 6px (输入框 / pill)
#

BRAND_NAME = "FinanceAgent"
BRAND_TAGLINE = "量化分析助手 · 双脑驱动"


def setup_page_style() -> None:
    """Configure Streamlit page and inject global CSS."""
    st.set_page_config(
        page_title=f"{BRAND_NAME} · 量化分析助手",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        /* ---------- 全局基础 ---------- */
        .stApp {
            background: #f5f7fb;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
                "PingFang SC", "Microsoft YaHei", sans-serif;
            color: #0f172a;
        }
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="collapsedControl"] { display: none !important; }

        /* 主区域内边距收窄 */
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* ---------- 排版 ---------- */
        h1, h2, h3, h4, h5 { color: #0f172a; font-weight: 600; letter-spacing: -0.01em; }
        h3 { font-size: 1.18rem; }
        h4 { font-size: 1.02rem; margin-top: 0.4rem; }
        p, li, span, label { color: #1e293b; }
        a { color: #2563eb; }
        hr { margin: 1.2rem 0; border-color: #e2e8f0; }

        /* ---------- 卡片 ---------- */
        .fa-card {
            background: #ffffff;
            padding: 18px 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin-bottom: 18px;
        }
        .fa-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .fa-card-title {
            font-size: 0.93rem;
            font-weight: 600;
            color: #0f172a;
            letter-spacing: 0.01em;
        }
        .fa-card-sub {
            font-size: 0.78rem;
            color: #64748b;
        }

        /* ---------- Section Label（替代满屏 emoji 标题）---------- */
        .fa-section {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 18px 0 10px;
        }
        .fa-section .bar {
            width: 3px;
            height: 16px;
            border-radius: 2px;
            background: #2563eb;
        }
        .fa-section .title {
            font-size: 1.0rem;
            font-weight: 600;
            color: #0f172a;
            letter-spacing: 0.01em;
        }
        .fa-section .desc {
            margin-left: 6px;
            font-size: 0.78rem;
            color: #64748b;
        }

        /* ---------- Hero（顶部品牌区）---------- */
        .fa-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            margin-bottom: 16px;
        }
        .fa-hero-left { display: flex; align-items: center; gap: 14px; }
        .fa-logo {
            width: 44px; height: 44px;
            border-radius: 10px;
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: #fff;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 18px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            letter-spacing: -0.02em;
        }
        .fa-brand {
            display: flex; flex-direction: column; gap: 2px;
        }
        .fa-brand-name { font-size: 1.16rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
        .fa-brand-sub { font-size: 0.78rem; color: #64748b; }
        .fa-hero-right { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }

        /* ---------- Pill ---------- */
        .fa-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 500;
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #e2e8f0;
            line-height: 1.4;
            white-space: nowrap;
        }
        .fa-pill.live   { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
        .fa-pill.bt     { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
        .fa-pill.warn   { background: #fffbeb; color: #b45309; border-color: #fde68a; }
        .fa-pill.muted  { background: #f8fafc; color: #64748b; border-color: #e2e8f0; }
        .fa-pill .dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: currentColor; opacity: 0.7;
        }

        /* ---------- 侧边栏 ---------- */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
            min-width: 280px;
            max-width: 320px;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
        [data-testid="stSidebar"] hr { margin: 0.8rem 0; }
        .fa-sb-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 4px 0 12px 0;
            border-bottom: 1px solid #f1f5f9;
            margin-bottom: 12px;
        }
        .fa-sb-brand .logo {
            width: 30px; height: 30px;
            border-radius: 8px;
            background: linear-gradient(135deg, #2563eb, #1e40af);
            color: #fff; font-weight: 700; font-size: 13px;
            display: flex; align-items: center; justify-content: center;
        }
        .fa-sb-brand .name { font-weight: 700; font-size: 0.98rem; color: #0f172a; }
        .fa-sb-brand .ver  { font-size: 0.7rem; color: #94a3b8; margin-top: 1px; }

        .fa-sb-label {
            font-size: 0.72rem; font-weight: 600;
            color: #64748b; letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 14px 0 4px;
        }

        /* ---------- 按钮：克制的设计 ---------- */
        .stButton > button {
            background: #ffffff;
            color: #0f172a;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            min-height: 38px;
            font-weight: 500;
            font-size: 0.86rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #2563eb;
            color: #2563eb;
            background: #f8fafc;
        }
        /* Primary 按钮（用 type="primary" 触发）*/
        .stButton > button[kind="primary"] {
            background: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
        }
        .stButton > button[kind="primary"]:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
            color: #ffffff;
        }

        /* ---------- Metric 指标块 ---------- */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: #64748b !important;
            font-weight: 500;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            color: #0f172a !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

        /* ---------- 输入控件 ---------- */
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox > div > div {
            border-radius: 8px !important;
            border-color: #cbd5e1 !important;
        }
        .stRadio > label, .stCheckbox > label { font-size: 0.85rem; }

        /* ---------- Chat 气泡 ---------- */
        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            margin-bottom: 10px;
        }
        [data-testid="stChatMessage"] p { line-height: 1.65; }

        /* ---------- 状态栏 ---------- */
        [data-testid="stStatusWidget"] {
            border: 1px solid #e2e8f0;
            background: #ffffff;
            border-radius: 10px;
        }

        /* ---------- 列间距 ---------- */
        [data-testid="column"] { padding-left: 0.4rem; padding-right: 0.4rem; }

        /* ---------- 新闻条目 ---------- */
        .fa-news-item {
            padding: 10px 12px;
            border-radius: 8px;
            background: #f8fafc;
            margin-bottom: 10px;
            border-left: 3px solid #cbd5e1;
        }
        .fa-news-item.pos { border-left-color: #22c55e; }
        .fa-news-item.neg { border-left-color: #ef4444; }
        .fa-news-meta { font-size: 0.72rem; color: #64748b; margin-bottom: 4px; }
        .fa-news-title { font-size: 0.88rem; color: #0f172a; line-height: 1.45; font-weight: 500; }
        .fa-news-score { font-size: 0.72rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 4px; }
        .fa-news-score.pos { color: #16a34a; }
        .fa-news-score.neg { color: #dc2626; }
        .fa-news-score.neu { color: #64748b; }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
        }

        /* ---------- 隐藏 plotly 工具栏的 stripe 噪点 ---------- */
        .modebar { opacity: 0.4; }
        .modebar:hover { opacity: 1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


setup_page_style()


# ---------- 通用 UI 工具 ---------- #
def section_label(title: str, desc: str = "") -> None:
    """渲染统一风格的小节标题（左侧蓝色色条 + 副描述）。"""
    desc_html = f'<span class="desc">{desc}</span>' if desc else ""
    st.markdown(
        f'<div class="fa-section"><span class="bar"></span>'
        f'<span class="title">{title}</span>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, variant: str = "muted") -> str:
    """生成 pill 标签 HTML。variant ∈ {muted, live, bt, warn}。"""
    return f'<span class="fa-pill {variant}"><span class="dot"></span>{text}</span>'

# ==========================================
# 2. 数据加载与模型初始化
# ==========================================
# 使用真实数据接口和模型进行预测
TICKER_NAME_MAP = {
    "NVDA": "英伟达", "EBAY": "eBay", "ORCL": "Oracle", "KO": "可口可口",
    "JNJ": "强生", "MS": "摩根士丹利", "HD": "家得宝", "MA": "万事达",
    "PEP": "百事", "QCOM": "高通",
}

# 延迟初始化 stock_universe（在 SUPPORTED_TICKERS 定义之后）
def get_stock_universe():
    """获取股票列表"""
    return [
        {"ticker": ticker, "name": TICKER_NAME_MAP.get(ticker, ticker)}
        for ticker in SUPPORTED_TICKERS
    ]

stock_universe = get_stock_universe()


def resolve_ticker(user_input: str) -> str:
    """将用户输入解析为股票代码。"""
    if not user_input: return ""
    user_input_upper = user_input.strip().upper()
    if user_input_upper in SUPPORTED_TICKERS: return user_input_upper
    for ticker, name in TICKER_NAME_MAP.items():
        if user_input_upper == name.upper() or user_input_upper in name.upper(): return ticker
    for item in stock_universe:
        if user_input_upper in item["ticker"] or user_input_upper in item["name"].upper(): return item["ticker"]
    return user_input_upper


def search_tickers(query: str, top_k: int = 5):
    """根据输入返回匹配的股票代码与名称"""
    q = query.strip().upper()
    if not q: return stock_universe[:top_k]
    results = [item for item in stock_universe if q in item["ticker"] or q in item["name"].upper()]
    return results[:top_k]


@st.cache_data
def load_stock_data_cached(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """缓存动态股票数据加载结果。"""
    return load_stock_data_dynamic(ticker, start_date, end_date)


@st.cache_resource
def get_data_provider() -> DataProvider:
    """全局共享的数据提供层（带 TTL 缓存与降级）。"""
    return DataProvider()


@st.cache_data(ttl=120, show_spinner=False)
def load_realtime_data(ticker: str, lookback_days: int = 400) -> Dict[str, Any]:
    """实时模式数据加载：用 DataProvider + RealtimeFeatureBuilder 现算特征。

    返回与回测模式对齐的结构，并附带数据来源（live/fallback）。
    依赖 pandas_ta；若缺失则返回 error，由上层提示切回回测模式。
    """
    import importlib

    # 主动清掉可能的失败模块缓存，处理"启动 streamlit 时依赖尚未装好"的边界
    importlib.invalidate_caches()
    try:
        from agents.realtime_features import RealtimeFeatureBuilder
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        # 区分两种典型情况：①真的没装 ②已装但 Python 之前缓存了失败状态
        hint = ""
        try:
            import pandas_ta  # noqa: F401
            hint = (
                "（pandas_ta 已在 venv 中存在，但当前 streamlit 进程未能导入。"
                "请重启 streamlit：关闭后用 `.venv\\Scripts\\python.exe run_app.py` 重新启动。）"
            )
        except Exception:
            hint = (
                "（请在 venv 中执行 `pip install pandas_ta` 后再重启 streamlit。）"
            )
        return {
            "error": f"实时特征模块加载失败：{exc} {hint}",
            "traceback": tb,
            "data": pd.DataFrame(),
        }

    provider = get_data_provider()
    builder = RealtimeFeatureBuilder(provider=provider)
    built = builder.build_features(ticker, lookback_days=lookback_days)
    return built


@st.cache_data(ttl=120, show_spinner=False)
def get_realtime_news_scored(ticker: str, limit: int = 8) -> List[Dict]:
    """获取最近新闻并用 FinBERT 打分（实时新闻情感流）。FinBERT 不可用时分数为 None。"""
    provider = get_data_provider()
    raw = provider.get_news(ticker, limit=limit)
    if not raw:
        return []
    titles = [n.get("title", "") for n in raw]
    scores: List[Optional[float]] = [None] * len(titles)
    try:
        analyzer = FinBERTAnalyzer()
        classifications = analyzer.classify_texts(titles)
        for i, res in enumerate(classifications):
            if isinstance(res, dict) and "scores" in res:
                s = res["scores"]
                scores[i] = float(s.get("Positive", s.get("positive", 0.0))) - float(
                    s.get("Negative", s.get("negative", 0.0))
                )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"实时新闻情感打分失败: {exc}")

    out = []
    for n, sc in zip(raw, scores):
        out.append(
            {
                "Date": n.get("publish_time", ""),
                "Title": n.get("title", ""),
                "Sentiment": float(sc) if sc is not None else 0.0,
                "Source": n.get("source", "News"),
            }
        )
    return out

@st.cache_data
def get_tech_indicators_cached(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """缓存技术指标计算结果。"""
    return generate_tech_indicators(df, normalize=normalize, return_scaler=False)

# ==========================================
# 预测工具已移至 agents.Tools.predict_t+1
# ==========================================


def compute_and_merge_tech_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标并合并到原始 DataFrame。"""
    try:
        if len(data) == 0:
            st.warning(f"⚠️ 数据量不足（0行），某些技术指标可能无法计算。")
            return data
        if len(data) < 20:
            st.warning(f"⚠️ 数据量不足（{len(data)} 行），某些技术指标可能无法计算。")

        data_with_indicators = get_tech_indicators_cached(data, normalize=False)
        tech_cols = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'SMA_5', 'SMA_20', 'Volatility']
        available_tech_cols = [col for col in tech_cols if col in data_with_indicators.columns]

        if available_tech_cols:
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            data_with_indicators['Date'] = pd.to_datetime(data_with_indicators['Date']).dt.date
            tech_data_to_merge = data_with_indicators[['Date'] + available_tech_cols]
            data = pd.merge(data, tech_data_to_merge, on='Date', how='left', suffixes=('', '_tech'))
            for col in available_tech_cols:
                if f'{col}_tech' in data.columns:
                    data[col] = data[f'{col}_tech']
                    data = data.drop(columns=[f'{col}_tech'])
    except Exception as e:
        warnings.warn(f"技术指标计算失败: {e}，将继续使用原始数据")
        st.error(f"❌ 技术指标计算失败: {e}")
        with st.expander("🔍 查看详细错误信息"):
            st.code(traceback.format_exc(), language="python")
    return data

# Hybrid 多任务模型预测器（缓存）
@st.cache_resource
def load_hybrid_predictor_cached():
    """加载 Hybrid 预测器（同时预测方向与幅度）。"""
    try:
        predictor = HybridPredictor()
        return predictor
    except FileNotFoundError as e:
        st.error(f"❌ Hybrid 模型文件未找到: {e}")
        st.info("💡 请确认已运行 `train_hybridmodel_with_optuna.py` 并已生成模型文件。")
        return None
    except Exception as e:
        st.error(f"❌ Hybrid 预测器加载失败: {e}")
        return None


# ==========================================
# 3. 侧边栏控制区 (Controller)
# ==========================================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

with st.sidebar:
    # ============ 品牌 ============ #
    st.markdown(
        f"""
        <div class="fa-sb-brand">
            <div class="logo">FA</div>
            <div>
                <div class="name">{BRAND_NAME}</div>
                <div class="ver">v2.0 · 量化分析助手</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============ 模型连接 ============ #
    st.markdown('<div class="fa-sb-label">模型连接</div>', unsafe_allow_html=True)
    deepseek_api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        help="输入您的 DeepSeek API Key 以启用 AI 决策分析。留空时仅本地量化工具可用。",
        placeholder="sk-...",
        label_visibility="collapsed",
    )

    if deepseek_api_key:
        if deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
            st.markdown(pill("API Key 已配置 · 智能体就绪", "live"), unsafe_allow_html=True)
        else:
            st.markdown(pill("Key 格式有误，请检查", "warn"), unsafe_allow_html=True)
    else:
        st.markdown(pill("未连接 · 仅可使用本地量化工具", "muted"), unsafe_allow_html=True)

    # ============ 数据模式 ============ #
    st.markdown('<div class="fa-sb-label">数据模式</div>', unsafe_allow_html=True)
    data_mode = st.radio(
        "数据来源",
        ["回测模式", "实时模式"],
        horizontal=True,
        label_visibility="collapsed",
        help="回测模式：本地历史数据 + 模拟日期回放。实时模式：yfinance 在线行情/新闻，失败自动回退本地。",
    )
    realtime_mode = data_mode == "实时模式"

    realtime_lookback = 400
    auto_refresh = False
    if realtime_mode:
        realtime_lookback = st.slider(
            "回溯窗口（天）",
            min_value=120, max_value=750, value=400, step=30,
            help="拉取多少天日线用于现算特征（Beta_60 等长窗口指标需要足够历史）。",
        )
        auto_refresh = st.checkbox(
            "每 30 秒自动刷新",
            value=False,
            help=(
                "开启后页面定时拉取最新行情。"
                if _AUTOREFRESH_AVAILABLE
                else "未安装 streamlit-autorefresh，已禁用。"
            ),
        )
        if auto_refresh and _AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=30_000, key="realtime_autorefresh")

    # ============ 标的与时间 ============ #
    st.markdown('<div class="fa-sb-label">分析标的</div>', unsafe_allow_html=True)

    supported_tickers_help = ", ".join(SUPPORTED_TICKERS)
    default_ticker = "EBAY" if "EBAY" in SUPPORTED_TICKERS else (SUPPORTED_TICKERS[0] if SUPPORTED_TICKERS else "MS")

    # 若用户从「历史对话」面板切换到另一只股票，下一轮 rerun 时把它同步到输入框
    _pending_ticker = st.session_state.pop("_pending_ticker_switch", None)
    if _pending_ticker:
        st.session_state["ticker_input"] = _pending_ticker

    ticker = st.text_input(
        "股票代码",
        st.session_state.get("ticker_input", default_ticker),
        help=f"代号或中文名均可（覆盖：{supported_tickers_help}）",
        placeholder="如 EBAY 或 eBay",
        key="ticker_input",
        label_visibility="collapsed",
    )

    suggestions = search_tickers(ticker)
    if suggestions:
        st.caption("匹配建议")
        # 使用更紧凑的内联展示
        chips_html = " ".join(
            f"<code style='background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:0.72rem;"
            f"color:#334155;border:1px solid #e2e8f0;margin-right:4px;'>"
            f"{item['ticker']} · {item['name']}</code>"
            for item in suggestions
        )
        st.markdown(chips_html, unsafe_allow_html=True)

    model_type = "混合模型"  # 原 Hybrid（多任务融合）

    if not realtime_mode:
        st.markdown('<div class="fa-sb-label">回测时间窗</div>', unsafe_allow_html=True)
        st.caption(
            f"可选范围 {KAGGLE_START_DATE.strftime('%Y-%m-%d')} ~ {KAGGLE_END_DATE.strftime('%Y-%m-%d')}"
        )

        col_start, col_end = st.columns(2)
        with col_start:
            default_start = max(KAGGLE_START_DATE, KAGGLE_END_DATE - timedelta(days=365))
            start_date = st.date_input(
                "开始", value=default_start,
                min_value=KAGGLE_START_DATE, max_value=KAGGLE_END_DATE,
            )
        with col_end:
            end_date = st.date_input(
                "结束", value=KAGGLE_END_DATE,
                min_value=KAGGLE_START_DATE, max_value=KAGGLE_END_DATE,
            )

        if start_date >= end_date:
            st.error("开始日期必须早于结束日期")
            st.stop()

        simulation_date = st.date_input(
            "模拟当前日（T 日）",
            value=end_date, min_value=start_date, max_value=end_date,
            help="把这个日期当作 T 日（即模拟的当前日），分析其前后窗口的数据。",
        )
    else:
        start_date = KAGGLE_START_DATE
        end_date = date.today()
        simulation_date = date.today()
        st.caption("实时模式 · 日期跟随最新行情，无需手动选择")

    # ============ 主行动按钮 ============ #
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    run_btn = st.button("运行分析", type="primary", use_container_width=True)
    if run_btn:
        st.session_state.run_analysis = True

    # =================================================== #
    # 历史对话面板（多会话 / 重载 / 重命名 / 删除）
    # =================================================== #
    if "_conversation_store" not in st.session_state:
        st.session_state["_conversation_store"] = ConversationStore()
    conv_store: ConversationStore = st.session_state["_conversation_store"]

    st.markdown('<div class="fa-sb-label">历史对话</div>', unsafe_allow_html=True)
    all_convs = conv_store.list()

    # —— 新建对话按钮 —— #
    if st.button("＋ 新建对话", use_container_width=True, key="conv_new_btn"):
        _new_ticker = (ticker or "").strip().upper() or "NEW"
        new_conv = conv_store.create(ticker=_new_ticker)
        st.session_state["active_conversation_id"] = new_conv.id
        st.session_state.pop("messages", None)
        st.session_state.pop("current_ticker", None)
        st.rerun()

    st.caption(f"已归档 {len(all_convs)} 条 · 自动落盘 `.cache/conversations.json`")

    if not all_convs:
        st.caption("（提一个问题之后，对话会自动出现在这里）")
    else:
        active_id = st.session_state.get("active_conversation_id")
        with st.container():
            for conv in all_convs[:30]:
                is_active = conv.id == active_id
                col_sel, col_act = st.columns([5, 1])
                with col_sel:
                    label_prefix = "● " if is_active else "○ "
                    btn_label = f"{label_prefix}{conv.title}"
                    if st.button(
                        btn_label,
                        key=f"conv_sel_{conv.id}",
                        help=f"{conv.ticker or '—'} · {conv.preview(40)} · {conv.updated_at}",
                        use_container_width=True,
                    ):
                        st.session_state["active_conversation_id"] = conv.id
                        st.session_state.pop("messages", None)
                        st.session_state.pop("current_ticker", None)
                        if conv.ticker:
                            st.session_state["_pending_ticker_switch"] = conv.ticker
                        st.rerun()
                with col_act:
                    if st.button("⋯", key=f"conv_menu_{conv.id}", help="重命名 / 删除"):
                        st.session_state["_managing_conv_id"] = (
                            None if st.session_state.get("_managing_conv_id") == conv.id else conv.id
                        )

                if st.session_state.get("_managing_conv_id") == conv.id:
                    with st.container(border=True):
                        new_title = st.text_input(
                            "重命名",
                            value=conv.title,
                            key=f"conv_rename_input_{conv.id}",
                            label_visibility="collapsed",
                        )
                        c_rename, c_delete, c_close = st.columns(3)
                        with c_rename:
                            if st.button("保存", key=f"conv_rename_save_{conv.id}",
                                         use_container_width=True):
                                conv_store.rename(conv.id, new_title)
                                st.session_state["_managing_conv_id"] = None
                                st.rerun()
                        with c_delete:
                            if st.button("删除", key=f"conv_delete_{conv.id}",
                                         use_container_width=True):
                                conv_store.delete(conv.id)
                                if st.session_state.get("active_conversation_id") == conv.id:
                                    st.session_state.pop("active_conversation_id", None)
                                    st.session_state.pop("messages", None)
                                st.session_state["_managing_conv_id"] = None
                                st.rerun()
                        with c_close:
                            if st.button("关闭", key=f"conv_close_{conv.id}",
                                         use_container_width=True):
                                st.session_state["_managing_conv_id"] = None
                                st.rerun()

    # ============ 系统状态 ============ #
    st.markdown('<div class="fa-sb-label">系统状态</div>', unsafe_allow_html=True)
    _status_pills: List[str] = []
    _status_pills.append(pill(
        f"数据 · {data_mode}",
        "live" if realtime_mode else "bt",
    ))
    _status_pills.append(pill("FinBERT 已加载", "muted"))
    _status_pills.append(pill("混合预测模型 就绪", "muted"))
    if deepseek_api_key and deepseek_api_key.strip():
        if deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
            _status_pills.append(pill(f"DeepSeek · {deepseek_settings.deepseek_model}", "live"))
        else:
            _status_pills.append(pill("DeepSeek · 配置异常", "warn"))
    else:
        _status_pills.append(pill("DeepSeek · 降级模式", "muted"))
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:6px;line-height:1.8;'>"
        + " ".join(_status_pills)
        + "</div>",
        unsafe_allow_html=True,
    )

# ==========================================
# 4. 主界面（Hero + 主体）
# ==========================================
try:
    resolved_ticker_display = resolve_ticker(ticker) if ticker else "EBAY"
except Exception:
    resolved_ticker_display = ticker if ticker else "EBAY"

# ---------- Hero 头部 ---------- #
if realtime_mode:
    _meta = st.session_state.get("_rt_meta", {})
    _src_label = {"live": "yfinance · 实时", "fallback": "本地回退", "none": "无数据"}.get(
        _meta.get("source", "live"), "实时"
    )
    _mode_pill = pill(f"实时 · {_src_label}", "live")
    _date_pill = pill(f"截至 {_meta.get('as_of', '最新')}", "muted")
else:
    _mode_pill = pill("回测模式", "bt")
    _date_pill = pill(
        f"{KAGGLE_START_DATE.strftime('%Y-%m')} ~ {KAGGLE_END_DATE.strftime('%Y-%m')}",
        "muted",
    )
_model_pill = pill(f"模型 · {model_type}", "muted")
_target_pill = pill(f"标的 · {resolved_ticker_display}", "muted")

st.markdown(
    f"""
    <div class="fa-hero">
        <div class="fa-hero-left">
            <div class="fa-logo">FA</div>
            <div class="fa-brand">
                <div class="fa-brand-name">{BRAND_NAME}</div>
                <div class="fa-brand-sub">{BRAND_TAGLINE}</div>
            </div>
        </div>
        <div class="fa-hero-right">
            {_target_pill}{_mode_pill}{_date_pill}{_model_pill}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

run_analysis = st.session_state.get("run_analysis", False)

if not run_analysis:
    # 更克制、更专业的"待启动"提示
    st.markdown(
        """
        <div class="fa-card" style="text-align:center;padding:40px 24px;">
            <div style="font-size:0.92rem;color:#0f172a;font-weight:600;margin-bottom:6px;">
                工作台已就绪
            </div>
            <div style="font-size:0.84rem;color:#64748b;line-height:1.7;">
                请在左侧选择分析标的与时间窗，确认无误后点击
                <span style="background:#2563eb;color:#fff;padding:2px 8px;border-radius:4px;
                font-size:0.78rem;margin:0 2px;">运行分析</span>
                开始本次会话。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        resolved_ticker = resolve_ticker(ticker)
        if realtime_mode:
            _rt = load_realtime_data(resolved_ticker, lookback_days=realtime_lookback)
            if _rt.get("error") or _rt.get("data") is None or _rt["data"].empty:
                st.error(f"实时数据获取失败：{_rt.get('error') or '无有效数据'}")
                if _rt.get("traceback"):
                    with st.expander("查看完整 traceback", expanded=False):
                        st.code(_rt["traceback"], language="python")
                # 给用户一个清理 streamlit 缓存的快捷按钮
                col_a, col_b, _ = st.columns([1, 1, 4])
                with col_a:
                    if st.button("清空缓存并重试", key="rt_clear_cache"):
                        st.cache_data.clear()
                        st.rerun()
                with col_b:
                    st.caption("或在左侧切换到「回测模式」继续使用本地历史数据。")
                st.stop()
            data = _rt["data"].copy()
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            st.session_state["_rt_meta"] = {
                "source": _rt.get("source", "live"),
                "as_of": _rt.get("as_of", ""),
            }
        else:
            data = load_stock_data_cached(resolved_ticker, start_date, end_date)
        
        # 检查 Sentiment_Score 列（数据中应该已有）
        if 'Sentiment_Score' not in data.columns:
            # 如果数据中没有，尝试计算
            if 'NewsTitles' in data.columns:
                try:
                    analyzer = FinBERTAnalyzer(use_finbert_tone=True)
                    sentiments = []
                    for news_text in data['NewsTitles']:
                        if pd.notna(news_text) and str(news_text).strip():
                            news_list = str(news_text).split(" | ")
                            sentiment = analyzer.calc_score(news_list)
                            sentiments.append(sentiment)
                        else:
                            sentiments.append(0.0)
                    data['Sentiment_Score'] = sentiments
                except Exception as e:
                    warnings.warn(f"计算情感分数失败: {e}，将使用默认值 0.0")
                    data['Sentiment_Score'] = 0.0
            else:
                data['Sentiment_Score'] = 0.0
        
        # 计算市场情绪指数（聚合情感分数）
        if 'Sentiment_Score' in data.columns and len(data) > 0:
            try:
                data['Aggregated_Sentiment'] = calculate_aggregated_sentiment(
                    data, 
                    window=7,  # 7天滚动窗口
                    use_market_adjustment=True  # 结合市场指标调整
                )
                aggregated_sentiment_available = True
            except Exception as e:
                warnings.warn(f"计算市场情绪指数失败: {e}，将使用默认值")
                data['Aggregated_Sentiment'] = 0.0
                aggregated_sentiment_available = False
        else:
            data['Aggregated_Sentiment'] = 0.0
            aggregated_sentiment_available = False
        
        # 检查数据是否为空
        if len(data) == 0:
            st.error(f"❌ 未找到股票 {resolved_ticker} 在日期范围 {start_date} 至 {end_date} 内的数据")
            st.info("💡 提示：请检查股票代码是否正确，或调整日期范围。")
            st.stop()
        
        data = compute_and_merge_tech_indicators(data)
        if realtime_mode:
            news = get_realtime_news_scored(resolved_ticker, limit=8)
        else:
            news = extract_news_before_date(resolved_ticker, simulation_date, days_before=5, max_news_per_day=3)
        
        # 查找小于等于模拟日期的数据，如果精确匹配不存在，找最近的交易日
        simulation_data = data[data['Date'] <= simulation_date]
        if len(simulation_data) == 0:
            # 尝试找最近的交易日
            available_dates = data[data['Date'] <= simulation_date + timedelta(days=30)]['Date']
            if len(available_dates) > 0:
                nearest_date = available_dates.max()
                st.warning(f"⚠️ 模拟日期 {simulation_date} 不是交易日，使用最近的交易日 {nearest_date}")
                simulation_data = data[data['Date'] <= nearest_date]
            else:
                st.error(f"❌ 模拟日期 {simulation_date} 在数据范围内没有找到对应的交易日数据")
                st.info(f"💡 可用日期范围：{data['Date'].min()} 至 {data['Date'].max()}")
                st.stop()
        
        current_price = simulation_data['Close'].iloc[-1]
        current_sentiment = simulation_data['Sentiment_Score'].iloc[-1]
        # 获取市场情绪指数（如果存在）
        if 'Aggregated_Sentiment' in simulation_data.columns:
            current_aggregated_sentiment = simulation_data['Aggregated_Sentiment'].iloc[-1]
            if pd.isna(current_aggregated_sentiment):
                current_aggregated_sentiment = None
        else:
            current_aggregated_sentiment = None
        current_date = simulation_data['Date'].iloc[-1]
        
        if len(simulation_data) > 1:
            prev_price = simulation_data['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            price_change_str = f"{price_change_pct:+.2f}% 较前日"
        else:
            price_change_str = "无前日数据"
        
        sentiment_trend = "正面趋势" if current_sentiment > 0.3 else "负面趋势" if current_sentiment < -0.3 else "中性趋势"
        
    except Exception as e:
        st.error(f"❌ 数据加载或预处理失败: {e}")
        with st.expander("🔍 查看详细错误信息"):
            st.code(traceback.format_exc(), language="python")
        st.stop()
    
    predicted_price: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    prediction_error: Optional[str] = None
    direction_prob: Optional[float] = None
    pred_magnitude: Optional[float] = None
    final_log_return: Optional[float] = None
    
    if model_type == "混合模型":
        try:
            predictor = load_hybrid_predictor_cached()
            
            if predictor:
                # 使用预测工具进行推理
                result = predictor.predict(
                    simulation_data=simulation_data,
                    current_price=current_price,
                    news_column='NewsTitles'
                )
                
                predicted_price = result["predicted_price"]
                predicted_change_pct = result["predicted_change_pct"]
                direction_prob = result["direction_prob"]
                pred_magnitude = result["pred_magnitude"]
                final_log_return = result["final_log_return"]
                prediction_error = result["error"]
            else:
                predicted_price = current_price * 0.965
                predicted_change_pct = -3.5
                prediction_error = "Hybrid 预测器加载失败"
                direction_prob = None
                pred_magnitude = None
                final_log_return = None
        except Exception as e:
            st.warning(f"⚠️ 流程严重错误: {e}")
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc(), language='python')
            predicted_price = current_price * 0.965
            predicted_change_pct = -3.5
            prediction_error = str(e)
            direction_prob = None
            pred_magnitude = None
            final_log_return = None

    # --- 关键指标计算（用于仪表盘） ---
    # 指标3: 宏观市场风险
    risk_regime = "N/A"
    risk_rationale = "计算失败"
    try:
        macro_analyzer = MacroAnalyzer()
        _macro_df = get_data_provider().get_macro() if realtime_mode else None
        macro_result = macro_analyzer.analyze_market_regime(
            current_date=simulation_date.strftime('%Y-%m-%d'), df=_macro_df
        )
        if "error" not in macro_result:
            risk_regime = macro_result.get("risk_regime", "N/A")
            risk_rationale = macro_result.get("rationale", "计算失败")
        else:
            risk_rationale = macro_result.get("error", "计算失败")
    except Exception as e:
        risk_rationale = f"计算失败: {str(e)}"
    
    # 指标4: 关键价位（支撑位和阻力位）
    support_price = None
    resistance_price = None
    support_resistance_status = "计算失败"
    try:
        sr_scanner = SupportResistanceScanner()
        sr_result = sr_scanner.calculate_levels(
            df=simulation_data,
            current_price=current_price,
            window=252,
            use_weighted=True
        )
        if sr_result.get("nearest_support") is not None:
            support_price = sr_result["nearest_support"]["price"]
        if sr_result.get("nearest_resistance") is not None:
            resistance_price = sr_result["nearest_resistance"]["price"]
        support_resistance_status = sr_result.get("status", "计算失败")
    except Exception as e:
        support_resistance_status = f"计算失败: {str(e)}"
    
    # 指标5: RSI
    rsi_value = None
    rsi_status = "数据不足"
    try:
        if 'RSI_14' in simulation_data.columns and len(simulation_data) > 0:
            rsi_value = simulation_data['RSI_14'].iloc[-1]
            if pd.notna(rsi_value):
                if rsi_value > 70:
                    rsi_status = "超买区域"
                elif rsi_value < 30:
                    rsi_status = "超卖区域"
                else:
                    rsi_status = "中性区域"
            else:
                rsi_status = "数据缺失"
    except Exception as e:
        rsi_status = f"计算失败: {str(e)}"

    # ============ 核心指标快照（前置，第一眼看见的信息）============ #
    section_label(
        "核心指标快照",
        f"模拟 T 日 · {simulation_date.strftime('%Y-%m-%d')}",
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if current_price is not None:
            delta_value = price_change_str if len(simulation_data) > 1 else "无前日数据"
            delta_color = "off"
            if len(simulation_data) > 1:
                delta_color = "inverse" if price_change_pct < 0 else "normal"
            st.metric(
                label="收盘价（T 日）",
                value=f"${current_price:,.2f}",
                delta=delta_value,
                delta_color=delta_color,
                help=f"模拟日期 {simulation_date.strftime('%Y-%m-%d')} 的收盘价。",
            )
        else:
            st.metric(label="收盘价（T 日）", value="N/A", delta="数据缺失")

    with col2:
        if predicted_price is not None and prediction_error is None:
            delta_value = f"{predicted_change_pct:+.2f}%"
            delta_color = "inverse" if predicted_change_pct < 0 else "normal"
            st.metric(
                label="T+1 预测",
                value=f"${predicted_price:,.2f}",
                delta=delta_value,
                delta_color=delta_color,
                help="基于多模态混合模型的下一交易日收盘价预测。",
            )
        else:
            error_msg = prediction_error if prediction_error else "计算失败"
            st.metric(label="T+1 预测", value="N/A", delta=error_msg)

    with col3:
        if risk_regime != "N/A":
            st.metric(
                label="宏观风险",
                value=risk_regime,
                delta=risk_rationale,
                help="基于 ^GSPC / ^VIX / ^TNX 的市场状态综合评估。",
            )
        else:
            st.metric(label="宏观风险", value="N/A", delta=risk_rationale)

    with col4:
        if support_price is not None or resistance_price is not None:
            support_str = f"{support_price:.2f}" if support_price is not None else "—"
            resistance_str = f"{resistance_price:.2f}" if resistance_price is not None else "—"
            st.metric(
                label="支撑 | 阻力",
                value=f"{support_str} / {resistance_str}",
                delta=support_resistance_status,
                help="基于 KDE 算法的历史成交密集区。",
            )
        else:
            st.metric(label="支撑 | 阻力", value="— / —", delta=support_resistance_status)

    with col5:
        if rsi_value is not None and pd.notna(rsi_value):
            delta_color = "inverse" if rsi_value > 70 else "normal" if rsi_value < 30 else "off"
            st.metric(
                label="RSI (14)",
                value=f"{rsi_value:.1f}",
                delta=rsi_status,
                delta_color=delta_color,
                help="相对强弱指数：>70 超买 / <30 超卖。",
            )
        else:
            st.metric(label="RSI (14)", value="N/A", delta=rsi_status)

    # ============ 价格 × 情感联动 + 近期新闻 ============ #
    section_label("价格走势与新闻情绪", "K 线 · FinBERT 情感分 · 聚合市场情绪")

    col_chart, col_news = st.columns([2, 1])

    with col_chart:
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="fa-card-header"><span class="fa-card-title">价格 × 情感联动</span>'
            '<span class="fa-card-sub">主纵轴：价格 · 副纵轴：情感分</span></div>',
            unsafe_allow_html=True,
        )
        
        fig = go.Figure()
        # 专业 K 线图（红涨绿跌），若缺 OHLC 则回退为收盘价折线
        if all(c in data.columns for c in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(go.Candlestick(
                x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
            ))
        else:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='股价', line=dict(color='#2563eb', width=2)))
        if 'Sentiment_Score' in data.columns:
            colors = ['#ef4444' if v < 0 else '#22c55e' for v in data['Sentiment_Score']]
            fig.add_trace(go.Bar(x=data['Date'], y=data['Sentiment_Score'], name='新闻情感（FinBERT）', yaxis='y2', marker_color=colors, opacity=0.5))
        
        if aggregated_sentiment_available and 'Aggregated_Sentiment' in data.columns:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Aggregated_Sentiment'], name='市场情绪指数（聚合）', yaxis='y2', line=dict(color='#8b5cf6', width=2, dash='dash'), opacity=0.8))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=380, xaxis_rangeslider_visible=False, yaxis=dict(title="价格 ($)", showgrid=True, gridcolor='#f1f5f9'), yaxis2=dict(title="情感得分", overlaying='y', side='right', range=[-1.5, 1.5], showgrid=False), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if 'RSI_14' in data.columns and 'MACD_12_26_9' in data.columns:
            st.markdown('<div class="fa-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="fa-card-header"><span class="fa-card-title">技术指标面板</span>'
                '<span class="fa-card-sub">RSI · MACD · SMA</span></div>',
                unsafe_allow_html=True,
            )
            
            fig_indicators = make_subplots(rows=3, cols=1, subplot_titles=('RSI', 'MACD', 'SMA'), vertical_spacing=0.15, row_heights=[0.28, 0.32, 0.40])
            
            rsi_data = data['RSI_14'].dropna()
            if len(rsi_data) > 0:
                fig_indicators.add_trace(go.Scatter(x=data.loc[rsi_data.index, 'Date'], y=rsi_data, name='RSI_14', line=dict(color='#8b5cf6', width=2), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'), row=1, col=1)
                fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买", row=1, col=1)
                fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖", row=1, col=1)
            
            macd_data, macds_data, macdh_data = data['MACD_12_26_9'].dropna(), data['MACDs_12_26_9'].dropna(), data['MACDh_12_26_9'].dropna()
            if len(macd_data) > 0:
                fig_indicators.add_trace(go.Scatter(x=data.loc[macd_data.index, 'Date'], y=macd_data, name='MACD', line=dict(color='#2563eb', width=2)), row=2, col=1)
                fig_indicators.add_trace(go.Scatter(x=data.loc[macds_data.index, 'Date'], y=macds_data, name='Signal', line=dict(color='#ef4444', width=2)), row=2, col=1)
                colors_hist = ['#22c55e' if v >= 0 else '#ef4444' for v in macdh_data]
                fig_indicators.add_trace(go.Bar(x=data.loc[macdh_data.index, 'Date'], y=macdh_data, name='Histogram', marker_color=colors_hist, opacity=0.6), row=2, col=1)
            
            close_data, sma5_data, sma20_data = data['Close'], data['SMA_5'].dropna(), data['SMA_20'].dropna()
            if len(sma5_data) > 0 and len(sma20_data) > 0:
                sma_idx = sma5_data.index.intersection(sma20_data.index)
                if len(sma_idx) > 0:
                    fig_indicators.add_trace(go.Scatter(x=data.loc[sma_idx, 'Date'], y=close_data.loc[sma_idx], name='收盘价', line=dict(color='#1e293b', width=2)), row=3, col=1)
                    fig_indicators.add_trace(go.Scatter(x=data.loc[sma_idx, 'Date'], y=sma5_data.loc[sma_idx], name='SMA_5', line=dict(color='#f59e0b', width=2, dash='dash')), row=3, col=1)
                    fig_indicators.add_trace(go.Scatter(x=data.loc[sma_idx, 'Date'], y=sma20_data.loc[sma_idx], name='SMA_20', line=dict(color='#3b82f6', width=2, dash='dash')), row=3, col=1)

            fig_indicators.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=700, showlegend=True, legend=dict(orientation="h", y=-0.12))
            fig_indicators.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
            fig_indicators.update_yaxes(title_text="MACD", row=2, col=1)
            fig_indicators.update_yaxes(title_text="价格 ($)", row=3, col=1)
            fig_indicators.update_xaxes(title_text="日期", row=3, col=1)
            st.plotly_chart(fig_indicators, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_news:
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)
        if realtime_mode:
            news_sub = f"{resolved_ticker_display} · 最近新闻 · FinBERT 实时打分"
        else:
            news_sub = f"T 日（{simulation_date.strftime('%Y-%m-%d')}）前 5 天新闻 · 取代表性 5 条"
        st.markdown(
            f'<div class="fa-card-header"><span class="fa-card-title">近期新闻情绪</span>'
            f'<span class="fa-card-sub">{news_sub}</span></div>',
            unsafe_allow_html=True,
        )

        if not news:
            st.markdown(
                '<div style="text-align:center;padding:24px 6px;color:#94a3b8;font-size:0.84rem;">'
                '该时间窗口内暂无新闻数据</div>',
                unsafe_allow_html=True,
            )
        else:
            def select_balanced_news(news_list: List[Dict], max_count: int = 5) -> List[Dict]:
                if len(news_list) <= max_count:
                    return news_list
                positive = sorted([n for n in news_list if n['Sentiment'] > 0],
                                  key=lambda x: abs(x['Sentiment']), reverse=True)
                negative = sorted([n for n in news_list if n['Sentiment'] < 0],
                                  key=lambda x: abs(x['Sentiment']), reverse=True)
                selected = positive[:(max_count // 2 + max_count % 2)] + negative[:max_count // 2]
                if len(selected) < max_count:
                    remaining = [n for n in news_list if n not in selected]
                    selected.extend(
                        sorted(remaining, key=lambda x: abs(x['Sentiment']), reverse=True)[
                            : max_count - len(selected)
                        ]
                    )
                return sorted(selected, key=lambda x: x['Date'], reverse=True)

            for item in select_balanced_news(news, max_count=5):
                if item['Sentiment'] > 0.05:
                    side_cls = "pos"; score_cls = "pos"
                elif item['Sentiment'] < -0.05:
                    side_cls = "neg"; score_cls = "neg"
                else:
                    side_cls = ""; score_cls = "neu"
                st.markdown(
                    f"""
                    <div class="fa-news-item {side_cls}">
                        <div class="fa-news-meta">{item['Date']} · {item['Source']}</div>
                        <div class="fa-news-title">{item['Title']}</div>
                        <div class="fa-news-score {score_cls}">情感 {item['Sentiment']:+.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # ============ 智能体对话区 ============ #
    section_label("FinanceAgent 对话", "可基于上方数据上下文进行深度推理与工具调用")

    # ---- 多会话：决定当前要展示哪一段历史 ---- #
    conv_store: ConversationStore = st.session_state["_conversation_store"]

    def _welcome_message(t_display: str) -> Dict[str, str]:
        return {
            "role": "assistant",
            "content": (
                f"您好，我是 **FinanceAgent**，本次会话已锁定分析标的 **{t_display}**。\n\n"
                "我可以为您完成：\n"
                "- T+1 走势预测与方向概率\n"
                "- 支撑 / 阻力位定位\n"
                "- 相似历史 K 线形态回测\n"
                "- SEC 财报健康度评估\n"
                "- 多模态综合投资决策报告\n\n"
                "您可以使用下方的「推荐分析」一键启动常用流程，或直接在输入框中描述您的问题。"
            ),
        }

    active_id: Optional[str] = st.session_state.get("active_conversation_id")
    active_conv: Optional[Conversation] = conv_store.get(active_id) if active_id else None

    # 没有激活会话 → 按 ticker 找一个"最近会话"或新建一个
    if active_conv is None:
        matching = [c for c in conv_store.list() if c.ticker.upper() == (resolved_ticker or "").upper()]
        if matching:
            active_conv = matching[0]
        else:
            active_conv = conv_store.create(
                ticker=resolved_ticker,
                initial_messages=[_welcome_message(resolved_ticker_display)],
            )
        st.session_state["active_conversation_id"] = active_conv.id

    # 把 active conversation 的 messages 拷贝到 session_state.messages
    needs_reload = (
        "messages" not in st.session_state
        or st.session_state.get("current_ticker") != resolved_ticker
        or st.session_state.get("_loaded_conv_id") != active_conv.id
    )
    if needs_reload:
        if not active_conv.messages:
            active_conv.messages = [_welcome_message(resolved_ticker_display)]
            conv_store.save(active_conv)
        st.session_state.messages = list(active_conv.messages)
        st.session_state.current_ticker = resolved_ticker
        st.session_state["_loaded_conv_id"] = active_conv.id

    # —— 会话标题栏 —— #
    _msg_count = sum(1 for m in active_conv.messages if m.get("role") in ("user", "assistant"))
    st.markdown(
        f"""
        <div class="fa-card" style="padding:14px 18px;margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                <div>
                    <div style="font-size:0.96rem;font-weight:600;color:#0f172a;">
                        {active_conv.title}
                    </div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">
                        ID `{active_conv.id[:8]}` · 标的 {active_conv.ticker or '—'}
                        · {_msg_count} 条消息 · 更新于 {active_conv.updated_at}
                    </div>
                </div>
                <div style="display:flex;gap:6px;">
                    {pill('已自动保存', 'live')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============ 初始化 FinancialAgent ============ #
    _AGENT_VER = AGENT_INTERFACE_VERSION
    agent_initialized = False
    agent = None
    if deepseek_api_key and deepseek_api_key.strip() and deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
        try:
            agent_key = f"financial_agent_{hash(deepseek_api_key)}_{_AGENT_VER}"
            for old_key in [k for k in st.session_state
                            if k.startswith(f"financial_agent_{hash(deepseek_api_key)}_") and k != agent_key]:
                del st.session_state[old_key]
            cached = st.session_state.get(agent_key)
            if cached is not None and getattr(cached, "_agent_ver", None) != _AGENT_VER:
                del st.session_state[agent_key]
                cached = None
            if cached is None:
                with st.spinner("初始化 FinanceAgent..."):
                    _new_agent = FinancialAgent(
                        api_key=deepseek_api_key.strip(),
                        realtime=realtime_mode,
                        data_provider=get_data_provider(),
                    )
                    setattr(_new_agent, "_agent_ver", _AGENT_VER)
                    st.session_state[agent_key] = _new_agent
            agent = st.session_state[agent_key]
            agent.realtime = realtime_mode
            agent_initialized = True
        except Exception as e:
            st.warning(f"FinanceAgent 初始化失败：{e}")
            agent_initialized = False
            agent = None
    else:
        agent_initialized = False
        agent = None

    # 未启用 AI 时的引导卡片
    if not agent_initialized:
        st.markdown(
            """
            <div class="fa-card" style="background:#fffbeb;border-color:#fde68a;">
                <div style="font-size:0.86rem;color:#92400e;font-weight:600;margin-bottom:4px;">
                    AI 决策模块未启用
                </div>
                <div style="font-size:0.8rem;color:#78350f;line-height:1.6;">
                    在左侧配置 DeepSeek API Key 后，将可调用 LLM
                    串联宏观 / 技术 / 基本面工具产出结构化分析报告。
                    当前页面的图表、量化指标与新闻情绪不受影响。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============ 推荐分析任务（chip 形式）============ #
    preset_questions = {
        "card1": f"请对 {resolved_ticker_display} 进行一次短期技术分析，重点关注 T+1 走势预测与关键支撑阻力位。",
        "card2": f"请挖掘 {resolved_ticker_display} 的历史数据，搜索与最近 20 天走势最相似的 K 线形态，并回测这些形态出现后的市场表现。",
        "card3": f"请对 {resolved_ticker_display} 进行深度基本面分析，调用 SEC 财报工具评估最新财务健康度与长期投资价值。",
        "card4": f"请为 {resolved_ticker_display} 生成一份综合性投资决策报告，兼顾短期量化技术信号与长期基本面价值，并进行交叉验证，最后明确指出核心风险。",
    }
    preset_labels = {
        "card1": ("短期技术分析", "T+1 预测 · 支撑阻力"),
        "card2": ("相似形态回测", "K 线相似度 · 历史路径"),
        "card3": ("财报健康度", "SEC 10-K · 长期价值"),
        "card4": ("综合决策报告", "宏观 + 技术 + 基本面"),
    }

    def set_user_input(question_key):
        st.session_state.user_input = preset_questions[question_key]

    st.markdown(
        '<div style="font-size:0.78rem;color:#64748b;font-weight:500;margin:4px 0 8px;'
        'letter-spacing:0.04em;text-transform:uppercase;">推荐分析任务</div>',
        unsafe_allow_html=True,
    )
    card_cols = st.columns(4, gap="small")
    for col, key in zip(card_cols, ["card1", "card2", "card3", "card4"]):
        with col:
            label_main, label_sub = preset_labels[key]
            st.button(
                f"{label_main}\n{label_sub}",
                help=preset_questions[key],
                on_click=set_user_input,
                args=(key,),
                key=f"{key}_btn",
                use_container_width=True,
            )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # ============ 聊天历史渲染 ============ #
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _persist_active_conversation() -> None:
        """把 session_state.messages 落盘到 ConversationStore。"""
        try:
            conv_store.update_messages(active_conv.id, st.session_state.messages)
        except Exception as _persist_err:  # noqa: BLE001
            warnings.warn(f"会话保存失败: {_persist_err}")

    def handle_user_query(user_input: str):
        """统一处理来自输入框或预设卡片的问题。"""
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        _persist_active_conversation()
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. 流式调用 Agent，可视化「思考 → 工具调用 → 图表 → 流式回答」全过程
        #
        # 渲染策略（顺序正确 + 实时流式）：
        #   ① thinking_delta → 折叠 expander，不占主流位置
        #   ② tool_call/tool_result → st.status 状态栏，图表在其后立即渲染
        #   ③ answer_delta → **多段流式占位符**：
        #        每次 LLM 从工具调用切回输出文字时，开一个全新的 st.empty()，
        #        delta 实时写入。这样：
        #          - 工具图表已经先于该段占位符渲染 → 顺序天然正确
        #          - 文本以打字机方式逐字出现 → 用户体感即时反馈
        #   ④ final → 元数据写在最末 caption；不再二次重写正文，避免覆盖流式渲染
        with st.chat_message("assistant"):
            if not agent_initialized or agent is None:
                st.warning("AI 决策模块未启用：请在左侧配置 DeepSeek API Key 后再发起对话。")
                return

            agent.realtime = realtime_mode

            # 过滤掉欢迎语，避免污染 LLM 上下文
            clean_history = [
                msg for msg in st.session_state.messages[:-1]
                if "FinanceAgent" not in msg.get("content", "")
                and "我是金融分析智能体" not in msg.get("content", "")
            ]

            thinking_box = None
            thinking_text = ""
            pending_status = None
            chart_idx = 0
            final_meta = {"tool_count": 0, "total_tool_time": 0.0}
            stream_error = None

            # ---- 流式文本：多段占位符 + 打字机光标 ---- #
            current_answer_box = None     # 当前正在流式渲染的 st.empty
            current_segment = ""          # 当前段已累积的文本
            full_answer = ""              # 跨段累计的完整答案（用于持久化）
            CURSOR = " ▌"                  # 打字机光标，渲染完毕时移除

            def _finalize_current_segment():
                """把当前段从"打字中"切换到"已定稿"（去掉光标）。"""
                nonlocal current_answer_box, current_segment
                if current_answer_box is not None and current_segment:
                    current_answer_box.markdown(current_segment)
                current_answer_box = None
                current_segment = ""

            try:
                # 传入真实日期：实时模式用今天，回测模式用模拟日期
                _agent_date = (
                    date.today().strftime("%Y-%m-%d")
                    if realtime_mode
                    else simulation_date.strftime("%Y-%m-%d")
                )
                for event in agent.run_stream(
                    user_prompt=user_input,
                    chat_history=clean_history,
                    stock_data=simulation_data,
                    current_price=current_price,
                    ticker=resolved_ticker,
                    current_date=_agent_date,
                ):
                    etype = event.get("type")

                    # ── 思考过程：折叠 expander ──
                    if etype == "thinking_delta":
                        thinking_text += event.get("text", "")
                        if thinking_box is None:
                            thinking_box = st.expander("模型推理过程", expanded=False).empty()
                        thinking_box.markdown(thinking_text + CURSOR)

                    # ── 工具调用：先把当前文字段定稿，再开 status ──
                    elif etype == "tool_call":
                        _finalize_current_segment()  # 关键：文字段封口，下面的工具块跟在它后面
                        name = event.get("name", "")
                        args = event.get("args", {})
                        pending_status = st.status(f"调用工具 · `{name}`", expanded=False)
                        with pending_status:
                            st.caption("入参")
                            st.json(args)

                    # ── 工具结果：更新状态栏 + 立即内联渲染图表 ──
                    elif etype == "tool_result":
                        name = event.get("name", "")
                        result = event.get("result")
                        elapsed = event.get("elapsed", 0.0)
                        if pending_status is not None:
                            is_err = isinstance(result, dict) and (
                                result.get("status") == "error" or bool(result.get("error"))
                            )
                            with pending_status:
                                st.caption(f"返回结果（耗时 {elapsed:.2f}s）")
                                st.json(result if isinstance(result, (dict, list)) else {"value": result})
                            pending_status.update(
                                label=f"{'失败' if is_err else '完成'} · `{name}` ({elapsed:.2f}s)",
                                state="error" if is_err else "complete",
                            )
                            pending_status = None
                        # 图表紧接在对应工具状态栏之后渲染
                        try:
                            fig = cr.render_tool_chart(name, result, data, current_price)
                            if fig is not None:
                                chart_idx += 1
                                st.plotly_chart(
                                    fig, use_container_width=True, key=f"toolchart_{chart_idx}"
                                )
                        except Exception as _chart_err:  # noqa: BLE001
                            warnings.warn(f"工具图表渲染失败: {_chart_err}")

                    # ── 回答增量：实时流式渲染到当前段占位符 ──
                    elif etype == "answer_delta":
                        text_chunk = event.get("text", "")
                        if not text_chunk:
                            continue
                        if current_answer_box is None:
                            # 新一段：占位符位置自然位于已渲染的工具/图表之后
                            current_answer_box = st.empty()
                            current_segment = ""
                        current_segment += text_chunk
                        full_answer += text_chunk
                        # 渲染带光标的"正在打字"效果
                        current_answer_box.markdown(current_segment + CURSOR)

                    # ── 最终事件：记录元数据；不再二次重写正文以免覆盖流式段 ──
                    elif etype == "final":
                        final_meta["tool_count"] = event.get("tool_count", 0)
                        final_meta["total_tool_time"] = event.get("total_tool_time", 0.0)
                        # 若中途完全没收到 answer_delta（罕见），用 final.content 补一次
                        final_content = event.get("content") or ""
                        if final_content and not full_answer:
                            current_answer_box = st.empty()
                            current_segment = final_content
                            full_answer = final_content

                    elif etype == "error":
                        stream_error = event.get("message", "未知错误")

            except Exception as e:  # noqa: BLE001
                stream_error = str(e)
                if not full_answer:
                    full_answer = f"抱歉，分析过程中出现错误：{e}"
                    if current_answer_box is None:
                        current_answer_box = st.empty()
                    current_segment = full_answer

            # ── 关闭最后一段（去掉打字机光标） ── #
            _finalize_current_segment()
            # 思考过程的光标也去掉
            if thinking_box is not None:
                thinking_box.markdown(thinking_text)

            # ── 末尾元数据 caption ── #
            answer_text = full_answer  # 持久化使用累计文本
            if stream_error and not answer_text:
                st.error(f"调用 FinanceAgent 失败：{stream_error}")
            elif answer_text and final_meta["tool_count"]:
                st.caption(
                    f"本次推理调用 {final_meta['tool_count']} 个工具 · "
                    f"工具耗时 {final_meta['total_tool_time']:.2f}s"
                )

            # 保存回复（仅保存文本到历史）
            if answer_text:
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                # 第一次提问后，若标题仍是默认的「TICKER · MM-DD HH:MM」，用首问内容作为标题
                first_user_msgs = [m for m in st.session_state.messages if m.get("role") == "user"]
                if len(first_user_msgs) == 1:
                    default_prefix = (active_conv.ticker.upper() + " · ") if active_conv.ticker else ""
                    if active_conv.title.startswith(default_prefix):
                        snippet = str(first_user_msgs[0].get("content") or "").strip().replace("\n", " ")
                        if snippet:
                            new_title = snippet[:24] + ("…" if len(snippet) > 24 else "")
                            conv_store.rename(active_conv.id, new_title)
                _persist_active_conversation()

    # 处理预设卡片的输入
    if preset_input := st.session_state.pop("user_input", None):
        handle_user_query(preset_input)

    # 聊天输入框
    if prompt := st.chat_input(f"向 FinanceAgent 提问关于 {resolved_ticker_display} 的任何问题…"):
        handle_user_query(prompt)