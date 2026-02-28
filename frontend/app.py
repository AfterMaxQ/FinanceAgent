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
        deepseek_model="deepseek-chat",
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
from agents.decision_agent import FinancialAgent

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
# 1. 页面配置与 CSS 极简美化 (UI Polish)
# ==========================================
def setup_page_style() -> None:
    """Configure Streamlit page and inject global CSS."""
    st.set_page_config(
        page_title="QuantAgent",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        /* 全局背景与字体 */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Inter', sans-serif;
        }
        
        /* 隐藏 Streamlit 默认菜单和 Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 卡片样式 */
        .css-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* 标题样式 */
        h1, h2, h3 {
            color: #1e293b;
            font-weight: 600;
        }
        
        /* 侧边栏美化 */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            min-width: 260px;
            max-width: 300px;
            visibility: visible !important;
            transform: translateX(0) !important;
        }
        /* 强制展开侧栏并移除折叠控件 */
        [data-testid="stSidebar"] > div:first-child {
            visibility: visible !important;
            display: block !important;
            transform: translateX(0) !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        
        /* 按钮美化 */
        .stButton>button {
            width: 100%;
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            height: 50px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
        }
        
        /* 列间距优化 */
        [data-testid="column"] {
            padding-left: 0.25rem;
            padding-right: 0.25rem;
        }
        
        /* 状态指示器美化 */
        [data-testid="stStatusWidget"] {
            border: 1px solid #e2e8f0;
            background-color: white;
        }
        
        /* Markdown 容器样式优化 */
        .analysis-container,
        .risk-container,
        .verdict-container {
            font-size: 14px;
        }
        
        .analysis-container p,
        .risk-container p,
        .verdict-container p {
            margin-bottom: 8px;
        }
        
        .analysis-container ul,
        .risk-container ul,
        .verdict-container ul {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        
        .analysis-container li,
        .risk-container li,
        .verdict-container li {
            margin-bottom: 6px;
        }
        
        .analysis-container strong,
        .risk-container strong,
        .verdict-container strong {
            color: #1e293b;
            font-weight: 600;
        }
        
        /* 加载动画样式 */
        [data-testid="stSpinner"] {
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)


setup_page_style()

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
    st.markdown("### ⚡ QuantAgent")
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    st.markdown("**🤖 AI 智能体配置**")
    deepseek_api_key = st.text_input(
        "DeepSeek API Key", type="password", help="输入您的 DeepSeek API 密钥以启用 AI 决策分析（可选）",
        placeholder="sk-xxx...（留空将使用降级模式）"
    )
    
    if deepseek_api_key:
        if deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
            st.success("🔗 API Key 已配置")
            # 检查 FinancialAgent 是否已初始化
            agent_key = f"financial_agent_{hash(deepseek_api_key)}"
            if agent_key in st.session_state:
                st.success("✅ 金融智能体已就绪")
        else:
            st.warning("⚠️ API Key 格式可能有误")
    else:
        st.info("💡 未配置 API Key，将使用规则引擎模式")
    
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    st.markdown("**📊 交易参数**")
    
    supported_tickers_help = ", ".join(SUPPORTED_TICKERS)
    # 使用实际存在于数据中的股票作为默认值（EBAY 在数据中存在）
    default_ticker = "EBAY" if "EBAY" in SUPPORTED_TICKERS else (SUPPORTED_TICKERS[0] if SUPPORTED_TICKERS else "MS")
    ticker = st.text_input("股票代码", default_ticker, help=f"输入股票代号或中文名称（支持：{supported_tickers_help}）", placeholder="如 EBAY 或 eBay")

    suggestions = search_tickers(ticker)
    st.caption("匹配建议（点击按钮前先确认代号是否正确）")
    for item in suggestions:
        st.markdown(f"- `{item['ticker']}` · {item['name']}")

    model_type = "混合模型" # 原Hybrid（多任务融合）
    st.info(f"当前模型：{model_type}")
    
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    st.markdown("**📅 日期范围选择**")
    st.caption(f"支持范围：{KAGGLE_START_DATE.strftime('%Y-%m-%d')} 至 {KAGGLE_END_DATE.strftime('%Y-%m-%d')}")
    
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = max(KAGGLE_START_DATE, KAGGLE_END_DATE - timedelta(days=365))
        start_date = st.date_input("开始日期", value=default_start, min_value=KAGGLE_START_DATE, max_value=KAGGLE_END_DATE)
    with col_end:
        end_date = st.date_input("结束日期", value=KAGGLE_END_DATE, min_value=KAGGLE_START_DATE, max_value=KAGGLE_END_DATE)
    
    if start_date >= end_date:
        st.error("⚠️ 开始日期必须早于结束日期！")
        st.stop()
    
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    st.markdown("**🕐 模拟时间点**")
    st.caption("选择要模拟分析的具体日期（在该日期范围内）")
    simulation_date = st.date_input("模拟日期", value=end_date, min_value=start_date, max_value=end_date)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    run_btn = st.button("启动挖掘与预测")
    if run_btn:
        st.session_state.run_analysis = True
    
    st.markdown("---")
    st.markdown("**🔧 系统状态**")
    st.caption(f"• 前端服务：**在线**")
    st.caption(f"• FinBERT 模型：**已加载**")
    st.caption(f"• 混合预测模型：**就绪**") 
    if deepseek_api_key and deepseek_api_key.strip():
        if deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
            st.caption("• DeepSeek API：**已配置** ✅")
        else:
            st.caption("• DeepSeek API：**配置异常** ⚠️")
    else:
        st.caption("• DeepSeek API：**降级模式** 💡")

# ==========================================
# 4. 主界面逻辑 (Main Interface)
# ==========================================
try:
    resolved_ticker_display = resolve_ticker(ticker) if ticker else "EBAY"
except:
    resolved_ticker_display = ticker if ticker else "EBAY"

st.markdown(f"## QuantAgent：**{resolved_ticker_display}**")
st.caption(f"📅 数据范围：{KAGGLE_START_DATE.strftime('%Y-%m-%d')} 至 {KAGGLE_END_DATE.strftime('%Y-%m-%d')}（Kaggle数据集）")

run_analysis = st.session_state.get("run_analysis", False)

if not run_analysis:
    st.info("👈 请在左侧确认参数后点击 **启动挖掘与预测** 开始流程。")
else:
    try:
        resolved_ticker = resolve_ticker(ticker)
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
        macro_result = macro_analyzer.analyze_market_regime(current_date=simulation_date.strftime('%Y-%m-%d'))
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

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # --- 阶段 2: 数据挖掘成果可视化 (Dual Axis Chart) ---
    col_chart, col_news = st.columns([2, 1])
    
    with col_chart:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 多模态特征关联")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='股价', fill='tozeroy', line=dict(color='rgba(37, 99, 235, 0.8)', width=2), fillcolor='rgba(37, 99, 235, 0.1)'))
        if 'Sentiment_Score' in data.columns:
            colors = ['#ef4444' if v < 0 else '#22c55e' for v in data['Sentiment_Score']]
            fig.add_trace(go.Bar(x=data['Date'], y=data['Sentiment_Score'], name='新闻情感（FinBERT）', yaxis='y2', marker_color=colors, opacity=0.6))
        
        if aggregated_sentiment_available and 'Aggregated_Sentiment' in data.columns:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Aggregated_Sentiment'], name='市场情绪指数（聚合）', yaxis='y2', line=dict(color='#8b5cf6', width=2, dash='dash'), opacity=0.8))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=350, yaxis=dict(title="价格 ($)", showgrid=True, gridcolor='#f1f5f9'), yaxis2=dict(title="情感得分", overlaying='y', side='right', range=[-1.5, 1.5], showgrid=False), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if 'RSI_14' in data.columns and 'MACD_12_26_9' in data.columns:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 技术指标分析")
            
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
            st.plotly_chart(fig_indicators, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_news:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 📰 结构化情报")
        st.caption(f"模拟日期 ({simulation_date.strftime('%Y-%m-%d')}) 前5天的新闻情感分析（显示5条）")
        
        if not news:
            st.info("📭 模拟日期前5天内暂无新闻数据")
        else:
            def select_balanced_news(news_list: List[Dict], max_count: int = 5) -> List[Dict]:
                if len(news_list) <= max_count: return news_list
                positive = sorted([n for n in news_list if n['Sentiment'] > 0], key=lambda x: abs(x['Sentiment']), reverse=True)
                negative = sorted([n for n in news_list if n['Sentiment'] < 0], key=lambda x: abs(x['Sentiment']), reverse=True)
                selected = positive[:(max_count // 2 + max_count % 2)] + negative[:max_count // 2]
                if len(selected) < max_count:
                    remaining = [n for n in news_list if n not in selected]
                    selected.extend(sorted(remaining, key=lambda x: abs(x['Sentiment']), reverse=True)[:max_count - len(selected)])
                return sorted(selected, key=lambda x: x['Date'], reverse=True)

            for item in select_balanced_news(news, max_count=5):
                sentiment_color = "#ef4444" if item['Sentiment'] < 0 else "#22c55e"
                st.markdown(f"""
                <div style="border-left: 3px solid {sentiment_color}; padding-left: 10px; margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #64748b;">{item['Date']} | {item['Source']}</div>
                    <div style="font-weight: 500; font-size: 14px;">{item['Title']}</div>
                    <div style="font-size: 12px; font-family: monospace; color: {sentiment_color};">情感分数: {item['Sentiment']:.3f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 关键指标仪表盘 ---
    st.markdown("---")
    st.markdown("### 📊 关键指标仪表盘")
    st.caption(f"模拟日期: {simulation_date.strftime('%Y-%m-%d')} | 股票代码: {resolved_ticker_display}")
    
    # 创建5列布局
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 指标1: 当前股价
    with col1:
        if current_price is not None:
            delta_value = price_change_str if len(simulation_data) > 1 else "无前日数据"
            delta_color = "off"
            if len(simulation_data) > 1:
                delta_color = "inverse" if price_change_pct < 0 else "normal"
            st.metric(
                label="当前股价",
                value=f"${current_price:.2f}",
                delta=delta_value,
                delta_color=delta_color,
                help=f"模拟日期 {simulation_date.strftime('%Y-%m-%d')} 的收盘价。"
            )
        else:
            st.metric(label="当前股价", value="N/A", delta="数据缺失")
    
    # 指标2: T+1 走势预测
    with col2:
        if predicted_price is not None and prediction_error is None:
            delta_value = f"{predicted_change_pct:+.2f}%"
            delta_color = "inverse" if predicted_change_pct < 0 else "normal"
            st.metric(
                label="T+1 预测",
                value=f"${predicted_price:.2f}",
                delta=delta_value,
                delta_color=delta_color,
                help="基于混合模型的T+1收盘价预测。"
            )
        else:
            error_msg = prediction_error if prediction_error else "计算失败"
            st.metric(label="T+1 预测", value="N/A", delta=error_msg)
    
    # 指标3: 宏观市场风险
    with col3:
        if risk_regime != "N/A":
            st.metric(
                label="市场风险",
                value=risk_regime,
                delta=risk_rationale,
                help="基于VIX, GSPC, TNX的综合宏观环境评估。"
            )
        else:
            st.metric(label="市场风险", value="N/A", delta=risk_rationale)
    
    # 指标4: 关键价位
    with col4:
        if support_price is not None or resistance_price is not None:
            support_str = f"{support_price:.2f}" if support_price is not None else "N/A"
            resistance_str = f"{resistance_price:.2f}" if resistance_price is not None else "N/A"
            value_str = f"{support_str} | {resistance_str}"
            st.metric(
                label="支撑 | 阻力",
                value=value_str,
                delta=support_resistance_status,
                help="基于KDE算法计算的历史成交密集区。"
            )
        else:
            st.metric(label="支撑 | 阻力", value="N/A | N/A", delta=support_resistance_status)
    
    # 指标5: RSI
    with col5:
        if rsi_value is not None and pd.notna(rsi_value):
            # 根据RSI值设置delta颜色：超买(>70)用红色，超卖(<30)用绿色，中性用灰色
            delta_color = "inverse" if rsi_value > 70 else "normal" if rsi_value < 30 else "off"
            st.metric(
                label="RSI (14)",
                value=f"{rsi_value:.2f}",
                delta=rsi_status,
                delta_color=delta_color,
                help="相对强弱指数，用于判断超买超卖状态。"
            )
        else:
            st.metric(label="RSI (14)", value="N/A", delta=rsi_status)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # --- 阶段 3: 交互式聊天界面 ---
    st.markdown("---")
    st.markdown("### 💬 金融分析聊天机器人")
    
    # 初始化聊天历史（如果不存在或股票代码变化）
    chat_key = f"chat_messages_{resolved_ticker}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": f"👋 您好！我是金融分析智能体，可以帮您分析 **{resolved_ticker_display}** 的投资机会。\n\n我可以：\n- 📈 预测股票 T+1 走势\n- 📊 分析支撑位和阻力位\n- 🔍 搜索相似的历史行情\n- 💭 分析新闻情感\n\n请选择下方的分析卡片，或直接向我提问！"
            }
        ]
    
    # 使用当前股票的聊天历史
    if "messages" not in st.session_state or st.session_state.get("current_ticker") != resolved_ticker:
        st.session_state.messages = st.session_state[chat_key].copy()
        st.session_state.current_ticker = resolved_ticker
    
    # 初始化 FinancialAgent（在应用启动时，仅在 API Key 存在时）
    agent_initialized = False
    agent = None
    if deepseek_api_key and deepseek_api_key.strip() and deepseek_api_key.startswith("sk-") and len(deepseek_api_key) > 20:
        try:
            agent_key = f"financial_agent_{hash(deepseek_api_key)}"
            if agent_key not in st.session_state:
                with st.spinner("正在初始化金融智能体..."):
                    st.session_state[agent_key] = FinancialAgent(api_key=deepseek_api_key.strip())
            agent = st.session_state[agent_key]
            agent_initialized = True
        except Exception as e:
            st.warning(f"⚠️ 金融智能体初始化失败: {e}")
            agent_initialized = False
            agent = None
    else:
        agent_initialized = False
        agent = None
    
    # 示例分析卡片
    st.markdown("#### 📋 快速分析")
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    
    # --- 新的交互逻辑：使用 on_click 回调 ---
    preset_questions = {
        "card1": f"请对 {resolved_ticker_display} 进行一次核心的短期技术分析，重点关注其 T+1 走势预测和关键的支撑阻力位。",
        "card2": f"请深入挖掘 {resolved_ticker_display} 的历史数据，搜索与最近20天走势最相似的K线形态，并回测这些形态出现后的市场表现。",
        "card3": f"请对 {resolved_ticker_display} 进行一次深度基本面分析，调用SEC财报工具，评估其最新的财务健康度和长期投资价值。",
        "card4": f"请为 {resolved_ticker_display} 生成一份顶级的、综合性的投资决策报告。报告必须兼顾短期量化技术信号与长期基本面价值，并进行交叉验证，最后明确指出潜在的核心风险。"
    }
    
    def set_user_input(question_key):
        st.session_state.user_input = preset_questions[question_key]

    # 使用更紧凑的列布局
    card_col1, card_col2, card_col3, card_col4 = st.columns(4, gap="small")
    
    with card_col1:
        st.button("📊 短期技术分析", help=preset_questions["card1"], on_click=set_user_input, args=("card1",), key="card1_btn", use_container_width=True)
    
    with card_col2:
        st.button("🔍 历史形态回测", help=preset_questions["card2"], on_click=set_user_input, args=("card2",), key="card2_btn", use_container_width=True)
    
    with card_col3:
        st.button("📜 财报健康度分析", help=preset_questions["card3"], on_click=set_user_input, args=("card3",), key="card3_btn", use_container_width=True)
    
    with card_col4:
        st.button("🎯 综合投资决策", help=preset_questions["card4"], on_click=set_user_input, args=("card4",), key="card4_btn", use_container_width=True)

    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_query(user_input: str):
        """统一处理来自输入框或预设卡片的问题。"""
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state[chat_key] = st.session_state.messages.copy()
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. 调用 Agent 并显示回复
        with st.chat_message("assistant"):
            if not agent_initialized or agent is None:
                st.warning("请先配置 API Key")
            else:
                with st.spinner("🤖 正在思考与分析..."):
                    # 构建干净的历史记录（去掉欢迎语）
                    clean_history = [
                        msg for msg in st.session_state.messages[:-1] 
                        if "我是金融分析智能体" not in msg.get("content", "")
                    ]
                    
                    # 同步调用 Agent
                    response = agent.run(
                        user_prompt=user_input,
                        chat_history=clean_history,
                        stock_data=simulation_data,
                        current_price=current_price,
                        ticker=resolved_ticker
                    )
                    
                    st.markdown(response)
                    
                    # 保存回复
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state[chat_key] = st.session_state.messages.copy()

    # 处理预设卡片的输入
    if preset_input := st.session_state.pop("user_input", None):
        handle_user_query(preset_input)

    # 聊天输入框
    if prompt := st.chat_input("向金融智能体提问..."):
        handle_user_query(prompt)