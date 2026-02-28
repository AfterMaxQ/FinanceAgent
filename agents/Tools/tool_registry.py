"""工具注册表：集中管理 Agent 工具的 API 规范与实现映射。

本模块职责：
1. 按照 DeepSeek API 规范暴露可调用的工具清单（tools）。
2. 将工具函数名映射到对应的 Python 类，用于运行时动态实例化和调用。
3. 提供便捷的查询方法（获取工具定义、类、名称列表），便于上层编排与调度。
"""

from typing import Any, Dict, List, Type
from difflib import get_close_matches

from agents.Tools.finbert_analyzer import (
    FinBERTAnalyzer,
)
from agents.Tools.predict_t1 import HybridPredictor
from agents.Tools.support_resistance import SupportResistanceScanner
from agents.Tools.similarity_search import SimilaritySearcher
from agents.Tools.macro_analyzer import MacroAnalyzer
# 新增：导入财报分析工具
from agents.Tools.financial_statement_analyzer import FinancialStatementAnalyzer

# DeepSeek API 工具定义列表
tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calc_score",
            "description": (
                "计算金融新闻文本列表的平均情感分数。"
                "使用 FinBERT 模型分析文本情感，返回 [-1, 1] 范围的情感分数，"
                "其中正数表示积极情感，负数表示消极情感。"
                "适用于分析单条或多条新闻的情感倾向。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "待分析的金融新闻文本列表。可以是单条或多条新闻。"
                    }
                },
                "required": ["texts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_texts",
            "description": (
                "对金融新闻文本列表进行三分类（正面/中性/负面）。"
                "返回每条文本的分类标签及三类概率，基于 finbert-tone 模型。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "待分类的金融新闻文本列表"
                    }
                },
                "required": ["texts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": (
                "使用 Hybrid 多任务模型预测股票 T+1 价格。"
                "结合 LSTM 时序特征和 Transformer 文本特征，"
                "同时输出方向概率（上涨概率）和波动幅度。"
                "需要提供股票代码（ticker）以获取历史数据和当前价格。"
                "返回预测价格、变化百分比、方向概率、波动幅度等信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "股票代码（如 'AAPL', '000001.SZ'），用于获取历史数据和当前价格"
                    },
                    "news_column": {
                        "type": "string",
                        "description": "DataFrame 中新闻标题的列名",
                        "default": "NewsTitles"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_levels",
            "description": (
                "使用 KDE (核密度估计) 算法识别股票历史价格中的关键支撑位和阻力位。"
                "通过分析价格分布的概率密度峰值，找出历史密集交易区（筹码峰），"
                "而非简单的最大/最小值。"
                "需要提供股票代码（ticker）以获取历史数据和当前价格。"
                "返回支撑位列表、阻力位列表、最近支撑位、最近阻力位和状态描述。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "股票代码（如 'AAPL', '000001.SZ'），用于获取历史数据和当前价格"
                    },
                    "window": {
                        "type": "integer",
                        "description": "回溯窗口大小（天数），默认 252 天（一年）",
                        "default": 252,
                        "minimum": 10
                    },
                    "use_weighted": {
                        "type": "boolean",
                        "description": "是否使用加权价格 (High+Low+Close)/3，否则仅使用 Close",
                        "default": True
                    },
                    "bandwidth": {
                        "type": ["number", "null"],
                        "description": "KDE 带宽参数，null 表示自动选择",
                        "default": None
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar_periods",
            "description": (
                "在历史数据中搜索与当前行情走势形状相似的历史片段（K线形态匹配）。"
                "使用滑动窗口和序列相似度算法，通过 Z-Score 标准化处理价格序列，"
                "只比较'形状'而非绝对价格水平，适用于不同价格区间的股票。"
                "需要提供股票代码（ticker）以获取历史数据，若上层已传入 DataFrame 则 ticker 可省略。"
                "返回最相似的 top_k 个历史片段，每个结果包含相似度得分和后续涨跌幅。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "股票代码（如 'AAPL', '000001.SZ'），用于获取历史数据"
                    },
                    "query_window": {
                        "type": "integer",
                        "description": "待匹配的当前 K 线长度（天数），默认 20 天",
                        "default": 20,
                        "minimum": 5
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相似的个数，默认 5",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "subsequent_days": {
                        "type": "integer",
                        "description": "统计后续涨跌幅的天数，默认 5 天",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 30
                    },
                    "similarity_method": {
                        "type": "string",
                        "description": "相似度计算方法",
                        "enum": ["euclidean", "cosine"],
                        "default": "euclidean"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market_regime",
            "description": (
                "分析宏观市场环境并返回量化指标（GSPC SMA20/50/200 及距离、"
                "VIX 水平与日变动、TNX 与 SMA50 趋势），用于在个股决策前评估系统性风险。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_date": {
                        "type": "string",
                        "description": "当前的模拟日期 (YYYY-MM-DD)"
                    }
                },
                "required": ["current_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_financial_statements",
            "description": (
                "获取并深入分析公司最新的年度财报(10-K)。"
                "此工具通过SEC官方API获取权威的原始财务数据，包括利润表、资产负债表和现金流量表的核心项目。"
                "它不仅返回原始数据，还计算关键财务比率，并从专业分析师视角评估公司的盈利能力、财务健康状况和现金创造能力，"
                "最终给出一个关于其长期运营价值的综合评级。适用于需要进行深度基本面分析或评估长期投资价值的场景。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "需要分析的股票代码 (例如 'AAPL', 'MSFT')"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# 工具函数名到 Python 类的映射
# 注意：对于需要实例化的类，映射到类本身；Agent 代码负责实例化和调用
TOOL_MAPPING: Dict[str, Type] = {
    "calc_score": FinBERTAnalyzer,
    "classify_texts": FinBERTAnalyzer,
    "predict": HybridPredictor,
    "calculate_levels": SupportResistanceScanner,
    "search_similar_periods": SimilaritySearcher,
    "analyze_market_regime": MacroAnalyzer,
    # 新增：财报分析工具的映射
    "analyze_financial_statements": FinancialStatementAnalyzer
}

# 工具初始化参数映射（用于需要特殊初始化参数的工具）
# 注意：这些参数需要在实例化工具类时传入 __init__ 方法
TOOL_INIT_PARAMS: Dict[str, Dict[str, Any]] = {
    "search_similar_periods": {
        # similarity_method 是 SimilaritySearcher.__init__ 的参数
        # 如果 LLM 调用时提供了 similarity_method，将用于初始化
        # 否则使用默认值 "euclidean"
    }
}

def get_tool_by_name(tool_name: str) -> Dict[str, Any]:
    """根据工具名称获取工具定义。
    
    Args:
        tool_name: 工具函数名称。
    
    Returns:
        工具定义字典。
    
    Raises:
        KeyError: 当工具不存在时。
    """
    for tool in tools:
        if tool["function"]["name"] == tool_name:
            return tool
    raise KeyError(f"工具 '{tool_name}' 不存在")

def get_tool_class(tool_name: str) -> Type:
    """根据工具名称获取对应的 Python 类。
    
    Args:
        tool_name: 工具函数名称。
    
    Returns:
        对应的 Python 类。
    
    Raises:
        KeyError: 当工具不存在时。
    """
    if tool_name not in TOOL_MAPPING:
        candidates = get_close_matches(tool_name, TOOL_MAPPING.keys(), n=1)
        suggestion = f" Do you mean '{candidates[0]}'?" if candidates else ""
        raise KeyError(f"工具 '{tool_name}' 不存在于 TOOL_MAPPING 中.{suggestion}")
    return TOOL_MAPPING[tool_name]

def get_all_tool_names() -> List[str]:
    """获取所有已注册的工具名称列表。
    
    Returns:
        工具名称列表。
    """
    return [tool["function"]["name"] for tool in tools]