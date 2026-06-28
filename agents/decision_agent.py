"""金融决策智能体 (Financial Agent)
=================================

本模块负责把 **DeepSeek LLM**（左脑：意图路由 + 函数调用 + 报告生成）
与 **本地量化工具**（右脑：T+1 预测、支撑阻力、相似形态、宏观、新闻情感、SEC 财报）
组装成一个完整的、可流式可视化的金融分析智能体。

模块结构（重构后）
------------------
1. :class:`_PromptBuilder`
   - 仅负责构建 system prompt + 单轮上下文，逻辑可独立复用与测试。
2. :class:`_ToolExecutor`
   - 工具路由 / 入参过滤 / 数据获取（实时 vs 本地）/ 结构化错误返回。
3. :class:`FinancialAgent`
   - 唯一对外暴露的「编排器」。保持原有 :meth:`run` / :meth:`run_stream` API 不变，
     使前端无需任何改动即可继续工作。

公开接口（向后兼容）
--------------------
- :meth:`FinancialAgent.run`            同步模式，返回最终文本
- :meth:`FinancialAgent.run_stream`     流式模式，逐步 yield 事件字典
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

# ---- 配置：失败则提供本地降级 ---- #
try:
    from src.config.settings import deepseek_settings
except ImportError:
    from pathlib import Path

    _current_file = Path(__file__).resolve()
    _project_root = _current_file.parent.parent

    class _SimpleSettings:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    deepseek_settings = _SimpleSettings(  # type: ignore[assignment]
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        # 升级到 DeepSeek V4 Pro（deepseek-chat 别名将于 2026-07-24 停用）
        deepseek_model="deepseek-v4-pro",
        deepseek_timeout=60,
    )

# ---- 工具注册表 & 数据提供层 ---- #
from agents.Tools.tool_registry import TOOL_MAPPING, tools
from agents.data_provider import DataProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

try:
    from openai import OpenAI

    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    logger.warning("openai SDK 未安装。请安装：pip install openai")


# 接口版本号：与前端缓存键约定，签名/字段变更时递增即可让旧实例自动失效
AGENT_INTERFACE_VERSION = "2026-06-28.v4"

# 工具调用上限（防止 LLM 死循环）
_MAX_TOOL_ROUNDS = 10


# --------------------------------------------------------------------------- #
# 1. PromptBuilder
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """

# 角色与使命 (Role & Mission)

你是一位顶级的量化投资策略师，为专业的对冲基金经理提供决策支持。你的核心使命是：**融合宏观环境、微观数据和历史规律，形成逻辑严密、数据驱动的、结构化的投资分析报告。** 你必须保持绝对的客观和中立，只基于数据和量化模型进行推理。



# 思考框架 (Thinking Framework) - 严格遵循此SOP

当你收到用户请求时，必须按照以下顺序进行思考和分析：



1.  **宏观优先 (Macro First):** 在进行任何个股分析之前，**必须**首先调用 `analyze_market_regime` 工具评估当前市场的整体风险状况。这是所有决策的基石。

2.  **【微调】问题定性与路径选择 (Query Qualification & Path Selection):**

    -   **识别用户意图：** 用户的核心问题是关于**短期交易**还是**长期投资**？

    -   **选择分析路径：**

        -   **A) 短期/战术分析路径:** 如果问题涉及"T+1走势"、"支撑阻力"、"K线形态"、"动量"，则优先使用 `predict`, `calculate_levels`, `search_similar_periods` 等量化工具。

        -   **B) 长期/基本面分析路径:** 如果问题涉及"公司价值"、"财报健康度"、"是否值得长持"、"基本面如何"，则**必须**以 `analyze_financial_statements` 工具为核心进行分析。

        -   **C) 全面综合分析路径:** 如果问题是"全面分析一下这只股票"或"给一份完整的投资报告"，则**必须同时执行A和B两条路径**，以形成最完整的视图。

3.  **工具协同 (Tool Synergy):** 根据选择的分析路径，规划并调用所有相关的工具来收集数据拼图。

4.  **数据综合 (Synthesize Data):** 将所有工具返回的JSON数据视为原始情报。你的任务不是简单复述数据，而是要**解读数据、建立关联、发现矛盾或共识**。

5.  **报告生成 (Generate Report):** 严格按照下方指定的「报告结构」输出你的最终分析，确保每一部分都有数据支撑。



# 报告结构 (Report Structure) - 你的最终输出必须遵循此格式

**【核心观点】**

*在此处用一句话总结你最核心的结论，例如："看涨，但需警惕宏观风险"或"中性偏空，等待关键支撑位确认"。*



**一、宏观环境分析**

- **市场风险等级：** [输出 `analyze_market_regime` 的 `risk_regime` 结果]

- **关键驱动因素：** [解读 `gspc`, `vix`, `tnx` 的数据，例如：VIX处于低位，标普500指数位于200日均线之上，市场情绪稳定。]



**二、多维度量化分析**

- **T+1走势预测:**

  - **预测方向:** [上涨/下跌] (基于 `direction_prob`)

  - **预测价格:** [输出 `predicted_price`]

  - **预测变动:** [输出 `predicted_change_pct`]

  - **模型解读:** [简要说明上涨概率和波动幅度的含义]

- **关键支撑与阻力:**

  - **最近支撑位:** [输出 `nearest_support` 的价格和强度]

  - **最近阻力位:** [输出 `nearest_resistance` 的价格和强度]

  - **当前状态:** [解读 `status`，例如：价格正处于强支撑位上方，短期下行空间有限。]

- **历史形态相似度:**

  - **相似片段数量:** [找到的相似片段数量]

  - **历史回测参考:** [总结所有相似片段的 `subsequent_return` 的平均值和正负概率，例如：历史上5次相似形态出现后，未来5日的平均涨幅为+2.1%，其中4次上涨，1次下跌。]

- **新闻情感分析:**

  - **近期情感倾向:** [正面/中性/负面] (基于 `calc_score` 的结果)

  - **关键新闻解读:** [如果可能，提及影响情感的关键新闻点]


** 三、长期基本面健康度评估**
*（当调用 `analyze_financial_statements` 工具时，此部分为必填项）*
- **综合财务评级：** [输出 `long_term_value_rating` 结果，例如：Excellent (优秀)]
- **盈利能力 (利润表):**
  - **核心指标:** [引用`revenue`, `net_income`等原始数据]
  - **分析师观点:** [引用并适当润色 `income_statement_analysis` 中的 `analyst_take`]
- **财务状况 (资产负债表):**
  - **核心指标:** [引用`total_assets`, `total_liabilities`, `current_ratio`等数据]
  - **分析师观点:** [引用并适当润色 `balance_sheet_analysis` 中的 `analyst_take`]
- **现金流 (现金流量表):**
  - **核心指标:** [引用`operating_cash_flow`, `free_cash_flow`等原始数据]
  - **分析师观点:** [引用并适当润色 `cash_flow_analysis` 中的 `analyst_take`]

**四、综合研判与风险提示**

- **综合结论:** [在这里，将**短期技术分析(第二部分)**与**长期基本面评估(第三部分)**进行最终的**交叉验证**。它们是相互印证，还是彼此矛盾？基于此形成一个完整的、多视角的投资逻辑。]
- **潜在风险:**
  - [明确列出1-2个最主要的风险点。例如：短期技术指标超买、财报显示杠杆率过高、宏观市场转为Risk-Off等。]



# 可用的快捷工具 (Quick Tools)

- `get_technical_snapshot`: 快速获取某股票当前 RSI/MACD/SMA/布林带读数及大白话解读，适合快速技术面体检。
- `fetch_recent_news`: 获取最近新闻并用 FinBERT 打分，适合判断近期市场情绪。



# 行为边界 (Behavioral Boundaries)

- **禁止提供直接投资建议:** 你的角色是分析师，不是投资顾问。严禁使用"你应该买入"、"建议加仓"等指令性语言。应使用"数据显示出看涨信号"、"价格接近关键支撑"等客观描述。

- **数据驱动:** 你的每一句结论都必须有前面工具返回的数据作为依据。

- **工具错误处理:** 若某个工具返回形如 `{"status":"error","reason":...}` 的结果，说明该项数据暂不可用。你应明确指出该维度数据缺失，并**基于其余可用数据继续分析**，绝不可编造缺失的数据。

- **高效沟通:** 如果用户只是闲聊或问候，直接友好回答，**不要**调用任何工具。

"""


class _PromptBuilder:
    """构建发送给 LLM 的 messages 列表。"""

    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build(
        self,
        user_prompt: str,
        chat_history: List[Dict[str, Any]],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str,
        current_date: str,
    ) -> List[Dict[str, Any]]:
        context = (
            f"股票: {ticker}\n"
            f"现价: {current_price}\n"
            f"当前日期: {current_date}（调用 analyze_market_regime 时必须使用此日期，禁止自行推断或捏造日期）\n"
            f"数据量: {len(stock_data) if stock_data is not None else 0}条\n"
            f"问题: {user_prompt}"
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(_sanitize_history(chat_history))
        messages.append({"role": "user", "content": context})
        return messages


def _sanitize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤掉前端展示用、不属于 LLM 上下文的消息字段。

    - 仅保留 role / content / tool_calls / tool_call_id / name 等 OpenAI 规范字段
    - 丢弃 None content（除非是 assistant 工具调用消息）
    """
    cleaned: List[Dict[str, Any]] = []
    allowed_keys = {"role", "content", "tool_calls", "tool_call_id", "name"}
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        item = {k: v for k, v in msg.items() if k in allowed_keys}
        if item.get("content") is None and "tool_calls" not in item:
            continue
        cleaned.append(item)
    return cleaned


# --------------------------------------------------------------------------- #
# 2. ToolExecutor
# --------------------------------------------------------------------------- #
class _ToolExecutor:
    """工具调用路由器：根据 LLM 返回的 function_name 找到对应工具并执行。

    所有 **业务工具的实际逻辑都没有改变**，本类只负责：
      - 入参清洗 / 过滤
      - 数据来源决策（实时 vs 本地）
      - 异常 → 结构化 `{"status":"error","reason":...}`
      - 计时
    """

    def __init__(self, data_provider: DataProvider, realtime: bool = False):
        self.data_provider = data_provider
        self.realtime = realtime
        self._feature_builder: Any = None  # 延迟加载

    # ---------- 公共入口 ---------- #
    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str,
        current_date: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float]:
        start = time.time()
        try:
            logger.info("执行工具: %s, 参数: %s", name, args)
            target = (args.get("ticker") or ticker or "").upper().strip()
            result = self._dispatch(name, args, stock_data, current_price, ticker, current_date, target)
            return result, time.time() - start
        except Exception as exc:  # noqa: BLE001
            logger.error("工具执行出错 (%s): %s", name, exc, exc_info=True)
            return _err(str(exc)), time.time() - start

    # ---------- 路由 ---------- #
    def _dispatch(
        self,
        name: str,
        args: Dict[str, Any],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str,
        current_date: Optional[str],
        target: str,
    ) -> Dict[str, Any]:

        # --- T+1 预测 --- #
        if name in ("get_stock_prediction", "predict"):
            df = self._resolve_feature_df(stock_data, target, ticker)
            if df is None or df.empty:
                return _err(
                    f"无法获取 {target} 的特征数据（实时/本地均失败，或缺少 pandas_ta 依赖）"
                )
            price = _price_from_df(df, current_price, target, ticker)
            tool = TOOL_MAPPING["predict"]()
            return tool.predict(simulation_data=df, current_price=price)

        # --- 支撑/阻力位 --- #
        if name in ("get_support_resistance", "calculate_levels"):
            df = self._resolve_ohlcv(stock_data, target, ticker)
            if df is None or df.empty:
                return _err(f"无法获取 {target} 的行情数据")
            price = _price_from_df(df, current_price, target, ticker)
            allowed = {"window", "use_weighted", "bandwidth"}
            filtered = {k: v for k, v in args.items() if k in allowed}
            tool = TOOL_MAPPING["calculate_levels"]()
            return tool.calculate_levels(df=df, current_price=price, **filtered)

        # --- 历史相似形态 --- #
        if name in ("find_similar_patterns", "search_similar_periods"):
            df = self._resolve_ohlcv(stock_data, target, ticker)
            if df is None or df.empty:
                return _err(f"无法获取 {target} 的行情数据")
            similarity_method = args.get("similarity_method", "euclidean")
            if similarity_method not in ("euclidean", "cosine"):
                similarity_method = "euclidean"
            tool = TOOL_MAPPING["search_similar_periods"](similarity_method=similarity_method)
            allowed = {"query_window", "top_k", "subsequent_days"}
            filtered = {k: v for k, v in args.items() if k in allowed}
            matches = tool.search_similar_periods(df=df, **filtered)
            return {"matches": matches}

        # --- 新闻情感（旧接口兼容） --- #
        if name in ("get_news_and_sentiment", "calc_score"):
            tool = TOOL_MAPPING["calc_score"]()
            if hasattr(tool, "get_news_and_sentiment"):
                df = self._resolve_ohlcv(stock_data, target, ticker)
                return tool.get_news_and_sentiment(df=df, end_date=df["Date"].max())
            texts = args.get("texts", [])
            return {"avg_sentiment": tool.calc_score(texts=texts), "count": len(texts)}

        if name == "classify_texts":
            tool = TOOL_MAPPING["classify_texts"]()
            return {"classifications": tool.classify_texts(texts=args.get("texts", []))}

        # --- 宏观环境 --- #
        if name == "analyze_market_regime":
            # 强制使用应用层传入的真实日期，忽略 LLM 自行推断/捏造
            regime_date = current_date or args.get("current_date") or pd.Timestamp.today().strftime("%Y-%m-%d")
            macro_df = self.data_provider.get_macro() if self.realtime else None
            tool = TOOL_MAPPING["analyze_market_regime"]()
            return tool.analyze_market_regime(current_date=regime_date, df=macro_df)

        # --- SEC 财报 --- #
        if name == "analyze_financial_statements":
            tool = TOOL_MAPPING["analyze_financial_statements"]()
            return tool.analyze_latest_filings(target)

        # --- 技术指标快照 --- #
        if name == "get_technical_snapshot":
            df = self._resolve_ohlcv(stock_data, target, ticker)
            if df is None or df.empty:
                return _err(f"无法获取 {target} 的行情数据")
            tool = TOOL_MAPPING["get_technical_snapshot"]()
            return tool.get_technical_snapshot(df=df, ticker=target)

        # --- 实时新闻情感 --- #
        if name == "fetch_recent_news":
            limit = int(args.get("limit", 8))
            tool = TOOL_MAPPING["fetch_recent_news"](provider=self.data_provider)
            return tool.fetch_recent_news(ticker=target, limit=limit)

        return _err(f"未知工具调用: {name}")

    # ---------- 数据来源辅助 ---------- #
    def _get_feature_builder(self) -> Any:
        if self._feature_builder is None:
            try:
                from agents.realtime_features import RealtimeFeatureBuilder

                self._feature_builder = RealtimeFeatureBuilder(provider=self.data_provider)
            except Exception as exc:  # noqa: BLE001
                logger.warning("实时特征构建器不可用: %s", exc)
                self._feature_builder = False
        return self._feature_builder or None

    def _resolve_ohlcv(self, stock_data: pd.DataFrame, target: str, default_ticker: str) -> pd.DataFrame:
        if (
            stock_data is not None
            and not stock_data.empty
            and target == (default_ticker or "").upper().strip()
        ):
            return stock_data
        return self.data_provider.get_ohlcv(target)

    def _resolve_feature_df(
        self, stock_data: pd.DataFrame, target: str, default_ticker: str
    ) -> Optional[pd.DataFrame]:
        if (
            stock_data is not None
            and not stock_data.empty
            and target == (default_ticker or "").upper().strip()
        ):
            return stock_data
        builder = self._get_feature_builder()
        if builder is None:
            return None
        built = builder.build_features(target)
        if built.get("error"):
            logger.warning("为 %s 现算特征失败: %s", target, built["error"])
        return built.get("data")


# --------------------------------------------------------------------------- #
# 静态工具函数
# --------------------------------------------------------------------------- #
def _err(reason: str) -> Dict[str, Any]:
    """统一的结构化错误返回，便于 LLM 理解并降级处理。"""
    return {"status": "error", "reason": reason}


def _price_from_df(df: pd.DataFrame, current_price: Any, target: str, default_ticker: str) -> Any:
    if target == (default_ticker or "").upper().strip() and current_price:
        return current_price
    if df is not None and not df.empty and "Close" in df.columns:
        return float(df["Close"].iloc[-1])
    return current_price


# --------------------------------------------------------------------------- #
# 3. FinancialAgent —— 编排器（对外唯一入口）
# --------------------------------------------------------------------------- #
class FinancialAgent:
    """金融决策智能体。

    用法（向后兼容）::

        agent = FinancialAgent(api_key=..., realtime=True, data_provider=...)
        for event in agent.run_stream(prompt, history, df, price, ticker, current_date=date):
            ...
    """

    INTERFACE_VERSION = AGENT_INTERFACE_VERSION

    def __init__(
        self,
        api_key: Optional[str] = None,
        realtime: bool = False,
        data_provider: Optional[DataProvider] = None,
    ):
        if not OPENAI_SDK_AVAILABLE:
            raise ValueError("OpenAI SDK 未安装")

        self.api_key = api_key or deepseek_settings.deepseek_api_key
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未设置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=deepseek_settings.deepseek_base_url,
            timeout=deepseek_settings.deepseek_timeout,
        )
        self.model = deepseek_settings.deepseek_model
        self.tools = tools
        self.tool_mapping = TOOL_MAPPING  # 兼容旧代码直接访问
        self.realtime = realtime
        self.data_provider = data_provider or DataProvider()

        # 子组件
        self._prompt_builder = _PromptBuilder()
        self._executor = _ToolExecutor(self.data_provider, realtime=self.realtime)

    # ----- 让外部修改 realtime 时同步到 executor ----- #
    @property
    def realtime(self) -> bool:  # type: ignore[override]
        return self._realtime

    @realtime.setter
    def realtime(self, value: bool) -> None:
        self._realtime = bool(value)
        # 当 executor 尚未初始化时（__init__ 内首次赋值），跳过同步
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.realtime = self._realtime

    # ========================================================== #
    # 同步入口（向后兼容）
    # ========================================================== #
    def run(
        self,
        user_prompt: str,
        chat_history: List[Dict[str, Any]],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str,
        current_date: Optional[str] = None,
    ) -> str:
        """同步模式：消费事件流并返回最终文本回答。"""
        final_parts: List[str] = []
        for event in self.run_stream(
            user_prompt,
            chat_history,
            stock_data,
            current_price,
            ticker,
            current_date=current_date,
        ):
            etype = event.get("type")
            if etype == "final":
                return event.get("content", "")
            if etype == "answer_delta":
                final_parts.append(event.get("text", ""))
            elif etype == "error":
                return f"抱歉，系统遇到错误: {event.get('message', '未知错误')}"
        return "".join(final_parts) or "（未生成回答）"

    # ========================================================== #
    # 流式入口
    # ========================================================== #
    def run_stream(
        self,
        user_prompt: str,
        chat_history: List[Dict[str, Any]],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str,
        current_date: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """流式执行智能体推理，逐步产出结构化事件。

        事件类型：
        - ``thinking_delta`` ``{text}``                 模型思考过程增量
        - ``answer_delta``   ``{text}``                 最终回答增量
        - ``tool_call``      ``{name, args}``           即将执行某工具
        - ``tool_result``    ``{name, result, elapsed}``工具执行完成
        - ``final``          ``{content, tool_count, total_tool_time}``
        - ``error``          ``{message}``
        """
        tool_count = 0
        total_tool_time = 0.0
        _current_date = current_date or pd.Timestamp.today().strftime("%Y-%m-%d")

        try:
            messages = self._prompt_builder.build(
                user_prompt, chat_history, stock_data, current_price, ticker, _current_date
            )

            for _round in range(_MAX_TOOL_ROUNDS):
                full_content, ordered_tool_calls = yield from self._stream_one_round(messages)

                # 本轮无工具调用 -> 最终回答
                if not ordered_tool_calls:
                    yield {
                        "type": "final",
                        "content": full_content,
                        "tool_count": tool_count,
                        "total_tool_time": round(total_tool_time, 2),
                    }
                    return

                # 把 assistant tool_calls 消息追加回历史
                assistant_tool_calls = [
                    {
                        "id": slot["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
                    }
                    for i, slot in enumerate(ordered_tool_calls)
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                # 逐个执行工具并产出事件
                logger.info("检测到 %d 个工具调用请求...", len(assistant_tool_calls))
                for call in assistant_tool_calls:
                    name = call["function"]["name"]
                    try:
                        parsed_args = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        parsed_args = {}

                    yield {"type": "tool_call", "name": name, "args": parsed_args}

                    result_obj, elapsed = self._executor.execute(
                        name,
                        parsed_args,
                        stock_data,
                        current_price,
                        ticker,
                        _current_date,
                    )
                    tool_count += 1
                    total_tool_time += elapsed

                    yield {
                        "type": "tool_result",
                        "name": name,
                        "result": result_obj,
                        "elapsed": round(elapsed, 2),
                    }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result_obj, ensure_ascii=False, default=str),
                        }
                    )

            yield {
                "type": "final",
                "content": "分析过程过长，已强制终止。请尝试简化您的问题。",
                "tool_count": tool_count,
                "total_tool_time": round(total_tool_time, 2),
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Agent 运行错误: %s", exc, exc_info=True)
            yield {"type": "error", "message": str(exc)}

    # ---------------------------------------------------------- #
    # 单轮流式：消费一次 chat.completions stream，
    # yield thinking/answer 增量；返回 (完整文本, ordered_tool_calls)
    # ---------------------------------------------------------- #
    def _stream_one_round(
        self, messages: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:  # type: ignore[override]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.3,  # 金融分析需结论可复现，低温度
            max_tokens=8192,
            stream=True,
        )

        content_parts: List[str] = []
        tool_calls_accum: Dict[int, Dict[str, str]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 思考过程（DeepSeek V4 thinking）
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "thinking_delta", "text": reasoning}

            # 正文
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield {"type": "answer_delta", "text": delta.content}

            # 工具调用（跨 chunk 累积）
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tool_calls_accum.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["args"] += tc.function.arguments

        full_content = "".join(content_parts)
        ordered = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
        # 把 (full_content, ordered) 作为 generator 的 return value
        return full_content, ordered


__all__ = ["FinancialAgent", "AGENT_INTERFACE_VERSION"]
