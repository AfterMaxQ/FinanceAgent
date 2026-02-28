import json
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

# 尝试导入配置
try:
    from src.config.settings import deepseek_settings
except ImportError:
    from pathlib import Path
    _current_file = Path(__file__).resolve()
    _project_root = _current_file.parent.parent
    
    class SimpleSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    deepseek_settings = SimpleSettings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout=60
    )

# 导入工具注册表
from agents.Tools.tool_registry import tools, TOOL_MAPPING

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    logger.warning("openai SDK 未安装。请安装：pip install openai")


class FinancialAgent:
    """
    金融决策智能体 (同步/非流式版本)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_SDK_AVAILABLE:
            raise ValueError("OpenAI SDK 未安装")
        
        self.api_key = api_key or deepseek_settings.deepseek_api_key
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未设置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=deepseek_settings.deepseek_base_url,
            timeout=deepseek_settings.deepseek_timeout
        )
        self.model = deepseek_settings.deepseek_model
        self.tools = tools
        self.tool_mapping = TOOL_MAPPING


    def run(
        self,
        user_prompt: str,
        chat_history: List[Dict[str, str]],
        stock_data: pd.DataFrame,
        current_price: float,
        ticker: str
    ) -> str:
        """
        执行智能体推理（同步模式）。
        会自动处理多轮工具调用，直到模型生成最终回答。
        """
        try:
            # 1. 构建初始消息
            messages = self._build_messages(user_prompt, chat_history, stock_data, current_price, ticker)
            
            # 2. 循环处理工具调用（最多 10 轮，防止死循环）
            for _ in range(10):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=1.0, # 调高温度，生成更具发散性的回答
                    max_tokens=8192 # 设置最大token数，支持更长的分析报告
                )
                
                message = response.choices[0].message
                
                # 将助手的消息加入历史
                # 兼容旧版和新版 OpenAI SDK
                if hasattr(message, 'model_dump'):
                    messages.append(message.model_dump(exclude_unset=True))
                else:
                    msg_dict = {"role": "assistant", "content": message.content}
                    if message.tool_calls:
                        msg_dict["tool_calls"] = message.tool_calls
                    messages.append(msg_dict)

                # 如果没有工具调用，说明已经得到最终结果，直接返回
                if not message.tool_calls:
                    return message.content

                # 3. 执行工具
                logger.info(f"检测到 {len(message.tool_calls)} 个工具调用请求...")
                for tool_call in message.tool_calls:
                    result_str = self._execute_tool(tool_call, stock_data, current_price, ticker)
                    
                    # 将工具结果加入消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str
                    })
            
            return "分析过程过长，已强制终止。请尝试简化您的问题。"


        except Exception as e:
            logger.error(f"Agent 运行错误: {e}", exc_info=True)
            return f"抱歉，系统遇到错误: {str(e)}"


    def _build_messages(self, user_prompt, chat_history, stock_data, current_price, ticker):
        """构建提示词"""
        system_content = """

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



# 行为边界 (Behavioral Boundaries)

- **禁止提供直接投资建议:** 你的角色是分析师，不是投资顾问。严禁使用"你应该买入"、"建议加仓"等指令性语言。应使用"数据显示出看涨信号"、"价格接近关键支撑"等客观描述。

- **数据驱动:** 你的每一句结论都必须有前面工具返回的数据作为依据。

- **高效沟通:** 如果用户只是闲聊或问候，直接友好回答，**不要**调用任何工具。

"""
        
        context = (
            f"股票: {ticker}\n"
            f"现价: {current_price}\n"
            f"数据量: {len(stock_data)}条\n"
            f"问题: {user_prompt}"
        )
        
        messages = [{"role": "system", "content": system_content}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": context})
        return messages


    def _execute_tool(self, tool_call, stock_data, current_price, ticker):
        """执行工具"""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            logger.info(f"执行工具: {name}, 参数: {args}")
            
            if name not in self.tool_mapping:
                return f"错误: 工具 {name} 不存在"
            
            tool_cls = self.tool_mapping[name]
            tool = tool_cls()
            
            # 根据不同工具分发
            if name == "get_stock_prediction" or name == "predict":
                res = tool.predict(simulation_data=stock_data, current_price=current_price)
            elif name == "get_support_resistance" or name == "calculate_levels":
                allowed = {"window", "use_weighted", "bandwidth"}
                filtered_args = {k: v for k, v in args.items() if k in allowed}
                res = tool.calculate_levels(
                    df=stock_data,
                    current_price=current_price,
                    **filtered_args
                )
            elif name == "find_similar_patterns" or name == "search_similar_periods":
                similarity_method = args.pop("similarity_method", "euclidean")
                if similarity_method not in ("euclidean", "cosine"):
                    logger.warning(
                        f"收到不支持的 similarity_method={similarity_method}，已回退为 euclidean"
                    )
                    similarity_method = "euclidean"
                tool = tool_cls(similarity_method=similarity_method)
                allowed = {"query_window", "top_k", "subsequent_days"}
                filtered_args = {k: v for k, v in args.items() if k in allowed}
                res = tool.search_similar_periods(
                    df=stock_data,
                    **filtered_args
                )
            elif name == "get_news_and_sentiment" or name == "calc_score":
                # 兼容旧版和新版工具名
                if hasattr(tool, 'get_news_and_sentiment'):
                    res = tool.get_news_and_sentiment(df=stock_data, end_date=stock_data['Date'].max(), **args)
                else:
                    texts = args.get('texts', [])
                    res = tool.calc_score(texts=texts)
            elif name == "classify_texts":
                res = tool.classify_texts(texts=args.get('texts', []))
            elif name == "analyze_market_regime":
                current_date = args.get("current_date")
                res = tool.analyze_market_regime(current_date=current_date)
            elif name == "analyze_financial_statements":
                # SEC财报分析工具
                ticker_arg = args.get("ticker", ticker)  # 优先使用参数中的ticker，否则使用传入的ticker
                res = tool.analyze_latest_filings(ticker_arg)
            else:
                return f"未知工具调用: {name}"
                
            return json.dumps(res, ensure_ascii=False, default=str)
            
        except Exception as e:
            return f"工具执行出错: {str(e)}"