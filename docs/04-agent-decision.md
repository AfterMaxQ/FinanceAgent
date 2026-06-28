# 04 · AI 决策智能体 (FinancialAgent)

> DeepSeek V3 Function Calling + 严格 SOP 提示词 + 工具路由 + 多轮对话。
> 实现"宏观先行 → 问题定性 → 工具协同 → 数据综合 → 报告生成"五步法。

---

## 1. 模块入口

[`agents/decision_agent.py`](../../agents/decision_agent.py) 导出 `FinancialAgent` 类，主要成员：

| 成员 | 说明 |
|---|---|
| `__init__(api_key)` | 创建 `openai.OpenAI` 客户端（指向 DeepSeek base_url） |
| `run(user_prompt, chat_history, stock_data, current_price, ticker)` | 同步主循环入口 |
| `_build_messages(...)` | 组装 system + history + 当前 query |
| `_execute_tool(tool_call, ...)` | 按 function name 路由到具体类 |

### 1.1 依赖与导入

```python
from openai import OpenAI
from agents.Tools.tool_registry import tools, TOOL_MAPPING
```

- `tools` — 7 个工具的 DeepSeek function schema
- `TOOL_MAPPING` — 名称 → Python 类的字典

### 1.2 配置回退

```python
try:
    from src.config.settings import deepseek_settings
except ImportError:
    class SimpleSettings: ...
    deepseek_settings = SimpleSettings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout=60,
    )
```

前端会注入一个 `SimpleSettings` 占位（见 [`frontend/app.py:18-58`](../../frontend/app.py)），避免生产环境对 `src.config` 的硬依赖。

---

## 2. 主循环 `run()`

```python
def run(self, user_prompt, chat_history, stock_data, current_price, ticker) -> str:
    try:
        messages = self._build_messages(user_prompt, chat_history, stock_data, current_price, ticker)

        for _ in range(10):                              # 最多 10 轮
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self.tools, tool_choice="auto",
                temperature=1.0, max_tokens=8192,
            )
            message = response.choices[0].message

            # 把助手消息写回历史
            if hasattr(message, 'model_dump'):
                messages.append(message.model_dump(exclude_unset=True))
            else:
                msg_dict = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls
                messages.append(msg_dict)

            # 没有 tool_calls → 模型已给出最终回答
            if not message.tool_calls:
                return message.content

            # 执行每个工具
            for tool_call in message.tool_calls:
                result_str = self._execute_tool(tool_call, stock_data, current_price, ticker)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        return "分析过程过长，已强制终止。请尝试简化您的问题。"

    except Exception as e:
        logger.error(f"Agent 运行错误: {e}", exc_info=True)
        return f"抱歉，系统遇到错误: {str(e)}"
```

### 2.1 关键设计

- **同步/非流式**：每轮拿到完整响应再决定下一步，比 streaming 更易处理多轮 tool_calls。
- **10 轮硬上限**：防止因工具定义不清或数据缺失导致死循环。
- **`model_dump(exclude_unset=True)`**：兼容新旧 OpenAI SDK，避免丢字段。
- **异常熔断**：任何 API 异常都会被 `try/except` 捕获并返回中文提示，前端不会崩溃。

### 2.2 消息数组结构

```
[
  {"role": "system", "content": <SOP 提示词>},
  {"role": "user", "content": <历史问题 1>},
  {"role": "assistant", "content": <历史回答 1>},
  ...
  {"role": "user", "content": <本轮 context: 股票/现价/数据量/问题>},
  {"role": "assistant", "content": ..., "tool_calls": [...]},   # 轮 1
  {"role": "tool", "tool_call_id": "...", "content": "<JSON 1>"},
  {"role": "tool", "tool_call_id": "...", "content": "<JSON 2>"},
  ...
  {"role": "assistant", "content": <最终 Markdown 报告>}      # 终止
]
```

---

## 3. SOP 系统提示词（`_build_messages`）

> 系统提示词约 110 行中文，详细定义角色、5 步思考框架、报告结构、行为边界。

### 3.1 思考框架（SOP）

```
1. 宏观优先 (Macro First)
   必须先调 analyze_market_regime，再做个股分析
2. 问题定性与路径选择 (Query Qualification)
   ├─ A) 短期 → predict / calculate_levels / search_similar_periods
   ├─ B) 长期 → analyze_financial_statements
   └─ C) 综合 → 同时走 A + B
3. 工具协同 (Tool Synergy)
   规划并调用所有相关工具
4. 数据综合 (Synthesize Data)
   解读/关联/发现矛盾或共识
5. 报告生成 (Generate Report)
   严格按 4 段格式输出
```

### 3.2 报告结构

```markdown
**【核心观点】**
（一句话总结）

**一、宏观环境分析**
- 市场风险等级
- 关键驱动因素

**二、多维度量化分析**
- T+1 走势预测（方向/价格/变动/解读）
- 关键支撑与阻力
- 历史形态相似度
- 新闻情感分析

**三、长期基本面健康度评估**（仅当调用了财报工具）
- 综合财务评级
- 盈利能力
- 财务状况
- 现金流

**四、综合研判与风险提示**
- 综合结论（短期 vs 长期交叉验证）
- 潜在风险
```

### 3.3 行为边界

- **禁止直接投资建议**：用"数据显示出看涨信号"代替"建议买入"。
- **数据驱动**：每句话都要有工具返回数据支撑。
- **闲聊不调工具**：避免误消耗 DeepSeek tokens。

### 3.4 上下文注入

```python
context = (
    f"股票: {ticker}\n"
    f"现价: {current_price}\n"
    f"数据量: {len(stock_data)}条\n"
    f"问题: {user_prompt}"
)
```

把当前对话的"基本事实"放在 user 消息里，避免模型凭空假设。

---

## 4. 工具路由 `_execute_tool()`

### 4.1 路由表

| 工具名（LLM 调用） | 实际入口 | 参数白名单 |
|---|---|---|
| `predict` / `get_stock_prediction` | `tool.predict(simulation_data, current_price)` | — |
| `calculate_levels` / `get_support_resistance` | `tool.calculate_levels(df, current_price, window, use_weighted, bandwidth)` | `window`, `use_weighted`, `bandwidth` |
| `search_similar_periods` / `find_similar_patterns` | `tool.search_similar_periods(df, query_window, top_k, subsequent_days)` | `query_window`, `top_k`, `subsequent_days` |
| `calc_score` / `get_news_and_sentiment` | `tool.calc_score(texts)` 或 `get_news_and_sentiment` | — |
| `classify_texts` | `tool.classify_texts(texts)` | — |
| `analyze_market_regime` | `tool.analyze_market_regime(current_date)` | — |
| `analyze_financial_statements` | `tool.analyze_latest_filings(ticker)` | — |

### 4.2 路由实现要点

```python
def _execute_tool(self, tool_call, stock_data, current_price, ticker):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    tool_cls = self.tool_mapping[name]
    tool = tool_cls()                       # 每次都新建实例

    if name == "predict" or name == "get_stock_prediction":
        res = tool.predict(simulation_data=stock_data, current_price=current_price)
    elif name == "calculate_levels" or name == "get_support_resistance":
        allowed = {"window", "use_weighted", "bandwidth"}
        filtered_args = {k: v for k, v in args.items() if k in allowed}
        res = tool.calculate_levels(df=stock_data, current_price=current_price, **filtered_args)
    ...
    return json.dumps(res, ensure_ascii=False, default=str)
```

**安全策略**：

- **白名单过滤**：每个工具接收的参数都被限定到 `allowed` 集合，LLM 注入多余字段不会传到工具。
- **JSON 序列化兜底**：`default=str` 处理不可序列化的对象（如 numpy、datetime、Path）。
- **同义函数名兼容**：保留 `get_stock_prediction`、`get_support_resistance` 等旧名，向后兼容旧 prompt。

### 4.3 异常隔离

```python
try:
    ...
    return json.dumps(res, ensure_ascii=False, default=str)
except Exception as e:
    return f"工具执行出错: {str(e)}"
```

任何工具的异常被转成字符串返回到 messages，模型会看到错误并自行决定下一步（重试 / 改用其他工具 / 给用户解释）。

---

## 5. 工具注册表 `tool_registry.py`

文件：[`agents/Tools/tool_registry.py`](../../agents/Tools/tool_registry.py)

### 5.1 数据结构

```python
tools: List[Dict[str, Any]] = [...]   # 7 个 DeepSeek function schema
TOOL_MAPPING: Dict[str, Type] = {...}  # 名称 → 类
TOOL_INIT_PARAMS: Dict[str, Dict] = {} # 特殊构造参数（当前仅 SimilaritySearcher）
```

### 5.2 工具定义示例

```python
{
    "type": "function",
    "function": {
        "name": "predict",
        "description": "使用 Hybrid 多任务模型预测股票 T+1 价格...",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "..."},
                "news_column": {"type": "string", "default": "NewsTitles"}
            },
            "required": ["ticker"]
        }
    }
}
```

> DeepSeek 的 function calling schema 与 OpenAI 完全兼容，因此可以直接复用 openai SDK。

### 5.3 辅助 API

| 函数 | 用途 |
|---|---|
| `get_tool_by_name(name)` | 取出 schema（用于向 LLM 展示） |
| `get_tool_class(name)` | 取出 Python 类（拼写错误时用 `difflib` 提示相似名） |
| `get_all_tool_names()` | 列出所有工具名 |

---

## 6. 配置容错

### 6.1 缺失 API Key

```python
self.api_key = api_key or deepseek_settings.deepseek_api_key
if not self.api_key:
    raise ValueError("DeepSeek API Key 未设置")
```

`FinancialAgent` 构造会失败，但前端检测到 Key 为空时**不创建** agent，转入"规则引擎模式"。

### 6.2 降级模式（前端）

[`frontend/app.py`](../../frontend/app.py) 在 sidebar 实时判断：

```
if deepseek_api_key:
    st.success("🔗 API Key 已配置")
else:
    st.info("💡 未配置 API Key，将使用规则引擎模式")
```

降级模式下：
- 图表、预测、支撑阻力、形态匹配、情感分析 全部可用
- AI 对话区域显示"请先配置 API Key"或直接禁用

---

## 7. 完整调用链

```
User 输入问题
   │
   ▼
Streamlit handle_user_query(prompt)
   │
   ▼
FinancialAgent(api_key=user_key)
   │
   ▼
agent.run(
  user_prompt=prompt,
  chat_history=st.session_state.messages,
  stock_data=df_filtered,
  current_price=latest_close,
  ticker=ticker
)
   │
   ├─ _build_messages(...)
   │
   └─ for i in 0..9:
        DeepSeek.chat.completions.create(messages, tools, tool_choice=auto)
            │
            ▼
        response.message
            │
            ├─ (no tool_calls) → return message.content
            │
            └─ (with tool_calls) → for each:
                  _execute_tool(tool_call, df, price, ticker)
                      │
                      ├─ tool = TOOL_MAPPING[name]()
                      ├─ 过滤 args 白名单
                      └─ 执行具体计算，返回 JSON 字符串
                  messages.append({"role": "tool", ...})
   │
   ▼
返回 Markdown 报告 → st.markdown(...)
```

---

## 8. 提示词调优建议

| 场景 | 调整 |
|---|---|
| LLM 频繁漏调 `analyze_market_regime` | 在 system prompt 中加 "**强制**第一调用 macro" |
| 输出过长 | `max_tokens=4096` 或要求"每段 100 字以内" |
| 输出过简 | 提高 `temperature` 至 1.2，或加 "请详尽分析" |
| 工具调用错误 | 在 description 中增加参数约束示例 |
| 多轮上下文混乱 | 在 `_build_messages` 中裁剪 `chat_history` 至最近 6 轮 |

---

## 9. 工具完整清单（按 `tool_registry.py` 顺序）

| # | name | class | 调用频率预期 |
|---|---|---|---|
| 1 | `calc_score` | `FinBERTAnalyzer` | 每次短期/综合分析 |
| 2 | `classify_texts` | `FinBERTAnalyzer` | 用户显式要求"分类"时 |
| 3 | `predict` | `HybridPredictor` | 每次短期/综合分析 |
| 4 | `calculate_levels` | `SupportResistanceScanner` | 每次短期/综合分析 |
| 5 | `search_similar_periods` | `SimilaritySearcher` | 每次短期/综合分析 |
| 6 | `analyze_market_regime` | `MacroAnalyzer` | **每轮必须** |
| 7 | `analyze_financial_statements` | `FinancialStatementAnalyzer` | 长期/综合分析 |

---

## 10. 安全与边界

- **不持久化数据**：agent 只在内存中处理股票切片，不写盘。
- **API Key 保护**：仅在本次会话的 `st.session_state` 中保留，不会回传到任何日志。
- **输入校验**：`_execute_tool` 通过 `try/except` + 白名单防止恶意 schema 注入。
- **限流友好**：每个工具内部已实现容错（KDE 失败返回空、SEC 失败返回错误对象），不会因单点故障中断 LLM 决策。
