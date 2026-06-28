# 06 · Streamlit 前端

> [`frontend/app.py`](../../frontend/app.py) 整体设计：UI 分区、缓存策略、降级模式、图表组件。

> **TL;DR**：单文件 ~1052 行；侧栏配置 + 主体三栏布局（K线/技术指标/情感曲线 + 仪表盘/新闻 + AI 对话）；无 API Key 时进入"规则引擎模式"。

---

## 1. 启动入口

```bash
# 标准启动
streamlit run app.py
# 等价于
streamlit run frontend/app.py
```

[`app.py`](../../app.py) 与 `frontend/app.py` 内容一致（仓库根的副本）。

---

## 2. 全局样式 (`setup_page_style`)

通过 `st.markdown(..., unsafe_allow_html=True)` 注入 CSS：

| 选择器 | 效果 |
|---|---|
| `.stApp` | 浅灰背景 (`#f8f9fa`) + Inter 字体 |
| `#MainMenu / footer / header` | 隐藏 Streamlit 默认 UI |
| `.css-card` | 圆角白底卡片 + 阴影 |
| `[data-testid="stSidebar"]` | 固定 260-300px 宽，可见性强制 |
| `[data-testid="collapsedControl"]` | 隐藏折叠按钮（强制展开） |
| `.stButton>button` | 蓝色背景 50px 高，圆角 |
| `.analysis-container / .risk-container / .verdict-container` | 自定义 Markdown 排版 |

---

## 3. 配置回退

`frontend/app.py` 顶部尝试导入正式配置，失败时使用 `SimpleSettings` 占位：

```python
try:
    from src.config.settings import finbert_settings, deepseek_settings
except ImportError:
    class SimpleSettings: ...

    finbert_settings = SimpleSettings(
        embedding_model_name="yiyanghkust/finbert-tone",
        embedding_max_length=128,
        model_name="yiyanghkust/finbert-tone",
        max_length=256,
        device=None,
        model_cache_dir=_PROJECT_ROOT / "models",
    )
    deepseek_settings = SimpleSettings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout=60,
    )

    # 注入到 sys.modules，让 agents 模块也能导入
    config_module = types.ModuleType('src.config.settings')
    config_module.finbert_settings = finbert_settings
    config_module.deepseek_settings = deepseek_settings
    sys.modules['src.config.settings'] = config_module
```

这样 **`agents/decision_agent.py` 与 `agents/Tools/finbert_analyzer.py` 都能找到 `deepseek_settings`**，不必引入 `src/` 目录。

---

## 4. 数据加载层

### 4.1 常量与缓存

```python
@st.cache_data
def _get_data_date_range():
    df = pd.read_csv(_STOCK_DATA_PATH, usecols=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df['Date'].min().date(), df['Date'].max().date()

KAGGLE_START_DATE, KAGGLE_END_DATE = _get_data_date_range()

@st.cache_data
def _get_supported_tickers():
    # 读 stock_statistics.csv 与实际数据交叉验证
    ...

@st.cache_data
def load_stock_data_cached(ticker, start_date, end_date) -> pd.DataFrame:
    return load_stock_data_dynamic(ticker, start_date, end_date)
```

`@st.cache_data` 让同一查询不重复读 CSV。

### 4.2 动态加载

```python
def load_stock_data_dynamic(ticker, start_date, end_date) -> pd.DataFrame:
    df = pd.read_csv(_STOCK_DATA_PATH, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    mask = (df['Ticker'] == ticker.upper()) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    return df[mask].sort_values('Date').reset_index(drop=True)
```

### 4.3 新闻提取

```python
def extract_news_before_date(ticker, target_date, days_before=5, max_news_per_day=3) -> List[Dict]:
    """从 stock_datas.csv 抽取指定日期前 N 天的新闻（按 | 分割多标题）"""
```

### 4.4 情感聚合

```python
def calculate_aggregated_sentiment(data, window=7, use_market_adjustment=True) -> pd.Series:
    """
    1. EMA 平滑 Sentiment_Score (span=window)
    2. 可选：叠加 GSPC_LogReturn 的 tanh 调整
    3. clip 到 [-1, 1]
    """
```

实现"个股情感曲线 + 市场背景调整"，使曲线更平滑、更有解释力。

---

## 5. 侧边栏 (Controller)

```
┌────────────────────────────┐
│ ⚡ QuantAgent              │
├────────────────────────────┤
│ 🤖 AI 智能体配置            │
│   DeepSeek API Key [•••]   │
│   ✓ API Key 已配置         │
│   (or) 💡 未配置 → 降级模式  │
├────────────────────────────┤
│ 📊 交易参数                │
│   股票代码 [EBAY]          │
│   匹配建议 (5 项)          │
│   当前模型: 混合模型        │
├────────────────────────────┤
│ 📅 日期范围选择            │
│   [2009-01-01] ~ [2019-12-31]│
├────────────────────────────┤
│ 🛠️ 技术指标参数            │
│   [RSI] [MACD] [BBANDS]  │
│   [VOLUME] [CANDLE] [EMA]  │
├────────────────────────────┤
│ [🚀 开始分析]              │
└────────────────────────────┘
```

### 5.1 关键交互

- **`st.text_input` 解析 ticker**：`resolve_ticker()` 支持中英文名/简写。
- **`search_tickers` 联想**：每输入一个字符就列出最多 5 个匹配。
- **API Key 校验**：`startswith("sk-") and len > 20` 才显示"已配置"。
- **日期范围** 用两个 `st.date_input`，默认拉到数据文件的真实起止。
- **指标开关**：布尔值多选，控制图表显示。

---

## 6. 主体三栏布局

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 股价 + 情感双轴图 (Plotly 双 Y 轴)                            │
│  · 主轴：Close 收盘价（蓝色线）                                    │
│  · 副轴：Sentiment_Score 聚合曲线（橙色线）                        │
├─────────────────────────────────────────────────────────────────┤
│  📈 技术指标三联图 (Plotly subplots)                               │
│  · RSI (14)                                                       │
│  · MACD (12,26,9) + Signal + Histogram                            │
│  · SMA 5 / SMA 20                                                 │
├──────────────────────────────┬──────────────────────────────────┤
│ 📊 关键指标仪表盘            │ 📰 最近 N 天新闻 + 情感           │
│  · 现价 / 预测价 / RSI       │  · 标题、日期、情感 emoji         │
│  · 支撑 / 阻力 / 风险等级     │  · Positive / Neutral / Negative │
│  · 宏观状态                  │                                  │
├──────────────────────────────┴──────────────────────────────────┤
│  💬 AI 对话区                                                     │
│  · 快捷指令按钮（4 个）                                           │
│  · 聊天记录 (st.chat_message)                                    │
│  · 输入框 (st.chat_input)                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1 Plotly 图表

**股价+情感双轴图**（核心可视化）：

```python
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['Date'], y=sentiment, name='Sentiment',
                         line=dict(color='orange')), secondary_y=True)
fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
```

**技术指标子图**：

```python
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.4, 0.3, 0.3],
                    subplot_titles=('RSI', 'MACD', 'SMA'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14']), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_12_26_9']), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['MACDs_12_26_9']), row=2, col=1)
fig.add_bar(x=df['Date'], y=df['MACDh_12_26_9']), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5']), row=3, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20']), row=3, col=1)
```

### 6.2 关键指标仪表盘

显示：
- 现价、预测价（来自 `HybridPredictor`）、变化百分比
- 风险等级（来自 `MacroAnalyzer`）
- 最近支撑 / 阻力（来自 `SupportResistanceScanner`）
- 上涨概率（`direction_prob`）

### 6.3 新闻列表

```python
for news in extract_news_before_date(ticker, today, days_before=5):
    sentiment_label = "😊" if sentiment > 0.1 else ("😐" if sentiment > -0.1 else "😟")
    st.markdown(f"- {sentiment_label} **{news['Date']}** · {news['Title']}")
```

---

## 7. AI 对话区

### 7.1 快捷指令

```python
quick_actions = [
    "📊 短期技术分析",  # 强制走 predict / calculate_levels / similarity
    "🔍 历史形态回测",
    "📜 财报深度解读",  # 强制走 analyze_financial_statements
    "🎯 综合投资报告",  # 走 A + B 路径
]
```

点击后注入对应 prompt 到 `st.session_state.messages` 并触发分析。

### 7.2 消息流

```python
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("正在分析..."):
            agent = FinancialAgent(api_key=api_key)
            response = agent.run(
                user_prompt=prompt,
                chat_history=st.session_state.messages,
                stock_data=df,
                current_price=latest_close,
                ticker=ticker,
            )
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 7.3 降级模式处理

```python
if not deepseek_api_key:
    st.warning("⚠️ 当前未配置 DeepSeek API Key，AI 对话功能暂不可用。请在左侧侧栏填写 API Key 后重试。")
    st.info("💡 提示：所有量化功能（图表、预测、支撑阻力、情感分析）仍可正常使用。")
    return  # 阻断后续
```

---

## 8. 缓存策略汇总

| 资源 | 装饰器 | 失效条件 |
|---|---|---|
| `_get_data_date_range` | `@st.cache_data` | 进程重启 |
| `_get_supported_tickers` | `@st.cache_data` | 进程重启 |
| `load_stock_data_cached` | `@st.cache_data` | `(ticker, start, end)` 变化 |
| `get_tech_indicators_cached` | `@st.cache_data` | 输入 df 哈希变化 |
| `load_hybrid_predictor_cached` | `@st.cache_resource` | 进程生命周期内单例 |

> `@st.cache_resource` 用于 HybridPredictor 这种**重资源对象**（避免每次刷新都重新加载 ~50MB 模型）。

---

## 9. 容错与 UX

- **数据加载失败** → `st.error("数据加载失败: ...")` + 返回空 DataFrame
- **技术指标计算失败** → `st.error(...)` + 折叠面板 `st.expander("🔍 查看详细错误信息")` 显示 traceback
- **模型文件缺失** → `st.error("Hybrid 模型文件未找到: ...")` + 引导运行训练脚本
- **KDE 失败** → 工具返回空 + 仪表盘显示"未找到支撑位"
- **SEC 请求失败** → 工具返回 `error` 字段 + LLM 自适应给出提示
- **API Key 无效** → DeepSeek 抛 401 → `st.error("API Key 验证失败")` 引导重输

---

## 10. 完整渲染流水线

```
[页面加载]
    │
    ▼
setup_page_style()        ← 注入 CSS
    │
    ▼
cache: _get_data_date_range, _get_supported_tickers
    │
    ▼
sidebar input widgets     ← ticker / dates / API key / indicators
    │
    ▼
[用户点击"开始分析"]
    │
    ▼
cache: load_stock_data_cached
    │
    ▼
load_hybrid_predictor_cached  ← 单例 HybridPredictor
    │
    ▼
render_price_sentiment_chart    ← 双 Y 轴 Plotly
render_tech_indicators          ← 3 行 subplot
render_metrics_dashboard        ← 现价/预测/风险/支撑
render_news_list                ← 5 天内新闻
    │
    ▼
[用户输入对话 or 点击快捷卡片]
    │
    ▼
FinancialAgent.run()    ← 同步阻塞 + st.spinner
    │
    ▼
st.markdown(报告)        ← 渲染到对话气泡
```

---

## 11. 已知限制与改进

| 限制 | 改进方向 |
|---|---|
| 整图 `st.rerun` 重绘所有组件 | 局部刷新 / `st.fragment` |
| 大数据量 Plotly 卡顿 | 抽样显示 + 局部缩放 |
| FinBERT 加载慢 | 启动时后台预热 |
| SEC 网络不稳 | 加前端 spinner + retry |
| 移动端布局不友好 | `st.columns` + 媒体查询 |
| 多用户并发状态混乱 | 改用 session_id 隔离 |
