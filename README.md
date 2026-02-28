```markdown
# ⚡ FinanceAgent — 多模态金融量化挖掘智能体


> **"像资深投资顾问一样思考，像量化交易员一样计算。"**

**FinanceAgent** 是一个结合 **生成式 AI (LLM)** 与 **深度学习量化模型** 的下一代金融决策支持系统。传统量化软件只有冰冷的数据，而通用的聊天机器人不懂复杂的金融计算。本项目采用 **"双脑协同"** 架构，将两者的优势融为一体：

| 🧠 左脑 — DeepSeek Agent | ⚙️ 右脑 — 量化工具链 |
|:---|:---|
| 理解用户意图与逻辑推理 | 高精度数学运算与模型推理 |
| 多轮对话与上下文管理 | 股价预测、因子计算、情感分析 |
| 生成通俗易懂的投资分析报告 | 支撑阻力位计算、形态相似度匹配 |
| SOP 路由：自动规划分析路径 | SEC 财报数据获取与财务比率分析 |

---

## 📑 目录

- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [数据处理流程](#-数据处理流程-pipeline)
- [启动应用](#-启动应用)
- [使用指南](#-使用指南)
- [工具箱详解](#-工具箱详解)
- [项目结构](#-项目结构)
- [技术细节](#-技术细节)
- [常见问题](#-常见问题)

---

## 🚀 核心功能

### 1. 📈 T+1 股价混合预测（Hybrid Prediction）

融合 **LSTM**（捕捉时序特征）和 **Transformer**（理解新闻语义）的多任务混合模型。不仅告诉你涨跌方向，还能预测 **明天的具体价格**、**波动幅度** 以及 **上涨概率**。

- 标量输入：18 维 VIF 筛选后的技术指标特征
- 文本输入：FinBERT 768 维新闻嵌入
- 双头输出：方向概率（Sigmoid）+ 波动幅度（回归）

### 2. 📰 金融舆情情感分析（Sentiment Analysis）

内置 **FinBERT（finbert-tone）** 金融领域预训练模型，对每日新闻标题进行三分类（Positive / Neutral / Negative），并计算情感分数（Score = P(positive) − P(negative)），量化消息面对股价的冲击。

### 3. 🎯 智能支撑 / 阻力位识别（Support & Resistance）

基于 **KDE（核密度估计）** 算法分析价格分布的概率密度峰值，科学识别历史筹码密集区。非主观画线，而是客观的统计方法，精准定位买卖关键价位。

### 4. 🔍 历史形态相似度匹配（Pattern Matching）

使用 **Z-Score 标准化 + 滑动窗口 + 欧氏距离/余弦相似度** 在长达数年的历史数据中，寻找与当前 K 线走势形状最相似的片段，并回测这些形态出现后的真实涨跌幅。

### 5. 🌍 宏观市场环境扫描（Macro Regime）

综合分析 **S&P 500（^GSPC）**、**VIX 恐慌指数（^VIX）**、**10年期美债收益率（^TNX）** 三大宏观指标，判断当前市场处于 `Risk-On`（风险偏好）、`Neutral`（中性）还是 `Risk-Off`（避险）状态。

### 6. 📜 SEC 财报深度分析（Fundamental Analysis）

通过 **SEC EDGAR XBRL API** 直接获取公司最新 10-K 年度报告的原始财务数据（利润表、资产负债表、现金流量表），计算毛利率、净利率、流动比率、自由现金流等核心指标，并给出定性评估。

### 7. 💬 交互式投资顾问

通过对话框直接提问（例如：*"分析一下 EBAY 的风险"*），AI 会按照预设的 **SOP 思考框架** 自动调度上述所有工具，生成一份有理有据、结构化的分析报告。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户与交互层 (Frontend)                       │
│  ┌──────────────┐    JSON / WebSocket    ┌──────────────────┐   │
│  │  👨‍💼 基金经理   │ ◄──────────────────► │ 💻 Streamlit 仪表盘│   │
│  └──────────────┘                        └────────┬─────────┘   │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
┌───────────────────────────────────────────────────┼─────────────┐
│                      AI 决策中枢 (Agent)           │             │
│  ┌────────────────────────────────────────────────▼──────────┐  │
│  │  🧠 FinancialAgent (decision_agent.py)                    │  │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐  │  │
│  │  │  SOP 路由 / 意图  │  │  🚀 DeepSeek V3 (LLM 推理)   │  │  │
│  │  │  识别 / 多轮对话  │◄►│  通用逻辑推理 & 报告生成      │  │  │
│  │  └──────────────────┘  └──────────────────────────────┘  │  │
│  └─────────┬────────────────────────────────────────────────┘  │
└────────────┼───────────────────────────────────────────────────┘
             │ Function Calling (tool_registry.py)
┌────────────┼───────────────────────────────────────────────────┐
│            ▼        专业分析工具箱 (Expert Tools)                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ 📈 T+1 预测   │ │ 📰 FinBERT   │ │ 🎯 支撑阻力   │           │
│  │ predict_t1   │ │ finbert      │ │ support_res  │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ 🔍 形态匹配   │ │ 🌍 宏观分析   │ │ 📜 SEC 财报   │           │
│  │ similarity   │ │ macro        │ │ financial    │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌────────────────────────────┼───────────────────────────────────┐
│                   数据基础设施 (Data)  │                         │
│  ┌──────────────┐ ┌───────▼──────┐ ┌──────────────┐           │
│  │ ⚙️ 离线工厂    │ │ 🛢️ CSV 数据库 │ │ 🏛️ SEC EDGAR │           │
│  │ 清洗/因子计算  │►│ 行情/因子/新闻 │ │  官方财报源   │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
└────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 快速开始

### 环境要求

- **Python** ≥ 3.10（推荐 3.13）
- **CUDA**（可选，GPU 加速推理）
- **网络**：需要访问 SEC EDGAR API（可选）和 DeepSeek API

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/FinanceAgent.git
cd FinanceAgent
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 核心依赖：

| 依赖           | 版本   | 用途                 |
| :------------- | :----- | :------------------- |
| `torch`        | 2.7.0  | 深度学习框架         |
| `transformers` | 4.51.3 | FinBERT 模型加载     |
| `pandas`       | 2.3.3  | 数据处理             |
| `numpy`        | 1.26.4 | 数值计算             |
| `scikit-learn` | latest | 标准化 / 评估        |
| `scipy`        | latest | KDE / 信号处理       |
| `yfinance`     | 0.2.66 | Yahoo Finance 数据源 |
| `streamlit`    | latest | Web 前端框架         |
| `plotly`       | latest | 交互式图表           |
| `pandas-ta`    | latest | 技术指标计算         |
| `statsmodels`  | latest | VIF 特征筛选         |

### 4. 模型准备

本项目依赖两个本地模型文件，请确保它们位于 `models/` 目录下：

#### FinBERT（情感分析 — 必需）

将 `yiyanghkust/finbert-tone` 的模型文件下载至 `models/yiyanghkust_finbert-tone/`：

```bash
# 使用 huggingface_hub 下载（推荐）
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='yiyanghkust/finbert-tone',
    local_dir='models/yiyanghkust_finbert-tone'
)
"
```

目录结构应为：

```
models/yiyanghkust_finbert-tone/
├── config.json
├── vocab.txt
├── tokenizer_config.json
├── model.safetensors (或 pytorch_model.bin)
└── special_tokens_map.json
```

#### Hybrid Model（股价预测 — 可选）

您需要先运行训练脚本生成模型，或将预训练好的权重放入指定目录：

```
models/hybrid_model/checkpoints/
├── hybrid_model.pt          # 模型权重
└── hybrid_scalers.npz       # 特征标准化参数
```

> 💡 如果没有该模型文件，系统会以降级模式运行（T+1 预测功能不可用，其他功能正常）。

---

## 📊 数据处理流程（Pipeline）

> **快速通道**：如果 `datas/` 文件夹中已经包含处理好的 `stock_datas.csv`（含所有特征列），您可以 **直接跳到** [启动应用](#-启动应用) 环节。

数据处理遵循严格的先后顺序，每一步覆盖输入文件并自动创建 `.bak` 备份。

```
stock_statistics.csv ──► get_data.py ──► stock_datas.csv (原始行情)
                                              │
analyst_ratings_processed.csv ──► merge_news_into_stock_data.py
                                              │
                                    stock_datas.csv (+ NewsTitles)
                                              │
                              feature_engineering.py
                                              │
                                    stock_datas.csv (+ 技术指标 + 情感)
                                              │
                                clean_stock_data.py
                                              │
                                    stock_datas.csv (清洗后完整数据)
                                              │
                              clean_stock_data_vif.py (可选)
                                              │
                                    stock_datas_vif.csv (精简21列)
```

### 步骤 1：下载原始数据

从 Yahoo Finance 下载股票和宏观指数（^VIX, ^TNX, ^GSPC）日线数据：

```bash
python get_data.py --start 2009-01-01 --end 2019-12-31
```

**输入**：`datas/stock_statistics.csv`（筛选 `Percentage >= 0.1` 的股票）

**输出**：`datas/stock_datas.csv`（列：Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Type）

| 参数            | 默认值       | 说明           |
| :-------------- | :----------- | :------------- |
| `--start`       | `2009-01-01` | 起始日期       |
| `--end`         | `2019-12-31` | 截止日期       |
| `--threshold`   | `0.1`        | 股票筛选阈值   |
| `--max-workers` | `8`          | 并发下载线程数 |
| `--retries`     | `4`          | 失败重试次数   |

### 步骤 2：合并新闻数据

将新闻数据集（`analyst_ratings_processed.csv`）按 `(Ticker, Date)` 聚合后合并到股价数据：

```bash
python merge_news_into_stock_data.py
```

**输出**：更新 `datas/stock_datas.csv`（新增 `NewsTitles` 列，多条新闻用 `" | "` 连接）

### 步骤 3：特征工程

计算技术指标并进行 FinBERT 情感打分：

```bash
python feature_engineering.py
```

新增特征列一览：

| 类别        | 特征                                                         |
| :---------- | :----------------------------------------------------------- |
| 宏观基准    | `GSPC_Close`, `GSPC_LogReturn`                               |
| 收益 & 波动 | `LogReturn`, `Volatility_20`                                 |
| 技术指标    | `RSI_14`, `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`   |
| 均线        | `SMA_5`, `SMA_20`, `EMA_12`, `EMA_26`                        |
| 布林带      | `BBL/BBM/BBU/BBB/BBP_20_2.0_2.0`                             |
| 量价        | `ATR_14`, `OBV`                                              |
| 衍生指标    | `Intraday_Range`, `Trend_Strength`, `Candle_Body`            |
| 因子        | `Beta_60`, `Alpha_60`, `Sharpe_60`（60 日滚动，Sharpe 年化 ×√252） |
| 时间编码    | `Month_Sin`（月份正弦编码）                                  |
| 情感        | `Sentiment_Score`（FinBERT：P(pos) − P(neg)）                |

### 步骤 4：数据清洗

去除异常值、处理缺失数据：

```bash
python clean_stock_data.py
```

- 填充缺失：`Sentiment_Score → 0`，`NewsTitles → ""`
- 删除滚动窗口产生的首段 NaN 行（按 Ticker）
- 异常值盖帽：收益/比例类列按 Ticker 做 1% / 99% 分位数裁剪

### 步骤 5：VIF 特征筛选（可选）

```bash
# 生成 VIF 分析报告
python vif_feature_selection.py --threshold 5.0

# 输出精简数据集（21 列）
python clean_stock_data_vif.py
```

VIF 保留的 18 个特征：

```
Volume, GSPC_Close, GSPC_LogReturn, LogReturn, MACDh_12_26_9, MACDs_12_26_9,
Intraday_Range, Trend_Strength, Candle_Body, Month_Sin, Sentiment_Score,
BBB_20_2.0_2.0, BBP_20_2.0_2.0, ATR_14, OBV, Beta_60, Alpha_60, Sharpe_60
```

---

## 🖥️ 启动应用

完成数据准备后，启动 Streamlit 前端界面：

```bash
streamlit run app.py
```

启动后，浏览器会自动打开 `http://localhost:8501`。

---

## 💡 使用指南

### 侧边栏配置

| 配置项               | 说明                                                         |
| :------------------- | :----------------------------------------------------------- |
| **DeepSeek API Key** | 输入 `sk-xxx...` 格式的密钥以启用 AI 对话分析。留空则使用规则引擎降级模式 |
| **股票代码**         | 输入股票代号（如 `EBAY`）或中文名称（如 `eBay`），下方显示匹配建议 |
| **日期范围**         | 选择数据的起止日期（受限于数据集覆盖范围）                   |
| **模拟日期**         | 选择要模拟分析的具体交易日（必须在日期范围内）               |
| **启动挖掘与预测**   | 点击按钮开始完整的分析流程                                   |

### 主界面布局

```
┌──────────────────────────────────────────────────────┐
│  📊 多模态特征关联图（股价 + 情感双轴）                   │
│  📈 技术指标分析（RSI / MACD / SMA 三联图）              │
├──────────────────────────┬───────────────────────────┤
│                          │  📰 结构化情报                │
│                          │  （最近5天新闻 + 情感评分）     │
├──────────────────────────┴───────────────────────────┤
│  📊 关键指标仪表盘（5 列 Metric 卡片）                    │
│  当前股价 | T+1 预测 | 市场风险 | 支撑|阻力 | RSI         │
├──────────────────────────────────────────────────────┤
│  💬 金融分析聊天机器人                                    │
│  ┌──────────────────────────────────────────────┐    │
│  │  📊 短期技术分析  │ 🔍 历史形态回测               │    │
│  │  📜 财报健康度    │ 🎯 综合投资决策               │    │
│  └──────────────────────────────────────────────┘    │
│  [用户输入框 / AI 对话区]                                │
└──────────────────────────────────────────────────────┘
```

### 快捷分析卡片

| 卡片               | 触发的工具                     | 分析内容                          |
| :----------------- | :----------------------------- | :-------------------------------- |
| 📊 **短期技术分析** | `predict` + `calculate_levels` | T+1 走势预测、支撑阻力位          |
| 🔍 **历史形态回测** | `search_similar_periods`       | 最近 20 天相似 K 线形态及后续表现 |
| 📜 **财报健康度**   | `analyze_financial_statements` | SEC 10-K 财报深度解读             |
| 🎯 **综合投资决策** | 全部工具                       | 完整的多维度投资分析报告          |

---

## 🔧 工具箱详解

所有工具通过 `tool_registry.py` 统一注册，由 `decision_agent.py` 按照 DeepSeek Function Calling 协议自动调度。

### 工具一览

| 工具名                         | 类                           | 文件                              | 功能                        |
| :----------------------------- | :--------------------------- | :-------------------------------- | :-------------------------- |
| `calc_score`                   | `FinBERTAnalyzer`            | `finbert_analyzer.py`             | 计算新闻情感分数 \[-1, 1\]  |
| `classify_texts`               | `FinBERTAnalyzer`            | `finbert_analyzer.py`             | 新闻三分类 + 概率           |
| `predict`                      | `HybridPredictor`            | `predict_t1.py`                   | T+1 价格预测（方向 + 幅度） |
| `calculate_levels`             | `SupportResistanceScanner`   | `support_resistance.py`           | KDE 支撑阻力位识别          |
| `search_similar_periods`       | `SimilaritySearcher`         | `similarity_search.py`            | 历史形态相似度搜索          |
| `analyze_market_regime`        | `MacroAnalyzer`              | `macro_analyzer.py`               | 宏观环境风险评估            |
| `analyze_financial_statements` | `FinancialStatementAnalyzer` | `financial_statement_analyzer.py` | SEC 财报分析                |

### Agent SOP 思考框架

Agent 接收到用户请求后，严格遵循以下决策流程：

```
1. 宏观优先    → 调用 analyze_market_regime，评估系统性风险
2. 问题定性    → 判断：短期交易 / 长期投资 / 全面分析
3. 路径选择    → A) 短期路径：predict + calculate_levels + search_similar_periods
                  B) 长期路径：analyze_financial_statements
                  C) 综合路径：A + B
4. 数据综合    → 解读、关联、发现矛盾或共识
5. 报告生成    → 按结构化模板输出（核心观点 → 宏观 → 量化 → 基本面 → 风险）
```

---

## 📂 项目结构

```
FinanceAgent/
├── app.py                            # Streamlit 前端主程序（根目录入口）
├── frontend/
│   └── app.py                        # Streamlit 界面实现
│
├── agents/
│   ├── __init__.py
│   ├── decision_agent.py             # AI 决策大脑（DeepSeek Function Calling）
│   └── Tools/
│       ├── __init__.py
│       ├── tool_registry.py          # 工具注册表（API 规范 + 类映射）
│       ├── finbert_analyzer.py       # FinBERT 情感分析器（单例）
│       ├── predict_t1.py             # Hybrid 多任务预测器
│       ├── support_resistance.py     # KDE 支撑阻力位扫描器
│       ├── similarity_search.py      # 历史形态相似度搜索器
│       ├── macro_analyzer.py         # 宏观市场环境分析器
│       └── financial_statement_analyzer.py  # SEC 财报分析器
│
├── models/
│   ├── __init__.py
│   ├── hybrid_model/
│   │   ├── hybrid_model.py           # 模型结构定义与推理接口
│   │   ├── hybrid_model_trainer.py   # 训练脚本
│   │   └── checkpoints/
│   │       ├── hybrid_model.pt       # 模型权重
│   │       └── hybrid_scalers.npz    # 特征标准化参数
│   └── yiyanghkust_finbert-tone/     # FinBERT 预训练模型文件
│       ├── config.json
│       ├── vocab.txt
│       └── model.safetensors
│
├── datas/                            # 数据存放目录
│   ├── stock_statistics.csv          # 股票统计信息（用于筛选）
│   ├── analyst_ratings_processed.csv # 新闻情绪数据
│   ├── stock_datas.csv               # 最终处理好的完整数据
│   └── stock_datas_vif.csv           # VIF 筛选后的精简数据
│
├── scripts/
│   └── test_macro_analyzer.py        # 宏观分析单元测试
│
├── get_data.py                       # Step 1：数据下载脚本
├── merge_news_into_stock_data.py     # Step 2：新闻合并脚本
├── feature_engineering.py            # Step 3：特征工程脚本
├── clean_stock_data.py               # Step 4：数据清洗脚本
├── clean_stock_data_vif.py           # Step 5：VIF 精简脚本
├── vif_feature_selection.py          # VIF 特征共线性分析 & 报告
├── draw_architecture.py              # 架构图绘制（Graphviz）
├── generate_requirements.py          # 自动扫描生成 requirements.txt
│
├── requirements.txt                  # 依赖库列表
├── selected_features.md              # VIF 特征筛选报告
├── stock_datas_explained.md          # 数据流程说明文档
└── README.md                         # 本文件
```

---

## 🔬 技术细节

### Hybrid 多任务模型架构

```
                 Input
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Scalar Branch          Text Branch
  (18-dim VIF)          (768-dim FinBERT)
        │                     │
   ┌────▼────┐          ┌─────▼─────┐
   │  LSTM   │          │ Transformer│
   │ Encoder │          │  Encoder   │
   └────┬────┘          └─────┬─────┘
        │                     │
        └──────────┬──────────┘
                   │ Concat
            ┌──────▼──────┐
            │  FC Fusion  │
            └──────┬──────┘
          ┌────────┴────────┐
          ▼                 ▼
   Direction Head     Magnitude Head
   (Sigmoid → P↑)    (Regression → |Δ|)
          │                 │
          └────────┬────────┘
                   ▼
        predicted_price = price × exp(±|Δ|)
```

### 特征工程总览

| 特征类别    |  数量  | 计算方法                              |
| :---------- | :----: | :------------------------------------ |
| 收益 & 波动 |   2    | 对数收益率、20 日滚动标准差           |
| 趋势指标    |   5    | SMA(5/20)、EMA(12/26)、Trend_Strength |
| 动量指标    |   4    | RSI(14)、MACD 三件套                  |
| 布林带      |   5    | BBL/BBM/BBU/BBB/BBP(20,2)             |
| 量价指标    |   3    | ATR(14)、OBV、Intraday_Range          |
| 形态指标    |   1    | Candle_Body                           |
| 因子        |   3    | Beta/Alpha/Sharpe(60 日滚动)          |
| 宏观        |   2    | GSPC_Close、GSPC_LogReturn            |
| 时间编码    |   1    | Month_Sin                             |
| 情感        |   1    | FinBERT Sentiment_Score               |
| **合计**    | **27** | 经 VIF 筛选后保留 **18** 个           |

### 数据文件说明

| 文件                            | 行数量级 | 列数 | 说明                     |
| :------------------------------ | :------- | :--: | :----------------------- |
| `stock_datas.csv`               | ~50 万行 | ~35  | 清洗后的完整数据集       |
| `stock_datas_vif.csv`           | ~50 万行 |  21  | VIF 筛选后的精简数据集   |
| `stock_statistics.csv`          | ~500 行  |  3+  | 股票代码和权重统计       |
| `analyst_ratings_processed.csv` | ~10 万行 |  3   | 新闻标题、日期、股票代码 |

---

## ❓ 常见问题

<details>
<summary><b>Q：没有 DeepSeek API Key 还能用吗？</b></summary>

可以。不配置 API Key 时，系统以**降级模式**运行：
- ✅ 股价图表、技术指标、情感分析、关键指标仪表盘 → **正常显示**
- ✅ T+1 预测、支撑阻力位、宏观风险 → **正常计算**
- ❌ AI 对话问答 → **不可用**（需要 LLM 推理能力）

</details>

<details>
<summary><b>Q：没有 Hybrid 模型文件怎么办？</b></summary>

系统会在初始化时检测模型文件是否存在。如果 `hybrid_model.pt` 不存在：
- T+1 预测功能会显示 `"Hybrid 预测器加载失败"` 并使用默认值
- 其他所有功能（情感分析、支撑阻力、宏观分析、AI 对话）不受影响

</details>

<details>
<summary><b>Q：数据下载时遇到网络问题？</b></summary>

`get_data.py` 默认使用代理 `http://127.0.0.1:7897`。如果您不需要代理，请注释掉文件开头的代理设置：

```python
# proxy = 'http://127.0.0.1:7897'
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy
```

此外，脚本内置了重试机制（默认 4 次）和并发下载（默认 8 线程）。

</details>

<details>
<summary><b>Q：SEC 财报分析报错 503？</b></summary>

SEC EDGAR API 有速率限制（每秒 ≤ 1 个请求）。程序已内置 `time.sleep(1)` 和重试机制。如果仍然报错：
- 稍后重试（SEC 服务器可能在维护）
- 确保 `User-Agent` 字段填写了有效的联系邮箱
- 直接访问 [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html) 手动查询

</details>

<details>
<summary><b>Q：如何自动生成 requirements.txt？</b></summary>

项目提供了 `generate_requirements.py` 脚本，自动扫描所有 `.py` 文件的 `import` 语句并匹配已安装的包版本：

```bash
python generate_requirements.py
```

</details>

---

## 📄 License

本项目仅供学习和研究使用。

---

<p align="center">
  <sub>Built with ❤️ using PyTorch, Transformers, Streamlit & DeepSeek</sub>
</p>
```
