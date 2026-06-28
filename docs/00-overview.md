# FinanceAgentV2.0 — 项目总览

> **多模态金融量化挖掘智能体**：融合 DeepSeek LLM 推理与本地 PyTorch/Transformers 量化模型，提供数据驱动的结构化投资分析报告。

---

## 1. 项目定位

FinanceAgentV2.0 是一个"**双脑协同**"架构的金融决策支持系统：

| 维度 | 左脑（LLM） | 右脑（Quant Tools） |
|---|---|---|
| 角色 | 决策规划者 | 计算执行者 |
| 实现 | DeepSeek V3（OpenAI SDK Function Calling） | 7 个本地 PyTorch/SciPy 工具 |
| 输入 | 用户自然语言问题 | 结构化股票数据 + 模型参数 |
| 输出 | 结构化投资分析报告 | 标准化的 JSON 计算结果 |
| 位置 | [`agents/decision_agent.py`](../../agents/decision_agent.py) | [`agents/Tools/`](../../agents/Tools/) |

**核心理念**：

- **有数据、有观点、有依据**——所有结论均由工具计算结果支撑，不让 LLM"凭空想象"。
- **可降级运行**——未配置 DeepSeek API Key 时，前端仍可使用图表、预测、支撑阻力、情感分析等所有量化功能。
- **多模态输入**——价格（数值时序）+ 财经新闻（文本）+ 财报（XBRL），通过 FinBERT 文本塔与 LSTM 时序塔融合预测。

---

## 2. 技术栈

| 类别 | 库 | 版本 | 用途 |
|---|---|---|---|
| 深度学习 | `torch` | 2.7.0 | 混合模型训练/推理 |
| 预训练模型 | `transformers` | 4.51.3 | FinBERT 加载与文本编码 |
| LLM 客户端 | `openai` | (运行时) | DeepSeek V3 Function Calling |
| 数据获取 | `yfinance` | 0.2.66 | Yahoo Finance 行情/宏观指数 |
| 数据处理 | `pandas` / `numpy` | 2.3.3 / 1.26.4 | 表格与数组 |
| 技术指标 | `pandas-ta` | - | MACD/RSI/BBANDS/OBV/ATR/EMA |
| 统计 | `scipy` / `statsmodels` | - | KDE、VIF |
| 机器学习 | `scikit-learn` | - | RobustScaler |
| 前端 | `streamlit` / `plotly` | - | 交互界面与图表 |
| 财报数据 | `requests` (内置) | - | SEC EDGAR XBRL API |

完整依赖见 [`requirements.txt`](../../requirements.txt)，可通过 `python generate_requirements.py` 自动从当前环境重新生成。

---

## 3. 目录结构

```
FinanceAgentV2.0/
├── agents/                          # 决策智能体层
│   ├── decision_agent.py            #   - FinancialAgent（DeepSeek 调用 + 工具调度）
│   └── Tools/                       #   - 7 个量化工具
│       ├── tool_registry.py         #     工具注册表（DeepSeek API 规范）
│       ├── finbert_analyzer.py      #     FinBERT 情感分析（单例）
│       ├── predict_t1.py            #     Hybrid 模型推理（T+1 价格）
│       ├── support_resistance.py    #     KDE 支撑/阻力位
│       ├── similarity_search.py     #     Z-Score 滑动窗口形态匹配
│       ├── macro_analyzer.py        #     宏观环境（^GSPC/^VIX/^TNX）
│       └── financial_statement_analyzer.py  # SEC EDGAR 10-K 财报分析
│
├── models/                          # 深度学习模型层
│   ├── hybrid_model/
│   │   ├── hybrid_model.py          #   - HybridModel（LSTM+Transformer+CrossAttn）
│   │   ├── hybrid_model_trainer.py  #   - 训练脚本（多任务：DA + Magnitude）
│   │   └── checkpoints/             #   - hybrid_model.pt + hybrid_scalers.npz
│   └── yiyanghkust_finbert-tone/    #   本地 FinBERT 权重（gitignored）
│
├── frontend/
│   └── app.py                       # Streamlit 界面
│
├── datas/                           # 数据层（gitignored）
│   ├── stock_statistics.csv         #   候选股票清单（含 Percentage 过滤阈值）
│   ├── stock_datas.csv              #   主数据表（OHLCV + 27 特征 + 新闻）
│   ├── stock_datas_vif.csv          #   VIF 筛选后的精简数据（18 特征）
│   ├── analyst_ratings_processed.csv
│   └── ...
│
├── docs/                            # 本文档目录
│
├── get_data.py                      # ① Yahoo Finance 数据下载
├── merge_news_into_stock_data.py    # ② 合并新闻标题
├── feature_engineering.py           # ③ 27 维特征工程
├── clean_stock_data.py              # ④ 异常值裁剪 + 缺失填充
├── vif_feature_selection.py         # ⑤ VIF 迭代降维（→ 18 维）
├── clean_stock_data_vif.py          # ⑥ VIF 特征子集输出
├── generate_requirements.py         # 工具：扫描 imports 生成 requirements.txt
├── app.py                           # 启动入口（→ frontend/app.py）
├── CLAUDE.md                        # 仓库级 AI 协作文档
└── README.md                        # 用户文档
```

> ⚠️ **目录警告**：历史遗留一个 `FinanceAgentV2.0/FinanceAgentV2.0/` 嵌套子目录，是非规范的复制产物。所有开发/运行均应在根目录 `F:\Python\FinanceAgentV2.0` 下进行。

---

## 4. 数据流概览

```
┌──────────────────────────────────────────────────────────────────────┐
│                         离线数据管道（Python 脚本）                     │
└──────────────────────────────────────────────────────────────────────┘

stock_statistics.csv
       │
       ▼
get_data.py ─────────────────► stock_datas.csv   (OHLCV + ^GSPC/^VIX/^TNX)
                                       │
analyst_ratings_processed.csv          │
       │                               │
       ▼                               │
merge_news_into_stock_data.py          │
       │                               │
       └─────────► stock_datas.csv   (含 NewsTitles 列)
                       │
                       ▼
              feature_engineering.py  (+ 27 维特征 + FinBERT 情感)
                       │
                       ▼
              stock_datas.csv  (37 列：基础+技术+情感+日历)
                       │
                       ▼
              clean_stock_data.py   (裁剪 1%/99% 分位 + 填 NaN)
                       │
                       ▼
              stock_datas.csv  (清洗后)
                       │
                       ▼
              vif_feature_selection.py  (迭代删除 VIF>5 的特征)
                       │
                       ▼
              clean_stock_data_vif.py
                       │
                       ▼
              stock_datas_vif.csv  (18 维 VIF 特征 + ID 列)
                       │
                       ▼
              hybrid_model_trainer.py
                       │
                       ▼
              hybrid_model.pt + hybrid_scalers.npz


┌──────────────────────────────────────────────────────────────────────┐
│                          在线运行时（Streamlit）                       │
└──────────────────────────────────────────────────────────────────────┘

用户 ──► Streamlit UI ──► FinancialAgent.run()
                              │
                              ├─► FinBERTAnalyzer      (情感)
                              ├─► HybridPredictor       (T+1)
                              ├─► SupportResistanceScanner (KDE 支撑/阻力)
                              ├─► SimilaritySearcher    (形态匹配)
                              ├─► MacroAnalyzer         (宏观)
                              └─► FinancialStatementAnalyzer (SEC 财报)
                              │
                              ▼
                      结构化分析报告 (Markdown)
```

---

## 5. 模块文档导航

| 编号 | 文档 | 涵盖内容 |
|---|---|---|
| 1 | [01-architecture.md](./01-architecture.md) | 系统整体架构、双脑协同、消息流、关键类与函数清单 |
| 2 | [02-data-pipeline.md](./02-data-pipeline.md) | 6 步数据管道：下载/合并/特征/清洗/VIF/输出 |
| 3 | [03-hybrid-model.md](./03-hybrid-model.md) | 双塔+融合+双头架构、训练循环、推理接口 |
| 4 | [04-agent-decision.md](./04-agent-decision.md) | DeepSeek Function Calling 循环、SOP 提示词、工具路由 |
| 5 | [05-tools.md](./05-tools.md) | 7 个量化工具的算法、参数、返回结构 |
| 6 | [06-frontend.md](./06-frontend.md) | Streamlit 页面布局、缓存、降级模式、Plotly 图表 |
| 7 | [07-deployment.md](./07-deployment.md) | 环境搭建、模型下载、训练、运行、代理配置 |

---

## 6. 快速验证清单

- ✅ **数据**：默认 `datas/stock_datas.csv` 必须存在（约 30 支 2009-2019 行情 + 3 个宏观指数）。
- ✅ **模型**：`models/hybrid_model/checkpoints/hybrid_model.pt` 与 `hybrid_scalers.npz` 必须存在。
- ✅ **FinBERT**：`models/yiyanghkust_finbert-tone/` 必须包含 `config.json` / `model.safetensors` / `vocab.txt` 等文件。
- ✅ **API Key（可选）**：用户在前端输入 DeepSeek Key，未提供时自动降级。
- ✅ **网络**：`analyze_financial_statements` 需访问 `data.sec.gov`，`get_data.py` 需访问 Yahoo Finance。
