# ⚡ FinanceAgent — 你的 AI 投资分析搭档

> **像资深投资顾问一样思考，像量化交易员一样计算。**

FinanceAgent 不是又一个股价预测工具。它是一个能**读懂新闻、看懂K线、算出财报健康度**的 AI 智能体。你只需要像跟分析师说话一样提问，它会自动调度量化模型给出数据驱动的答案。

---

## 🎯 它能帮你做什么？

| 场景 | 你问 | 它做 |
|---|---|---|
| 🔮 **明日走势预判** | "EBAY 明天涨还是跌？" | 跑 Hybrid 模型，输出方向概率 + 预测价格 + 波动幅度 |
| 📰 **消息面解读** | "最近关于 AAPL 的新闻情绪如何？" | 调 FinBERT，把每条新闻打分，告诉你市场是乐观还是恐慌 |
| 🎯 **找买卖关键位** | "TSLA 在什么价位有支撑？" | KDE 算法扫描历史筹码密集区，给出客观的支撑/阻力位 |
| 🔍 **历史会重演吗** | "现在 MSFT 的走势跟历史上哪段最像？" | 滑动窗口扫描数年数据，回测相似形态后的真实涨跌幅 |
| 🌍 **大局观判断** | "现在适合进场吗？" | 先看 S&P 500、VIX、美债收益率，告诉你市场是 Risk-On 还是 Risk-Off |
| 📜 **挖财报真相** | "NVDA 的基本面健康吗？" | 直连 SEC EDGAR，抓取 10-K 年报，算出毛利率、流动比率、自由现金流 |

**每个答案都有数据支撑，不是 AI 的凭空想象。**

---

## 🧠 它跟普通 AI 工具有什么不同？

```
普通 AI 聊天机器人          FinanceAgent
─────────────────────      ─────────────────────
"我觉得可能会涨"    →       "模型预测上涨概率 68%，波动幅度 2.3%"
"最近新闻偏正面"    →       "FinBERT 评分 +0.42，5条正面 2条中性 1条负面"
"大概在 150 有支撑"  →      "KDE 峰值在 $148.75，强度 0.73，是近一年最大筹码峰"
```

**左脑（DeepSeek LLM）** 负责理解你的问题、规划分析路径、把数据讲成人话。
**右脑（PyTorch 量化工具链）** 负责冷冰冰的数学计算，确保每个数字都有依据。

---

## 🚀 60 秒上手

```bash
# 1. 安装
git clone https://github.com/your-username/FinanceAgent.git
cd FinanceAgent
pip install -r requirements.txt

# 2. 下载模型（只需一次）
python -c "
from huggingface_hub import snapshot_download
snapshot_download('yiyanghkust/finbert-tone', local_dir='models/yiyanghkust_finbert-tone')
"

# 3. 启动
streamlit run app.py
```

打开浏览器，输入 DeepSeek API Key，选一只股票，点击分析 —— 搞定。

> 💡 **没有 API Key？** 所有量化功能照样跑：图表、预测、支撑阻力、情感分析都能用。只是 AI 对话功能需要 Key。

---

## 🏗️ 架构一览

```
你问一个问题
      │
      ▼
┌─────────────────────────────────┐
│  🧠 FinancialAgent (DeepSeek)   │  ← 理解意图 → 规划路径 → 调度工具
└──────────────┬──────────────────┘
               │ Function Calling
      ┌────────┼────────┬────────┬────────┐
      ▼        ▼        ▼        ▼        ▼
   📈T+1预测  📰情感   🎯支撑阻力 🔍形态   📜财报
   Hybrid    FinBERT    KDE    相似搜索   SEC API
      │        │        │        │        │
      └────────┴────────┴────────┴────────┘
               │
      ┌────────▼────────┐
      │  结构化分析报告   │
      │  有观点·有数据·有风险提示 │
      └─────────────────┘
```

---

## 📦 内置工具箱

| 工具 | 一句话说明 | 技术内核 |
|---|---|---|
| **T+1 预测** | 明天涨跌？涨多少？ | LSTM + Transformer 双塔融合，双头输出（方向+幅度） |
| **情感分析** | 市场在乐观还是恐慌？ | FinBERT 金融专属 BERT，三分类 + 连续情感分 |
| **支撑阻力** | 关键买卖位在哪？ | KDE 核密度估计，找筹码密集峰值 |
| **形态匹配** | 历史上有类似的走势吗？ | Z-Score 标准化 + 滑动窗口 + 欧氏/余弦相似度 |
| **宏观扫描** | 大盘什么环境？ | S&P 500 + VIX + 10Y 美债，三维度判定 Risk 状态 |
| **财报深挖** | 公司财务健康吗？ | SEC EDGAR XBRL API → 利润表/资产负债表/现金流 + 比率分析 |

---

## 🖥️ 界面长什么样？

```
┌──────────────────────────────────────────────┐
│  📊 股价 + 情感双轴关联图                       │
│  📈 RSI / MACD / SMA 三联技术指标               │
├────────────────────┬─────────────────────────┤
│  📊 关键指标仪表盘   │  📰 最近 5 天新闻 + 情感    │
│  现价 | 预测 | 风险  │  正面 / 中性 / 负面标签     │
│  支撑 | 阻力 | RSI   │                         │
├────────────────────┴─────────────────────────┤
│  💬 AI 对话区                                  │
│  [📊短期分析] [🔍历史回测] [📜财报解读] [🎯综合报告] │
│  输入你的问题...                                │
└──────────────────────────────────────────────┘
```

一键点击快捷卡片，或者直接打字提问 —— 两种方式都行。

---

## 🔧 数据处理（给想跑完整流程的人）

如果你有原始数据，按顺序跑这 5 步：

```bash
python get_data.py --start 2009-01-01 --end 2019-12-31   # ① 下载行情
python merge_news_into_stock_data.py                      # ② 合并新闻
python feature_engineering.py                             # ③ 特征工程（27→18维）
python clean_stock_data.py                                # ④ 清洗
python clean_stock_data_vif.py                            # ⑤ VIF 精简（可选）
```

每一步自动备份上一版数据为 `.bak`，放心跑。

---

## 📂 项目结构

```
FinanceAgent/
├── app.py                          # 启动入口（→ frontend/app.py）
├── frontend/app.py                 # Streamlit 界面
├── agents/
│   ├── decision_agent.py           # AI 决策大脑
│   └── Tools/                      # 7 个量化工具
├── models/
│   ├── hybrid_model/               # LSTM+Transformer 预测模型
│   └── yiyanghkust_finbert-tone/   # FinBERT 情感模型
├── datas/                          # 数据文件（CSV）
├── get_data.py                     # 数据下载
├── feature_engineering.py          # 特征工程
├── clean_stock_data.py             # 数据清洗
└── requirements.txt
```

---

## ⚙️ 技术栈

`PyTorch 2.7` · `Transformers 4.51` · `Streamlit` · `Plotly` · `DeepSeek V3` · `FinBERT` · `yfinance` · `scikit-learn` · `SEC EDGAR API`

---

<p align="center">
  <sub>Built with PyTorch, Transformers, Streamlit & DeepSeek</sub>
</p>
