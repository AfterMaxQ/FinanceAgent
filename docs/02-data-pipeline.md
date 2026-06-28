# 02 · 数据管道（Data Pipeline）

> 6 步数据处理管道：行情下载 → 新闻合并 → 特征工程 → 清洗 → VIF 降维 → 模型输入。

---

## 1. 全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 0                  datas/stock_statistics.csv                  │
│                          (含 Stock + Percentage 列)                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 过滤 Percentage ≥ 0.1
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1  get_data.py  ──►  datas/stock_datas.csv (OHLCV)             │
│            yf.Ticker().history() 多线程 + 指数代理 + 重试              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
┌─────────────────────────────┐  ┌────────────────────────────────────┐
│ Step 2  merge_news_into_    │  │ datas/analyst_ratings_processed.csv│
│        stock_data.py        │  │  (title, date, stock)              │
│  ──► stock_datas.csv        │  └────────────────────────────────────┘
│       (+ NewsTitles 列)      │
└─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3  feature_engineering.py  ──►  stock_datas.csv (+27 维特征)    │
│   · GSPC_Close / GSPC_LogReturn  (合并 ^GSPC)                         │
│   · LogReturn, Volatility_20                                              │
│   · BBANDS(20,2): BBL/BBM/BBU/BBB/BBP                                  │
│   · ATR_14, EMA_12, EMA_26                                              │
│   · OBV, RSI_14, MACD(12,26,9): MACD/MACDh/MACDs                       │
│   · SMA_5, SMA_20                                                        │
│   · Intraday_Range, Trend_Strength, Candle_Body                          │
│   · Month_Sin                                                            │
│   · Factor: Beta_60, Alpha_60, Sharpe_60                                │
│   · Sentiment_Score (FinBERT CLS → P-Pos - P-Neg)                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4  clean_stock_data.py  ──►  stock_datas.csv (cleaned)            │
│   · Sentiment_Score NaN → 0；NewsTitles NaN → ""                        │
│   · 丢弃每个 Ticker 前导 NaN（滚动指标未稳定）                            │
│   · ratio 类列 1%/99% 分位裁剪                                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5  vif_feature_selection.py  ──►  selected_features.md          │
│   · 标准化 → 迭代删除 max VIF>5 → 输出 18 维特征报告                    │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6  clean_stock_data_vif.py  ──►  stock_datas_vif.csv            │
│   · 保留 ID 列 + 18 个 VIF 特征                                          │
│   · 即为 hybrid_model_trainer 的输入                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step 1 — 数据下载（`get_data.py`）

### 2.1 核心功能

- **数据源**：Yahoo Finance（`yfinance`）
- **股票过滤**：读取 `datas/stock_statistics.csv`，仅保留 `Percentage ≥ 0.1` 的 Ticker
- **宏观指数**：固定下载 `^VIX`（恐慌指数）、`^TNX`（10 年美债收益率）、^GSPC（标普 500）
- **字段**：`Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Type`
- **日期范围**：默认 `2009-01-01` ~ `2019-12-31`（覆盖 1 个完整牛熊周期）
- **并发**：`ThreadPoolExecutor(max_workers=8)` 拉取个股，3 线程拉取宏观
- **重试**：4 次指数退避；区分"无数据"（终止）和"被限流"（继续重试）
- **代理**：脚本顶部硬编码 `http://127.0.0.1:7897`（按需注释）

### 2.2 关键函数

```python
def fetch_single_ticker_history(ticker, start, end, period) -> pd.DataFrame:
    """yf.Ticker().history() 单只拉取。返回的 index 转为无时区 datetime。"""

def download_all(tickers, ...) -> pd.DataFrame:
    """ThreadPoolExecutor 并行拉取，统一返回拼接后的 DataFrame。"""
```

### 2.3 运行

```bash
python get_data.py --start 2009-01-01 --end 2019-12-31
# 自定义阈值与代理
python get_data.py --threshold 0.05 --max-workers 4
```

---

## 3. Step 2 — 新闻合并（`merge_news_into_stock_data.py`）

### 3.1 输入/输出

| 输入 | 文件 | 必需列 |
|---|---|---|
| 主数据 | `datas/stock_datas.csv` | `Date, Ticker` |
| 新闻 | `datas/analyst_ratings_processed.csv` | `title, date, stock` |

输出：原主数据 + 一列 `NewsTitles`（同一天同一 Ticker 多条标题用 `" | "` 拼接）。

### 3.2 关键步骤

1. **日期归一化**：`pd.to_datetime(..., utc=True) → tz_convert(None) → normalize()`，消除时区与时间分量差异。
2. **代码大写对齐**：`Ticker` 列统一 `str.upper().str.strip()`，避免 `EBAY` vs `ebay` 错配。
3. **分组聚合**：`groupby(["Ticker", "Date"])["title"].apply(lambda s: " | ".join(s))`。
4. **左连接**：`stock_df.merge(news_grouped, how="left", on=["Ticker", "Date"])`，无新闻的日期 `NewsTitles` 为 NaN（后续清洗时填 `""`）。
5. **备份**：默认会写一份 `.bak`，可用 `--no-backup` 跳过。

---

## 4. Step 3 — 特征工程（`feature_engineering.py`）

### 4.1 输出特征清单（最终 27+ 维）

| 类别 | 特征 | 实现 |
|---|---|---|
| 行情 | Open, High, Low, Close, Volume, Adj Close | yfinance 原始 |
| 日历 | `Month_Sin` | `sin(2π·month/12)` |
| 宏观（合并 ^GSPC） | `GSPC_Close`, `GSPC_LogReturn` | 按 Date 映射 |
| 收益率与波动 | `LogReturn`, `Volatility_20` | 滚动 std |
| 布林带 (20,2) | `BBL_20_2.0_2.0`, `BBM_...`, `BBU_...`, `BBB_...`, `BBP_...` | `pandas_ta.bbands` |
| 趋势 | `EMA_12`, `EMA_26`, `SMA_5`, `SMA_20` | `pandas_ta.ema/sma` |
| MACD | `MACD_12_26_9`, `MACDh_...`, `MACDs_...` | `pandas_ta.macd` |
| 动量 | `RSI_14` | `pandas_ta.rsi` |
| 量能 | `OBV` | `pandas_ta.obv` |
| 波动 | `ATR_14` | `pandas_ta.atr` |
| 形态 | `Intraday_Range` = (H-L)/(C+ε), `Trend_Strength` = C/SMA_20, `Candle_Body` = (C-O)/(O+ε) | 手写 |
| 因子 | `Beta_60`, `Alpha_60`, `Sharpe_60` | 60 日滚动协方差/均值/标准差 |
| 情感 | `Sentiment_Score` | FinBERT P(Positive) − P(Negative) |

### 4.2 关键函数与实现

```python
def _compute_group_features(g: pd.DataFrame) -> pd.DataFrame:
    """对单只股票按时间排序后逐列计算所有技术指标。"""
    
def add_gspc_features(df):
    """从 df 中抽出 ^GSPC 行，按 Date 映射回主表。"""
    
def add_factor_metrics(df, window=60):
    """CAPM 风格：Beta = cov(ret, mret)/var(mret); Alpha = mean(ret) - Beta*mean(mret); Sharpe = (mean/std)*sqrt(252)"""

def compute_sentiment(df, finbert_path, batch_size=16, max_len=256):
    """FinBERT 批推理：每条 NewsTitles → P(Pos) - P(Neg) ∈ [-1, 1]"""
```

### 4.3 情感分数计算

```python
id2label = {0: 'positive', 1: 'negative', ...}  # 显式读 config 避免顺序错位
score = probs[:, pos_id] - probs[:, neg_id]      # ∈ [-1, 1]
```

- 缺失新闻 → `NaN`（Step 4 填 0）
- 有效新闻 → 连续值

### 4.4 并行优化

`_compute_group_features` 对每只股票独立计算，因此可以直接 `groupby("Ticker").apply(...)`。当前实现是显式 `for ... append` 的形式（已能处理 ~30 支股票 × 11 年）。

---

## 5. Step 4 — 清洗（`clean_stock_data.py`）

### 5.1 三类操作

1. **缺失填充**
   - `Sentiment_Score` NaN → `0`
   - `NewsTitles` NaN → `""`

2. **前导 NaN 裁剪**
   - 每只 Ticker 单独处理：丢弃所有滚动指标（MACD、SMA、BBANDS、ATR、OBV、Beta/Alpha/Sharpe 等）尚未稳定的起始行。
   - 触发条件：`group[rolling_cols].notnull().all(axis=1)` 第一次为 True 即从此行开始。

3. **1%/99% 分位裁剪**
   - 匹配正则 `(return|volatility|sharpe|beta|alpha|trend|candle|intraday|macd|rsi)` 选出"比率/收益率"类列。
   - `series.clip(q_low, q_high)` 双向裁剪，消除异常尖刺但保留分布形态。
   - 当 `q_low == q_high`（极端情况下常数列）时跳过。

### 5.2 关键函数

```python
def fill_missing(df)        # 填充 sentiment 与 news
def drop_leading_rolling_nans(group, required_cols)  # 每只股票从首个全有效行开始
def pick_ratio_cols(df)     # 正则选出"比率类"列
def cap_outliers(group, cols)  # 1%/99% clip
def clean_dataframe(df)     # 组合三个操作
```

---

## 6. Step 5 — VIF 迭代降维（`vif_feature_selection.py`）

### 6.1 算法

VIF（Variance Inflation Factor，方差膨胀因子）衡量某特征能否被其他特征线性表示。VIF 越大，多重共线性越严重。

**迭代过程**：

1. 读取 `stock_datas.csv`，仅保留数值列；剔除标准差为 0 的常数列。
2. 删除含 NaN 的行；按列做 z-score 标准化（避免 VIF 数值不稳定）。
3. 计算当前所有特征的 VIF，找出 `max_vif`。
4. 若 `max_vif > threshold`（默认 5），删除该特征，循环；否则停止。
5. 输出 Markdown 报告 `selected_features.md`，记录每次迭代与最终保留的 18 维特征。

### 6.2 VIF 公式

$$
\text{VIF}_i = \frac{1}{1 - R_i^2}
$$

其中 $R_i^2$ 是特征 $X_i$ 对其余特征的线性回归决定系数。

### 6.3 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 5.0 | VIF 上限；调高（如 10）保留更多特征 |
| `--max-rows` | None | 随机采样加速 VIF 计算 |
| `--random-state` | 42 | 采样种子 |

### 6.4 输出报告结构（`selected_features.md`）

```markdown
# VIF特征筛选报告
- 阈值: 5.0
- 用于计算的样本行数: 12345
- 初始迭代次数: 12
## 保留的特征
- Volume
- GSPC_Close
- ...
## 迭代VIF明细（按最大VIF排序）
### 迭代 1
- 最大VIF特征: Adj Close
- 最大VIF值: 87.31
| 特征 | VIF |
| --- | ---: |
| Adj Close | 87.31 |
| ... | ... |
## 最终特征VIF
| 特征 | VIF |
| --- | ---: |
| Volume | 1.34 |
| ... | ... |
```

### 6.5 最终保留的 18 维 VIF 特征

```
Volume, GSPC_Close, GSPC_LogReturn, LogReturn,
MACDh_12_26_9, MACDs_12_26_9, Intraday_Range,
Trend_Strength, Candle_Body, Month_Sin, Sentiment_Score,
BBB_20_2.0_2.0, BBP_20_2.0_2.0, ATR_14, OBV,
Beta_60, Alpha_60, Sharpe_60
```

这 18 个名字同时被硬编码在两个位置以保证完全一致：
- [`clean_stock_data_vif.py::VIF_FEATURES`](../../clean_stock_data_vif.py) — 离线过滤
- [`agents/Tools/predict_t1.py::VIF_FEATURES`](../../agents/Tools/predict_t1.py) — 在线推理

> 📌 若重新运行 VIF 调整了特征集，**必须同步更新**两处以及 [`hybrid_model_trainer.py::VIF_FEATURES`](../../models/hybrid_model/hybrid_model_trainer.py)。

---

## 7. Step 6 — VIF 特征子集输出（`clean_stock_data_vif.py`）

最简单的步骤：

```python
df = pd.read_csv(input_path)
keep_cols = ID_COLS + VIF_FEATURES  # 3 + 18 = 21 列
df[keep_cols].to_csv(output_path)
```

输出文件 `stock_datas_vif.csv` 即 [`hybrid_model_trainer.py`](../../models/hybrid_model/hybrid_model_trainer.py) 的输入。

---

## 8. 完整运行示例

```bash
# 一次性端到端
python get_data.py --start 2009-01-01 --end 2019-12-31
python merge_news_into_stock_data.py
python feature_engineering.py
python clean_stock_data.py
python vif_feature_selection.py --threshold 5.0
python clean_stock_data_vif.py

# 训练
python models/hybrid_model/hybrid_model_trainer.py \
  --csv-path datas/stock_datas_vif.csv \
  --finbert-dir models/yiyanghkust_finbert-tone \
  --output-dir models/hybrid_model/checkpoints

# 启动 UI
streamlit run app.py
```

---

## 9. 数据完整性自检

| 检查项 | 期望 |
|---|---|
| `datas/stock_datas.csv` 行数 | 约 30 支 × 11 年 × 252 ≈ 80k+ 行 |
| 必含列 | `Date, Ticker, Open, High, Low, Close, Volume, NewsTitles, Sentiment_Score, LogReturn, GSPC_Close, GSPC_LogReturn, ...` |
| 宏观指数 | `^GSPC, ^VIX, ^TNX` 各约 2500 行 |
| `datas/stock_datas_vif.csv` 列数 | 21 (`Date, Ticker, NewsTitles` + 18 VIF) |
| 日期范围 | 默认 2009-01-01 ~ 2019-12-31 |

---

## 10. 备份与容错

每个会覆写输入的脚本默认创建 `.bak`：

```
stock_datas.csv
stock_datas.csv.bak
```

可加 `--no-backup` 关闭。

所有步骤失败均不会破坏已存在的 `.bak`，可手动回滚。
