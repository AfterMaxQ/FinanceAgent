## 数据流程

所有脚本默认覆盖 `datas/stock_datas.csv`，写入前自动创建 `.bak` 备份。

### 1. `get_data.py` - 获取原始数据
- 从 `datas/stock_statistics.csv` 筛选 `Percentage >= 0.1` 的股票
- 使用 yfinance 下载股票和宏观指数（^VIX, ^TNX, ^GSPC）日线数据
- 输出：`datas/stock_datas.csv`（原始行情：Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Type）

### 2. `merge_news_into_stock_data.py` - 合并新闻
- 读取 `datas/analyst_ratings_processed.csv`
- 按 `(Ticker, Date)` 聚合新闻标题，用 `" | "` 连接
- 输出：更新 `datas/stock_datas.csv`（添加 `NewsTitles` 列）

### 3. `feature_engineering.py` - 特征工程
- **宏观**：`GSPC_Close`, `GSPC_LogReturn`（映射自 ^GSPC）
- **技术指标**：`LogReturn`, `Volatility_20`, `RSI_14`, `MACD_12_26_9`/`MACDh_12_26_9`/`MACDs_12_26_9`, `SMA_5`, `SMA_20`, `ATR_14`, `EMA_12`/`EMA_26`, `OBV`
- **布林带**：`BBL_20_2.0_2.0`/`BBM_20_2.0_2.0`/`BBU_20_2.0_2.0`/`BBB_20_2.0_2.0`/`BBP_20_2.0_2.0`
- **衍生指标**：`Intraday_Range`, `Trend_Strength`, `Candle_Body`
- **因子**：`Beta_60`, `Alpha_60`, `Sharpe_60`（60日滚动，Sharpe 年化 ×√252）
- **时间**：`Month_Sin`（月份正弦编码）
- **情感**：`Sentiment_Score`（FinBERT 模型，Score = P(positive) - P(negative)，默认 batch=16, maxlen=256）
- 输出：更新 `datas/stock_datas.csv`（添加所有特征列）

### 4. `clean_stock_data.py` - 数据清洗
- 填充缺失：`Sentiment_Score` → 0，`NewsTitles` → ""
- 删除滚动窗口产生的首段 NaN 行（按 Ticker）
- 异常值盖帽：收益/比例类列按 Ticker 做 1%/99% 分位数裁剪
- 输出：更新 `datas/stock_datas.csv`（清洗后的完整数据）

### 5. `clean_stock_data_vif.py` - VIF 特征筛选（可选）
- 从 `datas/stock_datas.csv` 筛选 VIF 特征集合
- 保留列：`Date`, `Ticker`, `NewsTitles` + 18 个 VIF 特征
- 输出：`datas/stock_datas_vif.csv`（精简版，用于模型训练）

## 数据文件说明

- **`datas/stock_datas.csv`**：清洗后的完整数据集（所有特征列）
- **`datas/stock_datas_vif.csv`**：VIF 筛选后的精简数据集（21 列）

## 运行顺序

```bash
python get_data.py 
python merge_news_into_stock_data.py
python feature_engineering.py
python clean_stock_data.py
python clean_stock_data_vif.py
```