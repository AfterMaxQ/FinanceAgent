# 05 · 量化工具箱（7 Tools）

> 7 个本地工具的算法细节、参数、返回结构、容错策略。
> 所有工具都遵循：**接受 pd.DataFrame → 内部计算 → 返回 Dict 字典**。

---

## 1. 总览

| 工具 | 类 | 算法 | 关键依赖 | 容错 |
|---|---|---|---|---|
| FinBERT 情感 | `FinBERTAnalyzer` | yiyanghkust/finbert-tone 三分类 | torch, transformers | 单例 + 懒加载 |
| T+1 预测 | `HybridPredictor` | HybridModel (LSTM+Transformer) | torch | 数据不足返回降级值 |
| 支撑阻力 | `SupportResistanceScanner` | KDE + 峰值检测 | scipy.stats.gaussian_kde | 全部 try/except |
| 形态匹配 | `SimilaritySearcher` | Z-Score 标准化 + 滑动窗口 | scipy.spatial.distance | 静默跳过无效片段 |
| 宏观环境 | `MacroAnalyzer` | SMA20/50/200 + VIX 阈值 | pandas, numpy | mtime 缓存 |
| 财报分析 | `FinancialStatementAnalyzer` | SEC EDGAR XBRL + 比率计算 | requests | 3 次重试 + 指数退避 |
| 工具注册 | (模块) | DeepSeek function schema | — | — |

---

## 2. FinBERTAnalyzer — 情感分析

文件：[`agents/Tools/finbert_analyzer.py`](../../agents/Tools/finbert_analyzer.py)

### 2.1 模型选择

- **仓库**：`yiyanghkust/finbert-tone`
- **任务**：金融文本三分类 (Positive / Negative / Neutral)
- **本地路径**：`models/yiyanghkust_finbert-tone/`

### 2.2 单例 + 懒加载

```python
class FinBERTAnalyzer:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
```

```python
def _find_local_model_path():
    """查找包含 vocab.txt + model.safetensors 的本地快照"""
    if LOCAL_MODEL_DIR.exists():
        candidates.append(LOCAL_MODEL_DIR)
    if SNAPSHOT_ROOT.exists():
        candidates.extend([p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir()])
    for path in candidates:
        if (path / "vocab.txt").exists() and (model_file).exists():
            return path
    return None
```

加载策略：本地优先 → 远程 HuggingFace Hub 兜底。

### 2.3 公开方法

#### `classify_texts(texts: List[str]) -> List[Dict]`

```python
inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
outputs = model(**inputs)
probs = softmax(outputs.logits, dim=-1)
results.append({"label": id2label[argmax], "scores": {id2label[j]: prob_j}})
```

返回示例：

```json
[{"label": "Positive", "scores": {"Positive": 0.82, "Negative": 0.05, "Neutral": 0.13}}]
```

#### `calc_score(texts: List[str]) -> float`

把三分类映射成连续分数：`Positive → 1, Negative → -1, Neutral → 0`，然后求平均。

```python
score_map = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0}
return total_score / valid_count   # ∈ [-1, 1]
```

#### `get_news_and_sentiment(df, end_date)`（兼容旧名）

> 在 `decision_agent._execute_tool` 中保留旧 API 路径；当前主要走 `calc_score`/`classify_texts`。

### 2.4 边界

- `texts` 为空或全是非字符串 → 返回 `[]` 或 `0.0`
- 模型推理失败 → `warnings.warn` 并返回 `[{error: ...}] * len(texts)`
- token 长度截断到 256（`MAX_LENGTH`）

---

## 3. HybridPredictor — T+1 价格预测

文件：[`agents/Tools/predict_t1.py`](../../agents/Tools/predict_t1.py)

详见 [03-hybrid-model.md](./03-hybrid-model.md)。这里只列出工具侧关键点。

### 3.1 关键常量

```python
VIF_FEATURES = [
    "Volume", "GSPC_Close", "GSPC_LogReturn", "LogReturn",
    "MACDh_12_26_9", "MACDs_12_26_9", "Intraday_Range",
    "Trend_Strength", "Candle_Body", "Month_Sin", "Sentiment_Score",
    "BBB_20_2.0_2.0", "BBP_20_2.0_2.0", "ATR_14", "OBV",
    "Beta_60", "Alpha_60", "Sharpe_60"
]
```

**必须与训练时的特征列表完全一致**，否则 `RobustScaler` 会因列错位产生 nan。

### 3.2 推理流水线

```python
def predict(self, simulation_data, current_price, news_column="NewsTitles"):
    seq_len = 30
    min_required = seq_len + 40
    if len(simulation_data) < min_required:
        return 降级结果    # predicted_price = current_price * 0.965

    df = simulation_data.tail(min_required).copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df[news_column] = df.get(news_column, "No significant news").fillna(...)

    scalar_tensor, text_tensor = self.prepare_tensors(df, news_column)
    direction_prob, pred_magnitude = predict_next_day(
        model, scalar_tensor, text_tensor, scaler_dict, self.device
    )
    pred_magnitude = abs(pred_magnitude)
    final_log_return = (1 if direction_prob >= 0.5 else -1) * pred_magnitude
    predicted_price = current_price * np.exp(final_log_return)
    return {
        "predicted_price": float(predicted_price),
        "predicted_change_pct": float((predicted_price - current_price) / current_price * 100),
        "direction_prob": float(direction_prob),
        "pred_magnitude": float(pred_magnitude),
        "final_log_return": float(final_log_return),
        "error": None,
    }
```

### 3.3 返回字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `predicted_price` | float | 预测 T+1 收盘价 |
| `predicted_change_pct` | float | 百分比变化 |
| `direction_prob` | float ∈ [0,1] | 上涨概率（>= 0.5 即认为上涨） |
| `pred_magnitude` | float ≥ 0 | 真实尺度波动幅度（log return） |
| `final_log_return` | float | `± magnitude` |
| `error` | str / None | 错误信息（成功时为 None） |

### 3.4 容错

- **数据不足** → 返回 `current_price * 0.965` 并附带 `"历史数据不足"` 错误。
- **任何异常** → `logger.error` + 返回降级字典，error 字段填 trace。

---

## 4. SupportResistanceScanner — KDE 支撑/阻力位

文件：[`agents/Tools/support_resistance.py`](../../agents/Tools/support_resistance.py)

### 4.1 算法

```
1. 取最近 window 天（默认 252）的价格
2. 计算加权价格 p = (H + L + C) / 3
3. 拟合 scipy.stats.gaussian_kde(p) — 高斯核密度估计
4. 在 [p_min - 0.1·range, p_max + 0.1·range] 上采 1000 点求密度
5. find_peaks(density, prominence=max_density*0.05, distance=20) 找峰
6. 按 current_price 分为 supports（<）和 resistances（>）
7. 排序后选 nearest
```

### 4.2 KDE 公式

$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)
$$

高斯核 $K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$，带宽 $h$ 默认 Scott 规则。

### 4.3 公开方法 `calculate_levels`

```python
def calculate_levels(
    self, df, current_price,
    window=252, use_weighted=True, bandwidth=None
) -> Dict:
    return {
        "supports": [{"price": 148.0, "strength": 0.73}, ...],
        "resistances": [{"price": 175.5, "strength": 0.61}, ...],
        "nearest_support": {...} | None,
        "nearest_resistance": {...} | None,
        "status": "价格位于强支撑位上方 0.87% 处 | 最近阻力位: 175.50 (距离 4.21%)"
    }
```

### 4.4 容错

- 缺 `Close`/`High`/`Low` 列 → 返回默认空字典
- 数据 < 10 行 → 状态 `"数据长度不足"`
- 找不到峰值（分布过于平滑）→ 状态 `"未找到支撑位或阻力位"`
- 任何未捕获异常 → `logger.error` + 返回 default_result

---

## 5. SimilaritySearcher — 历史形态匹配

文件：[`agents/Tools/similarity_search.py`](../../agents/Tools/similarity_search.py)

### 5.1 算法

```
1. 目标序列：df['Close'].tail(query_window)
2. Z-Score 标准化：(x - mean) / std
3. 历史序列：除最后 query_window 天外的所有滑动窗口
4. 对每窗口同样 Z-Score 标准化
5. 计算相似度（欧氏/余弦），转换为 [0, 100] 分
6. 排序取 top_k
7. 对每个匹配片段计算后续 N 天涨跌幅
```

### 5.2 相似度计算

#### 欧氏距离 → 相似度

```python
distance = euclidean(seq1, seq2)
scale = max(1.0, std(seq1) + std(seq2))
similarity = 100 * exp(-distance / scale)   # ∈ [0, 100]
```

#### 余弦距离 → 相似度

```python
distance = cosine(seq1, seq2)                # ∈ [0, 2]
similarity = 100 * (1 - distance / 2)        # ∈ [0, 100]
```

### 5.3 构造参数

```python
SimilaritySearcher(similarity_method='euclidean')   # 或 'cosine'
```

`decision_agent._execute_tool` 会从 LLM 参数中取出 `similarity_method` 并校验白名单。

### 5.4 公开方法 `search_similar_periods`

```python
def search_similar_periods(
    self, df, query_window=20, top_k=5, subsequent_days=5,
    similarity_method=None
) -> List[Dict]:
    return [
        {
            "end_index": 1234,
            "date": "2015-08-12",       # 可选，若 df 有 Date 列
            "similarity_score": 87.3,
            "subsequent_return": 2.34,  # 百分比
            "visualization_data": {
                "target": [...], "matched": [...],
                "target_raw": [...], "matched_raw": [...]
            }
        },
        ...
    ]
```

### 5.5 容错

- 数据不足 (`< query_window + subsequent_days`) → `[]`
- 标准差为 0（横盘） → 跳过该窗口
- 任何异常 → `logger.error` + 返回 `[]`

---

## 6. MacroAnalyzer — 宏观环境

文件：[`agents/Tools/macro_analyzer.py`](../../agents/Tools/macro_analyzer.py)

### 6.1 数据源

直接读 `datas/stock_datas.csv`（其中包含 `^GSPC, ^VIX, ^TNX` 三只宏观指数）。

```python
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "datas" / "stock_datas.csv"
```

### 6.2 缓存机制

```python
def _load_data_cached(self):
    mtime = self.data_path.stat().st_mtime
    if self._cache_df is not None and self._cache_mtime == mtime:
        return self._cache_df
    df = pd.read_csv(self.data_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["Date"])
    self._cache_df, self._cache_mtime = df, mtime
    return df
```

基于文件 mtime 失效，避免每次都重读 80k+ 行 CSV。

### 6.3 三类指标

#### GSPC (S&P 500)

```python
gspc = df[df["Ticker"] == "^GSPC"][["Date", "Close"]]
gspc["SMA20"] = gspc["Close"].rolling(20, min_periods=5).mean()
gspc["SMA50"] = gspc["Close"].rolling(50).mean()
gspc["SMA200"] = gspc["Close"].rolling(200).mean()
```

输出 `close, sma20, sma50, sma200, dist_to_sma20_pct, dist_to_sma200_pct, market_phase`。

**market_phase 规则**：

| 条件 | 阶段 |
|---|---|
| `close > sma200` | Bull Market |
| `close > sma50`（无 sma200） | Recovery/Neutral |
| 其他 | Bear Market |

#### VIX（波动率指数）

```python
if close >= 30: regime = "High Fear"
elif close >= 20: regime = "Elevated"
else: regime = "Calm"
```

输出 `close, pct_change_1d, regime`。

#### TNX（10 年美债收益率）

输出 `close, sma50, trend`（Uptrend if `close >= sma50` else Downtrend）。

### 6.4 综合 risk_regime

```python
if g_phase == "Bull Market" and vix_level < 20:
    risk_regime = "Risk-On"
elif g_phase == "Bear Market" or vix_level >= 25:
    risk_regime = "Risk-Off"
else:
    risk_regime = "Neutral"
```

### 6.5 公开方法 `analyze_market_regime(current_date)`

```python
return {
    "date_requested": "2018-06-01",
    "gspc": {...},
    "vix": {...},
    "tnx": {...},
    "risk_regime": "Risk-On",
    "rationale": "VIX=12.34; GSPC=Bull Market"
}
```

参数 `current_date` 是 "模拟日期"，用于在历史数据上做"假如我回到过去"的回放。

### 6.6 兼容性

```python
MacroRegimeAnalyzer = MacroAnalyzer  # 别名
```

---

## 7. FinancialStatementAnalyzer — 财报分析

文件：[`agents/Tools/financial_statement_analyzer.py`](../../agents/Tools/financial_statement_analyzer.py)

### 7.1 数据源

**SEC EDGAR XBRL API**（无需鉴权，仅需 User-Agent）：

- `https://www.sec.gov/include/ticker.txt` — Ticker → CIK 映射
- `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` — 公司全部财务事实

### 7.2 User-Agent（合规）

```python
USER_AGENT = "PersonalQuantProject/zhang.wei.123@email.com"
```

SEC 要求请求必须带可识别的 UA，否则会被限流。

### 7.3 重试与限流

```python
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2  # 秒

for attempt in range(1, MAX_RETRIES + 1):
    try:
        response = requests.get(url, headers=..., timeout=15)
        response.raise_for_status()
        time.sleep(1)   # 严格遵守 SEC 1 req/s
        return response.json()
    except:
        time.sleep(RETRY_WAIT_BASE * attempt)   # 2s, 4s, 6s 指数退避
```

### 7.4 抽取 14 个 XBRL 指标

| 报表 | 指标（us-gaap 概念） |
|---|---|
| 利润表 | Revenues, CostOfRevenue, GrossProfit, OperatingIncomeLoss, NetIncomeLoss |
| 资产负债表 | Assets, AssetsCurrent, Liabilities, LiabilitiesCurrent, StockholdersEquity |
| 现金流量表 | NetCashProvidedByUsedInOperatingActivities, NetCashProvidedByUsedInInvestingActivities, NetCashProvidedByUsedInInvestingActivities, PaymentsToAcquirePropertyPlantAndEquipment |

`_get_latest_annual_fact_value` 只取 `form in ["10-K", "10-K/A"]` 中 `end` 最大的记录。

### 7.5 计算的比率

| 比率 | 公式 | 意义 |
|---|---|---|
| `gross_margin_percent` | `gross_profit / revenue * 100` | 毛利空间 |
| `net_profit_margin_percent` | `net_income / revenue * 100` | 净利率 |
| `current_ratio` | `current_assets / current_liabilities` | 短期偿债 |
| `debt_to_equity_ratio` | `total_liabilities / equity` | 财务杠杆 |
| `free_cash_flow` | `ocf + capex` (capex 已取负) | 自由现金流 |
| `ocf_to_net_income_ratio` | `ocf / ni` | 盈利质量 |

### 7.6 定性 take 与综合评级

| 维度 | 评级规则 |
|---|---|
| 盈利能力 | `net_margin > 20%` → 极强；`> 10%` → 良好；`> 0` → 一般；否则亏损 |
| 财务健康 | `current_ratio > 2` → 非常强；`> 1.2` → 良好；否则有压力 + `debt_to_equity` 评估 |
| 现金流 | `ocf>0 && fcf>0` → 强大 + `ocf_to_ni` 评估 |

**综合 score**：

```
score = 0
if "极强" or "良好" in income_take:  +1
if "亏损" in income_take:            -1
if "稳健" or "良好" in balance_take: +1
if "风险" or "压力" in balance_take: -1
if "强大" or "健康" in cashflow_take: +1
if "危险信号" in cashflow_take:      -1

if score >= 3: rating = "Excellent (优秀)"
elif score >= 1: rating = "Solid (扎实)"
elif score == 0: rating = "Neutral (中性)"
else:            rating = "Caution Needed (需警惕)"
```

### 7.7 公开方法 `analyze_latest_filings(ticker)`

返回结构：

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "report_period": "2024-09-28",
  "income_statement_analysis": {
    "title": "利润表分析 (盈利能力)",
    "key_metrics": {"revenue": 391000000000, ...},
    "ratios_and_analysis": {
      "gross_margin_percent": 45.3, "net_profit_margin_percent": 25.1,
      "analyst_take": "盈利能力极强，净利率非常高，护城河显著。"
    }
  },
  "balance_sheet_analysis": {...},
  "cash_flow_analysis": {...},
  "overall_summary": {
    "title": "综合评估",
    "long_term_value_rating": "Excellent (优秀)",
    "summary_text": "盈利能力: ... 财务健康: ... 现金流: ..."
  }
}
```

### 7.8 容错

| 异常 | 处理 |
|---|---|
| 503 (Service Unavailable) | 返回友好提示，建议稍后重试或手动查 EDGAR |
| 404 (Not Found) | 提示 ticker 错误或公司未在 SEC 注册 |
| 网络错误 | 返回通用错误信息 |
| 缺核心字段 (Revenues) | 返回 `{"error": "无法从SEC获取最核心的年度'营业收入'数据"}` |
| 比率分母为 0 | 使用 `0` 或 `inf` 兜底 |

---

## 8. 工具注册表 `tool_registry.py`

文件：[`agents/Tools/tool_registry.py`](../../agents/Tools/tool_registry.py)

```python
tools: List[Dict] = [...]  # 7 个 DeepSeek function schema
TOOL_MAPPING: Dict[str, Type] = {
    "calc_score": FinBERTAnalyzer,
    "classify_texts": FinBERTAnalyzer,
    "predict": HybridPredictor,
    "calculate_levels": SupportResistanceScanner,
    "search_similar_periods": SimilaritySearcher,
    "analyze_market_regime": MacroAnalyzer,
    "analyze_financial_statements": FinancialStatementAnalyzer,
}
TOOL_INIT_PARAMS: Dict[str, Dict] = {
    "search_similar_periods": {},  # 占位
}

def get_tool_by_name(name) -> Dict: ...
def get_tool_class(name) -> Type:  # 拼写错误时用 difflib 提示相似名
def get_all_tool_names() -> List[str]: ...
```

### 8.1 添加工具的标准流程

1. 在 `agents/Tools/` 新建 `xxx.py`，实现 `XxxAnalyzer` 类（含主方法）。
2. 在 `tool_registry.py` 中：
   - 添加 schema 到 `tools` 列表
   - 在 `TOOL_MAPPING` 中加入 `name -> XxxAnalyzer` 映射
3. 若该工具有特殊构造参数，加到 `TOOL_INIT_PARAMS`，并在 `decision_agent._execute_tool` 中处理。
4. 更新 SOP 提示词（可选，让 LLM 知道新工具）。
5. 在前端加入"快捷卡片"（可选）。

---

## 9. 工具调用性能参考

| 工具 | 冷启动 | 热调用 | 备注 |
|---|---|---|---|
| FinBERT | ~5s（首次加载 700MB） | < 100ms / 16 条 | 单例共享 |
| Hybrid 预测 | ~2s（首次加载 .pt） | < 200ms | 每次新建实例 |
| KDE | < 50ms | < 50ms | scipy 缓存 |
| 形态匹配 | < 500ms（2000 窗口） | < 500ms | 与数据量线性 |
| Macro | < 100ms（缓存命中） | < 100ms | mtime 失效 |
| SEC 财报 | 2-5s | 2-5s | 网络依赖 + 1s 限流 |

> 单次完整分析（4-5 工具）约 5-10s，主要耗时在 FinBERT 加载与 SEC 请求。
