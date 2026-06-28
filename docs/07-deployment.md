# 07 · 部署与运行（Deployment Guide）

> 从零到上线：环境准备 → 数据准备 → 模型训练 → 启动应用 → 故障排查。

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11, macOS, Linux（代码已用 `Path` 跨平台） |
| Python | 3.10+（`f-string` 与 `match` 语句需要） |
| 内存 | ≥ 8 GB RAM（FinBERT 加载约 700MB，Hybrid 约 50MB） |
| GPU | 可选；CUDA 加速 Hybrid 训练（默认自动检测） |
| 磁盘 | ≥ 5 GB（数据 + 模型 + 缓存） |
| 网络 | 必需：Yahoo Finance、SEC EDGAR；可选：HuggingFace Hub |

---

## 2. 环境搭建

### 2.1 虚拟环境

```bash
# 创建 venv
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 关键版本：

```
numpy==1.26.4
pandas==2.3.3
pandas-ta
plotly
scikit-learn
scipy
statsmodels
streamlit
torch==2.7.0
transformers==4.51.3
yfinance==0.2.66
```

`openai` SDK **未在 requirements.txt**，但运行 AI 对话必需：

```bash
pip install openai
```

> 自动重生成：
> ```bash
> python generate_requirements.py
> ```

### 2.3 代理（仅 Windows 国内网络）

`get_data.py` 默认设置代理 `http://127.0.0.1:7897`。若无代理，**注释掉** 顶部两行：

```python
# proxy = 'http://127.0.0.1:7897'
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy
```

---

## 3. 数据准备（首次运行必做）

### 3.1 下载 FinBERT 权重

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('yiyanghkust/finbert-tone', local_dir='models/yiyanghkust_finbert-tone')
"
```

预期产出：

```
models/yiyanghkust_finbert-tone/
├── config.json
├── model.safetensors       (~700MB)
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

> 国内网络可能需挂代理或使用镜像：
> ```python
> snapshot_download(..., local_dir='...', endpoint='https://hf-mirror.com')
> ```

### 3.2 准备 stock_statistics.csv

`datas/stock_statistics.csv` 含两列：`Stock, Percentage`，由用户从 Kaggle 等渠道获取。

默认项目仓库应在 `datas/` 目录预先放置此文件。

### 3.3 准备 analyst_ratings_processed.csv

同上，列：`title, date, stock`（分析师评级新闻）。

### 3.4 依次运行 6 个数据管道

```bash
# ① 下载行情
python get_data.py --start 2009-01-01 --end 2019-12-31

# ② 合并新闻
python merge_news_into_stock_data.py

# ③ 特征工程
python feature_engineering.py

# ④ 数据清洗
python clean_stock_data.py

# ⑤ VIF 降维
python vif_feature_selection.py --threshold 5.0

# ⑥ VIF 特征子集输出
python clean_stock_data_vif.py
```

每一步会自动备份上版为 `.bak`（`--no-backup` 可关闭）。

### 3.5 数据自检

```python
import pandas as pd

df = pd.read_csv("datas/stock_datas.csv")
print("Shape:", df.shape)            # (80k+, 30+)
print("Tickers:", df['Ticker'].nunique())  # ~30 + 3
print("Date range:", df['Date'].min(), "~", df['Date'].max())

vif_df = pd.read_csv("datas/stock_datas_vif.csv")
print("VIF shape:", vif_df.shape)     # 80k+, 21
```

---

## 4. 模型训练

### 4.1 启动训练

```bash
python models/hybrid_model/hybrid_model_trainer.py \
  --csv-path datas/stock_datas_vif.csv \
  --finbert-dir models/yiyanghkust_finbert-tone \
  --output-dir models/hybrid_model/checkpoints
```

### 4.2 关键参数

| 参数 | 默认 | 建议 |
|---|---|---|
| `--sequence-length` | 30 | 30~60 都可 |
| `--batch-size` | 128 | GPU 显存 < 8GB → 64 |
| `--epochs` | 50 | 数据量大可 30 |
| `--lr` | 5e-5 | AdamW |
| `--weight-decay` | 1e-4 | — |
| `--huber-delta` | 1.0 | — |
| `--patience` | 15 | 防止过拟合 |
| `--train-ratio` | 0.8 | 时间顺序切分 |
| `--device` | auto | `cuda` / `cpu` |

### 4.3 训练输出

```
models/hybrid_model/checkpoints/
├── hybrid_model.pt        # 约 50MB
└── hybrid_scalers.npz     # 三个 RobustScaler
```

训练日志示例：

```
Epoch 001 | Train Loss: 0.6823 (DA 0.5231, RMSE 0.0421) | Val Loss: 0.6401 (DA 0.5312, DA Loss 0.6511, RMSE 0.0412)
Epoch 002 | Train Loss: 0.6512 (DA 0.5412, RMSE 0.0398) | Val Loss: 0.6102 (DA 0.5512, DA Loss 0.6122, RMSE 0.0388)
...
早停触发（patience=15），最佳验证 Val Loss 0.5432
训练完成。
```

### 4.4 训练耗时参考

| 配置 | 数据量 | 耗时 |
|---|---|---|
| CPU (8 核) | 30 支 × 11 年 | 6-10 小时 |
| GPU (RTX 3060) | 30 支 × 11 年 | 30-60 分钟 |
| GPU (RTX 4090) | 30 支 × 11 年 | 15-30 分钟 |

> 训练时 FinBERT 重新跑 768 维嵌入，这一步最耗时。

---

## 5. 启动应用

### 5.1 标准启动

```bash
streamlit run app.py
```

或：

```bash
streamlit run frontend/app.py
```

浏览器自动打开 `http://localhost:8501`。

### 5.2 自定义端口 / 远程访问

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

### 5.3 第一次进入

1. 左侧填写 DeepSeek API Key（可选）
2. 选择股票代码 / 日期范围 / 指标
3. 点击"开始分析"
4. 在对话框输入问题或点击快捷卡片

---

## 6. 故障排查（Troubleshooting）

### 6.1 模型文件缺失

```
❌ Hybrid 模型文件未找到: models\hybrid_model\checkpoints\hybrid_model.pt
```

**解决**：运行第 4 节训练流程。

### 6.2 FinBERT 找不到

```
ERROR [FinBERTAnalyzer]: 模型加载失败: ...
```

**解决**：检查 `models/yiyanghkust_finbert-tone/` 目录是否齐全；尝试重新 `snapshot_download`。

### 6.3 DeepSeek 401 / 403

```
OpenAI API Error: 401 Unauthorized
```

**解决**：检查 API Key 是否正确；以 `sk-` 开头且长度 > 20。

### 6.4 SEC EDGAR 503

```
{"error": "SEC API服务暂时不可用（503错误）..."}
```

**解决**：稍后重试；或直接到 [EDGAR Search](https://www.sec.gov/edgar/searchedgar/companysearch.html) 手动查询。

### 6.5 Yahoo Finance rate limit

```
Ticker AAPL attempt 1/4 failed (rate limited: ...); sleeping 0.1s
```

**解决**：`--retries 8 --retry-wait 1.0` 增加退避；或开启代理。

### 6.6 Streamlit `set_page_config` 错误

```
StreamlitAPIException: set_page_config() can only be called once per app
```

**解决**：确认 `setup_page_style()` 只在文件顶部调用一次。

### 6.7 CUDA 不可用但选用了 `cuda`

```
RuntimeError: CUDA 不可用，但指定了 cuda 设备
```

**解决**：训练时去掉 `--device cuda` 参数，让其自动检测；或安装匹配 PyTorch 版本的 CUDA 驱动。

---

## 7. 部署到服务器

### 7.1 Systemd 服务（Linux）

```ini
# /etc/systemd/system/quantagent.service
[Unit]
Description=QuantAgent Streamlit
After=network.target

[Service]
User=quant
WorkingDirectory=/opt/FinanceAgentV2.0
ExecStart=/opt/FinanceAgentV2.0/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
Environment="PYTHONPATH=/opt/FinanceAgentV2.0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now quantagent
sudo systemctl status quantagent
```

### 7.2 Docker

`Dockerfile` 模板：

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 下载 FinBERT（在构建期）
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('yiyanghkust/finbert-tone', local_dir='models/yiyanghkust_finbert-tone')"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建与运行：

```bash
docker build -t quantagent .
docker run -p 8501:8501 -v $(pwd)/datas:/app/datas quantagent
```

> 数据卷挂载 `datas/` 与 `models/hybrid_model/checkpoints/` 可避免每次重建都重新训练。

### 7.3 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name quant.example.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

---

## 8. 监控与日志

| 日志来源 | 路径 / 输出 |
|---|---|
| 数据管道 | stdout（`print`） |
| HybridPredictor / SimilaritySearcher 等 | `logging` 默认 INFO |
| Streamlit | stderr + `~/.streamlit/logs/` |
| DeepSeek 调用 | 由 OpenAI SDK 默认日志 |

可在启动前 export 环境变量调节日志等级：

```bash
export STREAMLIT_LOG_LEVEL=info
export PYTHONUNBUFFERED=1   # 实时刷新 print
```

---

## 9. 性能优化建议

| 优化点 | 做法 |
|---|---|
| FinBERT 加载慢 | 启动时后台预热；用 `device_map='auto'` 拆分 GPU/CPU |
| Hybrid 推理慢 | 关闭梯度 `torch.no_grad()`（已实现）+ 半精度 `model.half()` |
| Streamlit 重绘卡 | 改用 `st.fragment` 局部刷新 |
| 大 CSV 加载慢 | 转 `parquet` 格式 + `pyarrow` 引擎 |
| Yahoo 限流 | 多账号轮询 / 改用付费 API |

---

## 10. CI/CD 检查清单

部署前自动验证：

- [ ] `datas/stock_datas.csv` 行数 > 10000
- [ ] `datas/stock_datas_vif.csv` 列数 == 21
- [ ] `models/hybrid_model/checkpoints/hybrid_model.pt` 存在
- [ ] `models/yiyanghkust_finbert-tone/model.safetensors` 存在
- [ ] `python -c "from agents.Tools.tool_registry import tools; assert len(tools)==7"`
- [ ] `python -c "from models.hybrid_model.hybrid_model import load_hybrid_model; m,_ = load_hybrid_model('models/hybrid_model/checkpoints/hybrid_model.pt')"`
- [ ] `python -c "from agents.decision_agent import FinancialAgent"` 不抛错

---

## 11. 备份策略

| 数据 | 建议 |
|---|---|
| `datas/*.csv` | 每天 cron 备份到对象存储；`.bak` 自动保留 |
| `models/hybrid_model/checkpoints/` | 训练后归档；用 DVC / git-lfs |
| `models/yiyanghkust_finbert-tone/` | 首次下载后归档；约 700MB |
| DeepSeek prompts | 版本化到 git；可放 `prompts/` 目录 |

---

## 12. 安全注意事项

1. **不要把 API Key 硬编码到代码或 git**。前端用 `st.text_input` 接收；后端可读环境变量 `DEEPSEEK_API_KEY`。
2. **SEC User-Agent 必须是真实邮箱**，否则会被 IP 封禁。
3. **Yahoo Finance 数据有版权**，仅供研究/学习，不要用于商业再分发。
4. **所有预测仅供分析参考**，不构成投资建议（已在 SOP 提示词中明确禁止直接投资建议）。
