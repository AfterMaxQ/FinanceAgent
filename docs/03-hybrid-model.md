# 03 · 混合深度学习模型 (HybridModel)

> 双塔（LSTM + Transformer）+ 交叉注意力融合 + 双头（方向 + 幅度）多任务预测。
> 输入：18 维 VIF 时序特征 + 768 维 FinBERT 新闻嵌入。
> 输出：T+1 上涨概率 + 真实尺度波动幅度。

---

## 1. 架构总览

```
                       ┌─────────────────────────────────────┐
                       │     输入 (batch, seq_len=30)         │
                       └──────────────┬──────────────────────┘
                                      │
              ┌───────────────────────┴────────────────────────┐
              │                                                │
              ▼                                                ▼
   ┌──────────────────────┐                       ┌──────────────────────────┐
   │   TemporalTower      │                       │       TextTower          │
   │  (LSTM 主体)          │                       │  (Transformer 主体)       │
   │                      │                       │                          │
   │ Input: 18 维         │                       │ Input: 768 维 FinBERT    │
   │ ├ LSTM(18→256, 2层)   │                       │ ├ Linear(768→768)        │
   │ ├ LayerNorm(256)      │                       │ ├ + Learnable PosEnc     │
   │ └ Dropout(0.2)        │                       │ ├ 4×TransformerEncoder   │
   │                      │                       │ │  (768d, 8 heads, GELU) │
   │ Output: h_t (256,)    │                       │ ├ norm_first=True        │
   └──────────┬───────────┘                       │ └ Dropout(0.3)           │
              │                                   │                          │
              │                                   │ Output: H_text (B,L,768) │
              │                                   └────────────┬─────────────┘
              │                                                │
              │ 256                                            │ 768
              ▼                                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │                    FusionLayer                          │
        │  Cross-Attention(query=Linear(h_t), key/value=H_text)   │
        │  Gate = σ(MLP([q, attn_out]))                           │
        │  fused = gate * q + (1-gate) * attn_out                 │
        │  + Residual + LayerNorm + Dropout                       │
        │  → h_fused (768,)                                        │
        └─────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │   Shared FC           │
                      │  Linear(768→384)      │
                      │  ReLU + Dropout(0.2)  │
                      └───────────┬───────────┘
                                  │ 384
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
        ┌───────────────────┐         ┌────────────────────┐
        │  ClsHead          │         │  RegHead           │
        │  Linear(384→1)    │         │  Linear(384→1)     │
        │  Sigmoid          │         │  无激活（实数）      │
        │  → P(Up) ∈ [0,1]  │         │  → Magnitude       │
        └───────────────────┘         └────────────────────┘
```

---

## 2. 核心模块源码解读

文件：[`models/hybrid_model/hybrid_model.py`](../../models/hybrid_model/hybrid_model.py)

### 2.1 TemporalTower — LSTM 时序塔

```python
class TemporalTower(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                  # x: (B, L, 18)
        _, (h_n, _) = self.lstm(x)         # h_n: (num_layers, B, 256)
        return self.dropout(self.ln(h_n[-1]))   # 取最后一层 (B, 256)
```

**设计要点**：

- 只取 LSTM **最后时间步的隐状态**（`h_n[-1]`），相当于对 30 日窗口的"摘要向量"。
- `LayerNorm` 稳定训练，`Dropout` 防过拟合。
- 2 层足够捕获短期 + 中期依赖，更深容易过拟合。

### 2.2 TextTower — Transformer 文本塔

```python
class TextTower(nn.Module):
    def __init__(self, input_dim, hidden_size=768, num_layers=4,
                 num_heads=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)  # 保险，可恒等
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size*4,
            dropout, 'gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                # x: (B, L, 768)
        x = self.input_proj(x)           # (B, L, 768)
        if x.size(1) <= self.pos_enc.size(1):
            x = x + self.pos_enc[:, :x.size(1), :]
        else:
            # 用线性插值扩展位置编码到任意长度
            x = x + F.interpolate(
                self.pos_enc.transpose(1, 2), size=x.size(1),
                mode='linear', align_corners=False
            ).transpose(1, 2)
        return self.dropout(self.transformer(x))   # (B, L, 768)
```

**设计要点**：

- **位置编码** 是可学习的（`nn.Parameter`），覆盖至 1000 步；超长序列走线性插值。
- **`norm_first=True`**（Pre-LN）—— 现代 Transformer 默认做法，训练更稳定。
- 4 层、8 头、FFN 维度 3072（4×768），GELU 激活。
- 输出保留**完整序列** `(B, L, 768)`，交给 FusionLayer 做 Cross-Attention。

### 2.3 FusionLayer — 交叉注意力 + 门控

```python
class FusionLayer(nn.Module):
    def __init__(self, temporal_dim, text_dim, dropout=0.2):
        super().__init__()
        self.query_proj = nn.Linear(temporal_dim, text_dim)  # 256 → 768
        self.cross_attn = nn.MultiheadAttention(
            text_dim, 8, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(text_dim*2, text_dim), nn.ReLU(),
            nn.Linear(text_dim, 1), nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temp, text):                # temp: (B,256); text: (B,L,768)
        temp_q = self.query_proj(temp).unsqueeze(1)   # (B,1,768) — query
        attn_out, _ = self.cross_attn(temp_q, text, text)  # (B,1,768)
        attn_out = attn_out.squeeze(1)              # (B,768)
        temp_aligned = temp_q.squeeze(1)            # (B,768)
        gate_val = self.gate(torch.cat([temp_aligned, attn_out], dim=1))  # (B,1)
        fused = gate_val * temp_aligned + (1 - gate_val) * attn_out
        return self.dropout(self.ln(fused + temp_aligned))  # 残差 + LN
```

**设计要点**：

- **Query = LSTM 摘要**；**Key/Value = Transformer 序列**—— 让时序信号去"查询"相关文本片段。
- **门控 (Gating)** 决定信息来自时序还是文本：$g \in [0,1]$ 由 `[q, attn]` 经 MLP → Sigmoid 得出。
- 残差 + LayerNorm 保持训练稳定。

### 2.4 HybridModel — 双头

```python
class HybridModel(nn.Module):
    def __init__(self, scalar_input_dim, text_input_dim,
                 temporal_hidden_size=256, temporal_num_layers=2,
                 text_hidden_size=768, text_num_layers=4,
                 text_num_heads=8, ...):
        super().__init__()
        self.temporal_tower = TemporalTower(...)
        self.text_tower = TextTower(...)
        self.fusion_layer = FusionLayer(...)

        self.shared_fc = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size // 2),
            nn.ReLU(), nn.Dropout(fusion_dropout)
        )
        # 头1：分类（方向）
        self.cls_head = nn.Sequential(nn.Linear(384, 1), nn.Sigmoid())
        # 头2：回归（幅度）
        self.reg_head = nn.Linear(384, 1)

    def forward(self, scalar_seq, text_seq):
        temp_emb = self.temporal_tower(scalar_seq)
        text_emb = self.text_tower(text_seq)
        fused_emb = self.fusion_layer(temp_emb, text_emb)
        shared_feat = self.shared_fc(fused_emb)        # (B, 384)
        pred_prob = self.cls_head(shared_feat).squeeze(-1)  # (B,)
        pred_mag = self.reg_head(shared_feat).squeeze(-1)   # (B,)
        return pred_prob, pred_mag
```

---

## 3. 数据集与 Scaler

### 3.1 输入张量

| 名称 | 形状 | 来源 |
|---|---|---|
| `scalar_seq` | `(B, 30, 18)` | 18 维 VIF 特征 RobustScaler 后 |
| `text_seq`   | `(B, 30, 768)` | FinBERT CLS 隐状态 RobustScaler 后 |
| `target_da`  | `(B,)` | `1 if LogReturn > 0 else 0` |
| `target_mag` | `(B,)` | `LogReturn.abs().ewm(span=3).mean()`，RobustScaler 后 |

### 3.2 Scaler 持久化

```python
def save_scalers_to_npz(scaler_dict, save_path):
    np.savez_compressed(save_path,
        scalar_center=..., scalar_scale=...,
        text_center=..., text_scale=...,
        target_center=..., target_scale=...,
        sequence_length=..., scalar_feature_cols=...,
        text_feature_cols=...
    )

def load_scalers_from_npz(npz_path) -> Dict:
    # 重建 3 个 RobustScaler 及其元数据
    ...
```

文件 `hybrid_scalers.npz` 包含 **3 个 RobustScaler** 的 `center_` 与 `scale_`（标量、文本、目标）。

---

## 4. 训练流程

文件：[`models/hybrid_model/hybrid_model_trainer.py`](../../models/hybrid_model/hybrid_model_trainer.py)

### 4.1 整体流程

```
读取 stock_datas_vif.csv
       │
       ▼
prepare_features()
  · 按 Date 排序（避免未来信息泄露）
  · Target_DA = (LogReturn > 0)
  · Target_Magnitude = abs(LogReturn).ewm(span=3).mean()
       │
       ▼
build_text_embeddings()
  · 用本地 FinBERT 提取 768 维 CLS 向量
  · batch_size=32, max_length=256
       │
       ▼
build_datasets()
  · 按 8:2 时间顺序切分
  · 仅用训练部分拟合 3 个 RobustScaler
  · 构造训练/验证 Dataset
       │
       ▼
HybridModel(...)
       │
       ▼
for epoch in 1..50:
    train → BCE + Huber (各 0.5)
    val   → DA 准确率 + 反标 RMSE
    if Val Loss (0.7·BCE + 0.3·RMSE) 下降:
        save hybrid_model.pt + hybrid_scalers.npz
    else: 早停计数；连续 patience=15 无改进则停止
```

### 4.2 损失函数

```python
loss_da = BCELoss()(pred_prob, target_da)            # 方向损失
loss_mag = HuberLoss(delta=1.0)(pred_mag, target_mag) # 幅度损失
loss = 0.5 * loss_da + 0.5 * loss_mag                # 等权相加
```

- **BCELoss**：对方向预测施加对数损失，惩罚错误分类。
- **HuberLoss (delta=1.0)**：对小残差是 MSE，对大残差是 MAE，对异常值更鲁棒。
- **早停标准**：`Val Loss = 0.7 * DA_Loss + 0.3 * RMSE`，综合方向与幅度。

### 4.3 评估指标

```python
def compute_metrics(probs, targets_da, preds_mag, targets_mag, scaler):
    da_pred = (probs >= 0.5).astype(np.float32)
    da_acc = (da_pred == targets_da).mean()
    # DA Loss (BCE)
    da_loss = -mean(targets_da*log(probs) + (1-targets_da)*log(1-probs))
    # 反标后的 RMSE
    rmse = sqrt(mean((inverse_scale(preds_mag) - inverse_scale(targets_mag))**2))
    return da_acc, da_loss, rmse
```

### 4.4 关键超参

| 参数 | 默认 | 含义 |
|---|---|---|
| `--sequence-length` | 30 | 时间窗口天数 |
| `--batch-size` | 128 | 训练批大小 |
| `--epochs` | 50 | 最大轮数 |
| `--lr` | 5e-5 | AdamW 学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--huber-delta` | 1.0 | Huber 阈值 |
| `--patience` | 15 | 早停耐心 |
| `--train-ratio` | 0.8 | 训练比例（按时间切分） |

### 4.5 训练启动

```bash
python models/hybrid_model/hybrid_model_trainer.py \
  --csv-path datas/stock_datas_vif.csv \
  --finbert-dir models/yiyanghkust_finbert-tone \
  --output-dir models/hybrid_model/checkpoints
```

> ⚠️ 训练时**重新计算 FinBERT 嵌入**（不读缓存），保证与训练时的标量化器一致。

---

## 5. 推理流程

### 5.1 模型加载

```python
def load_hybrid_model(model_path, device=None):
    ckpt = torch.load(model_path, map_location=device)
    model = HybridModel(**ckpt['model_config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    scaler_dict = load_scalers_from_npz(model_path.parent / "hybrid_scalers.npz")
    return model, scaler_dict
```

`model_config` 是在训练时保存的关键字参数字典，推理时按它重建网络。

### 5.2 在线工具 `HybridPredictor`

文件：[`agents/Tools/predict_t1.py`](../../agents/Tools/predict_t1.py)

```python
class HybridPredictor:
    def __init__(self, model_path=None, device=None):
        # 默认 models/hybrid_model/checkpoints/hybrid_model.pt
        # 自动检测 cuda/cpu

    def prepare_tensors(self, simulation_data, news_column="NewsTitles"):
        # 取最后 30 行；按 18 维 VIF 顺序抽取并 RobustScaler 变换
        # 文本张量用零向量占位（保持维度一致）
        return scalar_tensor, text_tensor

    def predict(self, simulation_data, current_price, news_column="NewsTitles"):
        # 1. 至少需要 seq_len+40 行历史
        # 2. 准备张量
        # 3. predict_next_day(model, scalar, text, scaler_dict, device)
        # 4. 反标 mag, 计算 predicted_price = current_price * exp(direction * mag)
        return {
            "predicted_price": ...,
            "predicted_change_pct": ...,
            "direction_prob": ...,
            "pred_magnitude": ...,
            "final_log_return": ...,
            "error": None
        }
```

### 5.3 推理时文本简化

**注意**：`predict_t1.py` 在推理时使用**零向量**作为文本塔输入，理由：

- LLM 调度时**不会为每只股票重新跑 FinBERT**（700MB 模型 + 30 步推理延迟太高）。
- 文本塔在零输入下学到的"无信息"表示，由时序塔与门控机制降权处理。
- 真实情感已通过 `Sentiment_Score` 进入 18 维特征，所以文本塔更多是"占位/补充"角色。

如果未来需要把当日新闻实时注入，可在 `predict_t1.py::prepare_tensors` 中加入 FinBERT 编码。

---

## 6. 端到端示例

```python
from agents.Tools.predict_t1 import HybridPredictor
import pandas as pd

df = pd.read_csv("datas/stock_datas_vif.csv")
df = df[df["Ticker"] == "EBAY"].sort_values("Date").tail(200)

predictor = HybridPredictor()
result = predictor.predict(
    simulation_data=df,
    current_price=df["Close"].iloc[-1],
    news_column="NewsTitles"
)
print(result)
# {'predicted_price': 52.13, 'predicted_change_pct': 1.25,
#  'direction_prob': 0.68, 'pred_magnitude': 0.0124, ...}
```

---

## 7. 模型可解释性提示

虽然 HybridModel 本身是"黑盒"，但通过以下方式可获得一些可解释信息：

1. **方向概率** `direction_prob`：直接反映模型对 T+1 上涨的信心。
2. **幅度** `pred_magnitude`：模型预测的"典型波动"大小，与历史 `LogReturn.abs()` 同量纲。
3. **门控权重**（如果保存）可指示某日更依赖时序还是文本。
4. **VIF 特征** 已包含 `Sentiment_Score`，时序塔在"看"情感趋势。

---

## 8. 改进方向（Roadmap）

| 方向 | 描述 |
|---|---|
| 在线文本编码 | 推理时把当日新闻喂给 FinBERT，而非零向量 |
| 注意力可视化 | 提取 `cross_attn` 权重画热力图 |
| 多频率预测 | 同时预测 T+1 / T+5 / T+20 |
| 标的嵌入 | 增加 stock embedding 区分不同股票的特性 |
| 集成学习 | 多个 HybridModel 取平均/投票 |
| 不确定性估计 | MC-Dropout 输出预测区间 |
