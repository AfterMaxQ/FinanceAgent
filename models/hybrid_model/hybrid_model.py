"""
混合模型模块 - 多任务架构，同时预测方向（DA）和幅度（Magnitude）。

架构变更：
- HybridModel类现在是一个双头模型，共享一个主体网络。
- 一个分类头（cls_head）用于预测涨跌方向（概率）。
- 一个回归头（reg_head）用于预测波动幅度。
- forward方法返回两个输出：方向概率和幅度预测值。
- 数据处理函数已更新，以支持和缩放幅度目标。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# ==================== Scaler 管理 (已更新以适应多任务模型) ====================

def save_scalers_to_npz(scaler_dict: Dict[str, Any], save_path: Path):
    """保存特征Scaler（标量和文本）和目标Scaler（幅度）到.npz文件。"""
    scalar_scaler = scaler_dict['scalar_scaler']
    text_scaler = scaler_dict['text_scaler']
    target_scaler = scaler_dict['target_scaler'] # ## RE-INTRODUCED ##
    
    data_to_save = {
        'sequence_length': scaler_dict['sequence_length'],
        'scalar_feature_cols': scaler_dict['scalar_feature_cols'],
        'text_feature_cols': scaler_dict['text_feature_cols'],
        'scalar_center': scalar_scaler.center_,
        'scalar_scale': scalar_scaler.scale_,
        'text_center': text_scaler.center_,
        'text_scale': text_scaler.scale_,
        'target_center': target_scaler.center_, # ## RE-INTRODUCED ##
        'target_scale': target_scaler.scale_,   # ## RE-INTRODUCED ##
    }
    np.savez_compressed(save_path, **data_to_save)
    logger.info(f"特征和目标 Scalers 已保存至: {save_path}")

def load_scalers_from_npz(npz_path: Path) -> Dict[str, Any]:
    """从.npz文件加载并重建特征和目标Scaler。"""
    if not npz_path.exists(): raise FileNotFoundError(f"找不到 Scaler 文件: {npz_path}")
    loaded = np.load(npz_path, allow_pickle=True)
    
    scalar_scaler = RobustScaler()
    scalar_scaler.center_ = loaded['scalar_center']
    scalar_scaler.scale_ = loaded['scalar_scale']
    
    text_scaler = RobustScaler()
    text_scaler.center_ = loaded['text_center']
    text_scaler.scale_ = loaded['text_scale']
    
    target_scaler = RobustScaler() # ## RE-INTRODUCED ##
    target_scaler.center_ = loaded['target_center']
    target_scaler.scale_ = loaded['target_scale']
        
    return {
        'scalar_scaler': scalar_scaler, 'text_scaler': text_scaler, 'target_scaler': target_scaler,
        'sequence_length': int(loaded['sequence_length']),
        'scalar_feature_cols': list(loaded['scalar_feature_cols']),
        'text_feature_cols': list(loaded['text_feature_cols']),
    }

def fit_scalers_from_files(scalar_csv_path: Path, emb_npy_path: Path, sequence_length: int = 30, scalar_feature_cols: Optional[List[str]] = None, text_feature_cols: Optional[List[str]] = None, sample_ratio: float = 0.05) -> Dict[str, Any]:
    """从文件拟合特征和目标Scaler。"""
    logger.info(f"读取标量数据用于拟合 Scalers: {scalar_csv_path}")
    scalar_df = pd.read_csv(scalar_csv_path)
    if scalar_feature_cols is None:
        exclude = {'Date', 'Stock', 'LogReturn', 'Target_DA', 'Target_Magnitude', 'Unnamed: 0'}
        scalar_feature_cols = [c for c in scalar_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(scalar_df[c])]
    
    if text_feature_cols is None:
        emb_shape = np.load(emb_npy_path, mmap_mode='r').shape
        text_feature_cols = [f'emb_{i}' for i in range(emb_shape[1])]
    
    indices = np.random.choice(len(scalar_df), size=min(len(scalar_df), max(2000, int(len(scalar_df) * sample_ratio))), replace=False)
    scalar_samples = scalar_df.iloc[indices][scalar_feature_cols].values.astype(np.float32)
    embeddings = np.load(emb_npy_path, mmap_mode='r')
    text_samples = embeddings[indices].astype(np.float32)
    target_samples = scalar_df.iloc[indices][['Target_Magnitude']].values.astype(np.float32) # ## RE-INTRODUCED ##
    
    logger.info("拟合特征和目标 Scalers...")
    s_s, t_s, tgt_s = RobustScaler(), RobustScaler(), RobustScaler()
    s_s.fit(np.nan_to_num(scalar_samples))
    t_s.fit(np.nan_to_num(text_samples))
    tgt_s.fit(np.nan_to_num(target_samples)) # ## RE-INTRODUCED ##
    
    return {
        'scalar_scaler': s_s, 'text_scaler': t_s, 'target_scaler': tgt_s,
        'scalar_feature_cols': scalar_feature_cols, 'text_feature_cols': text_feature_cols, 
        'sequence_length': sequence_length
    }

# ==================== 数据集 (已更新以适应多任务模型) ====================

class MemmapHybridDataset(Dataset):
    """为多任务（方向和幅度）预测定制的数据集。"""
    def __init__(self, scalar_csv_path: Path, emb_npy_path: Path, start_idx: int, end_idx: int, scalar_feature_cols: List[str], sequence_length: int, scalar_scaler: RobustScaler, text_scaler: RobustScaler, target_scaler: RobustScaler):
        self.sequence_length = sequence_length
        self.scalar_feature_cols = scalar_feature_cols
        self.scalar_scaler = scalar_scaler
        self.text_scaler = text_scaler
        self.target_scaler = target_scaler # ## RE-INTRODUCED ##
        
        full_scalar_df = pd.read_csv(scalar_csv_path)
        
        self.start_idx = max(start_idx, sequence_length)
        self.end_idx = min(end_idx, len(full_scalar_df))
        self.n_samples = self.end_idx - self.start_idx
        if self.n_samples <= 0: raise ValueError("数据集为空")

        self.embeddings_mmap = np.load(emb_npy_path, mmap_mode='r')
        slice_start = self.start_idx - sequence_length
        
        # 准备特征数据块
        scalar_values = full_scalar_df.iloc[slice_start:self.end_idx][scalar_feature_cols].values.astype(np.float32)
        self.scalar_data_block = self.scalar_scaler.transform(np.nan_to_num(scalar_values))
        
        # ## MODIFIED ##: 准备两个目标的数据块
        self.target_da_block = full_scalar_df.iloc[slice_start:self.end_idx][['Target_DA']].values.astype(np.float32)
        magnitude_values = full_scalar_df.iloc[slice_start:self.end_idx][['Target_Magnitude']].values.astype(np.float32)
        self.target_mag_block = self.target_scaler.transform(np.nan_to_num(magnitude_values))
        
        self.offset = slice_start

    def __len__(self) -> int: return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_target_idx = self.start_idx + idx
        seq_start = abs_target_idx - self.sequence_length
        
        scalar_seq = self.scalar_data_block[seq_start - self.offset : abs_target_idx - self.offset]
        
        text_values = self.embeddings_mmap[seq_start:abs_target_idx]
        text_seq = self.text_scaler.transform(np.nan_to_num(text_values))
        
        # ## MODIFIED ##: 获取两个目标
        target_da = self.target_da_block[abs_target_idx - self.offset][0]
        target_mag = self.target_mag_block[abs_target_idx - self.offset][0]
        
        return (torch.from_numpy(scalar_seq.copy()).float(), 
                torch.from_numpy(text_seq.copy()).float(), 
                torch.tensor(target_da, dtype=torch.float32),
                torch.tensor(target_mag, dtype=torch.float32))

def create_dataloaders(scalar_csv_path: Path, emb_npy_path: Path, total_rows: int, scaler_dict: Dict[str, Any], batch_size: int = 128, train_ratio: float = 0.8, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """为多任务模型创建数据加载器。"""
    seq_len = scaler_dict['sequence_length']
    n_train = int((total_rows - seq_len) * train_ratio)
    train_start, train_end = seq_len, seq_len + n_train
    val_start, val_end = train_end, total_rows
    
    common = {
        "scalar_csv_path": scalar_csv_path, "emb_npy_path": emb_npy_path,
        "scalar_feature_cols": scaler_dict['scalar_feature_cols'],
        "sequence_length": seq_len, 
        "scalar_scaler": scaler_dict['scalar_scaler'],
        "text_scaler": scaler_dict['text_scaler'],
        "target_scaler": scaler_dict['target_scaler']
    }
    train_ds = MemmapHybridDataset(start_idx=train_start, end_idx=train_end, **common)
    val_ds = MemmapHybridDataset(start_idx=val_start, end_idx=val_end, **common)
    return DataLoader(train_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True), DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ==================== 模型核心架构 (主体部分与之前保持一致) ====================

class TemporalTower(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.dropout(self.ln(h_n[-1]))

class TextTower(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 512, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size*4, dropout, 'gelu', 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_proj(x)
        if x.size(1) <= self.pos_enc.size(1): x = x + self.pos_enc[:, :x.size(1), :]
        else: x = x + F.interpolate(self.pos_enc.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False).transpose(1, 2)
        return self.dropout(self.transformer(x))

class FusionLayer(nn.Module):
    def __init__(self, temporal_dim: int, text_dim: int, dropout: float = 0.2):
        super().__init__()
        self.query_proj = nn.Linear(temporal_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(text_dim, 8, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(text_dim*2, text_dim), nn.ReLU(), nn.Linear(text_dim, 1), nn.Sigmoid())
        self.ln = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, temp, text):
        temp_q = self.query_proj(temp).unsqueeze(1)
        attn_out, _ = self.cross_attn(temp_q, text, text)
        attn_out, temp_aligned = attn_out.squeeze(1), temp_q.squeeze(1)
        gate_val = self.gate(torch.cat([temp_aligned, attn_out], dim=1))
        fused = gate_val * temp_aligned + (1 - gate_val) * attn_out
        return self.dropout(self.ln(fused + temp_aligned))

# ==================== 混合模型 (修改为双头DA+幅度预测模型) ====================

class HybridModel(nn.Module):
    """
    专注于方向准确率（DA）和幅度（Magnitude）的双头混合模型。
    """
    def __init__(self, scalar_input_dim:int, text_input_dim:int, temporal_hidden_size:int=256, temporal_num_layers:int=2, temporal_dropout:float=0.2, text_hidden_size:int=768, text_num_layers:int=4, text_num_heads:int=8, text_dropout:float=0.3, fusion_dropout:float=0.2, **kwargs):
        super().__init__()
        self.temporal_tower = TemporalTower(scalar_input_dim, temporal_hidden_size, temporal_num_layers, temporal_dropout)
        self.text_tower = TextTower(text_input_dim, text_hidden_size, text_num_layers, text_num_heads, text_dropout)
        self.fusion_layer = FusionLayer(temporal_hidden_size, text_hidden_size, fusion_dropout)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )
        
        # --- 头1：分类头 (预测方向) ---
        self.cls_head = nn.Sequential(
            nn.Linear(text_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # --- 头2：回归头 (预测幅度) ---
        self.reg_head = nn.Linear(text_hidden_size // 2, 1)

    def forward(self, scalar_seq, text_seq) -> Tuple[torch.Tensor, torch.Tensor]:
        temp_emb = self.temporal_tower(scalar_seq)
        text_emb = self.text_tower(text_seq)
        fused_emb = self.fusion_layer(temp_emb, text_emb)
        
        shared_feat = self.shared_fc(fused_emb)
        
        # --- 关键修改：从共享特征并行输出两个预测 ---
        pred_prob = self.cls_head(shared_feat).squeeze(-1) # 方向概率
        pred_mag = self.reg_head(shared_feat).squeeze(-1)  # 幅度值
        
        return pred_prob, pred_mag

# ==================== 辅助函数 (已修改) ====================

def load_hybrid_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[HybridModel, Dict[str, Any]]:
    """加载多任务版本的HybridModel。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = HybridModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device); model.eval()
    scaler_path = Path(model_path).parent / "hybrid_scalers.npz"
    scaler_dict = load_scalers_from_npz(scaler_path) if scaler_path.exists() else {}
    return model, scaler_dict

def predict_next_day(
    model: HybridModel, 
    scalar_seq: torch.Tensor, 
    text_seq: torch.Tensor,
    scaler_dict: Dict[str, Any], # ## ADDED ## 需要scaler来反向缩放
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    使用多任务模型进行推理，返回上涨的概率和真实尺度的幅度。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        scalar_seq = scalar_seq.unsqueeze(0).to(device) if scalar_seq.ndim == 2 else scalar_seq.to(device)
        text_seq = text_seq.unsqueeze(0).to(device) if text_seq.ndim == 2 else text_seq.to(device)
        
        prob, mag_scaled = model(scalar_seq, text_seq)
        
        target_scaler = scaler_dict['target_scaler']
        mag_unscaled = target_scaler.inverse_transform(mag_scaled.cpu().numpy().reshape(-1, 1))
        
        return prob.item(), mag_unscaled.flatten()[0]