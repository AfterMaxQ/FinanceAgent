"""T+1 价格预测工具：使用 Hybrid 多任务模型进行股票价格预测。

本模块实现了完整的 Hybrid 模型推理管道，包括：
1. 模型加载和初始化
2. 数据预处理和特征准备
3. 文本嵌入提取
4. 多任务推理（方向概率 + 波动幅度）
5. 价格预测计算
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

# 添加项目根目录到路径
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
sys.path.insert(0, str(_project_root))

from models.hybrid_model.hybrid_model import load_hybrid_model, predict_next_day

# 配置日志
logger = logging.getLogger(__name__)

# VIF 筛选后的特征列表（与模型训练时一致）
VIF_FEATURES = [
    "Volume", "GSPC_Close", "GSPC_LogReturn", "LogReturn",
    "MACDh_12_26_9", "MACDs_12_26_9", "Intraday_Range",
    "Trend_Strength", "Candle_Body", "Month_Sin", "Sentiment_Score",
    "BBB_20_2.0_2.0", "BBP_20_2.0_2.0", "ATR_14", "OBV",
    "Beta_60", "Alpha_60", "Sharpe_60"
]


class HybridPredictor:
    """Hybrid 多任务模型预测器类。
    
    使用 Hybrid 模型进行 T+1 价格预测，同时输出方向概率和波动幅度。
    
    Attributes:
        model: HybridModel 实例。
        scaler_dict: 包含 scaler 和特征列信息的字典。
        embedding_analyzer: FinBERT 嵌入分析器实例。
        device: 模型运行设备（"cuda" 或 "cpu"）。
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """初始化 Hybrid 预测器。
        
        Args:
            model_path: 模型文件路径，默认使用 checkpoints/hybrid_model.pt。
            device: 设备类型（"cuda" 或 "cpu"），None 表示自动检测。
        （已简化，不再在推理阶段计算文本嵌入；若缺失则使用零向量。）
        
        Raises:
            FileNotFoundError: 当模型文件不存在时。
            RuntimeError: 当 CUDA 不可用但指定了 "cuda" 设备时。
        """
        # 设置模型路径
        if model_path is None:
            model_path = _project_root / "models" / "hybrid_model" / "checkpoints" / "hybrid_model.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 设备检测
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，但指定了 cuda 设备")
        
        # 加载模型和 scaler
        logger.info(f"正在加载 Hybrid 模型: {model_path}")
        self.model, self.scaler_dict = load_hybrid_model(str(model_path), device=self.device)
        logger.info("模型加载成功")
        
        # 简化：推理阶段不再计算文本嵌入，使用零向量占位
        self.embedding_analyzer = None
    
    def prepare_tensors(
        self,
        simulation_data: pd.DataFrame,
        news_column: str = "NewsTitles"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备 Hybrid 模型的输入张量。
        
        Args:
            simulation_data: 包含历史数据的 DataFrame。
            news_column: 新闻列名，默认 "NewsTitles"。
        
        Returns:
            scalar_tensor: 标量特征张量 [seq_len, scalar_feature_dim]。
            text_tensor: 文本嵌入张量 [seq_len, text_embedding_dim]。
        """
        sequence_length = self.scaler_dict.get('sequence_length', 30)
        scalar_feature_cols = self.scaler_dict.get('scalar_feature_cols', VIF_FEATURES)
        
        # 确保数据按日期排序
        if 'Date' in simulation_data.columns:
            simulation_data = simulation_data.sort_values('Date').reset_index(drop=True)
        
        # 取最后 sequence_length 行
        data_slice = simulation_data.tail(sequence_length).copy()
        
        # 准备标量特征
        scalar_features = []
        for col in scalar_feature_cols:
            if col in data_slice.columns:
                values = data_slice[col].fillna(0.0).astype(np.float32).values
                scalar_features.append(values)
            else:
                # 如果列不存在，填充零
                scalar_features.append(np.zeros(len(data_slice), dtype=np.float32))
        
        scalar_array = np.column_stack(scalar_features)
        
        # 应用 scaler
        if 'scalar_scaler' in self.scaler_dict:
            scalar_scaler = self.scaler_dict['scalar_scaler']
            scalar_array = scalar_scaler.transform(scalar_array)
        
        scalar_tensor = torch.tensor(scalar_array, dtype=torch.float32)
        
        # 准备文本嵌入：推理阶段默认使用零向量（保持维度一致）
        embedding_dim = self.scaler_dict.get('text_input_dim', 768)
        text_tensor = torch.zeros((sequence_length, embedding_dim), dtype=torch.float32)
        
        return scalar_tensor, text_tensor
    
    def predict(
        self,
        simulation_data: pd.DataFrame,
        current_price: float,
        news_column: str = "NewsTitles"
    ) -> Dict[str, Any]:
        """执行 T+1 价格预测。
        
        Args:
            simulation_data: 包含历史数据的 DataFrame。
            current_price: 当前价格。
            news_column: 新闻列名，默认 "NewsTitles"。
        
        Returns:
            包含预测结果的字典：
            - predicted_price: 预测价格
            - predicted_change_pct: 预测变化百分比
            - direction_prob: 上涨概率
            - pred_magnitude: 预测波动幅度
            - final_log_return: 最终对数收益率
            - error: 错误信息（如果有）
        """
        try:
            # 检查数据量
            seq_len = int(self.scaler_dict.get('sequence_length', 30))
            min_required = seq_len + 40  # 至少需要 seq_len + 滚动窗口(20) + 缓冲(20)
            
            if len(simulation_data) < min_required:
                return {
                    "predicted_price": current_price * 0.965,
                    "predicted_change_pct": -3.5,
                    "direction_prob": 0.0,
                    "pred_magnitude": 0.0,
                    "final_log_return": 0.0,
                    "error": f"历史数据不足（需要{min_required}天）"
                }
            
            # 准备数据
            historical_data = simulation_data.tail(min_required).copy()
            if 'Date' in historical_data.columns:
                historical_data["Date"] = pd.to_datetime(historical_data["Date"]).dt.date
            
            # 确保 NewsTitles 列存在
            if news_column not in historical_data.columns:
                historical_data[news_column] = 'No significant news'
            
            # 准备张量
            scalar_tensor, text_tensor = self.prepare_tensors(
                simulation_data=historical_data,
                news_column=news_column
            )
            
            # 执行推理
            direction_prob, pred_magnitude = predict_next_day(
                model=self.model,
                scalar_seq=scalar_tensor,
                text_seq=text_tensor,
                scaler_dict=self.scaler_dict,
                device=self.device
            )
            
            # 幅度必须为正
            pred_magnitude = abs(pred_magnitude)
            
            # 整合结果
            direction = 1 if direction_prob >= 0.5 else -1
            final_log_return = direction * pred_magnitude
            
            predicted_price = current_price * np.exp(final_log_return)
            predicted_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            return {
                "predicted_price": float(predicted_price),
                "predicted_change_pct": float(predicted_change_pct),
                "direction_prob": float(direction_prob),
                "pred_magnitude": float(pred_magnitude),
                "final_log_return": float(final_log_return),
                "error": None
            }
        
        except Exception as e:
            logger.error(f"预测过程出错: {e}", exc_info=True)
            return {
                "predicted_price": current_price * 0.965,
                "predicted_change_pct": -3.5,
                "direction_prob": 0.0,
                "pred_magnitude": 0.0,
                "final_log_return": 0.0,
                "error": str(e)
            }

