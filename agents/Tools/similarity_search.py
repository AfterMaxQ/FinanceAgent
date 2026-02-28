"""历史行情相似度搜索器：使用滑动窗口和序列相似度算法在历史数据中寻找与当前行情走势形状相似的片段。

本模块实现了基于时间序列相似度匹配的算法：
1. 使用 Z-Score 标准化处理价格序列，只比较"形状"而非绝对价格
2. 使用滑动窗口在历史数据上搜索相似片段
3. 计算欧氏距离或余弦相似度进行匹配
4. 统计匹配片段后续的涨跌幅作为参考
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

# 获取项目根目录（参考 finbert_analyzer.py 的方式）
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent

# 配置日志
logger = logging.getLogger(__name__)


class SimilaritySearcher:
    """历史行情相似度搜索器类。
    
    使用滑动窗口和序列相似度算法在历史数据中寻找与当前行情走势形状相似的片段。
    通过 Z-Score 标准化处理，只比较价格走势的"形状"而非绝对价格水平。
    
    Attributes:
        target_seq: 当前待匹配的价格序列（已标准化）。
        similarity_method: 相似度计算方法（'euclidean' 或 'cosine'）。
    """
    
    def __init__(self, similarity_method: str = 'euclidean') -> None:
        """初始化相似度搜索器。
        
        Args:
            similarity_method: 相似度计算方法，'euclidean'（欧氏距离）或 'cosine'（余弦相似度），默认 'euclidean'。
        
        Raises:
            ValueError: 当 similarity_method 不是 'euclidean' 或 'cosine' 时。
        """
        if similarity_method not in ['euclidean', 'cosine']:
            raise ValueError(
                f"similarity_method 必须是 'euclidean' 或 'cosine'，"
                f"当前值: {similarity_method}"
            )
        self.similarity_method = similarity_method
        self.target_seq: Optional[np.ndarray] = None
    
    def _zscore_normalize(self, sequence: np.ndarray) -> np.ndarray:
        """对序列进行 Z-Score 标准化。
        
        Args:
            sequence: 输入序列。
        
        Returns:
            标准化后的序列。
        
        Raises:
            ValueError: 当序列标准差为 0 时（无法标准化）。
        """
        mean = np.mean(sequence)
        std = np.std(sequence)
        
        if std == 0 or np.isnan(std) or np.isinf(std):
            logger.warning("序列标准差为 0 或无效，返回原序列")
            return sequence
        
        normalized = (sequence - mean) / std
        
        # 处理可能的 NaN 或 inf 值
        if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
            logger.warning("标准化后出现 NaN 或 inf 值，使用原序列")
            return sequence
        
        return normalized
    
    def _calculate_similarity(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray
    ) -> float:
        """计算两个序列的相似度。
        
        Args:
            seq1: 第一个序列（已标准化）。
            seq2: 第二个序列（已标准化）。
        
        Returns:
            相似度得分（0-100，越高越相似）。对于欧氏距离，转换为相似度得分。
        """
        if len(seq1) != len(seq2):
            raise ValueError(
                f"序列长度不匹配: seq1={len(seq1)}, seq2={len(seq2)}"
            )
        
        if self.similarity_method == 'euclidean':
            # 计算欧氏距离
            distance = euclidean(seq1, seq2)
            # 将距离转换为相似度得分（0-100）
            # 使用指数衰减函数：similarity = 100 * exp(-distance / scale)
            # scale 可以根据数据调整，这里使用经验值
            scale = max(1.0, np.std(seq1) + np.std(seq2))
            similarity_score = 100.0 * np.exp(-distance / scale)
            # 确保得分在合理范围内
            similarity_score = max(0.0, min(100.0, similarity_score))
        else:  # cosine
            # 计算余弦相似度
            # cosine 函数返回距离（0-2），需要转换为相似度
            distance = cosine(seq1, seq2)
            # 余弦相似度 = 1 - cosine_distance
            similarity_score = 100.0 * (1.0 - distance / 2.0)
            # 处理可能的 NaN
            if np.isnan(similarity_score) or np.isinf(similarity_score):
                similarity_score = 0.0
            similarity_score = max(0.0, min(100.0, similarity_score))
        
        return float(similarity_score)
    
    def _calculate_subsequent_return(
        self,
        df: pd.DataFrame,
        end_idx: int,
        days: int = 5
    ) -> Optional[float]:
        """计算指定日期之后 N 天的涨跌幅。
        
        Args:
            df: 包含 Close 列的 DataFrame。
            end_idx: 结束索引（历史片段的结束位置）。
            days: 后续天数，默认 5。
        
        Returns:
            涨跌幅（百分比），如果数据不足则返回 None。
        """
        if end_idx + days >= len(df):
            return None
        
        current_price = df.iloc[end_idx]['Close']
        future_price = df.iloc[end_idx + days]['Close']
        
        if pd.isna(current_price) or pd.isna(future_price):
            return None
        
        if current_price == 0:
            return None
        
        return_percent = ((future_price - current_price) / current_price) * 100.0
        
        # 处理异常值
        if np.isnan(return_percent) or np.isinf(return_percent):
            return None
        
        return float(return_percent)
    
    def search_similar_periods(
        self,
        df: pd.DataFrame,
        query_window: int = 20,
        top_k: int = 5,
        subsequent_days: int = 5,
        similarity_method: Optional[str] = None,
        **_: Dict
    ) -> List[Dict]:
        """搜索与当前行情走势相似的历史片段。
        
        Args:
            df: 包含 Close 列的历史数据 DataFrame。如果包含 Date 列，将用于返回结果。
            query_window: 待匹配的当前 K 线长度（例如最近 20 天），默认 20。
            top_k: 返回最相似的个数，默认 5。
            subsequent_days: 统计后续涨跌幅的天数，默认 5。
            similarity_method: 可选，动态覆盖实例的相似度计算方法，支持 'euclidean' 与 'cosine'。
            **_: 兼容占位，忽略未使用的多余参数，避免接口变更导致的报错。
        
        Returns:
            包含前 top_k 个匹配结果的列表。每个结果包含：
            - date: 历史片段的结束日期（如果有 Date 列）。
            - end_index: 历史片段的结束索引。
            - similarity_score: 相似度得分 (0-100，越高越相似)。
            - subsequent_return: 该历史片段之后 N 天的实际涨跌幅（百分比）。
            - visualization_data: 用于绘图的数据字典，包含 'target' 和 'matched' 序列。
        
        注意: 当数据不足或遇到错误时，返回空列表，不会抛出异常。
        """
        try:
            # 兼容：如果调用时传入 similarity_method（例如通过工具参数），动态切换实例配置
            if similarity_method:
                if similarity_method in ("euclidean", "cosine"):
                    self.similarity_method = similarity_method
                else:
                    logger.warning(
                        "收到不支持的 similarity_method=%s，继续使用默认方法 %s",
                        similarity_method,
                        self.similarity_method,
                    )
            # 检查空 DataFrame
            if df is None or df.empty:
                logger.warning("输入 DataFrame 为空，返回空列表")
                return []
            
            # 验证输入
            if 'Close' not in df.columns:
                logger.warning("DataFrame 必须包含 'Close' 列，返回空列表")
                return []
            
            # 检查最小数据要求（至少需要 query_window + subsequent_days 天）
            min_required = query_window + subsequent_days
            if len(df) < min_required:
                logger.warning(
                    f"数据长度 ({len(df)}) 不足，至少需要 {min_required} 天，返回空列表"
                )
                return []
            
            # 提取目标序列（最后 query_window 天）
            target_seq_raw = df['Close'].tail(query_window).values
            
            # 检查数据有效性
            if np.any(np.isnan(target_seq_raw)) or np.any(np.isinf(target_seq_raw)):
                logger.warning("目标序列包含 NaN 或 inf 值，尝试填充")
                target_series = pd.Series(target_seq_raw)
                target_series = target_series.ffill().bfill()
                target_seq_raw = target_series.values
                if np.any(np.isnan(target_seq_raw)) or np.any(np.isinf(target_seq_raw)):
                    logger.warning("目标序列包含无效值，无法处理，返回空列表")
                    return []
            
            # 标准化目标序列
            try:
                self.target_seq = self._zscore_normalize(target_seq_raw)
            except Exception as e:
                logger.warning(f"目标序列标准化失败: {str(e)}，返回空列表")
                return []
            
            # 准备历史数据（排除最后 query_window 天，以及需要后续数据）
            history_df = df.iloc[:-query_window].copy()
            search_end = len(history_df) - subsequent_days
            
            if search_end < query_window:
                logger.warning(
                    f"历史数据不足，无法进行搜索。"
                    f"需要至少 {query_window + subsequent_days} 天的历史数据，返回空列表"
                )
                return []
            
            # 使用向量化操作计算所有窗口的相似度
            similarities = []
            end_indices = []
            
            # 提取所有可能的历史窗口
            for i in range(query_window, search_end + 1):
                try:
                    # 提取历史片段
                    hist_seq_raw = history_df['Close'].iloc[i - query_window:i].values
                    
                    # 检查数据有效性
                    if np.any(np.isnan(hist_seq_raw)) or np.any(np.isinf(hist_seq_raw)):
                        continue
                    
                    # 标准化历史序列
                    try:
                        hist_seq_normalized = self._zscore_normalize(hist_seq_raw)
                    except Exception:
                        # 如果标准化失败，跳过这个片段
                        continue
                    
                    # 计算相似度
                    similarity = self._calculate_similarity(
                        self.target_seq,
                        hist_seq_normalized
                    )
                    
                    # 检查相似度是否有效
                    if np.isnan(similarity) or np.isinf(similarity):
                        continue
                    
                    similarities.append(similarity)
                    end_indices.append(i)
                except (IndexError, KeyError) as e:
                    # 索引错误，跳过这个片段
                    logger.debug(f"处理索引 {i} 时出错: {str(e)}，跳过")
                    continue
                except Exception as e:
                    # 其他错误，跳过这个片段
                    logger.debug(f"处理索引 {i} 时出错: {str(e)}，跳过")
                    continue
            
            if len(similarities) == 0:
                logger.warning("未找到任何有效的相似片段，返回空列表")
                return []
            
            # 转换为 numpy 数组以便排序
            similarities = np.array(similarities)
            end_indices = np.array(end_indices)
            
            # 获取 top_k 个最相似的结果
            top_k_actual = min(top_k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:top_k_actual]
            
            # 构建结果列表
            results = []
            has_date_column = 'Date' in df.columns
            
            for idx in top_indices:
                try:
                    end_idx = end_indices[idx]
                    similarity_score = similarities[idx]
                    
                    # 计算后续涨跌幅
                    subsequent_return = self._calculate_subsequent_return(
                        history_df,
                        end_idx,
                        days=subsequent_days
                    )
                    
                    # 获取历史片段数据用于可视化
                    hist_seq_raw = history_df['Close'].iloc[
                        end_idx - query_window:end_idx
                    ].values
                    hist_seq_normalized = self._zscore_normalize(hist_seq_raw)
                    
                    result = {
                        'end_index': int(end_idx),
                        'similarity_score': float(similarity_score),
                        'subsequent_return': subsequent_return,
                        'visualization_data': {
                            'target': self.target_seq.tolist(),
                            'matched': hist_seq_normalized.tolist(),
                            'target_raw': target_seq_raw.tolist(),
                            'matched_raw': hist_seq_raw.tolist()
                        }
                    }
                    
                    # 如果有 Date 列，添加日期信息
                    if has_date_column:
                        try:
                            result['date'] = str(history_df.iloc[end_idx]['Date'])
                        except (IndexError, KeyError):
                            # 如果日期获取失败，跳过日期字段
                            pass
                    
                    results.append(result)
                except Exception as e:
                    # 如果构建某个结果时出错，记录并跳过
                    logger.debug(f"构建结果时出错: {str(e)}，跳过该结果")
                    continue
            
            if len(results) > 0:
                logger.info(
                    f"成功找到 {len(results)} 个相似片段，"
                    f"相似度范围: {similarities[top_indices].min():.2f} - "
                    f"{similarities[top_indices].max():.2f}"
                )
            else:
                logger.warning("虽然找到相似片段，但构建结果时出错，返回空列表")
            
            return results
        
        except Exception as e:
            logger.error(
                f"搜索相似片段时发生未预期的错误: {str(e)}，返回空列表",
                exc_info=True
            )
            return []

