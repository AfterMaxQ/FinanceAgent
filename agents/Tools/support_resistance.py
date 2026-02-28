"""支撑位和阻力位扫描器：使用 KDE (核密度估计) 算法识别股票历史价格中的关键支撑位和阻力位。

本模块实现了基于核密度估计的支撑/阻力位识别算法：
1. 使用 KDE 分析价格分布的概率密度
2. 识别密度峰值作为潜在的支撑/阻力位
3. 根据当前价格分类支撑位和阻力位
4. 计算每个点位的强度（密度值）
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# 获取项目根目录
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent

# 配置日志
logger = logging.getLogger(__name__)


class SupportResistanceScanner:
    """支撑位和阻力位扫描器类。
    
    使用 KDE (核密度估计) 算法识别历史价格中的关键支撑位和阻力位。
    通过分析价格分布的概率密度峰值，找出历史密集交易区（筹码峰）。
    
    Attributes:
        kde: 核密度估计对象。
        price_data: 处理后的价格数据。
    """
    
    def __init__(self) -> None:
        """初始化支撑位和阻力位扫描器。
        
        无需参数，所有配置在 calculate_levels 方法中指定。
        """
        self.kde: Optional[stats.gaussian_kde] = None
        self.price_data: Optional[np.ndarray] = None
    
    def _prepare_price_data(
        self, 
        df: pd.DataFrame, 
        use_weighted: bool = True
    ) -> np.ndarray:
        """准备价格数据。
        
        Args:
            df: 包含 Close, High, Low 列的 DataFrame。
            use_weighted: 是否使用加权价格 (High+Low+Close)/3，否则仅使用 Close。
        
        Returns:
            处理后的价格数组。
        
        Raises:
            ValueError: 当 DataFrame 缺少必需的列时。
        """
        required_cols = ['Close']
        if use_weighted:
            required_cols.extend(['High', 'Low'])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"DataFrame 缺少必需的列: {missing_cols}. "
                f"需要: {required_cols}"
            )
        
        if use_weighted:
            # 使用加权价格 (High+Low+Close)/3
            prices = (df['High'] + df['Low'] + df['Close']) / 3.0
        else:
            # 仅使用收盘价
            prices = df['Close'].copy()
        
        # 移除 NaN 和无效值
        prices = prices.dropna()
        if len(prices) == 0:
            raise ValueError("价格数据为空，无法进行计算")
        
        return prices.values
    
    def _find_kde_peaks(
        self, 
        price_data: np.ndarray, 
        bandwidth: Optional[float] = None,
        num_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用 KDE 找到价格分布的峰值。
        
        Args:
            price_data: 价格数据数组。
            bandwidth: KDE 带宽参数，None 表示自动选择。
            num_points: 用于评估 KDE 的点数。
        
        Returns:
            (峰值价格数组, 对应的密度值数组) 元组。
        """
        # 创建 KDE 对象
        if bandwidth is None:
            # 使用 Scott's rule 估算带宽
            kde = stats.gaussian_kde(price_data)
        else:
            kde = stats.gaussian_kde(price_data)
            kde.set_bandwidth(bandwidth)
        
        self.kde = kde
        
        # 创建价格范围用于评估 KDE
        price_min = price_data.min()
        price_max = price_data.max()
        price_range = price_max - price_min
        # 扩展范围以捕获边界峰值
        price_eval = np.linspace(
            price_min - 0.1 * price_range,
            price_max + 0.1 * price_range,
            num_points
        )
        
        # 评估 KDE 密度
        density = kde(price_eval)
        
        # 找到局部峰值
        # prominence 参数控制峰值的最小突出度（相对于周围数据）
        # 使用密度值的 5% 作为最小突出度
        min_prominence = density.max() * 0.05
        peaks_indices, properties = find_peaks(
            density,
            prominence=min_prominence,
            distance=max(1, num_points // 50)  # 确保峰值之间有足够距离
        )
        
        # 获取峰值对应的价格和密度值
        peak_prices = price_eval[peaks_indices]
        peak_densities = density[peaks_indices]
        
        return peak_prices, peak_densities
    
    def _classify_levels(
        self,
        peak_prices: np.ndarray,
        peak_densities: np.ndarray,
        current_price: float
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """将峰值分类为支撑位和阻力位。
        
        Args:
            peak_prices: 峰值价格数组。
            peak_densities: 对应的密度值数组。
            current_price: 当前价格。
        
        Returns:
            (支撑位列表, 阻力位列表) 元组，每个元素包含 price 和 strength。
        """
        supports = []
        resistances = []
        
        for price, density in zip(peak_prices, peak_densities):
            level_info = {
                'price': float(price),
                'strength': float(density)
            }
            
            if price < current_price:
                supports.append(level_info)
            else:
                resistances.append(level_info)
        
        # 按价格排序：支撑位从高到低，阻力位从低到高
        supports.sort(key=lambda x: x['price'], reverse=True)
        resistances.sort(key=lambda x: x['price'])
        
        return supports, resistances
    
    def _find_nearest_levels(
        self,
        supports: List[Dict[str, float]],
        resistances: List[Dict[str, float]],
        current_price: float
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """找到距离当前价格最近的支撑位和阻力位。
        
        Args:
            supports: 支撑位列表。
            resistances: 阻力位列表。
            current_price: 当前价格。
        
        Returns:
            (最近支撑位, 最近阻力位) 元组。
        """
        nearest_support = None
        nearest_resistance = None
        
        if supports:
            # 找到最接近当前价格的支撑位（但低于当前价格）
            nearest_support = max(
                supports,
                key=lambda x: x['price'] if x['price'] < current_price else -np.inf
            )
        
        if resistances:
            # 找到最接近当前价格的阻力位（但高于当前价格）
            nearest_resistance = min(
                resistances,
                key=lambda x: x['price'] if x['price'] > current_price else np.inf
            )
        
        return nearest_support, nearest_resistance
    
    def _generate_status(
        self,
        current_price: float,
        nearest_support: Optional[Dict[str, float]],
        nearest_resistance: Optional[Dict[str, float]]
    ) -> str:
        """生成状态描述文本。
        
        Args:
            current_price: 当前价格。
            nearest_support: 最近的支撑位。
            nearest_resistance: 最近的阻力位。
        
        Returns:
            状态描述字符串。
        """
        status_parts = []
        
        if nearest_support:
            support_price = nearest_support['price']
            support_distance_pct = ((current_price - support_price) / support_price) * 100
            support_strength = nearest_support['strength']
            
            if support_distance_pct < 2:
                strength_desc = "强" if support_strength > 0.5 else "弱"
                status_parts.append(
                    f"价格位于{strength_desc}支撑位上方 {support_distance_pct:.2f}% 处"
                )
            else:
                status_parts.append(
                    f"最近支撑位: {support_price:.2f} (距离 {support_distance_pct:.2f}%)"
                )
        else:
            status_parts.append("未找到支撑位")
        
        if nearest_resistance:
            resistance_price = nearest_resistance['price']
            resistance_distance_pct = ((resistance_price - current_price) / current_price) * 100
            resistance_strength = nearest_resistance['strength']
            
            if resistance_distance_pct < 2:
                strength_desc = "强" if resistance_strength > 0.5 else "弱"
                status_parts.append(
                    f"价格位于{strength_desc}阻力位下方 {resistance_distance_pct:.2f}% 处"
                )
            else:
                status_parts.append(
                    f"最近阻力位: {resistance_price:.2f} (距离 {resistance_distance_pct:.2f}%)"
                )
        else:
            status_parts.append("未找到阻力位")
        
        return " | ".join(status_parts)
    
    def calculate_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
        window: int = 252,
        use_weighted: bool = True,
        bandwidth: Optional[float] = None
    ) -> Dict[str, Any]:
        """计算支撑位和阻力位。
        
        Args:
            df: 包含 Close, High, Low 列的股票数据 DataFrame。
            current_price: 当前最新价格。
            window: 回溯窗口大小（默认 252 天，即一年）。
            use_weighted: 是否使用加权价格 (High+Low+Close)/3，否则仅使用 Close。
            bandwidth: KDE 带宽参数，None 表示自动选择。
        
        Returns:
            包含以下键的字典：
            - supports: 排序后的支撑位列表（每个元素包含 price 和 strength）。
            - resistances: 排序后的阻力位列表（每个元素包含 price 和 strength）。
            - nearest_support: 距离当前价格最近的支撑位（Dict 或 None）。
            - nearest_resistance: 距离当前价格最近的阻力位（Dict 或 None）。
            - status: 文本描述当前价格相对于支撑/阻力位的位置。
        
        注意: 当数据不足或遇到错误时，返回安全的默认值（空列表和 None），不会抛出异常。
        """
        # 定义安全的默认返回值
        default_result = {
            'supports': [],
            'resistances': [],
            'nearest_support': None,
            'nearest_resistance': None,
            'status': '数据不足，无法计算支撑位和阻力位'
        }
        
        try:
            # 检查空 DataFrame
            if df is None or df.empty:
                logger.warning("输入 DataFrame 为空，返回默认值")
                return default_result
            
            # 检查必需的列
            required_cols = ['Close']
            if use_weighted:
                required_cols.extend(['High', 'Low'])
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"DataFrame 缺少必需的列: {missing_cols}，返回默认值"
                )
                default_result['status'] = f'缺少必需的列: {missing_cols}'
                return default_result
            
            # 检查数据长度（至少需要 10 个数据点才能进行 KDE 分析）
            min_required = 10
            if len(df) < min_required:
                logger.warning(
                    f"数据长度 ({len(df)}) 不足，至少需要 {min_required} 个数据点，返回默认值"
                )
                default_result['status'] = f'数据长度不足 ({len(df)} < {min_required})'
                return default_result
            
            # 检查数据长度
            if len(df) < window:
                logger.warning(
                    f"数据长度 ({len(df)}) 小于窗口大小 ({window})，使用全部数据"
                )
                df_window = df.copy()
            else:
                df_window = df.tail(window).copy()
            
            # 准备价格数据
            try:
                price_data = self._prepare_price_data(df_window, use_weighted)
            except (ValueError, KeyError) as e:
                logger.warning(f"准备价格数据时出错: {str(e)}，返回默认值")
                default_result['status'] = f'价格数据准备失败: {str(e)}'
                return default_result
            
            self.price_data = price_data
            
            # 检查有效数据点数量
            if len(price_data) < min_required:
                logger.warning(
                    f"有效价格数据点不足 ({len(price_data)})，至少需要 {min_required} 个数据点，返回默认值"
                )
                default_result['status'] = f'有效数据点不足 ({len(price_data)} < {min_required})'
                return default_result
            
            # 使用 KDE 找到峰值
            try:
                peak_prices, peak_densities = self._find_kde_peaks(
                    price_data,
                    bandwidth=bandwidth
                )
            except Exception as e:
                logger.warning(f"KDE 计算失败: {str(e)}，返回默认值")
                default_result['status'] = f'KDE 计算失败: {str(e)}'
                return default_result
            
            if len(peak_prices) == 0:
                logger.warning("未找到任何峰值，可能数据分布过于平滑，返回默认值")
                default_result['status'] = '未找到支撑位或阻力位（数据分布过于平滑）'
                return default_result
            
            # 分类支撑位和阻力位
            supports, resistances = self._classify_levels(
                peak_prices,
                peak_densities,
                current_price
            )
            
            # 找到最近的支撑位和阻力位
            nearest_support, nearest_resistance = self._find_nearest_levels(
                supports,
                resistances,
                current_price
            )
            
            # 生成状态描述
            status = self._generate_status(
                current_price,
                nearest_support,
                nearest_resistance
            )
            
            return {
                'supports': supports,
                'resistances': resistances,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'status': status
            }
        
        except Exception as e:
            logger.error(
                f"计算支撑位和阻力位时发生未预期的错误: {str(e)}，返回默认值",
                exc_info=True
            )
            default_result['status'] = f'计算过程出错: {str(e)}'
            return default_result

