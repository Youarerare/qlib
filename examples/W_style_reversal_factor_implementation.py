"""
W式切割反转因子实现
W-Style Splitting Reversal Factor Implementation

理论来源：魏建榕、傅开波（2018）
核心思想：将过去20日涨跌幅按照"平均单笔成交金额"进行切割，
         高平均单笔成交金额的10日涨跌幅加总 → M_high
         低平均单笔成交金额的10日涨跌幅加总 → M_low
         理想反转因子 M = M_high - M_low

研究发现反转效应主要来源于大单成交，因此M_high因子效果更好。
"""

import pandas as pd
import numpy as np
from typing import Union


class WStyleReversalFactor:
    """
    W式切割反转因子
    
    Parameters
    ----------
    window : int
        回溯窗口期，默认20日
    split_method : str
        切割方法：
        - 'median': 按中位数切割（高10日 vs 低10日）
        - 'quantile': 按指定分位数切割
        - 'high_quantile': 只取高分位部分（如13/16分位以上）
    quantile_threshold : float
        当split_method='quantile'或'high_quantile'时的分位数阈值
        默认0.8125 (13/16)
    """
    
    def __init__(self, window: int = 20, split_method: str = 'median', 
                 quantile_threshold: float = 0.8125):
        self.window = window
        self.split_method = split_method
        self.quantile_threshold = quantile_threshold
        
    def calculate_avg_trade_amount(self, amount: pd.Series, trade_count: pd.Series) -> pd.Series:
        """
        计算平均单笔成交金额
        
        Parameters
        ----------
        amount : pd.Series
            成交金额（元）
        trade_count : pd.Series
            成交笔数
            
        Returns
        -------
        avg_trade_amount : pd.Series
            平均单笔成交金额
        """
        return amount / trade_count.replace(0, np.nan)
    
    def get_rolling_quantile_rank(self, data: pd.Series, window: int) -> pd.Series:
        """
        计算滚动分位数排名
        
        Parameters
        ----------
        data : pd.Series
            输入数据
        window : int
            滚动窗口
            
        Returns
        -------
        rank : pd.Series
            排名值（1到window）
        """
        def rank_func(x):
            return pd.Series(x).rank().iloc[-1]
        
        return data.rolling(window=window, min_periods=1).apply(rank_func, raw=False)
    
    def calculate(self, returns: pd.Series, amount: pd.Series, 
                  trade_count: pd.Series = None) -> dict:
        """
        计算W式切割反转因子
        
        Parameters
        ----------
        returns : pd.Series
            日收益率（%）
        amount : pd.Series  
            成交金额（元）
        trade_count : pd.Series, optional
            成交笔数。如果为None，则使用(amount / volume)作为代理
            （需要volume数据）
            
        Returns
        -------
        factors : dict
            包含以下因子：
            - 'M_ideal': 理想反转因子 (M_high - M_low)
            - 'M_high': 高平均单笔成交金额日收益加总
            - 'M_low': 低平均单笔成交金额日收益加总
            - 'M_high_quantile': 高分位切割的M_high
            - 'Ret20': 传统20日反转因子
        """
        # 计算平均单笔成交金额
        if trade_count is not None:
            avg_trade_amount = self.calculate_avg_trade_amount(amount, trade_count)
        else:
            # 如果没有成交笔数，用成交金额本身作为代理
            avg_trade_amount = amount
        
        # 计算平均单笔成交金额的滚动排名
        avg_amount_rank = self.get_rolling_quantile_rank(avg_trade_amount, self.window)
        
        # 传统反转因子
        ret20 = -returns.rolling(window=self.window, min_periods=1).sum()
        
        # 方法1：中位数切割
        median_threshold = self.window / 2
        
        # 高平均单笔成交金额日（排名靠前的10日）
        high_mask = avg_amount_rank > median_threshold
        M_high = (returns * high_mask).rolling(window=self.window, min_periods=1).sum()
        
        # 低平均单笔成交金额日（排名靠后的10日）
        low_mask = avg_amount_rank <= median_threshold
        M_low = (returns * low_mask).rolling(window=self.window, min_periods=1).sum()
        
        # 理想反转因子
        M_ideal = M_high - M_low
        
        # 方法2：高分位切割（如13/16分位以上）
        quantile_threshold_value = self.window * self.quantile_threshold
        high_quantile_mask = avg_amount_rank >= quantile_threshold_value
        M_high_quantile = (returns * high_quantile_mask).rolling(
            window=self.window, min_periods=1).sum()
        
        return {
            'M_ideal': M_ideal,
            'M_high': M_high,
            'M_low': M_low,
            'M_high_quantile': M_high_quantile,
            'Ret20': ret20,
            'avg_trade_amount': avg_trade_amount,
            'avg_amount_rank': avg_amount_rank
        }
    
    def calculate_with_volume_proxy(self, returns: pd.Series, amount: pd.Series, 
                                     volume: pd.Series) -> dict:
        """
        当没有成交笔数数据时，使用成交金额/成交量作为代理
        
        Parameters
        ----------
        returns : pd.Series
            日收益率
        amount : pd.Series
            成交金额
        volume : pd.Series
            成交量（股数）
            
        Returns
        -------
        factors : dict
            因子字典
        """
        # 用(amount / volume)估算平均每笔交易规模
        # 虽然这不是真实的"平均单笔成交金额"，但可以反映交易特征
        avg_trade_size = amount / volume.replace(0, np.nan)
        
        # 计算排名
        avg_size_rank = self.get_rolling_quantile_rank(avg_trade_size, self.window)
        
        # 传统反转因子
        ret20 = -returns.rolling(window=self.window, min_periods=1).sum()
        
        # 中位数切割
        median_threshold = self.window / 2
        high_mask = avg_size_rank > median_threshold
        low_mask = avg_size_rank <= median_threshold
        
        M_high = (returns * high_mask).rolling(window=self.window, min_periods=1).sum()
        M_low = (returns * low_mask).rolling(window=self.window, min_periods=1).sum()
        M_ideal = M_high - M_low
        
        # 高分位切割
        quantile_threshold_value = self.window * self.quantile_threshold
        high_quantile_mask = avg_size_rank >= quantile_threshold_value
        M_high_quantile = (returns * high_quantile_mask).rolling(
            window=self.window, min_periods=1).sum()
        
        return {
            'M_ideal': M_ideal,
            'M_high': M_high,
            'M_low': M_low,
            'M_high_quantile': M_high_quantile,
            'Ret20': ret20,
            'avg_trade_size': avg_trade_size,
            'avg_size_rank': avg_size_rank
        }


def w_style_reversal_factor_worldquant_brain_syntax():
    """
    返回WorldQuant Brain平台可用的表达式
    
    由于WorldQuant Brain平台缺少"成交笔数"字段，
    这里提供基于可用数据的近似实现。
    """
    
    expressions = {
        'description': 'W式切割反转因子 - WorldQuant Brain实现版本',
        
        'traditional_ret20': {
            'expression': '-ts_sum(returns, 20)',
            'description': '传统20日反转因子（基准）'
        },
        
        'amount_based_split': {
            'expression': 'ts_sum(returns * ts_rank(volume * vwap, 20), 20) - ts_sum(returns * (20 - ts_rank(volume * vwap, 20)), 20)',
            'description': '基于成交金额估算的W式切割'
        },
        
        'high_quantile_M_high': {
            'expression': '-ts_sum(returns * power(ts_rank(volume * vwap, 20), 2), 20)',
            'description': '高分位切割版M_high（通过平方给予高成交金额日更大权重）'
        },
        
        'amount_deviation_split': {
            'expression': '-ts_sum(returns * ts_rank((volume * vwap) / ts_mean(volume * vwap, 20) - 1, 20), 20)',
            'description': '基于成交金额偏离度的切割'
        },
        
        'vwap_deviation_split': {
            'expression': 'rank(ts_sum(returns * ts_rank(abs(vwap - close) / close, 20), 20))',
            'description': '基于VWAP偏离度的切割'
        },
        
        'combined_version': {
            'expression': '-rank(ts_sum(returns * ts_rank(volume * vwap / ts_mean(volume * vwap, 20), 20), 20))',
            'description': '综合版本（推荐测试）'
        }
    }
    
    return expressions


# 示例：如何在qlib中使用
def example_usage_in_qlib():
    """
    在qlib框架中使用W式切割反转因子的示例
    """
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    
    # 初始化qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
    
    # 定义因子计算函数
    def compute_w_style_factor(df):
        """
        计算W式切割反转因子
        df应该包含：return, amount, trade_count（如果有）
        """
        factor_calculator = WStyleReversalFactor(
            window=20,
            split_method='high_quantile',
            quantile_threshold=13/16  # 高分位切割
        )
        
        # 如果有成交笔数数据
        if 'trade_count' in df.columns:
            factors = factor_calculator.calculate(
                returns=df['return'],
                amount=df['amount'],
                trade_count=df['trade_count']
            )
        else:
            # 使用volume作为代理
            factors = factor_calculator.calculate_with_volume_proxy(
                returns=df['return'],
                amount=df['amount'],
                volume=df['volume']
            )
        
        return pd.DataFrame(factors)
    
    # 获取数据
    instruments = ['sh600000', 'sh600004', 'sh600006']  # 示例股票
    start_time = '2020-01-01'
    end_time = '2023-12-31'
    
    # 加载数据（假设qlib数据中有amount, volume, return字段）
    # 实际使用时需要确保数据中有成交笔数字段，或使用代理变量
    
    print("W式切割反转因子示例")
    print("=" * 50)
    print("使用方法：")
    print("1. 如果有完整的逐笔成交数据，使用 calculate() 方法")
    print("2. 如果只有日度数据，使用 calculate_with_volume_proxy() 方法")
    print("3. 在WorldQuant Brain平台，使用上述表达式进行测试")
    print("=" * 50)
    
    # WorldQuant Brain表达式
    expressions = w_style_reversal_factor_worldquant_brain_syntax()
    print("\nWorldQuant Brain表达式：")
    for name, info in expressions.items():
        if name != 'description':
            print(f"\n{name}:")
            print(f"  表达式: {info['expression']}")
            print(f"  说明: {info['description']}")


if __name__ == '__main__':
    # 示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # 模拟数据
    returns = pd.Series(np.random.randn(100) * 2, index=dates)  # 日收益率
    amount = pd.Series(np.random.uniform(1e8, 1e9, 100), index=dates)  # 成交金额
    trade_count = pd.Series(np.random.randint(5000, 50000, 100), index=dates)  # 成交笔数
    volume = pd.Series(np.random.uniform(1e6, 1e7, 100), index=dates)  # 成交量
    
    print("=" * 60)
    print("W式切割反转因子 - 示例计算")
    print("=" * 60)
    
    # 计算因子
    calculator = WStyleReversalFactor(window=20, quantile_threshold=13/16)
    
    # 方法1：有成交笔数数据
    factors_with_trades = calculator.calculate(returns, amount, trade_count)
    
    print("\n方法1：使用成交笔数数据")
    print(f"理想反转因子 M_ideal 最后5个值:")
    print(factors_with_trades['M_ideal'].tail())
    
    print(f"\nM_high (高分位) 最后5个值:")
    print(factors_with_trades['M_high_quantile'].tail())
    
    # 方法2：没有成交笔数，使用volume代理
    factors_with_volume = calculator.calculate_with_volume_proxy(returns, amount, volume)
    
    print("\n方法2：使用成交量作为代理")
    print(f"理想反转因子 M_ideal 最后5个值:")
    print(factors_with_volume['M_ideal'].tail())
    
    # WorldQuant Brain表达式
    print("\n" + "=" * 60)
    print("WorldQuant Brain平台可用表达式")
    print("=" * 60)
    expressions = w_style_reversal_factor_worldquant_brain_syntax()
    
    for name, info in expressions.items():
        if name != 'description':
            print(f"\n{name}:")
            print(f"  {info['expression']}")
            print(f"  {info['description']}")
    
    # 运行示例
    print("\n" + "=" * 60)
    print("qlib框架使用示例")
    print("=" * 60)
    example_usage_in_qlib()
