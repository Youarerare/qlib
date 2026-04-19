"""
W式切割反转因子实现 - 独立版本
W-Style Splitting Reversal Factor Implementation - Standalone Version

理论来源：魏建榕、傅开波（2018）

核心思想：将过去20日涨跌幅按照"平均单笔成交金额"进行切割，
         高平均单笔成交金额的10日涨跌幅加总 → M_high
         低平均单笔成交金额的10日涨跌幅加总 → M_low
         理想反转因子 M = M_high - M_low

研究发现反转效应主要来源于大单成交，因此M_high因子效果更好。
"""

import numpy as np
from typing import Union, Dict


def calculate_rolling_rank(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算滚动排名
    
    Parameters
    ----------
    data : np.ndarray
        输入数据
    window : int
        滚动窗口
        
    Returns
    -------
    ranks : np.ndarray
        排名值（1到window）
    """
    n = len(data)
    ranks = np.zeros(n)
    
    for i in range(n):
        if i < window - 1:
            # 如果数据不足window个，使用已有数据
            window_data = data[:i+1]
        else:
            window_data = data[i-window+1:i+1]
        
        # 当前值在窗口中的排名
        ranks[i] = np.sum(window_data <= data[i])
    
    return ranks


def w_style_reversal_factor(returns: np.ndarray, 
                           avg_trade_amount: np.ndarray,
                           window: int = 20,
                           quantile_threshold: float = 0.8125) -> Dict[str, np.ndarray]:
    """
    计算W式切割反转因子
    
    Parameters
    ----------
    returns : np.ndarray
        日收益率序列
    avg_trade_amount : np.ndarray
        平均单笔成交金额序列
    window : int
        回溯窗口，默认20日
    quantile_threshold : float
        高分位阈值，默认0.8125 (13/16)
        
    Returns
    -------
    factors : dict
        包含多个因子
    """
    n = len(returns)
    
    # 计算平均单笔成交金额的滚动排名
    avg_amount_rank = calculate_rolling_rank(avg_trade_amount, window)
    
    # 传统反转因子
    ret20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        ret20[i] = -np.sum(returns[start_idx:i+1])
    
    # 中位数切割
    median_threshold = window / 2
    
    # 高平均单笔成交金额日
    M_high = np.zeros(n)
    M_low = np.zeros(n)
    M_ideal = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        
        # 获取窗口内的数据
        window_returns = returns[start_idx:i+1]
        window_ranks = avg_amount_rank[start_idx:i+1]
        
        # 高排名日（平均单笔成交金额高）
        high_mask = window_ranks > median_threshold
        M_high[i] = np.sum(window_returns * high_mask)
        
        # 低排名日
        low_mask = window_ranks <= median_threshold
        M_low[i] = np.sum(window_returns * low_mask)
        
        # 理想反转因子
        M_ideal[i] = M_high[i] - M_low[i]
    
    # 高分位切割
    quantile_threshold_value = window * quantile_threshold
    M_high_quantile = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_returns = returns[start_idx:i+1]
        window_ranks = avg_amount_rank[start_idx:i+1]
        
        # 只取高分位
        high_quantile_mask = window_ranks >= quantile_threshold_value
        M_high_quantile[i] = np.sum(window_returns * high_quantile_mask)
    
    return {
        'M_ideal': M_ideal,
        'M_high': M_high,
        'M_low': M_low,
        'M_high_quantile': M_high_quantile,
        'Ret20': ret20,
        'avg_amount_rank': avg_amount_rank
    }


def get_worldquant_brain_expressions() -> Dict[str, Dict[str, str]]:
    """
    返回WorldQuant Brain平台可用的表达式
    
    由于WorldQuant Brain平台缺少"成交笔数"字段，
    这里提供基于可用数据的近似实现。
    """
    
    expressions = {
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
        },
        
        'simplified_w_style': {
            'expression': '-rank(ts_sum(returns * ts_rank(volume, 20), 20))',
            'description': '简化版（仅使用成交量）'
        },
        
        'advanced_w_style': {
            'expression': '-rank(ts_sum(returns * ts_rank(volume * (vwap - ts_mean(vwap, 20)), 20), 20))',
            'description': '高级版（结合VWAP偏离）'
        }
    }
    
    return expressions


def main():
    """
    主函数 - 示例演示
    """
    print("=" * 70)
    print("W式切割反转因子实现")
    print("W-Style Splitting Reversal Factor Implementation")
    print("=" * 70)
    
    # 模拟数据
    np.random.seed(42)
    n_days = 100
    
    # 模拟日收益率
    returns = np.random.randn(n_days) * 2  # 平均日波动约2%
    
    # 模拟平均单笔成交金额（万元）
    # 假设有趋势和随机波动
    base_amount = 2.0  # 基础2万元
    trend = np.linspace(0, 0.5, n_days)  # 轻微上升趋势
    noise = np.random.randn(n_days) * 0.5
    avg_trade_amount = base_amount + trend + np.abs(noise)
    
    print("\n模拟数据概览：")
    print(f"  天数: {n_days}")
    print(f"  平均日收益率: {returns.mean():.4f}%")
    print(f"  平均单笔成交金额范围: {avg_trade_amount.min():.2f} - {avg_trade_amount.max():.2f} 万元")
    
    # 计算W式切割反转因子
    print("\n计算W式切割反转因子...")
    factors = w_style_reversal_factor(
        returns=returns,
        avg_trade_amount=avg_trade_amount,
        window=20,
        quantile_threshold=13/16  # 高分位切割
    )
    
    # 显示结果
    print("\n" + "=" * 70)
    print("因子计算结果（最后10个值）")
    print("=" * 70)
    
    print("\n1. 传统反转因子 Ret20:")
    print(f"   最后10个值: {factors['Ret20'][-10:]}")
    
    print("\n2. 理想反转因子 M_ideal (M_high - M_low):")
    print(f"   最后10个值: {factors['M_ideal'][-10:]}")
    
    print("\n3. 高平均单笔成交金额日收益 M_high:")
    print(f"   最后10个值: {factors['M_high'][-10:]}")
    
    print("\n4. 低平均单笔成交金额日收益 M_low:")
    print(f"   最后10个值: {factors['M_low'][-10:]}")
    
    print("\n5. 高分位切割M_high (13/16分位):")
    print(f"   最后10个值: {factors['M_high_quantile'][-10:]}")
    
    # WorldQuant Brain表达式
    print("\n" + "=" * 70)
    print("WorldQuant Brain平台可用表达式")
    print("=" * 70)
    
    expressions = get_worldquant_brain_expressions()
    
    for i, (name, info) in enumerate(expressions.items(), 1):
        print(f"\n{i}. {name}:")
        print(f"   表达式: {info['expression']}")
        print(f"   说明: {info['description']}")
    
    # 因子相关性分析
    print("\n" + "=" * 70)
    print("因子相关性分析")
    print("=" * 70)
    
    # 计算因子之间的相关性
    correlations = {}
    factor_names = ['M_ideal', 'M_high', 'M_low', 'M_high_quantile', 'Ret20']
    
    for i, name1 in enumerate(factor_names):
        for name2 in factor_names[i+1:]:
            corr = np.corrcoef(factors[name1], factors[name2])[0, 1]
            correlations[f"{name1} vs {name2}"] = corr
    
    print("\n因子相关性矩阵:")
    for pair, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {pair}: {corr:.4f}")
    
    # 理论总结
    print("\n" + "=" * 70)
    print("理论总结")
    print("=" * 70)
    
    print("""
W式切割反转因子的核心发现：

1. 反转效应的微观来源：
   - 主要来源于大单成交（机构投资者行为）
   - 小单成交更多体现动量特性（散户行为）

2. 切割方法的重要性：
   - 使用高分位切割（如13/16分位）效果最好
   - M_high因子的反转特性更强，IC绝对值更大
   - M_low因子的反转特性逐渐消失，甚至呈现动量特性

3. 实际应用建议：
   - 在有逐笔成交数据时，使用完整的W式切割
   - 在只有日度数据时，使用成交金额估算作为代理
   - 在WorldQuant Brain等平台，可使用提供的表达式进行测试

4. 因子改进方向：
   - 考虑结合其他微观结构特征
   - 可与成交量、VWAP等指标结合
   - 注意避免过拟合和因子拥挤

注意事项：
- WorldQuant Brain缺少"成交笔数"字段，只能使用代理变量
- 不同市场环境下因子效果可能不同
- 需要进行充分的样本外测试
    """)
    
    print("=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
