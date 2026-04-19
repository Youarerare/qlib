"""
遗传算法因子搜索功能测试脚本
- 测试简单公式解析
- 验证遗传算法初始化和适应度评估
- 输出详细日志
"""
import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clean.alpha_engine import AlphaEngine
from clean.ic_analyzer import calc_ic_series, calc_ic_summary, evaluate_factor
from clean.ga_search import ExpressionGenerator, GAFactorSearcher, _safe_spearman_ic, _safe_ir
from clean.config import GA, DATA_FIELDS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_ohlcv_data(n_days=30, n_stocks=20):
    """创建模拟OHLCV数据"""
    np.random.seed(42)
    
    # 生成日期和股票代码
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')  # 工作日
    stocks = [f'Stock_{i:03d}' for i in range(n_stocks)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 生成模拟数据
    data = {}
    for stock in stocks:
        # 基础价格
        base_price = np.random.uniform(10, 100)
        prices = base_price * (1 + np.random.randn(n_days).cumsum() * 0.02)
        prices = np.maximum(prices, 1)  # 确保价格为正
        
        data[stock] = {
            'open': prices * (1 + np.random.randn(n_days) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.02),
            'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.02),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, n_days) * (1 + np.random.randn(n_days) * 0.3),
        }
    
    # 组装DataFrame
    df_data = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_data[col] = []
        for stock in stocks:
            df_data[col].extend(data[stock][col])
    
    df = pd.DataFrame(df_data, index=index)
    df['volume'] = df['volume'].clip(lower=0)  # 成交量不能为负
    
    logger.info(f"创建模拟数据: {n_days}天, {n_stocks}只股票, 总计{len(df)}条记录")
    return df


def test_simple_formula():
    """测试简单公式解析"""
    print("\n" + "="*80)
    print("测试1: 简单公式解析 - close - open")
    print("="*80)
    
    # 创建小量测试数据
    np.random.seed(123)
    dates = pd.date_range('2024-01-01', periods=5, freq='B')
    stocks = ['Stock_A', 'Stock_B']
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 明确给定数值
    test_data = {
        'open': [10.0, 20.0, 10.5, 20.5, 11.0, 21.0, 10.8, 20.8, 11.2, 21.2],
        'close': [10.2, 20.2, 10.8, 20.8, 11.5, 21.5, 11.0, 21.0, 11.8, 21.8],
        'high': [10.5, 20.5, 11.0, 21.0, 12.0, 22.0, 11.5, 21.5, 12.5, 22.5],
        'low': [9.8, 19.8, 10.2, 20.2, 10.5, 20.5, 10.5, 20.5, 11.0, 21.0],
        'volume': [1e6, 2e6, 1.1e6, 2.1e6, 1.2e6, 2.2e6, 1.15e6, 2.15e6, 1.3e6, 2.3e6]
    }
    
    df = pd.DataFrame(test_data, index=index)
    
    print("\n[输入数据]")
    print(df.head(10).to_string())
    
    # 创建引擎并计算公式
    engine = AlphaEngine(df)
    formula = "close - open"
    
    print(f"\n[测试公式] {formula}")
    
    try:
        result = engine.calculate(formula)
        print(f"\n[解析器输出]")
        print(result.head(10).to_string())
        
        # 手动计算预期结果
        expected = df['close'] - df['open']
        print(f"\n[预期输出]")
        print(expected.head(10).to_string())
        
        # 验证
        is_match = np.allclose(result.values, expected.values, equal_nan=True)
        print(f"\n[结果] {'✅ 符合预期' if is_match else '❌ 不符合预期'}")
        
        return df, result
    except Exception as e:
        print(f"\n[错误] 计算失败: {e}")
        return None, None


def test_icir_calculation(df, factor_values):
    """测试ICIR计算"""
    print("\n" + "="*80)
    print("测试2: ICIR计算逻辑")
    print("="*80)
    
    # 创建label（下期收益率）
    # 重要：需要正确对齐时间
    returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)
    
    print("\n[Label定义] 未来一期收益率 (T+1日收益率)")
    print("说明: 使用shift(-1)将未来收益率对齐到当前时间")
    
    print("\n[收益率数据示例]")
    print(returns.head(10).to_string())
    
    # 计算IC序列
    ic_series = calc_ic_series(factor_values, returns, method="spearman", min_stocks=2)
    
    print(f"\n[IC序列] (共{len(ic_series)}个有效交易日)")
    print(ic_series.head().to_string())
    
    # 计算IC汇总
    ic_summary = calc_ic_summary(ic_series)
    
    print(f"\n[IC汇总指标]")
    print(f"  IC均值: {ic_summary['ic_mean']:.6f}")
    print(f"  IC标准差: {ic_summary['ic_std']:.6f}")
    print(f"  ICIR: {ic_summary['icir']:.6f}")
    print(f"  IC正率: {ic_summary['ic_positive_ratio']:.4f}")
    print(f"  有效周期数: {ic_summary['n_periods']}")
    
    return returns, ic_summary


def test_ga_initialization(df, returns):
    """测试遗传算法初始化"""
    print("\n" + "="*80)
    print("测试3: 遗传算法初始化与适应度评估")
    print("="*80)
    
    # 创建引擎和生成器
    engine = AlphaEngine(df)
    generator = ExpressionGenerator(engine, returns)
    
    print(f"\n[可用数据字段] {generator.data_fields}")
    print(f"[时序算子] {generator.ts_ops}")
    print(f"[截面算子] {generator.cs_ops}")
    
    # 生成小规模种群
    pop_size = 4
    print(f"\n[遗传算法初始化] 种群大小={pop_size}")
    
    population = []
    for i in range(pop_size):
        expr = generator.generate_random(max_depth=2)
        population.append(expr)
        print(f"个体{i+1}: 公式 \"{expr}\"")
    
    # 评估每个个体
    print("\n[适应度评估]")
    for i, expr in enumerate(population):
        try:
            factor, fitness = generator.evaluate_expression(expr)
            
            if factor is not None:
                # 获取前几个样本值
                sample_values = factor.head(5).tolist()
                
                # 计算IC和ICIR
                ic_mean = _safe_spearman_ic(factor, returns)
                icir = _safe_ir(factor, returns)
                
                print(f"个体{i+1}:")
                print(f"  公式: {expr}")
                print(f"  因子值(前5个): {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in sample_values]}")
                print(f"  IC均值: {ic_mean:.6f}")
                print(f"  ICIR: {icir:.6f}")
                print(f"  适应度: {fitness:.6f}")
            else:
                print(f"个体{i+1}: 公式 \"{expr}\" -> 计算失败, 适应度 = {fitness}")
        except Exception as e:
            print(f"个体{i+1}: 公式 \"{expr}\" -> 评估异常: {e}")


def test_ga_search_small(df, returns):
    """执行小规模遗传算法搜索"""
    print("\n" + "="*80)
    print("测试4: 小规模遗传算法搜索 (2代)")
    print("="*80)
    
    engine = AlphaEngine(df)
    
    # 创建小规模配置
    from dataclasses import dataclass
    @dataclass
    class SmallGAConfig:
        population_size: int = 6
        n_generations: int = 2
        crossover_prob: float = 0.5
        mutation_prob: float = 0.3
        max_tree_depth: int = 2
        n_jobs: int = 1
        ic_weight: float = 1.0
        ir_weight: float = 2.0
        turnover_penalty: float = 0.1
        correlation_penalty: float = 0.3
    
    small_config = SmallGAConfig()
    
    searcher = GAFactorSearcher(engine, returns, config=small_config)
    
    print(f"\n开始搜索: 种群={small_config.population_size}, 代数={small_config.n_generations}")
    
    # 执行搜索
    results = searcher.search()
    
    print(f"\n[搜索结果] 共找到{len(results)}个有效因子")
    print("\n[Top因子列表]")
    for i, item in enumerate(results[:5]):
        print(f"  #{i+1}: fitness={item['fitness']:.4f}, ic={item['ic_mean']:.4f}, "
              f"icir={item['icir']:.4f}, expr={item['expression'][:60]}")
    
    return results


def test_label_alignment():
    """测试标签对齐问题"""
    print("\n" + "="*80)
    print("测试5: Label对齐验证 (关键测试)")
    print("="*80)
    
    # 创建简单的3天2股票数据
    dates = pd.date_range('2024-01-01', periods=3, freq='B')
    stocks = ['Stock_A', 'Stock_B']
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 收盘价数据
    close_data = {
        'close': [100, 200, 105, 210, 110, 220]  # T, T+1, T+2
    }
    
    df = pd.DataFrame(close_data, index=index)
    
    print("\n[收盘价数据]")
    print(df.to_string())
    
    # 计算收益率
    raw_returns = df.groupby(level='instrument')['close'].pct_change()
    print("\n[原始收益率 (当期收益率)]")
    print(raw_returns.to_string())
    
    # 未来一期收益率
    future_returns = raw_returns.shift(-1)
    print("\n[未来一期收益率 (T+1收益率，用于label)]")
    print(future_returns.to_string())
    
    print("\n[关键说明]")
    print("1. 因子值在时间T计算")
    print("2. Label应该是T到T+1的收益率")
    print("3. 使用shift(-1)将未来收益率对齐到当前时间")
    print("4. 最后一天的future_returns会是NaN（因为没有T+1数据）")
    
    # 验证对齐
    print("\n[对齐验证]")
    for date in dates[:-1]:  # 排除最后一天
        for stock in stocks:
            try:
                factor_val = df.loc[(date, stock), 'close']
                label_val = future_returns.loc[(date, stock)]
                actual_return = (df.loc[(date + timedelta(days=1), stock), 'close'] / 
                               df.loc[(date, stock), 'close'] - 1)
                
                print(f"  {date} {stock}: 因子值={factor_val}, "
                      f"label={label_val:.4f}, 实际T+1收益={actual_return:.4f}, "
                      f"匹配={'✅' if abs(label_val - actual_return) < 1e-6 else '❌'}")
            except KeyError:
                pass


def main():
    """主测试流程"""
    print("="*80)
    print("遗传算法因子搜索功能完整测试")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试1: 简单公式解析
    df, factor_values = test_simple_formula()
    
    if df is not None and factor_values is not None:
        # 测试2: ICIR计算
        returns, ic_summary = test_icir_calculation(df, factor_values)
        
        # 测试3: GA初始化
        test_ga_initialization(df, returns)
        
        # 测试4: 小规模GA搜索
        test_ga_search_small(df, returns)
    
    # 测试5: Label对齐（独立测试）
    test_label_alignment()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
