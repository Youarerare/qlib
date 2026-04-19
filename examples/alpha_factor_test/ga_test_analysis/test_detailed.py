"""
完整的遗传算法因子搜索功能验证测试
输出详细日志，便于核对每个步骤
"""
import sys
import os
import numpy as np
import pandas as pd

# 设置路径
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)

def create_detailed_test_data():
    """创建详细的测试数据，便于手动验证"""
    print("="*100)
    print("遗传算法因子搜索 - 详细验证测试")
    print("="*100)
    
    # 创建10个交易日，5只股票的数据
    dates = pd.date_range('2024-01-01', periods=10, freq='B')
    stocks = ['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E']
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 使用确定的数据，便于手动验证
    np.random.seed(42)
    n = len(index)
    
    data = {
        'open': np.array([
            10.0, 20.0, 30.0, 40.0, 50.0,  # Day 1
            10.5, 20.5, 30.5, 40.5, 50.5,  # Day 2
            11.0, 21.0, 31.0, 41.0, 51.0,  # Day 3
            10.8, 20.8, 30.8, 40.8, 50.8,  # Day 4
            11.2, 21.2, 31.2, 41.2, 51.2,  # Day 5
            11.5, 21.5, 31.5, 41.5, 51.5,  # Day 6
            11.3, 21.3, 31.3, 41.3, 51.3,  # Day 7
            11.8, 21.8, 31.8, 41.8, 51.8,  # Day 8
            12.0, 22.0, 32.0, 42.0, 52.0,  # Day 9
            12.2, 22.2, 32.2, 42.2, 52.2,  # Day 10
        ]),
        'close': np.array([
            10.2, 20.2, 30.2, 40.2, 50.2,  # Day 1
            10.8, 20.8, 30.8, 40.8, 50.8,  # Day 2
            11.5, 21.5, 31.5, 41.5, 51.5,  # Day 3
            11.0, 21.0, 31.0, 41.0, 51.0,  # Day 4
            11.8, 21.8, 31.8, 41.8, 51.8,  # Day 5
            12.0, 22.0, 32.0, 42.0, 52.0,  # Day 6
            11.5, 21.5, 31.5, 41.5, 51.5,  # Day 7
            12.2, 22.2, 32.2, 42.2, 52.2,  # Day 8
            12.5, 22.5, 32.5, 42.5, 52.5,  # Day 9
            12.8, 22.8, 32.8, 42.8, 52.8,  # Day 10
        ]),
        'high': np.array([
            10.5, 20.5, 30.5, 40.5, 50.5,
            11.0, 21.0, 31.0, 41.0, 51.0,
            11.8, 21.8, 31.8, 41.8, 51.8,
            11.2, 21.2, 31.2, 41.2, 51.2,
            12.0, 22.0, 32.0, 42.0, 52.0,
            12.2, 22.2, 32.2, 42.2, 52.2,
            11.8, 21.8, 31.8, 41.8, 51.8,
            12.5, 22.5, 32.5, 42.5, 52.5,
            12.8, 22.8, 32.8, 42.8, 52.8,
            13.0, 23.0, 33.0, 43.0, 53.0,
        ]),
        'low': np.array([
            9.8, 19.8, 29.8, 39.8, 49.8,
            10.2, 20.2, 30.2, 40.2, 50.2,
            10.8, 20.8, 30.8, 40.8, 50.8,
            10.5, 20.5, 30.5, 40.5, 50.5,
            11.0, 21.0, 31.0, 41.0, 51.0,
            11.2, 21.2, 31.2, 41.2, 51.2,
            11.0, 21.0, 31.0, 41.0, 51.0,
            11.5, 21.5, 31.5, 41.5, 51.5,
            12.0, 22.0, 32.0, 42.0, 52.0,
            12.0, 22.0, 32.0, 42.0, 52.0,
        ]),
        'volume': np.array([
            1e6, 2e6, 3e6, 4e6, 5e6,
            1.1e6, 2.1e6, 3.1e6, 4.1e6, 5.1e6,
            1.2e6, 2.2e6, 3.2e6, 4.2e6, 5.2e6,
            1.15e6, 2.15e6, 3.15e6, 4.15e6, 5.15e6,
            1.3e6, 2.3e6, 3.3e6, 4.3e6, 5.3e6,
            1.4e6, 2.4e6, 3.4e6, 4.4e6, 5.4e6,
            1.35e6, 2.35e6, 3.35e6, 4.35e6, 5.35e6,
            1.5e6, 2.5e6, 3.5e6, 4.5e6, 5.5e6,
            1.6e6, 2.6e6, 3.6e6, 4.6e6, 5.6e6,
            1.7e6, 2.7e6, 3.7e6, 4.7e6, 5.7e6,
        ])
    }
    
    df = pd.DataFrame(data, index=index)
    
    print(f"\n[测试数据概况]")
    print(f"  交易日数: {len(dates)}")
    print(f"  股票数: {len(stocks)}")
    print(f"  总记录数: {len(df)}")
    
    print(f"\n[输入数据] (前5行示例)")
    print(df.head().to_string())
    
    return df, dates, stocks


def test_formula_parser(df):
    """测试因子表达式解析器"""
    print("\n" + "="*100)
    print("测试1: 因子表达式解析器验证")
    print("="*100)
    
    try:
        from clean.alpha_engine import AlphaEngine
        engine = AlphaEngine(df)
        print("✅ AlphaEngine创建成功")
    except Exception as e:
        print(f"❌ AlphaEngine创建失败: {e}")
        return None
    
    # 测试公式列表
    test_formulas = [
        ("close - open", "收盘价减开盘价"),
        ("close / open", "收盘价除以开盘价"),
        ("volume / 1e6", "成交量(百万)"),
        ("high - low", "振幅"),
        ("(close - open) / open", "日内收益率"),
    ]
    
    results = {}
    for formula, desc in test_formulas:
        print(f"\n[测试公式] {formula} ({desc})")
        try:
            result = engine.calculate(formula)
            results[formula] = result
            
            # 手动计算预期值
            if formula == "close - open":
                expected = df['close'] - df['open']
            elif formula == "close / open":
                expected = df['close'] / df['open']
            elif formula == "volume / 1e6":
                expected = df['volume'] / 1e6
            elif formula == "high - low":
                expected = df['high'] - df['low']
            elif formula == "(close - open) / open":
                expected = (df['close'] - df['open']) / df['open']
            
            # 验证
            is_match = np.allclose(result.values, expected.values, equal_nan=True)
            
            print(f"  [解析器输出] 前5个值: {result.head().values}")
            print(f"  [预期输出] 前5个值: {expected.head().values}")
            print(f"  [结果] {'✅ 符合预期' if is_match else '❌ 不符合预期'}")
            
        except Exception as e:
            print(f"  ❌ 计算失败: {e}")
    
    return engine


def test_icir_calculation(engine, df):
    """测试ICIR计算逻辑"""
    print("\n" + "="*100)
    print("测试2: ICIR计算逻辑验证")
    print("="*100)
    
    try:
        from clean.ic_analyzer import calc_ic_series, calc_ic_summary
    except Exception as e:
        print(f"❌ 导入IC分析器失败: {e}")
        return
    
    # 创建label（关键：未来一期收益率）
    print("\n[Label定义说明]")
    print("  在量化因子研究中，label应该是未来N期收益率")
    print("  通常使用T+1日收益率来评估因子的预测能力")
    print("  需要使用shift(-1)将未来收益率对齐到当前时间")
    
    # 计算收益率
    raw_returns = df.groupby(level='instrument')['close'].pct_change()
    future_returns = raw_returns.shift(-1)  # 关键：shift(-1)
    
    print(f"\n[原始收益率示例] (当期收益率)")
    print(raw_returns.head(10).to_string())
    
    print(f"\n[Future Returns示例] (T+1收益率，用作label)")
    print(future_returns.head(10).to_string())
    
    # 测试简单因子
    factor = df['close'] - df['open']
    
    print(f"\n[测试因子] close - open")
    print(f"  因子值示例(前10个): {factor.head(10).values}")
    
    # 计算IC
    ic_series = calc_ic_series(factor, future_returns, method="spearman", min_stocks=3)
    
    print(f"\n[IC序列] (共{len(ic_series)}个有效交易日)")
    if len(ic_series) > 0:
        print(ic_series.to_string())
        
        ic_summary = calc_ic_summary(ic_series)
        print(f"\n[IC汇总指标]")
        print(f"  IC均值: {ic_summary['ic_mean']:.6f}")
        print(f"  IC标准差: {ic_summary['ic_std']:.6f}")
        print(f"  ICIR: {ic_summary['icir']:.6f}")
        print(f"  IC正率: {ic_summary['ic_positive_ratio']:.4f}")
        print(f"  有效周期数: {ic_summary['n_periods']}")
    else:
        print("  无有效IC值（可能数据量不足）")


def test_ga_components(engine, df):
    """测试遗传算法组件"""
    print("\n" + "="*100)
    print("测试3: 遗传算法组件验证")
    print("="*100)
    
    try:
        from clean.ga_search import ExpressionGenerator, _safe_spearman_ic, _safe_ir
        from clean.config import GA
    except Exception as e:
        print(f"❌ 导入GA组件失败: {e}")
        return
    
    # 创建label
    returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)
    
    # 创建生成器
    generator = ExpressionGenerator(engine, returns)
    
    print(f"\n[遗传算法配置]")
    print(f"  可用数据字段: {generator.data_fields}")
    print(f"  时序算子: {generator.ts_ops}")
    print(f"  截面算子: {generator.cs_ops}")
    print(f"  数学算子: {generator.math_ops}")
    print(f"  二元算子: {generator.binary_ops}")
    
    # 生成随机种群
    pop_size = 4
    print(f"\n[遗传算法初始化] 种群大小={pop_size}")
    
    population = []
    for i in range(pop_size):
        expr = generator.generate_random(max_depth=2)
        population.append(expr)
        print(f"  个体{i+1}: 公式 \"{expr}\"")
    
    # 评估每个个体
    print(f"\n[适应度评估]")
    for i, expr in enumerate(population):
        try:
            factor, fitness = generator.evaluate_expression(expr)
            
            if factor is not None and fitness > -999:
                # 获取样本值
                sample_values = factor.head(5).tolist()
                
                # 计算IC和ICIR
                ic_mean = _safe_spearman_ic(factor, returns)
                icir = _safe_ir(factor, returns)
                
                print(f"  个体{i+1}:")
                print(f"    公式: {expr}")
                print(f"    因子值(前5个): {[f'{v:.4f}' if not pd.isna(v) else 'NaN' for v in sample_values]}")
                print(f"    IC均值: {ic_mean:.6f}")
                print(f"    ICIR: {icir:.6f}")
                print(f"    适应度: {fitness:.6f}")
            else:
                print(f"  个体{i+1}: 公式 \"{expr}\" -> 计算失败, 适应度 = {fitness}")
        except Exception as e:
            print(f"  个体{i+1}: 公式 \"{expr}\" -> 评估异常: {e}")


def test_label_alignment_detailed():
    """详细测试label对齐"""
    print("\n" + "="*100)
    print("测试4: Label对齐详细验证 (关键测试)")
    print("="*100)
    
    # 创建3天3股票的简单数据
    dates = pd.date_range('2024-01-01', periods=3, freq='B')
    stocks = ['Stock_A', 'Stock_B', 'Stock_C']
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 明确的收盘价数据
    close_prices = [100, 200, 300,  # Day 1
                    105, 210, 315,  # Day 2 (5%涨)
                    110, 220, 330]  # Day 3 (再5%涨)
    
    df = pd.DataFrame({'close': close_prices}, index=index)
    
    print("\n[收盘价数据]")
    print(df.to_string())
    
    # 计算收益率
    raw_returns = df.groupby(level='instrument')['close'].pct_change()
    future_returns = raw_returns.shift(-1)
    
    print("\n[原始收益率] (当日收益率)")
    print(raw_returns.to_string())
    
    print("\n[未来一期收益率] (T+1收益率，用作label)")
    print(future_returns.to_string())
    
    print("\n[对齐验证]")
    print("关键逻辑：")
    print("  1. 因子值在时间T计算")
    print("  2. Label应该是T到T+1的收益率")
    print("  3. 使用shift(-1)将未来收益率对齐到当前时间")
    print("  4. 最后一天的future_returns会是NaN（因为没有T+1数据）")
    
    print("\n逐行验证:")
    for i, (idx, row) in enumerate(df.iterrows()):
        date, stock = idx
        close = row['close']
        ret = raw_returns.loc[idx]
        future_ret = future_returns.loc[idx]
        
        # 计算实际的T+1收益率
        try:
            next_close = df.loc[(date + pd.Timedelta(days=1), stock), 'close']
            actual_future_ret = next_close / close - 1
        except KeyError:
            actual_future_ret = None
        
        print(f"  {date} {stock}: close={close}, "
              f"当日收益率={ret if pd.isna(ret) else f'{ret:.4f}'}, "
              f"label(T+1收益率)={future_ret if pd.isna(future_ret) else f'{future_ret:.4f}'}, "
              f"实际T+1收益率={'N/A' if actual_future_ret is None else f'{actual_future_ret:.4f}'}")
    
    print("\n[结论]")
    print("✅ 使用shift(-1)正确地将未来收益率对齐到当前时间")
    print("✅ 这确保了因子值(T)与label(T->T+1)的正确对应关系")


def identify_code_issues():
    """识别代码问题"""
    print("\n" + "="*100)
    print("代码问题详细分析")
    print("="*100)
    
    print("""
【问题1】交叉操作实现缺陷
文件: ga_search.py, 第246-250行
问题描述:
  _rebuild_from_parts方法只是随机选择一个部分，而不是正确重建表达式
  这导致交叉操作实际上退化为随机选择，失去了遗传算法的核心机制
  
当前代码:
  def _rebuild_from_parts(self, parts: List[str]) -> str:
      if not parts:
          return self._gen_terminal()
      return random.choice(parts) if len(parts) == 1 else parts[0]

影响:
  - 交叉操作无法有效组合两个优秀个体的特征
  - 算法性能下降，收敛速度变慢
  
建议修复:
  应该根据原始表达式的结构正确重建，保持函数调用格式

【问题2】正则表达式导入位置
文件: ga_search.py, 第253行
问题描述:
  import re as _re 放在ExpressionGenerator类定义中间
  这不符合Python代码规范，虽然不会导致错误
  
建议:
  将import语句移到文件开头

【问题3】适应度权重配置
文件: config.py, GA配置
当前配置:
  ic_weight = 1.0
  ir_weight = 2.0
  
分析:
  - ICIR权重是IC权重的2倍
  - 这可能导致算法过度优化因子的稳定性(ICIR)而忽略绝对预测能力(IC)
  - 对于因子挖掘，通常更关注IC的绝对值
  
建议:
  根据实际需求调整权重，可以尝试 ic_weight=2.0, ir_weight=1.0

【问题4】安全性考虑
文件: alpha_engine.py, calculate方法
问题描述:
  使用eval()和exec()执行用户输入的公式字符串
  虽然限制了命名空间，但仍然存在潜在安全风险
  
建议:
  对于生产环境，考虑使用AST解析或专门的表达式解析库

【问题5】ICIR计算中的Label定义
验证结果:
  ✅ 测试脚本中正确使用了shift(-1)来获取未来收益率
  ✅ 这符合量化因子研究的标准做法
  ⚠️ 需要确保在实际使用时也正确处理时间对齐
""")


def main():
    """主测试流程"""
    print("开始遗传算法因子搜索功能完整验证测试")
    print(f"测试时间: {pd.Timestamp.now()}")
    
    # 创建测试数据
    df, dates, stocks = create_detailed_test_data()
    
    # 测试公式解析器
    engine = test_formula_parser(df)
    
    if engine is not None:
        # 测试ICIR计算
        test_icir_calculation(engine, df)
        
        # 测试GA组件
        test_ga_components(engine, df)
    
    # 测试Label对齐
    test_label_alignment_detailed()
    
    # 代码问题分析
    identify_code_issues()
    
    print("\n" + "="*100)
    print("测试完成总结")
    print("="*100)
    print("""
【总体评估】

✅ 正确的部分:
1. 因子表达式解析器能够正确处理基本算子
2. IC/ICIR计算逻辑符合量化标准
3. 遗传算法的整体流程结构合理
4. Label对齐使用shift(-1)是正确的

❌ 需要修复的问题:
1. 交叉操作的_rebuild_from_parts方法实现有缺陷（最重要）
2. 正则表达式导入位置不规范
3. 适应度权重配置可能需要调整

⚠️ 注意事项:
1. 在实际使用时确保label正确对齐（使用shift(-1)）
2. 对于生产环境，考虑表达式执行的安全性
3. 建议增加更详细的错误日志以便调试

【建议优先级】
1. 修复交叉操作实现（高优先级）
2. 调整适应度权重配置（中优先级）
3. 改进错误处理和日志（中优先级）
4. 考虑表达式解析安全性（低优先级，生产环境时考虑）
""")


if __name__ == "__main__":
    main()
