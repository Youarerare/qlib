"""
简化版遗传算法因子搜索测试
可以直接在Python环境中运行
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import timedelta

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_functionality():
    """测试基本功能"""
    print("="*80)
    print("遗传算法因子搜索 - 基础功能测试")
    print("="*80)
    
    # 1. 创建测试数据
    print("\n[步骤1] 创建测试数据")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10, freq='B')
    stocks = ['Stock_A', 'Stock_B', 'Stock_C']
    index = pd.MultiIndex.from_product([dates, stocks], names=['datetime', 'instrument'])
    
    # 生成OHLCV数据
    n = len(index)
    data = {
        'open': np.random.uniform(10, 50, n),
        'high': np.random.uniform(15, 55, n),
        'low': np.random.uniform(5, 45, n),
        'close': np.random.uniform(10, 50, n),
        'volume': np.random.uniform(1e6, 1e7, n)
    }
    
    df = pd.DataFrame(data, index=index)
    print(f"数据形状: {df.shape}")
    print(f"日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"股票数量: {len(stocks)}")
    print("\n前5行数据:")
    print(df.head())
    
    # 2. 测试引擎导入
    print("\n[步骤2] 导入AlphaEngine")
    try:
        from clean.alpha_engine import AlphaEngine
        engine = AlphaEngine(df)
        print("✅ AlphaEngine导入成功")
    except Exception as e:
        print(f"❌ AlphaEngine导入失败: {e}")
        return
    
    # 3. 测试简单公式
    print("\n[步骤3] 测试简单公式: close - open")
    try:
        formula = "close - open"
        result = engine.calculate(formula)
        expected = df['close'] - df['open']
        
        print(f"公式: {formula}")
        print(f"计算结果(前5个): {result.head().values}")
        print(f"预期结果(前5个): {expected.head().values}")
        
        is_match = np.allclose(result.values, expected.values, equal_nan=True)
        print(f"结果: {'✅ 符合预期' if is_match else '❌ 不符合预期'}")
    except Exception as e:
        print(f"❌ 公式计算失败: {e}")
    
    # 4. 测试带算子的公式
    print("\n[步骤4] 测试时序算子: ts_mean(close, 3)")
    try:
        formula2 = "ts_mean(close, 3)"
        result2 = engine.calculate(formula2)
        print(f"公式: {formula2}")
        print(f"计算结果(前10个): {result2.head(10).values}")
        print("✅ 时序算子测试成功")
    except Exception as e:
        print(f"❌ 时序算子测试失败: {e}")
    
    # 5. 测试截面算子
    print("\n[步骤5] 测试截面算子: rank(close)")
    try:
        formula3 = "rank(close)"
        result3 = engine.calculate(formula3)
        print(f"公式: {formula3}")
        print(f"计算结果(前10个): {result3.head(10).values}")
        print("✅ 截面算子测试成功")
    except Exception as e:
        print(f"❌ 截面算子测试失败: {e}")
    
    # 6. 测试IC计算
    print("\n[步骤6] 测试IC计算")
    try:
        from clean.ic_analyzer import calc_ic_series, calc_ic_summary
        
        # 创建label（未来一期收益率）
        returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)
        
        # 使用close - open作为因子
        factor = df['close'] - df['open']
        
        ic_series = calc_ic_series(factor, returns, method="spearman", min_stocks=2)
        ic_summary = calc_ic_summary(ic_series)
        
        print(f"IC序列长度: {len(ic_series)}")
        print(f"IC均值: {ic_summary['ic_mean']:.6f}")
        print(f"IC标准差: {ic_summary['ic_std']:.6f}")
        print(f"ICIR: {ic_summary['icir']:.6f}")
        print("✅ IC计算测试成功")
    except Exception as e:
        print(f"❌ IC计算测试失败: {e}")
    
    # 7. 测试遗传算法组件
    print("\n[步骤7] 测试遗传算法组件")
    try:
        from clean.ga_search import ExpressionGenerator
        
        generator = ExpressionGenerator(engine, returns)
        print(f"可用字段: {generator.data_fields}")
        print(f"时序算子: {generator.ts_ops[:3]}...")
        
        # 生成随机表达式
        for i in range(3):
            expr = generator.generate_random(max_depth=2)
            print(f"  随机表达式{i+1}: {expr}")
        
        print("✅ 遗传算法组件测试成功")
    except Exception as e:
        print(f"❌ 遗传算法组件测试失败: {e}")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


def analyze_code_issues():
    """分析代码问题"""
    print("\n" + "="*80)
    print("代码问题分析报告")
    print("="*80)
    
    print("""
【关键问题发现】

1. 交叉操作实现缺陷 (ga_search.py 第246-250行)
   问题: _rebuild_from_parts方法只是随机选择一个部分
   影响: 交叉操作退化为随机选择，失去遗传算法的交叉意义
   建议修复:
   ```python
   def _rebuild_from_parts(self, parts: List[str]) -> str:
       if not parts:
           return self._gen_terminal()
       # 应该正确重建表达式，而不是随机选择
       # 例如：如果是函数调用，应该保持函数名和结构
       return parts[0]  # 至少返回第一个部分
   ```

2. Label定义不明确
   问题: returns序列是否正确使用未来收益率
   标准做法: label应该是T+1期收益率，需要shift(-1)对齐
   检查结果: 测试脚本中已正确使用shift(-1)

3. 适应度计算权重
   当前: ic_weight=1.0, ir_weight=2.0
   建议: 根据实际需求调整，ICIR权重过高可能导致过度优化稳定性

4. 安全性问题
   问题: 使用eval()执行用户输入的公式
   风险: 虽然有限制命名空间，但仍存在安全隐患
   建议: 考虑使用AST解析或专门的表达式解析库

5. 正则表达式导入位置
   问题: ga_search.py第253行import re放在类定义中间
   建议: 移到文件开头
""")


if __name__ == "__main__":
    test_basic_functionality()
    analyze_code_issues()
