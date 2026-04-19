"""
验证clean文件夹修复是否成功
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试导入是否成功"""
    print("="*80)
    print("测试1: 导入模块")
    print("="*80)
    
    try:
        from clean.ga_search import (
            ExpressionTreeNode,
            ExpressionParser,
            SubtreeCrossover,
            ExpressionGenerator,
            GAFactorSearcher
        )
        print("✅ 所有类导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_expression_parser():
    """测试表达式解析器"""
    print("\n" + "="*80)
    print("测试2: 表达式解析器")
    print("="*80)
    
    try:
        from clean.ga_search import ExpressionParser
        
        # 测试简单表达式
        expr1 = "add(close, open)"
        tree1 = ExpressionParser.parse(expr1)
        print(f"✅ 解析成功: {expr1}")
        print(f"   树结构: {tree1}")
        print(f"   深度: {tree1.depth()}")
        print(f"   大小: {tree1.size()}")
        
        # 测试复杂表达式
        expr2 = "ts_mean(add(close, open), 5)"
        tree2 = ExpressionParser.parse(expr2)
        print(f"\n✅ 解析成功: {expr2}")
        print(f"   树结构: {tree2}")
        print(f"   深度: {tree2.depth()}")
        print(f"   大小: {tree2.size()}")
        
        # 测试转换回字符串
        str1 = ExpressionParser.to_string(tree1)
        print(f"\n✅ 转换回字符串: {str1}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subtree_crossover():
    """测试子树交叉操作"""
    print("\n" + "="*80)
    print("测试3: 子树交叉操作")
    print("="*80)
    
    try:
        from clean.ga_search import SubtreeCrossover
        
        crossover_op = SubtreeCrossover(max_depth=5, max_size=30)
        
        # 测试用例
        expr1 = "add(close, open)"
        expr2 = "mul(volume, 0.5)"
        
        print(f"父个体1: {expr1}")
        print(f"父个体2: {expr2}")
        
        child1, child2 = crossover_op.crossover(expr1, expr2)
        
        print(f"\n✅ 交叉成功")
        print(f"子代1: {child1}")
        print(f"子代2: {child2}")
        
        # 验证语法有效性
        from clean.ga_search import ExpressionParser
        tree1 = ExpressionParser.parse(child1)
        tree2 = ExpressionParser.parse(child2)
        
        print(f"\n✅ 语法验证通过")
        print(f"子代1深度: {tree1.depth()}, 大小: {tree1.size()}")
        print(f"子代2深度: {tree2.depth()}, 大小: {tree2.size()}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置修改"""
    print("\n" + "="*80)
    print("测试4: 配置验证")
    print("="*80)
    
    try:
        from clean.config import GA
        
        print(f"IC权重: {GA.ic_weight}")
        print(f"ICIR权重: {GA.ir_weight}")
        
        if GA.ic_weight == 2.0 and GA.ir_weight == 1.0:
            print("✅ 配置修改正确")
            return True
        else:
            print(f"❌ 配置不符合预期 (应该是 ic_weight=2.0, ir_weight=1.0)")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n" + "="*80)
    print("测试5: 错误处理")
    print("="*80)
    
    try:
        from clean.ga_search import _safe_spearman_ic, _safe_ir
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        test_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 测试异常情况
        result1 = _safe_spearman_ic(test_series, test_series)
        result2 = _safe_ir(test_series, test_series)
        
        print(f"✅ 错误处理正常")
        print(f"  _safe_spearman_ic返回: {result1}")
        print(f"  _safe_ir返回: {result2}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试流程"""
    print("="*80)
    print("Clean文件夹修复验证测试")
    print("="*80)
    print(f"测试时间: {pd.Timestamp.now() if 'pd' in dir() else 'N/A'}")
    
    results = []
    
    # 运行所有测试
    results.append(("导入测试", test_imports()))
    results.append(("表达式解析器", test_expression_parser()))
    results.append(("子树交叉操作", test_subtree_crossover()))
    results.append(("配置验证", test_config()))
    results.append(("错误处理", test_error_handling()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！修复成功！")
        return True
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查")
        return False


if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)
