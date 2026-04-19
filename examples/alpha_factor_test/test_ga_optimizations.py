"""
测试优化后的GA因子搜索
验证:
1. 无意义表达式检测
2. 结构去重（系数缩放等价）
3. 避免生成常数乘法
"""
import sys
from pathlib import Path

# 添加路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clean.ga_search import _is_meaningless_expression, _extract_structure_signature

print("=" * 80)
print("测试1: 无意义表达式检测")
print("=" * 80)

test_cases_meaningless = [
    "(close - close)",
    "(high - high)",
    "ts_delta((close - close), 1)",
    "123",  # 纯常数
    "1 + 2 * 3",  # 纯常数运算
]

test_cases_meaningful = [
    "(close - open)",
    "(high - low)",
    "ts_delta(close, 1)",
    "ts_mean(volume, 10)",
    "sqrt(abs(close - open))",
]

print("\n应该被检测为无意义的表达式:")
for expr in test_cases_meaningless:
    result = _is_meaningless_expression(expr)
    status = "✓ 正确检测" if result else "✗ 未检测到"
    print(f"  {status}: {expr}")

print("\n应该被检测为有意义的表达式:")
for expr in test_cases_meaningful:
    result = _is_meaningless_expression(expr)
    status = "✓ 正确检测" if not result else "✗ 误判"
    print(f"  {status}: {expr}")

print("\n" + "=" * 80)
print("测试2: 结构签名提取（系数缩放等价检测）")
print("=" * 80)

test_cases_structure = [
    ("-20 * ts_delta(((close - low) - (open - close)) / (high - low), 1)",),
    ("-1 * ts_delta(((close - low) - (open - close)) / (high - low), 1)",),
    ("-60 * ts_delta(((close - low) - (open - close)) / (high - low), 1)",),
    ("ts_delta(close, 10)",),
    ("ts_delta(close, 20)",),
    ("sqrt(scale(ts_zscore(max(sign(high), cs_mean(adv80)), 10)))",),
    ("sqrt(scale(ts_zscore(max(sign(high), cs_mean(adv80)), 60)))",),
]

print("\n表达式及其结构签名:")
signatures = []
for (expr,) in test_cases_structure:
    sig = _extract_structure_signature(expr)
    signatures.append(sig)
    print(f"  原始: {expr[:70]}")
    print(f"  签名: {sig[:70]}")
    print()

# 检测等价组
print("等价因子组检测:")
from collections import Counter
sig_counts = Counter(signatures)
for sig, count in sig_counts.items():
    if count > 1:
        print(f"  发现 {count} 个等价因子 (签名: {sig[:50]}...)")

print("\n" + "=" * 80)
print("测试3: 实际GA搜索验证")
print("=" * 80)
print("\n建议运行以下命令验证完整流程:")
print("  python -m factor_library.batch_search --ga --ga-per-seed")
print("\n优化效果应该体现在:")
print("  1. 日志中不再出现 (close-close) 等无意义表达式")
print("  2. Top因子列表中不会出现仅系数不同的重复因子")
print("  3. 每代日志显示 structures 数量，反映真实多样性")
print("  4. 最终结果会显示结构去重统计信息")
