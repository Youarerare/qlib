# Alpha因子挖掘与遗传算法搜索

## 项目概述

本项目实现了基于遗传算法的Alpha因子自动搜索系统，支持：
- 自动搜索和组合因子表达式
- 基于IC/ICIR的适应度评估
- Alpha101/Alpha191公式解析
- XGBoost模型训练与对比

## 快速开始

### 1. 基本使用

```python
from clean.run_pipeline import run_full_pipeline

# 运行完整流程
results = run_full_pipeline()
```

### 2. 遗传算法搜索

```python
from clean.data_manager import load_ohlcv
from clean.alpha_engine import AlphaEngine
from clean.ga_search import GAFactorSearcher

# 加载数据
df = load_ohlcv()

# 创建引擎
engine = AlphaEngine(df)

# 创建label（未来一期收益率）
returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)

# 执行搜索
searcher = GAFactorSearcher(engine, returns)
results = searcher.search()
```

### 3. 使用Alpha101/191公式

```python
from clean.formula_parser import load_alpha101_formulas, load_alpha191_formulas
from clean.alpha_engine import compute_factors

# 加载公式
formulas_101 = load_alpha101_formulas()
formulas_191 = load_alpha191_formulas()

# 批量计算因子
factors = compute_factors(df, formulas_101)
```

## 模块说明

### 核心模块

| 模块 | 说明 |
|------|------|
| `config.py` | 全局配置（路径、参数、算子定义） |
| `formula_parser.py` | Alpha101/191公式解析器 |
| `alpha_engine.py` | 因子计算引擎（支持丰富算子） |
| `ic_analyzer.py` | IC/ICIR分析器 |
| `data_manager.py` | 数据管理（qlib数据加载） |
| `ga_search.py` | 遗传算法因子搜索 |
| `model_trainer.py` | XGBoost模型训练与对比 |
| `run_pipeline.py` | 一键执行流水线 |

### 遗传算法特性

**子树交叉操作**（已优化）：
- 真正实现信息交换
- 能够组合两个父个体的优秀特征
- 保持语法有效性
- 符合遗传规划（GP）标准

**适应度评估**：
- 使用Rank IC（Spearman相关系数）
- 基于未来一期收益率（T+1）
- 支持IC和ICIR加权
- 包含相关性惩罚

**遗传算子**：
- 选择：锦标赛选择
- 交叉：子树交叉（Subtree Crossover）
- 变异：4种策略（替换子树、改变窗口、改变字段、改变操作符）
- 精英保留：保留每代最优个体

## 配置说明

### 遗传算法配置（config.py）

```python
@dataclass
class GAConfig:
    population_size: int = 200      # 种群大小
    n_generations: int = 50         # 进化代数
    crossover_prob: float = 0.7     # 交叉概率
    mutation_prob: float = 0.2      # 变异概率
    max_tree_depth: int = 5         # 最大树深度
    ic_weight: float = 1.0          # IC权重
    ir_weight: float = 2.0          # ICIR权重
    correlation_penalty: float = 0.3  # 相关性惩罚
```

### 支持的算子

**时序算子**：
- `ts_sum`, `ts_mean`, `ts_std_dev`, `ts_min`, `ts_max`
- `ts_rank`, `ts_delta`, `ts_delay`, `ts_corr`, `ts_covariance`
- `ts_scale`, `ts_decay_linear`, `ts_arg_max`, `ts_arg_min`
- `ts_product`, `ts_av_diff`

**截面算子**：
- `rank`, `scale`

**数学算子**：
- `abs`, `log`, `sign`, `sqrt`, `signed_power`

**二元算子**：
- `max`, `min`（元素级）

**逻辑算子**：
- `if_else`

## 使用示例

### 示例1：简单因子搜索

```python
# 搜索简单因子
from clean.ga_search import ExpressionGenerator

generator = ExpressionGenerator(engine, returns)

# 生成随机表达式
expr = generator.generate_random(max_depth=2)
print(f"随机表达式: {expr}")

# 评估表达式
factor, fitness = generator.evaluate_expression(expr)
print(f"适应度: {fitness}")
```

### 示例2：基于Alpha101公式搜索

```python
# 加载Alpha101公式
from clean.formula_parser import load_alpha101_formulas

formulas = load_alpha101_formulas()

# 使用公式作为初始种群
from clean.ga_search import GAFactorSearcher

searcher = GAFactorSearcher(engine, returns)

# 可以用已有公式初始化种群
# （需要修改search方法支持自定义初始种群）
```

### 示例3：批量计算和评估

```python
# 批量计算因子
from clean.alpha_engine import compute_factors
from clean.ic_analyzer import evaluate_all_factors

# 计算公式
factors = compute_factors(df, formulas)

# 评估因子
results = evaluate_all_factors(factors, returns)
print(results.head())
```

## 修复历史

### 2026-04-18 重要更新

**遗传算法交叉操作优化**：
- ✅ 实现了真正的子树交叉（Subtree Crossover）
- ✅ 修复了原有的交叉操作缺陷
- ✅ 添加了表达式解析器
- ✅ 改进了错误处理

**详见**：`FIXES_APPLIED.md`

## 性能建议

1. **数据准备**：
   - 确保有足够的时间序列数据（建议>1年）
   - 股票池建议50-300只

2. **参数调优**：
   - 小规模测试：`population_size=50, n_generations=20`
   - 正式搜索：`population_size=200, n_generations=50`

3. **性能优化**：
   - 使用并行计算：设置`n_jobs > 1`
   - 添加表达式缓存（已实现）
   - 限制表达式复杂度：`max_tree_depth=5`

## 常见问题

### Q1: Label是什么？

**A**: Label是未来一期收益率（T+1日收益率），使用`shift(-1)`对齐。

```python
returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)
```

### Q2: 如何自定义算子？

**A**: 在`AlphaEngine._setup_operator_map`中添加：

```python
def _setup_operator_map(self):
    self.ts_ops['my_custom_op'] = self.my_custom_op
    
def my_custom_op(self, s, w):
    # 实现自定义逻辑
    pass
```

### Q3: 交叉操作如何工作？

**A**: 使用子树交叉：
1. 将表达式解析为树结构
2. 随机选择两个父个体的子树
3. 交换子树
4. 转换回字符串

示例：
```
父个体1: add(close, open)
父个体2: mul(volume, 0.5)

子代1: add(close, 0.5)    # 交换了子树
子代2: mul(volume, open)  # 交换了子树
```

## 参考资料

- Alpha101公式：WorldQuant的101个Alpha因子
- Alpha191公式：国泰君安191个Alpha因子
- 遗传规划：Genetic Programming (GP)
- IC/ICIR：信息系数/信息比率

## 许可证

本项目仅供学习和研究使用。

---

**最后更新**：2026-04-18  
**版本**：v2.0（包含交叉操作优化）
