# Alpha公式遗传算法搜索使用指南

## 概述

本项目提供两种遗传算法搜索方式：
1. **批量搜索**：基于Alpha101/191公式集进行搜索
2. **单条公式搜索**：基于单条公式进行变异和进化

---

## 方式1：基于Alpha101/191的批量搜索

### 使用场景
- 想要从Alpha101或Alpha191公式集中搜索最优因子
- 使用已有公式作为初始种群，加速收敛

### 命令行使用

```bash
# 基本使用 - Alpha101
python run_alpha_ga_search.py --alpha-type 101

# Alpha191
python run_alpha_ga_search.py --alpha-type 191

# 两者结合
python run_alpha_ga_search.py --alpha-type both

# 自定义参数
python run_alpha_ga_search.py \
    --alpha-type 101 \
    --instruments csi300 \
    --start-time 2020-01-01 \
    --end-time 2023-06-01 \
    --n-additional 50 \
    --n-generations 30 \
    --output my_results.csv
```

### Python代码使用

```python
from run_alpha_ga_search import run_alpha_based_search

# 运行搜索
results = run_alpha_based_search(
    alpha_type="101",          # "101", "191", 或 "both"
    instruments="csi300",      # 股票池
    start_time="2020-01-01",   # 开始时间
    end_time="2023-06-01",     # 结束时间
    n_additional=50,           # 额外随机个体数
    n_generations=30,          # 进化代数
    output_file="results.csv"  # 输出文件
)

# 查看结果
print(f"找到 {len(results)} 个因子")
print(results[:5])  # Top 5
```

### 工作原理

```
1. 加载Alpha101/191公式
   ↓
2. 使用公式作为初始种群的一部分
   ↓
3. 补充随机生成的个体
   ↓
4. 执行遗传算法进化
   - 选择：锦标赛选择
   - 交叉：子树交叉
   - 变异：4种策略
   ↓
5. 输出Top因子
```

---

## 方式2：单条公式搜索

### 使用场景
- 有一条已知的有效公式
- 想要基于该公式搜索相似的更优因子
- 探索公式的变体

### 命令行使用

```bash
# 基本使用
python run_single_formula_ga.py --formula "add(close, open)"

# 复杂公式
python run_single_formula_ga.py --formula "ts_mean(add(close, open), 5)"

# 自定义参数
python run_single_formula_ga.py \
    --formula "add(close, open)" \
    --instruments csi300 \
    --start-time 2020-01-01 \
    --end-time 2023-06-01 \
    --population-size 100 \
    --n-generations 30 \
    --output single_results.csv
```

### Python代码使用

```python
from run_single_formula_ga import run_single_formula_search

# 运行搜索
results = run_single_formula_search(
    formula="add(close, open)",  # 原始公式
    instruments="csi300",        # 股票池
    start_time="2020-01-01",     # 开始时间
    end_time="2023-06-01",       # 结束时间
    population_size=100,         # 种群大小
    n_generations=30,            # 进化代数
    output_file="single_results.csv"
)

# 查看结果
print(f"找到 {len(results)} 个因子")
for i, item in enumerate(results[:5]):
    print(f"#{i+1}: {item['expression']}")
```

### 工作原理

```
1. 输入单条公式
   ↓
2. 通过变异生成初始种群
   - 30%个体：变异原始公式
   - 70%个体：随机生成
   ↓
3. 执行遗传算法进化
   ↓
4. 输出Top因子（包含原始公式的优化版本）
```

---

## 示例

### 示例1：搜索Alpha101中最优因子

```bash
python run_alpha_ga_search.py --alpha-type 101 --n-generations 50
```

**预期输出**：
```
================================================================================
Top 10 因子
================================================================================

#1:
  适应度: 1.2345
  IC均值: 0.1234
  ICIR:   0.4567
  表达式: add(ts_mean(close, 5), rank(volume))

#2:
  适应度: 1.1234
  IC均值: 0.1123
  ICIR:   0.4234
  表达式: mul(close, ts_rank(volume, 10))
...
```

### 示例2：基于单条公式搜索

```bash
python run_single_formula_ga.py --formula "add(close, open)" --n-generations 30
```

**预期输出**：
```
================================================================================
单条公式遗传算法搜索
================================================================================
原始公式: add(close, open)
加载数据...
创建label...
从公式生成初始种群: add(close, open)
初始种群大小: 100
原始公式: add(close, open)
示例变异: add(close, 5)
开始进化: 30代
  Gen 0: best_fitness=0.8765, expr=add(close, open)
  Gen 5: best_fitness=0.9234, expr=add(ts_mean(close, 5), open)
  Gen 10: best_fitness=1.0123, expr=add(ts_mean(close, 5), rank(volume))
...
搜索完成: 找到95个有效因子
```

### 示例3：批量搜索多个公式

```python
from run_single_formula_ga import run_single_formula_search

# 要搜索的公式列表
formulas = [
    "add(close, open)",
    "sub(high, low)",
    "mul(close, volume)",
    "ts_mean(close, 5)",
    "rank(volume)",
]

# 批量搜索
all_results = {}
for formula in formulas:
    print(f"\n搜索公式: {formula}")
    results = run_single_formula_search(
        formula=formula,
        n_generations=20,
        output_file=f"results_{formula[:20]}.csv"
    )
    all_results[formula] = results

# 汇总结果
print("\n所有搜索完成")
for formula, results in all_results.items():
    print(f"{formula}: 找到{len(results)}个因子, 最佳IC={results[0]['ic_mean']:.4f}")
```

---

## 参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--instruments` | csi300 | 股票池（csi300, csi500等） |
| `--start-time` | 2020-01-01 | 数据开始时间 |
| `--end-time` | 2023-06-01 | 数据结束时间 |
| `--n-generations` | 30 | 进化代数 |
| `--output` | 自动生成 | 输出文件路径 |

### 批量搜索专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha-type` | 101 | 公式类型（101, 191, both） |
| `--n-additional` | 50 | 额外随机个体数 |

### 单条公式专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--formula` | (必需) | 原始公式 |
| `--population-size` | 100 | 种群大小 |

---

## 结果文件说明

### CSV格式

```csv
expression,ic_mean,icir,fitness
"add(close, open)",0.1234,0.4567,1.2345
"ts_mean(close, 5)",0.1123,0.4234,1.1234
...
```

**字段说明**：
- `expression`: 因子表达式
- `ic_mean`: IC均值（预测能力）
- `icir`: ICIR（稳定性）
- `fitness`: 适应度（综合评分）

---

## 常见问题

### Q1: 如何选择alpha-type？

**A**: 
- **Alpha101**：WorldQuant的101个因子，质量较高，推荐初学者使用
- **Alpha191**：国泰君安的191个因子，数量更多，覆盖面更广
- **both**：两者结合，搜索空间更大，但计算时间更长

### Q2: 单条公式搜索 vs 批量搜索？

**A**:
- **单条公式搜索**：适合优化已知有效公式
- **批量搜索**：适合从大量公式中搜索最优解

### Q3: 如何加快搜索速度？

**A**:
1. 减少进化代数：`--n-generations 20`
2. 减少种群大小：`--population-size 50`
3. 减少额外个体：`--n-additional 20`
4. 使用并行计算（修改config.py中的n_jobs）

### Q4: 结果中的IC和ICIR是什么？

**A**:
- **IC（Information Coefficient）**：因子值与未来收益率的相关性，衡量预测能力
- **ICIR（Information Ratio）**：IC均值/IC标准差，衡量稳定性
- **适应度**：`|IC| * ic_weight + |ICIR| * ir_weight`

---

## 性能建议

### 小规模测试
```bash
python run_alpha_ga_search.py --alpha-type 101 --n-generations 10 --n-additional 20
```

### 正式搜索
```bash
python run_alpha_ga_search.py --alpha-type 101 --n-generations 50 --n-additional 50
```

### 深度搜索
```bash
python run_alpha_ga_search.py --alpha-type both --n-generations 100 --n-additional 100
```

---

## 下一步

1. **分析结果**：查看CSV文件，分析Top因子
2. **验证因子**：使用历史数据验证因子有效性
3. **组合因子**：将多个优秀因子组合使用
4. **模型训练**：使用`model_trainer.py`训练预测模型

---

**最后更新**：2026-04-18  
**版本**：v2.0
