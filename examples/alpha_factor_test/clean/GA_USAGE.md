# 遗传算法因子搜索 - 使用指南

## 📊 当前支持的字段和算子

### ✅ 支持的基础字段 (7个)
```python
open, high, low, close, volume, vwap, returns
```

### ✅ 支持的成交量字段 (12个)
```python
adv5, adv10, adv15, adv20, adv30, adv40, 
adv50, adv60, adv80, adv120, adv150, adv180
```

### ✅ 支持的时间序列算子 (17个)
```python
ts_sum, ts_mean, ts_std_dev, ts_min, ts_max, 
ts_rank, ts_delta, ts_delay, ts_corr, ts_covariance,
ts_scale, ts_decay_linear, ts_arg_max, ts_arg_min, 
ts_product, ts_av_diff, ts_zscore
```

### ✅ 支持的截面算子 (3个)
```python
rank, scale, cs_mean
```

### ✅ 支持的数学算子 (5个)
```python
abs, log, sign, sqrt, signed_power (或 power)
```

### ✅ 支持的二元算子 (6个)
```python
add, subtract, multiply, divide, max, min
```

### ✅ 支持的逻辑算子 (1个)
```python
if_else
```

---

## 🚀 使用方式

### 方式1：完全随机搜索（默认）

```python
from clean.search_one import run_single_formula_search

results = run_single_formula_search(
    n_generations=30,
    population_size=100,
)
```

### 方式2：从种子公式开始搜索 ⭐ **推荐**

修改 `search_one.py` 的 `__main__` 部分：

```python
if __name__ == "__main__":
    # 你的种子公式
    seed_formula = "-ts_decay_linear(power((close / ts_delay(close, 1) - 1) - ts_mean(close / ts_delay(close, 1) - 1, 60), 2), 20)"
    
    results = run_single_formula_search(
        formula=seed_formula,  # 作为种子公式
        n_generations=50,
        population_size=200,
    )
```

**注意**：当前版本会评估种子公式，但**不会直接变异种子公式**。种子公式会作为基准对比。

---

## 🔧 你的公式修正

你的公式：
```
-ts_decay_linear(power((close / ts_delay(close, 1) - 1) - ts_mean(close / ts_delay(close, 1) - 1, 60), 2), 20)
```

**已支持！** 所有算子都在支持列表中：
- ✅ `ts_decay_linear` - 线性衰减
- ✅ `power` - 幂运算（别名，实际调用 `signed_power`）
- ✅ `ts_delay` - 延迟
- ✅ `ts_mean` - 均值
- ✅ 基础字段：`close`

---

## 📝 完整示例

### 示例1：评估单个公式

```python
from clean.search_one import evaluate_single_formula

result = evaluate_single_formula(
    formula="-ts_decay_linear(power((close / ts_delay(close, 1) - 1) - ts_mean(close / ts_delay(close, 1) - 1, 60), 2), 20)",
    data_months=3,        # IC评估用3个月
    load_data_months=12   # 加载12个月数据
)

print(f"IC: {result['ic']:.4f}")
print(f"RankICIR: {result['rank_icir']:.4f}")
```

### 示例2：从种子公式开始搜索

修改 `ga_search.py` 的 `GAFactorSearcher.search()` 调用：

```python
# 在 search_one.py 中
seed_formulas = [
    "-ts_decay_linear(power((close / ts_delay(close, 1) - 1) - ts_mean(close / ts_delay(close, 1) - 1, 60), 2), 20)",
    "rank(ts_corr(rank(high), rank(ts_mean(volume, 15)), 9))",
]

# 将种子公式传入
results = searcher.search(seed_formulas=seed_formulas)
```

---

## 🎯 典型因子表达式

### Alpha101 风格
```python
# 量价相关性
"-1 * rank(ts_corr(rank(high), rank(volume), 5))"

# 动量反转
"ts_delta(close, 5) / ts_std_dev(close, 20)"

# 波动率调整收益
"(close - ts_mean(close, 20)) / ts_std_dev(close, 20)"
```

### Alpha191 风格
```python
# 成交量加权价格
"ts_corr(vwap, volume, 10)"

# 价格趋势
"ts_regression(close, ts_step(1), 20, rettype=2)"
```

---

## ⚠️ 注意事项

1. **字段大小写**：全部小写（`close` 不是 `Close`）
2. **除法保护**：`divide(a, b)` 会自动处理除零（返回 NaN）
3. **power 算子**：实际调用 `signed_power`，处理负数幂
4. **数据范围**：
   - IC评估：默认最近 3 个月
   - 数据加载：默认 IC起始日期 前推 12 个月

---

## 📈 性能优化建议

```python
# 快速测试（5分钟）
run_single_formula_search(
    n_generations=20,
    population_size=50,
    data_months=3,
)

# 标准搜索（30分钟）
run_single_formula_search(
    n_generations=50,
    population_size=100,
    data_months=3,
)

# 深度搜索（2小时）
run_single_formula_search(
    n_generations=100,
    population_size=200,
    data_months=6,
    load_data_months=18,
)
```

---

## 🐛 常见问题

### Q1: 公式计算失败
```
公式执行失败: xxx | 错误: 字段不存在: xxx
```
**解决**：检查字段名是否拼写错误（必须小写）

### Q2: 内存泄漏
**解决**：已修复！缓存限制为 5000 条，每 5 代自动 GC

### Q3: 种群早熟
**解决**：已实现早停机制，连续 10 代停滞自动重启 50% 种群
