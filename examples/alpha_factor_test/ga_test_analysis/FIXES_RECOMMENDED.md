# 遗传算法因子搜索 - 代码修复建议

本文档提供具体的代码修复方案，按优先级排序。

---

## 🔴 高优先级修复

### 修复1：交叉操作的 `_rebuild_from_parts` 方法

**文件**：`clean/ga_search.py`  
**位置**：第246-250行  
**严重程度**：高（影响遗传算法核心功能）

#### 当前代码（有问题）

```python
def _rebuild_from_parts(self, parts: List[str]) -> str:
    """从拆分部分重建表达式"""
    if not parts:
        return self._gen_terminal()
    return random.choice(parts) if len(parts) == 1 else parts[0]
```

#### 问题分析

1. 当 `len(parts) == 1` 时，随机选择（但只有一个选择，所以总是返回它）
2. 当 `len(parts) > 1` 时，总是返回 `parts[0]`，失去了随机性
3. 没有真正"重建"表达式，只是选择了一个部分

#### 修复方案A（简单修复）

```python
def _rebuild_from_parts(self, parts: List[str]) -> str:
    """从拆分部分重建表达式"""
    if not parts:
        return self._gen_terminal()
    
    # 过滤空字符串
    valid_parts = [p for p in parts if p.strip()]
    if not valid_parts:
        return self._gen_terminal()
    
    # 随机选择一个有效的子表达式
    # 这保持了遗传算法的随机性和多样性
    return random.choice(valid_parts)
```

#### 修复方案B（更完整的修复）

```python
def _rebuild_from_parts(self, parts: List[str], keep_structure: bool = True) -> str:
    """
    从拆分部分重建表达式
    
    Parameters
    ----------
    parts : List[str]
        拆分后的表达式部分
    keep_structure : bool
        是否保持函数调用结构（默认True）
    """
    if not parts:
        return self._gen_terminal()
    
    # 过滤空字符串
    valid_parts = [p.strip() for p in parts if p.strip()]
    if not valid_parts:
        return self._gen_terminal()
    
    # 如果只有一个部分，直接返回
    if len(valid_parts) == 1:
        return valid_parts[0]
    
    # 如果有多个部分，随机选择一个作为新的子树
    # 这模拟了遗传算法中的子树交换
    selected = random.choice(valid_parts)
    
    # 验证选择的部分是否有效
    if selected and selected.strip():
        return selected.strip()
    else:
        return self._gen_terminal()
```

#### 修复方案C（最完整，保持函数结构）

如果你想保持函数调用的完整结构：

```python
def _rebuild_from_parts(self, parts: List[str], original_func: str = None) -> str:
    """
    从拆分部分重建表达式，尽量保持函数结构
    
    Parameters
    ----------
    parts : List[str]
        拆分后的表达式部分，第一个元素通常是函数名和开括号
    original_func : str, optional
        原始函数名（如果知道的话）
    """
    if not parts:
        return self._gen_terminal()
    
    # 过滤空字符串
    valid_parts = [p.strip() for p in parts if p.strip()]
    if not valid_parts:
        return self._gen_terminal()
    
    # 检查第一部分是否包含函数调用
    first_part = valid_parts[0]
    if '(' in first_part and ')' in first_part:
        # 这是一个完整的函数调用，直接返回
        return first_part
    
    # 否则随机选择一个部分
    return random.choice(valid_parts)
```

#### 推荐方案

**推荐使用方案A**，原因：
1. 简单有效
2. 保持随机性
3. 不会引入新的bug
4. 符合遗传算法的子树替换思想

---

## 🟡 中优先级修复

### 修复2：正则表达式导入位置

**文件**：`clean/ga_search.py`  
**位置**：第253行

#### 当前代码（不规范）

```python
class ExpressionGenerator:
    # ... 类定义 ...

import re as _re  # ❌ 放在类定义中间

def re_findall_numbers(s: str) -> List[str]:
    return _re.findall(r"\b(\d+)\b", s)
```

#### 修复后

在文件开头添加导入（第1-10行之间）：

```python
"""
基于DEAP的遗传算法因子搜索
"""
import logging
import random
import operator
import re  # ✅ 移到这里
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import GA, BACKTEST, ALL_OPERATORS, TS_OPERATORS, CS_OPERATORS, DATA_FIELDS, ADV_FIELDS
from .alpha_engine import AlphaEngine
from .ic_analyzer import calc_ic_series, calc_ic_summary

logger = logging.getLogger(__name__)
```

然后删除第253行的 `import re as _re`，并将函数改为：

```python
def re_findall_numbers(s: str) -> List[str]:
    return re.findall(r"\b(\d+)\b", s)
```

---

### 修复3：适应度权重配置

**文件**：`clean/config.py`  
**位置**：第42-53行

#### 当前配置

```python
@dataclass
class GAConfig:
    population_size: int = 200
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    max_tree_depth: int = 5
    n_jobs: int = 4
    ic_weight: float = 1.0      # ⚠️ IC权重较低
    ir_weight: float = 2.0      # ⚠️ ICIR权重较高
    turnover_penalty: float = 0.1
    correlation_penalty: float = 0.3
```

#### 建议配置A（更关注预测能力）

```python
@dataclass
class GAConfig:
    population_size: int = 200
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    max_tree_depth: int = 5
    n_jobs: int = 4
    ic_weight: float = 2.0      # ✅ 提高IC权重
    ir_weight: float = 1.0      # ✅ 降低ICIR权重
    turnover_penalty: float = 0.1
    correlation_penalty: float = 0.3
```

#### 建议配置B（平衡配置）

```python
@dataclass
class GAConfig:
    # ... 其他配置 ...
    ic_weight: float = 1.5      # 适中
    ir_weight: float = 1.5      # 适中
    # ... 其他配置 ...
```

#### 选择建议

- **因子挖掘阶段**：使用配置A（更关注IC）
- **因子优化阶段**：使用配置B（平衡）
- **当前配置**：适合需要高稳定性的场景

---

## 🟢 低优先级改进

### 改进1：错误处理细化

**文件**：`clean/ga_search.py`  
**位置**：第20-41行

#### 当前代码

```python
def _safe_spearman_ic(factor_values: pd.Series, returns: pd.Series) -> float:
    """安全计算Rank IC均值"""
    try:
        ic_s = calc_ic_series(factor_values, returns, method="spearman", min_stocks=10)
        if len(ic_s) < 20:
            return 0.0
        summary = calc_ic_summary(ic_s)
        return summary["ic_mean"]
    except Exception:
        return 0.0
```

#### 改进后

```python
def _safe_spearman_ic(factor_values: pd.Series, returns: pd.Series) -> float:
    """安全计算Rank IC均值"""
    try:
        ic_s = calc_ic_series(factor_values, returns, method="spearman", min_stocks=10)
        if len(ic_s) < 20:
            logger.debug(f"IC序列过短: {len(ic_s)} < 20")
            return 0.0
        summary = calc_ic_summary(ic_s)
        return summary["ic_mean"]
    except Exception as e:
        logger.debug(f"IC计算失败: {e}")
        return 0.0
```

---

### 改进2：增加类型注解

**文件**：`clean/ga_search.py`

#### 示例

```python
def mutate(self, expr: str) -> str:
    """
    变异操作
    
    Parameters
    ----------
    expr : str
        原始表达式
    
    Returns
    -------
    str
        变异后的表达式
    """
    ops = ["replace_subtree", "change_window", "change_field", "change_op"]
    op = random.choice(ops)
    # ... 其余代码 ...
```

---

### 改进3：增加日志输出

**文件**：`clean/ga_search.py`  
**位置**：search方法

#### 改进建议

在关键步骤增加日志：

```python
def search(self, existing_factors: Optional[List[pd.Series]] = None) -> List[Dict]:
    """执行遗传算法搜索"""
    if existing_factors:
        self.generator.set_existing_factors(existing_factors)

    logger.info(f"开始遗传算法搜索: 种群={self.cfg.population_size}, 代数={self.cfg.n_generations}")
    logger.info(f"适应度权重: IC={self.cfg.ic_weight}, ICIR={self.cfg.ir_weight}")

    # 初始化种群
    population = []
    for i in range(self.cfg.population_size):
        expr = self.generator.generate_random(max_depth=self.cfg.max_tree_depth)
        population.append(expr)
    
    logger.info(f"种群初始化完成: {len(population)}个个体")

    # 进化
    for gen in range(self.cfg.n_generations):
        # ... 其余代码 ...
```

---

## 📋 完整修复清单

### 必须修复（高优先级）

- [ ] 修复 `_rebuild_from_parts` 方法（ga_search.py 第246-250行）

### 建议修复（中优先级）

- [ ] 移动正则表达式导入到文件开头（ga_search.py 第253行）
- [ ] 调整适应度权重配置（config.py 第49-50行）

### 可选改进（低优先级）

- [ ] 细化错误处理，增加日志
- [ ] 增加类型注解和文档字符串
- [ ] 增加关键步骤的日志输出
- [ ] 考虑表达式解析的安全性（生产环境）

---

## 🧪 修复后验证

修复完成后，运行测试脚本验证：

```bash
cd C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\ga_test_analysis
python final_test.py
```

检查点：
1. ✅ 所有测试通过
2. ✅ 交叉操作能够正确组合表达式
3. ✅ 遗传算法能够收敛到更优解
4. ✅ 日志输出清晰完整

---

## 📝 注意事项

1. **备份原始代码**：修复前请备份clean文件夹
2. **逐步修复**：建议先修复高优先级问题，验证后再修复其他问题
3. **测试验证**：每次修复后都要运行测试脚本
4. **性能监控**：修复交叉操作后，监控算法收敛速度

---

**创建时间**：2026-04-18  
**适用版本**：clean文件夹当前版本  
**状态**：待修复
