# W式切割反转因子 - WorldQuant Brain实现

## 理论背景

W式切割反转因子由魏建榕、傅开波在2018年提出，核心思想：
- 将过去20日涨跌幅按照"平均单笔成交金额"进行切割
- 高平均单笔成交金额的10日涨跌幅加总 → M_high
- 低平均单笔成交金额的10日涨跌幅加总 → M_low
- 理想反转因子 M = M_high - M_low

研究发现：
- 反转效应主要来源于大单成交
- 使用高分位作为切割标准效果更好
- M_high因子比传统Ret20更稳健

## WorldQuant Brain平台的限制

### 可用数据字段（pv1数据集）：
- `close` - 收盘价
- `open` - 开盘价
- `high` - 最高价
- `low` - 最低价
- `volume` - 成交量（股数）
- `vwap` - 成交量加权平均价
- `returns` - 日收益率

### 核心限制：
❌ 缺少"成交笔数"（number of trades）字段
❌ 无法直接计算"平均单笔成交金额"

## 可行实现方案

### 方案1：基于成交量的切割（近似方案）

```python
# 传统反转因子作为基准
alpha_ret20 = -ts_sum(returns, 20)

# 使用成交量作为切割标准（替代平均单笔成交金额）
# 计算过去20日每日成交量排名
vol_rank = ts_rank(volume, 20)

# 定义高低成交量日（前10日为高，后10日为低）
# 高成交量日的收益加总
M_high_vol = ts_sum(returns * (vol_rank > 10), 20)
# 或者用成交量权重
M_high_vol_weighted = ts_sum(returns * rank(volume), 20)

# 低成交量日的收益加总
M_low_vol = ts_sum(returns * (vol_rank <= 10), 20)
# 或者用反向成交量权重
M_low_vol_weighted = ts_sum(returns * (20 - rank(volume)), 20)

# 理想反转因子（成交量切割版）
M_vol = M_high_vol_weighted - M_low_vol_weighted
```

**WorldQuant Brain表达式：**
```
-ts_sum(returns * ts_rank(volume, 20), 20) + ts_sum(returns * (20 - ts_rank(volume, 20)), 20)
```

简化版：
```
2 * ts_sum(returns * (2 * ts_rank(volume, 20) - 20), 20)
```

### 方案2：基于VWAP的切割

VWAP（成交量加权平均价）可以反映大单交易的影响：

```python
# VWAP相对于收盘价的偏离度
vwap_premium = (vwap - close) / close

# VWAP偏离度高的日子，可能有更多大单交易
vwap_rank = ts_rank(vwap_premium, 20)

# 按VWAP偏离度切割
M_high_vwap = ts_sum(returns * (vwap_rank > 10), 20)
M_low_vwap = ts_sum(returns * (vwap_rank <= 10), 20)
M_vwap = M_high_vwap - M_low_vwap
```

**WorldQuant Brain表达式：**
```
ts_sum(returns * ts_rank((vwap - close) / close, 20), 20) - ts_sum(returns * (20 - ts_rank((vwap - close) / close, 20)), 20)
```

### 方案3：基于成交金额估算

如果平台有成交额字段（amount/turnover），可以使用：

```python
# 假设有amount字段
# 平均单笔成交金额 ≈ amount / volume (粗略估计，假设每笔交易规模)

# 但WorldQuant Brain可能没有amount字段
# 可以用 volume * vwap 估算成交金额
estimated_amount = volume * vwap

# 按估算成交金额排名
amount_rank = ts_rank(estimated_amount, 20)

# 高成交金额日
M_high_amt = ts_sum(returns * (amount_rank > 10), 20)
# 低成交金额日
M_low_amt = ts_sum(returns * (amount_rank <= 10), 20)
M_amount = M_high_amt - M_low_amt
```

**WorldQuant Brain表达式：**
```
ts_sum(returns * ts_rank(volume * vwap, 20), 20) - ts_sum(returns * (20 - ts_rank(volume * vwap, 20)), 20)
```

### 方案4：高阶改进版（推荐）

结合多个维度：

```python
# 1. 成交金额估算
amount_proxy = volume * vwap

# 2. 成交金额的变化（相对于均值的偏离）
amount_ma = ts_mean(amount_proxy, 20)
amount_deviation = amount_proxy / amount_ma - 1

# 3. 按偏离度排名（高偏离度可能意味着大单活跃）
deviation_rank = ts_rank(amount_deviation, 20)

# 4. 只取高分位（如前13/16，约81%分位）
threshold = 13 / 16
M_high_13_16 = ts_sum(returns * (deviation_rank > 20 * threshold), 20)

# 5. 或者使用加权
M_weighted = ts_sum(returns * deviation_rank, 20)
```

**WorldQuant Brain表达式（简化版）：**
```
-ts_sum(returns * ts_rank((volume * vwap) / ts_mean(volume * vwap, 20), 20), 20)
```

## 推荐在WorldQuant Brain测试的表达式

### 表达式1：基础版（基于成交金额估算）
```
rank(ts_sum(returns * ts_rank(volume * vwap, 20), 20) - ts_sum(returns * (20 - ts_rank(volume * vwap, 20)), 20))
```

### 表达式2：改进版（高分位切割）
```
-rank(ts_sum(returns * power(ts_rank(volume * vwap, 20), 2), 20))
```
说明：通过平方操作，给高成交金额日更大权重，模拟高分位切割效果

### 表达式3：成交金额偏离度版
```
-rank(ts_sum(returns * ts_rank((volume * vwap) / ts_mean(volume * vwap, 20) - 1, 20), 20))
```

### 表达式4：VWAP切割版
```
rank(ts_sum(returns * ts_rank(abs(vwap - close) / close, 20), 20))
```

## 验证建议

1. **对比测试**：
   - 测试传统Ret20因子：`-ts_sum(returns, 20)`
   - 测试W式切割版本（上述表达式）
   - 比较IR、月度胜率、最大回撤

2. **参数优化**：
   - 尝试不同的时间窗口（15日、20日、30日）
   - 尝试不同的切割比例（1/2, 13/16等）

3. **稳健性检验**：
   - 不同市场环境下的表现
   - 不同股票池的效果

## 注意事项

1. **数据限制**：WorldQuant Brain的pv1数据集缺少成交笔数，只能用代理变量
2. **逻辑差异**：用成交金额替代平均单笔成交金额，逻辑上有一定偏差
3. **平台特性**：WorldQuant Brain的表达式语法可能需要调整
4. **过拟合风险**：避免过度优化参数

## 理论启示

虽然无法完全复现W式切割，但理论研究给我们的启示：

1. **反转效应有微观结构**：不同交易行为产生的反转强度不同
2. **大单信息含量高**：机构投资者行为对未来收益预测力更强
3. **因子改进思路**：可以通过交易行为特征改进传统因子
4. **避免因子拥挤**：改进后的因子可能避免传统因子的拥挤和回撤

## 参考资源

- 原始论文：魏建榕、傅开波（2018）《W式切割反转因子》
- WorldQuant Brain平台：https://platform.worldquantbrain.com
- WorldQuant 101 Alpha因子库：用于参考表达式语法
