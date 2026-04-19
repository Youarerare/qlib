# XGBoost Top50因子模型 vs Alpha158

## 概述

本项目实现了使用经过ICIR筛选的Top50因子（来自Alpha101和Alpha191）作为特征，训练XGBoost模型，并与默认的Alpha158基线进行对比。

## 核心思想

**为什么Top50可能比Alpha158更强？**

- Alpha158: 158个固定因子，未经筛选，存在大量冗余
- Top50因子: 经过ICIR严格筛选，保留最有预测能力的因子
- 因子质量 > 因子数量

## 快速开始

### 方案1: 预计算特征 + XGBoost训练（推荐）

```bash
# 1. 预计算Top50因子特征（约10-20分钟）
python topk_alpha_handler.py --prepare

# 2. 运行XGBoost Top50模型
python qlib/workflow/cli.py run workflow_config_xgboost_top50.yaml

# 3. 查看结果
# 结果在 experiments/ 目录下
```

### 方案2: 端到端快速测试

```bash
python run_top50_xgboost.py
```

### 方案3: 对比测试（Alpha158 vs Top50）

```bash
python run_xgboost_comparison.py
```

### 查看Top50因子列表

```bash
python run_top50_xgboost_simple.py
```

## 文件说明

### 核心模块
- `formula_parser.py` - Alpha101/191公式解析器
- `alpha_calculator.py` - 因子计算器
- `icir_calculator.py` - ICIR计算器
- `run_alpha_icir_test.py` - Alpha因子ICIR回测主脚本

### Top50 XGBoost相关
- `topk_alpha_handler.py` - Top50因子预计算工具
- `workflow_config_xgboost_top50.yaml` - XGBoost Top50模型配置
- `run_top50_xgboost.py` - 端到端Top50 XGBoost脚本
- `run_xgboost_comparison.py` - 对比回测脚本
- `run_top50_xgboost_simple.py` - 快速查看Top50因子列表

### 数据文件
- `top50_by_rank_icir.csv` - Top50因子列表（按Rank ICIR排序）
- `top50_by_icir.csv` - Top50因子列表（按ICIR排序）
- `all_alpha_results.csv` - 全部Alpha101+191因子回测结果
- `top50_features.pkl` - 预计算的Top50因子特征（中间文件）
- `xgboost_comparison_results.csv` - XGBoost对比结果

### 文档
- `IMPLEMENTATION_PLAN.txt` - 完整实现方案
- `README.md` - 本文件

## Top50因子表现

| 排名 | 因子名 | 来源 | IC均值 | ICIR | RankICIR |
|------|--------|------|--------|------|----------|
| 1 | alpha171 | alpha191 | 0.0359 | 0.3506 | **0.2788** |
| 2 | alpha083 | alpha101 | 0.0231 | 0.1647 | 0.2439 |
| 3 | alpha114 | alpha191 | 0.0231 | 0.1646 | 0.2438 |
| 4 | alpha054 | alpha101 | 0.0209 | 0.1556 | 0.2434 |
| 5 | alpha100 | alpha101 | 0.0118 | 0.0826 | 0.2305 |

## XGBoost配置

### 模型参数
- 学习率 (eta): 0.0421
- 最大深度 (max_depth): 8
- 树数量 (n_estimators): 647
- 列采样 (colsample_bytree): 0.8879
- 行采样 (subsample): 0.8789

### 数据集划分
- 训练集: 2020-01-01 ~ 2022-01-01
- 验证集: 2022-01-01 ~ 2022-06-01
- 测试集: 2022-06-01 ~ 2023-06-01

### 股票池和策略
- 股票池: CSI300
- 策略: TopkDropoutStrategy
- topk: 50（每天选50只股票）
- n_drop: 5（每天换仓5只）

## 未来数据泄露风险

**无泄露风险** ✓

- 因子计算只使用当日及之前的历史数据
- IC计算使用未来收益率只是评估因子预测能力的标准做法
- 不参与因子本身的计算

## 预期结果

### 乐观估计
- IC均值: 0.03 ~ 0.05（Alpha158约0.02）
- ICIR: 0.25 ~ 0.40（Alpha158约0.20）
- RankICIR: 0.25 ~ 0.35（Alpha158约0.20）

### 风险提示
1. 无行业中性化 - 行业暴露可能增加风险
2. 因子共线性 - Top50中可能存在高相关因子
3. 过拟合风险 - 需要交叉验证
4. 样本外表现 - 训练期以外的表现不确定

## 依赖

- Python 3.8+
- Qlib
- XGBoost
- Pandas, NumPy, Scikit-learn

## 注意事项

1. 数据范围: 使用2020-2023年CSI300数据
2. 没有使用行业数据: qlib默认数据不包含行业分类
3. 因子公式: 来自 `C:\Users\syk\Desktop\git_repo\auto_alpha\` 目录
4. Alpha101中有39个公式解析失败（主要是复杂语法），Alpha191中有2个失败
