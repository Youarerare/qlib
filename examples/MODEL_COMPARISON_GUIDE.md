# Qlib 三模型对比实验指南

## 实验概述

本实验对比三个经典量化投资模型：
1. **XGBoost** - 梯度提升树模型
2. **LSTM** - 长短期记忆网络
3. **TabNet** - 注意力机制的表格神经网络

## 模型特点对比

| 特性 | XGBoost | LSTM | TabNet |
|------|---------|------|--------|
| **类型** | 树模型 | 深度学习 | 深度学习 |
| **训练速度** | 快 | 慢 | 中等 |
| **需要GPU** | 否 | 建议 | 建议 |
| **可解释性** | 中 | 低 | 高 |
| **适合场景** | 通用 | 序列数据 | 表格数据 |
| **参数数量** | 中 | 高 | 高 |

## 运行状态

### 已运行的模型

根据实验记录，以下模型正在或已完成训练：

1. **XGBoost** ✓
   - 状态: 已完成
   - 模型类型: XGBModel
   - 配置: Alpha158因子

2. **LSTM** (运行中)
   - 状态: 训练中
   - 模型类型: LSTM
   - 配置: Alpha158因子
   - 注意: 需要200个epoch，可能需要较长时间

3. **TabNet** (运行中)
   - 状态: 训练中
   - 模型类型: TabnetModel
   - 配置: Alpha158因子

## 如何运行

### 方法1: 逐个运行

```bash
# 运行XGBoost（最快，推荐先运行）
python run_single_model.py xgboost

# 运行LSTM（需要GPU加速）
python run_single_model.py lstm

# 运行TabNet
python run_single_model.py tabnet
```

### 方法2: 批量运行

```bash
# 依次运行所有模型
python run_model_comparison_simple.py
```

### 方法3: 使用qrun命令

```bash
# XGBoost
python -m qlib.cli.run benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml

# LSTM
python -m qlib.cli.run benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml

# TabNet
python -m qlib.cli.run benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml
```

## 查看结果

### 方法1: 使用Python脚本

```bash
python check_model_status.py
```

### 方法2: 使用MLflow UI

```bash
# 安装mlflow
pip install mlflow

# 启动UI
mlflow ui

# 在浏览器打开: http://localhost:5000
```

### 方法3: 手动查看

```bash
# 进入实验目录
cd mlruns

# 查看实验列表
ls

# 进入具体实验
cd <experiment_id>

# 查看运行记录
cd <run_id>

# 查看参数
cat params/model.class

# 查看指标
cat metrics/*
```

## 预期结果

### 关键指标

训练完成后，可以查看以下指标：

| 指标 | 含义 | 期望值 |
|------|------|--------|
| **IC** | 信息系数 | 越高越好 |
| **Rank IC** | 排名信息系数 | 越高越好 |
| **ICIR** | IC信息比率 | > 0.5 较好 |
| **Annualized Return** | 年化收益率 | 越高越好 |
| **Information Ratio** | 信息比率 | > 1.0 较好 |

### 模型结果目录结构

```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── params/          # 模型参数
        │   ├── model.class
        │   ├── model.module_path
        │   └── ...
        ├── metrics/         # 评估指标
        │   ├── ic
        │   ├── rank_ic
        │   └── ...
        ├── artifacts/       # 生成的文件
        │   ├── task         # 任务配置
        │   └── ...
        └── tags/            # 标签信息
```

## 配置说明

### 数据配置

所有模型使用相同的配置：

- **数据源**: ~/.qlib/qlib_data/cn_data
- **市场**: CSI300（沪深300）
- **训练期**: 2008-01-01 到 2014-12-31
- **验证期**: 2015-01-01 到 2016-12-31
- **测试期**: 2017-01-01 到 2020-08-01
- **特征**: Alpha158（158个量化因子）

### 模型特定配置

#### XGBoost配置
- n_estimators: 647
- max_depth: 8
- learning_rate (eta): 0.0421
- colsample_bytree: 0.8879
- subsample: 0.8789

#### LSTM配置
- hidden_size: 64
- num_layers: 2
- dropout: 0.0
- n_epochs: 200
- batch_size: 800
- learning_rate: 1e-3
- early_stop: 10

#### TabNet配置
- d_feat: 158
- pretrain: True
- seed: 993

## 注意事项

### 1. GPU支持

深度学习模型（LSTM和TabNet）建议使用GPU加速：

```python
# 检查GPU是否可用
import torch
print(torch.cuda.is_available())
```

如果没有GPU，可以在配置文件中修改：
```yaml
model:
    kwargs:
        GPU: -1  # 使用CPU
```

### 2. 训练时间

预估训练时间（基于CPU）：
- XGBoost: 5-10分钟
- LSTM: 1-3小时
- TabNet: 30分钟-1小时

### 3. 内存需求

- XGBoost: 约2GB
- LSTM: 约4GB
- TabNet: 约3GB

### 4. 常见问题

**Q: 训练时出现CUDA错误怎么办？**
A: 修改配置文件中的GPU参数为-1，使用CPU训练。

**Q: 数据不存在怎么办？**
A: 运行数据下载脚本：
```bash
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

**Q: 如何调整超参数？**
A: 修改对应的yaml配置文件中的model.kwargs部分。

## 结果分析

### 性能对比维度

1. **预测性能**
   - IC/Rank IC：衡量因子预测能力
   - ICIR：衡量因子稳定性

2. **回测性能**
   - 年化收益率：投资收益
   - 信息比率：风险调整后收益
   - 最大回撤：风险控制能力

3. **计算效率**
   - 训练时间
   - 预测速度

### 可视化分析

使用qlib内置的分析工具：

```python
import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord

# 加载实验
recorder = R.get_recorder(rec_id=<run_id>)

# 查看分析结果
recorder.list_metrics()
```

## 下一步

1. 等待所有模型训练完成
2. 对比各模型的IC、ICIR等指标
3. 分析回测结果
4. 尝试调整超参数优化性能
5. 尝试不同的特征集（如Alpha360）

## 参考资源

- [Qlib官方文档](https://qlib.readthedocs.io/)
- [XGBoost文档](https://xgboost.readthedocs.io/)
- [PyTorch LSTM文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [TabNet论文](https://arxiv.org/abs/1908.07442)

---

**最后更新**: 2026-04-14
**作者**: AI Assistant
