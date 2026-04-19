# Qlib 示例学习指南

> 本文档整理了 Qlib 项目中所有示例的说明、启动命令和学习路径，帮助您快速上手量化投资研究。

---

## 📚 目录

1. [快速开始](#快速开始)
2. [树模型示例](#树模型示例)
3. [深度学习模型示例](#深度学习模型示例)
4. [图神经网络示例](#图神经网络示例)
5. [高频交易示例](#高频交易示例)
6. [强化学习示例](#强化学习示例)
7. [数据处理示例](#数据处理示例)
8. [组合优化示例](#组合优化示例)
9. [学习路径建议](#学习路径建议)

---

## 快速开始

### 环境准备

```bash
# 安装 Qlib
pip install pyqlib

# 初始化数据（首次使用）
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 最简示例

```bash
cd examples
python workflow_by_code.py  # Python 代码方式
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml  # 配置文件方式
```

---

## 树模型示例

### 1. LightGBM（推荐入门）

**说明**：高效的梯度提升决策树，适合结构化数据

**启动命令**：
```bash
cd examples
# CSI300 Alpha158
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# CSI500 Alpha158
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

# Alpha360 特征
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml

# 多频率数据
qrun benchmarks/LightGBM/workflow_config_lightgbm_multi_freq.yaml
```

**学习要点**：
- 理解 `Alpha158` 和 `Alpha360` 特征差异
- 学习 YAML 配置文件结构
- 观察训练/验证/测试集划分方式

---

### 2. XGBoost

**说明**：另一个流行的梯度提升框架

**启动命令**：
```bash
cd examples
qrun benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml
qrun benchmarks/XGBoost/workflow_config_xgboost_Alpha360.yaml
```

**学习要点**：
- 对比 LightGBM 和 XGBoost 的性能差异
- 了解不同超参数配置

---

### 3. CatBoost

**说明**：对类别特征友好的梯度提升库

**启动命令**：
```bash
cd examples
qrun benchmarks/CatBoost/workflow_config_catboost_Alpha158.yaml
qrun benchmarks/CatBoost/workflow_config_catboost_Alpha360.yaml
qrun benchmarks/CatBoost/workflow_config_catboost_Alpha158_csi500.yaml
```

---

### 4. TabNet

**说明**：深度表格学习模型，结合了树模型和深度学习的优势

**启动命令**：
```bash
cd examples
qrun benchmarks/TabNet/workflow_config_TabNet_Alpha158.yaml
qrun benchmarks/TabNet/workflow_config_TabNet_Alpha360.yaml
```

---

## 深度学习模型示例

### 1. LSTM（推荐深度学习入门）

**说明**：长短期记忆网络，经典时序模型

**启动命令**：
```bash
cd examples
qrun benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml
qrun benchmarks/LSTM/workflow_config_lstm_Alpha360.yaml
```

**配置差异**（与树模型对比）：
```yaml
# 树模型使用 DatasetH
dataset:
    class: DatasetH

# LSTM 使用 TSDatasetH（时序数据集）
dataset:
    class: TSDatasetH
    kwargs:
        step_len: 20  # 时间窗口长度

# LSTM 模型参数
model:
    class: LSTM
    kwargs:
        d_feat: 20          # 输入特征维度
        hidden_size: 64     # 隐藏层大小
        num_layers: 2       # LSTM 层数
        dropout: 0.0        # Dropout 率
        n_epochs: 200       # 训练轮数
        lr: 1e-3           # 学习率
        early_stop: 10      # 早停轮数
        batch_size: 800     # 批大小
        GPU: 0              # GPU 编号
```

**学习要点**：
- `TSDatasetH` 会将特征组织成时间序列窗口
- `step_len` 决定了 LSTM 的输入时间跨度
- 深度模型通常需要特征筛选（`FilterCol`）

---

### 2. ALSTM（Attention LSTM）

**说明**：带注意力机制的 LSTM

**启动命令**：
```bash
cd examples
qrun benchmarks/ALSTM/workflow_config_alstm_Alpha158.yaml
qrun benchmarks/ALSTM/workflow_config_alstm_Alpha360.yaml
```

---

### 3. GRU

**说明**：门控循环单元，计算效率高于 LSTM

**启动命令**：
```bash
cd examples
qrun benchmarks/GRU/workflow_config_gru_Alpha158.yaml
qrun benchmarks/GRU/workflow_config_gru_Alpha360.yaml
```

---

### 4. Transformer

**说明**：基于自注意力机制的模型

**启动命令**：
```bash
cd examples
qrun benchmarks/Transformer/workflow_config_transformer_Alpha158.yaml
qrun benchmarks/Transformer/workflow_config_transformer_Alpha360.yaml
```

**学习要点**：
- 自注意力机制对时序模式建模
- 相对 RNN 更好的并行化能力

---

### 5. TFT（Temporal Fusion Transformer）

**说明**：可解释的多时序预测模型

**启动命令**：
```bash
cd examples
# 需要先查看 TFT 特定要求
cat benchmarks/TFT/README.md
python benchmarks/TFT/workflow_by_code_tft.py
```

**注意事项**：
- 需要 Python 3.6-3.7
- 必须使用 GPU
- 需要特定的 CUDA 版本

---

### 6. TCN（时间卷积网络）

**说明**：使用卷积进行时序建模

**启动命令**：
```bash
cd examples
qrun benchmarks/TCN/workflow_config_tcn_Alpha158.yaml
qrun benchmarks/TCN/workflow_config_tcn_Alpha360.yaml
```

---

### 7. MLP

**说明**：多层感知机基线

**启动命令**：
```bash
cd examples
qrun benchmarks/MLP/workflow_config_mlp_Alpha158.yaml
qrun benchmarks/MLP/workflow_config_mlp_Alpha360.yaml
qrun benchmarks/MLP/workflow_config_mlp_Alpha158_csi500.yaml
```

---

### 8. Linear

**说明**：线性模型基线

**启动命令**：
```bash
cd examples
qrun benchmarks/Linear/workflow_config_linear_Alpha158.yaml
```

---

### 9. TRA（Temporal Routing Adapter）

**说明**：时间路由适配器模型

**启动命令**：
```bash
cd examples
qrun benchmarks/TRA/workflow_config_tra_Alpha158.yaml
qrun benchmarks/TRA/workflow_config_tra_Alpha360.yaml
```

---

### 10. TCTS

**说明**：时间协变量转移模型

**启动命令**：
```bash
cd examples
qrun benchmarks/TCTS/workflow_config_tcts_Alpha360.yaml
```

---

### 11. ADARNN

**说明**：自适应深度循环网络

**启动命令**：
```bash
cd examples
qrun benchmarks/ADARNN/workflow_config_adarnn_Alpha360.yaml
```

---

## 图神经网络示例

### 1. GATs（Graph Attention Networks）

**说明**：图注意力网络，建模股票间关系

**启动命令**：
```bash
cd examples
qrun benchmarks/GATs/workflow_config_gats_Alpha158.yaml
qrun benchmarks/GATs/workflow_config_gats_Alpha360.yaml
```

**学习要点**：
- 如何将股票关系建模为图结构
- 图神经网络在金融数据上的应用

---

### 2. HIST

**说明**：基于行业/概念的超图模型

**启动命令**：
```bash
cd examples
qrun benchmarks/HIST/workflow_config_hist_Alpha360.yaml
```

---

## 高频交易示例

### highfreq

**说明**：高频交易数据和策略示例

**启动命令**：
```bash
cd examples/highfreq

# 获取高频数据
python workflow.py get_data

# 运行工作流
python workflow.py

# 数据集序列化示例
python workflow.py dump_and_load_dataset
```

**学习要点**：
- 高频数据处理方法
- 自定义算子（`highfreq_ops.py`）
- 自定义 Handler（`highfreq_handler.py`）

---

## 强化学习示例

### 1. rl_order_execution

**说明**：订单执行强化学习

**启动命令**：
```bash
cd examples/rl_order_execution

# 1. 数据处理
python -m qlib.cli.data qlib_data --target_dir ./data/bin --region hs300 --interval 5min

# 2. 生成 pickle 数据
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py

# 3. 训练（以 OPDS 为例）
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest

# 4. 回测
python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml
```

**学习要点**：
- PPO（Proximal Policy Optimization）
- OPDS（Oracle Policy Distillation）
- TWAP 基线策略

---

### 2. rl/simple_example.ipynb

**说明**：强化学习基础示例

**启动命令**：
```bash
jupyter notebook examples/rl/simple_example.ipynb
```

---

### 3. nested_decision_execution

**说明**：嵌套决策执行

**启动命令**：
```bash
cd examples/nested_decision_execution
python workflow.py
```

---

## 数据处理示例

### 1. data_demo

**说明**：数据缓存和内存复用

**启动命令**：
```bash
cd examples/data_demo
python data_cache_demo.py
python data_mem_resuse_demo.py
```

---

### 2. orderbook_data

**说明**：订单簿数据处理

**启动命令**：
```bash
cd examples/orderbook_data
python create_dataset.py
python example.py
```

---

### 3. rolling_process_data

**说明**：滚动数据处理

**启动命令**：
```bash
cd examples/rolling_process_data
python workflow.py
```

---

## 组合优化示例

### portfolio

**说明**：投资组合优化

**启动命令**：
```bash
cd examples/portfolio

# 准备风险数据
python prepare_riskdata.py

# 运行增强指数策略
qrun config_enhanced_indexing.yaml
```

---

## 其他示例

### 1. tutorial/detailed_workflow.ipynb

**说明**：详细的工作流教程，逐步构建各个组件

**启动命令**：
```bash
jupyter notebook examples/tutorial/detailed_workflow.ipynb
```

---

### 2. workflow_by_code.py

**说明**：用 Python 代码构建工作流（非 YAML 配置）

**启动命令**：
```bash
cd examples
python workflow_by_code.py
```

---

### 3. model_rolling

**说明**：模型滚动训练

**启动命令**：
```bash
cd examples/model_rolling
python task_manager_rolling.py
```

---

### 4. online_srv

**说明**：在线预测服务

**启动命令**：
```bash
cd examples/online_srv
python update_online_pred.py
python rolling_online_management.py
python online_management_simulate.py
```

---

### 5. model_interpreter

**说明**：模型解释和特征重要性分析

**启动命令**：
```bash
cd examples/model_interpreter
python feature.py
```

---

### 6. DoubleEnsemble

**说明**：双集成方法

**启动命令**：
```bash
cd examples
qrun benchmarks/DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml
qrun benchmarks/DoubleEnsemble/workflow_config_doubleensemble_Alpha360.yaml
```

---

### 7. hyperparameter/LightGBM

**说明**：超参数调优示例

**启动命令**：
```bash
cd examples/hyperparameter/LightGBM
python hyperparameter_158.py
python hyperparameter_360.py
```

---

## 学习路径建议

### 初学者路径（推荐）

1. **第一周：基础概念**
   - 运行 `workflow_by_code.py`，理解基本流程
   - 阅读 `tutorial/detailed_workflow.ipynb`
   - 运行 LightGBM Alpha158 示例

2. **第二周：树模型对比**
   - 运行 XGBoost 和 CatBoost 示例
   - 对比不同模型在相同特征上的表现
   - 理解 Alpha158 vs Alpha360 特征差异

3. **第三周：深度学习模型**
   - 从 LSTM 开始
   - 对比 ALSTM、GRU 的差异
   - 理解 TSDatasetH 和 DatasetH 的区别

4. **第四周：进阶主题**
   - 图神经网络（GATs）
   - 高频交易示例
   - 强化学习示例

---

### 进阶路径

1. **特征工程**
   - 修改 `Alpha158` 配置
   - 创建自定义特征
   - 尝试不同的 rolling 算子组合

2. **模型优化**
   - 超参数调优示例
   - DoubleEnsemble 方法
   - 模型解释

3. **策略研究**
   - 组合优化
   - 滚动训练
   - 在线服务

---

## 关键概念总结

### 数据集类型

- `DatasetH`：常规数据集，适用于树模型
- `TSDatasetH`：时序数据集，适用于深度学习模型（需指定 `step_len`）

### 特征集合

- **Alpha158**：158 个技术特征（K线形态、价格比率、滚动统计等）
- **Alpha360**：360 个原始价格序列特征（过去 60 天的 OHLCV）

### 处理器

- `FilterCol`：特征筛选
- `RobustZScoreNorm`：稳健标准化
- `Fillna`：填充缺失值
- `DropnaLabel`：删除缺失标签
- `CSRankNorm`：截面排名标准化

---

## 常见问题

### Q: 如何选择模型？

A:
- 数据量小、特征工程成熟：树模型（LightGBM/XGBoost）
- 数据量大、时序关系强：深度学习模型（LSTM/Transformer）
- 考虑股票关系：图神经网络（GATs/HIST）

### Q: 如何自定义特征？

A: 参考 `qlib/contrib/data/loader.py`，修改 `Alpha158` 的配置或创建新的 Handler 类。

### Q: 如何评估模型？

A: 主要指标：
- IC / Rank IC：预测能力
- ICIR：预测稳定性
- 年化收益 / IR：策略表现
- 最大回撤：风险控制

---

## 参考资源

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [论文列表](https://github.com/microsoft/qlib#references)

---

> 最后更新：2024 年
> 整理者：CatPaw AI Assistant

