# Qlib 三模型对比实验 - 最终状态报告

## 🎯 实验概览

**实验目的**: 对比XGBoost、LSTM和TabNet三个量化投资模型在A股市场的表现

**更新时间**: 2026-04-14 22:00

---

## ✅ 问题解决记录

### 遇到的问题

1. **setuptools_scm缺失**
   - 错误: `ModuleNotFoundError: No module named 'setuptools_scm'`
   - 解决: `pip install setuptools_scm`

2. **模块路径错误**
   - 错误: `ModuleNotFoundError: No module named 'qlib.run'`
   - 解决: 使用正确的路径 `qlib.cli.data` 和 `qlib.cli.run`

3. **C扩展未编译**
   - 错误: `ModuleNotFoundError: No module named 'qlib.data._libs.rolling'`
   - 解决: `pip install -e . --no-build-isolation`

4. **PyTorch DLL加载失败**
   - 错误: `OSError: [WinError 1114] DLL初始化例程失败`
   - 解决: 重新安装PyTorch CPU版本
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

---

## 📊 当前状态

### 模型运行状态

| 模型 | 状态 | 配置文件 | 预计时间 |
|------|------|----------|----------|
| **XGBoost** | ✅ 运行中 | Alpha158 | 5-10分钟 |
| **LSTM** | ✅ 运行中 | Alpha158 | 1-3小时 |
| **TabNet** | ✅ 运行中 | Alpha158 | 30分钟-1小时 |

**所有模型已启动！** 🎉

---

## 🔧 环境配置

### 已安装的依赖

```
✓ qlib 0.1.dev2063
✓ setuptools_scm 10.0.5
✓ cython 3.2.4
✓ xgboost 3.2.0
✓ torch 2.11.0+cpu
✓ 数据: ~/.qlib/qlib_data/cn_data (547MB)
```

### 数据配置

- **数据源**: 中国A股市场
- **市场**: CSI300 (沪深300)
- **时间范围**:
  - 训练: 2008-01-01 到 2014-12-31 (7年)
  - 验证: 2015-01-01 到 2016-12-31 (2年)
  - 测试: 2017-01-01 到 2020-08-01 (3.5年)
- **特征**: Alpha158 (158个量化因子)

---

## 📁 文件结构

### 已创建的脚本

```
examples/
├── run_single_model.py           # 运行单个模型
├── check_progress.py             # 实时监控进度 ⭐推荐
├── check_model_status.py         # 检查模型状态
├── run_model_comparison.py       # 批量对比运行
├── run_model_comparison_simple.py # 简化版对比
├── show_results.py               # 显示结果
├── MODEL_COMPARISON_GUIDE.md     # 详细指南
└── TRAINING_STATUS.md            # 本文件
```

### 实验结果目录

```
mlruns/
└── 746607912206639406/          # 实验ID
    ├── c6fbfd74.../             # XGBoost运行
    ├── a51b8f25.../             # LSTM运行
    └── af585cbd.../             # TabNet运行
```

---

## 🚀 使用指南

### 1. 查看实时进度（推荐）

```bash
cd c:/Users/syk/Desktop/git_repo/qlib/examples
python check_progress.py
```

这会每30秒自动刷新，显示：
- 每个模型的状态
- 已完成数量
- 运行中数量

### 2. 查看详细状态

```bash
python check_model_status.py
```

### 3. 使用MLflow可视化（训练完成后）

```bash
# 安装mlflow
pip install mlflow

# 启动UI
mlflow ui

# 浏览器访问
# http://localhost:5000
```

### 4. 手动运行单个模型

```bash
# XGBoost（最快）
python run_single_model.py xgboost

# LSTM（最慢）
python run_single_model.py lstm

# TabNet
python run_single_model.py tabnet
```

---

## 📈 预期结果

### 关键指标

训练完成后，查看以下指标评估模型性能：

| 指标 | 含义 | 好的标准 |
|------|------|----------|
| **IC** | Information Coefficient | > 0.05 |
| **Rank IC** | Rank IC | > 0.05 |
| **ICIR** | IC Information Ratio | > 0.5 |
| **Rank ICIR** | Rank IC IR | > 0.5 |
| **Annual Return** | 年化收益率 | > 10% |
| **Information Ratio** | 信息比率 | > 1.0 |
| **Max Drawdown** | 最大回撤 | < 20% |

### 模型特点对比

| 特性 | XGBoost | LSTM | TabNet |
|------|---------|------|--------|
| **类型** | 树模型 | 深度学习 | 深度学习 |
| **速度** | ⚡ 快 | 🐢 慢 | 🚗 中等 |
| **可解释性** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **适合场景** | 通用 | 序列 | 表格 |
| **调参难度** | 中 | 高 | 高 |

---

## ⚠️ 注意事项

### 1. 训练时间

基于CPU训练，预计总时间：
- XGBoost: ~10分钟
- LSTM: ~2小时（200 epochs）
- TabNet: ~45分钟

**总计**: 约3小时

### 2. 系统资源

- **内存**: 建议8GB以上
- **CPU**: 多核处理器
- **磁盘**: 约2GB（包括数据）

### 3. 常见问题

**Q: 如何停止正在运行的模型？**
```bash
# 找到Python进程
tasklist | findstr python

# 结束进程（管理员权限）
taskkill /F /PID <进程ID>
```

**Q: 如何修改训练参数？**
```bash
# 编辑对应的YAML配置文件
# 例如: benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml
```

**Q: 如何使用GPU加速？**
```bash
# 1. 安装CUDA版本的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. 修改配置文件
# 将 GPU: 0 改为 GPU: 0（使用第一个GPU）
```

---

## 🎓 学习资源

### Qlib文档
- [官方文档](https://qlib.readthedocs.io/)
- [GitHub仓库](https://github.com/microsoft/qlib)
- [论文](https://arxiv.org/abs/2009.11189)

### 模型论文
- [XGBoost](https://arxiv.org/abs/1603.02754)
- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [TabNet](https://arxiv.org/abs/1908.07442)

---

## 📝 下一步

### 训练完成后

1. **查看结果**
   ```bash
   python check_progress.py
   mlflow ui
   ```

2. **对比分析**
   - 比较IC、ICIR等指标
   - 分析回测曲线
   - 评估风险收益

3. **模型优化**
   - 调整超参数
   - 尝试不同的特征集（Alpha360）
   - 组合多个模型

4. **实战应用**
   - 选择最佳模型
   - 部署到生产环境
   - 持续监控性能

### 进阶实验

- 尝试其他市场（CSI500、CSI100）
- 使用分钟级数据
- 添加自定义因子
- 实现组合策略

---

## 📞 支持

如有问题，请查看：
1. 本文档
2. `MODEL_COMPARISON_GUIDE.md`
3. [Qlib官方文档](https://qlib.readthedocs.io/)
4. [GitHub Issues](https://github.com/microsoft/qlib/issues)

---

**祝实验顺利！** 🎉

**最后更新**: 2026-04-14 22:00
