# 快速命令参考

## 🎯 当前状态

✅ **三个模型都在运行中！**

- XGBoost: 运行中
- LSTM: 运行中
- TabNet: 运行中

## 📊 查看进度

### 方法1: 自动刷新（推荐）
```bash
cd c:/Users/syk/Desktop/git_repo/qlib/examples
python check_progress.py
```
每30秒自动刷新，显示训练进度。

### 方法2: 查看状态
```bash
python check_model_status.py
```

### 方法3: MLflow UI（训练完成后）
```bash
pip install mlflow
mlflow ui
# 浏览器打开: http://localhost:5000
```

## 🔄 如果需要重新运行

### 单个模型
```bash
# XGBoost（最快，~10分钟）
python run_single_model.py xgboost

# LSTM（最慢，~2小时）
python run_single_model.py lstm

# TabNet（中等，~45分钟）
python run_single_model.py tabnet
```

### 批量运行
```bash
python run_model_comparison_simple.py
```

## ⚠️ 已解决的问题

1. ✅ setuptools_scm缺失 → 已安装
2. ✅ C扩展未编译 → 已编译
3. ✅ PyTorch DLL错误 → 已修复（使用CPU版本）

## 📈 预计完成时间

- XGBoost: ~10分钟
- LSTM: ~2小时
- TabNet: ~45分钟

**总计**: 约3小时（从现在开始）

## 📁 结果位置

```
mlruns/
└── 746607912206639406/
    ├── c6fbfd74.../  # XGBoost
    ├── a51b8f25.../  # LSTM
    └── af585cbd.../  # TabNet
```

## 🎓 详细文档

- `TRAINING_STATUS.md` - 完整状态报告
- `MODEL_COMPARISON_GUIDE.md` - 详细使用指南

---

**现在只需等待训练完成！** ☕
