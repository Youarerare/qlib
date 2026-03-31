# LSTM + Alpha360 运行指南（Linux 容器）

由于 macOS 上存在多进程兼容性问题（段错误），建议在 Linux 容器中运行以下步骤：

## 1. 构建镜像
```bash
cd /Users/syk/Desktop/git_repo/qlib
docker build -t qlib-lstm -f examples/Dockerfile.lstm_alpha360 .
```

## 2. 准备数据
- 你需要提前下载 Qlib 的 CN 数据，并挂载到容器内：
```bash
# 假设数据已下载到本地 ~/.qlib/qlib_data/cn_data
docker run --rm -v ~/.qlib/qlib_data/cn_data:/root/.qlib/qlib_data/cn_data qlib-lstm
```

## 3. 查看结果
- 容器运行结束后，可在 `mlruns` 目录下查看 IC、ICIR、回测结果。
- 你也可以进入容器交互：
```bash
docker run -it --rm -v ~/.qlib/qlib_data/cn_data:/root/.qlib/qlib_data/cn_data qlib-lstm bash
# 在容器内：
qrun examples/benchmarks/LSTM/workflow_config_lstm_Alpha360.yaml
```

## 4. 结果解读
- 运行完成后，日志会输出 test IC、Rank IC、ICIR，以及回测的年化收益、IR、最大回撤。
- 可与 `LightGBM + Alpha360` 的结果对比，判断 LSTM 在 Alpha360 上是否更合理。

---
如果你希望继续在 macOS 上尝试，可以：
- 确保 numpy==1.26.4、torch==2.2.2 兼容。
- 在 `qrun` 或自定义脚本中强制使用 `spawn` 启动、`multiprocessing` 后端、并行数=1。
- 但仍可能遇到段错误或卡死，建议优先使用容器方案。

