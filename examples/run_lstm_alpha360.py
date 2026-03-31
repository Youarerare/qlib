# run_lstm_alpha360.py
# 在本地 macOS 运行，尽量规避段错误：多进程启动方式改 spawn，并行数降到 1
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from pathlib import Path
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from qlib import init
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha360
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord, SigAnaRecord
from qlib.contrib.evaluate import risk_analysis
# TopkDropoutStrategy 会导入 cvxpy，而当前 numpy 1.26.4 与 cvxpy 不兼容，因此延迟导入
# from qlib.contrib.strategy import TopkDropoutStrategy

# 本地路径
YAML_PATH = Path(__file__).parent / "benchmarks/LSTM/workflow_config_lstm_Alpha360.yaml"
PRED_OUT = Path(__file__).parent / "mlruns/lstm_alpha360_pred.pkl"
PORT_OUT = Path(__file__).parent / "mlruns/lstm_alpha360_port.pkl"

def main():
    # 初始化 Qlib
    init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    # 读取配置
    cfg = yaml.safe_load(YAML_PATH.read_text())
    model_cfg = cfg["task"]["model"]
    dataset_cfg = cfg["task"]["dataset"]
    port_cfg = cfg["port_analysis_config"]

    # 构建 dataset
    dataset = DatasetH(**dataset_cfg["kwargs"])

    # 准备数据
    df_train, df_valid, df_test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key="DK_L"
    )
    X_train, y_train = df_train["feature"], df_train["label"]
    X_valid, y_valid = df_valid["feature"], df_valid["label"]
    X_test, y_test = df_test["feature"], df_test["label"]

    # 转为 np 并 reshape 为 (N, T, D)，T=60, D=6
    def to_tensor(df):
        arr = df.values
        N, FD = arr.shape
        T, D = 60, 6
        assert FD == T * D, f"特征维度应为 {T*D}，实际为 {FD}"
        return torch.from_numpy(arr.reshape(N, T, D)).float()

    train_ds = torch.utils.data.TensorDataset(to_tensor(X_train), torch.from_numpy(y_train.values).float())
    valid_ds = torch.utils.data.TensorDataset(to_tensor(X_valid), torch.from_numpy(y_valid.values).float())
    train_loader = DataLoader(train_ds, batch_size=800, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=800, shuffle=False)

    # 模型
    class LSTMModel(nn.Module):
        def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)  # (N, T, H)
            out = out[:, -1, :]    # 取最后一个时间步
            out = self.fc(out)
            return out.squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    patience = 20
    wait = 0

    for epoch in range(200):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb.squeeze()).item())
        val_loss = np.mean(val_losses)
        print(f"epoch {epoch} val_loss {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    model.eval()

    # 预测测试集
    test_tensor = to_tensor(X_test).to(device)
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy()
    pred_series = pd.Series(pred, index=X_test.index)

    # 保存预测结果
    with open(PRED_OUT, "wb") as f:
        pickle.dump(pred_series, f)
    print("saved predictions to", PRED_OUT)

    # 计算 IC（截面相关）
    # 注意：Alpha360 的标签定义为 Ref($close, -2)/Ref($close, -1) - 1，预测的是 T+1 收益率
    # pred_series 与 y_test 的索引一致，都是 (instrument, datetime)
    # 按 datetime 分组计算 rank IC
    def rank_ic(pred, label):
        """按日期计算 rank IC，返回每日 IC 的均值与 ICIR"""
        df = pd.DataFrame({"pred": pred, "label": label})
        ic_per_day = df.groupby(level="datetime").apply(lambda x: x["pred"].rank().corr(x["label"].rank()))
        return ic_per_day.mean(), ic_per_day.mean() / ic_per_day.std() if ic_per_day.std() != 0 else np.nan

    mean_ic, icir = rank_ic(pred_series, y_test.squeeze())
    print(f"mean rank IC: {mean_ic:.6f}, ICIR: {icir:.6f}")

    # 简单回测（使用 Qlib 的 backtest_executor）
    # 这里直接调用 TopkDropoutStrategy 的信号回测
    from qlib.contrib.backtest import backtest
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    # 构造信号 DataFrame
    signal = pred_series.to_frame("score")
    # 对齐时间
    signal = signal.reset_index()
    signal.columns = ["instrument", "datetime", "score"]
    signal = signal.set_index(["instrument", "datetime"])

    # 回测参数
    backtest_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": signal,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # 执行回测
    portfolio = backtest(**backtest_config["backtest"], strategy=backtest_config["strategy"])
    # 保存回测结果
    with open(PORT_OUT, "wb") as f:
        pickle.dump(portfolio, f)
    print("saved portfolio to", PORT_OUT)

    # 风险分析
    analysis = risk_analysis(portfolio["return"])
    print(analysis)

if __name__ == "__main__":
    main()

