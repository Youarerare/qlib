# run_lstm_alpha360_minimal.py
# 最小化版本：只加载 Alpha360 数据、训练 LSTM 并输出 rank IC/ICIR
# 适配 macOS：spawn 启动、单进程数据加载
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from qlib import init
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha360

def main():
    # 初始化 Qlib
    init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    # 构建 dataset（直接用 Alpha360 类）
    handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "csi300",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }
    handler = Alpha360(**handler_config)
    dataset = DatasetH(handler=handler)

    # 准备数据
    print("Loading data...")
    df_train, df_valid, df_test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key="DK_L"
    )
    print("Data loaded. train:", df_train["feature"].shape, "valid:", df_valid["feature"].shape, "test:", df_test["feature"].shape)

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
    train_loader = DataLoader(train_ds, batch_size=800, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=800, shuffle=False, num_workers=0)

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
    best_state = None

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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 预测测试集
    test_tensor = to_tensor(X_test).to(device)
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy()
    pred_series = pd.Series(pred, index=X_test.index)

    # 计算 rank IC / ICIR
    def rank_ic(pred, label):
        df = pd.DataFrame({"pred": pred, "label": label})
        ic_per_day = df.groupby(level="datetime").apply(lambda x: x["pred"].rank().corr(x["label"].rank()))
        return ic_per_day.mean(), ic_per_day.mean() / ic_per_day.std() if ic_per_day.std() != 0 else np.nan

    mean_ic, icir = rank_ic(pred_series, y_test.squeeze())
    print(f"mean rank IC: {mean_ic:.6f}, ICIR: {icir:.6f}")

    # 保存预测结果
    out_path = Path(__file__).parent / "mlruns/lstm_alpha360_pred_minimal.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pred_series, f)
    print("saved predictions to", out_path)

if __name__ == "__main__":
    main()

