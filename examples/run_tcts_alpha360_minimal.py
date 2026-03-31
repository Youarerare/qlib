# run_tcts_alpha360_minimal.py
# 单进程、内存数组方式运行 TCTS Alpha360，绕过 macOS joblib 共享内存导致的段错误
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ['JOBLIB_NUM_WORKERS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from qlib import init
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha360

def main():
    init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

    # 直接构造 DatasetH，避免内部多进程
    print('Loading data (single-process)...')
    ds = DatasetH(
        handler={'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': dict(
            start_time='2008-01-01',
            end_time='2020-08-01',
            fit_start_time='2008-01-01',
            fit_end_time='2014-12-31',
            instruments='csi300',
            infer_processors=[
                {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}},
            ],
            learn_processors=[
                {'class': 'DropnaLabel'},
                {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}},
            ],
            label=["Ref($close, -2) / Ref($close, -1) - 1", "Ref($close, -3) / Ref($close, -1) - 1", "Ref($close, -4) / Ref($close, -1) - 1"],
        )},
        segments={
            'train': ('2008-01-01', '2014-12-31'),
            'valid': ('2015-01-01', '2016-12-31'),
            'test': ('2017-01-01', '2020-08-01'),
        },
    )
    # fetch processed data (learn set) instead of using data_key
    df_train = ds.prepare('train', col_set=['feature', 'label'])
    df_valid = ds.prepare('valid', col_set=['feature', 'label'])
    df_test = ds.prepare('test', col_set=['feature', 'label'])
    print('Shapes:', df_train['feature'].shape, df_valid['feature'].shape, df_test['feature'].shape)

    X_train, y_train = df_train['feature'].values, df_train['label'].values
    X_valid, y_valid = df_valid['feature'].values, df_valid['label'].values
    X_test, y_test = df_test['feature'].values, df_test['label'].values

    # reshape to (N, T=60, D=6)
    N, FD = X_train.shape
    T, D = 60, 6
    assert FD == T * D
    X_train = X_train.reshape(N, T, D)
    X_valid = X_valid.reshape(X_valid.shape[0], T, D)
    X_test = X_test.reshape(X_test.shape[0], T, D)

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=800, shuffle=True, num_workers=0)
    valid_dl = DataLoader(TensorDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).float()), batch_size=800, shuffle=False, num_workers=0)

    class TCTSModel(nn.Module):
        def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.3, output_dim=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_dim)
        def forward(self, x):
            out, _ = self.lstm(x)  # (N, T, H)
            out = out[:, -1, :]    # last timestep
            out = self.fc(out)     # (N, output_dim)
            return out

    device = torch.device('cpu')
    model = TCTSModel().to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    patience = 20
    wait = 0
    best_state = None

    for epoch in range(200):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = np.mean(val_losses)
        print(f'epoch {epoch} val_loss {val_loss:.6f}')
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print('early stop')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # test prediction
    with torch.no_grad():
        pred_test = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()

    # only use main task (label 0) for evaluation
    pred_series = pd.Series(pred_test[:, 0], index=df_test['feature'].index)
    y_test_main = pd.Series(y_test[:, 0], index=df_test['feature'].index)

    def rank_ic(pred, label):
        df = pd.DataFrame({'pred': pred, 'label': label})
        ic_per_day = df.groupby(level='datetime').apply(lambda x: x['pred'].rank().corr(x['label'].rank()))
        return ic_per_day.mean(), ic_per_day.mean() / ic_per_day.std() if ic_per_day.std() != 0 else np.nan

    mean_ic, icir = rank_ic(pred_series, y_test_main)
    print(f'TCTS test mean rank IC: {mean_ic:.6f}, ICIR: {icir:.6f}')

    # 保存预测
    out_path = Path(__file__).parent / 'mlruns/tcts_alpha360_pred.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(pred_series, f)
    print('saved to', out_path)

if __name__ == "__main__":
    main()

