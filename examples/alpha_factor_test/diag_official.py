"""用官方qrun方式跑Alpha158 XGBoost，获取基准IC"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import qlib
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    import numpy as np

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    handler = Alpha158(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2023-06-01",
        fit_start_time="2020-01-01",
        fit_end_time="2021-12-31",
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2020-01-01", "2021-12-31"),
            "valid": ("2022-01-01", "2022-05-31"),
            "test": ("2022-06-01", "2023-06-01"),
        },
    )

    model = XGBModel(
        eval_metric="rmse",
        colsample_bytree=0.8879,
        eta=0.0421,
        max_depth=8,
        n_estimators=647,
        subsample=0.8789,
        nthread=20,
    )

    model.fit(dataset)

    # 预测: 用DK_I的feature
    pred = model.predict(dataset)
    print(f"预测值: {len(pred)}条, 均值={pred.mean():.6f}, 标准差={pred.std():.6f}")

    # 获取原始label: 用DK_R
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(label, pd.DataFrame):
        label = label.squeeze()
    print(f"Label: {len(label)}条, 均值={float(label.mean()):.6f}, 标准差={float(label.std()):.6f}")

    # 对齐
    common_idx = pred.index.intersection(label.dropna().index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]
    print(f"对齐后: {len(common_idx)}条")

    # 逐日计算IC
    result_df = pd.DataFrame({"pred": pred_aligned, "label": label_aligned}, index=common_idx)
    ic_list, rank_ic_list = [], []
    for date in result_df.index.get_level_values(0).unique():
        day = result_df.loc[date]
        if len(day) < 10:
            continue
        try:
            ic, _ = pearsonr(day["pred"], day["label"])
            rank_ic, _ = spearmanr(day["pred"], day["label"])
            ic_list.append(ic)
            rank_ic_list.append(rank_ic)
        except:
            continue

    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list)

    print(f"\n官方qrun方式 Alpha158 XGBoost 结果:")
    print(f"  IC均值: {ic_arr.mean():.4f}")
    print(f"  ICIR: {ic_arr.mean()/ic_arr.std():.4f}")
    print(f"  RankIC均值: {rank_ic_arr.mean():.4f}")
    print(f"  RankICIR: {rank_ic_arr.mean()/rank_ic_arr.std():.4f}")
    print(f"  测试期数: {len(ic_arr)}")
