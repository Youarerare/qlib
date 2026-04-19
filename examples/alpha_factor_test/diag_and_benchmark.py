"""1. 诊断Alpha158 label NaN问题 2. 按官方设置复现benchmark"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP
    from scipy.stats import pearsonr, spearmanr

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    # ============ 1. 诊断label NaN ============
    print("=" * 80)
    print("1. 诊断Alpha158 label NaN问题")
    print("=" * 80)

    handler = Alpha158(instruments="csi300", start_time="2020-01-01", end_time="2023-06-01")

    a158_label_r = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
    print(f"DK_R label类型: {type(a158_label_r)}")
    print(f"DK_R label形状: {a158_label_r.shape}")
    print(f"DK_R label列名: {a158_label_r.columns.tolist() if isinstance(a158_label_r, pd.DataFrame) else 'Series'}")
    print(f"DK_R label索引前5: {a158_label_r.index[:5].tolist()}")
    print(f"DK_R label索引层级: {a158_label_r.index.names}")
    print(f"DK_R label前5行:\n{a158_label_r.head()}")

    # 用loc取SH600000
    sample_stock = "SH600000"
    try:
        stock_label = a158_label_r.loc[(slice(None), sample_stock), :]
        print(f"\n用loc取SH600000: {len(stock_label)}行")
        print(stock_label.dropna().head(5))
    except Exception as e:
        print(f"loc取SH600000失败: {e}")

    # 用不同方式取
    try:
        stock_label2 = a158_label_r.xs(sample_stock, level=1)
        print(f"\n用xs取SH600000: {len(stock_label2)}行")
        print(stock_label2.dropna().head(5))
    except Exception as e:
        print(f"xs取SH600000失败: {e}")

    # ============ 2. 按官方benchmark设置复现 ============
    print("\n" + "=" * 80)
    print("2. 按官方benchmark设置复现 Alpha158 XGBoost")
    print("=" * 80)

    # 官方配置: 2008-2020, train=2008-2014, valid=2015-2016, test=2017-2020
    handler_bench = Alpha158(
        instruments="csi300",
        start_time="2008-01-01",
        end_time="2020-08-01",
        fit_start_time="2008-01-01",
        fit_end_time="2014-12-31",
    )

    dataset = DatasetH(
        handler=handler_bench,
        segments={
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
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

    print("开始训练...")
    model.fit(dataset)
    print("训练完成!")

    # 预测
    pred = model.predict(dataset)
    print(f"预测值: {len(pred)}条, 均值={pred.mean():.6f}, 标准差={pred.std():.6f}")

    # 获取原始label
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(label, pd.DataFrame):
        label = label.squeeze()
    print(f"Label: {len(label)}条, 均值={float(label.mean()):.6f}, 标准差={float(label.std()):.6f}")

    # 对齐
    common_idx = pred.index.intersection(label.dropna().index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]
    print(f"对齐后: {len(common_idx)}条")

    # 逐日IC
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

    print(f"\n官方benchmark设置 Alpha158 XGBoost 结果:")
    print(f"  IC均值: {ic_arr.mean():.4f}")
    print(f"  ICIR: {ic_arr.mean()/ic_arr.std():.4f}")
    print(f"  Rank IC均值: {rank_ic_arr.mean():.4f}")
    print(f"  Rank ICIR: {rank_ic_arr.mean()/rank_ic_arr.std():.4f}")
    print(f"  测试期数: {len(ic_arr)}")
    print(f"\n官方README基准: IC=0.0498, ICIR=0.3779, RankIC=0.0505, RankICIR=0.4131")
