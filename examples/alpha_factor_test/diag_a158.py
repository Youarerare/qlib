"""诊断Alpha158特征质量问题"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from clean.data_manager import init_qlib, load_alpha158_data

    init_qlib()

    # 分别加载DK_I和DK_L
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    handler = Alpha158(instruments="csi300", start_time="2020-01-01", end_time="2023-06-01")

    feat_i = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    feat_l = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_L)
    label_l = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L)
    label_r = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)

    print("=== DK_I (infer) 特征 ===")
    print(f"  形状: {feat_i.shape}")
    print(f"  inf数: {np.isinf(feat_i.values).sum()}")
    print(f"  NaN数: {np.isnan(feat_i.values).sum()}")
    print(f"  NaN比例: {feat_i.isna().mean().mean():.4f}")

    print("\n=== DK_L (learn) 特征 ===")
    print(f"  形状: {feat_l.shape}")
    print(f"  inf数: {np.isinf(feat_l.values).sum()}")
    print(f"  NaN数: {np.isnan(feat_l.values).sum()}")
    print(f"  NaN比例: {feat_l.isna().mean().mean():.4f}")

    print("\n=== DK_L label ===")
    print(f"  均值: {label_l.mean():.6f}")
    print(f"  标准差: {label_l.std():.6f}")
    print(f"  最小: {label_l.min():.6f}")
    print(f"  最大: {label_l.max():.6f}")

    print("\n=== DK_R label ===")
    print(f"  均值: {label_r.mean():.6f}")
    print(f"  标准差: {label_r.std():.6f}")
    print(f"  最小: {label_r.min():.6f}")
    print(f"  最大: {label_r.max():.6f}")

    # DK_I vs DK_L 的行数差异
    print(f"\nDK_I行数: {len(feat_i)}, DK_L行数: {len(feat_l)}")
    print(f"DropnaLabel删除了: {len(feat_i) - len(feat_l)}行")
