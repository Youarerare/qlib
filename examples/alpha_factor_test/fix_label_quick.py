"""快速修复: 只更新pkl中的LABEL0，不需要重新计算因子"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from clean.data_manager import init_qlib, load_ohlcv

    init_qlib()
    df = load_ohlcv()

    # 新label: T+1→T+2收益率 (与Alpha158一致)
    new_label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    # Winsorize
    new_label = new_label.clip(lower=-0.2, upper=0.2)

    # 加载旧pkl
    pkl_path = r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\clean\output\all_features.pkl'
    data = pd.read_pickle(pkl_path)
    data.index = data.index.rename(['datetime', 'instrument'])

    old_label = data['LABEL0'].copy()
    data['LABEL0'] = new_label.reindex(data.index)

    print(f"旧LABEL0: 均值={old_label.mean():.6f}, 最大={old_label.max():.6f}, NaN={old_label.isna().sum()}")
    print(f"新LABEL0: 均值={data['LABEL0'].mean():.6f}, 最大={data['LABEL0'].max():.6f}, NaN={data['LABEL0'].isna().sum()}")

    # 保存
    data.to_pickle(pkl_path)
    print(f"\n已更新保存到: {pkl_path}")
    print(f"形状: {data.shape}")
