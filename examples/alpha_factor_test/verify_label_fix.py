"""验证label修复效果 - 对比修复前后的label统计和样本对齐"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    print("=" * 80)
    print("验证Label修复效果")
    print("=" * 80)

    # 1. 加载Top50原始数据（旧label）
    top50 = pd.read_pickle(r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\clean\output\all_features.pkl')
    top50.index = top50.index.rename(['datetime', 'instrument'])

    old_label = top50['LABEL0']
    print(f"\n旧Top50 LABEL0 (T→T+1收益率):")
    print(f"  均值: {old_label.mean():.6f}")
    print(f"  标准差: {old_label.std():.6f}")
    print(f"  最小: {old_label.min():.6f}")
    print(f"  最大: {old_label.max():.6f}")
    print(f"  NaN数: {old_label.isna().sum()}")

    # 2. 计算新label (Alpha158公式: T+1→T+2收益率)
    from clean.data_manager import init_qlib, load_ohlcv
    init_qlib()
    df = load_ohlcv()

    new_label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    print(f"\n新Top50 LABEL0 (T+1→T+2收益率, 与Alpha158一致):")
    print(f"  均值: {new_label.mean():.6f}")
    print(f"  标准差: {new_label.std():.6f}")
    print(f"  最小: {new_label.min():.6f}")
    print(f"  最大: {new_label.max():.6f}")
    print(f"  NaN数: {new_label.isna().sum()}")

    # 3. Winsorize后的新label
    new_label_winsorized = new_label.clip(lower=-0.2, upper=0.2)
    print(f"\n新Top50 LABEL0 (winsorize后):")
    print(f"  均值: {new_label_winsorized.mean():.6f}")
    print(f"  标准差: {new_label_winsorized.std():.6f}")
    print(f"  最小: {new_label_winsorized.min():.6f}")
    print(f"  最大: {new_label_winsorized.max():.6f}")
    print(f"  NaN数: {new_label_winsorized.isna().sum()}")

    # 4. 加载Alpha158的label做对比
    from clean.data_manager import load_alpha158_data
    a158_data = load_alpha158_data()
    a158_label = a158_data['LABEL0']
    print(f"\nAlpha158 LABEL0 (Ref($close,-2)/Ref($close,-1)-1):")
    print(f"  均值: {a158_label.mean():.6f}")
    print(f"  标准差: {a158_label.std():.6f}")
    print(f"  最小: {a158_label.min():.6f}")
    print(f"  最大: {a158_label.max():.6f}")
    print(f"  NaN数: {a158_label.isna().sum()}")

    # 5. 计算新旧label与Alpha158 label的相关性
    common_idx_old = old_label.dropna().index.intersection(a158_label.dropna().index)
    common_idx_new = new_label_winsorized.dropna().index.intersection(a158_label.dropna().index)

    from scipy.stats import pearsonr, spearmanr

    if len(common_idx_old) > 100:
        corr_old, _ = pearsonr(old_label.loc[common_idx_old], a158_label.loc[common_idx_old])
        print(f"\n旧Top50 label vs Alpha158 label: 相关系数={corr_old:.6f} (样本={len(common_idx_old)})")

    if len(common_idx_new) > 100:
        corr_new, _ = pearsonr(new_label_winsorized.loc[common_idx_new], a158_label.loc[common_idx_new])
        print(f"新Top50 label vs Alpha158 label: 相关系数={corr_new:.6f} (样本={len(common_idx_new)})")

    # 6. 样本对齐分析
    print("\n" + "=" * 80)
    print("样本对齐分析")
    print("=" * 80)

    top50_stocks = set(top50.index.get_level_values('instrument').unique())
    a158_stocks = set(a158_data.index.get_level_values('instrument').unique())
    common_stocks = top50_stocks & a158_stocks
    print(f"Top50股票数: {len(top50_stocks)}, Alpha158股票数: {len(a158_stocks)}, 共同: {len(common_stocks)}")

    top50_dates = set(top50.index.get_level_values('datetime').unique())
    a158_dates = set(a158_data.index.get_level_values('datetime').unique())
    common_dates = sorted(top50_dates & a158_dates)
    print(f"Top50交易日数: {len(top50_dates)}, Alpha158交易日数: {len(a158_dates)}, 共同: {len(common_dates)}")

    # 模拟对齐后的样本数
    top50_aligned = top50.loc[
        top50.index.get_level_values('instrument').isin(common_stocks) &
        top50.index.get_level_values('datetime').isin(common_dates)
    ]
    a158_aligned = a158_data.loc[
        a158_data.index.get_level_values('instrument').isin(common_stocks) &
        a158_data.index.get_level_values('datetime').isin(common_dates)
    ]
    print(f"对齐后 Top50: {top50_aligned.shape[0]}条")
    print(f"对齐后 Alpha158: {a158_aligned.shape[0]}条")

    # 特征NaN导致的差异
    X1 = top50_aligned.drop(columns=['LABEL0']).replace([np.inf, -np.inf], np.nan)
    X2 = a158_aligned.drop(columns=['LABEL0']).replace([np.inf, -np.inf], np.nan)

    X1_valid = (X1.isna().mean(axis=1) <= 0.5)
    X2_valid = (X2.isna().mean(axis=1) <= 0.5)

    common_valid_idx = X1_valid.index.intersection(X2_valid.index)
    print(f"Top50特征有效行: {X1_valid.sum()}")
    print(f"Alpha158特征有效行: {X2_valid.sum()}")
    print(f"共同有效行: {len(common_valid_idx)}")
    print(f"\n修复后两个模型的训练/测试样本数将完全一致: {len(common_valid_idx)}条")
