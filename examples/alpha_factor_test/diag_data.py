"""打印特征和label的样本数据"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.config import OUTPUT_DIR

    init_qlib()

    # ============ 1. Top50 特征 ============
    print("=" * 100)
    print("1. Top50 特征数据")
    print("=" * 100)

    top50 = pd.read_pickle(OUTPUT_DIR / "all_features.pkl")
    top50.index = top50.index.rename(["datetime", "instrument"])

    ic_csv = OUTPUT_DIR / "top50_by_rank_icir.csv"
    if ic_csv.exists():
        top50_names = pd.read_csv(ic_csv)["name"].head(50).tolist()
        available = [c for c in top50_names if c in top50.columns]
        print(f"Top50因子名: {available[:10]}... (共{len(available)}个)")
    else:
        available = [c for c in top50.columns if c != "LABEL0"][:50]

    sample_stock = "SH600000"
    stock_data = top50.loc[(slice(None), sample_stock), available[:5] + ["LABEL0"]].dropna().head(10)
    print(f"\nSH600000 前10个有效交易日:")
    print(stock_data.to_string())

    # ============ 2. Alpha158 特征 ============
    print("\n" + "=" * 100)
    print("2. Alpha158 特征数据")
    print("=" * 100)

    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    handler = Alpha158(instruments="csi300", start_time="2020-01-01", end_time="2023-06-01")

    a158_feat_i = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    a158_feat_i.index = a158_feat_i.index.rename(["datetime", "instrument"])
    print(f"Alpha158 DK_I特征: {a158_feat_i.shape}")

    a158_stock_feat = a158_feat_i.loc[(slice(None), sample_stock), :].dropna().head(10)
    print(f"\nSH600000 Alpha158特征 前10个有效交易日 (前5列):")
    print(a158_stock_feat.iloc[:, :5].to_string())

    # ============ 3. Label对比 ============
    print("\n" + "=" * 100)
    print("3. Label对比 (SH600000)")
    print("=" * 100)

    a158_label_r = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
    a158_label_r.index = a158_label_r.index.rename(["datetime", "instrument"])
    if isinstance(a158_label_r, pd.DataFrame):
        a158_label_r = a158_label_r.squeeze()

    a158_label_l = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L)
    a158_label_l.index = a158_label_l.index.rename(["datetime", "instrument"])
    if isinstance(a158_label_l, pd.DataFrame):
        a158_label_l = a158_label_l.squeeze()

    df = load_ohlcv()
    raw_label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    # 统一用日期索引对齐
    top50_label = top50.loc[(slice(None), sample_stock), "LABEL0"]
    a158_r = a158_label_r.loc[(slice(None), sample_stock)]
    a158_l = a158_label_l.loc[(slice(None), sample_stock)]
    raw = raw_label.loc[(slice(None), sample_stock)]

    # 取共同日期
    common_dates = sorted(
        set(top50_label.dropna().index.get_level_values("datetime")) &
        set(a158_r.dropna().index.get_level_values("datetime")) &
        set(raw.dropna().index.get_level_values("datetime"))
    )[:10]

    print(f"\n{'日期':<20} {'Top50_LABEL0':>14} {'A158_DK_R':>14} {'A158_DK_L':>14} {'OHLCV自算':>14}")
    print("-" * 80)
    for dt in common_dates:
        t50 = top50_label.get((dt, sample_stock), np.nan)
        ar = a158_r.get((dt, sample_stock), np.nan)
        al = a158_l.get((dt, sample_stock), np.nan)
        rw = raw.get((dt, sample_stock), np.nan)
        print(f"{str(dt)[:10]:<20} {t50:>14.6f} {ar:>14.6f} {al:>14.6f} {rw:>14.6f}")

    # ============ 4. 整体统计 ============
    print("\n" + "=" * 100)
    print("4. 整体统计")
    print("=" * 100)

    print(f"Top50 LABEL0:  均值={top50['LABEL0'].mean():.6f}, 标准差={top50['LABEL0'].std():.6f}, "
          f"最小={top50['LABEL0'].min():.6f}, 最大={top50['LABEL0'].max():.6f}, NaN={top50['LABEL0'].isna().sum()}")
    print(f"Alpha158 DK_R: 均值={a158_label_r.mean():.6f}, 标准差={a158_label_r.std():.6f}, "
          f"最小={a158_label_r.min():.6f}, 最大={a158_label_r.max():.6f}, NaN={a158_label_r.isna().sum()}")
    print(f"Alpha158 DK_L: 均值={a158_label_l.mean():.6f}, 标准差={a158_label_l.std():.6f}, "
          f"最小={a158_label_l.min():.6f}, 最大={a158_label_l.max():.6f}, NaN={a158_label_l.isna().sum()}")
    print(f"OHLCV自算:     均值={raw_label.mean():.6f}, 标准差={raw_label.std():.6f}, "
          f"最小={raw_label.min():.6f}, 最大={raw_label.max():.6f}, NaN={raw_label.isna().sum()}")

    # ============ 5. 重复因子检查 ============
    print("\n" + "=" * 100)
    print("5. 重复因子检查 (alpha191_alpha185 vs alpha101_alpha033)")
    print("=" * 100)

    if "alpha191_alpha185" in top50.columns and "alpha101_alpha033" in top50.columns:
        a185 = top50["alpha191_alpha185"].dropna()
        a033 = top50["alpha101_alpha033"].dropna()
        common_idx = a185.index.intersection(a033.index)
        corr = a185.loc[common_idx].corr(a033.loc[common_idx])
        diff = (a185.loc[common_idx] - a033.loc[common_idx]).abs()
        print(f"alpha191_alpha185 vs alpha101_alpha033:")
        print(f"  相关系数: {corr:.6f}")
        print(f"  最大差异: {diff.max():.8f}")
        print(f"  平均差异: {diff.mean():.8f}")
        print(f"  alpha191_alpha185公式: rank((-1*(power((1-(open/close)), 2))))")
        print(f"  alpha101_alpha033公式: rank(-(1 - (open / close)))")
        print(f"  → 如果相关系数=1, 说明power(x,2)没有被正确解析!")

    # ============ 6. 日期范围 ============
    print("\n" + "=" * 100)
    print("6. 日期范围")
    print("=" * 100)

    from clean.config import BACKTEST
    t50_dates = sorted(top50.index.get_level_values("datetime").unique())
    a158_dates = sorted(a158_feat_i.index.get_level_values("datetime").unique())
    print(f"Top50: {t50_dates[0]} ~ {t50_dates[-1]}, 共{len(t50_dates)}天")
    print(f"Alpha158: {a158_dates[0]} ~ {a158_dates[-1]}, 共{len(a158_dates)}天")
    print(f"训练截止: {BACKTEST.train_end}, 测试开始: {BACKTEST.test_start}")
