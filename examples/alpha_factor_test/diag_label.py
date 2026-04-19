"""诊断Top50 LABEL0异常值和Alpha158 label处理流程"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # 1. 诊断Top50 LABEL0异常值
    print("=" * 80)
    print("1. 诊断Top50 LABEL0异常值")
    print("=" * 80)

    top50 = pd.read_pickle(r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\clean\output\all_features.pkl')
    top50.index = top50.index.rename(['datetime', 'instrument'])

    y = top50['LABEL0']
    print(f"LABEL0统计:")
    print(f"  均值: {y.mean():.6f}")
    print(f"  标准差: {y.std():.6f}")
    print(f"  最小: {y.min():.6f}")
    print(f"  最大: {y.max():.6f}")
    print(f"  中位: {y.median():.6f}")

    # 找出异常值（|收益率| > 0.2）
    extreme = y[abs(y) > 0.2]
    print(f"\n|收益率| > 20% 的样本数: {len(extreme)}")
    print(f"  占比: {len(extreme) / len(y.dropna()) * 100:.2f}%")

    # 看看最大值的详情
    if len(extreme) > 0:
        top10 = extreme.abs().nlargest(10)
        print(f"\nTop10极端收益率:")
        for idx in top10.index:
            val = y.loc[idx]
            print(f"  {idx}: {val:.6f} ({val*100:.2f}%)")

    # 检查这些极端值是否来自停牌复牌
    print(f"\n检查是否来自停牌复牌:")
    for idx in extreme.abs().nlargest(5).index:
        dt, inst = idx
        stock_data = top50.loc[(slice(None), inst), :]
        stock_data_sorted = stock_data.sort_index()
        dt_idx = stock_data_sorted.index.get_level_values('datetime').tolist().index(dt) if dt in stock_data_sorted.index.get_level_values('datetime') else -1
        if dt_idx > 0:
            prev_dt = stock_data_sorted.index.get_level_values('datetime')[dt_idx - 1]
            print(f"  {inst} @ {dt}: return={y.loc[idx]:.4f}, 前一交易日={prev_dt}")

    # 2. 分析Alpha158 label处理
    print("\n" + "=" * 80)
    print("2. Alpha158 label处理流程")
    print("=" * 80)

    from clean.data_manager import init_qlib, load_ohlcv
    init_qlib()

    # 加载原始OHLCV数据
    df = load_ohlcv()

    # Top50的label计算方式
    label_top50 = df.groupby(level="instrument")["close"].pct_change(1).shift(-1)
    print(f"Top50 label计算: close.pct_change(1).shift(-1)")
    print(f"  均值: {label_top50.mean():.6f}")
    print(f"  最大: {label_top50.max():.6f}")
    print(f"  最小: {label_top50.min():.6f}")

    # Alpha158的label计算方式
    from qlib.contrib.data.handler import Alpha158
    handler = Alpha158(instruments="csi300", start_time="2020-01-01", end_time="2023-06-01")
    a158_data = handler.fetch()
    a158_label = a158_data['LABEL0']

    print(f"\nAlpha158 label (经过DropnaLabel + CSZScoreNorm):")
    print(f"  均值: {a158_label.mean():.6f}")
    print(f"  最大: {a158_label.max():.6f}")
    print(f"  最小: {a158_label.min():.6f}")

    # 对比：如果Top50的label也做CSZScoreNorm
    from qlib.data.dataset.processor import CSZScoreNorm
    csz = CSZScoreNorm()
    label_top50_normed = csz(label_top50.to_frame('LABEL0'))['LABEL0']
    print(f"\nTop50 label经过CSZScoreNorm后:")
    print(f"  均值: {label_top50_normed.mean():.6f}")
    print(f"  最大: {label_top50_normed.max():.6f}")
    print(f"  最小: {label_top50_normed.min():.6f}")

    # 3. 样本对齐分析
    print("\n" + "=" * 80)
    print("3. 样本对齐分析")
    print("=" * 80)

    # Top50的样本
    top50_valid = top50.dropna(subset=['LABEL0'])
    top50_stocks = set(top50_valid.index.get_level_values('instrument').unique())
    top50_dates = set(top50_valid.index.get_level_values('datetime').unique())

    # Alpha158的样本
    a158_valid = a158_data.dropna(subset=['LABEL0'])
    a158_stocks = set(a158_valid.index.get_level_values('instrument').unique())
    a158_dates = set(a158_valid.index.get_level_values('datetime').unique())

    print(f"Top50: {len(top50_stocks)}只股票, {len(top50_dates)}天, {len(top50_valid)}条")
    print(f"Alpha158: {len(a158_stocks)}只股票, {len(a158_dates)}天, {len(a158_valid)}条")

    common_stocks = top50_stocks & a158_stocks
    common_dates = top50_dates & a158_dates
    print(f"共同: {len(common_stocks)}只股票, {len(common_dates)}天")

    # 构建完全对齐的索引
    common_idx = top50_valid.index.intersection(a158_valid.index)
    print(f"共同有效索引: {len(common_idx)}条")

    # 检查为什么Top50有更多数据
    top50_only = top50_valid.index.difference(a158_valid.index)
    print(f"\nTop50独有的样本: {len(top50_only)}条")

    # 分析独有样本的原因
    if len(top50_only) > 0:
        top50_only_df = pd.DataFrame(index=top50_only)
        top50_only_df['datetime'] = top50_only_df.index.get_level_values('datetime')
        top50_only_df['instrument'] = top50_only_df.index.get_level_values('instrument')

        # 按日期统计
        by_date = top50_only_df.groupby('datetime').size()
        print(f"  按日期分布:")
        print(f"    最早: {by_date.index.min()}")
        print(f"    最晚: {by_date.index.max()}")
        print(f"    集中在前N天的比例: {(by_date.index < '2020-07-01').sum() / len(by_date) * 100:.1f}%")

        # 按股票统计
        by_stock = top50_only_df.groupby('instrument').size()
        print(f"  按股票分布:")
        print(f"    涉及股票数: {len(by_stock)}")
        print(f"    每只股票平均: {by_stock.mean():.1f}条")
