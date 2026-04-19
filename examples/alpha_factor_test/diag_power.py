"""验证power()函数是否正确工作"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine

    init_qlib()
    df = load_ohlcv()
    engine = AlphaEngine(df)

    # 测试power函数
    close = df["close"]
    open_ = df["open"]

    # alpha191_alpha185: rank((-1*(power((1-(open/close)), 2))))
    # alpha101_alpha033: rank(-(1 - (open / close)))

    # 手动计算
    ratio = 1 - (open_ / close)
    print("1-(open/close) 统计:")
    print(f"  均值={ratio.mean():.6f}, 标准差={ratio.std():.6f}, 最小={ratio.min():.6f}, 最大={ratio.max():.6f}")

    # power(ratio, 2)
    powered = engine._signed_power(ratio, 2)
    print(f"\npower(ratio, 2) 统计:")
    print(f"  均值={powered.mean():.6f}, 标准差={powered.std():.6f}, 最小={powered.min():.6f}, 最大={powered.max():.6f}")

    # ratio^2 vs ratio
    print(f"\nratio vs power(ratio,2) 前10个值:")
    for i in range(10):
        print(f"  ratio={ratio.iloc[i]:.6f}, ratio^2={powered.iloc[i]:.6f}, np.power={np.power(ratio.iloc[i], 2):.6f}")

    # rank比较
    rank_ratio = engine.rank(-ratio)
    rank_powered = engine.rank(-powered)
    print(f"\nrank(-ratio) vs rank(-power(ratio,2)):")
    print(f"  rank(-ratio) 前5: {rank_ratio.head(5).values}")
    print(f"  rank(-powered) 前5: {rank_powered.head(5).values}")

    # 完整公式计算
    f185 = "rank((-1*(power((1-(open/close)), 2))))"
    f033 = "rank(-(1 - (open / close)))"

    r185 = engine.calculate(f185)
    r033 = engine.calculate(f033)

    common = r185.dropna().index.intersection(r033.dropna().index)
    corr = r185.loc[common].corr(r033.loc[common])
    diff = (r185.loc[common] - r033.loc[common]).abs()
    print(f"\n完整公式计算结果:")
    print(f"  相关系数: {corr:.6f}")
    print(f"  最大差异: {diff.max():.8f}")
    print(f"  alpha185 前5: {r185.dropna().head(5).values}")
    print(f"  alpha033 前5: {r033.dropna().head(5).values}")

    # 关键: rank是单调变换，如果x和x^2的rank相同，说明x全是正或全是负
    # 对于 1-(open/close), 当close>open时为正, close<open时为负
    # rank(-x) 和 rank(-x^2) 只有在x全为正或全为负时才相同
    # 但实际上x有正有负，所以rank(-x)和rank(-x^2)应该不同
    print(f"\n1-(open/close) 正值比例: {(ratio > 0).mean():.4f}")
    print(f"1-(open/close) 负值比例: {(ratio < 0).mean():.4f}")
