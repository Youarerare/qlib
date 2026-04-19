"""检查qlib数据可用日期范围"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import qlib
    import pandas as pd
    from qlib.data import D

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    # 用calendar获取交易日历
    cal = D.calendar(start_time="2005-01-01", end_time="2026-12-31", freq="day")
    print(f"交易日历范围: {cal[0]} ~ {cal[-1]}")
    print(f"总交易日数: {len(cal)}")

    # 按年统计
    years = sorted(set(pd.Timestamp(d).year for d in cal))
    print(f"\n按年统计:")
    for y in years:
        count = sum(1 for d in cal if pd.Timestamp(d).year == y)
        print(f"  {y}: {count}个交易日")

    # 检查最新数据
    print(f"\n最新5个交易日:")
    for d in cal[-5:]:
        print(f"  {pd.Timestamp(d).strftime('%Y-%m-%d')}")

    # 检查CSI300成分股
    stock_list = D.list_instruments(D.instruments("csi300"), start_time="2020-01-01", end_time="2026-04-18", freq="day", as_list=True)
    print(f"\nCSI300成分股数: {len(stock_list)}")
