"""批量校验所有公式，找出计算异常的因子"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import logging
    from clean.formula_parser import load_all_formulas
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine

    logging.basicConfig(level=logging.WARNING)
    init_qlib()
    df = load_ohlcv(start_time="2023-01-01", end_time="2023-06-01")
    engine = AlphaEngine(df)

    a101, a191, _ = load_all_formulas()
    all_formulas = {**a101, **a191}

    ok_count = 0
    fail_count = 0
    issues = []

    for name, formula in all_formulas.items():
        try:
            result = engine.calculate(formula)
            if result is None:
                issues.append((name, "返回None"))
                fail_count += 1
                continue
            if isinstance(result, pd.Series):
                valid = result.dropna()
                if len(valid) == 0:
                    issues.append((name, "全NaN"))
                    fail_count += 1
                    continue
                if np.isinf(valid.values).sum() > len(valid) * 0.5:
                    issues.append((name, f"inf过多({np.isinf(valid.values).sum()}/{len(valid)})"))
                    fail_count += 1
                    continue
                ok_count += 1
            else:
                issues.append((name, f"返回类型异常: {type(result)}"))
                fail_count += 1
        except Exception as e:
            issues.append((name, f"异常: {str(e)[:80]}"))
            fail_count += 1

    print(f"\n校验结果: 成功={ok_count}, 失败={fail_count}, 总计={len(all_formulas)}")
    if issues:
        print(f"\n问题因子 ({len(issues)}个):")
        for name, issue in issues:
            print(f"  {name}: {issue}")
    else:
        print("所有公式校验通过!")
