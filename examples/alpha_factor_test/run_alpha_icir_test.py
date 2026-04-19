"""
Alpha因子ICIR回测脚本
测试alpha101和alpha191在A股的IC和RankICIR表现

用法:
    python run_alpha_icir_test.py --alpha-type alpha101
    python run_alpha_icir_test.py --alpha-type alpha191
    python run_alpha_icir_test.py --alpha-type all
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from formula_parser import load_alpha101_formulas, load_alpha191_formulas
from alpha_calculator import AlphaCalculatorAuto
from qlib.contrib.eva.alpha import calc_ic


def load_qlib_data(instruments="csi300", start_time="2020-01-01", end_time="2023-12-31"):
    """从Qlib加载数据（不包含行业数据）"""
    import qlib
    from qlib.data import D
    
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
    
    fields = ['$open', '$high', '$low', '$close', '$volume', '$factor', '$change']
    
    df = D.instruments(instruments)
    df_data = D.features(df, fields, start_time=start_time, end_time=end_time)
    
    df_data = df_data.rename(columns={
        '$open': 'open', '$high': 'high', '$low': 'low',
        '$close': 'close', '$volume': 'volume', '$factor': 'factor', '$change': 'change'
    })
    
    return df_data


def calculate_ic_single(factor_values, df_data, shift=1):
    """计算单个因子的IC和RankICIR"""
    label = df_data.groupby(level='instrument')['close'].pct_change(shift).shift(-shift)
    
    common_index = factor_values.index.intersection(label.index)
    factor_aligned = factor_values.loc[common_index]
    label_aligned = label.loc[common_index]
    
    valid_mask = ~(factor_aligned.isna() | label_aligned.isna())
    factor_valid = factor_aligned[valid_mask]
    label_valid = label_aligned[valid_mask]
    
    if len(factor_valid) == 0:
        return None
    
    try:
        ic, rank_ic = calc_ic(factor_valid, label_valid, date_col='datetime')
        ic_clean = ic.dropna()
        rank_ic_clean = rank_ic.dropna()
        
        if len(ic_clean) == 0:
            return None
        
        return {
            'IC_mean': ic_clean.mean(),
            'IC_std': ic_clean.std(),
            'ICIR': ic_clean.mean() / ic_clean.std() if ic_clean.std() != 0 else np.nan,
            'IC_positive_ratio': (ic_clean > 0).sum() / len(ic_clean),
            'Rank_IC_mean': rank_ic_clean.mean(),
            'Rank_IC_std': rank_ic_clean.std(),
            'Rank_ICIR': rank_ic_clean.mean() / rank_ic_clean.std() if rank_ic_clean.std() != 0 else np.nan,
            'Rank_IC_positive_ratio': (rank_ic_clean > 0).sum() / len(rank_ic_clean),
            'num_periods': len(ic_clean)
        }
    except:
        return None


def run_alpha_test(calc, alpha_dict, df_data, alpha_type_name, forecast_period=1):
    """运行Alpha因子测试"""
    results_list = []
    alpha_names = sorted(alpha_dict.keys())
    total = len(alpha_names)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, alpha_name in enumerate(alpha_names, 1):
        formula = alpha_dict[alpha_name]
        elapsed = time.time() - start_time
        
        if i > 1:
            avg_time = elapsed / (i - 1)
            remaining = avg_time * (total - i + 1)
            eta_str = f", 预计剩余: {remaining:.0f}秒"
        else:
            eta_str = ""
        
        print(f"[{i}/{total}] {alpha_name} (已用: {elapsed:.1f}秒{eta_str})", flush=True)
        
        try:
            factor_values = calc.calculate_alpha(formula)
            ic_result = calculate_ic_single(factor_values, df_data, forecast_period)
            
            if ic_result:
                results_list.append({'alpha_name': alpha_name, 'source': alpha_type_name, **ic_result})
                success_count += 1
                print(f"  OK IC={ic_result['IC_mean']:.6f}, ICIR={ic_result['ICIR']:.4f}, RankICIR={ic_result['Rank_ICIR']:.4f}", flush=True)
            else:
                fail_count += 1
                print(f"  SKIP - IC无效", flush=True)
                
        except Exception as e:
            error_msg = str(e)[:60]
            fail_count += 1
            print(f"  FAIL: {error_msg}", flush=True)
        
        if i % 10 == 0:
            print(f"\n--- 进度: {i}/{total} ({i/total*100:.1f}%) | 成功: {success_count} | 失败: {fail_count} ---\n", flush=True)
    
    print(f"\n{alpha_type_name} 完成! 成功: {success_count}, 失败: {fail_count}, 耗时: {time.time() - start_time:.1f}秒\n")
    
    return pd.DataFrame(results_list)


def print_top50(results_df):
    """打印Top50因子结果"""
    if len(results_df) == 0:
        print("无有效结果")
        return
    
    results_df['ICIR_abs'] = results_df['ICIR'].abs()
    results_df['Rank_ICIR_abs'] = results_df['Rank_ICIR'].abs()
    
    print("=" * 120)
    print("TOP 50 因子 - 按 ICIR 绝对值排序")
    print("=" * 120)
    top50_icir = results_df.nlargest(50, 'ICIR_abs')
    print(top50_icir[['alpha_name', 'source', 'IC_mean', 'IC_std', 'ICIR', 'ICIR_abs', 'IC_positive_ratio', 'num_periods']].to_string(index=False))
    
    print("\n" + "=" * 130)
    print("TOP 50 因子 - 按 Rank_ICIR 绝对值排序")
    print("=" * 130)
    top50_rank = results_df.nlargest(50, 'Rank_ICIR_abs')
    print(top50_rank[['alpha_name', 'source', 'Rank_IC_mean', 'Rank_IC_std', 'Rank_ICIR', 'Rank_ICIR_abs', 'Rank_IC_positive_ratio', 'num_periods']].to_string(index=False))
    
    top50_icir.to_csv('top50_by_icir.csv', index=False)
    top50_rank.to_csv('top50_by_rank_icir.csv', index=False)
    results_df.to_csv('all_alpha_results.csv', index=False)
    
    print("\n\n结果已保存到:")
    print("  - top50_by_icir.csv")
    print("  - top50_by_rank_icir.csv")
    print("  - all_alpha_results.csv")


def main():
    parser = argparse.ArgumentParser(description='Alpha因子ICIR测试')
    parser.add_argument('--alpha-type', default='alpha101', choices=['alpha101', 'alpha191', 'all'])
    parser.add_argument('--instruments', default='csi300')
    parser.add_argument('--start-time', default='2020-01-01')
    parser.add_argument('--end-time', default='2023-12-31')
    parser.add_argument('--forecast-period', type=int, default=1)
    parser.add_argument('--alpha101-path', default=r'C:\Users\syk\Desktop\git_repo\auto_alpha\research_formula_candidates.txt')
    parser.add_argument('--alpha191-path', default=r'C:\Users\syk\Desktop\git_repo\auto_alpha\alpha191.txt')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Alpha因子ICIR回测")
    print("=" * 80)
    print(f"Alpha类型: {args.alpha_type}")
    print(f"股票池: {args.instruments}")
    print(f"时间范围: {args.start_time} ~ {args.end_time}")
    print(f"预测周期: {args.forecast_period}天")
    print(f"开始时间: {datetime.now()}")
    print("=" * 80)
    print()
    
    print("加载Qlib数据...")
    df_data = load_qlib_data(
        instruments=args.instruments,
        start_time=args.start_time,
        end_time=args.end_time
    )
    print(f"加载了 {len(df_data)} 条数据, {df_data.index.get_level_values(1).nunique()} 只股票\n")
    
    print("创建计算器...")
    calc = AlphaCalculatorAuto(df_data)
    
    all_results = []
    
    if args.alpha_type in ['alpha101', 'all']:
        print("=" * 80)
        print("开始测试 Alpha101 因子")
        print("=" * 80)
        alpha101_dict = load_alpha101_formulas(args.alpha101_path)
        print(f"加载了 {len(alpha101_dict)} 个Alpha101公式\n")
        results = run_alpha_test(calc, alpha101_dict, df_data, 'alpha101', args.forecast_period)
        all_results.append(results)
    
    if args.alpha_type in ['alpha191', 'all']:
        print("=" * 80)
        print("开始测试 Alpha191 因子")
        print("=" * 80)
        alpha191_dict = load_alpha191_formulas(args.alpha191_path)
        print(f"加载了 {len(alpha191_dict)} 个Alpha191公式\n")
        results = run_alpha_test(calc, alpha191_dict, df_data, 'alpha191', args.forecast_period)
        all_results.append(results)
    
    if len(all_results) > 0:
        combined_results = pd.concat(all_results, ignore_index=True)
        print("=" * 80)
        print(f"全部测试完成 - 共 {len(combined_results)} 个有效因子")
        print("=" * 80)
        print()
        print_top50(combined_results)


if __name__ == "__main__":
    main()
