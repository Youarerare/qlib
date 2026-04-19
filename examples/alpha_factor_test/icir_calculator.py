"""
ICIR计算和评估模块
使用Qlib内置的IC计算功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 使用Qlib内置的IC计算函数
from qlib.contrib.eva.alpha import calc_ic, calc_all_ic


def calculate_icir_stats(ic: pd.Series) -> Dict[str, float]:
    """
    计算ICIR统计量
    
    Parameters
    ----------
    ic : pd.Series
        每日IC值
        
    Returns
    -------
    dict
        ICIR统计量
    """
    ic_clean = ic.dropna()
    
    if len(ic_clean) == 0:
        return {
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan,
            'num_periods': 0
        }
    
    return {
        'ic_mean': ic_clean.mean(),
        'ic_std': ic_clean.std(),
        'icir': ic_clean.mean() / ic_clean.std() if ic_clean.std() != 0 else np.nan,
        'ic_positive_ratio': (ic_clean > 0).sum() / len(ic_clean),
        'num_periods': len(ic_clean)
    }


def calculate_label(df: pd.DataFrame, shift: int = 1) -> pd.Series:
    """
    计算标签值（未来收益率）
    
    Parameters
    ----------
    df : pd.DataFrame
        包含close列的DataFrame
    shift : int
        预测期数，默认1天
        
    Returns
    -------
    pd.Series
        未来收益率
    """
    returns = df.groupby(level='instrument')['close'].pct_change(shift)
    return returns.shift(-shift)


def evaluate_single_alpha(factor_values: pd.Series, 
                         df_data: pd.DataFrame,
                         shift: int = 1) -> Dict:
    """
    评估单个alpha因子的ICIR
    
    Parameters
    ----------
    factor_values : pd.Series
        因子值，MultiIndex (datetime, instrument)
    df_data : pd.DataFrame
        原始数据
    shift : int
        预测期数
        
    Returns
    -------
    dict
        评估结果
    """
    # 计算标签（未来收益率）
    label = calculate_label(df_data, shift)
    
    # 使用Qlib内置函数计算IC和Rank IC
    ic, rank_ic = calc_ic(factor_values, label, date_col='datetime')
    
    # 计算ICIR统计量
    ic_stats = calculate_icir_stats(ic)
    rank_ic_stats = calculate_icir_stats(rank_ic)
    
    return {
        'IC_mean': ic_stats['ic_mean'],
        'IC_std': ic_stats['ic_std'],
        'ICIR': ic_stats['icir'],
        'IC_positive_ratio': ic_stats['ic_positive_ratio'],
        'Rank_IC_mean': rank_ic_stats['ic_mean'],
        'Rank_IC_std': rank_ic_stats['ic_std'],
        'Rank_ICIR': rank_ic_stats['icir'],
        'Rank_IC_positive_ratio': rank_ic_stats['ic_positive_ratio'],
        'num_periods': ic_stats['num_periods'],
        'ic_series': ic,
        'rank_ic_series': rank_ic
    }


def evaluate_multiple_alphas(alpha_dict: Dict[str, pd.Series],
                            df_data: pd.DataFrame,
                            shift: int = 1) -> pd.DataFrame:
    """
    评估多个alpha因子的ICIR
    
    Parameters
    ----------
    alpha_dict : dict
        {alpha_name: factor_values} 字典
    df_data : pd.DataFrame
        原始数据
    shift : int
        预测期数
        
    Returns
    -------
    pd.DataFrame
        评估结果汇总
    """
    results = []
    
    for alpha_name, factor_values in alpha_dict.items():
        try:
            print(f"正在评估 {alpha_name}...")
            stats = evaluate_single_alpha(factor_values, df_data, shift)
            
            results.append({
                'alpha_name': alpha_name,
                'IC_mean': stats['IC_mean'],
                'IC_std': stats['IC_std'],
                'ICIR': stats['ICIR'],
                'IC_positive_ratio': stats['IC_positive_ratio'],
                'Rank_IC_mean': stats['Rank_IC_mean'],
                'Rank_IC_std': stats['Rank_IC_std'],
                'Rank_ICIR': stats['Rank_ICIR'],
                'Rank_IC_positive_ratio': stats['Rank_IC_positive_ratio'],
                'num_periods': stats['num_periods']
            })
        except Exception as e:
            print(f"评估 {alpha_name} 失败: {str(e)}")
            results.append({
                'alpha_name': alpha_name,
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'IC_positive_ratio': np.nan,
                'Rank_IC_mean': np.nan,
                'Rank_IC_std': np.nan,
                'Rank_ICIR': np.nan,
                'Rank_IC_positive_ratio': np.nan,
                'num_periods': 0
            })
    
    return pd.DataFrame(results)


def format_icir_report(results_df: pd.DataFrame) -> str:
    """
    格式化ICIR报告
    
    Parameters
    ----------
    results_df : pd.DataFrame
        评估结果
        
    Returns
    -------
    str
        格式化的报告
    """
    report = []
    report.append("=" * 80)
    report.append("Alpha因子ICIR评估报告")
    report.append("=" * 80)
    
    # 按ICIR降序排序
    sorted_df = results_df.dropna(subset=['ICIR']).sort_values('ICIR', ascending=False)
    
    report.append(f"\n有效因子数量: {len(sorted_df)}")
    report.append(f"\n{'Alpha':<15} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'IC_pos%':>10} {'RankICIR':>10}")
    report.append("-" * 80)
    
    for _, row in sorted_df.iterrows():
        report.append(
            f"{row['alpha_name']:<15} "
            f"{row['IC_mean']:>10.6f} "
            f"{row['IC_std']:>10.6f} "
            f"{row['ICIR']:>10.4f} "
            f"{row['IC_positive_ratio']:>10.4f} "
            f"{row['Rank_ICIR']:>10.4f}"
        )
    
    report.append("\n" + "=" * 80)
    report.append("Top 10 Alpha by ICIR:")
    report.append("=" * 80)
    
    for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
        report.append(f"{i:2d}. {row['alpha_name']} - ICIR: {row['ICIR']:.4f}, IC: {row['IC_mean']:.6f}")
    
    return '\n'.join(report)


def save_results(results_df: pd.DataFrame, output_path: str):
    """
    保存评估结果
    
    Parameters
    ----------
    results_df : pd.DataFrame
        评估结果
    output_path : str
        输出文件路径
    """
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    # 测试ICIR计算
    print("ICIR计算模块已就绪，使用Qlib内置的calc_ic函数")
