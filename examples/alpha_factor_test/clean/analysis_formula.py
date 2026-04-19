"""
单公式因子评估工具
计算 IC, IR, RankIC, RankICIR, 胜率等指标
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from clean.data_manager import load_ohlcv
from clean.alpha_engine import AlphaEngine

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_formula")


def compute_metrics(factor: pd.Series, returns: pd.Series, start_date: str = None, end_date: str = None):
    """
    计算因子的完整指标
    
    Parameters
    ----------
    factor : pd.Series
        因子值 (MultiIndex: instrument, datetime)
    returns : pd.Series
        收益率 (MultiIndex: instrument, datetime)
    start_date : str, optional
        评估起始日期
    end_date : str, optional
        评估结束日期
    
    Returns
    -------
    dict : 包含所有指标
    """
    # 日期过滤
    if start_date:
        start_date = pd.Timestamp(start_date)
        factor = factor.loc[factor.index.get_level_values('datetime') >= start_date]
        returns = returns.loc[returns.index.get_level_values('datetime') >= start_date]
    
    if end_date:
        end_date = pd.Timestamp(end_date)
        factor = factor.loc[factor.index.get_level_values('datetime') <= end_date]
        returns = returns.loc[returns.index.get_level_values('datetime') <= end_date]
    
    # 对齐因子和收益率
    common_index = factor.index.intersection(returns.index)
    factor = factor.loc[common_index]
    returns = returns.loc[common_index]
    
    # 去除 NaN
    valid_mask = ~(factor.isna() | returns.isna())
    factor_valid = factor[valid_mask]
    returns_valid = returns[valid_mask]
    
    if len(factor_valid) == 0:
        return {"error": "无有效数据"}
    
    # 按日期分组计算每日 IC
    dates = factor_valid.index.get_level_values('datetime').unique()
    daily_ic = []
    daily_rank_ic = []
    daily_correct = []
    skipped_count = 0
    
    for date in dates:
        try:
            # 提取该日期的因子和收益（MultiIndex: instrument, datetime）
            f = factor_valid.xs(date, level='datetime')
            r = returns_valid.xs(date, level='datetime')
            
            if len(f) < 10:  # 至少10只股票
                skipped_count += 1
                continue
            
            # Pearson IC
            if f.std() > 0 and r.std() > 0:
                ic = f.corr(r)
                daily_ic.append(ic)
            else:
                skipped_count += 1
            
            # Spearman Rank IC
            if len(f.unique()) > 1 and len(r.unique()) > 1:
                rank_ic, _ = stats.spearmanr(f, r)
                daily_rank_ic.append(rank_ic)
            
            # 方向正确率（因子值与下期收益同号）
            # 简化版：因子排名高的股票，收益是否也高
            if len(f) >= 10:
                f_top = f.nlargest(len(f) // 3)  # 前1/3
                r_top = r.loc[f_top.index]
                correct = (r_top > 0).mean()  # 上涨比例
                daily_correct.append(correct)
                
        except Exception as e:
            skipped_count += 1
            continue
    
    if len(daily_ic) == 0:
        logger.warning(f"⚠️  没有计算到任何IC值！跳过了 {skipped_count} 个日期")
        logger.warning(f"  可能原因：因子值无变化(std=0) 或 收益率无变化")
    
    # 汇总指标
    results = {
        "num_dates": len(dates),
        "num_valid_pairs": len(factor_valid),
    }
    
    if len(daily_ic) > 0:
        ic_series = pd.Series(daily_ic)
        results["ic_mean"] = ic_series.mean()
        results["ic_std"] = ic_series.std()
        results["icir"] = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
        results["ic_positive_ratio"] = (ic_series > 0).mean()
        results["ic_max"] = ic_series.max()
        results["ic_min"] = ic_series.min()
    
    if len(daily_rank_ic) > 0:
        rank_ic_series = pd.Series(daily_rank_ic)
        results["rank_ic_mean"] = rank_ic_series.mean()
        results["rank_ic_std"] = rank_ic_series.std()
        results["rank_icir"] = rank_ic_series.mean() / rank_ic_series.std() if rank_ic_series.std() > 0 else 0
        results["rank_ic_positive_ratio"] = (rank_ic_series > 0).mean()
        results["rank_ic_max"] = rank_ic_series.max()
        results["rank_ic_min"] = rank_ic_series.min()
    
    if len(daily_correct) > 0:
        correct_series = pd.Series(daily_correct)
        results["win_rate"] = correct_series.mean()
        results["win_rate_std"] = correct_series.std()
    
    return results


def test_formula(formula: str, data_months: int = 3, load_data_months: int = 12):
    """
    测试单个公式
    
    Parameters
    ----------
    formula : str
        因子公式
    data_months : int
        IC评估周期（月）
    load_data_months : int
        数据加载周期（月）
    """
    logger.info("=" * 80)
    logger.info(f"测试公式: {formula}")
    logger.info("=" * 80)
    
    # 初始化 qlib
    from qlib.constant import REG_CN
    import qlib
    qlib.init(provider_uri="C:/Users/syk/.qlib/qlib_data/cn_data", region=REG_CN)
    logger.info("qlib初始化完成")
    
    # 计算日期范围
    from qlib.data import D
    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()
    ic_end_date = max_date
    ic_start_date = max_date - pd.DateOffset(months=data_months)
    load_start = (ic_start_date - pd.DateOffset(months=load_data_months)).strftime("%Y-%m-%d")
    load_end = ic_end_date.strftime("%Y-%m-%d")
    
    logger.info(f"数据加载范围: {load_start} ~ {load_end}")
    logger.info(f"IC评估范围: {ic_start_date.strftime('%Y-%m-%d')} ~ {ic_end_date.strftime('%Y-%m-%d')} ({data_months}个月)")
    
    # 加载数据
    df = load_ohlcv(start_time=load_start, end_time=load_end)
    logger.info(f"加载OHLCV: {len(df)}条, {df.index.get_level_values('instrument').nunique()}只股票")
    
    # 计算因子
    engine = AlphaEngine(df)
    try:
        factor = engine.calculate(formula)
        logger.info(f"因子计算成功: {len(factor)}个值")
        logger.info(f"  因子统计: mean={factor.mean():.6f}, std={factor.std():.6f}, "
                   f"min={factor.min():.6f}, max={factor.max():.6f}")
        logger.info(f"  NaN比例: {factor.isna().sum() / len(factor):.1%}")
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 计算收益率（下期1日收益）
    df['return_1d'] = df.groupby(level='instrument')['close'].pct_change().shift(-1)
    returns = df['return_1d']
    
    # 计算指标
    logger.info("\n" + "=" * 80)
    logger.info("评估结果")
    logger.info("=" * 80)
    
    metrics = compute_metrics(factor, returns, 
                              start_date=ic_start_date, end_date=ic_end_date)
    
    if "error" in metrics:
        logger.error(f"评估失败: {metrics['error']}")
        return
    
    # 打印结果
    logger.info(f"\n📊 基础统计:")
    logger.info(f"  评估天数: {metrics.get('num_dates', 'N/A')}")
    logger.info(f"  有效样本: {metrics.get('num_valid_pairs', 'N/A')}")
    
    logger.info(f"\n📈 Pearson IC 指标:")
    logger.info(f"  IC均值:   {metrics.get('ic_mean', 0):+.4f}")
    logger.info(f"  IC标准差: {metrics.get('ic_std', 0):.4f}")
    logger.info(f"  ICIR:     {metrics.get('icir', 0):+.4f}")
    logger.info(f"  IC正率:   {metrics.get('ic_positive_ratio', 0):.1%}")
    logger.info(f"  IC最大值: {metrics.get('ic_max', 0):+.4f}")
    logger.info(f"  IC最小值: {metrics.get('ic_min', 0):+.4f}")
    
    logger.info(f"\n🎯 Rank IC (Spearman) 指标:")
    logger.info(f"  RankIC均值: {metrics.get('rank_ic_mean', 0):+.4f}")
    logger.info(f"  RankIC标准差: {metrics.get('rank_ic_std', 0):.4f}")
    logger.info(f"  RankICIR:   {metrics.get('rank_icir', 0):+.4f}")
    logger.info(f"  RankIC正率: {metrics.get('rank_ic_positive_ratio', 0):.1%}")
    logger.info(f"  RankIC最大值: {metrics.get('rank_ic_max', 0):+.4f}")
    logger.info(f"  RankIC最小值: {metrics.get('rank_ic_min', 0):+.4f}")
    
    logger.info(f"\n🏆 胜率指标:")
    logger.info(f"  胜率: {metrics.get('win_rate', 0):.1%} (±{metrics.get('win_rate_std', 0):.4f})")
    
    # 综合评价
    logger.info(f"\n💡 综合评价:")
    rank_icir = abs(metrics.get('rank_icir', 0))
    win_rate = metrics.get('win_rate', 0)
    
    if rank_icir > 0.5:
        logger.info(f"  ✅ RankICIR绝对值={rank_icir:.2f} > 0.5，预测能力较强")
    elif rank_icir > 0.2:
        logger.info(f"  ⚠️ RankICIR绝对值={rank_icir:.2f}，预测能力一般")
    else:
        logger.info(f"  ❌ RankICIR绝对值={rank_icir:.2f} < 0.2，预测能力较弱")
    
    if win_rate > 0.55:
        logger.info(f"  ✅ 胜率={win_rate:.1%} > 55%，方向判断较准确")
    elif win_rate > 0.50:
        logger.info(f"  ⚠️ 胜率={win_rate:.1%}，方向判断一般")
    else:
        logger.info(f"  ❌ 胜率={win_rate:.1%} < 50%，方向判断较差")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试单个因子公式')
    parser.add_argument('--formula', type=str, 
                        default='-ts_decay_linear(power((close / ts_delay(close, 1) - 1) - ts_mean(close / ts_delay(close, 1) - 1, 60), 2), 20)',
                        help='因子公式')
    parser.add_argument('--months', type=int, default=3,
                        help='IC评估周期（月），默认3')
    parser.add_argument('--load_months', type=int, default=12,
                        help='数据加载周期（月），默认12')
    parser.add_argument('--long_test', action='store_true',
                        help='长时间回测模式：测试3年、2年、1年、6个月、3个月')
    
    args = parser.parse_args()
    
    if args.long_test:
        # 长时间回测：多个时间段
        logger.info("=" * 80)
        logger.info("长时间回测模式")
        logger.info("=" * 80)
        
        test_periods = [
            (36, 48, "3年"),   # 36个月评估，加载48个月数据
            (24, 36, "2年"),
            (12, 24, "1年"),
            (6, 18, "6个月"),
            (3, 12, "3个月"),
        ]
        
        results_summary = []
        
        for eval_months, load_months, label in test_periods:
            logger.info(f"\n{'='*80}")
            logger.info(f"测试周期: {label} (评估{eval_months}个月, 加载{load_months}个月)")
            logger.info(f"{'='*80}")
            
            try:
                metrics = test_formula(
                    formula=args.formula,
                    data_months=eval_months,
                    load_data_months=load_months
                )
                
                if metrics and "error" not in metrics:
                    results_summary.append({
                        "周期": label,
                        "IC": metrics.get('ic_mean', 0),
                        "ICIR": metrics.get('icir', 0),
                        "RankIC": metrics.get('rank_ic_mean', 0),
                        "RankICIR": metrics.get('rank_icir', 0),
                        "胜率": metrics.get('win_rate', 0),
                        "样本数": metrics.get('num_valid_pairs', 0),
                    })
            except Exception as e:
                logger.error(f"{label}测试失败: {e}")
        
        # 打印汇总
        logger.info(f"\n{'='*80}")
        logger.info("长时间回测汇总")
        logger.info(f"{'='*80}")
        
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            logger.info(f"\n{summary_df.to_string(index=False)}")
            
            # 保存结果
            output_file = os.path.join(os.path.dirname(__file__), 'output', 'long_backtest_results.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            summary_df.to_csv(output_file, index=False)
            logger.info(f"\n结果已保存: {output_file}")
            
            # 稳定性分析
            logger.info(f"\n💡 稳定性分析:")
            rank_icirs = [r['RankICIR'] for r in results_summary]
            if len(rank_icirs) > 1:
                std = np.std(rank_icirs)
                mean = np.mean(rank_icirs)
                if std < 0.5:
                    logger.info(f"  ✅ RankICIR标准差={std:.2f}，因子表现稳定")
                elif std < 1.0:
                    logger.info(f"  ⚠️ RankICIR标准差={std:.2f}，因子表现一般")
                else:
                    logger.info(f"  ❌ RankICIR标准差={std:.2f}，因子不稳定（可能过拟合）")
    else:
        # 单次测试
        test_formula(
            formula=args.formula,
            data_months=args.months,
            load_data_months=args.load_months
        )
