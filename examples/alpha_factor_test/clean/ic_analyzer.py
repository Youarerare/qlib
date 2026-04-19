"""
IC/ICIR分析器
"""
import logging
import warnings
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def calc_ic_series(
    factor: pd.Series,
    return_next: pd.Series,
    method: str = "spearman",
    min_stocks: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    计算逐日IC序列
    Parameters
    ----------
    factor : pd.Series, MultiIndex(datetime, instrument)
    return_next : pd.Series, 同索引，下期收益率
    method : "spearman" 或 "pearson"
    min_stocks : 截面最少股票数
    start_date : 可选，起始日期过滤（如 "2023-10-01"）
    end_date : 可选，截止日期过滤（如 "2024-01-01"）
    """
    df = pd.DataFrame({"factor": factor, "ret": return_next}).dropna()

    # 日期过滤：只保留指定范围内的数据
    dates_all = sorted(df.index.get_level_values("datetime").unique())
    if start_date is not None:
        dates_all = [d for d in dates_all if d >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates_all = [d for d in dates_all if d <= pd.Timestamp(end_date)]

    ic_list = []

    for date in dates_all:
        try:
            day = df.loc[date]
        except KeyError:
            continue
        if len(day) < min_stocks:
            continue

        # 常量检测：因子值或收益率全为常数时跳过，避免 spearmanr 报 ConstantInputWarning
        factor_vals = day["factor"].values
        ret_vals = day["ret"].values
        if _is_constant(factor_vals) or _is_constant(ret_vals):
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="An input array is constant")
                if method == "spearman":
                    ic, _ = spearmanr(factor_vals, ret_vals)
                else:
                    ic, _ = pearsonr(factor_vals, ret_vals)
            # spearmanr 在常数输入时可能返回 nan
            if np.isnan(ic):
                continue
            ic_list.append({"datetime": date, "ic": ic})
        except Exception:
            continue

    if not ic_list:
        return pd.Series(dtype=float)

    return pd.DataFrame(ic_list).set_index("datetime")["ic"]


def _is_constant(arr: np.ndarray) -> bool:
    """检测数组是否为常数（所有值相同或只剩一个有效值）"""
    valid = arr[~np.isnan(arr)]
    if len(valid) <= 1:
        return True
    return bool(np.allclose(valid, valid[0]))


def calc_ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    """从IC序列计算汇总指标"""
    if len(ic_series) == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "rank_icir": np.nan, "ic_positive_ratio": np.nan, "n_periods": 0}

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_icir": icir,  # 与 icir 一致（输入已经是 rank IC 时）
        "ic_positive_ratio": (ic_series > 0).mean(),
        "n_periods": len(ic_series),
    }


def evaluate_factor(
    factor: pd.Series,
    return_next: pd.Series,
    name: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """评估单个因子，返回IC和Rank IC汇总"""
    ic_s = calc_ic_series(factor, return_next, method="pearson",
                          start_date=start_date, end_date=end_date)
    rank_ic_s = calc_ic_series(factor, return_next, method="spearman",
                               start_date=start_date, end_date=end_date)

    result = {"name": name}
    result.update({f"ic_{k}": v for k, v in calc_ic_summary(ic_s).items()})
    result.update({f"rank_{k}": v for k, v in calc_ic_summary(rank_ic_s).items()})
    # rank_icir: 从 rank IC 序列直接计算
    rank_ic_mean = rank_ic_s.mean() if len(rank_ic_s) > 0 else np.nan
    rank_ic_std = rank_ic_s.std() if len(rank_ic_s) > 0 else np.nan
    result["rank_icir"] = rank_ic_mean / rank_ic_std if rank_ic_std and rank_ic_std > 0 else 0
    # IR: pearson IC 均值 / pearson IC 标准差
    ic_mean = ic_s.mean() if len(ic_s) > 0 else np.nan
    ic_std = ic_s.std() if len(ic_s) > 0 else np.nan
    result["ir"] = ic_mean / ic_std if ic_std and ic_std > 0 else 0
    return result


def evaluate_all_factors(
    feature_df: pd.DataFrame,
    return_next: pd.Series,
    show_progress: bool = True,
) -> pd.DataFrame:
    """批量评估所有因子"""
    results = []
    for i, col in enumerate(feature_df.columns):
        if col == "LABEL0":
            continue
        try:
            r = evaluate_factor(feature_df[col], return_next, name=col)
            results.append(r)
        except Exception as e:
            logger.debug(f"评估失败: {col} - {e}")

        if show_progress and (i + 1) % 10 == 0:
            logger.info(f"  评估进度: {i + 1}/{len(feature_df.columns)}")

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values("rank_icir", key=abs, ascending=False)
        df = df.reset_index(drop=True)

    return df


def get_top_k(
    results_df: pd.DataFrame,
    k: int = 50,
    by: str = "rank_icir",
) -> pd.DataFrame:
    """获取Top K因子"""
    df = results_df.copy()
    df["abs_score"] = df[by].abs()
    df = df.sort_values("abs_score", ascending=False).head(k)
    return df.drop(columns=["abs_score"]).reset_index(drop=True)
