"""
统一回测引擎 - 复用 clean 模块的回测逻辑，提供分组收益、多空收益等扩展指标
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .config import BACKTEST, GROUP_BT, THRESHOLD
from .database import add_factor, should_auto_ingest, exists

logger = logging.getLogger(__name__)

# QLib 初始化标记
_qlib_initialized = False


def _ensure_qlib():
    """确保 QLib 已初始化"""
    global _qlib_initialized
    if not _qlib_initialized:
        import qlib
        from .config import QLIB_DATA_URI
        qlib.init(provider_uri=QLIB_DATA_URI)
        _qlib_initialized = True
        logger.info("QLib 初始化完成（backtest_engine）")


def _load_data(instruments: str = None, start_time: str = None,
               end_time: str = None, load_data_months: int = None):
    """加载数据（复用 clean.data_manager）"""
    # 确保 clean 包可导入
    _alpha_dir = Path(__file__).resolve().parent.parent / "examples" / "alpha_factor_test"
    if str(_alpha_dir) not in sys.path:
        sys.path.insert(0, str(_alpha_dir))

    from clean.data_manager import init_qlib, load_ohlcv
    from qlib.data import D

    init_qlib()

    instruments = instruments or BACKTEST.instruments
    load_months = load_data_months or BACKTEST.load_data_months

    # 计算数据加载范围
    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()

    data_months = BACKTEST.data_months
    ic_start_date = max_date - pd.DateOffset(months=data_months)
    load_start = (ic_start_date - pd.DateOffset(months=load_months)).strftime("%Y-%m-%d")
    load_end = max_date.strftime("%Y-%m-%d")

    df = load_ohlcv(instruments=instruments, start_time=load_start, end_time=load_end)
    return df


def _compute_returns(df: pd.DataFrame) -> pd.Series:
    """计算下期收益率（与 clean 模块一致：shift(-2)/shift(-1) - 1）"""
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    return returns


def _compute_ic_dates(df: pd.DataFrame, data_months: int = None):
    """计算 IC 评估的日期范围"""
    data_months = data_months or BACKTEST.data_months
    dates = df.index.get_level_values("datetime")
    max_date = dates.max()
    start_date = (max_date - pd.DateOffset(months=data_months)).strftime("%Y-%m-%d")
    end_date = max_date.strftime("%Y-%m-%d")
    return start_date, end_date


def compute_group_returns(factor: pd.Series, returns: pd.Series,
                          n_groups: int = 5, start_date: str = None,
                          end_date: str = None) -> Dict[str, float]:
    """
    计算分组收益

    Parameters
    ----------
    factor : pd.Series, MultiIndex(datetime, instrument)
    returns : pd.Series, 同索引，下期收益率
    n_groups : int
        分组数
    start_date, end_date : str
        评估日期范围

    Returns
    -------
    dict
        {"g1": 0.05, "g2": 0.03, ..., "long_short": 0.02, "top_excess": 0.01}
    """
    df = pd.DataFrame({"factor": factor, "ret": returns}).dropna()

    # 日期过滤：需要在 dropna 之后重新获取 dates
    if start_date or end_date:
        dates = df.index.get_level_values("datetime")
        if start_date:
            df = df[dates >= pd.Timestamp(start_date)]
            dates = df.index.get_level_values("datetime")  # 重新获取
        if end_date:
            df = df[dates <= pd.Timestamp(end_date)]

    group_rets = {}
    for gi in range(1, n_groups + 1):
        group_rets[f"g{gi}"] = 0.0

    n_days = 0
    for date, day_df in df.groupby(level="datetime"):
        if len(day_df) < n_groups * 2:
            continue

        # 按因子值分组
        day_df = day_df.sort_values("factor", ascending=False)
        day_df["group"] = pd.qcut(day_df["factor"], n_groups, labels=False, duplicates="drop") + 1

        day_rets = {}
        for gi in range(1, n_groups + 1):
            g = day_df[day_df["group"] == gi]
            if len(g) > 0:
                day_rets[gi] = g["ret"].mean()

        for gi in range(1, n_groups + 1):
            if gi in day_rets:
                group_rets[f"g{gi}"] += day_rets[gi]

        n_days += 1

    if n_days > 0:
        for gi in range(1, n_groups + 1):
            group_rets[f"g{gi}"] /= n_days

    # 多空收益 = 第一组均值 - 最后一组均值
    group_rets["long_short"] = group_rets.get("g1", 0) - group_rets.get(f"g{n_groups}", 0)
    # 第一组超额 = 第一组均值 - 所有组均值
    all_mean = np.mean([group_rets.get(f"g{gi}", 0) for gi in range(1, n_groups + 1)])
    group_rets["top_excess"] = group_rets.get("g1", 0) - all_mean

    return group_rets


def run_backtest(expression: str, instruments: str = None,
                 data_months: int = None, load_data_months: int = None,
                 compute_groups: bool = True) -> Dict[str, Any]:
    """
    统一回测接口：对单个因子表达式执行完整回测

    Parameters
    ----------
    expression : str
        因子表达式
    instruments : str, optional
        股票池
    data_months : int, optional
        IC评估月数
    load_data_months : int, optional
        数据加载月数
    compute_groups : bool
        是否计算分组收益

    Returns
    -------
    dict
        回测结果，包含 ic_mean, icir, ic_win_rate, rank_ic, rank_icir,
        long_short_return, top_group_excess, group_returns 等
    """
    # 确保 clean 包可导入
    _alpha_dir = Path(__file__).resolve().parent.parent / "examples" / "alpha_factor_test"
    if str(_alpha_dir) not in sys.path:
        sys.path.insert(0, str(_alpha_dir))

    from clean.alpha_engine import AlphaEngine
    from clean.ic_analyzer import calc_ic_series, calc_ic_summary
    from clean.ga_search import _compute_detailed_metrics

    instruments = instruments or BACKTEST.instruments
    data_months = data_months or BACKTEST.data_months
    load_months = load_data_months or BACKTEST.load_data_months

    logger.info(f"回测因子: {expression[:80]}")

    # 加载数据
    df = _load_data(instruments, load_data_months=load_months)
    engine = AlphaEngine(df)

    # 计算因子值
    try:
        factor = engine.calculate(expression)
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        return {"expression": expression, "error": str(e)}

    # 计算下期收益
    returns = _compute_returns(df)

    # IC评估日期范围
    start_date, end_date = _compute_ic_dates(df, data_months)

    # 计算 IC/ICIR 等指标（复用 clean 模块）
    detailed = _compute_detailed_metrics(factor, returns, start_date=start_date, end_date=end_date)

    result = {"expression": expression}
    result.update(detailed)

    # 字段名映射（统一命名）
    result["ic_mean"] = result.get("ic", 0)
    result["ic_win_rate"] = result.get("ic_positive_ratio", 0)

    # 分组收益
    group_returns = {}
    if compute_groups:
        group_returns = compute_group_returns(
            factor, returns,
            n_groups=GROUP_BT.n_groups,
            start_date=start_date, end_date=end_date
        )
        result["long_short_return"] = group_returns.get("long_short", 0)
        result["top_group_excess"] = group_returns.get("top_excess", 0)

    result["group_returns"] = group_returns
    result["test_start_date"] = start_date
    result["test_end_date"] = end_date
    result["asset_universe"] = instruments

    logger.info(
        f"  IC均值={result['ic_mean']:+.4f} | ICIR={result.get('icir', 0):+.4f} | "
        f"RankICIR={result.get('rank_icir', 0):+.4f} | "
        f"多空收益={result.get('long_short_return', 0):+.4f} | "
        f"第一组超额={result.get('top_group_excess', 0):+.4f}"
    )

    return result


def run_backtest_and_ingest(expression: str, description: str = "",
                            tags: str = "", auto_tag: bool = True,
                            **backtest_kwargs) -> Dict[str, Any]:
    """
    执行回测并判断是否自动入库

    Parameters
    ----------
    expression : str
        因子表达式
    description : str
        因子描述
    tags : str
        额外标签
    auto_tag : bool
        是否添加"自动入库"标签
    **backtest_kwargs
        传递给 run_backtest 的参数

    Returns
    -------
    dict
        回测结果 + 入库状态
    """
    result = run_backtest(expression, **backtest_kwargs)

    if "error" in result:
        result["ingested"] = False
        result["ingest_reason"] = f"回测失败: {result['error']}"
        return result

    # 判断是否入库
    if should_auto_ingest(result):
        all_tags = tags
        if auto_tag:
            all_tags = (all_tags + ",自动入库").strip(",")

        group_returns_str = json.dumps(result.get("group_returns", {}), ensure_ascii=False)

        factor_id = add_factor(
            expression=expression,
            metrics=result,
            description=description,
            tags=all_tags,
            asset_universe=result.get("asset_universe", "csi300"),
            test_start_date=result.get("test_start_date", ""),
            test_end_date=result.get("test_end_date", ""),
            group_returns=group_returns_str,
        )
        result["ingested"] = True
        result["ingest_reason"] = f"满足入库阈值，factor_id={factor_id}"
    else:
        result["ingested"] = False
        result["ingest_reason"] = (
            f"未满足入库阈值: |ICIR|={abs(result.get('icir', 0)):.4f}<={THRESHOLD.icir} 或 "
            f"|IC均值|={abs(result.get('ic_mean', 0)):.4f}<={THRESHOLD.ic_mean} 或 "
            f"IC胜率={result.get('ic_win_rate', 0):.4f}<={THRESHOLD.ic_win_rate}"
        )

    return result


def run_batch_backtest(expressions: list, **backtest_kwargs) -> pd.DataFrame:
    """
    批量回测

    Parameters
    ----------
    expressions : list[str]
        因子表达式列表
    **backtest_kwargs
        传递给 run_backtest 的参数

    Returns
    -------
    pd.DataFrame
        回测结果汇总
    """
    results = []
    for i, expr in enumerate(expressions):
        logger.info(f"[{i+1}/{len(expressions)}] 回测: {expr[:60]}")
        try:
            r = run_backtest(expr, **backtest_kwargs)
            results.append(r)
        except Exception as e:
            logger.error(f"  回测异常: {e}")
            results.append({"expression": expr, "error": str(e)})

    return pd.DataFrame(results)


def run_batch_backtest_and_ingest(expressions: list, **backtest_kwargs) -> pd.DataFrame:
    """批量回测并自动入库"""
    results = []
    for i, expr in enumerate(expressions):
        logger.info(f"[{i+1}/{len(expressions)}] 回测+入库: {expr[:60]}")
        try:
            r = run_backtest_and_ingest(expr, **backtest_kwargs)
            results.append(r)
        except Exception as e:
            logger.error(f"  回测异常: {e}")
            results.append({"expression": expr, "error": str(e), "ingested": False})

    return pd.DataFrame(results)
