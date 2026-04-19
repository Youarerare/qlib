"""
数据管理 - qlib数据加载、股票池对齐、特征预处理
"""
import logging
from typing import Optional, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import QLIB_DATA_URI, BACKTEST

logger = logging.getLogger(__name__)

_qlib_initialized = False


def init_qlib():
    """初始化qlib（仅一次）"""
    global _qlib_initialized
    if not _qlib_initialized:
        import qlib
        qlib.init(provider_uri=QLIB_DATA_URI)
        _qlib_initialized = True
        logger.info("qlib初始化完成")


def get_stock_list(
    instruments: str = "csi300",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[str]:
    """获取指定时间范围内的股票池成分股列表"""
    from qlib.data import D

    init_qlib()
    start_time = start_time or BACKTEST.start_time
    end_time = end_time or BACKTEST.end_time

    instruments_config = D.instruments(instruments)
    stock_list = D.list_instruments(
        instruments_config,
        start_time=start_time,
        end_time=end_time,
        freq="day",
        as_list=True,
    )
    logger.info(f"{instruments}在{start_time}~{end_time}期间: {len(stock_list)}只成分股")
    return stock_list


def load_ohlcv(
    instruments: str = "csi300",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载OHLCV数据
    返回: DataFrame, MultiIndex(datetime, instrument), columns=[open,high,low,close,volume,factor]
    """
    from qlib.data import D

    init_qlib()
    start_time = start_time or BACKTEST.start_time
    end_time = end_time or BACKTEST.end_time

    stock_list = get_stock_list(instruments, start_time, end_time)

    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    df = D.features(stock_list, fields, start_time=start_time, end_time=end_time)

    df = df.rename(columns={
        "$open": "open", "$high": "high", "$low": "low",
        "$close": "close", "$volume": "volume", "$factor": "factor",
    })

    df.index = df.index.rename(["datetime", "instrument"])

    if df.index.get_level_values(0).dtype == object:
        df = df.swaplevel().sort_index()
        df.index = df.index.rename(["datetime", "instrument"])

    logger.info(f"加载OHLCV: {df.shape[0]}条, {df.index.get_level_values('instrument').nunique()}只股票")
    return df


def load_alpha158_data(
    instruments: str = "csi300",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """加载Alpha158特征数据 (DK_I特征 + DK_R原始label)"""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    init_qlib()
    start_time = start_time or BACKTEST.start_time
    end_time = end_time or BACKTEST.end_time

    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )

    # 特征用DK_I (infer, 无处理), label用DK_R (raw, 原始值)
    feat = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)

    # 特征清理: inf→NaN, NaN→0
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    data = feat.copy()
    data["LABEL0"] = label
    data.index = data.index.rename(["datetime", "instrument"])
    logger.info(f"加载Alpha158: {data.shape}")
    return data


def get_common_stocks(*dataframes: pd.DataFrame) -> Set[str]:
    """获取多个DataFrame的共同股票池"""
    stock_sets = []
    for df in dataframes:
        stocks = set(df.index.get_level_values("instrument").unique())
        stock_sets.append(stocks)

    common = stock_sets[0]
    for s in stock_sets[1:]:
        common = common & s

    logger.info(f"共同股票池: {len(common)}只")
    return common


def filter_by_stocks(df: pd.DataFrame, stocks: Set[str]) -> pd.DataFrame:
    """按股票池过滤DataFrame"""
    mask = df.index.get_level_values("instrument").isin(stocks)
    return df[mask]


def split_train_test(
    df: pd.DataFrame,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间划分训练/测试集"""
    train_end = train_end or BACKTEST.train_end
    test_start = test_start or BACKTEST.test_start

    dt = df.index.get_level_values("datetime")
    train = df[dt < train_end]
    test = df[dt >= test_start]
    return train, test


def clean_features(X: pd.DataFrame, y: pd.Series, max_nan_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    清理特征和标签: 去除inf，填充NaN，去除NaN过多的行
    max_nan_ratio: 允许的最大NaN比例，超过此比例的行被删除
    """
    X = X.replace([np.inf, -np.inf], np.nan)

    # 去除标签NaN
    valid_label = ~y.isna()
    X = X[valid_label]
    y = y[valid_label]

    # 去除NaN比例过高的行
    nan_ratio = X.isna().mean(axis=1)
    valid = nan_ratio <= max_nan_ratio
    X = X[valid]
    y = y.loc[X.index]

    # 剩余NaN用0填充
    X = X.fillna(0)

    return X, y


def apply_cszscorenorm(X: pd.DataFrame) -> pd.DataFrame:
    """应用CSZScoreNorm截面标准化"""
    from qlib.data.dataset.processor import CSZScoreNorm
    csz = CSZScoreNorm()
    X = csz(X)
    return X


def winsorize_label(y: pd.Series, lower: float = -0.2, upper: float = 0.2) -> pd.Series:
    """截断label极端值，与Alpha158的label范围对齐"""
    return y.clip(lower=lower, upper=upper)
