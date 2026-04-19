"""
XGBoost模型训练与对比
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import XGBOOST, BACKTEST
from .data_manager import (
    init_qlib, load_ohlcv, load_alpha158_data,
    get_common_stocks, filter_by_stocks, split_train_test,
    clean_features, apply_cszscorenorm, winsorize_label,
)

logger = logging.getLogger(__name__)


def train_xgboost(X_train, y_train, X_test, X_valid=None, y_valid=None, config=None):
    """训练XGBoost模型"""
    import xgboost as xgb
    cfg = config or XGBOOST

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    watchlist = [(dtrain, "train")]

    if X_valid is not None and y_valid is not None:
        dvalid = xgb.DMatrix(X_valid.values, label=y_valid.values)
        watchlist.append((dvalid, "valid"))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "nthread": cfg.n_jobs if cfg.n_jobs > 0 else -1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    dtest = xgb.DMatrix(X_test.values)
    return model.predict(dtest)


def evaluate_predictions(pred, y_true, index) -> Dict[str, float]:
    """评估预测结果"""
    result_df = pd.DataFrame({"pred": pred, "label": y_true.values}, index=index)

    ic_list, rank_ic_list = [], []
    for date in result_df.index.get_level_values("datetime").unique():
        try:
            day = result_df.loc[date]
        except KeyError:
            continue
        if len(day) < 10:
            continue
        try:
            ic, _ = pearsonr(day["pred"], day["label"])
            rank_ic, _ = spearmanr(day["pred"], day["label"])
            ic_list.append(ic)
            rank_ic_list.append(rank_ic)
        except Exception:
            continue

    if not ic_list:
        return {}

    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list)

    return {
        "ic_mean": ic_arr.mean(),
        "ic_std": ic_arr.std(),
        "icir": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        "ic_positive_ratio": (ic_arr > 0).mean(),
        "rank_ic_mean": rank_ic_arr.mean(),
        "rank_ic_std": rank_ic_arr.std(),
        "rank_icir": rank_ic_arr.mean() / rank_ic_arr.std() if rank_ic_arr.std() > 0 else 0,
        "n_test_periods": len(ic_arr),
    }


def run_comparison(top50_pkl_path: str) -> pd.DataFrame:
    """
    运行Top50 vs Alpha158公平对比

    1. Label: 统一使用Alpha158的label (T+1→T+2收益率), 不做标准化
    2. Winsorize: clip(-0.2, 0.2) 去除极端值
    3. 样本: 共同有效索引强制对齐
    4. Top50: 只使用IC评估中RankICIR最高的50个因子
    """
    init_qlib()

    top50_raw = pd.read_pickle(top50_pkl_path)
    top50_raw.index = top50_raw.index.rename(["datetime", "instrument"])

    # 读取IC评估结果，筛选Top50因子
    from .config import OUTPUT_DIR
    ic_csv = OUTPUT_DIR / "top50_by_rank_icir.csv"
    if ic_csv.exists():
        top50_names = pd.read_csv(ic_csv)["name"].head(50).tolist()
        available = [c for c in top50_names if c in top50_raw.columns]
        logger.info(f"Top50因子: 需{len(top50_names)}个, 可用{len(available)}个")
        keep_cols = available + ["LABEL0"]
        top50_raw = top50_raw[keep_cols]
    else:
        logger.warning(f"未找到{ic_csv}, 使用全部{top50_raw.shape[1]-1}个因子")

    a158_raw = load_alpha158_data()

    common = get_common_stocks(top50_raw, a158_raw)
    top50_filtered = filter_by_stocks(top50_raw, common)
    a158_filtered = filter_by_stocks(a158_raw, common)

    top50_dt = set(top50_filtered.index.get_level_values("datetime").unique())
    a158_dt = set(a158_filtered.index.get_level_values("datetime").unique())
    common_dates = sorted(top50_dt & a158_dt)

    top50_filtered = top50_filtered.loc[top50_filtered.index.get_level_values("datetime").isin(common_dates)]
    a158_filtered = a158_filtered.loc[a158_filtered.index.get_level_values("datetime").isin(common_dates)]

    logger.info(f"共同股票池: {len(common)}只, 共同交易日: {len(common_dates)}天")
    logger.info(f"Top50过滤后: {top50_filtered.shape[0]}条")
    logger.info(f"Alpha158过滤后: {a158_filtered.shape[0]}条")

    # 统一label: Alpha158的label, winsorize去极端值, 不做标准化
    a158_label = a158_filtered["LABEL0"].copy()
    a158_label = winsorize_label(a158_label)
    logger.info(f"统一Label统计: 均值={a158_label.mean():.6f}, 标准差={a158_label.std():.6f}, "
                f"最小={a158_label.min():.6f}, 最大={a158_label.max():.6f}, NaN={a158_label.isna().sum()}")

    # ---- 三段分割: train / valid / test ----
    X1_raw = top50_filtered.drop(columns=["LABEL0"])
    X2_raw = a158_filtered.drop(columns=["LABEL0"])

    train_end = BACKTEST.train_end
    valid_start = BACKTEST.train_end
    valid_end = BACKTEST.test_start
    test_start = BACKTEST.test_start

    dt1 = X1_raw.index.get_level_values("datetime")
    dt2 = X2_raw.index.get_level_values("datetime")

    X1_train_raw = X1_raw[dt1 < train_end]
    X1_valid_raw = X1_raw[(dt1 >= valid_start) & (dt1 < valid_end)]
    X1_test_raw = X1_raw[dt1 >= test_start]

    X2_train_raw = X2_raw[dt2 < train_end]
    X2_valid_raw = X2_raw[(dt2 >= valid_start) & (dt2 < valid_end)]
    X2_test_raw = X2_raw[dt2 >= test_start]

    # Top50特征做CSZScoreNorm
    X1_train_raw = apply_cszscorenorm(X1_train_raw)
    X1_valid_raw = apply_cszscorenorm(X1_valid_raw)
    X1_test_raw = apply_cszscorenorm(X1_test_raw)

    # label分割
    dt_y = a158_label.index.get_level_values("datetime")
    y_train_raw = a158_label[dt_y < train_end]
    y_valid_raw = a158_label[(dt_y >= valid_start) & (dt_y < valid_end)]
    y_test_raw = a158_label[dt_y >= test_start]

    # ---- 找共同有效索引 ----
    def _valid_idx(X, y):
        Xc = X.replace([np.inf, -np.inf], np.nan)
        x_ok = (Xc.isna().mean(axis=1) <= 0.5)
        y_ok = ~y.isna()
        return x_ok.index.intersection(y_ok[y_ok].index)

    train_idx = _valid_idx(X1_train_raw, y_train_raw).intersection(
        _valid_idx(X2_train_raw, y_train_raw))
    valid_idx = _valid_idx(X1_valid_raw, y_valid_raw).intersection(
        _valid_idx(X2_valid_raw, y_valid_raw))
    test_idx = _valid_idx(X1_test_raw, y_test_raw).intersection(
        _valid_idx(X2_test_raw, y_test_raw))

    logger.info(f"对齐后: 训练={len(train_idx)}, 验证={len(valid_idx)}, 测试={len(test_idx)}")

    # ---- 提取对齐数据, label不做标准化 ----
    X1_train = X1_train_raw.loc[train_idx].fillna(0)
    X1_valid = X1_valid_raw.loc[valid_idx].fillna(0)
    X1_test = X1_test_raw.loc[test_idx].fillna(0)

    X2_train = X2_train_raw.loc[train_idx].fillna(0)
    X2_valid = X2_valid_raw.loc[valid_idx].fillna(0)
    X2_test = X2_test_raw.loc[test_idx].fillna(0)

    y_train = y_train_raw.loc[train_idx]
    y_valid = y_valid_raw.loc[valid_idx]
    y_test = y_test_raw.loc[test_idx]

    # ---- 模型1: Top50因子 ----
    logger.info("=" * 60)
    logger.info("模型1: Top50因子")
    pred1 = train_xgboost(X1_train, y_train, X1_test, X1_valid, y_valid)
    m1 = evaluate_predictions(pred1, y_test, X1_test.index)
    m1["model"] = "Top50"
    m1["n_features"] = X1_train.shape[1]
    m1["train_samples"] = len(X1_train)
    m1["test_samples"] = len(X1_test)

    # ---- 模型2: Alpha158因子 ----
    logger.info("=" * 60)
    logger.info("模型2: Alpha158")
    pred2 = train_xgboost(X2_train, y_train, X2_test, X2_valid, y_valid)
    m2 = evaluate_predictions(pred2, y_test, X2_test.index)
    m2["model"] = "Alpha158"
    m2["n_features"] = X2_train.shape[1]
    m2["train_samples"] = len(X2_train)
    m2["test_samples"] = len(X2_test)

    results = pd.DataFrame([m1, m2])
    return results
