"""
一键模型训练辅助模块

提供从因子库筛选因子到完成模型训练的完整流程，
支持命令行调用和编程接口。

用法:
    # 使用默认配置训练（从因子库选择ICIR最高的50个因子）
    python -m factor_library.model_trainer

    # 指定参数
    python -m factor_library.model_trainer --top-k 30 --icir-min 0.3 --instruments csi300
"""
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from factor_library.database import get_factors_for_training, get_all_factors
from factor_library.config import BACKTEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model_trainer")


def train_from_factor_library(
    top_k: int = 50,
    icir_min: float = 0.0,
    instruments: str = "csi300",
    n_estimators: int = 1000,
    max_depth: int = 8,
    learning_rate: float = 0.0421,
    subsample: float = 0.8789,
    colsample_bytree: float = 0.8879,
) -> Dict:
    """
    从因子库选取因子，训练 XGBoost 模型

    Parameters
    ----------
    top_k : int
        选取前K个因子
    icir_min : float
        ICIR 最小值筛选
    instruments : str
        股票池
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree :
        XGBoost 超参数

    Returns
    -------
    dict
        训练结果指标
    """
    # 1. 获取因子列表
    filter_by = {}
    if icir_min > 0:
        filter_by["icir_min"] = icir_min

    expressions = get_factors_for_training(filter_by=filter_by, limit=top_k)

    if not expressions:
        logger.error("因子库中无满足条件的因子!")
        return {}

    logger.info(f"从因子库选择 {len(expressions)} 个因子用于训练")

    # 2. 加载数据
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine, compute_factors
    from clean.model_trainer import train_xgboost, evaluate_predictions
    from clean.data_manager import winsorize_label
    from clean.config import BACKTEST as CLEAN_BACKTEST

    init_qlib()
    df = load_ohlcv(instruments=instruments,
                    start_time=CLEAN_BACKTEST.start_time,
                    end_time=CLEAN_BACKTEST.end_time)

    # 3. 计算因子
    formulas_dict = {f"f_{i}": expr for i, expr in enumerate(expressions)}
    feature_df = compute_factors(df, formulas_dict)

    # 4. 准备标签
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    feature_df["LABEL0"] = winsorize_label(returns)

    label = feature_df["LABEL0"]
    features = feature_df.drop(columns=["LABEL0"])

    # 5. 分割数据
    dt = features.index.get_level_values("datetime")
    train_mask = dt < CLEAN_BACKTEST.train_end
    test_mask = dt >= CLEAN_BACKTEST.test_start

    X_train = features[train_mask].fillna(0).replace([np.inf, -np.inf], 0)
    X_test = features[test_mask].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = label[train_mask]
    y_test = label[test_mask]

    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 6. 训练
    from clean.config import XGBoostConfig
    config = XGBoostConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )

    pred = train_xgboost(X_train, y_train, X_test, config=config)

    # 7. 评估
    metrics = evaluate_predictions(pred, y_test, X_test.index)
    metrics["n_features"] = len(expressions)
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)

    logger.info("=" * 60)
    logger.info("训练结果:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键模型训练")
    parser.add_argument("--top-k", type=int, default=50, help="选取前K个因子")
    parser.add_argument("--icir-min", type=float, default=0.0, help="ICIR最小值筛选")
    parser.add_argument("--instruments", type=str, default="csi300", help="股票池")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0421)
    parser.add_argument("--subsample", type=float, default=0.8789)
    parser.add_argument("--colsample", type=float, default=0.8879)

    args = parser.parse_args()

    try:
        metrics = train_from_factor_library(
            top_k=args.top_k,
            icir_min=args.icir_min,
            instruments=args.instruments,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=args.subsample,
            colsample_bytree=args.colsample,
        )

        if metrics:
            print("\n" + "=" * 60)
            print("训练完成! 关键指标:")
            print(f"  IC均值: {metrics.get('ic_mean', 0):+.4f}")
            print(f"  ICIR: {metrics.get('icir', 0):+.4f}")
            print(f"  Rank ICIR: {metrics.get('rank_icir', 0):+.4f}")
            print(f"  IC胜率: {metrics.get('ic_positive_ratio', 0):.2%}")
            print(f"  因子数: {metrics.get('n_features', 0)}")
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        input("按回车键退出...")
