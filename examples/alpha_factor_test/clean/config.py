"""
全局配置 - 路径、参数、常量
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

BASE_DIR = Path(__file__).parent
ALPHA_FACTOR_DIR = BASE_DIR.parent

QLIB_DATA_URI = "~/.qlib/qlib_data/cn_data"

ALPHA101_FORMULA_PATH = r"C:\Users\syk\Desktop\git_repo\auto_alpha\research_formula_candidates.txt"
ALPHA191_FORMULA_PATH = r"C:\Users\syk\Desktop\git_repo\auto_alpha\alpha191.txt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class BacktestConfig:
    instruments: str = "csi300"
    start_time: str = "2008-01-01"
    end_time: str = "2026-04-13"
    train_end: str = "2022-01-01"
    test_start: str = "2022-06-01"
    forecast_period: int = 1


@dataclass
class XGBoostConfig:
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate: float = 0.0421
    subsample: float = 0.8789
    colsample_bytree: float = 0.8879
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class GAConfig:
    population_size: int = 200
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    max_tree_depth: int = 5
    n_jobs: int = 4
    ic_weight: float = 2.0      # 提高IC权重，更关注预测能力
    ir_weight: float = 1.0      # 降低ICIR权重，避免过度优化稳定性
    turnover_penalty: float = 0.1
    correlation_penalty: float = 0.3
    evolution_data_months: int = 3  # 进化评估只用最近N个月的数据（加速搜索）


BACKTEST = BacktestConfig()
XGBOOST = XGBoostConfig()
GA = GAConfig()

TS_OPERATORS = [
    "ts_sum", "ts_mean", "ts_std_dev", "ts_min", "ts_max",
    "ts_rank", "ts_delta", "ts_delay", "ts_corr", "ts_covariance",
    "ts_scale", "ts_decay_linear", "ts_arg_max", "ts_arg_min",
    "ts_product", "ts_regression", "ts_av_diff", "ts_zscore",
]

CS_OPERATORS = ["rank", "scale", "normalize", "zscore"]

MATH_OPERATORS = ["abs", "log", "sign", "sqrt", "signed_power"]

BINARY_OPERATORS = ["add", "subtract", "multiply", "divide", "max", "min"]

LOGIC_OPERATORS = ["if_else", "and", "or"]

GROUP_OPERATORS = ["group_neutralize", "group_rank", "group_zscore"]

DATA_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"]
ADV_FIELDS = [f"adv{d}" for d in [5, 10, 15, 20, 30, 40, 50, 60, 80, 120, 150, 180]]

ALL_OPERATORS = TS_OPERATORS + CS_OPERATORS + MATH_OPERATORS + BINARY_OPERATORS + LOGIC_OPERATORS + GROUP_OPERATORS
