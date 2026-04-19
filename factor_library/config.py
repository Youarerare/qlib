"""
因子库全局配置
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ===== 路径 =====
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "factor_library.db"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

# 引用已有的 clean 模块路径
CLEAN_DIR = Path(__file__).resolve().parent.parent / "examples" / "alpha_factor_test" / "clean"

# ===== 入库阈值 =====


@dataclass
class IngestThreshold:
    """因子自动入库阈值"""
    icir: float = 0.5           # ICIR > 此值才考虑入库
    ic_mean: float = 0.02       # |IC均值| > 此值才考虑入库
    ic_win_rate: float = 0.5    # IC胜率 > 此值才考虑入库
    # 改进因子对比阈值（需同时满足）
    improved_icir: float = 0.5
    improved_ic_mean: float = 0.03


# ===== 回测参数 =====


@dataclass
class BacktestParams:
    """回测参数配置"""
    instruments: str = "csi300"         # 股票池
    start_time: str = "2008-01-01"     # 回测开始日期
    end_time: str = "2026-04-13"       # 回测结束日期
    data_months: int = 3               # IC评估用最近N个月数据
    load_data_months: int = 12         # 数据加载范围（月）
    forecast_period: int = 1           # 预测周期


# ===== 分组回测参数 =====


@dataclass
class GroupBacktestParams:
    """分组回测参数"""
    n_groups: int = 5                  # 分组数
    long_group: int = 1                # 做多组（1=最高组）
    short_group: int = 5               # 做空组（5=最低组）


# ===== QLib 数据路径 =====
QLIB_DATA_URI = "~/.qlib/qlib_data/cn_data"

# ===== 实例化默认配置 =====
THRESHOLD = IngestThreshold()
BACKTEST = BacktestParams()
GROUP_BT = GroupBacktestParams()

# ===== 改进因子定义 =====
ORIGINAL_FACTOR = "ts_arg_min(sqrt(max(cs_mean(ts_av_diff(adv5, 2)), abs(adv150))), 5)"

IMPROVED_FACTORS = {
    "A": {
        "expression": "divide(ts_av_diff(adv5, 2), cs_mean(ts_av_diff(adv5, 2)))",
        "description": "个股成交量偏离度 / 市场平均偏离度",
        "tags": "改进因子,成交量类,偏离度",
    },
    "B": {
        "expression": "rank(ts_av_diff(adv5, 2))",
        "description": "个股成交量偏离度在市场的排名",
        "tags": "改进因子,成交量类,排名",
    },
    "C": {
        "expression": "subtract(ts_av_diff(adv5, 2), cs_mean(ts_av_diff(adv5, 2)))",
        "description": "个股成交量减去市场平均成交量",
        "tags": "改进因子,成交量类,差值",
    },
}
