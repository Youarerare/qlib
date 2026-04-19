"""
单因子搜索与自动入库脚本

基于 clean 模块的搜索逻辑，增加自动入库功能。
回测完成后自动判断是否入库，满足阈值的因子自动保存到因子库。

用法:
    # 从 qlib 项目根目录执行:
    python -m factor_library.search_one

    # 或直接运行:
    python factor_library/search_one.py
"""
import sys
import logging
from pathlib import Path

# 确保 clean 包和项目根目录可导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
from clean.config import GA, OUTPUT_DIR
from clean.data_manager import init_qlib, load_ohlcv
from clean.alpha_engine import AlphaEngine
from clean.ic_analyzer import evaluate_factor, calc_ic_series, calc_ic_summary
from clean.ga_search import GAFactorSearcher, _compute_detailed_metrics

from factor_library.database import add_factor, should_auto_ingest, exists
from factor_library.config import THRESHOLD, BACKTEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("search_one_lib")


def evaluate_single_formula(formula: str, data_months: int = None,
                            load_data_months: int = None) -> dict:
    """
    评估单个公式，返回 IC/ICIR 等指标

    Parameters
    ----------
    formula : str
        因子公式
    data_months : int
        IC/IR 计算只用最近 N 个月的数据，0 表示使用全部日期
    load_data_months : int
        加载数据范围（月），默认12个月
    """
    data_months = data_months or BACKTEST.data_months
    load_data_months = load_data_months or BACKTEST.load_data_months

    logger.info(f"=" * 60)
    logger.info(f"评估公式: {formula}")

    init_qlib()

    from qlib.data import D
    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()
    ic_end_date = max_date
    ic_start_date = max_date - pd.DateOffset(months=data_months)

    load_start = (ic_start_date - pd.DateOffset(months=load_data_months)).strftime("%Y-%m-%d")
    load_end = ic_end_date.strftime("%Y-%m-%d")

    logger.info(f"  数据加载范围: {load_start} ~ {load_end}")
    logger.info(f"  (IC评估范围: {ic_start_date.strftime('%Y-%m-%d')} ~ {ic_end_date.strftime('%Y-%m-%d')})")
    df = load_ohlcv(start_time=load_start, end_time=load_end)
    engine = AlphaEngine(df)

    try:
        factor = engine.calculate(formula)
    except Exception as e:
        logger.error(f"公式计算失败: {e}")
        return {"formula": formula, "error": str(e)}

    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    if data_months > 0:
        dates = returns.index.get_level_values("datetime")
        max_date = dates.max()
        start_date = (max_date - pd.DateOffset(months=data_months)).strftime("%Y-%m-%d")
        end_date = max_date.strftime("%Y-%m-%d")
        logger.info(f"  IC/IR 评估日期范围: {start_date} ~ {end_date}")
    else:
        start_date, end_date = None, None
        logger.info("  IC/IR 评估日期范围: 全部数据")

    detailed = _compute_detailed_metrics(factor, returns, start_date=start_date, end_date=end_date)
    result = {"formula": formula}
    result.update(detailed)

    # 映射字段名
    result["ic_mean"] = result.get("ic", 0)
    result["ic_win_rate"] = result.get("ic_positive_ratio", 0)

    logger.info(f"  IC={detailed['ic']:+.4f} | IR={detailed['ir']:+.4f} | "
                f"ICIR={detailed['icir']:+.4f} | RankIC={detailed['rank_ic']:+.4f} | "
                f"RankICIR={detailed['rank_icir']:+.4f}")

    return result


def evaluate_and_ingest(formula: str, description: str = "", tags: str = "",
                        data_months: int = None, load_data_months: int = None) -> dict:
    """
    评估因子并自动入库（如果满足阈值）

    Parameters
    ----------
    formula : str
        因子公式
    description : str
        因子描述
    tags : str
        额外标签
    data_months : int
        IC评估月数
    load_data_months : int
        数据加载月数

    Returns
    -------
    dict
        评估结果 + 入库状态
    """
    result = evaluate_single_formula(formula, data_months, load_data_months)

    if "error" in result:
        result["ingested"] = False
        result["ingest_reason"] = f"评估失败: {result['error']}"
        return result

    # 判断是否入库
    if should_auto_ingest(result):
        all_tags = (tags + ",自动入库").strip(",")
        factor_id = add_factor(
            expression=formula,
            metrics=result,
            description=description,
            tags=all_tags,
            asset_universe=BACKTEST.instruments,
        )
        result["ingested"] = True
        result["factor_id"] = factor_id
        result["ingest_reason"] = f"满足入库阈值，已入库 {factor_id}"
        logger.info(f"  >> 自动入库: {factor_id}")
    else:
        result["ingested"] = False
        result["ingest_reason"] = "未满足入库阈值"
        logger.info(f"  >> 未入库: 不满足阈值")

    return result


def run_single_formula_search(
    formula: str = "add(close, open)",
    n_generations: int = 30,
    population_size: int = 100,
    output_file: str = "single_results.csv",
    data_months: int = None,
    load_data_months: int = None,
    auto_ingest: bool = True,
) -> list:
    """
    以指定公式为起点，执行遗传算法搜索，搜索结果中满足阈值的因子自动入库

    Parameters
    ----------
    formula : str
        起始公式
    n_generations : int
        进化代数
    population_size : int
        种群大小
    output_file : str
        结果保存文件名
    data_months : int
        IC/IR 评估月数
    load_data_months : int
        数据加载月数
    auto_ingest : bool
        是否自动入库满足阈值的因子
    """
    data_months = data_months or BACKTEST.data_months
    load_data_months = load_data_months or BACKTEST.load_data_months

    logger.info(f"=" * 60)
    logger.info(f"单公式遗传搜索: {formula}")
    logger.info(f"  种群={population_size}, 代数={n_generations}, 自动入库={auto_ingest}")

    # 1. 评估原始公式
    eval_result = evaluate_single_formula(formula, data_months, load_data_months)

    # 2. 初始化 GA 搜索
    init_qlib()
    from qlib.data import D
    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()
    load_start = (max_date - pd.DateOffset(months=load_data_months)).strftime("%Y-%m-%d")
    load_end = max_date.strftime("%Y-%m-%d")

    df = load_ohlcv(start_time=load_start, end_time=load_end)
    engine = AlphaEngine(df)
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    from clean import config as _cfg
    _cfg.GA.n_generations = n_generations
    _cfg.GA.population_size = population_size

    searcher = GAFactorSearcher(engine, returns)
    results = searcher.search()

    # 3. 补充原始公式评估结果
    if "error" not in eval_result:
        results.append({
            "expression": formula,
            "ic": eval_result.get("ic", 0),
            "ir": eval_result.get("ir", 0),
            "icir": eval_result.get("icir", 0),
            "rank_ic": eval_result.get("rank_ic", 0),
            "rank_icir": eval_result.get("rank_icir", 0),
            "fitness": eval_result.get("ic", 0) * 2 + eval_result.get("icir", 0) * 1,
        })
        results.sort(key=lambda x: x["fitness"], reverse=True)

    # 4. 自动入库
    if auto_ingest:
        ingested_count = 0
        for item in results:
            expr = item.get("expression", "")
            if not expr:
                continue
            metrics = {
                "ic_mean": item.get("ic", 0),
                "icir": item.get("icir", 0),
                "rank_ic": item.get("rank_ic", 0),
                "rank_icir": item.get("rank_icir", 0),
                "ic_win_rate": item.get("ic_positive_ratio", 0),
            }
            if should_auto_ingest(metrics):
                fid = add_factor(
                    expression=expr,
                    metrics=metrics,
                    tags="自动入库,GA搜索",
                    asset_universe=BACKTEST.instruments,
                )
                ingested_count += 1
                logger.info(f"  自动入库 #{ingested_count}: {fid} | {expr[:50]}")
        logger.info(f"自动入库完成: {ingested_count}/{len(results)} 个因子")

    # 5. 保存结果
    output_path = OUTPUT_DIR / output_file
    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info(f"结果已保存: {output_path}")

    # 打印 Top5
    logger.info("Top 5 因子:")
    for i, item in enumerate(results[:5]):
        expr = item.get("expression", "?")
        logger.info(
            f"  #{i + 1}: fitness={item['fitness']:.4f}, "
            f"IC={item.get('ic', 0):.4f}, IR={item.get('ir', 0):.4f}, "
            f"ICIR={item.get('icir', 0):.4f}, RankICIR={item.get('rank_icir', 0):.4f}, "
            f"expr={expr[:60]}"
        )

    return results


if __name__ == "__main__":
    try:
        # 评估原始因子并自动入库
        logger.info("=" * 60)
        logger.info("步骤1: 评估原始因子")
        original = "ts_arg_min(sqrt(max(cs_mean(ts_av_diff(adv5, 2)), abs(adv150))), 5)"
        eval_orig = evaluate_and_ingest(
            original,
            description="原始因子: ts_arg_min + sqrt + max + cs_mean",
            tags="原始因子,成交量类",
        )
        logger.info(f"原始因子评估结果: ICIR={eval_orig.get('icir', 0):.4f}, "
                     f"IC均值={eval_orig.get('ic_mean', 0):.4f}")

        # 评估改进因子
        logger.info("\n" + "=" * 60)
        logger.info("步骤2: 评估改进因子")
        improved_factors = {
            "A": "divide(ts_av_diff(adv5, 2), cs_mean(ts_av_diff(adv5, 2)))",
            "B": "rank(ts_av_diff(adv5, 2))",
            "C": "subtract(ts_av_diff(adv5, 2), cs_mean(ts_av_diff(adv5, 2)))",
        }
        improved_results = {}
        for name, expr in improved_factors.items():
            logger.info(f"\n--- 改进因子 {name}: {expr}")
            r = evaluate_and_ingest(
                expr,
                description=f"改进因子{name}: {expr}",
                tags=f"改进因子{name},成交量类",
            )
            improved_results[name] = r

        # 对比输出
        logger.info("\n" + "=" * 60)
        logger.info("对比结果:")
        logger.info(f"  原始因子: ICIR={eval_orig.get('icir', 0):+.4f}, IC={eval_orig.get('ic_mean', 0):+.4f}")
        for name, r in improved_results.items():
            logger.info(f"  改进{name}: ICIR={r.get('icir', 0):+.4f}, IC={r.get('ic_mean', 0):+.4f}, "
                         f"入库={'是' if r.get('ingested') else '否'}")

        # GA搜索
        logger.info("\n" + "=" * 60)
        logger.info("步骤3: GA搜索（可选）")
        results = run_single_formula_search(
            formula= original,
            n_generations=10,
            output_file="single_results.csv",
            auto_ingest=True,
        )
        logger.info(f"\n完成! 共找到 {len(results)} 个有效因子")
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)
        input("按回车键退出...")
