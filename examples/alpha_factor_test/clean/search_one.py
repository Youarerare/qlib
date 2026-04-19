"""
单公式评估 & 遗传算法搜索

用法:
    # 直接运行（从 alpha_factor_test 目录执行）:
    python -m clean.search_one

    # 或从项目根目录执行:
    cd alpha_factor_test
    python -m clean.search_one
"""
import sys
import logging
from pathlib import Path

# 确保可以 import clean 包
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from clean.config import GA, OUTPUT_DIR
from clean.data_manager import init_qlib, load_ohlcv
from clean.alpha_engine import AlphaEngine
from clean.ic_analyzer import evaluate_factor, calc_ic_series, calc_ic_summary
from clean.ga_search import GAFactorSearcher, _compute_detailed_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("search_one")


def evaluate_single_formula(formula: str, data_months: int = 3, load_data_months: int = 12) -> dict:
    """评估单个公式，返回 IC/ICIR 等指标

    Parameters
    ----------
    formula : str
        因子公式
    data_months : int
        IC/IR 计算只用最近 N 个月的数据，0 表示使用全部日期
    load_data_months : int
        加载数据范围（月），默认12个月（1年），提供足够的历史窗口给时序算子
    """
    logger.info(f"=" * 60)
    logger.info(f"评估公式: {formula}")

    init_qlib()

    # 先获取最新日期，计算 IC 评估的起止日期
    from qlib.data import D
    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()
    ic_end_date = max_date
    ic_start_date = max_date - pd.DateOffset(months=data_months)

    # 数据加载从 IC起始日期 的前一年开始（给时序算子留足历史窗口）
    load_start = (ic_start_date - pd.DateOffset(months=load_data_months)).strftime("%Y-%m-%d")
    load_end = ic_end_date.strftime("%Y-%m-%d")

    logger.info(f"  数据加载范围: {load_start} ~ {load_end}")
    logger.info(f"  (IC评估范围: {(ic_start_date).strftime('%Y-%m-%d')} ~ {ic_end_date.strftime('%Y-%m-%d')}, {data_months}个月)")
    df = load_ohlcv(start_time=load_start, end_time=load_end)
    engine = AlphaEngine(df)

    # 计算因子
    try:
        factor = engine.calculate(formula)
    except Exception as e:
        logger.error(f"公式计算失败: {e}")
        return {"formula": formula, "error": str(e)}

    # 计算 Label（与 pipeline 一致）
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    # 计算评估日期范围（只用最近 N 个月）
    if data_months > 0:
        dates = returns.index.get_level_values("datetime")
        max_date = dates.max()
        start_date = (max_date - pd.DateOffset(months=data_months)).strftime("%Y-%m-%d")
        end_date = max_date.strftime("%Y-%m-%d")
        logger.info(f"  IC/IR 评估日期范围: {start_date} ~ {end_date} ({data_months}个月)")
    else:
        start_date, end_date = None, None
        logger.info("  IC/IR 评估日期范围: 全部数据")

    # 详细评估（传入日期范围）
    detailed = _compute_detailed_metrics(factor, returns, start_date=start_date, end_date=end_date)
    result = {"formula": formula}
    result.update(detailed)

    logger.info(f"  IC={detailed['ic']:+.4f} | IR={detailed['ir']:+.4f} | "
                f"ICIR={detailed['icir']:+.4f} | RankIC={detailed['rank_ic']:+.4f} | "
                f"RankICIR={detailed['rank_icir']:+.4f}")

    return result


def run_single_formula_search(
    formula: str = "add(close, open)",
    n_generations: int = 30,
    population_size: int = 100,
    output_file: str = "single_results.csv",
    data_months: int = 3,
    load_data_months: int = 12,
) -> list:
    """
    以指定公式为起点，执行遗传算法搜索

    Parameters
    ----------
    formula : str
        起始公式（也会参与评估）
    n_generations : int
        进化代数
    population_size : int
        种群大小
    output_file : str
        结果保存文件名
    data_months : int
        IC/IR 计算只用最近 N 个月的数据，0 表示使用全部日期
    load_data_months : int
        加载数据范围（月），默认12个月（1年），提供足够的历史窗口给时序算子

    Returns
    -------
    list[dict]
        Top 因子列表
    """
    logger.info(f"=" * 60)
    logger.info(f"单公式遗传搜索: {formula}")
    logger.info(f"  种群={population_size}, 代数={n_generations}, IC评估周期={data_months}个月, 数据加载={load_data_months}个月")

    # 1. 先评估原始公式
    eval_result = evaluate_single_formula(formula, data_months=data_months, load_data_months=load_data_months)

    # 2. 初始化 GA 搜索
    init_qlib()

    # GA 搜索也用同样的数据加载范围
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

    # 覆盖默认代数
    from clean import config as _cfg
    _cfg.GA.n_generations = n_generations
    _cfg.GA.population_size = population_size

    searcher = GAFactorSearcher(engine, returns)

    # 将原始公式作为种子注入GA搜索，确保进化从该公式出发
    logger.info(f"  将原始公式作为种子注入GA种群...")
    results = searcher.search(seed_expressions=[formula])

    # 3. 把原始公式评估结果也加入
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

    # 4. 保存结果
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
    import traceback
    
    print("=" * 60)
    print("单公式遗传算法搜索")
    print("=" * 60)
    
    try:
        results = run_single_formula_search(
            formula="(-1 * rank(ts_corr(rank(high), rank(ts_mean(volume, 15)), 9)))",
            n_generations=30,
            output_file="single_results.csv",
        )
        logger.info(f"\n完成! 共找到 {len(results)} 个有效因子")
    except Exception as e:
        logger.error(f"\n运行出错: {e}")
        logger.error("\n详细错误信息:")
        traceback.print_exc()
    finally:
        # 无论成功还是失败，都等待用户确认
        print("\n" + "=" * 60)
        input("按回车键退出...")  # 防止闪退，方便看错误信息
