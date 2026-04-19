"""
批量因子搜索与自动入库脚本

两种模式:
  1. 批量回测模式（默认）: 逐个回测 all_formulas.txt 中的公式并自动入库
  2. GA 种子搜索模式: 用公式文件中的表达式作为 GA 初始种群，进化搜索更优因子

用法:
    # 批量回测（默认）
    python -m factor_library.batch_search
    python -m factor_library.batch_search --input my_formulas.txt

    # GA 种子搜索（用 all_formulas.txt 作为种子）
    python -m factor_library.batch_search --ga
    python -m factor_library.batch_search --ga --ga-per-seed    # 逐种子搜索（推荐）
    python -m factor_library.batch_search --ga --ga-generations 20 --ga-population 60

    # 先批量回测，再用优质因子做 GA 搜索
    python -m factor_library.batch_search --ga --ga-top-seed 10

    # 准备公式文件（首次运行前执行）
    python -m factor_library.prepare_formulas
"""
import sys
import logging
import argparse
from pathlib import Path

# 确保可导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from factor_library.database import add_factor, should_auto_ingest
from factor_library.backtest_engine import run_backtest_and_ingest
from factor_library.config import BACKTEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("batch_search")

# 本地公式文件
_ALL_FORMULAS_FILE = Path(__file__).resolve().parent / "all_formulas.txt"


def _strip_prefix(line: str) -> str:
    """去掉 alphaXXX:: 或 alphaXXX: 前缀"""
    import re
    m = re.match(r"alpha\d+\s*::\s*(.*)", line)
    if m:
        return m.group(1).strip()
    m = re.match(r"alpha\d+\s*:\s*(.*)", line)
    if m:
        return m.group(1).strip()
    return line.strip()


def load_formulas(filepath: str) -> list:
    """
    从文件加载因子表达式列表

    支持:
    - CSV 文件（需有 expression 列）
    - 文本文件（每行一个表达式，自动去掉 alphaXXX:: / alphaXXX: 前缀）
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
        if "expression" in df.columns:
            return df["expression"].dropna().tolist()
        elif "formula" in df.columns:
            return df["formula"].dropna().tolist()
        else:
            return df.iloc[:, 0].dropna().tolist()
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            formulas = [_strip_prefix(line) for line in f
                        if line.strip() and not line.strip().startswith("#")]
        formulas = [f for f in formulas if f]
        return formulas


# ==================== 批量回测模式 ====================

def batch_search(input_path: str, output_path: str = None,
                 auto_ingest: bool = True, tags: str = "批量搜索",
                 instruments: str = None) -> pd.DataFrame:
    """
    批量回测因子并自动入库
    """
    formulas = load_formulas(input_path)
    logger.info(f"加载 {len(formulas)} 个因子表达式")

    results = []
    for i, expr in enumerate(formulas):
        logger.info(f"\n[{i+1}/{len(formulas)}] 回测: {expr[:60]}")
        try:
            if auto_ingest:
                r = run_backtest_and_ingest(
                    expr,
                    tags=f"{tags},批量搜索",
                    instruments=instruments or BACKTEST.instruments,
                )
            else:
                from factor_library.backtest_engine import run_backtest
                r = run_backtest(expr, instruments=instruments)
                r["ingested"] = False
            results.append(r)
        except Exception as e:
            logger.error(f"  回测异常: {e}")
            results.append({"expression": expr, "error": str(e), "ingested": False})

    df = pd.DataFrame(results)

    if output_path is None:
        from factor_library.config import EXPORT_DIR
        from datetime import datetime
        output_path = str(EXPORT_DIR / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n批量搜索完成! 结果保存: {output_path}")

    ingested = df.get("ingested", pd.Series(dtype=bool))
    n_ingested = ingested.sum() if len(ingested) > 0 else 0
    logger.info(f"  总计: {len(formulas)} 个因子, 入库: {n_ingested} 个")

    return df


# ==================== GA 种子搜索模式 ====================

def ga_seed_search(input_path: str, n_generations: int = 20,
                   population_size: int = 60, auto_ingest: bool = True,
                   top_seed: int = 0, instruments: str = None,
                   output_path: str = None,
                   per_seed: bool = False,
                   per_seed_generations: int = 10,
                   per_seed_population: int = 30) -> pd.DataFrame:
    """
    用公式文件中的表达式作为 GA 初始种群种子，执行遗传算法搜索

    支持两种模式:
      1. 混合模式（默认, per_seed=False）: 所有种子混入1个大种群，跑1次GA
      2. 逐种子模式（per_seed=True）: 每条公式作为1个独立种子，各自起GA搜索

    Parameters
    ----------
    input_path : str
        公式文件路径
    n_generations : int
        GA 进化代数（混合模式用）
    population_size : int
        种群大小（混合模式用）
    auto_ingest : bool
        是否自动入库
    top_seed : int
        如果 > 0，先用批量回测评估所有公式，只取 Top N 作为种子
        如果 = 0，全部公式都作为种子
    instruments : str
        股票池
    output_path : str
        结果输出路径
    per_seed : bool
        是否启用逐种子独立GA搜索模式（每条公式独立进化）
    per_seed_generations : int
        逐种子模式下，每次GA的进化代数（默认10）
    per_seed_population : int
        逐种子模式下，每次GA的种群大小（默认30）

    Returns
    -------
    pd.DataFrame
        GA 搜索结果
    """
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine
    from clean.ga_search import GAFactorSearcher, _compute_detailed_metrics
    from clean.config import GA as GA_CONFIG
    from qlib.data import D

    formulas = load_formulas(input_path)
    logger.info(f"加载 {len(formulas)} 个因子表达式作为种子池")

    # 如果 top_seed > 0，先批量回测筛选 Top N
    if top_seed > 0 and top_seed < len(formulas):
        logger.info(f"\n=== 先批量回测筛选 Top {top_seed} 种子 ===")
        seed_formulas = []
        for i, expr in enumerate(formulas):
            logger.info(f"  [{i+1}/{len(formulas)}] 预评估: {expr[:50]}")
            try:
                from factor_library.backtest_engine import run_backtest
                r = run_backtest(expr, instruments=instruments)
                icir = abs(r.get("icir", 0))
                seed_formulas.append((expr, icir))
            except Exception:
                seed_formulas.append((expr, 0))

        seed_formulas.sort(key=lambda x: x[1], reverse=True)
        seeds = [expr for expr, _ in seed_formulas[:top_seed]]
        logger.info(f"  Top {top_seed} 种子 ICIR 范围: "
                     f"{seed_formulas[min(top_seed-1, len(seed_formulas)-1)][1]:.4f} ~ {seed_formulas[0][1]:.4f}")
    else:
        seeds = formulas

    # 初始化 QLib 和数据（所有模式共用）
    init_qlib()
    instruments = instruments or BACKTEST.instruments

    latest_dates = D.calendar(freq="day")
    max_date = latest_dates[-1] if len(latest_dates) > 0 else pd.Timestamp.now()
    ic_start = max_date - pd.DateOffset(months=BACKTEST.data_months)
    load_start = (ic_start - pd.DateOffset(months=BACKTEST.load_data_months)).strftime("%Y-%m-%d")
    load_end = max_date.strftime("%Y-%m-%d")

    df = load_ohlcv(instruments=instruments, start_time=load_start, end_time=load_end)
    engine = AlphaEngine(df)
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    ingested_count = 0

    # ==================================================================
    # 逐种子模式: 每条公式独立起GA，从自身出发变异进化
    # ==================================================================
    if per_seed:
        logger.info(f"\n=== GA 逐种子搜索模式 ===")
        logger.info(f"  种子总数: {len(seeds)}")
        logger.info(f"  1条种子 = 1次独立GA搜索")
        logger.info(f"  每次种群: {per_seed_population}, 代数: {per_seed_generations}")

        import re
        all_ga_results = []
        all_seen_expr = set()
        all_structures = {}

        def _extract_structure(expr: str) -> str:
            """提取因子结构签名（数字→#，用于检测结构相似）"""
            return re.sub(r'\s+', '', re.sub(r'\b\d+\.?\d*\b', '#', expr))

        for seed_idx, seed_expr in enumerate(seeds):
            logger.info(f"\n{'='*60}")
            logger.info(f"种子 [{seed_idx+1}/{len(seeds)}]: {seed_expr[:80]}")
            logger.info(f"{'='*60}")

            GA_CONFIG.n_generations = per_seed_generations
            GA_CONFIG.population_size = per_seed_population

            searcher = GAFactorSearcher(engine, returns)
            seed_results = searcher.search(seed_expressions=[seed_expr])

            new_count = 0
            for item in seed_results:
                expr = item.get("expression", "")
                if not expr or expr in all_seen_expr:
                    continue
                all_seen_expr.add(expr)

                struct_key = _extract_structure(expr)
                item_icir = abs(item.get("icir", 0))

                if struct_key in all_structures:
                    old_expr, old_icir, _ = all_structures[struct_key]
                    if item_icir <= old_icir:
                        continue
                    logger.info(f"  结构去重替换: {old_expr[:50]} (ICIR={old_icir:.4f}) → {expr[:50]} (ICIR={item_icir:.4f})")
                    all_ga_results = [x for x in all_ga_results if x.get("expression") != old_expr]

                all_structures[struct_key] = (expr, item_icir, item)
                all_ga_results.append(item)
                new_count += 1

            # 每种子搜索完立即入库最好的
            if auto_ingest and seed_results:
                best = seed_results[0]
                best_expr = best.get("expression", "")
                metrics = {
                    "ic_mean": best.get("ic", 0),
                    "icir": best.get("icir", 0),
                    "rank_ic": best.get("rank_ic", 0),
                    "rank_icir": best.get("rank_icir", 0),
                }
                if should_auto_ingest(metrics):
                    fid = add_factor(
                        expression=best_expr,
                        metrics=metrics,
                        tags=f"GA逐种子,种子{seed_idx+1},自动入库",
                        asset_universe=instruments,
                    )
                    ingested_count += 1
                    logger.info(f"  >> 自动入库: {fid} | ICIR={metrics['icir']:+.4f}")
                else:
                    best_icir = abs(best.get("icir", 0))
                    logger.info(f"  >> 未入库: 不满足阈值 (|ICIR|={best_icir:.4f})")

            logger.info(f"  本种子产出: {len(seed_results)} 个因子, 新增 {new_count} 个（累计 {len(all_ga_results)} 个）")

        ga_results = all_ga_results

    # ==================================================================
    # 混合模式: 所有种子混入1个大种群（原有逻辑）
    # ==================================================================
    else:
        logger.info(f"\n=== GA 种子搜索（混合模式）===")
        logger.info(f"  种子数: {len(seeds)}")
        logger.info(f"  种群大小: {population_size}")
        logger.info(f"  进化代数: {n_generations}")

        GA_CONFIG.n_generations = n_generations
        GA_CONFIG.population_size = population_size

        searcher = GAFactorSearcher(engine, returns)
        ga_results = searcher.search(seed_expressions=seeds)

    # ==================================================================
    # 自动入库（仅混合模式用；逐种子模式已在循环中入库）
    # ==================================================================
    if not per_seed:
        for item in ga_results:
            if not auto_ingest:
                break
            expr = item.get("expression", "")
            if not expr:
                continue
            metrics = {
                "ic_mean": item.get("ic", 0),
                "icir": item.get("icir", 0),
                "rank_ic": item.get("rank_ic", 0),
                "rank_icir": item.get("rank_icir", 0),
            }
            if should_auto_ingest(metrics):
                fid = add_factor(
                    expression=expr,
                    metrics=metrics,
                    tags="GA种子搜索,自动入库",
                    asset_universe=instruments,
                )
                ingested_count += 1
                logger.info(f"  自动入库 #{ingested_count}: {fid} | ICIR={metrics['icir']:+.4f} | {expr[:50]}")

    logger.info(f"\nGA 搜索完成! 找到 {len(ga_results)} 个有效因子, 入库 {ingested_count} 个")

    # 保存结果
    result_df = pd.DataFrame(ga_results)
    if output_path is None:
        from factor_library.config import EXPORT_DIR
        from datetime import datetime
        output_path = str(EXPORT_DIR / f"ga_seed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"结果保存: {output_path}")

    return result_df


# ==================== CLI ====================

def _resolve_input_path(args) -> str:
    """根据命令行参数解析输入文件路径"""
    if args.input:
        return args.input
    elif _ALL_FORMULAS_FILE.exists():
        logger.info(f"使用默认输入: {_ALL_FORMULAS_FILE}")
        return str(_ALL_FORMULAS_FILE)
    else:
        print(f"错误: 默认文件 {_ALL_FORMULAS_FILE} 不存在")
        print("  请先运行: python -m factor_library.prepare_formulas")
        print("  或用 --input 指定文件路径")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量因子搜索与自动入库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m factor_library.batch_search                      # 批量回测
  python -m factor_library.batch_search --ga                  # GA 混合模式搜索
  python -m factor_library.batch_search --ga --ga-per-seed    # GA 逐种子搜索（每条公式独立进化）
  python -m factor_library.batch_search --ga --ga-top-seed 10 # 先回测筛选Top10再做GA
  python -m factor_library.batch_search --ga --ga-per-seed --ga-per-gen 15 --ga-per-pop 40
  python -m factor_library.prepare_formulas                   # 生成 all_formulas.txt
        """,
    )
    # 输入
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入文件路径（默认 factor_library/all_formulas.txt）")

    # 模式
    parser.add_argument("--ga", action="store_true",
                        help="启用 GA 种子搜索模式（用公式文件中的表达式作为 GA 初始种群）")

    # 批量回测参数
    parser.add_argument("--no-ingest", action="store_true",
                        help="不自动入库")
    parser.add_argument("--tags", type=str, default="批量搜索",
                        help="附加标签")
    parser.add_argument("--instruments", type=str, default=None,
                        help="股票池（如 csi300）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="结果输出路径")

    # GA 参数
    parser.add_argument("--ga-generations", type=int, default=20,
                        help="GA 进化代数（默认20）")
    parser.add_argument("--ga-population", type=int, default=60,
                        help="GA 种群大小（默认60）")
    parser.add_argument("--ga-top-seed", type=int, default=0,
                        help="先用批量回测筛选 Top N 作为种子（0=全部公式作为种子）")
    parser.add_argument("--ga-per-seed", action="store_true",
                        help="启用逐种子独立GA搜索（每条公式独立进化）")
    parser.add_argument("--ga-per-gen", type=int, default=10,
                        help="逐种子模式下每次GA进化代数（默认10）")
    parser.add_argument("--ga-per-pop", type=int, default=30,
                        help="逐种子模式下每次GA种群大小（默认30）")

    args = parser.parse_args()
    input_path = _resolve_input_path(args)

    try:
        if args.ga:
            # GA 种子搜索模式
            df = ga_seed_search(
                input_path=input_path,
                n_generations=args.ga_generations,
                population_size=args.ga_population,
                auto_ingest=not args.no_ingest,
                top_seed=args.ga_top_seed,
                instruments=args.instruments,
                output_path=args.output,
                per_seed=args.ga_per_seed,
                per_seed_generations=args.ga_per_gen,
                per_seed_population=args.ga_per_pop,
            )
        else:
            # 批量回测模式
            df = batch_search(
                input_path=input_path,
                output_path=args.output,
                auto_ingest=not args.no_ingest,
                tags=args.tags,
                instruments=args.instruments,
            )
    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        input("按回车键退出...")
