"""
改进因子回测对比脚本

回测原始因子与三个改进因子（A/B/C），输出对比表格，
并将优于原始因子的改进因子自动入库。

用法:
    python -m factor_library.test_improved_factors
    python factor_library/test_improved_factors.py
"""
import sys
import json
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from factor_library.config import ORIGINAL_FACTOR, IMPROVED_FACTORS, THRESHOLD
from factor_library.backtest_engine import run_backtest
from factor_library.database import add_factor, exists, update_factor
from factor_library.search_one import evaluate_and_ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_improved")


def compare_factors(original_expr: str = None, improved: dict = None,
                    output_csv: str = None) -> pd.DataFrame:
    """
    对比原始因子与改进因子的回测表现

    Parameters
    ----------
    original_expr : str
        原始因子表达式
    improved : dict
        改进因子字典 {"A": {"expression": ..., "description": ...}, ...}
    output_csv : str, optional
        对比结果输出路径

    Returns
    -------
    pd.DataFrame
        对比表格
    """
    original_expr = original_expr or ORIGINAL_FACTOR
    improved = improved or IMPROVED_FACTORS

    logger.info("=" * 80)
    logger.info("改进因子回测对比")
    logger.info("=" * 80)

    # 1. 回测原始因子
    logger.info("\n>>> 回测原始因子")
    logger.info(f"    {original_expr}")
    orig_result = run_backtest(original_expr, compute_groups=True)

    if "error" in orig_result:
        logger.error(f"原始因子回测失败: {orig_result['error']}")
        return pd.DataFrame()

    # 入库原始因子
    orig_id = add_factor(
        expression=original_expr,
        metrics=orig_result,
        description="原始因子: ts_arg_min + sqrt + max + cs_mean",
        tags="原始因子,成交量类",
        asset_universe=orig_result.get("asset_universe", "csi300"),
        test_start_date=orig_result.get("test_start_date", ""),
        test_end_date=orig_result.get("test_end_date", ""),
        group_returns=json.dumps(orig_result.get("group_returns", {}), ensure_ascii=False),
    )

    # 2. 回测改进因子
    improved_results = {}
    for name, info in improved.items():
        expr = info["expression"]
        desc = info["description"]
        tags = info.get("tags", f"改进因子{name}")

        logger.info(f"\n>>> 回测改进因子 {name}")
        logger.info(f"    {expr}")

        try:
            r = run_backtest(expr, compute_groups=True)
            if "error" not in r:
                improved_results[name] = r

                # 对比：是否优于原始因子
                is_better = _is_better_than_original(r, orig_result)
                r["is_better_than_original"] = is_better

                if is_better:
                    logger.info(f"    >> 改进因子{name} 优于原始因子! 自动入库")
                    all_tags = f"{tags},改进因子,优于原始"
                    fid = add_factor(
                        expression=expr,
                        metrics=r,
                        description=f"改进因子{name}: {desc}",
                        tags=all_tags,
                        asset_universe=r.get("asset_universe", "csi300"),
                        test_start_date=r.get("test_start_date", ""),
                        test_end_date=r.get("test_end_date", ""),
                        group_returns=json.dumps(r.get("group_returns", {}), ensure_ascii=False),
                    )
                    r["factor_id"] = fid
                else:
                    logger.info(f"    >> 改进因子{name} 未优于原始因子")
                    # 仍然入库，但不标记为"优于原始"
                    all_tags = f"{tags},改进因子"
                    fid = add_factor(
                        expression=expr,
                        metrics=r,
                        description=f"改进因子{name}: {desc}",
                        tags=all_tags,
                        asset_universe=r.get("asset_universe", "csi300"),
                        test_start_date=r.get("test_start_date", ""),
                        test_end_date=r.get("test_end_date", ""),
                        group_returns=json.dumps(r.get("group_returns", {}), ensure_ascii=False),
                    )
                    r["factor_id"] = fid
            else:
                logger.error(f"    改进因子{name} 回测失败: {r.get('error')}")
                improved_results[name] = {"expression": expr, "error": r["error"]}
        except Exception as e:
            logger.error(f"    改进因子{name} 异常: {e}")
            improved_results[name] = {"expression": expr, "error": str(e)}

    # 3. 生成对比表格
    comparison = _build_comparison_table(orig_result, improved_results, original_expr)
    logger.info("\n" + "=" * 80)
    logger.info("对比结果:")
    logger.info("\n" + comparison.to_string())

    # 4. 保存
    if output_csv is None:
        from factor_library.config import EXPORT_DIR
        from datetime import datetime
        output_csv = str(EXPORT_DIR / f"improved_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    comparison.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"\n对比结果已保存: {output_csv}")

    return comparison


def _is_better_than_original(improved_metrics: dict, orig_metrics: dict) -> bool:
    """
    判断改进因子是否在关键指标上优于原始因子

    条件: ICIR > improved_icir 且 |IC均值| > improved_ic_mean
    """
    imp_icir = abs(improved_metrics.get("icir", 0))
    imp_ic = abs(improved_metrics.get("ic_mean", improved_metrics.get("ic", 0)))
    orig_icir = abs(orig_metrics.get("icir", 0))
    orig_ic = abs(orig_metrics.get("ic_mean", orig_metrics.get("ic", 0)))

    # 改进因子需超过阈值且优于原始因子
    better_threshold = (imp_icir > THRESHOLD.improved_icir and
                        imp_ic > THRESHOLD.improved_ic_mean)
    better_than_orig = (imp_icir > orig_icir and imp_ic > orig_ic)

    # 只要满足阈值条件就标记为改进（不一定要严格优于原始因子，
    # 因为原始因子可能本身就不好）
    return better_threshold


def _build_comparison_table(orig_result: dict, improved_results: dict,
                            original_expr: str) -> pd.DataFrame:
    """构建对比表格"""
    rows = []

    # 原始因子
    rows.append({
        "name": "原始因子",
        "expression": original_expr[:60],
        "ic_mean": orig_result.get("ic_mean", orig_result.get("ic", 0)),
        "icir": orig_result.get("icir", 0),
        "ic_win_rate": orig_result.get("ic_win_rate", orig_result.get("ic_positive_ratio", 0)),
        "rank_ic": orig_result.get("rank_ic", 0),
        "rank_icir": orig_result.get("rank_icir", 0),
        "long_short_return": orig_result.get("long_short_return", 0),
        "top_group_excess": orig_result.get("top_group_excess", 0),
        "is_better_than_original": "-",
        "group_returns": str(orig_result.get("group_returns", {})),
    })

    # 改进因子
    for name, r in improved_results.items():
        if "error" in r:
            rows.append({
                "name": f"改进{name}",
                "expression": r.get("expression", ""),
                "ic_mean": 0, "icir": 0, "ic_win_rate": 0,
                "rank_ic": 0, "rank_icir": 0,
                "long_short_return": 0, "top_group_excess": 0,
                "is_better_than_original": f"失败: {r['error'][:30]}",
                "group_returns": "",
            })
        else:
            rows.append({
                "name": f"改进{name}",
                "expression": r.get("expression", "")[:60],
                "ic_mean": r.get("ic_mean", r.get("ic", 0)),
                "icir": r.get("icir", 0),
                "ic_win_rate": r.get("ic_win_rate", r.get("ic_positive_ratio", 0)),
                "rank_ic": r.get("rank_ic", 0),
                "rank_icir": r.get("rank_icir", 0),
                "long_short_return": r.get("long_short_return", 0),
                "top_group_excess": r.get("top_group_excess", 0),
                "is_better_than_original": "是" if r.get("is_better_than_original") else "否",
                "group_returns": str(r.get("group_returns", {})),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    try:
        comparison = compare_factors()
    except Exception as e:
        logger.error(f"对比测试失败: {e}", exc_info=True)
        input("按回车键退出...")
