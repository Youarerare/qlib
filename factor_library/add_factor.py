"""
手动添加因子到因子库的命令行工具

用法:
    # 交互式添加:
    python -m factor_library.add_factor

    # 命令行添加:
    python -m factor_library.add_factor --expr "rank(close)" --ic 0.05 --icir 0.6 --tags "手动添加,量价类"

    # 从 CSV 批量导入:
    python -m factor_library.add_factor --import factors.csv
"""
import sys
import argparse
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_library.database import (
    init_db, add_factor, exists, get_factor_by_expression,
    get_all_factors, export_to_csv, import_from_csv,
)
from factor_library.config import BACKTEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("add_factor")


def interactive_add():
    """交互式添加因子"""
    print("=" * 60)
    print("因子库 - 手动添加因子")
    print("=" * 60)

    expression = input("因子表达式: ").strip()
    if not expression:
        print("表达式不能为空!")
        return

    if exists(expression):
        print(f"因子已存在! 表达式: {expression}")
        info = get_factor_by_expression(expression)
        if info:
            print(f"  factor_id: {info['factor_id']}")
            print(f"  IC均值: {info['ic_mean']}")
            print(f"  ICIR: {info['icir']}")
        update = input("是否更新指标? (y/n): ").strip().lower()
        if update != "y":
            return

    description = input("描述 (可选): ").strip()
    tags = input("标签 (逗号分隔, 如: 手动添加,量价类): ").strip()

    print("\n回测指标 (直接回车跳过):")
    ic_mean = _input_float("IC均值: ")
    icir = _input_float("ICIR: ")
    ic_win_rate = _input_float("IC胜率: ")
    rank_ic = _input_float("Rank IC: ")
    rank_icir = _input_float("Rank ICIR: ")
    long_short = _input_float("多空收益: ")
    top_excess = _input_float("第一组超额: ")
    turnover = _input_float("换手率: ")
    max_dd = _input_float("最大回撤: ")

    print(f"\n回测日期范围 (直接回车使用默认值 {BACKTEST.start_time} ~ {BACKTEST.end_time}):")
    test_start = input("  开始日期: ").strip() or BACKTEST.start_time
    test_end = input("  结束日期: ").strip() or BACKTEST.end_time
    universe = input(f"股票池 (默认 {BACKTEST.instruments}): ").strip() or BACKTEST.instruments

    metrics = {
        "ic_mean": ic_mean,
        "icir": icir,
        "ic_win_rate": ic_win_rate,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "long_short_return": long_short,
        "top_group_excess": top_excess,
        "turnover": turnover,
        "max_drawdown": max_dd,
    }

    factor_id = add_factor(
        expression=expression,
        metrics=metrics,
        description=description,
        tags=tags,
        asset_universe=universe,
        test_start_date=test_start,
        test_end_date=test_end,
    )

    print(f"\n因子入库成功! factor_id={factor_id}")


def _input_float(prompt: str) -> float:
    """安全读取浮点数输入"""
    while True:
        val = input(prompt).strip()
        if not val:
            return 0.0
        try:
            return float(val)
        except ValueError:
            print("  请输入有效数字!")


def cli_add(args):
    """命令行模式添加因子"""
    metrics = {
        "ic_mean": args.ic or 0,
        "icir": args.icir or 0,
        "ic_win_rate": args.ic_win_rate or 0,
        "rank_ic": args.rank_ic or 0,
        "rank_icir": args.rank_icir or 0,
        "long_short_return": args.long_short or 0,
        "top_group_excess": args.top_excess or 0,
    }

    factor_id = add_factor(
        expression=args.expr,
        metrics=metrics,
        description=args.desc or "",
        tags=args.tags or "手动添加",
        asset_universe=args.universe or BACKTEST.instruments,
        test_start_date=args.start or BACKTEST.start_time,
        test_end_date=args.end or BACKTEST.end_time,
    )

    print(f"因子入库成功! factor_id={factor_id}")


def list_factors(args):
    """列出因子库中的因子"""
    df = get_all_factors(sort_by="icir", ascending=False)
    if df.empty:
        print("因子库为空!")
        return

    # 显示关键列
    display_cols = ["factor_id", "expression", "ic_mean", "icir", "ic_win_rate", "rank_icir", "tags"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))
    print(f"\n共 {len(df)} 个因子")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子库管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # add 命令
    add_parser = subparsers.add_parser("add", help="添加因子")
    add_parser.add_argument("--expr", "-e", type=str, required=True, help="因子表达式")
    add_parser.add_argument("--desc", "-d", type=str, default="", help="因子描述")
    add_parser.add_argument("--tags", "-t", type=str, default="手动添加", help="标签")
    add_parser.add_argument("--ic", type=float, default=0, help="IC均值")
    add_parser.add_argument("--icir", type=float, default=0, help="ICIR")
    add_parser.add_argument("--ic-win-rate", type=float, default=0, help="IC胜率")
    add_parser.add_argument("--rank-ic", type=float, default=0, help="Rank IC")
    add_parser.add_argument("--rank-icir", type=float, default=0, help="Rank ICIR")
    add_parser.add_argument("--long-short", type=float, default=0, help="多空收益")
    add_parser.add_argument("--top-excess", type=float, default=0, help="第一组超额")
    add_parser.add_argument("--start", type=str, default="", help="回测开始日期")
    add_parser.add_argument("--end", type=str, default="", help="回测结束日期")
    add_parser.add_argument("--universe", type=str, default="", help="股票池")

    # list 命令
    list_parser = subparsers.add_parser("list", help="列出因子")

    # import 命令
    import_parser = subparsers.add_parser("import", help="从CSV导入因子")
    import_parser.add_argument("--file", "-f", type=str, required=True, help="CSV文件路径")

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出因子库为CSV")
    export_parser.add_argument("--file", "-f", type=str, default=None, help="输出文件路径")

    # interactive 命令
    subparsers.add_parser("interactive", help="交互式添加因子")

    args = parser.parse_args()

    if args.command == "add":
        cli_add(args)
    elif args.command == "list":
        list_factors(args)
    elif args.command == "import":
        count = import_from_csv(args.file)
        print(f"导入完成: {count} 条因子")
    elif args.command == "export":
        path = export_to_csv(args.file)
        print(f"导出完成: {path}")
    elif args.command == "interactive":
        interactive_add()
    else:
        # 默认进入交互模式
        interactive_add()
