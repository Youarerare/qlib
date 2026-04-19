"""
整理 alpha191.txt + research_formula_candidates.txt → all_formulas.txt

合并两个源文件，去除前缀，过滤不兼容表达式，输出纯因子表达式（每行一个）。
输出文件: factor_library/all_formulas.txt

用法:
    python -m factor_library.prepare_formulas
    python -m factor_library.prepare_formulas --keep-incompatible
"""
import re
import argparse
from pathlib import Path

AUTO_ALPHA_DIR = Path(r"C:\Users\syk\Desktop\git_repo\auto_alpha")
OUTPUT_FILE = Path(__file__).resolve().parent / "all_formulas.txt"

# AlphaEngine 已在 _add_derived_fields 中自动生成 adv5~adv180
# 所以所有 adv 字段都支持！只需要排除不支持的算子/语法
UNSUPPORTED_PATTERNS = [
    # 不支持的算子
    (r"\bts_lag\b", "ts_lag→ts_delay"),
    (r"\bts_weighted_mean\b", "ts_weighted_mean"),
    (r"\bIndNeutralize\b", "IndNeutralize"),
    # 不支持的语法
    (r"\?", "三元运算符?:"),
]


def strip_prefix(line: str) -> str:
    """去掉 alphaXXX:: 或 alphaXXX: 前缀"""
    m = re.match(r"alpha\d+\s*::\s*(.*)", line.strip())
    if m:
        return m.group(1).strip()
    m = re.match(r"alpha\d+\s*:\s*(.*)", line.strip())
    if m:
        return m.group(1).strip()
    return line.strip()


def is_compatible(expr: str) -> tuple:
    """判断表达式是否兼容 AlphaEngine，返回 (bool, reason)"""
    if not expr:
        return False, "空表达式"
    # 未闭合括号
    if expr.count("(") != expr.count(")"):
        return False, "括号不匹配"
    # 不支持的模式
    for pattern, reason in UNSUPPORTED_PATTERNS:
        if re.search(pattern, expr):
            return False, reason
    # ^ 幂运算符（但 .001 之类的小数不算）
    if re.search(r"\w\^", expr):
        return False, "^幂运算"
    return True, ""


def load_source_file(filepath: Path) -> list:
    """从源文件加载表达式列表"""
    if not filepath.exists():
        print(f"  跳过（不存在）: {filepath}")
        return []
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                expr = strip_prefix(line)
                if expr:
                    lines.append(expr)
    return lines


def main():
    parser = argparse.ArgumentParser(description="整理因子表达式文件")
    parser.add_argument("--keep-incompatible", action="store_true",
                        help="保留不兼容的表达式（加 # 注释）")
    args = parser.parse_args()

    print(f"源目录: {AUTO_ALPHA_DIR}")

    # 1. 读取 alpha191.txt
    alpha191_exprs = load_source_file(AUTO_ALPHA_DIR / "alpha191.txt")
    print(f"  alpha191.txt: {len(alpha191_exprs)} 条")

    # 2. 读取 research_formula_candidates.txt
    candidates_exprs = load_source_file(AUTO_ALPHA_DIR / "research_formula_candidates.txt")
    print(f"  research_formula_candidates.txt: {len(candidates_exprs)} 条")

    # 3. 合并去重
    all_exprs = alpha191_exprs + candidates_exprs
    seen = set()
    unique = []
    for expr in all_exprs:
        if expr not in seen:
            seen.add(expr)
            unique.append(expr)
    print(f"  合并去重后: {len(unique)} 条")

    # 4. 兼容性过滤
    valid, skipped = [], []
    for expr in unique:
        ok, reason = is_compatible(expr)
        if ok:
            valid.append(expr)
        else:
            skipped.append((expr, reason))

    # 5. 写入（纯表达式，无前缀）
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for expr in valid:
            f.write(expr + "\n")
        if args.keep_incompatible and skipped:
            f.write("\n# === 以下表达式不兼容，仅供参考 ===\n")
            for expr, reason in skipped:
                f.write(f"# [{reason}] {expr}\n")

    print(f"\n整理完成!")
    print(f"  兼容表达式: {len(valid)} 条 → {OUTPUT_FILE}")
    print(f"  跳过(不兼容): {len(skipped)} 条")
    if skipped:
        print(f"\n跳过原因统计:")
        reasons = {}
        for _, r in skipped:
            reasons[r] = reasons.get(r, 0) + 1
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {r}: {cnt} 条")


if __name__ == "__main__":
    main()
