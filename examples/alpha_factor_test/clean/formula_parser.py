"""
Alpha101/191公式解析器
- 支持单冒号和双冒号格式
- 自动补全括号
- 记录不可修复的公式
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import ALPHA101_FORMULA_PATH, ALPHA191_FORMULA_PATH

logger = logging.getLogger(__name__)


class FormulaParser:
    """Alpha公式文件解析器"""

    def parse_single_colon(self, filepath: str) -> Dict[str, str]:
        """解析 alpha001: formula 格式"""
        formulas = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"(alpha\d{3}):\s*(.+)", line)
                if match:
                    formulas[match.group(1)] = match.group(2)
        return formulas

    def parse_double_colon(self, filepath: str) -> Dict[str, str]:
        """解析 alpha001:: formula 格式"""
        formulas = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"(alpha\d{3})::\s*(.+)", line)
                if match:
                    formulas[match.group(1)] = match.group(2)
        return formulas


def fix_brackets(formula: str) -> Tuple[str, bool]:
    """
    自动补全括号
    返回: (修复后的公式, 是否修复成功)
    """
    stack = []
    for i, ch in enumerate(formula):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                return formula, False

    if not stack:
        return formula, True

    fixed = formula + ")" * len(stack)
    logger.info(f"括号补全: 添加了{len(stack)}个右括号")
    return fixed, True


def validate_formula(formula: str) -> Tuple[bool, Optional[str]]:
    """
    验证公式基本有效性
    返回: (是否有效, 错误信息)
    """
    if not formula.strip():
        return False, "空公式"

    fixed, ok = fix_brackets(formula)
    if not ok:
        return False, "括号不匹配且无法自动修复"

    unsupported = []
    for func in ["IndNeutralize", "group_neutralize", "group_mean"]:
        if func in formula:
            unsupported.append(func)

    if unsupported:
        return True, f"包含不支持的行业中性化函数: {unsupported} (将被移除)"

    return True, None


def load_alpha101_formulas(filepath: Optional[str] = None) -> Dict[str, str]:
    """加载Alpha101公式 (双冒号格式)"""
    filepath = filepath or ALPHA101_FORMULA_PATH
    parser = FormulaParser()
    return parser.parse_double_colon(filepath)


def load_alpha191_formulas(filepath: Optional[str] = None) -> Dict[str, str]:
    """加载Alpha191公式 (单冒号格式)"""
    filepath = filepath or ALPHA191_FORMULA_PATH
    parser = FormulaParser()
    return parser.parse_single_colon(filepath)


def load_all_formulas(
    alpha101_path: Optional[str] = None,
    alpha191_path: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str], List[dict]]:
    """
    加载所有公式并验证
    返回: (alpha101公式, alpha191公式, 问题列表)
    """
    a101 = load_alpha101_formulas(alpha101_path)
    a191 = load_alpha191_formulas(alpha191_path)

    issues = []
    for name, formula in {**a101, **a191}.items():
        valid, error = validate_formula(formula)
        if not valid:
            issues.append({"name": name, "formula": formula, "error": error})
            logger.warning(f"公式验证失败: {name} - {error}")

    logger.info(f"加载完成: Alpha101={len(a101)}, Alpha191={len(a191)}, 问题={len(issues)}")
    return a101, a191, issues
