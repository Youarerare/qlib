"""
Alpha因子公式解析器
解析alpha101和alpha191的公式定义

Alpha101 (100个) = research_formula_candidates.txt (双冒号格式)
Alpha191 (191个) = alpha191.txt (单冒号格式)
"""

import re
from typing import Dict, List, Tuple
from pathlib import Path


class AlphaFormulaParser:
    """Alpha因子公式解析器"""
    
    def __init__(self):
        self.formulas = {}
        
    def parse_single_colon(self, file_path: str) -> Dict[str, str]:
        """解析单冒号格式: alpha001: formula"""
        formulas = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'(alpha\d+):\s*(.+)', line)
                if match:
                    formulas[match.group(1)] = match.group(2)
        self.formulas.update(formulas)
        return formulas
    
    def parse_double_colon(self, file_path: str) -> Dict[str, str]:
        """解析双冒号格式: alpha001:: formula"""
        formulas = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'(alpha\d+)::\s*(.+)', line)
                if match:
                    formulas[match.group(1)] = match.group(2)
        self.formulas.update(formulas)
        return formulas
    
    # 兼容旧方法名
    def parse_alpha101_file(self, file_path: str) -> Dict[str, str]:
        return self.parse_single_colon(file_path)
    
    def parse_alpha191_file(self, file_path: str) -> Dict[str, str]:
        return self.parse_double_colon(file_path)


def load_alpha101_formulas(file_path: str = None) -> Dict[str, str]:
    """加载Alpha101公式 (100个，双冒号格式)"""
    parser = AlphaFormulaParser()
    if file_path is None:
        file_path = r'C:\Users\syk\Desktop\git_repo\auto_alpha\research_formula_candidates.txt'
    return parser.parse_double_colon(file_path)


def load_alpha191_formulas(file_path: str = None) -> Dict[str, str]:
    """加载Alpha191公式 (191个，单冒号格式)"""
    parser = AlphaFormulaParser()
    if file_path is None:
        file_path = r'C:\Users\syk\Desktop\git_repo\auto_alpha\alpha191.txt'
    return parser.parse_single_colon(file_path)


def load_all_formulas(alpha101_path: str = None, alpha191_path: str = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """加载所有公式"""
    alpha101 = load_alpha101_formulas(alpha101_path)
    alpha191 = load_alpha191_formulas(alpha191_path)
    return alpha101, alpha191


if __name__ == "__main__":
    alpha101, alpha191 = load_all_formulas()
    print(f"Alpha101公式数量: {len(alpha101)}")
    print(f"Alpha191公式数量: {len(alpha191)}")
