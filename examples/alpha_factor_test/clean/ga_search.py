"""
基于DEAP的遗传算法因子搜索
"""
import logging
import random
import operator
import re
import gc  # 垃圾回收
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import GA, BACKTEST, ALL_OPERATORS, TS_OPERATORS, CS_OPERATORS, DATA_FIELDS, ADV_FIELDS
from .alpha_engine import AlphaEngine
from .ic_analyzer import calc_ic_series, calc_ic_summary

logger = logging.getLogger(__name__)


# ===== 表达式树和子树交叉操作 =====

class ExpressionTreeNode:
    """表达式树节点"""
    def __init__(self, value: str, children: List['ExpressionTreeNode'] = None):
        self.value = value
        self.children = children or []
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def depth(self) -> int:
        """计算以该节点为根的子树深度"""
        if self.is_leaf():
            return 0
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """计算子树大小（节点数）"""
        if self.is_leaf():
            return 1
        return 1 + sum(child.size() for child in self.children)


class ExpressionParser:
    """表达式解析器"""
    
    @classmethod
    def parse(cls, expr: str) -> ExpressionTreeNode:
        """将字符串表达式解析为树"""
        expr = expr.strip()
        if '(' not in expr:
            return ExpressionTreeNode(expr)
        
        paren_idx = expr.index('(')
        func_name = expr[:paren_idx].strip()
        args_str = expr[paren_idx + 1:expr.rindex(')')]
        args = cls._split_arguments(args_str)
        children = [cls.parse(arg) for arg in args]
        
        return ExpressionTreeNode(func_name, children)
    
    @classmethod
    def _split_arguments(cls, args_str: str) -> List[str]:
        """分割参数，考虑嵌套括号"""
        args = []
        current = []
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                arg = ''.join(current).strip()
                if arg:
                    args.append(arg)
                current = []
            else:
                current.append(char)
        
        if current:
            arg = ''.join(current).strip()
            if arg:
                args.append(arg)
        
        return args
    
    @classmethod
    def to_string(cls, tree: ExpressionTreeNode) -> str:
        """将树转换回字符串"""
        if tree.is_leaf():
            return tree.value
        children_str = ', '.join(cls.to_string(c) for c in tree.children)
        return f"{tree.value}({children_str})"


class SubtreeCrossover:
    """子树交叉操作器"""
    
    def __init__(self, max_depth: int = 5, max_size: int = 30):
        self.max_depth = max_depth
        self.max_size = max_size
    
    def crossover(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """
        执行子树交叉
        
        Parameters
        ----------
        expr1, expr2 : str
            两个父个体表达式
        
        Returns
        -------
        Tuple[str, str]
            两个子代表达式
        """
        try:
            tree1 = ExpressionParser.parse(expr1)
            tree2 = ExpressionParser.parse(expr2)
            
            nodes1 = self._collect_all_nodes(tree1)
            nodes2 = self._collect_all_nodes(tree2)
            
            if not nodes1 or not nodes2:
                return expr1, expr2
            
            subtree1 = random.choice(nodes1)
            subtree2 = random.choice(nodes2)
            
            child1_tree = self._replace_subtree(tree1, subtree1, subtree2)
            child2_tree = self._replace_subtree(tree2, subtree2, subtree1)
            
            if (child1_tree.depth() > self.max_depth or 
                child2_tree.depth() > self.max_depth or
                child1_tree.size() > self.max_size or
                child2_tree.size() > self.max_size):
                return expr1, expr2
            
            child1 = ExpressionParser.to_string(child1_tree)
            child2 = ExpressionParser.to_string(child2_tree)
            
            return child1, child2
            
        except Exception as e:
            logger.debug(f"交叉操作失败: {e}")
            return expr1, expr2
    
    def _collect_all_nodes(self, tree: ExpressionTreeNode) -> List[ExpressionTreeNode]:
        """收集树中的所有节点"""
        nodes = []
        def traverse(node):
            nodes.append(node)
            for child in node.children:
                traverse(child)
        traverse(tree)
        return nodes
    
    def _replace_subtree(self, tree: ExpressionTreeNode, 
                         target: ExpressionTreeNode,
                         replacement: ExpressionTreeNode) -> ExpressionTreeNode:
        """替换子树（返回新树，不修改原树）"""
        if tree is target:
            return self._deep_copy(replacement)
        if tree.is_leaf():
            return tree
        new_children = []
        for child in tree.children:
            new_child = self._replace_subtree(child, target, replacement)
            new_children.append(new_child)
        return ExpressionTreeNode(tree.value, new_children)
    
    def _deep_copy(self, tree: ExpressionTreeNode) -> ExpressionTreeNode:
        """深拷贝树"""
        if tree.is_leaf():
            return ExpressionTreeNode(tree.value)
        new_children = [self._deep_copy(child) for child in tree.children]
        return ExpressionTreeNode(tree.value, new_children)


# ===== IC计算函数 =====


def _safe_spearman_ic(factor_values: pd.Series, returns: pd.Series,
                       start_date: str = None, end_date: str = None) -> float:
    """安全计算Rank IC均值"""
    try:
        ic_s = calc_ic_series(factor_values, returns, method="spearman", min_stocks=10,
                              start_date=start_date, end_date=end_date)
        if len(ic_s) < 5:
            logger.debug(f"IC序列过短: {len(ic_s)} < 5")
            return 0.0
        summary = calc_ic_summary(ic_s)
        return summary["ic_mean"]
    except Exception as e:
        logger.debug(f"IC计算失败: {e}")
        return 0.0


def _safe_ir(factor_values: pd.Series, returns: pd.Series,
             start_date: str = None, end_date: str = None) -> float:
    """安全计算Rank ICIR"""
    try:
        ic_s = calc_ic_series(factor_values, returns, method="spearman", min_stocks=10,
                              start_date=start_date, end_date=end_date)
        if len(ic_s) < 5:
            logger.debug(f"IC序列过短: {len(ic_s)} < 5")
            return 0.0
        summary = calc_ic_summary(ic_s)
        return summary["icir"]
    except Exception as e:
        logger.debug(f"ICIR计算失败: {e}")
        return 0.0


def _compute_detailed_metrics(factor_values: pd.Series, returns: pd.Series,
                               start_date: str = None, end_date: str = None) -> Dict[str, float]:
    """计算详细评估指标: IC, IR, ICIR, Rank ICIR"""
    result = {"ic": 0.0, "ir": 0.0, "icir": 0.0, "rank_ic": 0.0, "rank_icir": 0.0,
              "n_periods": 0, "ic_positive_ratio": 0.0}
    try:
        # Pearson IC
        pearson_ic_s = calc_ic_series(factor_values, returns, method="pearson", min_stocks=10,
                                      start_date=start_date, end_date=end_date)
        # Spearman Rank IC
        rank_ic_s = calc_ic_series(factor_values, returns, method="spearman", min_stocks=10,
                                   start_date=start_date, end_date=end_date)

        if len(pearson_ic_s) >= 5:
            ic_mean = pearson_ic_s.mean()
            ic_std = pearson_ic_s.std()
            result["ic"] = ic_mean
            result["ir"] = ic_mean / ic_std if ic_std > 0 else 0.0
            result["icir"] = result["ir"]  # ICIR = IC均值/IC标准差 = IR
            result["ic_positive_ratio"] = (pearson_ic_s > 0).mean()

        if len(rank_ic_s) >= 5:
            rank_ic_mean = rank_ic_s.mean()
            rank_ic_std = rank_ic_s.std()
            result["rank_ic"] = rank_ic_mean
            result["rank_icir"] = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0

        result["n_periods"] = max(len(pearson_ic_s), len(rank_ic_s))
    except Exception as e:
        logger.debug(f"详细指标计算失败: {e}")
    return result


class ExpressionGenerator:
    """表达式树生成器（优化版）"""

    def __init__(self, engine: AlphaEngine, returns: pd.Series):
        self.engine = engine
        self.returns = returns

        self.ts_ops = list(engine.ts_ops.keys())
        self.cs_ops = list(engine.cs_ops.keys())
        self.math_ops = list(engine.math_ops.keys())
        self.binary_ops = list(engine.binary_ops.keys())
        
        # 【优化1】分离基础字段和adv字段，确保多样性
        self.base_fields = [f for f in DATA_FIELDS if f in engine.df.columns]  # open, high, low, close, volume, vwap, returns
        self.adv_fields = [f for f in ADV_FIELDS if f in engine.df.columns]    # adv5, adv10, ...
        self.data_fields = self.base_fields + self.adv_fields
        
        self.constants = [1, 2, 3, 5, 10, 20, 30, 60, 0.5, 0.1, -1]

        self._cache: Dict[str, float] = {}
        self._cache_max_size = 5000  # 限制缓存大小，防止内存泄漏
        self._existing_factors: List[pd.Series] = []
        self._max_corr: float = 0.7
        
        # 子树交叉操作器
        self.crossover_operator = SubtreeCrossover(
            max_depth=GA.max_tree_depth, 
            max_size=30
        )
        
        # 【优化2】跟踪字段使用统计
        self._field_usage: Dict[str, int] = {f: 0 for f in self.data_fields}
        self._generation_count = 0

    def set_existing_factors(self, factors: List[pd.Series]):
        """设置已有因子列表，用于相关性惩罚"""
        self._existing_factors = factors

    def generate_random(self, max_depth: int = 3, force_field_category: str = None) -> str:
        """
        随机生成一个表达式
        
        Parameters
        ----------
        force_field_category : str, optional
            强制使用某类字段：'base'(基础量价), 'volume'(成交量), 'mixed'(混合)
        """
        return self._gen_expr(max_depth, is_root=True, force_category=force_field_category)

    def _gen_expr(self, depth: int, is_root: bool = False, force_category: str = None) -> str:
        if depth <= 0:
            return self._gen_terminal(force_category=force_category)

        choices = ["ts", "cs", "math", "binary", "terminal"]
        if is_root:
            weights = [0.4, 0.2, 0.15, 0.15, 0.1]  # 增加ts算子概率
        else:
            weights = [0.3, 0.15, 0.15, 0.2, 0.2]

        choice = random.choices(choices, weights=weights, k=1)[0]

        if choice == "ts":
            op = random.choice(self.ts_ops)
            arg = self._gen_expr(depth - 1, force_category=force_category)
            window = random.choice([3, 5, 10, 15, 20, 30, 60])
            if op in ["ts_corr", "ts_covariance"]:
                # 【优化3】ts_corr 强制用不同字段，增加有意义的相关性
                arg2 = self._gen_expr(depth - 1, force_category=force_category)
                return f"{op}({arg}, {arg2}, {window})"
            return f"{op}({arg}, {window})"

        elif choice == "cs":
            op = random.choice(self.cs_ops)
            arg = self._gen_expr(depth - 1, force_category=force_category)
            return f"{op}({arg})"

        elif choice == "math":
            op = random.choice(self.math_ops)
            arg = self._gen_expr(depth - 1, force_category=force_category)
            if op in ["signed_power"]:
                exp = random.choice([2, 0.5, 0.5])
                return f"{op}({arg}, {exp})"
            return f"{op}({arg})"

        elif choice == "binary":
            op = random.choice(self.binary_ops)
            a = self._gen_expr(depth - 1, force_category=force_category)
            b = self._gen_expr(depth - 1, force_category=force_category)
            
            # 【新增】避免生成无意义的二元运算
            # 1. 避免 multiply(常数, expr) 这种仅缩放系数的形式
            # 2. 避免 subtract(同字段, 同字段) 这种恒为0的形式
            if op == "multiply":
                # 检查是否是 常数 * expr 的形式
                a_is_const = a.replace('.', '').replace('-', '').isdigit()
                b_is_const = b.replace('.', '').replace('-', '').isdigit()
                if a_is_const or b_is_const:
                    # 如果是常数乘法，直接返回非恒定部分（避免无意义缩放）
                    if a_is_const and not b_is_const:
                        return b
                    elif b_is_const and not a_is_const:
                        return a
                    else:
                        # 两个都是常数，返回其乘积
                        try:
                            return str(float(a) * float(b))
                        except:
                            return a
            
            # 避免 subtract(field, field) 这种形式
            if op == "subtract" and a == b:
                return a  # 返回原表达式，避免生成0
            
            return f"{op}({a}, {b})"

        else:
            return self._gen_terminal(force_category=force_category)

    def _gen_terminal(self, force_category: str = None) -> str:
        """
        生成终端节点（字段或常数）
        
        【优化4】分层采样：保证30%个体使用base字段，避免全adv早熟
        """
        # 根据强制类别选择字段
        if force_category == "base":
            field_pool = self.base_fields
        elif force_category == "volume":
            field_pool = self.adv_fields if self.adv_fields else self.base_fields
        else:
            field_pool = self.data_fields
        
        # 80%概率选字段，20%选常数
        if random.random() < 0.8:
            field = random.choice(field_pool)
            self._field_usage[field] = self._field_usage.get(field, 0) + 1
            return field
        return str(random.choice(self.constants))

    def evaluate_expression(self, expr: str,
                            start_date: str = None, end_date: str = None
                            ) -> Tuple[Optional[pd.Series], float]:
        """
        计算表达式并返回(因子值, 适应度)
        
        【优化5】综合适应度函数：
        - IC/IR/RankICIR 绝对值越大越好（正负都有价值）
        - fitness = abs(RankIC) * ic_weight + abs(RankICIR) * ir_weight + 稳定性奖励 - 复杂度惩罚
        """
        if expr in self._cache:
            return None, self._cache[expr]

        # 【新增】检测无意义表达式
        if _is_meaningless_expression(expr):
            self._cache[expr] = -999
            return None, -999

        try:
            factor = self.engine.calculate(expr)
            if factor.isna().all():
                return None, -999
            
            # 【新增】检测因子是否为常数（标准差接近0）
            factor_std = factor.std()
            if factor_std < 1e-10:  # 几乎是常数
                self._cache[expr] = -999
                return None, -999

            # 计算多种指标
            ic_mean = _safe_spearman_ic(factor, self.returns, start_date, end_date)
            icir = _safe_ir(factor, self.returns, start_date, end_date)
            
            # 【优化6】计算 Pearson IC 和 IC 稳定性
            try:
                pearson_ic_s = calc_ic_series(factor, self.returns, method="pearson", min_stocks=10,
                                              start_date=start_date, end_date=end_date)
                pearson_ic_mean = pearson_ic_s.mean() if len(pearson_ic_s) > 0 else 0
                ic_positive_ratio = (pearson_ic_s > 0).mean() if len(pearson_ic_s) > 0 else 0
            except:
                pearson_ic_mean = 0
                ic_positive_ratio = 0

            # 【优化7】综合适应度：绝对值越大越好（正IC=正向预测，负IC=反向预测）
            fitness = abs(ic_mean) * GA.ic_weight + abs(icir) * GA.ir_weight
            
            # IC稳定性奖励：无论正负，只要IC方向一致就奖励
            # ic_positive_ratio 接近 0 或 1 都表示稳定（全负或全正），0.5 表示随机
            ic_consistency = max(ic_positive_ratio, 1 - ic_positive_ratio)  # 0.5~1.0
            if ic_consistency > 0.7:  # 70%以上时间方向一致
                fitness += 0.1 * ic_consistency

            # 【优化8】复杂度惩罚：鼓励简洁因子
            try:
                tree = ExpressionParser.parse(expr)
                size = tree.size()
                # 节点数 > 10 时开始惩罚，> 15 时重度惩罚
                if size > 15:
                    fitness *= 0.7
                elif size > 10:
                    fitness *= (1.0 - 0.03 * (size - 10))
            except:
                pass

            # 相关性惩罚
            if self._existing_factors and len(self._existing_factors) > 0:
                max_corr = 0
                for existing in self._existing_factors[:20]:
                    try:
                        valid = ~(factor.isna() | existing.isna())
                        if valid.sum() > 100:
                            c = factor[valid].corr(existing[valid])
                            if abs(c) > max_corr:
                                max_corr = abs(c)
                    except Exception as e:
                        logger.debug(f"相关性计算失败: {e}")
                        pass
                if max_corr > self._max_corr:
                    fitness -= GA.correlation_penalty * (max_corr - self._max_corr)

            # 限制缓存大小，防止内存泄漏
            if len(self._cache) >= self._cache_max_size:
                # 清理一半缓存（保留最近的）
                items = list(self._cache.items())
                self._cache = dict(items[len(items)//2:])
            
            self._cache[expr] = fitness
            return factor, fitness

        except Exception:
            # 限制缓存大小
            if len(self._cache) >= self._cache_max_size:
                items = list(self._cache.items())
                self._cache = dict(items[len(items)//2:])
            self._cache[expr] = -999
            return None, -999

    def mutate(self, expr: str) -> str:
        """变异操作"""
        ops = ["replace_subtree", "change_window", "change_field", "change_op"]
        op = random.choice(ops)

        if op == "replace_subtree":
            depth = random.randint(1, 2)
            new_subtree = self._gen_expr(depth)
            
            try:
                tree = ExpressionParser.parse(expr)
                nodes = self._collect_mutable_nodes(tree)
                
                if nodes:
                    target_node = random.choice(nodes)
                    new_tree = self._replace_subtree_node(tree, target_node, new_subtree)
                    
                    if new_tree.depth() <= GA.max_tree_depth and new_tree.size() <= 30:
                        return ExpressionParser.to_string(new_tree)
                
                return new_subtree
            except Exception as e:
                logger.debug(f"子树替换变异失败: {e}")
                return new_subtree

        elif op == "change_window":
            nums = re_findall_numbers(expr)
            if nums:
                old = random.choice(nums)
                new = random.choice([3, 5, 10, 15, 20, 30, 60])
                expr = expr.replace(str(old), str(new), 1)
            return expr

        elif op == "change_field":
            for field in self.data_fields:
                if field in expr:
                    new_field = random.choice(self.data_fields)
                    expr = expr.replace(field, new_field, 1)
                    break
            return expr

        else:
            for ts_op in self.ts_ops:
                if ts_op in expr:
                    new_op = random.choice(self.ts_ops)
                    expr = expr.replace(ts_op, new_op, 1)
                    break
            return expr
    
    def _collect_mutable_nodes(self, tree: ExpressionTreeNode) -> List[ExpressionTreeNode]:
        """收集所有可被替换的节点（排除根节点）"""
        nodes = []
        
        def _traverse(node, is_root=True):
            if not is_root:
                nodes.append(node)
            for child in node.children:
                _traverse(child, is_root=False)
        
        _traverse(tree)
        return nodes
    
    def _replace_subtree_node(self, tree: ExpressionTreeNode, 
                               target: ExpressionTreeNode,
                               replacement_expr: str) -> ExpressionTreeNode:
        """替换树中的指定节点"""
        if tree is target:
            return ExpressionParser.parse(replacement_expr)
        
        if tree.is_leaf():
            return tree
        
        new_children = []
        for child in tree.children:
            new_child = self._replace_subtree_node(child, target, replacement_expr)
            new_children.append(new_child)
        
        return ExpressionTreeNode(tree.value, new_children)

    def crossover(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """
        交叉操作（优化版 - 子树交叉）
        
        真正组合两个父个体的特征，体现遗传算法的进化思想
        
        Parameters
        ----------
        expr1, expr2 : str
            两个父个体表达式
        
        Returns
        -------
        Tuple[str, str]
            两个子代表达式
        """
        return self.crossover_operator.crossover(expr1, expr2)


def re_findall_numbers(s: str) -> List[str]:
    """查找字符串中的所有数字"""
    return re.findall(r"\b(\d+\.?\d*)\b", s)


def _is_meaningless_expression(expr: str) -> bool:
    """
    检测无意义表达式
    
    检测模式:
    1. 相同字段相减: (close - close), (high - high)
    2. 除以零风险: ... / (close - close)
    3. 常数表达式: 只包含数字和运算符
    4. 对负数开方: sqrt(...负数...)
    5. 对负数取log: log(...负数...)
    """
    # 1. 检测相同字段相减
    fields = ["open", "high", "low", "close", "volume", "vwap", "returns"]
    for field in fields:
        # 匹配 (field - field) 或 (field-field)
        pattern = rf'\(\s*{field}\s*-\s*{field}\s*\)'
        if re.search(pattern, expr):
            return True
    
    # 2. 检测常数表达式（过于简单的）
    # 如果不包含任何字段，且只包含数字和运算符
    if not any(field in expr for field in fields):
        # 可能是纯常数表达式
        test_expr = re.sub(r'[\d\s\+\-\*/\(\)\.]', '', expr)
        if not test_expr:  # 只剩下数字和运算符
            return True
    
    # 3. 检测除以零的风险模式
    # 匹配 / (xxx - xxx) 或 /(xxx-xxx)
    div_zero_pattern = r'/\s*\([^)]*-\s*\1\)'  # 简化检测
    for field in fields:
        pattern = rf'/\s*\(\s*[^)]*{field}\s*-\s*{field}\s*\)'
        if re.search(pattern, expr):
            return True
    
    return False


def _extract_structure_signature(expr: str) -> str:
    """
    提取结构签名，用于检测仅系数不同的等价因子
    
    将所有数字替换为 #，这样:
    - -20 * ts_delta(..., 1) → -# * ts_delta(..., #)
    - -1 * ts_delta(..., 1)  → -# * ts_delta(..., #)
    - -60 * ts_delta(..., 1) → -# * ts_delta(..., #)
    三者结构签名相同，视为等价
    """
    # 先将所有数字（包括小数）替换为 #
    signature = re.sub(r'\b\d+\.?\d*\b', '#', expr)
    # 去掉空格
    signature = re.sub(r'\s+', '', signature)
    return signature


class GAFactorSearcher:
    """遗传算法因子搜索器"""

    def __init__(self, engine: AlphaEngine, returns: pd.Series, config=None):
        self.engine = engine
        self.returns = returns
        self.cfg = config or GA
        self.generator = ExpressionGenerator(engine, returns)
        self.history: List[Dict] = []
        self.best_expressions: List[Dict] = []
        self.all_evaluated: List[Dict] = []  # 记录所有评估过的因子

        # Hall of Fame: 记录进化过程中每代的最优个体，防止历史最优因丢失
        self._hof: Dict[str, float] = {}  # {expression: fitness}

        # 计算进化评估的日期范围（只用最近 N 个月数据）
        self._start_date, self._end_date = self._compute_evolution_date_range()
        logger.info(f"进化评估日期范围: {self._start_date} ~ {self._end_date}")

    def _compute_evolution_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """根据 evolution_data_months 配置，计算评估用的起止日期"""
        months = self.cfg.evolution_data_months
        if months <= 0:
            return None, None  # 不限制，用全部数据

        # 从 returns 的最大日期往回推 N 个月
        dates = self.returns.index.get_level_values("datetime")
        max_date = dates.max()
        start = max_date - pd.DateOffset(months=months)
        start_str = start.strftime("%Y-%m-%d")
        end_str = max_date.strftime("%Y-%m-%d")
        return start_str, end_str

    def search(self, existing_factors: Optional[List[pd.Series]] = None,
               seed_expressions: Optional[List[str]] = None) -> List[Dict]:
        """
        执行遗传算法搜索
        返回: Top因子列表 [{expression, ic, ir, icir, rank_ic, rank_icir, fitness}, ...]

        Parameters
        ----------
        existing_factors : list[pd.Series], optional
            已有因子列表，用于相关性惩罚（避免生成太相似的）
        seed_expressions : list[str], optional
            种子表达式列表，注入初始种群作为进化起点
        """
        if existing_factors:
            self.generator.set_existing_factors(existing_factors)

        n_seeds = 0
        if seed_expressions:
            # 过滤掉无效的种子（计算会失败的）
            valid_seeds = []
            for expr in seed_expressions:
                try:
                    self.engine.calculate(expr)
                    valid_seeds.append(expr)
                except Exception:
                    pass
            n_seeds = min(len(valid_seeds), self.cfg.population_size)
            logger.info(f"  种子表达式: {len(seed_expressions)} 条输入, {len(valid_seeds)} 条有效, 注入 {n_seeds} 条到初始种群")

        logger.info(f"开始遗传算法搜索: 种群={self.cfg.population_size}, 代数={self.cfg.n_generations}")
        logger.info(f"  评估数据范围: {self._start_date} ~ {self._end_date} "
                    f"({self.cfg.evolution_data_months}个月)")

        # 初始化种群：种子 + 种子的变异体（少随机）
        population = []
        # 先放入种子
        if seed_expressions and n_seeds > 0:
            import random as _rnd
            _rnd.shuffle(valid_seeds)
            population.extend(valid_seeds[:n_seeds])

            # 当种子数少（<=3）时，用种子的变异体填充大部分种群，而非完全随机
            # 这样保证进化是从种子出发，而不是被随机个体主导
            if n_seeds <= 3:
                n_mutants = max(0, int(self.cfg.population_size * 0.7) - n_seeds)
                logger.info(f"  种群策略: {n_seeds}条种子 + {n_mutants}个变异体 + {self.cfg.population_size - n_seeds - n_mutants}个随机")
                for i in range(n_mutants):
                    parent_seed = valid_seeds[_rnd.randint(0, n_seeds - 1)]
                    mutated = self.generator.mutate(parent_seed)
                    population.append(mutated)
                    if i < 3:  # 只打印前3个变异体
                        logger.info(f"    变异体{i+1}: {parent_seed[:40]} → {mutated[:60]}")
                # 剩余用随机填充
                while len(population) < self.cfg.population_size:
                    expr = self.generator.generate_random(max_depth=self.cfg.max_tree_depth)
                    population.append(expr)
            else:
                # 种子多时，直接用种子+随机
                while len(population) < self.cfg.population_size:
                    expr = self.generator.generate_random(max_depth=self.cfg.max_tree_depth)
                    population.append(expr)
        else:
            # 没有种子，完全随机
            while len(population) < self.cfg.population_size:
                expr = self.generator.generate_random(max_depth=self.cfg.max_tree_depth)
                population.append(expr)

        # 进化
        for gen in range(self.cfg.n_generations):
            gen_start_time = pd.Timestamp.now()

            # 评估
            scored = []
            for expr in population:
                _, fitness = self.generator.evaluate_expression(
                    expr, start_date=self._start_date, end_date=self._end_date)
                scored.append((expr, fitness))

            scored.sort(key=lambda x: x[1], reverse=True)

            # ---- 更新 Hall of Fame ----
            # HoF 现在保存完整信息字典：{"fitness": ..., "ic": ..., ...}
            for expr, fit in scored:
                if fit > -999:
                    # 检查是否需要更新 HoF（不存在或新的 fitness 更高）
                    if expr not in self._hof:
                        # 新因子，计算并保存完整指标
                        try:
                            factor = self.engine.calculate(expr)
                            metrics = _compute_detailed_metrics(
                                factor, self.returns,
                                start_date=self._start_date, end_date=self._end_date)
                            self._hof[expr] = {
                                "fitness": fit,
                                "ic": metrics["ic"],
                                "ir": metrics["ir"],
                                "icir": metrics["icir"],
                                "rank_ic": metrics["rank_ic"],
                                "rank_icir": metrics["rank_icir"],
                            }
                        except Exception:
                            self._hof[expr] = {"fitness": fit, "ic": 0, "ir": 0, "icir": 0, 
                                               "rank_ic": 0, "rank_icir": 0}
                    else:
                        # 已存在，比较 fitness（需要访问字典中的 fitness 字段）
                        old_fitness = self._hof[expr].get("fitness", -999)
                        if fit > old_fitness:
                            # 更新为更好的结果
                            try:
                                factor = self.engine.calculate(expr)
                                metrics = _compute_detailed_metrics(
                                    factor, self.returns,
                                    start_date=self._start_date, end_date=self._end_date)
                                self._hof[expr] = {
                                    "fitness": fit,
                                    "ic": metrics["ic"],
                                    "ir": metrics["ir"],
                                    "icir": metrics["icir"],
                                    "rank_ic": metrics["rank_ic"],
                                    "rank_icir": metrics["rank_icir"],
                                }
                            except Exception:
                                self._hof[expr] = {"fitness": fit, "ic": 0, "ir": 0, "icir": 0, 
                                                   "rank_ic": 0, "rank_icir": 0}

            # ---- 计算每代详细指标 ----
            valid_fitness = [f for _, f in scored if f > -999]
            best_expr, best_fit = scored[0]

            # 最优个体的详细指标
            best_metrics = {"ic": 0, "ir": 0, "icir": 0, "rank_ic": 0, "rank_icir": 0, "n_periods": 0}
            if best_fit > -999:
                try:
                    best_factor = self.engine.calculate(best_expr)
                    best_metrics = _compute_detailed_metrics(
                        best_factor, self.returns,
                        start_date=self._start_date, end_date=self._end_date)
                except Exception:
                    pass

            # 种群平均表现
            pop_avg_fitness = np.mean(valid_fitness) if valid_fitness else 0
            pop_valid_ratio = len(valid_fitness) / len(scored) if scored else 0

            # 记录历史
            history_record = {
                "generation": gen,
                "best_expression": best_expr,
                "best_fitness": best_fit,
                "best_ic": best_metrics["ic"],
                "best_ir": best_metrics["ir"],
                "best_icir": best_metrics["icir"],
                "best_rank_ic": best_metrics["rank_ic"],
                "best_rank_icir": best_metrics["rank_icir"],
                "pop_avg_fitness": pop_avg_fitness,
                "pop_valid_ratio": pop_valid_ratio,
            }
            self.history.append(history_record)

            # 每代都输出详细指标
            gen_elapsed = (pd.Timestamp.now() - gen_start_time).total_seconds()
            
            # 【新增】计算当前代的结构多样性
            current_structures = set()
            for expr, fit in scored:
                if fit > -999:
                    current_structures.add(_extract_structure_signature(expr))
            
            logger.info(
                f"  Gen {gen:3d} | "
                f"fitness={best_fit:+.4f} | "
                f"IC={best_metrics['ic']:+.4f} | "
                f"IR={best_metrics['ir']:+.4f} | "
                f"ICIR={best_metrics['icir']:+.4f} | "
                f"RankIC={best_metrics['rank_ic']:+.4f} | "
                f"RankICIR={best_metrics['rank_icir']:+.4f} | "
                f"avg_fit={pop_avg_fitness:.4f} | "
                f"valid={pop_valid_ratio:.0%} | "
                f"structures={len(current_structures)} | "
                f"{gen_elapsed:.1f}s | "
                f"{best_expr[:120]}"  # 增加到120个字符
            )

            # 选择 - 锦标赛选择
            elite_count = max(2, self.cfg.population_size // 10)
            new_pop = [expr for expr, _ in scored[:elite_count]]

            while len(new_pop) < self.cfg.population_size:
                if random.random() < self.cfg.crossover_prob:
                    p1 = self._tournament_select(scored)
                    p2 = self._tournament_select(scored)
                    c1, c2 = self.generator.crossover(p1, p2)
                    new_pop.append(c1)
                    if len(new_pop) < self.cfg.population_size:
                        new_pop.append(c2)
                else:
                    p = self._tournament_select(scored)
                    new_pop.append(self.generator.mutate(p))

            population = new_pop[:self.cfg.population_size]
            
            # 【优化13】每代结束后清理内存
            if gen % 5 == 0:  # 每5代清理一次
                gc.collect()

        # 最终评估：直接使用 HoF 中保存的结果，避免重新计算导致不一致
        # 合并最后一代种群（重新评估） + HoF 中的历史最优（直接使用已保存的指标）
        final_scored = []
        seen = set()
        seen_structures = {}  # {structure_signature: (expr, fitness, item)} 用于结构去重
        
        # 1. 重新评估最后一代种群
        for expr in population:
            if expr in seen:
                continue
            
            # 【新增】结构去重：检测仅系数不同的等价因子
            struct_sig = _extract_structure_signature(expr)
            if struct_sig in seen_structures:
                # 已存在相同结构的因子，比较fitness，保留更好的
                old_expr, old_fit, old_item = seen_structures[struct_sig]
                # 跳过这个，因为已经有一个更好的了
                continue
            
            seen.add(expr)
            
            try:
                factor, fitness = self.generator.evaluate_expression(
                    expr, start_date=self._start_date, end_date=self._end_date)
                
                if factor is not None and fitness > -999:
                    detailed = _compute_detailed_metrics(
                        factor, self.returns,
                        start_date=self._start_date, end_date=self._end_date)
                    item = {
                        "expression": expr,
                        "ic": detailed["ic"],
                        "ir": detailed["ir"],
                        "icir": detailed["icir"],
                        "rank_ic": detailed["rank_ic"],
                        "rank_icir": detailed["rank_icir"],
                        "fitness": fitness,
                    }
                    final_scored.append(item)
                    seen_structures[struct_sig] = (expr, fitness, item)
            except Exception:
                pass
        
        # 2. 添加 HoF 中的历史最优（使用已保存的指标，不重新计算）
        hof_added = 0
        for hof_expr, hof_data in self._hof.items():
            if hof_expr in seen:
                continue
            
            # 【新增】结构去重
            struct_sig = _extract_structure_signature(hof_expr)
            if struct_sig in seen_structures:
                # 已存在相同结构，比较fitness
                old_expr, old_fit, old_item = seen_structures[struct_sig]
                if hof_data.get("fitness", -999) <= old_fit:
                    continue  # 已有的更好，跳过
                else:
                    # 新的更好，替换旧的
                    final_scored = [x for x in final_scored if x.get("expression") != old_expr]
            
            seen.add(hof_expr)
            
            # 直接使用 HoF 中保存的指标
            item = {
                "expression": hof_expr,
                "ic": hof_data.get("ic", 0),
                "ir": hof_data.get("ir", 0),
                "icir": hof_data.get("icir", 0),
                "rank_ic": hof_data.get("rank_ic", 0),
                "rank_icir": hof_data.get("rank_icir", 0),
                "fitness": hof_data.get("fitness", -999),
            }
            final_scored.append(item)
            seen_structures[struct_sig] = (hof_expr, item["fitness"], item)
            hof_added += 1
        
        logger.info(f"最终评估: 种群{len(population)}个(重新评估) + HoF补充{hof_added}个(直接使用) = {len(final_scored)}个候选")
        logger.info(f"  结构去重: 从 {len(seen)} 个唯一表达式过滤到 {len(final_scored)} 个不同结构")

        final_scored.sort(key=lambda x: x["fitness"], reverse=True)
        self.best_expressions = final_scored[:20]
        
        # 保存所有评估过的因子
        self.all_evaluated = final_scored

        logger.info(f"搜索完成: 找到{len(final_scored)}个有效因子, Top5:")
        for i, item in enumerate(self.best_expressions[:5]):
            expr_display = item['expression'][:150]  # 显示150个字符
            logger.info(
                f"  #{i + 1}: fitness={item['fitness']:.4f}, "
                f"IC={item['ic']:+.4f}, IR={item['ir']:+.4f}, "
                f"ICIR={item['icir']:+.4f}, RankICIR={item['rank_icir']:+.4f}, "
                f"expr={expr_display}"
            )
            # 如果表达式被截断，再单独打印完整版本
            if len(item['expression']) > 150:
                logger.info(f"     完整公式: {item['expression']}")
        
        logger.info(f"  总共评估: {len(self.all_evaluated)}个唯一因子")
        logger.info(f"  历史代数: {len(self.history)}代")
        logger.info(f"  Hall of Fame: 累计{len(self._hof)}个唯一有效因子")

        return self.best_expressions

    def _tournament_select(self, scored, k=5):
        """锦标赛选择"""
        candidates = random.sample(scored, min(k, len(scored)))
        return max(candidates, key=lambda x: x[1])[0]
