"""
Alpha因子计算引擎
基于Qlib框架计算alpha101和alpha191因子

支持两种方式：
1. 手动调用计算函数
2. 自动解析公式字符串执行
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Callable
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')


class AlphaCalculator:
    """Alpha因子计算器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化计算器
        
        Parameters
        ----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame，MultiIndex (datetime, instrument)
            需要的列: open, high, low, close, volume, vwap
        """
        self.df = df.copy()
        self._add_derived_fields()
        self._cache = {}  # 缓存计算结果
    
    def _add_derived_fields(self):
        """添加衍生字段"""
        if 'vwap' not in self.df.columns:
            self.df['vwap'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df.groupby(level='instrument')['close'].pct_change()
        
        # 计算adv20 (20日平均成交量)
        self.df['adv20'] = self.df.groupby(level='instrument')['volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        
        # 计算更多平均成交量
        for days in [5, 10, 15, 30, 40, 50, 60, 80, 120, 150, 180]:
            self.df[f'adv{days}'] = self.df.groupby(level='instrument')['volume'].transform(
                lambda x: x.rolling(days, min_periods=1).mean()
            )
    
    def _get_series(self, name: str) -> pd.Series:
        """获取指定列的Series"""
        if name in self.df.columns:
            return self.df[name]
        raise ValueError(f"Column {name} not found in data")
    
    # ====== 基础操作函数 ======
    
    def ts_sum(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列求和"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    
    def ts_mean(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列均值"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    def ts_std_dev(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列标准差"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    def ts_min(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列最小值"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
    
    def ts_max(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列最大值"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    
    def ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列排名 - 返回当前值在窗口内的百分位排名"""
        def rank_func(x):
            if len(x) <= 1:
                return 0.5
            return stats.percentileofscore(x[:-1], x[-1]) / 100.0
        
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(rank_func, raw=False)
        )
    
    def ts_delta(self, series: pd.Series, period: int = 1) -> pd.Series:
        """时间序列差分"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.diff(period)
        )
    
    def ts_delay(self, series: pd.Series, period: int) -> pd.Series:
        """时间序列滞后"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda x: x.shift(period)
        )
    
    def ts_corr(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """时间序列相关系数"""
        def calc_corr(group):
            s1 = series1.loc[group.index]
            s2 = series2.loc[group.index]
            return s1.rolling(window, min_periods=2).corr(s2)
        
        return series1.groupby(level='instrument', group_keys=False).apply(calc_corr)
    
    def ts_covariance(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """时间序列协方差"""
        def calc_cov(group):
            s1 = series1.loc[group.index]
            s2 = series2.loc[group.index]
            return s1.rolling(window, min_periods=2).cov(s2)
        
        return series1.groupby(level='instrument', group_keys=False).apply(calc_cov)
    
    def rank(self, series: pd.Series) -> pd.Series:
        """横截面排名 - 按日期对所有股票排名"""
        return series.groupby(level='datetime', group_keys=False).rank(pct=True)
    
    def scale(self, series: pd.Series) -> pd.Series:
        """横截面缩放 - 按日期标准化到和为1"""
        def scale_group(x):
            x_abs = x.abs()
            s = x_abs.sum()
            if s == 0:
                return x * 0
            return x / s
        return series.groupby(level='datetime', group_keys=False).transform(scale_group)
    
    def ts_scale(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列缩放 - 在窗口内缩放到[0,1]"""
        def scale_func(x):
            if len(x) == 0:
                return np.nan
            min_val = x.min()
            max_val = x.max()
            if max_val == min_val:
                return 0.5
            return (x[-1] - min_val) / (max_val - min_val)
        
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(scale_func, raw=False)
        )
    
    def ts_decay_linear(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列线性衰减加权平均"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        def decay_func(x):
            n = len(x)
            if n < window:
                w = np.arange(1, n + 1)
                w = w / w.sum()
                return np.dot(x, w)
            return np.dot(x, weights)
        
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(decay_func, raw=True)
        )
    
    def ts_arg_max(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列最大值位置"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(lambda x: np.argmax(x) + 1, raw=True)
        )
    
    def ts_arg_min(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列最小值位置"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(lambda x: np.argmin(x) + 1, raw=True)
        )
    
    def ts_product(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列乘积"""
        return series.groupby(level='instrument', group_keys=False).transform(
            lambda s: s.rolling(window, min_periods=1).apply(np.prod, raw=True)
        )
    
    def ts_regression(self, series: pd.Series, independent: pd.Series, window: int, 
                     lag: int = 0, rettype: int = 2) -> pd.Series:
        """
        时间序列回归
        rettype: 0=截距, 1=斜率, 2=残差
        """
        def calc_regression(group):
            y = series.loc[group.index]
            X = independent.loc[group.index]
            if lag > 0:
                X = X.shift(lag)
            
            def reg_func(window_data):
                idx = window_data.index
                y_window = y.loc[idx]
                x_window = X.loc[idx]
                
                valid = ~(y_window.isna() | x_window.isna())
                if valid.sum() < 2:
                    return np.nan
                
                y_valid = y_window[valid]
                x_valid = x_window[valid]
                
                slope, intercept, _, _, _ = stats.linregress(x_valid, y_valid)
                
                if rettype == 2:
                    return y_valid.iloc[-1] - (slope * x_valid.iloc[-1] + intercept)
                elif rettype == 1:
                    return slope
                else:
                    return intercept
            
            return y.rolling(window, min_periods=2).apply(reg_func, raw=False)
        
        return series.groupby(level='instrument', group_keys=False).apply(calc_regression)
    
    def ts_step(self, window: int = 1) -> pd.Series:
        """时间序列步进"""
        result = pd.Series(index=self.df.index, dtype=float)
        for instrument in self.df.index.get_level_values('instrument').unique():
            mask = self.df.index.get_level_values('instrument') == instrument
            result[mask] = np.arange(mask.sum())
        return result
    
    def ts_av_diff(self, series: pd.Series, window: int) -> pd.Series:
        """时间序列与均值的差"""
        ma = self.ts_mean(series, window)
        return series - ma
    
    # ====== 逻辑操作 ======
    
    def if_else(self, condition: pd.Series, true_val: pd.Series, false_val: pd.Series) -> pd.Series:
        """条件选择"""
        if not isinstance(condition, pd.Series):
            condition = pd.Series(condition, index=self.df.index)
        if not isinstance(true_val, pd.Series):
            true_val = pd.Series(true_val, index=self.df.index)
        if not isinstance(false_val, pd.Series):
            false_val = pd.Series(false_val, index=self.df.index)
        
        return pd.Series(np.where(condition.fillna(False), true_val, false_val), index=self.df.index)
    
    def sign(self, series: pd.Series) -> pd.Series:
        """符号函数"""
        return np.sign(series)
    
    def abs(self, series: pd.Series) -> pd.Series:
        """绝对值"""
        return np.abs(series)
    
    def log(self, series: pd.Series) -> pd.Series:
        """自然对数"""
        return np.log(series.clip(lower=1e-10))
    
    def power(self, series: pd.Series, exponent) -> pd.Series:
        """幂函数"""
        if isinstance(exponent, pd.Series):
            # 处理序列指数
            result = np.sign(series) * np.power(series.abs().fillna(0), exponent.abs())
            return result
        return np.power(series, exponent)
    
    def max(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """两个序列逐元素取最大值"""
        return pd.Series(np.maximum(series1, series2), index=self.df.index)
    
    def min(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """两个序列逐元素取最小值"""
        return pd.Series(np.minimum(series1, series2), index=self.df.index)
    
    def max_rolling(self, series: pd.Series, window: int) -> pd.Series:
        """滚动窗口最大值"""
        return self.ts_max(series, window)
    
    def min_rolling(self, series: pd.Series, window: int) -> pd.Series:
        """滚动窗口最小值"""
        return self.ts_min(series, window)
    
    def and_op(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """逻辑与"""
        return series1 & series2
    
    def or_op(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """逻辑或"""
        return series1 | series2
    
    def sqrt(self, series: pd.Series) -> pd.Series:
        """平方根"""
        return np.sqrt(series.clip(lower=0))
    
    def subtract(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """减法"""
        return series1 - series2
    
    def divide(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """除法"""
        return series1 / series2.replace(0, np.nan)
    
    # ====== 行业中性化函数 ======
    
    def indneutralize(self, series: pd.Series, group: str = 'industry') -> pd.Series:
        """
        行业中性化
        将因子值减去同行业内的均值，实现行业中性化
        
        Parameters
        ----------
        series : pd.Series
            需要中性化的因子值
        group : str
            分组字段，默认为 'industry'
        """
        if group not in self.df.columns:
            # 如果没有行业数据，返回原始序列
            return series
        
        # 获取行业分组
        industry = self.df[group]
        
        # 计算每个日期每个行业的均值
        def neutralize_group(g):
            # g 是一个分组，包含同一日期同一行业的所有股票
            return g - g.mean()
        
        # 按日期和行业分组，进行中性化
        result = series.groupby([series.index.get_level_values('datetime'), industry]).transform(neutralize_group)
        return result
    
    def group_neutralize(self, series: pd.Series, group_field: str = 'industry') -> pd.Series:
        """group_neutralize 别名"""
        return self.indneutralize(series, group_field)
    
    def IndNeutralize(self, series: pd.Series, group_field: str = 'industry') -> pd.Series:
        """IndNeutralize 别名（大小写兼容）"""
        return self.indneutralize(series, group_field)


class AlphaFormulaParser:
    """
    Alpha公式自动解析器
    将公式字符串解析为可执行的Python代码
    """
    
    def __init__(self, calc: AlphaCalculator):
        self.calc = calc
        self._setup_operators()
    
    def _setup_operators(self):
        """设置运算符映射"""
        self.function_map = {
            # 时间序列函数
            'ts_sum': self.calc.ts_sum,
            'ts_mean': self.calc.ts_mean,
            'ts_std_dev': self.calc.ts_std_dev,
            'ts_min': self.calc.ts_min,
            'ts_max': self.calc.ts_max,
            'ts_rank': self.calc.ts_rank,
            'ts_delta': self.calc.ts_delta,
            'ts_delay': self.calc.ts_delay,
            'ts_corr': self.calc.ts_corr,
            'ts_covariance': self.calc.ts_covariance,
            'ts_scale': self.calc.ts_scale,
            'ts_decay_linear': self.calc.ts_decay_linear,
            'ts_arg_max': self.calc.ts_arg_max,
            'ts_arg_min': self.calc.ts_arg_min,
            'ts_product': self.calc.ts_product,
            'ts_regression': self.calc.ts_regression,
            'ts_step': self.calc.ts_step,
            'ts_av_diff': self.calc.ts_av_diff,
            
            # 横截面函数
            'rank': self.calc.rank,
            'scale': self.calc.scale,
            
            # 数学函数
            'abs': self.calc.abs,
            'log': self.calc.log,
            'sign': self.calc.sign,
            'power': self.calc.power,
            'sqrt': self.calc.sqrt,
            'subtract': self.calc.subtract,
            'divide': self.calc.divide,
            
            # 逻辑函数
            'if_else': self.calc.if_else,
            'max': self.calc.max,
            'min': self.calc.min,
            'min_rolling': self.calc.min_rolling,
            'max_rolling': self.calc.max_rolling,
            'and': self.calc.and_op,
            'or': self.calc.or_op,
            'and_op': self.calc.and_op,
            'or_op': self.calc.or_op,
            
            # 行业中性化函数
            'indneutralize': self.calc.indneutralize,
            'IndNeutralize': self.calc.IndNeutralize,
            'group_neutralize': self.calc.group_neutralize,
        }
        
        # 数据字段 - 动态获取
        self.data_fields = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 'industry']
        # 添加adv字段
        for days in [5, 10, 15, 20, 30, 40, 50, 60, 80, 120, 150, 180]:
            self.data_fields.append(f'adv{days}')
    
    def parse_and_calculate(self, formula: str) -> pd.Series:
        """解析公式字符串并计算结果"""
        processed = self._preprocess_formula(formula)
        namespace = self._build_namespace()
        
        try:
            # 处理多行公式（带分号和赋值）
            if ';' in processed:
                # 将分号替换为换行符，并去除每行前后空格
                parts = [p.strip() for p in processed.split(';') if p.strip()]
                code = '\n'.join(parts)
                # 执行多行代码
                exec(code, {"__builtins__": {}}, namespace)
                # 获取最后一个表达式的结果
                last_expr = parts[-1]
                if '=' not in last_expr:
                    result = eval(last_expr, {"__builtins__": {}}, namespace)
                else:
                    # 取最后一个赋值的变量
                    var_name = last_expr.split('=')[0].strip()
                    result = namespace.get(var_name)
                    if result is None:
                        result = pd.Series(0.0, index=self.calc.df.index)
                return result
            else:
                result = eval(processed, {"__builtins__": {}}, namespace)
                return result
        except Exception as e:
            import traceback
            raise ValueError(f"公式解析错误: {formula[:100]}...\n处理后: {processed[:100]}...\n错误信息: {str(e)}\n{traceback.format_exc()}")
    
    def _preprocess_formula(self, formula: str) -> str:
        """预处理公式字符串"""
        formula = formula.strip()
        
        # 移除换行符，保持公式在一行
        formula = formula.replace('\n', ' ').replace('\r', ' ')
        
        # 处理 ifelse -> if_else
        formula = re.sub(r'\bifelse\b', 'if_else', formula)
        
        # 处理 ./ 和 .* 操作符 (MATLAB风格)
        formula = formula.replace('.*', '*')
        formula = formula.replace('./', '/')
        
        # 处理 & 和 | 逻辑运算符
        formula = formula.replace('&&', ' & ')
        formula = formula.replace('||', ' | ')
        
        # 处理 and(...) 和 or(...) 函数调用
        formula = re.sub(r'\band\s*\(', 'and_op(', formula)
        formula = re.sub(r'\bor\s*\(', 'or_op(', formula)
        
        # 处理比较表达式与 & 和 | 的组合
        # 使用更通用的方法：找出所有比较表达式并用括号包裹
        # 比较 运算符: >, <, >=, <=, ==, !=
        
        # 递归处理括号内的表达式
        def process_comparison(expr):
            """处理比较表达式，为 & 和 | 添加括号"""
            # 找到所有括号对，先处理括号内的
            result = []
            i = 0
            while i < len(expr):
                if expr[i] == '(':
                    # 找到匹配的右括号
                    depth = 1
                    j = i + 1
                    while j < len(expr) and depth > 0:
                        if expr[j] == '(':
                            depth += 1
                        elif expr[j] == ')':
                            depth -= 1
                        j += 1
                    # 递归处理括号内的内容
                    inner = expr[i+1:j-1]
                    result.append('(' + process_comparison(inner) + ')')
                    i = j
                else:
                    result.append(expr[i])
                    i += 1
            
            expr = ''.join(result)
            
            # 处理比较表达式中的 & 和 |
            # 模式: 比较表达式 & 或 | 比较表达式
            # 比较表达式: 非括号内容 + 比较运算符 + 非括号内容
            # 简化处理：匹配形如 "xxx>yyy & zzz>www" 的模式
            
            # 使用循环处理多个 & 和 |
            changed = True
            while changed:
                changed = False
                # 匹配: ) 比较 运算符 ... & ... 比较 运算符 (
                # 或: 开头 比较 运算符 ... & ... 比较 运算符 结尾
                # 或复杂的函数调用形式
                
                # 更简单的方法：找到 & 或 |，然后向两边扩展直到找到比较运算符
                match = re.search(r'([^()]+(?:>|<|>=|<=|==|!=)[^()]+)\s*(&|\|)\s*([^()]+(?:>|<|>=|<=|==|!=)[^()]+)', expr)
                if match:
                    left = match.group(1)
                    op = match.group(2)
                    right = match.group(3)
                    # 如果 left 或 right 已经被括号包裹，不处理
                    if not (left.strip().startswith('(') and left.strip().endswith(')')):
                        new_left = '(' + left + ')'
                    else:
                        new_left = left
                    if not (right.strip().startswith('(') and right.strip().endswith(')')):
                        new_right = '(' + right + ')'
                    else:
                        new_right = right
                    new_expr = new_left + ' ' + op + ' ' + new_right
                    expr = expr[:match.start()] + new_expr + expr[match.end():]
                    changed = True
            
            return expr
        
        try:
            formula = process_comparison(formula)
        except:
            pass  # 如果处理失败，继续使用原始公式
        
        # 处理 group_mean - 需要市值数据，暂不支持，返回0
        formula = re.sub(r'group_mean\([^)]+\)', '0', formula)
        
        # 处理 IndNeutralize 和 group_neutralize - 去掉中性化，直接返回原始值
        # 使用简单方法：找到函数名，然后找到匹配的括号对
        def remove_neutralize_func(text, func_name):
            result = text
            while True:
                idx = result.find(func_name)
                if idx == -1:
                    break
                # 找到函数名后的左括号
                start = idx + len(func_name)
                while start < len(result) and result[start] in ' \t':
                    start += 1
                if start >= len(result) or result[start] != '(':
                    break
                # 找到匹配的右括号
                depth = 1
                end = start + 1
                while end < len(result) and depth > 0:
                    if result[end] == '(':
                        depth += 1
                    elif result[end] == ')':
                        depth -= 1
                    end += 1
                if depth != 0:
                    break
                # 提取括号内的内容
                inner = result[start+1:end-1]
                # 找到第一个逗号（不在嵌套括号内的）
                comma_idx = -1
                bracket_depth = 0
                for i, c in enumerate(inner):
                    if c == '(':
                        bracket_depth += 1
                    elif c == ')':
                        bracket_depth -= 1
                    elif c == ',' and bracket_depth == 0:
                        comma_idx = i
                        break
                if comma_idx != -1:
                    first_arg = inner[:comma_idx].strip()
                else:
                    first_arg = inner.strip()
                result = result[:idx] + first_arg + result[end:]
            return result
        
        formula = remove_neutralize_func(formula, 'IndNeutralize')
        formula = remove_neutralize_func(formula, 'group_neutralize')
        
        # 处理 IndClass.industry, IndClass.sector 等行业分类参数
        formula = re.sub(r'IndClass\.industry', "'industry'", formula)
        formula = re.sub(r'IndClass\.sector', "'sector'", formula)
        formula = re.sub(r'IndClass\.subindustry', "'subindustry'", formula)
        
        # 处理 cap, market, sector, industry, subindustry 等不存在的字段
        formula = re.sub(r'\bcap\b', '1', formula)
        formula = re.sub(r'\bmarket\b', '1', formula)
        formula = re.sub(r'\bsector\b', '1', formula)
        formula = re.sub(r'\bindustry\b', '1', formula)
        formula = re.sub(r'\bsubindustry\b', '1', formula)
        
        return formula
    
    def _build_namespace(self) -> dict:
        """构建公式执行的命名空间"""
        namespace = {}
        
        # 添加函数
        namespace.update(self.function_map)
        
        # 添加数据字段
        for field in self.data_fields:
            try:
                namespace[field] = self.calc._get_series(field)
            except ValueError:
                pass  # 忽略不存在的字段
        
        # 添加常量
        namespace['np'] = np
        namespace['nan'] = np.nan
        namespace['inf'] = np.inf
        namespace['True'] = True
        namespace['False'] = False
        namespace['true'] = True
        namespace['false'] = False
        
        return namespace


class AlphaCalculatorAuto(AlphaCalculator):
    """支持自动解析公式的Alpha计算器"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.formula_parser = AlphaFormulaParser(self)
    
    def calculate_alpha(self, formula: str) -> pd.Series:
        """计算给定的alpha公式"""
        return self.formula_parser.parse_and_calculate(formula)


# ====== Alpha001-Alpha020 手动实现 ======

def calculate_alpha_001(calc: AlphaCalculator) -> pd.Series:
    """alpha001: (-1*ts_corr(rank(ts_delta(log(volume),1)),rank(((close-open)/open)),6))"""
    volume = calc._get_series('volume')
    close = calc._get_series('close')
    open_price = calc._get_series('open')
    
    log_volume = calc.log(volume)
    delta_log_volume = calc.ts_delta(log_volume, 1)
    rank1 = calc.rank(delta_log_volume)
    
    close_open_ratio = (close - open_price) / open_price
    rank2 = calc.rank(close_open_ratio)
    
    corr = calc.ts_corr(rank1, rank2, 6)
    return -1 * corr


def calculate_alpha_002(calc: AlphaCalculator) -> pd.Series:
    """alpha002: -1 * ts_delta(((close - low) - (high - close)) / (high - low), 1)"""
    close = calc._get_series('close')
    high = calc._get_series('high')
    low = calc._get_series('low')
    
    numerator = (close - low) - (high - close)
    denominator = high - low
    ratio = numerator / denominator.replace(0, np.nan)
    delta = calc.ts_delta(ratio, 1)
    return -1 * delta


def calculate_alpha_003(calc: AlphaCalculator) -> pd.Series:
    """alpha003: ts_sum(if_else(close == ts_delay(close, 1), 0, close - if_else(close > ts_delay(close, 1), min(low, ts_delay(close, 1)), max(high, ts_delay(close, 1)))), 6)"""
    close = calc._get_series('close')
    high = calc._get_series('high')
    low = calc._get_series('low')
    
    close_delayed = calc.ts_delay(close, 1)
    cond1 = close == close_delayed
    cond2 = close > close_delayed
    
    min_low_close = calc.min(low, close_delayed)
    max_high_close = calc.max(high, close_delayed)
    inner_if = calc.if_else(cond2, min_low_close, max_high_close)
    value = close - inner_if
    result = calc.if_else(cond1, pd.Series(0, index=calc.df.index), value)
    
    return calc.ts_sum(result, 6)


def calculate_alpha_004(calc: AlphaCalculator) -> pd.Series:
    """alpha004: 复杂条件判断"""
    close = calc._get_series('close')
    volume = calc._get_series('volume')
    adv20 = calc._get_series('adv20')
    
    mean_8 = calc.ts_mean(close, 8)
    std_8 = calc.ts_std_dev(close, 8)
    mean_2 = calc.ts_mean(close, 2)
    
    cond1 = (mean_8 + std_8) < mean_2
    cond2 = mean_2 < (mean_8 - std_8)
    vol_ratio = volume / adv20.replace(0, np.nan)
    cond3 = (1 < vol_ratio) | (vol_ratio == 1)
    
    result = calc.if_else(cond1, pd.Series(-1.0, index=calc.df.index),
                          calc.if_else(cond2, pd.Series(1.0, index=calc.df.index),
                                       calc.if_else(cond3, pd.Series(1.0, index=calc.df.index),
                                                    pd.Series(-1.0, index=calc.df.index))))
    return result


def calculate_alpha_005(calc: AlphaCalculator) -> pd.Series:
    """alpha005: -ts_rank(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""
    volume = calc._get_series('volume')
    high = calc._get_series('high')
    
    rank_vol = calc.ts_rank(volume, 5)
    rank_high = calc.ts_rank(high, 5)
    corr = calc.ts_corr(rank_vol, rank_high, 5)
    return -calc.ts_rank(corr, 3)


def calculate_alpha_006(calc: AlphaCalculator) -> pd.Series:
    """alpha006: (rank(sign(ts_delta((((open*0.85)+(high*0.15))),4)))*-1)"""
    open_price = calc._get_series('open')
    high = calc._get_series('high')
    
    weighted = open_price * 0.85 + high * 0.15
    delta = calc.ts_delta(weighted, 4)
    sign_delta = calc.sign(delta)
    return -calc.rank(sign_delta)


def calculate_alpha_007(calc: AlphaCalculator) -> pd.Series:
    """alpha007: ((rank(max((vwap-close),3))+rank(min((vwap-close),3)))*rank(ts_delta(volume,3)))"""
    vwap = calc._get_series('vwap')
    close = calc._get_series('close')
    volume = calc._get_series('volume')
    
    vwap_close = vwap - close
    max_3 = calc.ts_max(vwap_close, 3)
    min_3 = calc.ts_min(vwap_close, 3)
    
    return (calc.rank(max_3) + calc.rank(min_3)) * calc.rank(calc.ts_delta(volume, 3))


def calculate_alpha_008(calc: AlphaCalculator) -> pd.Series:
    """alpha008: rank(ts_delta(((((high+low)/2)*0.2)+(vwap*0.8)),4)*-1)"""
    high = calc._get_series('high')
    low = calc._get_series('low')
    vwap = calc._get_series('vwap')
    
    weighted = (high + low) / 2 * 0.2 + vwap * 0.8
    return calc.rank(-calc.ts_delta(weighted, 4))


def calculate_alpha_009(calc: AlphaCalculator) -> pd.Series:
    """alpha009: ts_mean((((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume), 7)"""
    high = calc._get_series('high')
    low = calc._get_series('low')
    volume = calc._get_series('volume')
    
    hl_mean = (high + low) / 2
    hl_mean_delayed = (calc.ts_delay(high, 1) + calc.ts_delay(low, 1)) / 2
    diff = hl_mean - hl_mean_delayed
    ratio = diff * (high - low) / volume.replace(0, np.nan)
    return calc.ts_mean(ratio, 7)


def calculate_alpha_010(calc: AlphaCalculator) -> pd.Series:
    """alpha010: rank(power(if_else(returns < 0, ts_std_dev(returns, 20), close), 2))"""
    close = calc._get_series('close')
    returns = calc._get_series('returns')
    
    cond = returns < 0
    std_returns = calc.ts_std_dev(returns, 20)
    value = calc.if_else(cond, std_returns, close)
    return calc.rank(calc.power(value, 2))


def calculate_alpha_011(calc: AlphaCalculator) -> pd.Series:
    """alpha011: (ts_sum(((close-low)-(high-close))/(high-low)*volume,6))"""
    close = calc._get_series('close')
    high = calc._get_series('high')
    low = calc._get_series('low')
    volume = calc._get_series('volume')
    
    ratio = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return calc.ts_sum(ratio * volume, 6)


def calculate_alpha_012(calc: AlphaCalculator) -> pd.Series:
    """alpha012: (rank((open-(ts_sum(vwap,10)/10))))*(-1*(rank(abs((close-vwap)))))"""
    open_price = calc._get_series('open')
    close = calc._get_series('close')
    vwap = calc._get_series('vwap')
    
    return calc.rank(open_price - calc.ts_sum(vwap, 10) / 10) * (-calc.rank(calc.abs(close - vwap)))


def calculate_alpha_013(calc: AlphaCalculator) -> pd.Series:
    """alpha013: ((power((high*low), 0.5))-vwap)"""
    high = calc._get_series('high')
    low = calc._get_series('low')
    vwap = calc._get_series('vwap')
    
    return calc.sqrt(high * low) - vwap


def calculate_alpha_014(calc: AlphaCalculator) -> pd.Series:
    """alpha014: close-ts_delay(close,5)"""
    close = calc._get_series('close')
    return close - calc.ts_delay(close, 5)


def calculate_alpha_015(calc: AlphaCalculator) -> pd.Series:
    """alpha015: open/ts_delay(close,1)-1"""
    open_price = calc._get_series('open')
    close = calc._get_series('close')
    return open_price / calc.ts_delay(close, 1).replace(0, np.nan) - 1


def calculate_alpha_016(calc: AlphaCalculator) -> pd.Series:
    """alpha016: -ts_rank(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)"""
    volume = calc._get_series('volume')
    vwap = calc._get_series('vwap')
    
    corr = calc.ts_corr(calc.rank(volume), calc.rank(vwap), 5)
    return -calc.ts_rank(calc.rank(corr), 5)


def calculate_alpha_017(calc: AlphaCalculator) -> pd.Series:
    """alpha017: power(rank((vwap-max(vwap,15))), ts_delta(close,5))"""
    vwap = calc._get_series('vwap')
    close = calc._get_series('close')
    
    rank_diff = calc.rank(vwap - calc.ts_max(vwap, 15))
    delta_close = calc.ts_delta(close, 5)
    return calc.power(rank_diff, delta_close)


def calculate_alpha_018(calc: AlphaCalculator) -> pd.Series:
    """alpha018: close/ts_delay(close,5)"""
    close = calc._get_series('close')
    return close / calc.ts_delay(close, 5).replace(0, np.nan)


def calculate_alpha_019(calc: AlphaCalculator) -> pd.Series:
    """alpha019: if_else条件判断"""
    close = calc._get_series('close')
    close_delayed = calc.ts_delay(close, 5)
    
    cond1 = close < close_delayed
    cond2 = close == close_delayed
    
    value1 = (close - close_delayed) / close_delayed.replace(0, np.nan)
    value3 = (close - close_delayed) / close.replace(0, np.nan)
    
    return calc.if_else(cond1, value1, calc.if_else(cond2, pd.Series(0.0, index=calc.df.index), value3))


def calculate_alpha_020(calc: AlphaCalculator) -> pd.Series:
    """alpha020: (close-ts_delay(close,6))/ts_delay(close,6)*100"""
    close = calc._get_series('close')
    close_delayed = calc.ts_delay(close, 6)
    return (close - close_delayed) / close_delayed.replace(0, np.nan) * 100


# Alpha名称到计算函数的映射
ALPHA_FUNCTIONS = {
    'alpha001': calculate_alpha_001,
    'alpha002': calculate_alpha_002,
    'alpha003': calculate_alpha_003,
    'alpha004': calculate_alpha_004,
    'alpha005': calculate_alpha_005,
    'alpha006': calculate_alpha_006,
    'alpha007': calculate_alpha_007,
    'alpha008': calculate_alpha_008,
    'alpha009': calculate_alpha_009,
    'alpha010': calculate_alpha_010,
    'alpha011': calculate_alpha_011,
    'alpha012': calculate_alpha_012,
    'alpha013': calculate_alpha_013,
    'alpha014': calculate_alpha_014,
    'alpha015': calculate_alpha_015,
    'alpha016': calculate_alpha_016,
    'alpha017': calculate_alpha_017,
    'alpha018': calculate_alpha_018,
    'alpha019': calculate_alpha_019,
    'alpha020': calculate_alpha_020,
}


def calculate_alpha_by_name(calc: AlphaCalculator, alpha_name: str) -> pd.Series:
    """根据名称计算alpha因子"""
    if alpha_name in ALPHA_FUNCTIONS:
        return ALPHA_FUNCTIONS[alpha_name](calc)
    else:
        raise NotImplementedError(f"Alpha {alpha_name} 未实现")


if __name__ == "__main__":
    print("Alpha计算器模块加载成功")
    print(f"已手动实现的alpha数量: {len(ALPHA_FUNCTIONS)}")
