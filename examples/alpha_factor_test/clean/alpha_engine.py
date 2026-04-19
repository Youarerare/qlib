"""
因子计算引擎 - 支持Alpha101/191公式自动解析与执行
"""
import re
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AlphaEngine:
    """Alpha因子计算引擎"""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV数据, MultiIndex(datetime, instrument)
        """
        self.df = df.copy()
        self._add_derived_fields()
        self._setup_operator_map()

    def _add_derived_fields(self):
        if "vwap" not in self.df.columns:
            self.df["vwap"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        if "returns" not in self.df.columns:
            self.df["returns"] = self.df.groupby(level="instrument")["close"].pct_change()
        for d in [5, 10, 15, 20, 30, 40, 50, 60, 80, 120, 150, 180]:
            col = f"adv{d}"
            if col not in self.df.columns:
                self.df[col] = self.df.groupby(level="instrument")["volume"].transform(
                    lambda x: x.rolling(d, min_periods=1).mean()
                )

    def _setup_operator_map(self):
        self.ts_ops = {
            "ts_sum": self.ts_sum, "ts_mean": self.ts_mean,
            "ts_std_dev": self.ts_std_dev, "ts_min": self.ts_min,
            "ts_max": self.ts_max, "ts_rank": self.ts_rank,
            "ts_delta": self.ts_delta, "ts_delay": self.ts_delay,
            "ts_corr": self.ts_corr, "ts_covariance": self.ts_covariance,
            "ts_scale": self.ts_scale, "ts_decay_linear": self.ts_decay_linear,
            "ts_arg_max": self.ts_arg_max, "ts_arg_min": self.ts_arg_min,
            "ts_product": self.ts_product, "ts_av_diff": self.ts_av_diff,
            "ts_zscore": self.ts_zscore, "ts_step": self.ts_step,
            "ts_regression": self.ts_regression,
        }
        self.cs_ops = {
            "rank": self.rank, "scale": self.scale, "cs_mean": self.cs_mean,
        }
        self.math_ops = {
            "abs": np.abs, "log": self._log, "sign": np.sign,
            "sqrt": self._sqrt, "signed_power": self._signed_power,
            "power": self._signed_power,  # power 别名
        }
        self.binary_ops = {
            "max": self._elem_max, "min": self._elem_min,
        }
        self.logic_ops = {
            "if_else": self.if_else,
        }

    def _get(self, name: str) -> pd.Series:
        if name in self.df.columns:
            return self.df[name]
        raise ValueError(f"字段不存在: {name}")

    # ===== 时间序列算子 =====
    def _ts_rolling(self, series, window, func):
        return series.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(window, min_periods=1).apply(func, raw=True) if callable(func) else getattr(x.rolling(window, min_periods=1), func)()
        )

    def ts_sum(self, s, w): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.rolling(w, min_periods=1).sum())
    def ts_mean(self, s, w): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.rolling(w, min_periods=1).mean())
    def ts_std_dev(self, s, w): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.rolling(w, min_periods=1).std())
    def ts_min(self, s, w): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.rolling(w, min_periods=1).min())
    def ts_max(self, s, w): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.rolling(w, min_periods=1).max())
    def ts_delta(self, s, d=1): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.diff(d))
    def ts_delay(self, s, d): return s.groupby(level="instrument", group_keys=False).transform(lambda x: x.shift(d))

    def ts_rank(self, s, w):
        def _rank(x):
            if len(x) <= 1: return 0.5
            return stats.percentileofscore(x[:-1], x[-1]) / 100.0
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(_rank, raw=False)
        )

    def ts_corr(self, s1, s2, w):
        # 逐 instrument 计算 rolling corr，再拼回与原始索引对齐的 Series
        parts = []
        for inst, idx in s1.groupby(level="instrument", group_keys=False).groups.items():
            a = s1.loc[idx]
            b = s2.loc[idx]
            corr = a.rolling(w, min_periods=2).corr(b)
            corr.index = idx  # 确保索引完全一致
            parts.append(corr)
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    def ts_covariance(self, s1, s2, w):
        # 逐 instrument 计算 rolling cov，再拼回与原始索引对齐的 Series
        parts = []
        for inst, idx in s1.groupby(level="instrument", group_keys=False).groups.items():
            a = s1.loc[idx]
            b = s2.loc[idx]
            cov = a.rolling(w, min_periods=2).cov(b)
            cov.index = idx  # 确保索引完全一致
            parts.append(cov)
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    def ts_scale(self, s, w):
        def _scale(x):
            mn, mx = x.min(), x.max()
            return (x[-1] - mn) / (mx - mn) if mx != mn else 0.5
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(_scale, raw=False)
        )

    def ts_decay_linear(self, s, w):
        weights = np.arange(1, w + 1, dtype=float)
        weights /= weights.sum()
        def _decay(x):
            n = len(x)
            if n < w:
                wt = np.arange(1, n + 1, dtype=float); wt /= wt.sum()
                return np.dot(x, wt)
            return np.dot(x, weights)
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(_decay, raw=True)
        )

    def ts_arg_max(self, s, w):
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(lambda v: np.argmax(v) + 1, raw=True)
        )

    def ts_arg_min(self, s, w):
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(lambda v: np.argmin(v) + 1, raw=True)
        )

    def ts_product(self, s, w):
        return s.groupby(level="instrument", group_keys=False).transform(
            lambda x: x.rolling(w, min_periods=1).apply(np.prod, raw=True)
        )

    def ts_av_diff(self, s, w):
        return s - self.ts_mean(s, w)

    def ts_zscore(self, s, w):
        mean = self.ts_mean(s, w)
        std = self.ts_std_dev(s, w)
        return (s - mean) / std.replace(0, np.nan)

    def ts_step(self, n):
        return pd.Series(np.ones(len(self.df)), index=self.df.index)

    def ts_regression(self, y, x, d, lag=0, rettype=0):
        """
        OLS滚动回归
        rettype: 0=残差, 1=alpha(截距), 2=beta(斜率), 3=y估计值,
                 4=SSE, 5=SST, 6=R², 7=MSE, 8=SE(beta), 9=SE(alpha)
        """
        rettype = int(rettype)
        lag = int(lag)

        def _regress(window_y, window_x):
            n = len(window_y)
            if n < 2:
                return np.nan
            x_arr = np.array(window_x, dtype=float)
            y_arr = np.array(window_y, dtype=float)
            valid = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isinf(x_arr) | np.isinf(y_arr))
            x_v = x_arr[valid]
            y_v = y_arr[valid]
            n_valid = len(x_v)
            if n_valid < 2:
                return np.nan
            x_mean = x_v.mean()
            y_mean = y_v.mean()
            ss_xy = np.sum((x_v - x_mean) * (y_v - y_mean))
            ss_xx = np.sum((x_v - x_mean) ** 2)
            if ss_xx == 0:
                return np.nan
            beta = ss_xy / ss_xx
            alpha = y_mean - beta * x_mean
            y_est = alpha + beta * x_v
            resid = y_v - y_est
            sse = np.sum(resid ** 2)
            sst = np.sum((y_v - y_mean) ** 2)
            if rettype == 0:
                return resid[-1]
            elif rettype == 1:
                return alpha
            elif rettype == 2:
                return beta
            elif rettype == 3:
                return alpha + beta * x_v[-1]
            elif rettype == 4:
                return sse
            elif rettype == 5:
                return sst
            elif rettype == 6:
                return 1 - sse / sst if sst > 0 else np.nan
            elif rettype == 7:
                return sse / (n_valid - 2) if n_valid > 2 else np.nan
            elif rettype == 8:
                mse = sse / (n_valid - 2) if n_valid > 2 else np.nan
                return np.sqrt(mse / ss_xx) if ss_xx > 0 and not np.isnan(mse) else np.nan
            elif rettype == 9:
                mse = sse / (n_valid - 2) if n_valid > 2 else np.nan
                se_alpha = np.sqrt(mse * (1.0 / n_valid + x_mean ** 2 / ss_xx)) if ss_xx > 0 and not np.isnan(mse) else np.nan
                return se_alpha
            return np.nan

        if lag > 0:
            x = x.groupby(level="instrument", group_keys=False).shift(lag)

        result = pd.Series(np.nan, index=y.index)
        for inst, group in y.groupby(level="instrument", group_keys=False):
            y_vals = group.values
            x_vals = x.loc[group.index].values
            res = []
            for i in range(len(y_vals)):
                start = max(0, i - d + 1)
                wy = y_vals[start:i + 1]
                wx = x_vals[start:i + 1]
                res.append(_regress(wy, wx))
            result.loc[group.index] = res
        return result

    # ===== 截面算子 =====
    def rank(self, s):
        return s.groupby(level="datetime", group_keys=False).rank(pct=True)

    def scale(self, s):
        def _scale(g):
            total = g.abs().sum()
            return g / total if total > 0 else g * 0
        return s.groupby(level="datetime", group_keys=False).transform(_scale)

    def cs_mean(self, s):
        return s.groupby(level="datetime", group_keys=False).transform("mean")

    # ===== 数学/逻辑算子 =====
    def _log(self, s): return np.log(s.clip(lower=1e-10))
    def _sqrt(self, s): return np.sqrt(s.clip(lower=0))
    
    def _power(self, s, e):
        """标准幂运算：直接调用 np.power(x, e)"""
        return np.power(s, e)
    
    def _signed_power(self, s, e):
        """带符号幂运算：sign(x) * |x|^e"""
        return np.sign(s) * np.power(np.abs(s), e)

    def _elem_max(self, a, b):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            if not a.index.equals(b.index):
                b = b.reindex(a.index)
        return pd.Series(np.maximum(a, b), index=self.df.index)
    def _elem_min(self, a, b):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            if not a.index.equals(b.index):
                b = b.reindex(a.index)
        return pd.Series(np.minimum(a, b), index=self.df.index)

    def if_else(self, cond, true_v, false_v):
        # 统一为 Series 并对齐到 df.index
        target_idx = self.df.index
        if isinstance(cond, pd.Series):
            if not cond.index.equals(target_idx):
                cond = cond.reindex(target_idx).fillna(False)
        else:
            cond = pd.Series(bool(cond), index=target_idx)

        if isinstance(true_v, pd.Series):
            if not true_v.index.equals(target_idx):
                true_v = true_v.reindex(target_idx)
        else:
            true_v = pd.Series(float(true_v), index=target_idx)

        if isinstance(false_v, pd.Series):
            if not false_v.index.equals(target_idx):
                false_v = false_v.reindex(target_idx)
        else:
            false_v = pd.Series(float(false_v), index=target_idx)

        return pd.Series(np.where(cond.fillna(False), true_v, false_v), index=target_idx)

    # ===== 公式自动解析与执行 =====
    def calculate(self, formula: str) -> pd.Series:
        """解析并计算公式"""
        processed = self._preprocess(formula)
        ns = self._build_namespace()
        try:
            if ";" in processed:
                parts = [p.strip() for p in processed.split(";") if p.strip()]
                code = "\n".join(parts)
                exec(code, {"__builtins__": {}}, ns)
                last = parts[-1]
                if "=" not in last:
                    result = eval(last, {"__builtins__": {}}, ns)
                else:
                    var = last.split("=")[0].strip()
                    result = ns.get(var, pd.Series(0.0, index=self.df.index))
            else:
                result = eval(processed, {"__builtins__": {}}, ns)

            # 确保返回 Series 且索引与 self.df 完全对齐
            if isinstance(result, pd.Series):
                if len(result) != len(self.df):
                    # 索引长度不一致，reindex 对齐
                    result = result.reindex(self.df.index)
                elif not result.index.equals(self.df.index):
                    # 长度一致但索引不完全相同，按索引对齐
                    result = result.reindex(self.df.index)
            else:
                # 标量结果，扩展为 Series
                result = pd.Series(float(result), index=self.df.index)

            return result
        except Exception as e:
            raise ValueError(f"公式执行失败: {formula[:80]}... | 错误: {e}")

    def _preprocess(self, formula: str) -> str:
        f = formula.strip().replace("\n", " ").replace("\r", " ")
        f = re.sub(r"\bifelse\b", "if_else", f)
        f = f.replace(".*", "*").replace("./", "/")
        f = f.replace("&&", " & ").replace("||", " | ")
        f = re.sub(r"\band\s*\(", "and_op(", f)
        f = re.sub(r"\bor\s*\(", "or_op(", f)

        # 移除行业中性化（保留内部表达式）
        for func in ["IndNeutralize", "group_neutralize"]:
            f = self._remove_neutralize(f, func)

        f = re.sub(r"IndClass\.industry", "'industry'", f)
        f = re.sub(r"IndClass\.sector", "'sector'", f)
        f = re.sub(r"\bcap\b", "1", f)
        f = re.sub(r"\bmarket\b", "1", f)
        f = re.sub(r"\bsector\b", "1", f)
        f = re.sub(r"\bindustry\b", "1", f)
        f = re.sub(r"\bsubindustry\b", "1", f)
        f = re.sub(r"group_mean\(([^,]+),\s*[^,]+,\s*[^)]+\)", r"cs_mean(\1)", f)
        return f

    def _remove_neutralize(self, text: str, func_name: str) -> str:
        result = text
        while True:
            idx = result.find(func_name)
            if idx == -1:
                break
            start = idx + len(func_name)
            while start < len(result) and result[start] in " \t":
                start += 1
            if start >= len(result) or result[start] != "(":
                break
            depth, end = 1, start + 1
            while end < len(result) and depth > 0:
                if result[end] == "(": depth += 1
                elif result[end] == ")": depth -= 1
                end += 1
            if depth != 0:
                break
            inner = result[start + 1:end - 1]
            comma = -1
            bd = 0
            for i, c in enumerate(inner):
                if c == "(": bd += 1
                elif c == ")": bd -= 1
                elif c == "," and bd == 0:
                    comma = i
                    break
            first_arg = inner[:comma].strip() if comma != -1 else inner.strip()
            result = result[:idx] + first_arg + result[end:]
        return result

    def _build_namespace(self) -> dict:
        ns = {}
        ns.update(self.ts_ops)
        ns.update(self.cs_ops)
        ns.update(self.math_ops)
        ns.update(self.binary_ops)
        ns.update(self.logic_ops)
        ns["and_op"] = lambda a, b: a & b
        ns["or_op"] = lambda a, b: a | b
        
        # 添加二元算子（支持 open - close 等表达式）
        ns["subtract"] = lambda a, b: a - b
        ns["divide"] = lambda a, b: a / b.replace(0, np.nan)
        ns["multiply"] = lambda a, b: a * b
        ns["add"] = lambda a, b: a + b
        
        ns["power"] = self._signed_power

        for field in list(self.df.columns):
            try:
                ns[field] = self.df[field]
            except Exception:
                pass

        ns["np"] = np
        ns["nan"] = np.nan
        ns["inf"] = np.inf
        ns["True"] = True
        ns["False"] = False
        ns["true"] = True
        ns["false"] = False
        return ns


def compute_factors(
    df: pd.DataFrame,
    formulas: Dict[str, str],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    批量计算因子
    返回: DataFrame, 每列一个因子
    """
    engine = AlphaEngine(df)
    results = {}
    failed = []

    items = list(formulas.items())
    for i, (name, formula) in enumerate(items):
        try:
            results[name] = engine.calculate(formula)
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"  进度: {i + 1}/{len(items)}")
        except Exception as e:
            failed.append({"name": name, "error": str(e)[:100]})
            logger.debug(f"  失败: {name} - {e}")

    feature_df = pd.DataFrame(results)
    logger.info(f"因子计算完成: 成功={len(results)}, 失败={len(failed)}")

    if failed:
        import pandas as _pd
        fail_df = _pd.DataFrame(failed)
        logger.warning(f"失败因子列表:\n{fail_df.to_string()}")

    return feature_df
