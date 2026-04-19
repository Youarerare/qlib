"""
单条公式遗传算法搜索脚本

功能：
1. 输入单条Alpha公式
2. 基于该公式进行变异和进化
3. 搜索相似结构的最优因子
"""
import sys
import os
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean.data_manager import load_ohlcv
from clean.alpha_engine import AlphaEngine
from clean.ga_search import GAFactorSearcher, ExpressionGenerator
from clean.config import GA

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleFormulaGASearcher:
    """单条公式遗传算法搜索器"""
    
    def __init__(self, formula: str):
        """
        Parameters
        ----------
        formula : str
            单条Alpha公式，例如 "add(close, open)"
        """
        self.formula = formula
        
    def generate_population_from_formula(
        self,
        generator: ExpressionGenerator,
        population_size: int,
        mutation_rate: float = 0.3,
    ) -> List[str]:
        """
        从单条公式生成初始种群
        
        Parameters
        ----------
        generator : ExpressionGenerator
            表达式生成器
        population_size : int
            种群大小
        mutation_rate : float
            变异率
        
        Returns
        -------
        List[str]
            种群表达式列表
        """
        population = [self.formula]  # 原始公式
        
        # 通过变异生成其他个体
        for _ in range(population_size - 1):
            if np.random.random() < mutation_rate:
                # 变异原始公式
                mutated = generator.mutate(self.formula)
                population.append(mutated)
            else:
                # 随机生成
                new_expr = generator.generate_random(max_depth=GA.max_tree_depth)
                population.append(new_expr)
        
        return population
    
    def search(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
        population_size: int = 100,
        n_generations: int = 30,
    ) -> List[Dict]:
        """
        执行遗传算法搜索
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV数据
        returns : pd.Series
            收益率序列
        population_size : int
            种群大小
        n_generations : int
            进化代数
        
        Returns
        -------
        List[Dict]
            Top因子列表
        """
        engine = AlphaEngine(df)
        generator = ExpressionGenerator(engine, returns)
        
        # 1. 生成初始种群
        logger.info(f"从公式生成初始种群: {self.formula}")
        population = self.generate_population_from_formula(
            generator, population_size
        )
        
        logger.info(f"初始种群大小: {len(population)}")
        logger.info(f"原始公式: {self.formula}")
        logger.info(f"示例变异: {population[1] if len(population) > 1 else 'N/A'}")
        
        # 2. 创建搜索器
        searcher = GAFactorSearcher(engine, returns)
        searcher.generator = generator
        
        # 3. 执行进化
        logger.info(f"开始进化: {n_generations}代")
        
        for gen in range(n_generations):
            # 评估
            scored = []
            for expr in population:
                _, fitness = generator.evaluate_expression(expr)
                scored.append((expr, fitness))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # 记录最佳
            best_expr, best_fit = scored[0]
            if gen % 5 == 0 or gen == n_generations - 1:
                logger.info(f"  Gen {gen}: best_fitness={best_fit:.4f}, "
                          f"expr={best_expr[:60]}")
            
            # 选择
            elite_count = max(2, len(population) // 10)
            new_pop = [expr for expr, _ in scored[:elite_count]]
            
            # 交叉和变异
            while len(new_pop) < len(population):
                if np.random.random() < GA.crossover_prob:
                    # 锦标赛选择
                    k = 5
                    candidates = np.random.choice(len(scored), size=min(k, len(scored)), replace=False)
                    p1 = scored[candidates[0]][0]
                    p2 = scored[candidates[1]][0]
                    
                    # 交叉
                    c1, c2 = generator.crossover(p1, p2)
                    new_pop.append(c1)
                    if len(new_pop) < len(population):
                        new_pop.append(c2)
                else:
                    # 变异
                    candidates = np.random.choice(len(scored), size=min(5, len(scored)), replace=False)
                    p = scored[candidates[0]][0]
                    new_pop.append(generator.mutate(p))
            
            population = new_pop[:len(population)]
        
        # 4. 最终评估
        logger.info("最终评估...")
        final_scored = []
        seen = set()
        
        for expr in population:
            if expr in seen:
                continue
            seen.add(expr)
            
            factor, fitness = generator.evaluate_expression(expr)
            if factor is not None and fitness > -999:
                from clean.ga_search import _safe_spearman_ic, _safe_ir
                
                ic_mean = _safe_spearman_ic(factor, returns)
                icir = _safe_ir(factor, returns)
                
                final_scored.append({
                    "expression": expr,
                    "ic_mean": ic_mean,
                    "icir": icir,
                    "fitness": fitness,
                })
        
        final_scored.sort(key=lambda x: x["fitness"], reverse=True)
        
        logger.info(f"搜索完成: 找到{len(final_scored)}个有效因子")
        for i, item in enumerate(final_scored[:5]):
            logger.info(f"  #{i+1}: fitness={item['fitness']:.4f}, "
                       f"ic={item['ic_mean']:.4f}, icir={item['icir']:.4f}")
        
        return final_scored[:20]


def run_single_formula_search(
    formula: str,
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2023-06-01",
    population_size: int = 100,
    n_generations: int = 30,
    output_file: str = None,
):
    """
    运行单条公式的遗传算法搜索
    
    Parameters
    ----------
    formula : str
        单条Alpha公式
    instruments : str
        股票池
    start_time, end_time : str
        时间范围
    population_size : int
        种群大小
    n_generations : int
        进化代数
    output_file : str
        输出文件路径
    """
    logger.info("="*80)
    logger.info("单条公式遗传算法搜索")
    logger.info("="*80)
    logger.info(f"原始公式: {formula}")
    
    # 1. 加载数据
    logger.info("加载数据...")
    df = load_ohlcv(instruments=instruments, start_time=start_time, end_time=end_time)
    
    # 2. 创建label
    logger.info("创建label...")
    returns = df.groupby(level='instrument')['close'].pct_change().shift(-1)
    
    # 3. 创建搜索器
    searcher = SingleFormulaGASearcher(formula)
    
    # 4. 执行搜索
    results = searcher.search(
        df=df,
        returns=returns,
        population_size=population_size,
        n_generations=n_generations,
    )
    
    # 5. 保存结果
    if output_file is None:
        safe_formula = formula.replace('(', '_').replace(')', '_').replace(',', '_')[:50]
        output_file = f"single_formula_{safe_formula}_results.csv"
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"结果已保存到: {output_file}")
    
    # 6. 打印Top结果
    print("\n" + "="*80)
    print("Top 10 因子")
    print("="*80)
    for i, item in enumerate(results[:10]):
        print(f"\n#{i+1}:")
        print(f"  适应度: {item['fitness']:.4f}")
        print(f"  IC均值: {item['ic_mean']:.4f}")
        print(f"  ICIR:   {item['icir']:.4f}")
        print(f"  表达式: {item['expression'][:80]}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="单条公式遗传算法搜索")
    parser.add_argument("--formula", type=str, required=True,
                       help="Alpha公式，例如 \"add(close, open)\"")
    parser.add_argument("--instruments", type=str, default="csi300",
                       help="股票池")
    parser.add_argument("--start-time", type=str, default="2020-01-01",
                       help="开始时间")
    parser.add_argument("--end-time", type=str, default="2023-06-01",
                       help="结束时间")
    parser.add_argument("--population-size", type=int, default=100,
                       help="种群大小")
    parser.add_argument("--n-generations", type=int, default=30,
                       help="进化代数")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    run_single_formula_search(
        formula=args.formula,
        instruments=args.instruments,
        start_time=args.start_time,
        end_time=args.end_time,
        population_size=args.population_size,
        n_generations=args.n_generations,
        output_file=args.output,
    )
