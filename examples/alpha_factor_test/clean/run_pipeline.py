"""
一键执行流水线 - 从因子计算、IC评估、模型对比到遗传算法搜索
"""
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from .config import BACKTEST, OUTPUT_DIR
from .formula_parser import load_all_formulas
from .data_manager import init_qlib, load_ohlcv, get_stock_list
from .alpha_engine import compute_factors, AlphaEngine
from .ic_analyzer import evaluate_all_factors, get_top_k
from .model_trainer import run_comparison
from .ga_search import GAFactorSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def step1_load_formulas():
    """Step 1: 加载并验证公式"""
    logger.info("=" * 60)
    logger.info("Step 1: 加载公式")
    a101, a191, issues = load_all_formulas()
    logger.info(f"  Alpha101: {len(a101)}个公式")
    logger.info(f"  Alpha191: {len(a191)}个公式")
    logger.info(f"  问题公式: {len(issues)}个")
    if issues:
        for item in issues[:10]:
            logger.warning(f"    - {item['name']}: {item['error']}")

    # 合并公式，处理重复名称
    all_formulas = {}
    for name, formula in a101.items():
        all_formulas[f"alpha101_{name}"] = formula
    for name, formula in a191.items():
        all_formulas[f"alpha191_{name}"] = formula

    logger.info(f"  合并后唯一公式: {len(all_formulas)}个")
    return all_formulas, a101, a191, issues


def step2_compute_factors(formulas: dict):
    """Step 2: 计算所有因子"""
    logger.info("=" * 60)
    logger.info("Step 2: 计算因子")
    init_qlib()
    df = load_ohlcv()
    feature_df = compute_factors(df, formulas)
    feature_df.index = df.index[:len(feature_df)]

    # 添加标签
    label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    feature_df["LABEL0"] = label

    # 保存
    pkl_path = OUTPUT_DIR / "all_features.pkl"
    feature_df.to_pickle(pkl_path)
    logger.info(f"  保存到: {pkl_path}")
    logger.info(f"  形状: {feature_df.shape}")

    return feature_df, df


def step3_evaluate_ic(feature_df: pd.DataFrame):
    """Step 3: 评估所有因子IC/ICIR"""
    logger.info("=" * 60)
    logger.info("Step 3: 评估IC/ICIR")

    return_next = feature_df["LABEL0"]
    X = feature_df.drop(columns=["LABEL0"])

    results = evaluate_all_factors(X, return_next)
    results.to_csv(OUTPUT_DIR / "ic_results.csv", index=False)
    logger.info(f"  评估完成: {len(results)}个因子")

    # Top 50
    top50 = get_top_k(results, k=50, by="rank_icir")
    top50.to_csv(OUTPUT_DIR / "top50_by_rank_icir.csv", index=False)
    logger.info(f"  Top50因子已保存")

    # 打印Top10
    logger.info("  Top 10因子:")
    for i, row in top50.head(10).iterrows():
        logger.info(f"    #{i + 1}: {row['name']} | RankICIR={row.get('rank_icir', 'N/A'):.4f}")

    return results, top50


def step4_model_comparison(top50_pkl_path: str = None):
    """Step 4: Top50 vs Alpha158模型对比"""
    logger.info("=" * 60)
    logger.info("Step 4: 模型对比")

    if top50_pkl_path is None:
        top50_pkl_path = str(OUTPUT_DIR / "all_features.pkl")

    results = run_comparison(top50_pkl_path)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    logger.info("  对比结果:")
    for _, row in results.iterrows():
        logger.info(f"    {row['model']}: ICIR={row['icir']:.4f}, RankICIR={row['rank_icir']:.4f}, "
                     f"训练样本={row['train_samples']}, 测试样本={row['test_samples']}")

    return results


def step5_ga_search(feature_df: pd.DataFrame, top50_results: pd.DataFrame):
    """Step 5: 遗传算法因子搜索"""
    logger.info("=" * 60)
    logger.info("Step 5: 遗传算法因子搜索")

    init_qlib()
    df = load_ohlcv()
    engine = AlphaEngine(df)
    returns = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    # 设置已有Top因子用于相关性惩罚
    existing = []
    top_names = top50_results["name"].head(20).tolist()
    for name in top_names:
        if name in feature_df.columns:
            existing.append(feature_df[name].dropna())

    searcher = GAFactorSearcher(engine, returns)
    new_factors = searcher.search(existing_factors=existing)

    if new_factors:
        new_df = pd.DataFrame(new_factors)
        new_df.to_csv(OUTPUT_DIR / "ga_new_factors.csv", index=False)
        logger.info(f"  发现{len(new_factors)}个新因子")
        for i, f in enumerate(new_factors[:10]):
            logger.info(f"    #{i + 1}: fitness={f['fitness']:.4f}, "
                         f"IC={f.get('ic', 0):.4f}, IR={f.get('ir', 0):.4f}, "
                         f"ICIR={f.get('icir', 0):.4f}, RankICIR={f.get('rank_icir', 0):.4f}")
            logger.info(f"         expr={f['expression'][:80]}")

    return new_factors


def run_full_pipeline(skip_ga: bool = False):
    """执行完整流水线"""
    start = time.time()
    logger.info(f"流水线开始: {datetime.now()}")
    logger.info(f"配置: instruments={BACKTEST.instruments}, "
                f"时间={BACKTEST.start_time}~{BACKTEST.end_time}")

    # Step 1
    formulas, a101, a191, issues = step1_load_formulas()

    # Step 2
    feature_df, raw_df = step2_compute_factors(formulas)

    # Step 3
    ic_results, top50 = step3_evaluate_ic(feature_df)

    # Step 4
    model_results = step4_model_comparison()

    # Step 5
    if not skip_ga:
        new_factors = step5_ga_search(feature_df, top50)
    else:
        new_factors = []
        logger.info("跳过遗传算法搜索")

    elapsed = time.time() - start
    logger.info(f"\n流水线完成! 耗时: {elapsed / 60:.1f}分钟")
    logger.info(f"结果保存在: {OUTPUT_DIR}")

    return {
        "formulas": {"alpha101": len(a101), "alpha191": len(a191), "issues": len(issues)},
        "ic_results": ic_results,
        "top50": top50,
        "model_comparison": model_results,
        "new_factors": new_factors,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ga", action="store_true", help="跳过遗传算法搜索")
    parser.add_argument("--step", type=int, help="只运行指定步骤(1-5)")
    args = parser.parse_args()

    if args.step:
        if args.step == 1:
            step1_load_formulas()
        elif args.step == 2:
            formulas, _, _, _ = step1_load_formulas()
            step2_compute_factors(formulas)
        elif args.step == 3:
            feature_df = pd.read_pickle(OUTPUT_DIR / "all_features.pkl")
            step3_evaluate_ic(feature_df)
        elif args.step == 4:
            step4_model_comparison()
        elif args.step == 5:
            feature_df = pd.read_pickle(OUTPUT_DIR / "all_features.pkl")
            top50 = pd.read_csv(OUTPUT_DIR / "top50_by_rank_icir.csv")
            step5_ga_search(feature_df, top50)
    else:
        run_full_pipeline(skip_ga=args.skip_ga)
