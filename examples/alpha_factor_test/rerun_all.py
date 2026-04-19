"""一键重跑: 修复power()后重新计算因子+IC评估+模型对比"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import logging
    from clean.config import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("rerun")

    # Step 1: 加载公式
    logger.info("=" * 60)
    logger.info("Step 1: 加载公式")
    from clean.formula_parser import load_all_formulas
    a101, a191, issues = load_all_formulas()
    formulas = {**a101, **a191}
    logger.info(f"共{len(formulas)}个公式")

    # Step 2: 计算因子
    logger.info("=" * 60)
    logger.info("Step 2: 计算因子 (power()已修复)")
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import compute_factors
    init_qlib()
    df = load_ohlcv()
    feature_df = compute_factors(df, formulas)

    # Label: T+1→T+2收益率 (与Alpha158一致), winsorize
    label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    label = label.clip(lower=-0.2, upper=0.2)
    feature_df["LABEL0"] = label

    feature_df.to_pickle(OUTPUT_DIR / "all_features.pkl")
    logger.info(f"因子计算完成: {feature_df.shape}, 保存到 all_features.pkl")

    # Step 3: IC评估
    logger.info("=" * 60)
    logger.info("Step 3: IC评估")
    from clean.ic_analyzer import compute_all_ic
    ic_results = compute_all_ic(feature_df)
    ic_results.to_csv(OUTPUT_DIR / "top50_by_rank_icir.csv", index=False)
    logger.info(f"IC评估完成, Top5:")
    for _, row in ic_results.head(5).iterrows():
        logger.info(f"  {row['name']}: RankICIR={row['rank_icir']:.4f}")

    # Step 4: 模型对比
    logger.info("=" * 60)
    logger.info("Step 4: 模型对比")
    from clean.model_trainer import run_comparison
    results = run_comparison(str(OUTPUT_DIR / "all_features.pkl"))
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info("对比结果:")
    for _, row in results.iterrows():
        logger.info(f"  {row['model']}: ICIR={row['icir']:.4f}, RankICIR={row['rank_icir']:.4f}, "
                     f"训练={row['train_samples']}, 测试={row['test_samples']}, 特征={row['n_features']}")

    logger.info("全部完成!")
