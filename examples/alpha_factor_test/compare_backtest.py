"""
Alpha158 vs Top50 对比回测
对齐官方yaml的日期划分方式:
  训练: 2008-01-01 ~ 2021-12-31 (更长训练期, 和官方类似)
  验证: 2022-01-01 ~ 2023-12-31
  测试: 2024-01-01 ~ 2026-04-13 (最近2年多)

Alpha158: 用qlib官方XGBModel + Alpha158 handler
Top50: 用已有Top50因子, 其他参数对齐Alpha158
"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import logging
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from pathlib import Path
    import time
    import gc

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("compare_backtest")

    OUTPUT_DIR = Path(r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\clean\output')
    OUTPUT_DIR.mkdir(exist_ok=True)

    TRAIN_START = "2008-01-01"
    TRAIN_END = "2021-12-31"
    VALID_END = "2023-12-31"
    TEST_END = "2026-04-13"

    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.0421,
        "max_depth": 8,
        "subsample": 0.8789,
        "colsample_bytree": 0.8879,
        "nthread": 20,
    }
    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING = 50

    def calc_ic(pred, label):
        common_idx = pred.dropna().index.intersection(label.dropna().index)
        result = pd.DataFrame({"pred": pred.loc[common_idx], "label": label.loc[common_idx]}, index=common_idx)
        ic_list, rank_ic_list = [], []
        for date in result.index.get_level_values(0).unique():
            day = result.loc[date]
            if len(day) < 10:
                continue
            try:
                ic, _ = pearsonr(day["pred"], day["label"])
                rank_ic, _ = spearmanr(day["pred"], day["label"])
                ic_list.append(ic)
                rank_ic_list.append(rank_ic)
            except:
                continue
        ic_arr = np.array(ic_list)
        rank_ic_arr = np.array(rank_ic_list)
        return {
            "ic_mean": ic_arr.mean() if len(ic_arr) > 0 else 0,
            "icir": ic_arr.mean() / ic_arr.std() if len(ic_arr) > 0 and ic_arr.std() > 0 else 0,
            "rank_ic_mean": rank_ic_arr.mean() if len(rank_ic_arr) > 0 else 0,
            "rank_icir": rank_ic_arr.mean() / rank_ic_arr.std() if len(rank_ic_arr) > 0 and rank_ic_arr.std() > 0 else 0,
            "test_periods": len(ic_arr),
            "test_samples": len(common_idx),
        }

    # ============ Step 1: Alpha158 官方方式回测 ============
    logger.info("=" * 60)
    logger.info("Step 1: Alpha158 官方方式回测")
    logger.info("=" * 60)

    import qlib
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    handler_a158 = Alpha158(
        instruments="csi300",
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
    )

    dataset_a158 = DatasetH(
        handler=handler_a158,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": ("2022-01-01", VALID_END),
            "test": ("2024-01-01", TEST_END),
        },
    )

    model_a158 = XGBModel(
        eval_metric="rmse",
        colsample_bytree=0.8879,
        eta=0.0421,
        max_depth=8,
        n_estimators=647,
        subsample=0.8789,
        nthread=20,
    )

    logger.info("Alpha158 训练中...")
    t0 = time.time()
    model_a158.fit(dataset_a158)
    logger.info(f"Alpha158 训练完成, 耗时={(time.time()-t0)/60:.1f}分钟")

    pred_a158 = model_a158.predict(dataset_a158)
    label_a158 = dataset_a158.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(label_a158, pd.DataFrame):
        label_a158 = label_a158.squeeze()

    a158_result = calc_ic(pred_a158, label_a158)
    a158_result["model"] = "Alpha158"
    a158_result["best_iteration"] = getattr(model_a158, "best_iteration", "N/A")

    logger.info(f"Alpha158: IC={a158_result['ic_mean']:.4f}, ICIR={a158_result['icir']:.4f}, "
                f"RankIC={a158_result['rank_ic_mean']:.4f}, RankICIR={a158_result['rank_icir']:.4f}, "
                f"测试期={a158_result['test_periods']}, 样本={a158_result['test_samples']}")

    del handler_a158, dataset_a158, model_a158, pred_a158, label_a158
    gc.collect()

    # ============ Step 2: Top50 因子计算 ============
    logger.info("=" * 60)
    logger.info("Step 2: Top50 因子计算")
    logger.info("=" * 60)

    top50_csv = OUTPUT_DIR / "top50_by_rank_icir.csv"
    top50_df = pd.read_csv(top50_csv)
    top50_names = top50_df["name"].head(50).tolist()
    logger.info(f"从已有IC结果加载Top50因子: {len(top50_names)}个")

    from clean.formula_parser import load_all_formulas
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine

    a101, a191, _ = load_all_formulas()
    a101_prefixed = {f"alpha101_{k}": v for k, v in a101.items()}
    a191_prefixed = {f"alpha191_{k}": v for k, v in a191.items()}
    all_formulas = {**a101_prefixed, **a191_prefixed}

    top50_formulas = {}
    for name in top50_names:
        if name in all_formulas:
            top50_formulas[name] = all_formulas[name]
    logger.info(f"Top50公式匹配: {len(top50_formulas)}个")

    df = load_ohlcv(start_time=TRAIN_START, end_time=TEST_END)
    logger.info(f"数据加载完成: {df.shape}")

    engine = AlphaEngine(df)

    label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    ).clip(lower=-0.2, upper=0.2)

    logger.info("计算Top50因子值...")
    top50_features = {}
    failed = []
    for i, (name, formula) in enumerate(top50_formulas.items()):
        try:
            factor = engine.calculate(formula)
            if factor is not None and not factor.dropna().empty:
                top50_features[name] = factor
            else:
                failed.append(name)
        except:
            failed.append(name)
        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(top50_formulas)}, 成功={len(top50_features)}, 失败={len(failed)}")

    logger.info(f"Top50因子计算完成: 成功={len(top50_features)}, 失败={len(failed)}")

    feature_df = pd.DataFrame(top50_features)
    feature_df["LABEL0"] = label
    logger.info(f"特征矩阵: {feature_df.shape}")

    del df, engine
    gc.collect()

    # ============ Step 3: Top50 XGBoost训练 ============
    logger.info("=" * 60)
    logger.info("Step 3: Top50 XGBoost训练")
    logger.info("=" * 60)

    import xgboost as xgb
    from qlib.data.dataset.processor import CSZScoreNorm

    feat_cols = [c for c in feature_df.columns if c != "LABEL0"]
    X = feature_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y = feature_df["LABEL0"]

    dt = X.index.get_level_values("datetime")
    X_train = X[dt <= TRAIN_END]
    X_valid = X[(dt > TRAIN_END) & (dt <= VALID_END)]
    X_test = X[dt > VALID_END]
    y_train = y[dt <= TRAIN_END]
    y_valid = y[(dt > TRAIN_END) & (dt <= VALID_END)]
    y_test = y[dt > VALID_END]

    csz = CSZScoreNorm()
    X_train = csz(X_train).fillna(0)
    X_valid = csz(X_valid).fillna(0)
    X_test = csz(X_test).fillna(0)

    valid_train = ~y_train.isna()
    valid_valid = ~y_valid.isna()
    valid_test = ~y_test.isna()
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_valid, y_valid = X_valid[valid_valid], y_valid[valid_valid]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    logger.info(f"Top50: 训练={len(X_train)}, 验证={len(X_valid)}, 测试={len(X_test)}, 特征={X_train.shape[1]}")

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dvalid = xgb.DMatrix(X_valid.values, label=y_valid.values)
    dtest = xgb.DMatrix(X_test.values)

    logger.info("Top50 XGBoost 训练中...")
    t0 = time.time()
    model_top50 = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=100,
    )
    logger.info(f"Top50 训练完成, 耗时={(time.time()-t0)/60:.1f}分钟, best_round={model_top50.best_iteration}")

    pred_top50 = pd.Series(model_top50.predict(dtest), index=X_test.index)
    top50_result = calc_ic(pred_top50, y_test)
    top50_result["model"] = "Top50"
    top50_result["n_features"] = len(feat_cols)
    top50_result["best_iteration"] = model_top50.best_iteration

    # ============ 最终对比 ============
    logger.info("\n" + "=" * 80)
    logger.info(f"最终对比结果")
    logger.info(f"  训练: {TRAIN_START} ~ {TRAIN_END}")
    logger.info(f"  验证: 2022-01-01 ~ {VALID_END}")
    logger.info(f"  测试: 2024-01-01 ~ {TEST_END}")
    logger.info("=" * 80)
    logger.info(f"  Alpha158: IC={a158_result['ic_mean']:.4f}, ICIR={a158_result['icir']:.4f}, "
                f"RankIC={a158_result['rank_ic_mean']:.4f}, RankICIR={a158_result['rank_icir']:.4f}, "
                f"测试期={a158_result['test_periods']}, 样本={a158_result['test_samples']}, "
                f"best_iter={a158_result['best_iteration']}")
    logger.info(f"  Top50:    IC={top50_result['ic_mean']:.4f}, ICIR={top50_result['icir']:.4f}, "
                f"RankIC={top50_result['rank_ic_mean']:.4f}, RankICIR={top50_result['rank_icir']:.4f}, "
                f"测试期={top50_result['test_periods']}, 样本={top50_result['test_samples']}, "
                f"特征={top50_result['n_features']}, best_iter={top50_result['best_iteration']}")

    results = pd.DataFrame([a158_result, top50_result])
    results.to_csv(OUTPUT_DIR / "compare_backtest_results.csv", index=False)
    logger.info(f"结果已保存: {OUTPUT_DIR / 'compare_backtest_results.csv'}")
    logger.info("全部完成!")
