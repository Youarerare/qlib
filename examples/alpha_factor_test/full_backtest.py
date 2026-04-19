"""一键完整回测: 2008-2026, Alpha158 vs Top50"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import logging
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    import xgboost as xgb

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("full_backtest")

    # ============ Step 1: 生成Top50特征 (2008-2026) ============
    logger.info("=" * 60)
    logger.info("Step 1: 生成Top50特征 (2008-2026)")

    from clean.formula_parser import load_all_formulas
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import compute_factors
    from clean.ic_analyzer import evaluate_all_factors, get_top_k
    from clean.config import OUTPUT_DIR, BACKTEST, XGBOOST

    init_qlib()
    a101, a191, _ = load_all_formulas()
    a101_prefixed = {f"alpha101_{k}": v for k, v in a101.items()}
    a191_prefixed = {f"alpha191_{k}": v for k, v in a191.items()}
    formulas = {**a101_prefixed, **a191_prefixed}
    logger.info(f"共{len(formulas)}个公式 (Alpha101={len(a101_prefixed)}, Alpha191={len(a191_prefixed)})")

    df = load_ohlcv(start_time="2008-01-01", end_time="2026-04-13")
    feature_df = compute_factors(df, formulas)

    label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    label = label.clip(lower=-0.2, upper=0.2)
    feature_df["LABEL0"] = label
    feature_df.to_pickle(OUTPUT_DIR / "all_features_full.pkl")
    logger.info(f"特征计算完成: {feature_df.shape}")

    # ============ Step 2: IC评估, 选Top50 ============
    logger.info("=" * 60)
    logger.info("Step 2: IC评估")
    ic_results = evaluate_all_factors(feature_df)
    ic_results = get_top_k(ic_results, k=50)
    ic_results.to_csv(OUTPUT_DIR / "top50_full.csv", index=False)
    top50_names = ic_results["name"].head(50).tolist()
    top5_info = [(r['name'], round(r['rank_icir'], 4)) for _, r in ic_results.head(5).iterrows()]
    logger.info(f"Top5因子: {top5_info}")

    # ============ Step 3: Alpha158 官方方式回测 ============
    logger.info("=" * 60)
    logger.info("Step 3: Alpha158 官方方式回测 (2008-2026)")

    import qlib
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    # 训练: 2008-2021, 验证: 2022, 测试: 2023-2026
    handler_a158 = Alpha158(
        instruments="csi300",
        start_time="2008-01-01",
        end_time="2026-04-13",
        fit_start_time="2008-01-01",
        fit_end_time="2021-12-31",
    )

    dataset_a158 = DatasetH(
        handler=handler_a158,
        segments={
            "train": ("2008-01-01", "2021-12-31"),
            "valid": ("2022-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2026-04-13"),
        },
    )

    model_a158 = XGBModel(
        eval_metric="rmse",
        colsample_bytree=0.8879,
        eta=0.0421,
        max_depth=8,
        n_estimators=1000,
        subsample=0.8789,
        nthread=20,
    )

    model_a158.fit(dataset_a158)
    pred_a158 = model_a158.predict(dataset_a158)

    label_a158 = dataset_a158.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(label_a158, pd.DataFrame):
        label_a158 = label_a158.squeeze()

    common_idx = pred_a158.index.intersection(label_a158.dropna().index)
    result_a158 = pd.DataFrame({"pred": pred_a158.loc[common_idx], "label": label_a158.loc[common_idx]}, index=common_idx)

    ic_list, rank_ic_list = [], []
    for date in result_a158.index.get_level_values(0).unique():
        day = result_a158.loc[date]
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

    a158_result = {
        "model": "Alpha158",
        "ic_mean": ic_arr.mean(),
        "icir": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        "rank_ic_mean": rank_ic_arr.mean(),
        "rank_icir": rank_ic_arr.mean() / rank_ic_arr.std() if rank_ic_arr.std() > 0 else 0,
        "test_periods": len(ic_arr),
        "test_samples": len(common_idx),
    }

    logger.info(f"Alpha158结果: IC={a158_result['ic_mean']:.4f}, ICIR={a158_result['icir']:.4f}, "
                f"RankIC={a158_result['rank_ic_mean']:.4f}, RankICIR={a158_result['rank_icir']:.4f}, "
                f"测试期={a158_result['test_periods']}, 样本={a158_result['test_samples']}")

    # ============ Step 4: Top50因子回测 ============
    logger.info("=" * 60)
    logger.info("Step 4: Top50因子回测 (2008-2026)")

    top50_data = pd.read_pickle(OUTPUT_DIR / "all_features_full.pkl")
    top50_data.index = top50_data.index.rename(["datetime", "instrument"])

    available = [c for c in top50_names if c in top50_data.columns]
    logger.info(f"Top50因子可用: {len(available)}个")

    X_top50 = top50_data[available].copy()
    y_top50 = top50_data["LABEL0"].copy()

    # 三段分割
    dt = X_top50.index.get_level_values("datetime")
    X_train = X_top50[dt < "2022-01-01"]
    X_valid = X_top50[(dt >= "2022-01-01") & (dt < "2023-01-01")]
    X_test = X_top50[dt >= "2023-01-01"]

    y_train = y_top50[dt < "2022-01-01"]
    y_valid = y_top50[(dt >= "2022-01-01") & (dt < "2023-01-01")]
    y_test = y_top50[dt >= "2023-01-01"]

    # CSZScoreNorm特征
    from clean.data_manager import apply_cszscorenorm
    X_train = apply_cszscorenorm(X_train)
    X_valid = apply_cszscorenorm(X_valid)
    X_test = apply_cszscorenorm(X_test)

    # 清理NaN
    def clean(X, y):
        Xc = X.replace([np.inf, -np.inf], np.nan)
        valid = (Xc.isna().mean(axis=1) <= 0.5) & (~y.isna())
        return Xc.loc[valid].fillna(0), y.loc[valid]

    X_train, y_train = clean(X_train, y_train)
    X_valid, y_valid = clean(X_valid, y_valid)
    X_test, y_test = clean(X_test, y_test)

    logger.info(f"Top50: 训练={len(X_train)}, 验证={len(X_valid)}, 测试={len(X_test)}, 特征={X_train.shape[1]}")

    # XGBoost训练
    cfg = XGBOOST
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dvalid = xgb.DMatrix(X_valid.values, label=y_valid.values)
    dtest = xgb.DMatrix(X_test.values)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "nthread": cfg.n_jobs if cfg.n_jobs > 0 else -1,
    }

    model_top50 = xgb.train(params, dtrain, num_boost_round=cfg.n_estimators,
                             evals=[(dtrain, "train"), (dvalid, "valid")],
                             early_stopping_rounds=50, verbose_eval=100)

    pred_top50 = model_top50.predict(dtest)

    # IC评估
    result_top50 = pd.DataFrame({"pred": pred_top50, "label": y_test.values}, index=X_test.index)
    ic_list2, rank_ic_list2 = [], []
    for date in result_top50.index.get_level_values("datetime").unique():
        day = result_top50.loc[date]
        if len(day) < 10:
            continue
        try:
            ic, _ = pearsonr(day["pred"], day["label"])
            rank_ic, _ = spearmanr(day["pred"], day["label"])
            ic_list2.append(ic)
            rank_ic_list2.append(rank_ic)
        except:
            continue

    ic_arr2 = np.array(ic_list2)
    rank_ic_arr2 = np.array(rank_ic_list2)

    top50_result = {
        "model": "Top50",
        "ic_mean": ic_arr2.mean(),
        "icir": ic_arr2.mean() / ic_arr2.std() if ic_arr2.std() > 0 else 0,
        "rank_ic_mean": rank_ic_arr2.mean(),
        "rank_icir": rank_ic_arr2.mean() / rank_ic_arr2.std() if rank_ic_arr2.std() > 0 else 0,
        "test_periods": len(ic_arr2),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
    }

    # ============ 最终对比 ============
    logger.info("\n" + "=" * 80)
    logger.info("最终对比结果 (2008-2026, 测试期: 2023-01-01 ~ 2026-04-13)")
    logger.info("=" * 80)
    logger.info(f"  Alpha158: IC={a158_result['ic_mean']:.4f}, ICIR={a158_result['icir']:.4f}, "
                f"RankIC={a158_result['rank_ic_mean']:.4f}, RankICIR={a158_result['rank_icir']:.4f}, "
                f"测试期={a158_result['test_periods']}, 样本={a158_result['test_samples']}")
    logger.info(f"  Top50:    IC={top50_result['ic_mean']:.4f}, ICIR={top50_result['icir']:.4f}, "
                f"RankIC={top50_result['rank_ic_mean']:.4f}, RankICIR={top50_result['rank_icir']:.4f}, "
                f"测试期={top50_result['test_periods']}, 样本={top50_result['test_samples']}, 特征={top50_result['n_features']}")

    results = pd.DataFrame([a158_result, top50_result])
    results.to_csv(OUTPUT_DIR / "full_backtest_results.csv", index=False)
    logger.info("全部完成!")
