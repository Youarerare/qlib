"""缩短版回测: 2021-2026, 带详细中间过程"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import logging
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    import xgboost as xgb
    import time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("quick_backtest")

    from clean.formula_parser import load_all_formulas
    from clean.data_manager import init_qlib, load_ohlcv
    from clean.alpha_engine import AlphaEngine, compute_factors
    from clean.config import OUTPUT_DIR, XGBOOST

    init_qlib()
    a101, a191, _ = load_all_formulas()
    a101_prefixed = {f"alpha101_{k}": v for k, v in a101.items()}
    a191_prefixed = {f"alpha191_{k}": v for k, v in a191.items()}
    formulas = {**a101_prefixed, **a191_prefixed}
    logger.info(f"共{len(formulas)}个公式 (Alpha101={len(a101_prefixed)}, Alpha191={len(a191_prefixed)})")

    df = load_ohlcv(start_time="2021-01-01", end_time="2026-04-13")
    logger.info(f"数据加载完成: {df.shape}")

    # ============ Step 1: 逐个计算因子，实时显示IC ============
    logger.info("=" * 60)
    logger.info("Step 1: 计算因子 + 实时IC评估")
    logger.info("=" * 60)

    engine = AlphaEngine(df)

    # 先计算label
    label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    label = label.clip(lower=-0.2, upper=0.2)

    results_list = []
    failed = []
    t0 = time.time()

    items = list(formulas.items())
    for i, (name, formula) in enumerate(items):
        try:
            factor = engine.calculate(formula)
            if factor is None or (isinstance(factor, pd.Series) and factor.dropna().empty):
                failed.append(name)
                continue

            # 计算该因子的IC
            common = factor.dropna().index.intersection(label.dropna().index)
            if len(common) < 100:
                failed.append(name)
                continue

            f_vals = factor.loc[common]
            l_vals = label.loc[common]

            # 逐日IC
            result_df = pd.DataFrame({"factor": f_vals, "label": l_vals}, index=common)
            ic_list, rank_ic_list = [], []
            for date in result_df.index.get_level_values("datetime").unique():
                day = result_df.loc[date]
                if len(day) < 10:
                    continue
                try:
                    ic, _ = pearsonr(day["factor"], day["label"])
                    rank_ic, _ = spearmanr(day["factor"], day["label"])
                    ic_list.append(ic)
                    rank_ic_list.append(rank_ic)
                except:
                    continue

            if len(ic_list) < 10:
                failed.append(name)
                continue

            ic_arr = np.array(ic_list)
            rank_ic_arr = np.array(rank_ic_list)

            ic_mean = ic_arr.mean()
            icir = ic_mean / ic_arr.std() if ic_arr.std() > 0 else 0
            rank_ic_mean = rank_ic_arr.mean()
            rank_icir = rank_ic_mean / rank_ic_arr.std() if rank_ic_arr.std() > 0 else 0

            results_list.append({
                "name": name,
                "ic_mean": ic_mean,
                "icir": icir,
                "rank_ic_mean": rank_ic_mean,
                "rank_icir": rank_icir,
                "ic_positive_ratio": (ic_arr > 0).mean(),
            })

            del factor, f_vals, l_vals, result_df, ic_arr, rank_ic_arr

            if (i + 1) % 5 == 0 or (i + 1) == len(items):
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (i + 1) * (len(items) - i - 1)
                top5 = sorted(results_list, key=lambda x: abs(x["rank_icir"]), reverse=True)[:5]
                top5_str = ", ".join([f"{r['name']}({r['rank_icir']:.3f})" for r in top5])
                logger.info(f"  进度: {i+1}/{len(items)}, 成功={len(results_list)}, "
                            f"失败={len(failed)}, 耗时={elapsed:.1f}min, 剩余≈{eta:.0f}min")
                logger.info(f"  当前Top5: {top5_str}")

        except Exception as e:
            failed.append(name)
            logger.debug(f"  失败: {name} - {str(e)[:60]}")

    logger.info(f"\n因子计算完成: 成功={len(results_list)}, 失败={len(failed)}, "
                f"总耗时={(time.time()-t0)/60:.1f}分钟")

    # ============ Step 2: Top50因子 ============
    logger.info("=" * 60)
    logger.info("Step 2: 选Top50因子")
    logger.info("=" * 60)

    results_sorted = sorted(results_list, key=lambda x: abs(x["rank_icir"]), reverse=True)
    top50_info = results_sorted[:50]
    top50_names = [r["name"] for r in top50_info]

    logger.info("Top10因子:")
    for j, r in enumerate(top50_info[:10]):
        logger.info(f"  #{j+1}: {r['name']}, IC={r['ic_mean']:.4f}, ICIR={r['icir']:.4f}, "
                     f"RankIC={r['rank_ic_mean']:.4f}, RankICIR={r['rank_icir']:.4f}")

    # 重新计算Top50因子（避免内存爆炸）
    logger.info("重新计算Top50因子...")
    top50_factors = {}
    for name in top50_names:
        formula = formulas[name]
        try:
            top50_factors[name] = engine.calculate(formula)
        except:
            pass
    feature_df = pd.DataFrame(top50_factors)
    feature_df["LABEL0"] = label
    feature_df.to_pickle(OUTPUT_DIR / "top50_features_2021_2026.pkl")
    logger.info(f"Top50特征保存: {feature_df.shape}")

    # 保存IC结果
    ic_df = pd.DataFrame(results_sorted)
    ic_df.to_csv(OUTPUT_DIR / "ic_results_2021_2026.csv", index=False)

    # ============ Step 3: Alpha158 官方方式回测 ============
    logger.info("=" * 60)
    logger.info("Step 3: Alpha158 官方方式回测")
    logger.info("=" * 60)

    import qlib
    from qlib.contrib.model.xgboost import XGBModel
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    handler_a158 = Alpha158(
        instruments="csi300",
        start_time="2021-01-01",
        end_time="2026-04-13",
        fit_start_time="2021-01-01",
        fit_end_time="2023-12-31",
    )

    dataset_a158 = DatasetH(
        handler=handler_a158,
        segments={
            "train": ("2021-01-01", "2023-12-31"),
            "valid": ("2024-01-01", "2024-06-30"),
            "test": ("2024-07-01", "2026-04-13"),
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

    logger.info("Alpha158训练中...")
    model_a158.fit(dataset_a158)
    pred_a158 = model_a158.predict(dataset_a158)
    logger.info(f"Alpha158预测完成: {len(pred_a158)}条")

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
    logger.info("Step 4: Top50因子回测")
    logger.info("=" * 60)

    top50_names = [r["name"] for r in top50_info]
    X_top50 = feature_df[top50_names].copy()
    y_top50 = feature_df["LABEL0"].copy()

    dt = X_top50.index.get_level_values("datetime")
    X_train = X_top50[dt < "2024-01-01"]
    X_valid = X_top50[(dt >= "2024-01-01") & (dt < "2024-07-01")]
    X_test = X_top50[dt >= "2024-07-01"]

    y_train = y_top50[dt < "2024-01-01"]
    y_valid = y_top50[(dt >= "2024-01-01") & (dt < "2024-07-01")]
    y_test = y_top50[dt >= "2024-07-01"]

    from clean.data_manager import apply_cszscorenorm
    X_train = apply_cszscorenorm(X_train)
    X_valid = apply_cszscorenorm(X_valid)
    X_test = apply_cszscorenorm(X_test)

    def clean(X, y):
        Xc = X.replace([np.inf, -np.inf], np.nan)
        valid = (Xc.isna().mean(axis=1) <= 0.5) & (~y.isna())
        return Xc.loc[valid].fillna(0), y.loc[valid]

    X_train, y_train = clean(X_train, y_train)
    X_valid, y_valid = clean(X_valid, y_valid)
    X_test, y_test = clean(X_test, y_test)

    logger.info(f"Top50: 训练={len(X_train)}, 验证={len(X_valid)}, 测试={len(X_test)}, 特征={X_train.shape[1]}")

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

    logger.info("Top50 XGBoost训练中...")
    model_top50 = xgb.train(params, dtrain, num_boost_round=cfg.n_estimators,
                             evals=[(dtrain, "train"), (dvalid, "valid")],
                             early_stopping_rounds=50, verbose_eval=100)

    pred_top50 = model_top50.predict(dtest)

    result_top50_df = pd.DataFrame({"pred": pred_top50, "label": y_test.values}, index=X_test.index)
    ic_list2, rank_ic_list2 = [], []
    for date in result_top50_df.index.get_level_values("datetime").unique():
        day = result_top50_df.loc[date]
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
    logger.info("最终对比结果 (2021-2026, 测试期: 2024-07-01 ~ 2026-04-13)")
    logger.info("=" * 80)
    logger.info(f"  Alpha158: IC={a158_result['ic_mean']:.4f}, ICIR={a158_result['icir']:.4f}, "
                f"RankIC={a158_result['rank_ic_mean']:.4f}, RankICIR={a158_result['rank_icir']:.4f}, "
                f"测试期={a158_result['test_periods']}, 样本={a158_result['test_samples']}")
    logger.info(f"  Top50:    IC={top50_result['ic_mean']:.4f}, ICIR={top50_result['icir']:.4f}, "
                f"RankIC={top50_result['rank_ic_mean']:.4f}, RankICIR={top50_result['rank_icir']:.4f}, "
                f"测试期={top50_result['test_periods']}, 样本={top50_result['test_samples']}, 特征={top50_result['n_features']}")

    results = pd.DataFrame([a158_result, top50_result])
    results.to_csv(OUTPUT_DIR / "quick_backtest_results.csv", index=False)
    logger.info("全部完成!")
