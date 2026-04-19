"""对比: 不对label做任何处理，用原始真实收益率"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from clean.data_manager import (
        init_qlib, load_ohlcv, load_alpha158_data,
        get_common_stocks, filter_by_stocks, apply_cszscorenorm,
    )
    from clean.config import XGBOOST, BACKTEST, OUTPUT_DIR
    import xgboost as xgb

    init_qlib()

    # ---- 加载Top50数据 ----
    top50_raw = pd.read_pickle(OUTPUT_DIR / "all_features.pkl")
    top50_raw.index = top50_raw.index.rename(["datetime", "instrument"])

    # 筛选Top50因子
    ic_csv = OUTPUT_DIR / "top50_by_rank_icir.csv"
    if ic_csv.exists():
        top50_names = pd.read_csv(ic_csv)["name"].head(50).tolist()
        available = [c for c in top50_names if c in top50_raw.columns]
        keep_cols = available + ["LABEL0"]
        top50_raw = top50_raw[keep_cols]
        print(f"Top50因子: {len(available)}个")

    # ---- 加载Alpha158数据 ----
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    handler = Alpha158(instruments="csi300", start_time="2020-01-01", end_time="2023-06-01")
    a158_feat = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    a158_feat = a158_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    a158_feat.index = a158_feat.index.rename(["datetime", "instrument"])

    # ---- 用OHLCV计算真实收益率作为label (不做任何处理) ----
    df = load_ohlcv()
    raw_label = df.groupby(level="instrument")["close"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    print(f"原始Label统计: 均值={raw_label.mean():.6f}, 标准差={raw_label.std():.6f}, "
          f"最小={raw_label.min():.6f}, 最大={raw_label.max():.6f}, NaN={raw_label.isna().sum()}")

    # ---- 对齐 ----
    common = get_common_stocks(top50_raw, a158_feat.to_frame() if isinstance(a158_feat, pd.Series) else a158_feat)
    top50_filtered = filter_by_stocks(top50_raw, common)
    a158_filtered = a158_feat.loc[a158_feat.index.get_level_values("instrument").isin(common)]

    top50_dt = set(top50_filtered.index.get_level_values("datetime").unique())
    a158_dt = set(a158_filtered.index.get_level_values("datetime").unique())
    common_dates = sorted(top50_dt & a158_dt)

    top50_filtered = top50_filtered.loc[top50_filtered.index.get_level_values("datetime").isin(common_dates)]
    a158_filtered = a158_filtered.loc[a158_filtered.index.get_level_values("datetime").isin(common_dates)]

    # label也过滤到共同范围
    raw_label_filtered = raw_label.loc[
        raw_label.index.get_level_values("instrument").isin(common) &
        raw_label.index.get_level_values("datetime").isin(common_dates)
    ]

    print(f"共同股票: {len(common)}只, 共同交易日: {len(common_dates)}天")

    # ---- 三段分割 ----
    train_end = BACKTEST.train_end
    valid_end = BACKTEST.test_start

    def split3(df):
        dt = df.index.get_level_values("datetime")
        return df[dt < train_end], df[(dt >= train_end) & (dt < valid_end)], df[dt >= valid_end]

    X1_train, X1_valid, X1_test = split3(top50_filtered.drop(columns=["LABEL0"]))
    X2_train, X2_valid, X2_test = split3(a158_filtered)
    y_train, y_valid, y_test = split3(raw_label_filtered)

    # Top50特征CSZScoreNorm
    X1_train = apply_cszscorenorm(X1_train)
    X1_valid = apply_cszscorenorm(X1_valid)
    X1_test = apply_cszscorenorm(X1_test)

    # 找共同有效索引
    def valid_rows(X, y):
        Xc = X.replace([np.inf, -np.inf], np.nan)
        x_ok = (Xc.isna().mean(axis=1) <= 0.5)
        y_ok = ~y.isna()
        return x_ok.index.intersection(y_ok[y_ok].index)

    train_idx = valid_rows(X1_train, y_train).intersection(valid_rows(X2_train, y_train))
    valid_idx = valid_rows(X1_valid, y_valid).intersection(valid_rows(X2_valid, y_valid))
    test_idx = valid_rows(X1_test, y_test).intersection(valid_rows(X2_test, y_test))

    print(f"对齐后: 训练={len(train_idx)}, 验证={len(valid_idx)}, 测试={len(test_idx)}")

    X1_train, X1_valid, X1_test = X1_train.loc[train_idx].fillna(0), X1_valid.loc[valid_idx].fillna(0), X1_test.loc[test_idx].fillna(0)
    X2_train, X2_valid, X2_test = X2_train.loc[train_idx].fillna(0), X2_valid.loc[valid_idx].fillna(0), X2_test.loc[test_idx].fillna(0)
    y_train, y_valid, y_test = y_train.loc[train_idx], y_valid.loc[valid_idx], y_test.loc[test_idx]

    # ---- 训练+评估 ----
    def train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, name):
        cfg = XGBOOST
        dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
        dvalid = xgb.DMatrix(X_va.values, label=y_va.values)
        dtest = xgb.DMatrix(X_te.values)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "nthread": cfg.n_jobs if cfg.n_jobs > 0 else -1,
        }

        model = xgb.train(params, dtrain, num_boost_round=cfg.n_estimators,
                           evals=[(dtrain, "train"), (dvalid, "valid")],
                           early_stopping_rounds=50, verbose_eval=False)

        pred = model.predict(dtest)

        # 逐日IC
        result_df = pd.DataFrame({"pred": pred, "label": y_te.values}, index=X_te.index)
        ic_list, rank_ic_list = [], []
        for date in result_df.index.get_level_values("datetime").unique():
            day = result_df.loc[date]
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
            "model": name,
            "n_features": X_tr.shape[1],
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
            "ic_mean": ic_arr.mean(),
            "icir": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
            "rank_ic_mean": rank_ic_arr.mean(),
            "rank_icir": rank_ic_arr.mean() / rank_ic_arr.std() if rank_ic_arr.std() > 0 else 0,
        }

    m1 = train_eval(X1_train, y_train, X1_valid, y_valid, X1_test, y_test, "Top50")
    m2 = train_eval(X2_train, y_train, X2_valid, y_valid, X2_test, y_test, "Alpha158")

    print("\n" + "=" * 80)
    print("原始收益率Label (不做任何处理):")
    print("=" * 80)
    for m in [m1, m2]:
        print(f"  {m['model']}: ICIR={m['icir']:.4f}, RankICIR={m['rank_icir']:.4f}, "
              f"IC均值={m['ic_mean']:.4f}, RankIC均值={m['rank_ic_mean']:.4f}, "
              f"训练={m['train_samples']}, 测试={m['test_samples']}, 特征={m['n_features']}")
