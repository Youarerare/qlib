import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def compute_ic(
    predictions: pd.Series,
    labels: pd.Series,
) -> float:
    valid = predictions.notna() & labels.notna()
    if valid.sum() < 5:
        return np.nan
    return predictions[valid].corr(labels[valid], method="pearson")


def compute_rank_ic(
    predictions: pd.Series,
    labels: pd.Series,
) -> float:
    valid = predictions.notna() & labels.notna()
    if valid.sum() < 5:
        return np.nan
    return predictions[valid].corr(labels[valid], method="spearman")


def compute_daily_ic(
    pred_df: pd.DataFrame,
    label_col: str = "LABEL0",
    pred_col: str = "pred",
) -> pd.DataFrame:
    results = []
    for date, group in pred_df.groupby(level="datetime"):
        pred = group[pred_col]
        label = group[label_col]
        ic = compute_ic(pred, label)
        rank_ic = compute_rank_ic(pred, label)
        results.append({"datetime": date, "IC": ic, "Rank_IC": rank_ic, "count": len(group)})

    return pd.DataFrame(results).set_index("datetime")


def compute_long_short_return(
    pred_df: pd.DataFrame,
    label_col: str = "LABEL0",
    pred_col: str = "pred",
    n_groups: int = 5,
) -> pd.DataFrame:
    results = []
    for date, group in pred_df.groupby(level="datetime"):
        if len(group) < n_groups * 2:
            continue
        group = group.copy()
        group["group"] = pd.qcut(group[pred_col], n_groups, labels=False, duplicates="drop")
        long_ret = group[group["group"] == n_groups - 1][label_col].mean()
        short_ret = group[group["group"] == 0][label_col].mean()
        ls_ret = long_ret - short_ret
        results.append({
            "datetime": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "ls_return": ls_ret,
        })

    return pd.DataFrame(results).set_index("datetime")


def compute_performance_metrics(daily_ic_df: pd.DataFrame, ls_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}

    ic_series = daily_ic_df["IC"].dropna()
    rank_ic_series = daily_ic_df["Rank_IC"].dropna()

    metrics["IC_mean"] = ic_series.mean()
    metrics["IC_std"] = ic_series.std()
    metrics["ICIR"] = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
    metrics["Rank_IC_mean"] = rank_ic_series.mean()
    metrics["Rank_IC_std"] = rank_ic_series.std()
    metrics["Rank_ICIR"] = rank_ic_series.mean() / rank_ic_series.std() if rank_ic_series.std() > 0 else 0

    if len(ls_df) > 0:
        ls_ret = ls_df["ls_return"]
        metrics["LS_annual_return"] = ls_ret.mean() * 252
        metrics["LS_annual_vol"] = ls_ret.std() * np.sqrt(252)
        metrics["LS_sharpe"] = metrics["LS_annual_return"] / metrics["LS_annual_vol"] if metrics["LS_annual_vol"] > 0 else 0

        cum_ret = (1 + ls_ret).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        metrics["LS_max_drawdown"] = drawdown.min()

    return metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None,
) -> Tuple[Any, pd.DataFrame]:
    import lightgbm as lgb

    default_params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 64,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 4,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train.values, label=y_train.values)
    valid_data = lgb.Dataset(X_test.values, label=y_test.values, reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict(X_test.values)
    pred_series = pd.Series(pred, index=X_test.index, name="pred")

    result_df = X_test.copy()
    result_df["pred"] = pred_series
    result_df["LABEL0"] = y_test

    return model, result_df


def _get_multiindex_cols(df, group_key):
    loc = df.columns.get_loc(group_key)
    if isinstance(loc, slice):
        return df.columns[loc]
    elif isinstance(loc, list):
        return [df.columns[i] for i, v in enumerate(loc) if v]
    else:
        return df.columns[loc]


def prepare_features_and_labels(
    df: pd.DataFrame,
    is_multi_index_cols: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    if is_multi_index_cols and isinstance(df.columns, pd.MultiIndex):
        feature_col_tuples = _get_multiindex_cols(df, "feature")

        X = df[feature_col_tuples].copy()
        X.columns = X.columns.droplevel(0)
        y = df[("label", "LABEL0")]
    else:
        feature_cols = [c for c in df.columns if not str(c).startswith("LABEL")]

        X = df[feature_cols].copy()
        y = df["LABEL0"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def label_zscore_comparison(
    aligned_df: pd.DataFrame,
    is_multi_index_cols: bool = True,
    test_start: str = None,
    test_end: str = None,
    lgbm_params: Dict[str, Any] = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("LABEL ZSCORE COMPARISON (3.1)")
    print("=" * 80)

    X, y_raw = prepare_features_and_labels(aligned_df, is_multi_index_cols)

    y_zscore = y_raw.groupby(level="datetime", group_keys=False).apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    y_zscore.name = y_raw.name

    if test_start is None:
        dates = X.index.get_level_values("datetime")
        test_start = str(dates.max() - pd.Timedelta(days=365))

    train_mask = X.index.get_level_values("datetime") < test_start
    if test_end:
        test_mask = X.index.get_level_values("datetime") >= test_start
        test_mask &= X.index.get_level_values("datetime") < test_end
    else:
        test_mask = X.index.get_level_values("datetime") >= test_start

    X_train, X_test = X[train_mask], X[test_mask]
    y_raw_train, y_raw_test = y_raw[train_mask], y_raw[test_mask]
    y_zscore_train, y_zscore_test = y_zscore[train_mask], y_zscore[test_mask]

    print(f"\n  Train: {X_train.index.get_level_values('datetime').min()} ~ "
          f"{X_train.index.get_level_values('datetime').max()} ({len(X_train)} samples)")
    print(f"  Test:  {X_test.index.get_level_values('datetime').min()} ~ "
          f"{X_test.index.get_level_values('datetime').max()} ({len(X_test)} samples)")

    print("\n  [Method A] Training with raw label...")
    try:
        model_a, result_a = train_lightgbm(X_train, y_raw_train, X_test, y_raw_test, lgbm_params)
        daily_ic_a = compute_daily_ic(result_a, label_col="LABEL0", pred_col="pred")
        ls_a = compute_long_short_return(result_a, label_col="LABEL0", pred_col="pred")
        metrics_a = compute_performance_metrics(daily_ic_a, ls_a)
        print("    Done.")
    except Exception as e:
        logger.error(f"Method A failed: {e}")
        metrics_a = {}
        daily_ic_a = pd.DataFrame()
        ls_a = pd.DataFrame()

    print("  [Method B] Training with z-score label...")
    try:
        model_b, result_b = train_lightgbm(X_train, y_zscore_train, X_test, y_zscore_test, lgbm_params)
        daily_ic_b = compute_daily_ic(result_b, label_col="LABEL0", pred_col="pred")
        ls_b = compute_long_short_return(result_b, label_col="LABEL0", pred_col="pred")
        metrics_b = compute_performance_metrics(daily_ic_b, ls_b)
        print("    Done.")
    except Exception as e:
        logger.error(f"Method B failed: {e}")
        metrics_b = {}
        daily_ic_b = pd.DataFrame()
        ls_b = pd.DataFrame()

    print("\n  --- Performance Comparison ---")
    print(f"  {'Metric':<25} {'Raw Label (A)':<20} {'ZScore Label (B)':<20}")
    print("  " + "-" * 65)
    for key in ["IC_mean", "IC_std", "ICIR", "Rank_IC_mean", "Rank_IC_std", "Rank_ICIR",
                "LS_annual_return", "LS_max_drawdown", "LS_sharpe"]:
        val_a = metrics_a.get(key, float("nan"))
        val_b = metrics_b.get(key, float("nan"))
        print(f"  {key:<25} {val_a:<20.6f} {val_b:<20.6f}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_df = pd.DataFrame({
            "Metric": list(metrics_a.keys()),
            "Raw_Label_A": list(metrics_a.values()),
            "ZScore_Label_B": [metrics_b.get(k, float("nan")) for k in metrics_a.keys()],
        })
        comparison_df.to_csv(output_dir / "label_zscore_comparison.csv", index=False)

        if len(daily_ic_a) > 0:
            daily_ic_a.to_csv(output_dir / "daily_ic_raw_label.csv")
        if len(daily_ic_b) > 0:
            daily_ic_b.to_csv(output_dir / "daily_ic_zscore_label.csv")

        try:
            _plot_label_zscore_comparison(daily_ic_a, daily_ic_b, ls_a, ls_b, output_dir)
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    return {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "daily_ic_a": daily_ic_a,
        "daily_ic_b": daily_ic_b,
        "ls_a": ls_a,
        "ls_b": ls_b,
    }


def data_window_comparison(
    aligned_df: pd.DataFrame,
    is_multi_index_cols: bool = True,
    test_years: int = 1,
    lgbm_params: Dict[str, Any] = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("DATA WINDOW COMPARISON: 3yr vs 6yr (3.2)")
    print("=" * 80)

    X, y = prepare_features_and_labels(aligned_df, is_multi_index_cols)

    dates = X.index.get_level_values("datetime")
    max_date = dates.max()
    test_start = max_date - pd.Timedelta(days=test_years * 365)

    train_start_3yr = max_date - pd.Timedelta(days=4 * 365)
    train_start_6yr = max_date - pd.Timedelta(days=7 * 365)

    train_mask_3yr = (dates >= train_start_3yr) & (dates < test_start)
    train_mask_6yr = (dates >= train_start_6yr) & (dates < test_start)
    test_mask = dates >= test_start

    X_train_3yr, X_test = X[train_mask_3yr], X[test_mask]
    y_train_3yr, y_test = y[train_mask_3yr], y[test_mask]
    X_train_6yr = X[train_mask_6yr]
    y_train_6yr = y[train_mask_6yr]

    print(f"\n  3yr train: {X_train_3yr.index.get_level_values('datetime').min()} ~ "
          f"{X_train_3yr.index.get_level_values('datetime').max()} ({len(X_train_3yr)} samples)")
    print(f"  6yr train: {X_train_6yr.index.get_level_values('datetime').min()} ~ "
          f"{X_train_6yr.index.get_level_values('datetime').max()} ({len(X_train_6yr)} samples)")
    print(f"  Test:      {X_test.index.get_level_values('datetime').min()} ~ "
          f"{X_test.index.get_level_values('datetime').max()} ({len(X_test)} samples)")

    print("\n  [3yr Window] Training...")
    try:
        model_3yr, result_3yr = train_lightgbm(X_train_3yr, y_train_3yr, X_test, y_test, lgbm_params)
        daily_ic_3yr = compute_daily_ic(result_3yr, label_col="LABEL0", pred_col="pred")
        ls_3yr = compute_long_short_return(result_3yr, label_col="LABEL0", pred_col="pred")
        metrics_3yr = compute_performance_metrics(daily_ic_3yr, ls_3yr)
        print("    Done.")
    except Exception as e:
        logger.error(f"3yr training failed: {e}")
        metrics_3yr = {}
        daily_ic_3yr = pd.DataFrame()
        ls_3yr = pd.DataFrame()

    print("  [6yr Window] Training...")
    try:
        model_6yr, result_6yr = train_lightgbm(X_train_6yr, y_train_6yr, X_test, y_test, lgbm_params)
        daily_ic_6yr = compute_daily_ic(result_6yr, label_col="LABEL0", pred_col="pred")
        ls_6yr = compute_long_short_return(result_6yr, label_col="LABEL0", pred_col="pred")
        metrics_6yr = compute_performance_metrics(daily_ic_6yr, ls_6yr)
        print("    Done.")
    except Exception as e:
        logger.error(f"6yr training failed: {e}")
        metrics_6yr = {}
        daily_ic_6yr = pd.DataFrame()
        ls_6yr = pd.DataFrame()

    print("\n  --- Performance Comparison ---")
    print(f"  {'Metric':<25} {'3yr Window':<20} {'6yr Window':<20}")
    print("  " + "-" * 65)
    for key in ["IC_mean", "IC_std", "ICIR", "Rank_IC_mean", "Rank_IC_std", "Rank_ICIR",
                "LS_annual_return", "LS_max_drawdown", "LS_sharpe"]:
        val_3 = metrics_3yr.get(key, float("nan"))
        val_6 = metrics_6yr.get(key, float("nan"))
        print(f"  {key:<25} {val_3:<20.6f} {val_6:<20.6f}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_df = pd.DataFrame({
            "Metric": list(metrics_3yr.keys()),
            "3yr_Window": list(metrics_3yr.values()),
            "6yr_Window": [metrics_6yr.get(k, float("nan")) for k in metrics_3yr.keys()],
        })
        comparison_df.to_csv(output_dir / "data_window_comparison.csv", index=False)

        if len(daily_ic_3yr) > 0:
            daily_ic_3yr.to_csv(output_dir / "daily_ic_3yr.csv")
        if len(daily_ic_6yr) > 0:
            daily_ic_6yr.to_csv(output_dir / "daily_ic_6yr.csv")

        try:
            _plot_data_window_comparison(daily_ic_3yr, daily_ic_6yr, ls_3yr, ls_6yr, output_dir)
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    return {
        "metrics_3yr": metrics_3yr,
        "metrics_6yr": metrics_6yr,
        "daily_ic_3yr": daily_ic_3yr,
        "daily_ic_6yr": daily_ic_6yr,
        "ls_3yr": ls_3yr,
        "ls_6yr": ls_6yr,
    }


def _plot_label_zscore_comparison(daily_ic_a, daily_ic_b, ls_a, ls_b, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    if len(daily_ic_a) > 0 and len(daily_ic_b) > 0:
        ax = axes[0, 0]
        ic_a = daily_ic_a["IC"].rolling(20, min_periods=1).mean()
        ic_b = daily_ic_b["IC"].rolling(20, min_periods=1).mean()
        ax.plot(ic_a.index, ic_a.values, label="Raw Label (A)", alpha=0.8)
        ax.plot(ic_b.index, ic_b.values, label="ZScore Label (B)", alpha=0.8)
        ax.set_title("IC Series Comparison (20-day MA)")
        ax.set_ylabel("IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ric_a = daily_ic_a["Rank_IC"].rolling(20, min_periods=1).mean()
        ric_b = daily_ic_b["Rank_IC"].rolling(20, min_periods=1).mean()
        ax.plot(ric_a.index, ric_a.values, label="Raw Label (A)", alpha=0.8)
        ax.plot(ric_b.index, ric_b.values, label="ZScore Label (B)", alpha=0.8)
        ax.set_title("Rank IC Series Comparison (20-day MA)")
        ax.set_ylabel("Rank IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(ls_a) > 0 and len(ls_b) > 0:
        ax = axes[1, 0]
        cum_a = (1 + ls_a["ls_return"]).cumprod()
        cum_b = (1 + ls_b["ls_return"]).cumprod()
        ax.plot(cum_a.index, cum_a.values, label="Raw Label (A)", alpha=0.8)
        ax.plot(cum_b.index, cum_b.values, label="ZScore Label (B)", alpha=0.8)
        ax.set_title("Long-Short Cumulative Return")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    metrics_a = compute_performance_metrics(daily_ic_a, ls_a)
    metrics_b = compute_performance_metrics(daily_ic_b, ls_b)
    ax = axes[1, 1]
    metrics_names = ["IC_mean", "ICIR", "Rank_IC_mean", "Rank_ICIR"]
    vals_a = [metrics_a.get(m, 0) for m in metrics_names]
    vals_b = [metrics_b.get(m, 0) for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width / 2, vals_a, width, label="Raw Label (A)")
    ax.bar(x + width / 2, vals_b, width, label="ZScore Label (B)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45)
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "label_zscore_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_dir / 'label_zscore_comparison.png'}")


def _plot_data_window_comparison(daily_ic_3yr, daily_ic_6yr, ls_3yr, ls_6yr, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    if len(daily_ic_3yr) > 0 and len(daily_ic_6yr) > 0:
        ax = axes[0, 0]
        ic_3 = daily_ic_3yr["IC"].rolling(20, min_periods=1).mean()
        ic_6 = daily_ic_6yr["IC"].rolling(20, min_periods=1).mean()
        ax.plot(ic_3.index, ic_3.values, label="3yr Window", alpha=0.8)
        ax.plot(ic_6.index, ic_6.values, label="6yr Window", alpha=0.8)
        ax.set_title("IC Series Comparison (20-day MA)")
        ax.set_ylabel("IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ric_3 = daily_ic_3yr["Rank_IC"].rolling(20, min_periods=1).mean()
        ric_6 = daily_ic_6yr["Rank_IC"].rolling(20, min_periods=1).mean()
        ax.plot(ric_3.index, ric_3.values, label="3yr Window", alpha=0.8)
        ax.plot(ric_6.index, ric_6.values, label="6yr Window", alpha=0.8)
        ax.set_title("Rank IC Series Comparison (20-day MA)")
        ax.set_ylabel("Rank IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(ls_3yr) > 0 and len(ls_6yr) > 0:
        ax = axes[1, 0]
        cum_3 = (1 + ls_3yr["ls_return"]).cumprod()
        cum_6 = (1 + ls_6yr["ls_return"]).cumprod()
        ax.plot(cum_3.index, cum_3.values, label="3yr Window", alpha=0.8)
        ax.plot(cum_6.index, cum_6.values, label="6yr Window", alpha=0.8)
        ax.set_title("Long-Short Cumulative Return")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    metrics_3yr = compute_performance_metrics(daily_ic_3yr, ls_3yr)
    metrics_6yr = compute_performance_metrics(daily_ic_6yr, ls_6yr)
    ax = axes[1, 1]
    metrics_names = ["IC_mean", "ICIR", "Rank_IC_mean", "Rank_ICIR"]
    vals_3 = [metrics_3yr.get(m, 0) for m in metrics_names]
    vals_6 = [metrics_6yr.get(m, 0) for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width / 2, vals_3, width, label="3yr Window")
    ax.bar(x + width / 2, vals_6, width, label="6yr Window")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45)
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "data_window_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_dir / 'data_window_comparison.png'}")
