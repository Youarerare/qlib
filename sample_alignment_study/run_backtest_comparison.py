"""
Alpha158 vs Alpha50 (aligned) - 5-year backtest comparison
Steps:
  1. Load Alpha158 and Alpha50 data
  2. Align Alpha50 to Alpha158 (stock pool + label + processors)
  3. Verify sample counts match
  4. Run 5-year rolling backtest with LightGBM for both
  5. Compare IC, Rank IC, ICIR, long-short returns
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent
QLIB_REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(QLIB_REPO_ROOT))

from align_and_compare import (
    load_alpha158_data,
    load_alpha50_data,
    _get_multiindex_cols,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def align_and_get_data(
    instruments="csi300",
    start_time="2018-01-01",
    end_time="2023-06-01",
):
    print("=" * 80)
    print("STEP 1: Load and Align Data")
    print("=" * 80)

    print("\n[1/4] Loading Alpha158...")
    a158_raw, _ = load_alpha158_data(instruments, start_time, end_time)
    print(f"  Alpha158 raw: {len(a158_raw)}")

    print("\n[2/4] Loading Alpha50...")
    a50_raw = load_alpha50_data(instruments, start_time, end_time)
    print(f"  Alpha50 raw: {len(a50_raw)}")

    a158_is_multi = isinstance(a158_raw.columns, pd.MultiIndex)
    print(f"  Alpha158 columns: {'MultiIndex' if a158_is_multi else 'Flat'}")

    print("\n[3/4] Aligning Alpha50 -> Alpha158...")
    a158_keys = set(a158_raw.index)
    a50_keys = set(a50_raw.index)
    common_keys = a158_keys & a50_keys
    a50_aligned = a50_raw.loc[a50_raw.index.isin(common_keys)].sort_index()
    print(f"  Stock pool filter: {len(a50_raw)} -> {len(a50_aligned)}")

    if a158_is_multi:
        a158_label = a158_raw[("label", "LABEL0")].copy()
    else:
        a158_label = a158_raw["LABEL0"].copy()
    a50_aligned["LABEL0"] = a158_label.reindex(a50_aligned.index)
    print(f"  Label replaced from Alpha158")

    a50_before = len(a50_aligned)
    a50_aligned = a50_aligned.dropna(subset=["LABEL0"])
    print(f"  DropnaLabel: {a50_before} -> {len(a50_aligned)}")

    a158_processed = a158_raw.copy()
    if a158_is_multi:
        lt = _get_multiindex_cols(a158_processed, "label")
        a158_processed = a158_processed[~a158_processed[lt].isna().any(axis=1)]
    else:
        a158_processed = a158_processed.dropna(subset=["LABEL0"])

    print("\n[4/4] Verification:")
    a158_final_keys = set(a158_processed.index)
    a50_final_keys = set(a50_aligned.index)
    intersection = a158_final_keys & a50_final_keys
    a158_only = a158_final_keys - a50_final_keys
    a50_only = a50_final_keys - a158_final_keys

    print(f"  Alpha158 final: {len(a158_processed)}")
    print(f"  Alpha50  final: {len(a50_aligned)}")
    print(f"  Intersection:   {len(intersection)}")
    print(f"  Alpha158 only:  {len(a158_only)}")
    print(f"  Alpha50 only:   {len(a50_only)}")
    print(f"  ALIGNED: {'YES' if len(a158_only) == 0 and len(a50_only) == 0 else 'NO'}")

    return a158_processed, a50_aligned, a158_is_multi


def prepare_Xy(df, is_multi):
    if is_multi and isinstance(df.columns, pd.MultiIndex):
        feat_tuples = _get_multiindex_cols(df, "feature")
        X = df[feat_tuples].copy()
        X.columns = X.columns.droplevel(0)
        y = df[("label", "LABEL0")].copy()
    else:
        feat_cols = [c for c in df.columns if not str(c).startswith("LABEL")]
        X = df[feat_cols].copy()
        y = df["LABEL0"].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def compute_daily_ic(pred_df, pred_col="pred", label_col="LABEL0"):
    results = []
    for date, group in pred_df.groupby(level="datetime"):
        pred = group[pred_col]
        label = group[label_col]
        valid = pred.notna() & label.notna()
        if valid.sum() < 5:
            continue
        ic = pred[valid].corr(label[valid], method="pearson")
        ric = pred[valid].corr(label[valid], method="spearman")
        results.append({"datetime": date, "IC": ic, "Rank_IC": ric})
    return pd.DataFrame(results).set_index("datetime") if results else pd.DataFrame()


def compute_long_short(pred_df, pred_col="pred", label_col="LABEL0", n_groups=5):
    results = []
    for date, group in pred_df.groupby(level="datetime"):
        if len(group) < n_groups * 2:
            continue
        group = group.copy()
        try:
            group["group"] = pd.qcut(group[pred_col], n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        long_ret = group[group["group"] == n_groups - 1][label_col].mean()
        short_ret = group[group["group"] == 0][label_col].mean()
        results.append({
            "datetime": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "ls_return": long_ret - short_ret,
        })
    return pd.DataFrame(results).set_index("datetime") if results else pd.DataFrame()


def rolling_backtest(X, y, test_start, train_years=3, valid_years=1,
                     step_trading_days=20, horizon=2, lgbm_params=None):
    """
    Qlib-style fixed-window sliding rolling backtest.

    Pattern (same as Qlib RollingGen with rtype=SHIFT_SD):
    - train: fixed [train_years] window
    - valid: fixed [valid_years] window right after train
    - test:  [step_trading_days] days right after valid
    - All three windows slide forward by [step_trading_days] each iteration
    - trunc_days = horizon + 1 to prevent label leakage

    Example with train=3yr, valid=1yr, step=20d:
      Roll 1: train=[2018-01, 2020-12], valid=[2021-01, 2021-12], test=[2022-01-01, 2022-01-28]
      Roll 2: train=[2018-01+20d, 2020-12+20d], valid=[...], test=[2022-01-29, 2022-03-xx]
      ...
    """
    import lightgbm as lgb

    if lgbm_params is None:
        lgbm_params = {
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

    dates = X.index.get_level_values("datetime")
    all_unique_dates = sorted(dates.unique())
    test_start_dt = pd.Timestamp(test_start)

    test_start_idx = None
    for i, d in enumerate(all_unique_dates):
        if d >= test_start_dt:
            test_start_idx = i
            break
    if test_start_idx is None:
        print("  ERROR: test_start not found in data date range")
        return pd.DataFrame(), pd.DataFrame(), {}

    train_days = train_years * 252
    valid_days = valid_years * 252
    trunc_days = horizon + 1

    all_preds = []
    roll_count = 0

    test_idx = test_start_idx
    while test_idx < len(all_unique_dates):
        test_end_idx = min(test_idx + step_trading_days - 1, len(all_unique_dates) - 1)

        valid_end_idx = test_idx - 1
        valid_start_idx = valid_end_idx - valid_days + 1

        train_end_idx = valid_start_idx - 1 - trunc_days
        train_start_idx = train_end_idx - train_days + 1

        if train_start_idx < 0 or valid_start_idx < 0:
            test_idx += step_trading_days
            continue

        train_start = all_unique_dates[train_start_idx]
        train_end = all_unique_dates[train_end_idx]
        valid_start = all_unique_dates[valid_start_idx]
        valid_end = all_unique_dates[valid_end_idx]
        test_start_date = all_unique_dates[test_idx]
        test_end_date = all_unique_dates[test_end_idx]

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) < 100 or len(X_test) == 0:
            test_idx += step_trading_days
            continue

        train_data = lgb.Dataset(X_train.values, label=y_train.values)

        if valid_years > 0 and valid_start_idx >= 0:
            valid_mask = (dates >= valid_start) & (dates <= valid_end)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            if len(X_valid) > 0:
                valid_data = lgb.Dataset(X_valid.values, label=y_valid.values, reference=train_data)
                model = lgb.train(
                    lgbm_params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                )
            else:
                model = lgb.train(
                    lgbm_params,
                    train_data,
                    num_boost_round=300,
                    callbacks=[lgb.log_evaluation(0)],
                )
        else:
            model = lgb.train(
                lgbm_params,
                train_data,
                num_boost_round=300,
                callbacks=[lgb.log_evaluation(0)],
            )

        pred = model.predict(X_test.values)
        pred_df = pd.DataFrame({
            "pred": pred,
            "LABEL0": y_test.values,
        }, index=X_test.index)
        all_preds.append(pred_df)

        roll_count += 1
        if roll_count % 5 == 0:
            print(f"    Roll {roll_count}: test=[{test_start_date.strftime('%Y-%m-%d')}, "
                  f"{test_end_date.strftime('%Y-%m-%d')}], "
                  f"train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}")

        test_idx += step_trading_days

    print(f"  Total rolling windows: {roll_count}")

    if not all_preds:
        return pd.DataFrame(), pd.DataFrame(), {}

    result_df = pd.concat(all_preds)
    result_df = result_df[~result_df.index.duplicated(keep="last")]
    result_df = result_df.sort_index()

    daily_ic = compute_daily_ic(result_df)
    ls_df = compute_long_short(result_df)

    metrics = {}
    if len(daily_ic) > 0:
        ic = daily_ic["IC"].dropna()
        ric = daily_ic["Rank_IC"].dropna()
        metrics["IC_mean"] = ic.mean()
        metrics["IC_std"] = ic.std()
        metrics["ICIR"] = ic.mean() / ic.std() if ic.std() > 0 else 0
        metrics["Rank_IC_mean"] = ric.mean()
        metrics["Rank_IC_std"] = ric.std()
        metrics["Rank_ICIR"] = ric.mean() / ric.std() if ric.std() > 0 else 0
        metrics["IC>0_ratio"] = (ic > 0).mean()
        metrics["Rank_IC>0_ratio"] = (ric > 0).mean()

    if len(ls_df) > 0:
        ls_ret = ls_df["ls_return"]
        metrics["LS_annual_return"] = ls_ret.mean() * 252
        metrics["LS_annual_vol"] = ls_ret.std() * np.sqrt(252)
        metrics["LS_sharpe"] = (
            metrics["LS_annual_return"] / metrics["LS_annual_vol"]
            if metrics["LS_annual_vol"] > 0 else 0
        )
        cum = (1 + ls_ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        metrics["LS_max_drawdown"] = dd.min()

    return result_df, daily_ic, metrics


def plot_comparison(daily_ic_158, daily_ic_50, ls_158, ls_50, metrics_158, metrics_50, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Alpha158 vs Alpha50 (Aligned) - 5-Year Backtest Comparison", fontsize=14)

    if len(daily_ic_158) > 0 and len(daily_ic_50) > 0:
        ax = axes[0, 0]
        ic_158_ma = daily_ic_158["IC"].rolling(20, min_periods=1).mean()
        ic_50_ma = daily_ic_50["IC"].rolling(20, min_periods=1).mean()
        ax.plot(ic_158_ma.index, ic_158_ma.values, label="Alpha158", alpha=0.8)
        ax.plot(ic_50_ma.index, ic_50_ma.values, label="Alpha50 (aligned)", alpha=0.8)
        ax.set_title("IC (20-day MA)")
        ax.set_ylabel("IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ric_158_ma = daily_ic_158["Rank_IC"].rolling(20, min_periods=1).mean()
        ric_50_ma = daily_ic_50["Rank_IC"].rolling(20, min_periods=1).mean()
        ax.plot(ric_158_ma.index, ric_158_ma.values, label="Alpha158", alpha=0.8)
        ax.plot(ric_50_ma.index, ric_50_ma.values, label="Alpha50 (aligned)", alpha=0.8)
        ax.set_title("Rank IC (20-day MA)")
        ax.set_ylabel("Rank IC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(ls_158) > 0 and len(ls_50) > 0:
        ax = axes[1, 0]
        cum_158 = (1 + ls_158["ls_return"]).cumprod()
        cum_50 = (1 + ls_50["ls_return"]).cumprod()
        ax.plot(cum_158.index, cum_158.values, label="Alpha158", alpha=0.8)
        ax.plot(cum_50.index, cum_50.values, label="Alpha50 (aligned)", alpha=0.8)
        ax.set_title("Long-Short Cumulative Return")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    key_metrics = ["IC_mean", "ICIR", "Rank_IC_mean", "Rank_ICIR", "LS_sharpe"]
    vals_158 = [metrics_158.get(k, 0) for k in key_metrics]
    vals_50 = [metrics_50.get(k, 0) for k in key_metrics]
    x = np.arange(len(key_metrics))
    width = 0.35
    ax.bar(x - width / 2, vals_158, width, label="Alpha158")
    ax.bar(x + width / 2, vals_50, width, label="Alpha50 (aligned)")
    ax.set_xticks(x)
    ax.set_xticklabels(key_metrics, rotation=45)
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "alpha158_vs_alpha50_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_dir / 'alpha158_vs_alpha50_backtest.png'}")


def main():
    instruments = "csi300"
    start_time = "2018-01-01"
    end_time = "2023-06-01"
    test_start = "2022-07-01"
    train_years = 2
    valid_years = 0

    a158_df, a50_df, a158_is_multi = align_and_get_data(
        instruments, start_time, end_time
    )

    print("\n" + "=" * 80)
    print("STEP 2: Qlib-Style Fixed-Window Rolling Backtest")
    print("=" * 80)
    print(f"  train_years={train_years}, valid_years={valid_years}, step=20 trading days, horizon=2")
    print(f"  (Same as Qlib RollingGen with rtype=SHIFT_SD)")

    print("\nPreparing features and labels...")
    X_158, y_158 = prepare_Xy(a158_df, a158_is_multi)
    X_50, y_50 = prepare_Xy(a50_df, False)

    print(f"  Alpha158: {X_158.shape[0]} samples, {X_158.shape[1]} features")
    print(f"  Alpha50:  {X_50.shape[0]} samples, {X_50.shape[1]} features")
    print(f"  Test start: {test_start}")

    print("\n[1/2] Running Alpha158 backtest...")
    result_158, ic_158, metrics_158 = rolling_backtest(
        X_158, y_158, test_start, train_years=train_years, valid_years=valid_years,
        step_trading_days=20, horizon=2,
    )
    print(f"  Test predictions: {len(result_158)}")
    print(f"  IC_mean={metrics_158.get('IC_mean', 0):.4f}, "
          f"ICIR={metrics_158.get('ICIR', 0):.4f}, "
          f"Rank_IC_mean={metrics_158.get('Rank_IC_mean', 0):.4f}, "
          f"LS_sharpe={metrics_158.get('LS_sharpe', 0):.4f}")

    print("\n[2/2] Running Alpha50 (aligned) backtest...")
    result_50, ic_50, metrics_50 = rolling_backtest(
        X_50, y_50, test_start, train_years=train_years, valid_years=valid_years,
        step_trading_days=20, horizon=2,
    )
    print(f"  Test predictions: {len(result_50)}")
    print(f"  IC_mean={metrics_50.get('IC_mean', 0):.4f}, "
          f"ICIR={metrics_50.get('ICIR', 0):.4f}, "
          f"Rank_IC_mean={metrics_50.get('Rank_IC_mean', 0):.4f}, "
          f"LS_sharpe={metrics_50.get('LS_sharpe', 0):.4f}")

    print("\n" + "=" * 80)
    print("STEP 3: Comparison Summary")
    print("=" * 80)

    all_keys = set(metrics_158.keys()) | set(metrics_50.keys())
    print(f"\n  {'Metric':<25} {'Alpha158':<15} {'Alpha50(aligned)':<15}")
    print("  " + "-" * 55)
    for k in sorted(all_keys):
        v158 = metrics_158.get(k, float("nan"))
        v50 = metrics_50.get(k, float("nan"))
        print(f"  {k:<25} {v158:<15.6f} {v50:<15.6f}")

    output_dir = PROJECT_DIR / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(ic_158) > 0:
        ic_158.to_csv(PROJECT_DIR / "output" / "data" / "backtest_ic_alpha158.csv")
    if len(ic_50) > 0:
        ic_50.to_csv(PROJECT_DIR / "output" / "data" / "backtest_ic_alpha50.csv")

    comparison = pd.DataFrame({
        "Metric": sorted(all_keys),
        "Alpha158": [metrics_158.get(k, float("nan")) for k in sorted(all_keys)],
        "Alpha50_aligned": [metrics_50.get(k, float("nan")) for k in sorted(all_keys)],
    })
    comparison.to_csv(PROJECT_DIR / "output" / "data" / "backtest_comparison.csv", index=False)
    print(f"\n  CSV saved: {PROJECT_DIR / 'output' / 'data' / 'backtest_comparison.csv'}")

    try:
        plot_comparison(ic_158, ic_50, 
                        compute_long_short(result_158) if len(result_158) > 0 else pd.DataFrame(),
                        compute_long_short(result_50) if len(result_50) > 0 else pd.DataFrame(),
                        metrics_158, metrics_50, output_dir)
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
