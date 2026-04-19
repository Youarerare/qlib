import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

QLIB_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(QLIB_REPO_ROOT))


@dataclass
class AlignmentResult:
    alpha158_raw_count: int = 0
    alpha50_raw_count: int = 0
    alpha158_processed_count: int = 0
    alpha50_processed_count: int = 0
    alpha158_aligned_count: int = 0
    alpha50_aligned_count: int = 0
    intersection_count: int = 0
    alpha158_only_count: int = 0
    alpha50_only_count: int = 0
    alpha158_only_samples: Optional[pd.DataFrame] = None
    alpha50_only_samples: Optional[pd.DataFrame] = None
    difference_analysis: List[Dict[str, Any]] = field(default_factory=list)
    is_aligned: bool = False


def load_alpha158_data(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2023-06-01",
    freq: str = "day",
) -> pd.DataFrame:
    try:
        import qlib
        from qlib.config import C
        from qlib.data import D
        from qlib.contrib.data.handler import Alpha158

        provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
        if not C.registered:
            qlib.init(provider_uri=str(provider_uri), region="cn")

        handler = Alpha158(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=[],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            process_type="append",
        )

        raw_df = handler.fetch(col_set=handler.CS_RAW, data_key=handler.DK_R)
        learn_df = handler.fetch(col_set=handler.CS_ALL, data_key=handler.DK_L)

        return raw_df, learn_df

    except Exception as e:
        logger.error(f"Failed to load Alpha158 data: {e}")
        logger.info("Attempting manual data loading...")
        return _load_alpha158_manual(instruments, start_time, end_time, freq)


def _load_alpha158_manual(
    instruments: str,
    start_time: str,
    end_time: str,
    freq: str,
) -> pd.DataFrame:
    import qlib
    from qlib.config import C
    from qlib.data import D

    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    if not C.registered:
        qlib.init(provider_uri=str(provider_uri), region="cn")

    from qlib.contrib.data.loader import Alpha158DL

    fields, names = Alpha158DL.get_feature_config()
    label_fields = ["Ref($close, -2)/Ref($close, -1) - 1"]
    label_names = ["LABEL0"]

    feature_config = (fields, names)
    label_config = (label_fields, label_names)

    instruments_config = D.instruments(instruments)

    feature_data = D.features(
        instruments_config,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )
    feature_data.columns = names

    label_data = D.features(
        instruments_config,
        fields=label_fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )
    label_data.columns = label_names

    raw_df = pd.concat(
        {"feature": feature_data, "label": label_data},
        axis=1,
    )

    return raw_df, None


def load_alpha50_data(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2023-06-01",
    pickle_path: str = None,
) -> pd.DataFrame:
    if pickle_path is None:
        pickle_path = str(QLIB_REPO_ROOT / "examples/alpha_factor_test/top50_features.pkl")

    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Alpha50 pickle file not found: {pickle_path}\n"
            f"Please run: python examples/alpha_factor_test/topk_alpha_handler.py --prepare"
        )

    data = pd.read_pickle(pickle_path)

    if start_time:
        data = data.loc[data.index.get_level_values("datetime") >= start_time]
    if end_time:
        data = data.loc[data.index.get_level_values("datetime") <= end_time]

    return data


def _get_multiindex_cols(df, group_key):
    loc = df.columns.get_loc(group_key)
    if isinstance(loc, slice):
        return df.columns[loc]
    elif isinstance(loc, list):
        return [df.columns[i] for i, v in enumerate(loc) if v]
    else:
        return df.columns[loc]


def apply_unified_preprocessing(
    df: pd.DataFrame,
    is_multi_index_cols: bool = True,
    mad_threshold: float = 3.0,
    zscore_clip: float = 3.0,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    result = df.copy()

    if is_multi_index_cols and isinstance(result.columns, pd.MultiIndex):
        feature_col_tuples = _get_multiindex_cols(result, "feature")
        label_col_tuples = _get_multiindex_cols(result, "label")

        feature_df = result[feature_col_tuples].copy()
        feature_df = feature_df.groupby(level="instrument", group_keys=False).apply(
            lambda x: x.ffill(limit=ffill_limit)
        )
        result[feature_col_tuples] = feature_df

        for col_tuple in feature_col_tuples:
            series = result[col_tuple]
            median = series.groupby(level="datetime").transform("median")
            mad = (series - median).abs().groupby(level="datetime").transform("median") * 1.4826
            mad = mad.replace(0, 1e-8)
            z = (series - median) / mad
            result[col_tuple] = z.clip(-mad_threshold, mad_threshold)

        for col_tuple in label_col_tuples:
            series = result[col_tuple]
            mean = series.groupby(level="datetime").transform("mean")
            std = series.groupby(level="datetime").transform("std")
            std = std.replace(0, 1e-8)
            z = (series - mean) / std
            result[col_tuple] = z.clip(-zscore_clip, zscore_clip)
    else:
        feature_cols = [c for c in result.columns if not str(c).startswith("LABEL")]
        label_cols = [c for c in result.columns if str(c).startswith("LABEL")]

        result[feature_cols] = result[feature_cols].groupby(
            level="instrument", group_keys=False
        ).apply(lambda x: x.ffill(limit=ffill_limit))

        for col in feature_cols:
            series = result[col]
            median = series.groupby(level="datetime").transform("median")
            mad = (series - median).abs().groupby(level="datetime").transform("median") * 1.4826
            mad = mad.replace(0, 1e-8)
            z = (series - median) / mad
            result[col] = z.clip(-mad_threshold, mad_threshold)

        for col in label_cols:
            series = result[col]
            mean = series.groupby(level="datetime").transform("mean")
            std = series.groupby(level="datetime").transform("std")
            std = std.replace(0, 1e-8)
            z = (series - mean) / std
            result[col] = z.clip(-zscore_clip, zscore_clip)

    return result


def apply_unified_sample_rules(
    df: pd.DataFrame,
    is_multi_index_cols: bool = True,
    min_consecutive_days: int = 60,
    drop_any_nan: bool = True,
) -> pd.DataFrame:
    result = df.copy()

    if drop_any_nan:
        if is_multi_index_cols and isinstance(result.columns, pd.MultiIndex):
            result = result.dropna()
        else:
            result = result.dropna()

    if min_consecutive_days > 0:
        def filter_short_stocks(group):
            if len(group) < min_consecutive_days:
                return group.head(0)
            return group

        result = result.groupby(level="instrument", group_keys=False).apply(
            filter_short_stocks
        )

    return result


def align_and_compare(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2023-06-01",
    freq: str = "day",
    alpha50_pickle_path: str = None,
    mad_threshold: float = 3.0,
    zscore_clip: float = 3.0,
    min_consecutive_days: int = 60,
    drop_any_nan: bool = True,
) -> AlignmentResult:
    result = AlignmentResult()

    print("\n" + "=" * 80)
    print("ALIGNMENT EXPERIMENT")
    print("=" * 80)

    print("\n[1/6] Loading Alpha158 raw data...")
    try:
        a158_raw, a158_learn = load_alpha158_data(instruments, start_time, end_time, freq)
        result.alpha158_raw_count = len(a158_raw)
        print(f"  Alpha158 raw samples: {result.alpha158_raw_count}")
    except Exception as e:
        logger.error(f"Failed to load Alpha158: {e}")
        print(f"  ERROR: {e}")
        return result

    print("\n[2/6] Loading Alpha50 raw data...")
    try:
        a50_raw = load_alpha50_data(instruments, start_time, end_time, alpha50_pickle_path)
        result.alpha50_raw_count = len(a50_raw)
        print(f"  Alpha50 raw samples: {result.alpha50_raw_count}")
    except Exception as e:
        logger.error(f"Failed to load Alpha50: {e}")
        print(f"  ERROR: {e}")
        return result

    print("\n[3/6] Applying unified preprocessing...")
    a158_is_multi = isinstance(a158_raw.columns, pd.MultiIndex)
    a50_is_multi = isinstance(a50_raw.columns, pd.MultiIndex)

    a158_processed = apply_unified_preprocessing(
        a158_raw, is_multi_index_cols=a158_is_multi,
        mad_threshold=mad_threshold, zscore_clip=zscore_clip,
    )
    a50_processed = apply_unified_preprocessing(
        a50_raw, is_multi_index_cols=a50_is_multi,
        mad_threshold=mad_threshold, zscore_clip=zscore_clip,
    )

    print("\n[4/6] Applying unified sample rules...")
    a158_aligned = apply_unified_sample_rules(
        a158_processed, is_multi_index_cols=a158_is_multi,
        min_consecutive_days=min_consecutive_days, drop_any_nan=drop_any_nan,
    )
    a50_aligned = apply_unified_sample_rules(
        a50_processed, is_multi_index_cols=a50_is_multi,
        min_consecutive_days=min_consecutive_days, drop_any_nan=drop_any_nan,
    )

    result.alpha158_processed_count = len(a158_processed)
    result.alpha50_processed_count = len(a50_processed)
    result.alpha158_aligned_count = len(a158_aligned)
    result.alpha50_aligned_count = len(a50_aligned)

    print(f"  Alpha158: raw={result.alpha158_raw_count} -> "
          f"processed={result.alpha158_processed_count} -> "
          f"aligned={result.alpha158_aligned_count}")
    print(f"  Alpha50:  raw={result.alpha50_raw_count} -> "
          f"processed={result.alpha50_processed_count} -> "
          f"aligned={result.alpha50_aligned_count}")

    print("\n[5/6] Computing intersection and differences...")
    a158_keys = set(a158_aligned.index)
    a50_keys = set(a50_aligned.index)

    intersection = a158_keys & a50_keys
    a158_only = a158_keys - a50_keys
    a50_only = a50_keys - a158_keys

    result.intersection_count = len(intersection)
    result.alpha158_only_count = len(a158_only)
    result.alpha50_only_count = len(a50_only)

    print(f"  Intersection: {result.intersection_count}")
    print(f"  Alpha158 only: {result.alpha158_only_count}")
    print(f"  Alpha50 only:  {result.alpha50_only_count}")

    if a158_only:
        a158_only_df = a158_aligned.loc[list(a158_only)]
        result.alpha158_only_samples = a158_only_df
    if a50_only:
        a50_only_df = a50_aligned.loc[list(a50_only)]
        result.alpha50_only_samples = a50_only_df

    print("\n[6/6] Analyzing differences...")
    result.difference_analysis = _analyze_differences(
        a158_raw, a50_raw, a158_aligned, a50_aligned,
        a158_only, a50_only, a158_is_multi, a50_is_multi,
    )

    result.is_aligned = (result.alpha158_only_count == 0 and result.alpha50_only_count == 0)

    _print_alignment_result(result)

    return result


def _analyze_differences(
    a158_raw, a50_raw, a158_aligned, a50_aligned,
    a158_only, a50_only, a158_is_multi, a50_is_multi,
) -> List[Dict[str, Any]]:
    analysis = []

    a158_dates = a158_raw.index.get_level_values("datetime")
    a50_dates = a50_raw.index.get_level_values("datetime")
    a158_instruments = a158_raw.index.get_level_values("instrument")
    a50_instruments = a50_raw.index.get_level_values("instrument")

    a158_date_range = (a158_dates.min(), a158_dates.max())
    a50_date_range = (a50_dates.min(), a50_dates.max())

    analysis.append({
        "aspect": "Date Range",
        "alpha158": f"{a158_date_range[0]} ~ {a158_date_range[1]}",
        "alpha50": f"{a50_date_range[0]} ~ {a50_date_range[1]}",
        "match": a158_date_range == a50_date_range,
    })

    a158_unique_dates = set(a158_dates.unique())
    a50_unique_dates = set(a50_dates.unique())
    dates_only_in_158 = a158_unique_dates - a50_unique_dates
    dates_only_in_50 = a50_unique_dates - a158_unique_dates

    analysis.append({
        "aspect": "Trading Days",
        "alpha158": f"{len(a158_unique_dates)} unique dates",
        "alpha50": f"{len(a50_unique_dates)} unique dates",
        "match": len(dates_only_in_158) == 0 and len(dates_only_in_50) == 0,
        "detail": f"Dates only in Alpha158: {len(dates_only_in_158)}, "
                  f"only in Alpha50: {len(dates_only_in_50)}",
    })

    a158_unique_stocks = set(a158_instruments.unique())
    a50_unique_stocks = set(a50_instruments.unique())
    stocks_only_in_158 = a158_unique_stocks - a50_unique_stocks
    stocks_only_in_50 = a50_unique_stocks - a158_unique_stocks

    analysis.append({
        "aspect": "Stock Pool",
        "alpha158": f"{len(a158_unique_stocks)} unique stocks",
        "alpha50": f"{len(a50_unique_stocks)} unique stocks",
        "match": len(stocks_only_in_158) == 0 and len(stocks_only_in_50) == 0,
        "detail": f"Stocks only in Alpha158: {len(stocks_only_in_158)}, "
                  f"only in Alpha50: {len(stocks_only_in_50)}",
    })

    if stocks_only_in_158:
        sample_stocks = list(stocks_only_in_158)[:10]
        analysis.append({
            "aspect": "Stocks only in Alpha158 (sample)",
            "value": str(sample_stocks),
            "reason": "These stocks are in the dynamic instrument pool but not in the pre-computed pickle",
        })

    if stocks_only_in_50:
        sample_stocks = list(stocks_only_in_50)[:10]
        analysis.append({
            "aspect": "Stocks only in Alpha50 (sample)",
            "value": str(sample_stocks),
            "reason": "These stocks were in the pickle but may not be in the current dynamic instrument pool",
        })

    if a158_only:
        a158_only_dates = pd.Index([idx[0] for idx in a158_only])
        a158_only_stocks = pd.Index([idx[1] for idx in a158_only])
        date_dist = a158_only_dates.value_counts().head(10)
        stock_dist = a158_only_stocks.value_counts().head(10)

        analysis.append({
            "aspect": "Alpha158-only samples distribution",
            "total": len(a158_only),
            "top_dates": date_dist.to_dict(),
            "top_stocks": stock_dist.to_dict(),
        })

    if a50_only:
        a50_only_dates = pd.Index([idx[0] for idx in a50_only])
        a50_only_stocks = pd.Index([idx[1] for idx in a50_only])
        date_dist = a50_only_dates.value_counts().head(10)
        stock_dist = a50_only_stocks.value_counts().head(10)

        analysis.append({
            "aspect": "Alpha50-only samples distribution",
            "total": len(a50_only),
            "top_dates": date_dist.to_dict(),
            "top_stocks": stock_dist.to_dict(),
        })

    a158_nan_before = a158_raw.isna().any(axis=1).sum() if not a158_is_multi else \
        a158_raw.isna().any(axis=1).sum()
    a50_nan_before = a50_raw.isna().any(axis=1).sum()

    analysis.append({
        "aspect": "NaN rows before processing",
        "alpha158": int(a158_nan_before),
        "alpha50": int(a50_nan_before),
    })

    return analysis


def _print_alignment_result(result: AlignmentResult):
    print("\n" + "-" * 60)
    print("ALIGNMENT RESULT SUMMARY")
    print("-" * 60)
    print(f"  Alpha158: {result.alpha158_aligned_count} samples after alignment")
    print(f"  Alpha50:  {result.alpha50_aligned_count} samples after alignment")
    print(f"  Intersection: {result.intersection_count}")
    print(f"  Alpha158 only: {result.alpha158_only_count}")
    print(f"  Alpha50 only:  {result.alpha50_only_count}")
    print(f"  Fully aligned: {'YES' if result.is_aligned else 'NO'}")

    if not result.is_aligned:
        print("\n  DIFFERENCE ANALYSIS:")
        for item in result.difference_analysis:
            aspect = item.get("aspect", "")
            if "match" in item:
                match_str = "MATCH" if item["match"] else "MISMATCH"
                print(f"    [{match_str}] {aspect}")
                if "detail" in item:
                    print(f"      Detail: {item['detail']}")
                if "alpha158" in item and "alpha50" in item:
                    print(f"      Alpha158: {item['alpha158']}")
                    print(f"      Alpha50:  {item['alpha50']}")
            else:
                print(f"    {aspect}: {item.get('value', item.get('total', 'N/A'))}")
                if "reason" in item:
                    print(f"      Reason: {item['reason']}")

    print("-" * 60)


def _recompute_label_alpha158_style(
    df: pd.DataFrame,
    instruments: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    import qlib
    from qlib.config import C
    from qlib.data import D

    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    if not C.registered:
        qlib.init(provider_uri=str(provider_uri), region="cn")

    label_fields = ["Ref($close, -2)/Ref($close, -1) - 1"]
    instruments_config = D.instruments(instruments)
    label_data = D.features(
        instruments_config,
        fields=label_fields,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    label_data.columns = ["LABEL0"]
    return label_data["LABEL0"]


def _get_dynamic_stock_pool_per_date(
    instruments: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    import qlib
    from qlib.config import C
    from qlib.data import D

    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    if not C.registered:
        qlib.init(provider_uri=str(provider_uri), region="cn")

    inst_config = D.instruments(instruments)
    stock_list = D.list_instruments(
        inst_config, start_time=start_time, end_time=end_time,
        freq="day", as_list=True,
    )

    all_dates = D.calendar(start_time=start_time, end_time=end_time, freq="day")

    pool_data = []
    for date in all_dates:
        date_inst = D.list_instruments(
            D.instruments(instruments),
            start_time=date,
            end_time=date,
            freq="day",
            as_list=True,
        )
        for stock in date_inst:
            pool_data.append({"datetime": date, "instrument": stock})

    if not pool_data:
        return pd.DataFrame()

    pool_df = pd.DataFrame(pool_data)
    pool_df["datetime"] = pd.to_datetime(pool_df["datetime"])
    pool_df = pool_df.set_index(["datetime", "instrument"])
    pool_df["in_pool"] = True
    return pool_df


def align_alpha50_to_alpha158(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2023-06-01",
    freq: str = "day",
    alpha50_pickle_path: str = None,
    min_consecutive_days: int = 60,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("ALIGN ALPHA50 -> ALPHA158 (对齐实验)")
    print("=" * 80)
    print("\n对齐策略:")
    print("  1. 股票池: 用Alpha158的动态股票池过滤Alpha50数据")
    print("  2. 标签公式: Alpha50的 pct_change(1).shift(-1) -> Alpha158的 Ref($close,-2)/Ref($close,-1)-1")
    print("  3. 处理器管线: Alpha50的 infer_processors 清空, 仅在 learn_processors 中保留 DropnaLabel + CSZScoreNorm(label)")
    print("  4. NaN策略: 统一dropna, 与Alpha158一致")

    print("\n" + "-" * 60)
    print("[1/6] 加载Alpha158数据 (作为基准)...")
    try:
        a158_raw, a158_learn = load_alpha158_data(instruments, start_time, end_time, freq)
        print(f"  Alpha158 raw samples: {len(a158_raw)}")
    except Exception as e:
        logger.error(f"Failed to load Alpha158: {e}")
        print(f"  ERROR: {e}")
        return {}

    print("\n[2/6] 加载Alpha50原始数据...")
    try:
        a50_raw = load_alpha50_data(instruments, start_time, end_time, alpha50_pickle_path)
        print(f"  Alpha50 raw samples: {len(a50_raw)}")
    except Exception as e:
        logger.error(f"Failed to load Alpha50: {e}")
        print(f"  ERROR: {e}")
        return {}

    a158_is_multi = isinstance(a158_raw.columns, pd.MultiIndex)
    a50_is_multi = isinstance(a50_raw.columns, pd.MultiIndex)
    print(f"  Alpha158 columns type: {'MultiIndex' if a158_is_multi else 'Flat'}")
    print(f"  Alpha50 columns type: {'MultiIndex' if a50_is_multi else 'Flat'}")

    print("\n[3/6] 用Alpha158动态股票池过滤Alpha50...")
    a158_keys = set(a158_raw.index)
    a50_keys = set(a50_raw.index)
    common_keys = a158_keys & a50_keys
    a50_aligned = a50_raw.loc[a50_raw.index.isin(common_keys)]
    a50_aligned = a50_aligned.sort_index()
    print(f"  Alpha50 过滤前: {len(a50_raw)}, 过滤后: {len(a50_aligned)}")
    print(f"  移除了 {len(a50_raw) - len(a50_aligned)} 个不在Alpha158动态股票池中的样本")

    print("\n[4/6] 重算Alpha50标签 (Alpha158公式: Ref($close,-2)/Ref($close,-1)-1)...")
    try:
        if a158_is_multi:
            a158_label = a158_raw[("label", "LABEL0")].copy()
        else:
            a158_label = a158_raw["LABEL0"].copy()

        new_label_aligned = a158_label.reindex(a50_aligned.index)
        old_label_nan = a50_aligned["LABEL0"].isna().sum() if "LABEL0" in a50_aligned.columns else "N/A"
        new_label_nan = new_label_aligned.isna().sum()
        a50_aligned["LABEL0"] = new_label_aligned
        print(f"  旧标签NaN数: {old_label_nan}")
        print(f"  新标签NaN数: {new_label_nan}")
        print(f"  标签来源: 直接从Alpha158 raw数据中提取")
    except Exception as e:
        logger.error(f"重算标签失败: {e}")
        print(f"  ERROR: {e}")
        return {}

    print("\n[5/6] 应用Alpha158的处理器管线 (DropnaLabel + CSZScoreNorm)...")
    a50_before_dropna = len(a50_aligned)

    label_cols = [c for c in a50_aligned.columns if str(c).startswith("LABEL")]
    a50_aligned = a50_aligned.dropna(subset=label_cols)
    print(f"  Alpha50 DropnaLabel: {a50_before_dropna} -> {len(a50_aligned)} (移除 {a50_before_dropna - len(a50_aligned)})")

    for col in label_cols:
        series = a50_aligned[col]
        mean = series.groupby(level="datetime").transform("mean")
        std = series.groupby(level="datetime").transform("std")
        std = std.replace(0, 1e-8)
        z = (series - mean) / std
        a50_aligned[col] = z.clip(-3, 3)

    a158_processed = a158_raw.copy()
    a158_before_dropna = len(a158_processed)

    if a158_is_multi:
        label_col_tuples_158 = _get_multiindex_cols(a158_processed, "label")
        label_nan_mask_158 = a158_processed[label_col_tuples_158].isna().any(axis=1)
        a158_processed = a158_processed[~label_nan_mask_158]
        print(f"  Alpha158 DropnaLabel: {a158_before_dropna} -> {len(a158_processed)} (移除 {a158_before_dropna - len(a158_processed)})")

        for col_tuple in label_col_tuples_158:
            series = a158_processed[col_tuple]
            mean = series.groupby(level="datetime").transform("mean")
            std = series.groupby(level="datetime").transform("std")
            std = std.replace(0, 1e-8)
            z = (series - mean) / std
            a158_processed[col_tuple] = z.clip(-3, 3)
    else:
        label_cols_158 = [c for c in a158_processed.columns if str(c).startswith("LABEL")]
        a158_processed = a158_processed.dropna(subset=label_cols_158)
        print(f"  Alpha158 DropnaLabel: {a158_before_dropna} -> {len(a158_processed)}")

        for col in label_cols_158:
            series = a158_processed[col]
            mean = series.groupby(level="datetime").transform("mean")
            std = series.groupby(level="datetime").transform("std")
            std = std.replace(0, 1e-8)
            z = (series - mean) / std
            a158_processed[col] = z.clip(-3, 3)

    print("\n[6/6] 比较对齐后的样本量...")
    a158_keys_final = set(a158_processed.index)
    a50_keys_final = set(a50_aligned.index)
    intersection = a158_keys_final & a50_keys_final
    a158_only = a158_keys_final - a50_keys_final
    a50_only = a50_keys_final - a158_keys_final

    print("\n" + "=" * 80)
    print("对齐结果")
    print("=" * 80)
    print(f"  Alpha158 最终样本量: {len(a158_processed)}")
    print(f"  Alpha50  最终样本量: {len(a50_aligned)}")
    print(f"  交集: {len(intersection)}")
    print(f"  Alpha158独有: {len(a158_only)}")
    print(f"  Alpha50独有:  {len(a50_only)}")
    print(f"  样本量是否一致: {'YES' if len(a158_only) == 0 and len(a50_only) == 0 else 'NO'}")

    if len(a158_only) > 0 or len(a50_only) > 0:
        print("\n  差异分析:")

        if a158_only:
            a158_only_dates = pd.Index([idx[0] for idx in a158_only])
            a158_only_stocks = pd.Index([idx[1] for idx in a158_only])
            print(f"\n  Alpha158独有样本 ({len(a158_only)}):")
            print(f"    涉及日期数: {a158_only_dates.nunique()}")
            print(f"    涉及股票数: {a158_only_stocks.nunique()}")
            top_stocks = a158_only_stocks.value_counts().head(5)
            print(f"    最多的5只股票: {top_stocks.to_dict()}")

            a158_only_df = a158_processed.loc[list(a158_only)]
            if a158_is_multi:
                feature_col_tuples = _get_multiindex_cols(a158_only_df, "feature")
                nan_in_features = a158_only_df[feature_col_tuples].isna().any(axis=1).sum()
            else:
                feature_cols = [c for c in a158_only_df.columns if not str(c).startswith("LABEL")]
                nan_in_features = a158_only_df[feature_cols].isna().any(axis=1).sum()
            print(f"    其中特征含NaN的行数: {nan_in_features}")

            a158_only_sample = list(a158_only)[:3]
            print(f"    示例(date,stock): {a158_only_sample}")

        if a50_only:
            a50_only_dates = pd.Index([idx[0] for idx in a50_only])
            a50_only_stocks = pd.Index([idx[1] for idx in a50_only])
            print(f"\n  Alpha50独有样本 ({len(a50_only)}):")
            print(f"    涉及日期数: {a50_only_dates.nunique()}")
            print(f"    涉及股票数: {a50_only_stocks.nunique()}")
            top_stocks = a50_only_stocks.value_counts().head(5)
            print(f"    最多的5只股票: {top_stocks.to_dict()}")

            a50_only_sample = list(a50_only)[:3]
            print(f"    示例(date,stock): {a50_only_sample}")
    else:
        print("\n  [SUCCESS] 两个数据集样本量完全一致! 对齐成功!")

    print("=" * 80)

    result = {
        "a158_final_count": len(a158_processed),
        "a50_final_count": len(a50_aligned),
        "intersection": len(intersection),
        "a158_only": len(a158_only),
        "a50_only": len(a50_only),
        "is_aligned": len(a158_only) == 0 and len(a50_only) == 0,
        "a158_processed": a158_processed,
        "a50_aligned": a50_aligned,
    }

    output_dir = Path(__file__).parent / "output" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
    pd.Series(summary).to_csv(output_dir / "alpha50_to_alpha158_alignment.csv")
    print(f"\n结果已保存到: {output_dir / 'alpha50_to_alpha158_alignment.csv'}")

    return result


def save_alignment_data(result: AlignmentResult, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "alpha158_raw_count": result.alpha158_raw_count,
        "alpha50_raw_count": result.alpha50_raw_count,
        "alpha158_processed_count": result.alpha158_processed_count,
        "alpha50_processed_count": result.alpha50_processed_count,
        "alpha158_aligned_count": result.alpha158_aligned_count,
        "alpha50_aligned_count": result.alpha50_aligned_count,
        "intersection_count": result.intersection_count,
        "alpha158_only_count": result.alpha158_only_count,
        "alpha50_only_count": result.alpha50_only_count,
        "is_aligned": result.is_aligned,
    }

    pd.Series(summary).to_csv(output_dir / "alignment_summary.csv")

    if result.alpha158_only_samples is not None:
        result.alpha158_only_samples.to_csv(output_dir / "alpha158_only_samples.csv")
    if result.alpha50_only_samples is not None:
        result.alpha50_only_samples.to_csv(output_dir / "alpha50_only_samples.csv")

    diff_df = pd.DataFrame(result.difference_analysis)
    diff_df.to_csv(output_dir / "difference_analysis.csv", index=False)

    logger.info(f"Alignment data saved to {output_dir}")
