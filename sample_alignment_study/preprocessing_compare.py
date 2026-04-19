import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStep:
    name: str
    description: str
    parameters: Dict[str, Any]
    affects_sample_count: bool
    sample_impact: str = ""


@dataclass
class PreprocessingComparison:
    alpha158_steps: List[PreprocessingStep] = field(default_factory=list)
    alpha50_steps: List[PreprocessingStep] = field(default_factory=list)
    differences: List[Dict[str, Any]] = field(default_factory=list)
    unified_pipeline: List[PreprocessingStep] = field(default_factory=list)


class PreprocessingComparator:
    def __init__(self):
        self.comparison = PreprocessingComparison()

    def analyze_alpha158_preprocessing(self) -> List[PreprocessingStep]:
        steps = [
            PreprocessingStep(
                name="1. Data Loading (QlibDataLoader)",
                description="Load raw OHLCV data via Qlib expression engine. "
                            "Features computed on-the-fly using Qlib operators (Mean, Std, Ref, etc.). "
                            "Qlib engine handles NaN by skipping invalid windows.",
                parameters={"source": "Qlib expression engine", "nan_handling": "automatic"},
                affects_sample_count=False,
                sample_impact="No direct sample loss; NaN produced where insufficient history",
            ),
            PreprocessingStep(
                name="2. Feature Computation (Alpha158DL)",
                description="Compute 158 features: 9 kbar + price ratios + rolling operators. "
                            "Rolling windows: [5, 10, 20, 30, 60]. Max history needed: 60 days. "
                            "Qlib expression engine uses exact window (no min_periods), "
                            "producing NaN for first N-1 days of each rolling feature.",
                parameters={
                    "rolling_windows": [5, 10, 20, 30, 60],
                    "max_history_window": 60,
                    "nan_policy": "exact_window_no_min_periods",
                },
                affects_sample_count=True,
                sample_impact="First 59 trading days per stock will have NaN in rolling features",
            ),
            PreprocessingStep(
                name="3. Label Computation",
                description="Label = Ref($close, -2)/Ref($close, -1) - 1. "
                            "This is the return from T+1 close to T+2 close. "
                            "The last 2 trading days have NaN labels (future data unavailable).",
                parameters={
                    "formula": "Ref($close, -2)/Ref($close, -1) - 1",
                    "horizon": 2,
                    "nan_at_end": 2,
                },
                affects_sample_count=True,
                sample_impact="Last 2 trading days per stock have NaN label",
            ),
            PreprocessingStep(
                name="4. Infer Processors (empty)",
                description="Alpha158 uses empty infer_processors by default. "
                            "No processing applied to inference data.",
                parameters={},
                affects_sample_count=False,
            ),
            PreprocessingStep(
                name="5. Learn Processors: DropnaLabel",
                description="Drop rows where label is NaN. This removes samples where "
                            "future return cannot be computed (last 2 days) or label is invalid.",
                parameters={"fields_group": "label"},
                affects_sample_count=True,
                sample_impact="Drops ~2 days per stock at end of period; drops any stock with missing close prices",
            ),
            PreprocessingStep(
                name="6. Learn Processors: CSZScoreNorm (label)",
                description="Cross-sectional z-score normalization on labels. "
                            "For each date: y_norm = (y - mean(y)) / std(y). "
                            "Does NOT change sample count, only transforms values.",
                parameters={"fields_group": "label", "method": "zscore"},
                affects_sample_count=False,
            ),
        ]
        self.comparison.alpha158_steps = steps
        return steps

    def analyze_alpha50_preprocessing(self) -> List[PreprocessingStep]:
        steps = [
            PreprocessingStep(
                name="1. Data Loading (TopkAlphaLoader)",
                description="Load pre-computed factor data from pickle file. "
                            "Data was computed offline using AlphaCalculatorAuto. "
                            "The pickle file contains a fixed set of stocks and dates "
                            "determined at computation time.",
                parameters={
                    "source": "precomputed pickle",
                    "nan_handling": "rolling with min_periods=1",
                },
                affects_sample_count=True,
                sample_impact="Pickle data may cover different stock pool / date range than Alpha158",
            ),
            PreprocessingStep(
                name="2. Feature Computation (AlphaCalculatorAuto)",
                description="Compute Top50 factors from Alpha101/Alpha191 formulas. "
                            "Uses pandas rolling with min_periods=1, which produces values "
                            "even with insufficient history (unlike Qlib's exact window). "
                            "Rolling windows: [5, 10, 15, 20, 30, 40, 50, 60, 80, 120, 150, 180]. "
                            "Max history needed: 180 days.",
                parameters={
                    "rolling_windows": [5, 10, 15, 20, 30, 40, 50, 60, 80, 120, 150, 180],
                    "max_history_window": 180,
                    "nan_policy": "min_periods=1 (produces values even with 1 data point)",
                },
                affects_sample_count=True,
                sample_impact="More early samples survive (min_periods=1), but values may be unreliable. "
                              "Some Alpha101/191 formulas may fail and produce all-NaN columns.",
            ),
            PreprocessingStep(
                name="3. Label Computation",
                description="Label = close.pct_change(1).shift(-1). "
                            "This is the return from T close to T+1 close. "
                            "The last 1 trading day has NaN label.",
                parameters={
                    "formula": "close.pct_change(1).shift(-1)",
                    "horizon": 1,
                    "nan_at_end": 1,
                },
                affects_sample_count=True,
                sample_impact="Last 1 trading day per stock has NaN label. "
                              "DIFFERENT from Alpha158 which loses 2 days!",
            ),
            PreprocessingStep(
                name="4. Infer Processors: DropnaLabel",
                description="Drop rows where label is NaN. Applied in infer_processors "
                            "(unusual - typically DropnaLabel is only in learn_processors).",
                parameters={"fields_group": "label"},
                affects_sample_count=True,
                sample_impact="Drops last 1 day per stock; also drops rows with NaN label",
            ),
            PreprocessingStep(
                name="5. Infer Processors: CSZScoreNorm (feature)",
                description="Cross-sectional z-score normalization on FEATURES. "
                            "Applied in infer_processors. This is unusual - typically "
                            "feature normalization is done in learn_processors or via ConfigSectionProcessor.",
                parameters={"fields_group": "feature", "method": "zscore"},
                affects_sample_count=False,
                sample_impact="Does not change sample count, but normalization differs from Alpha158",
            ),
            PreprocessingStep(
                name="6. Learn Processors: DropnaLabel",
                description="Drop rows where label is NaN (redundant with infer step).",
                parameters={"fields_group": "label"},
                affects_sample_count=False,
                sample_impact="Already applied in infer_processors",
            ),
            PreprocessingStep(
                name="7. Learn Processors: CSZScoreNorm (label)",
                description="Cross-sectional z-score normalization on labels.",
                parameters={"fields_group": "label", "method": "zscore"},
                affects_sample_count=False,
            ),
        ]
        self.comparison.alpha50_steps = steps
        return steps

    def compare(self) -> PreprocessingComparison:
        self.analyze_alpha158_preprocessing()
        self.analyze_alpha50_preprocessing()

        differences = [
            {
                "aspect": "Data Loading",
                "alpha158": "Qlib expression engine (on-the-fly, dynamic stock pool)",
                "alpha50": "Pre-computed pickle (fixed stock pool at computation time)",
                "impact": "Alpha50 may miss stocks that entered the index after computation, "
                          "or include stocks that were later removed",
                "severity": "HIGH",
            },
            {
                "aspect": "Feature NaN Policy",
                "alpha158": "Exact window (NaN for insufficient history)",
                "alpha50": "min_periods=1 (values even with 1 data point)",
                "impact": "Alpha50 has fewer NaN in early periods, but values may be unreliable. "
                          "Alpha158 has more NaN in early periods, leading to more sample drops.",
                "severity": "HIGH",
            },
            {
                "aspect": "Max History Window",
                "alpha158": "60 days (rolling windows [5,10,20,30,60])",
                "alpha50": "180 days (rolling windows up to 180 for adv calculations)",
                "impact": "Alpha50 needs 180 days of warm-up data for full feature coverage. "
                          "Alpha158 only needs 60 days.",
                "severity": "MEDIUM",
            },
            {
                "aspect": "Label Formula",
                "alpha158": "Ref($close,-2)/Ref($close,-1)-1 (T+1 close to T+2 close)",
                "alpha50": "pct_change(1).shift(-1) (T close to T+1 close)",
                "impact": "Different label definitions! Alpha158 loses 2 days at end, Alpha50 loses 1 day. "
                          "Labels are shifted by 1 day relative to each other.",
                "severity": "CRITICAL",
            },
            {
                "aspect": "Infer Processors",
                "alpha158": "Empty (no processing)",
                "alpha50": "DropnaLabel + CSZScoreNorm(feature)",
                "impact": "Alpha50 drops NaN-label samples in BOTH infer and learn paths. "
                          "Alpha158 only drops in learn path. This causes sample count differences.",
                "severity": "HIGH",
            },
            {
                "aspect": "Feature Normalization",
                "alpha158": "None (or ConfigSectionProcessor if explicitly configured)",
                "alpha50": "CSZScoreNorm on features in infer_processors",
                "impact": "Different feature distributions, but does not affect sample count",
                "severity": "LOW",
            },
        ]
        self.comparison.differences = differences

        self.comparison.unified_pipeline = [
            PreprocessingStep(
                name="1. Load Raw Data",
                description="Load both datasets with identical instruments, start_time, end_time, freq. "
                            "For Alpha50, re-compute with matching parameters or filter existing pickle data.",
                parameters={"method": "unified_config"},
                affects_sample_count=True,
                sample_impact="Ensures same stock pool and date range",
            ),
            PreprocessingStep(
                name="2. Forward Fill Missing Values",
                description="Fill NaN values using forward fill within each stock's time series. "
                            "This handles trading suspension and missing data consistently.",
                parameters={"method": "ffill", "limit": 5},
                affects_sample_count=False,
            ),
            PreprocessingStep(
                name="3. MAD Outlier Removal",
                description="Remove extreme values using Median Absolute Deviation. "
                            "Values beyond threshold * MAD from median are clipped.",
                parameters={"threshold": 3.0, "method": "clip"},
                affects_sample_count=False,
            ),
            PreprocessingStep(
                name="4. Cross-sectional Z-score Standardization",
                description="Standardize features and labels cross-sectionally (per date). "
                            "y_norm = (y - mean(y)) / std(y) for each trading day.",
                parameters={"method": "zscore", "clip": 3.0},
                affects_sample_count=False,
            ),
            PreprocessingStep(
                name="5. Unified Label Computation",
                description="Use Ref($close,-2)/Ref($close,-1)-1 as label for both datasets. "
                            "This ensures identical label definition and NaN pattern.",
                parameters={"formula": "Ref($close,-2)/Ref($close,-1)-1", "horizon": 2},
                affects_sample_count=True,
                sample_impact="Last 2 trading days per stock have NaN label",
            ),
            PreprocessingStep(
                name="6. Drop NaN Samples",
                description="Drop any sample where label OR any feature is NaN. "
                            "This ensures both datasets have exactly the same valid samples.",
                parameters={"drop_features_nan": True, "drop_label_nan": True},
                affects_sample_count=True,
                sample_impact="Removes all samples with any missing data",
            ),
        ]

        return self.comparison

    def print_comparison(self):
        if not self.comparison.alpha158_steps:
            self.compare()

        print("\n" + "=" * 80)
        print("PREPROCESSING COMPARISON")
        print("=" * 80)

        print("\n--- Alpha158 Preprocessing Steps ---")
        for step in self.comparison.alpha158_steps:
            impact_flag = " [AFFECTS SAMPLE COUNT]" if step.affects_sample_count else ""
            print(f"\n  {step.name}{impact_flag}")
            print(f"    Description: {step.description}")
            print(f"    Parameters: {step.parameters}")
            if step.sample_impact:
                print(f"    Sample Impact: {step.sample_impact}")

        print("\n--- Alpha50 Preprocessing Steps ---")
        for step in self.comparison.alpha50_steps:
            impact_flag = " [AFFECTS SAMPLE COUNT]" if step.affects_sample_count else ""
            print(f"\n  {step.name}{impact_flag}")
            print(f"    Description: {step.description}")
            print(f"    Parameters: {step.parameters}")
            if step.sample_impact:
                print(f"    Sample Impact: {step.sample_impact}")

        print("\n--- Key Differences ---")
        for diff in self.comparison.differences:
            print(f"\n  [{diff['severity']}] {diff['aspect']}")
            print(f"    Alpha158: {diff['alpha158']}")
            print(f"    Alpha50:  {diff['alpha50']}")
            print(f"    Impact: {diff['impact']}")

        print("\n--- Unified Preprocessing Pipeline ---")
        for step in self.comparison.unified_pipeline:
            impact_flag = " [AFFECTS SAMPLE COUNT]" if step.affects_sample_count else ""
            print(f"\n  {step.name}{impact_flag}")
            print(f"    Description: {step.description}")
            print(f"    Parameters: {step.parameters}")

        print("\n" + "=" * 80)


def apply_unified_preprocessing(
    df: pd.DataFrame,
    mad_threshold: float = 3.0,
    zscore_clip: float = 3.0,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        feature_cols = result.columns.get_loc("feature")
        label_cols = result.columns.get_loc("label")
    else:
        feature_cols = [c for c in result.columns if not c.startswith("LABEL")]
        label_cols = [c for c in result.columns if c.startswith("LABEL")]

    if isinstance(result.columns, pd.MultiIndex):
        for col in feature_cols:
            result[("feature", col)] = result[("feature", col)].groupby(
                level="instrument", group_keys=False
            ).apply(lambda x: x.ffill(limit=ffill_limit))
    else:
        result[feature_cols] = result[feature_cols].groupby(
            level="instrument", group_keys=False
        ).apply(lambda x: x.ffill(limit=ffill_limit))

    if isinstance(result.columns, pd.MultiIndex):
        for col in feature_cols:
            series = result[("feature", col)]
            median = series.groupby(level="datetime").transform("median")
            mad = (series - median).abs().groupby(level="datetime").transform("median") * 1.4826
            mad = mad.replace(0, 1e-8)
            z = (series - median) / mad
            result[("feature", col)] = z.clip(-mad_threshold, mad_threshold)
    else:
        for col in feature_cols:
            series = result[col]
            median = series.groupby(level="datetime").transform("median")
            mad = (series - median).abs().groupby(level="datetime").transform("median") * 1.4826
            mad = mad.replace(0, 1e-8)
            z = (series - median) / mad
            result[col] = z.clip(-mad_threshold, mad_threshold)

    if isinstance(result.columns, pd.MultiIndex):
        for col_group in [feature_cols, label_cols]:
            cols_to_norm = result[col_group]
            if isinstance(cols_to_norm, pd.DataFrame):
                for col in cols_to_norm.columns:
                    series = result[(col_group, col)] if isinstance(col_group, str) else result[col]
                    mean = series.groupby(level="datetime").transform("mean")
                    std = series.groupby(level="datetime").transform("std")
                    std = std.replace(0, 1e-8)
                    z = (series - mean) / std
                    if isinstance(col_group, str):
                        result[(col_group, col)] = z.clip(-zscore_clip, zscore_clip)
                    else:
                        result[col] = z.clip(-zscore_clip, zscore_clip)
    else:
        for col in feature_cols + label_cols:
            series = result[col]
            mean = series.groupby(level="datetime").transform("mean")
            std = series.groupby(level="datetime").transform("std")
            std = std.replace(0, 1e-8)
            z = (series - mean) / std
            result[col] = z.clip(-zscore_clip, zscore_clip)

    return result
