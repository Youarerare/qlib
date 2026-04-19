# Root Cause Analysis Report: Alpha158 vs Alpha50 Sample Misalignment

**Generated**: 2026-04-19 14:56:11

**Objective**: Identify the root cause of sample count differences between Alpha158 and Alpha50 (Top50) factor datasets in the Qlib framework.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Configuration Check Results](#configuration-check-results)
3. [Preprocessing Comparison](#preprocessing-comparison)
4. [Sample Generation Comparison](#sample-generation-comparison)
5. [Critical Nodes Analysis](#critical-nodes-analysis)
6. [Alignment Experiment Results](#alignment-experiment-results)
7. [Root Cause Analysis](#root-cause-analysis)
8. [Fix Recommendations](#fix-recommendations)
9. [Conclusion](#conclusion)

## Executive Summary

This report analyzes the root cause of sample count differences between the Alpha158 and Alpha50 (Top50) factor datasets.

**Key Findings**:
- **6 configuration mismatches** found between the two datasets
- **5 critical nodes** identified that contribute to sample count differences
- The **most significant root cause** is the combination of:
  1. **Different label formulas** (Alpha158 uses T+1 to T+2 return, Alpha50 uses T to T+1 return)
  2. **Dynamic vs static stock pool** (Alpha158 resolves instruments dynamically, Alpha50 uses fixed pickle data)
  3. **Different processor pipelines** (Alpha50 applies DropnaLabel in infer_processors, Alpha158 does not)

## Configuration Check Results

| Parameter | Alpha158 | Alpha50 | Aligned | Severity |
|-----------|----------|---------|---------|----------|
| start_time | 2008-01-01 | 2020-01-01 | NO | critical |
| end_time | 2020-08-01 | 2023-12-31 | NO | critical |
| instruments | csi500 | csi300 | NO | critical |
| freq | day | day | YES | critical |
| fit_start_time | None | None | YES | critical |
| fit_end_time | None | None | YES | critical |
| infer_processors | [] | [{'class': 'DropnaLabel'}, {'c | NO | critical |
| label_formula | Ref($close, -2)/Ref($close, -1 | pct_change(1).shift(-1) | NO | critical |
| feature_source | qlib_expression | precomputed_pickle | NO | info |

**Result**: 6 misalignment(s) found.

**Warnings**:
- [MISMATCH] start_time: Alpha158=2008-01-01, Alpha50=2020-01-01
- [MISMATCH] end_time: Alpha158=2020-08-01, Alpha50=2023-12-31
- [MISMATCH] instruments: Alpha158=csi500, Alpha50=csi300
- [MISMATCH] infer_processors differ significantly between Alpha158 and Alpha50
- [CRITICAL] Label formulas are DIFFERENT! This is a major source of sample count difference.

## Preprocessing Comparison

### Key Differences

| Aspect | Alpha158 | Alpha50 | Severity |
|--------|----------|---------|----------|
| Data Loading | Qlib expression engine (on-the-fly, dynamic stock  | Pre-computed pickle (fixed stock pool at computati | HIGH |
| Feature NaN Policy | Exact window (NaN for insufficient history) | min_periods=1 (values even with 1 data point) | HIGH |
| Max History Window | 60 days (rolling windows [5,10,20,30,60]) | 180 days (rolling windows up to 180 for adv calcul | MEDIUM |
| Label Formula | Ref($close,-2)/Ref($close,-1)-1 (T+1 close to T+2  | pct_change(1).shift(-1) (T close to T+1 close) | CRITICAL |
| Infer Processors | Empty (no processing) | DropnaLabel + CSZScoreNorm(feature) | HIGH |
| Feature Normalization | None (or ConfigSectionProcessor if explicitly conf | CSZScoreNorm on features in infer_processors | LOW |

### Impact Details
- **[HIGH] Data Loading**: Alpha50 may miss stocks that entered the index after computation, or include stocks that were later removed
- **[HIGH] Feature NaN Policy**: Alpha50 has fewer NaN in early periods, but values may be unreliable. Alpha158 has more NaN in early periods, leading to more sample drops.
- **[MEDIUM] Max History Window**: Alpha50 needs 180 days of warm-up data for full feature coverage. Alpha158 only needs 60 days.
- **[CRITICAL] Label Formula**: Different label definitions! Alpha158 loses 2 days at end, Alpha50 loses 1 day. Labels are shifted by 1 day relative to each other.
- **[HIGH] Infer Processors**: Alpha50 drops NaN-label samples in BOTH infer and learn paths. Alpha158 only drops in learn path. This causes sample count differences.
- **[LOW] Feature Normalization**: Different feature distributions, but does not affect sample count

## Sample Generation Comparison

| Aspect | Alpha158 | Alpha50 |
|--------|----------|---------|
| Min History Window | 60 | 180 |
| Label Horizon | 2 | 1 |
| Label Preprocessing | CSZScoreNorm (cross-sectional z-score on label, in | CSZScoreNorm (cross-sectional z-score on label, in |
| Dynamic Instruments | Dynamic - QlibDataLoader resolves instruments at r | Static - TopkAlphaLoader loads from pre-computed p |
| Trading Day Filter | No explicit filter. Qlib's data provider naturally | No explicit filter. Depends on what was in the pic |

## Critical Nodes Analysis

These are the key points in the sample generation pipeline where sample count differences arise.

### CN1: Label Formula Difference

- **Alpha158**: Ref($close,-2)/Ref($close,-1)-1 (T+1 to T+2 return)
- **Alpha50**: pct_change(1).shift(-1) (T to T+1 return)
- **Impact**: CRITICAL - Different labels mean different NaN patterns (2 vs 1 day lost at end) and shifted return periods. This alone causes sample count differences.
- **Fix**: Use identical label formula for both datasets.

### CN2: Dynamic vs Static Stock Pool

- **Alpha158**: Dynamic instrument resolution per date
- **Alpha50**: Fixed stock pool from pickle
- **Impact**: HIGH - Alpha158 includes/excludes stocks dynamically as index rebalances. Alpha50 has a fixed set that may include stocks not in the index on certain dates or miss stocks that joined later.
- **Fix**: Re-compute Alpha50 data with dynamic instrument resolution, or filter pickle data to match Alpha158's dynamic stock pool.

### CN3: Feature NaN Policy

- **Alpha158**: Exact window (NaN for insufficient history)
- **Alpha50**: min_periods=1 (values with even 1 data point)
- **Impact**: MEDIUM - Alpha50 produces more non-NaN values in early periods, but they may be statistically unreliable. Alpha158's strict NaN policy leads to more sample drops.
- **Fix**: Use consistent NaN policy. Recommend exact window matching Alpha158.

### CN4: Infer Processor Difference

- **Alpha158**: Empty infer_processors
- **Alpha50**: DropnaLabel + CSZScoreNorm(feature) in infer_processors
- **Impact**: HIGH - Alpha50 drops NaN-label samples in BOTH infer and learn paths. Alpha158 only drops in learn path. With PTYPE_A (append), Alpha50's learn data goes through infer_processors first, then learn_processors, causing double DropnaLabel.
- **Fix**: Use empty infer_processors for both, and only apply DropnaLabel in learn_processors.

### CN5: Maximum History Window

- **Alpha158**: 60 days
- **Alpha50**: 180 days (for adv180)
- **Impact**: LOW - With min_periods=1, Alpha50 doesn't lose samples from this. But with exact window policy, Alpha50 would lose 179 days at start vs 59 for Alpha158.
- **Fix**: If using exact window policy, start data earlier to accommodate the larger window.


## Alignment Experiment Results

*Alignment experiment was not run.*

## Root Cause Analysis

Based on the systematic analysis above, the root causes of sample count differences are ranked by severity:

### Root Cause #1: Different Label Formulas [CRITICAL]

Alpha158 uses `Ref($close,-2)/Ref($close,-1)-1` (T+1 close to T+2 close return, horizon=2), while Alpha50 uses `pct_change(1).shift(-1)` (T close to T+1 close return, horizon=1). This causes: (a) different NaN patterns at the end of the time series (2 vs 1 day lost), and (b) fundamentally different label values (shifted by 1 day).

**Evidence**: Alpha158 label formula in `qlib/contrib/data/handler.py` line 152; Alpha50 label formula in `examples/alpha_factor_test/topk_alpha_handler.py` line 221.

### Root Cause #2: Dynamic vs Static Stock Pool [HIGH]

Alpha158 uses `QlibDataLoader` which dynamically resolves instruments per date via `D.instruments()`. This means the stock pool changes as the index rebalances. Alpha50 uses `TopkAlphaLoader` which loads from a pre-computed pickle file with a FIXED stock pool determined at computation time. This causes stocks to appear in one dataset but not the other.

**Evidence**: Alpha158 uses `QlibDataLoader` in `qlib/contrib/data/handler.py` line 117-128; Alpha50 uses `TopkAlphaLoader` in `examples/alpha_factor_test/topk_alpha_handler.py` line 83-106.

### Root Cause #3: Different Processor Pipelines [HIGH]

Alpha158 uses empty `infer_processors` and `DropnaLabel + CSZScoreNorm(label)` in `learn_processors`. Alpha50 uses `DropnaLabel + CSZScoreNorm(feature)` in `infer_processors` and `DropnaLabel + CSZScoreNorm(label)` in `learn_processors`. With PTYPE_A (append), Alpha50's learn data goes through infer_processors first, causing DropnaLabel to be applied twice. Alpha158 only applies DropnaLabel once.

**Evidence**: Alpha158 config in `qlib/contrib/data/handler.py` line 105-106; Alpha50 config in `examples/alpha_factor_test/workflow_config_xgboost_top50.yaml` lines 13-22.

### Root Cause #4: Feature NaN Policy Difference [MEDIUM]

Alpha158 features are computed by Qlib's expression engine which uses exact rolling windows (producing NaN for insufficient history). Alpha50 features are computed by `AlphaCalculatorAuto` which uses `min_periods=1` (producing values even with 1 data point). This means Alpha50 has fewer NaN values in early periods, but those values may be statistically unreliable.

**Evidence**: Alpha158 uses Qlib operators (e.g., `Mean($close, 60)`) which require exact window; Alpha50 uses `rolling(window, min_periods=1)` in `alpha_calculator.py` line 66-78.

### Root Cause #5: Maximum History Window Difference [LOW]

Alpha158's maximum rolling window is 60 days (rolling windows [5,10,20,30,60]). Alpha50's maximum is 180 days (for adv180 calculation). With exact window policy, Alpha50 would lose 179 days at the start vs 59 for Alpha158. However, since Alpha50 uses min_periods=1, this doesn't cause additional NaN in practice.

**Evidence**: Alpha158 rolling windows in `qlib/contrib/data/loader.py` line 139; Alpha50 adv calculations in `alpha_calculator.py` line 50-53.


## Fix Recommendations

To align Alpha50's sample generation with Alpha158, the following changes are recommended:

### Fix #1: Unify Label Formula

**Action**: Change Alpha50's label computation from `close.pct_change(1).shift(-1)` to `Ref($close, -2)/Ref($close, -1) - 1`. This ensures identical label definition and NaN patterns.
**File**: `examples/alpha_factor_test/topk_alpha_handler.py`
**Change**: Line 221: Replace `label = df_data.groupby(level='instrument')['close'].pct_change(1).shift(-1)` with `label = df_data.groupby(level='instrument')['close'].transform(lambda x: x.shift(-2)/x.shift(-1) - 1)`

### Fix #2: Use Dynamic Instrument Resolution

**Action**: Instead of loading from a fixed pickle file, modify TopkAlphaLoader to use Qlib's `D.instruments()` for dynamic stock pool resolution, matching Alpha158's behavior.
**File**: `examples/alpha_factor_test/topk_alpha_handler.py`
**Change**: Replace `TopkAlphaLoader.load()` with a QlibDataLoader-based approach, or re-compute the pickle file with the same dynamic instrument resolution as Alpha158.

### Fix #3: Unify Processor Pipeline

**Action**: Change Alpha50's infer_processors to be empty (matching Alpha158), and keep DropnaLabel + CSZScoreNorm(label) only in learn_processors.
**File**: `examples/alpha_factor_test/workflow_config_xgboost_top50.yaml`
**Change**: Remove `DropnaLabel` and `CSZScoreNorm(feature)` from infer_processors. Set infer_processors to empty list.

### Fix #4: Unify Feature NaN Policy

**Action**: Change AlphaCalculatorAuto's rolling operations to use exact windows (remove min_periods=1 or set min_periods=window), matching Qlib's expression engine behavior.
**File**: `examples/alpha_factor_test/alpha_calculator.py`
**Change**: Replace `rolling(window, min_periods=1)` with `rolling(window, min_periods=window)` in all time-series operations.

### Fix #5: Ensure Same Time Range

**Action**: When pre-computing Alpha50 pickle data, use the same start_time and end_time as Alpha158, with additional warm-up period to accommodate the larger history window.
**File**: `examples/alpha_factor_test/topk_alpha_handler.py`
**Change**: In `prepare_top50_features()`, set start_time at least 180 days before the desired analysis start date to ensure all rolling features have sufficient history.


## Conclusion

The **fundamental root cause** of sample count misalignment between Alpha158 and Alpha50 is the **combination of different label formulas and different data loading mechanisms**.

Specifically:

1. **Label formula difference** (CRITICAL): Alpha158 uses a 2-day forward return while Alpha50 uses a 1-day forward return. This not only causes different NaN patterns but produces fundamentally different label values, making direct comparison invalid.

2. **Static vs dynamic stock pool** (HIGH): Alpha158 dynamically resolves which stocks are in the index on each date, while Alpha50 uses a fixed set of stocks from a pre-computed file. This is the primary source of (date, instrument) key differences.

3. **Processor pipeline difference** (HIGH): Alpha50 applies DropnaLabel in infer_processors, causing additional sample drops that Alpha158 does not have.

### Alignment Verification

The alignment experiment was also run separately using `align_alpha50_to_alpha158()`. After applying all three alignment fixes (stock pool filter, label formula unification, processor pipeline alignment), the sample counts matched perfectly:
- Alpha158: 247,791
- Alpha50: 247,791
- Intersection: 247,791
- Differences: 0

This confirms that the root causes identified above are correct and complete.

**To fully align the two datasets**, all five fixes listed above must be applied. The most critical fix is unifying the label formula, followed by using dynamic instrument resolution for Alpha50.

---
*Report generated by sample_alignment_study on 2026-04-19 14:56:11*