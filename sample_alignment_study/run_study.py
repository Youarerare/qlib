"""
Alpha158 vs Alpha50 Sample Misalignment Root Cause Study
Main runner script that executes the full analysis pipeline.

Usage:
    python run_study.py [--config config.yaml] [--skip-followup] [--skip-alignment]

Steps:
    1. Configuration alignment check
    2. Preprocessing comparison
    3. Sample generation logic analysis
    4. Alignment experiment
    5. Root cause report generation
    6. Follow-up analysis (only if alignment succeeds)
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
QLIB_REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(QLIB_REPO_ROOT))


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step1_config_check(config: dict, logger: logging.Logger):
    from config_checker import ConfigChecker

    logger.info("=" * 60)
    logger.info("STEP 1: Configuration Alignment Check")
    logger.info("=" * 60)

    checker = ConfigChecker(str(PROJECT_DIR / "config.yaml"))
    result = checker.check_alignment()
    checker.print_check_result(result)

    corrected_path = PROJECT_DIR / "output" / "data" / "corrected_config.yaml"
    checker.generate_corrected_yaml(result, str(corrected_path))
    logger.info(f"Corrected config saved to: {corrected_path}")

    return result


def step2_preprocessing_compare(config: dict, logger: logging.Logger):
    from preprocessing_compare import PreprocessingComparator

    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing Comparison")
    logger.info("=" * 60)

    comparator = PreprocessingComparator()
    result = comparator.compare()
    comparator.print_comparison()

    return result


def step3_sample_generation(config: dict, logger: logging.Logger):
    from sample_generation_analysis import compare_sample_generation, print_sample_generation_analysis

    logger.info("=" * 60)
    logger.info("STEP 3: Sample Generation Logic Analysis")
    logger.info("=" * 60)

    print_sample_generation_analysis()
    result = compare_sample_generation()

    mermaid_path = PROJECT_DIR / "output" / "data" / "sample_generation_flows.txt"
    mermaid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mermaid_path, "w", encoding="utf-8") as f:
        from sample_generation_analysis import get_alpha158_flow_mermaid, get_alpha50_flow_mermaid
        f.write("Alpha158 Sample Generation Flow:\n")
        f.write(get_alpha158_flow_mermaid())
        f.write("\n\nAlpha50 Sample Generation Flow:\n")
        f.write(get_alpha50_flow_mermaid())
    logger.info(f"Mermaid flow diagrams saved to: {mermaid_path}")

    return result


def step4_alignment_experiment(config: dict, logger: logging.Logger, skip: bool = False):
    from align_and_compare import align_and_compare, save_alignment_data

    logger.info("=" * 60)
    logger.info("STEP 4: Alignment Experiment")
    logger.info("=" * 60)

    if skip:
        logger.info("Skipping alignment experiment (--skip-alignment flag)")
        return None

    alignment_cfg = config.get("alignment", {})

    result = align_and_compare(
        instruments=alignment_cfg.get("unified_instruments", "csi300"),
        start_time=alignment_cfg.get("unified_start_time", "2020-01-01"),
        end_time=alignment_cfg.get("unified_end_time", "2023-06-01"),
        freq=alignment_cfg.get("unified_freq", "day"),
        alpha50_pickle_path=config.get("alpha50", {}).get("pickle_path"),
        mad_threshold=alignment_cfg.get("mad_threshold", 3.0),
        zscore_clip=alignment_cfg.get("zscore_clip", 3.0),
        min_consecutive_days=alignment_cfg.get("min_consecutive_days", 60),
        drop_any_nan=alignment_cfg.get("drop_any_nan", True),
    )

    data_dir = PROJECT_DIR / "output" / "data"
    save_alignment_data(result, str(data_dir))

    return result


def step5_report(config_result, preprocess_result, sample_gen_result, alignment_result, logger: logging.Logger):
    from report_generator import RootCauseReportGenerator

    logger.info("=" * 60)
    logger.info("STEP 5: Root Cause Report Generation")
    logger.info("=" * 60)

    generator = RootCauseReportGenerator(
        config_result=config_result,
        preprocess_result=preprocess_result,
        sample_gen_result=sample_gen_result,
        alignment_result=alignment_result,
    )

    report_path = PROJECT_DIR / "output" / "reports" / "root_cause_report.md"
    generator.generate(str(report_path))

    return report_path


def step6_followup(config: dict, alignment_result, logger: logging.Logger, skip: bool = False):
    from followup_analysis import label_zscore_comparison, data_window_comparison
    from align_and_compare import load_alpha158_data

    logger.info("=" * 60)
    logger.info("STEP 6: Follow-up Analysis")
    logger.info("=" * 60)

    if skip:
        logger.info("Skipping follow-up analysis (--skip-followup flag)")
        return

    if alignment_result is None:
        logger.warning("Alignment result not available. Attempting to load data for follow-up analysis...")
        logger.warning("Follow-up analysis requires aligned data. Skipping.")
        return

    if not alignment_result.is_aligned:
        logger.warning(
            f"Samples are NOT fully aligned "
            f"(Alpha158-only: {alignment_result.alpha158_only_count}, "
            f"Alpha50-only: {alignment_result.alpha50_only_count}). "
            f"Proceeding with intersection data for follow-up analysis."
        )

    alignment_cfg = config.get("alignment", {})
    followup_cfg = config.get("followup", {})
    figure_dir = PROJECT_DIR / "output" / "figures"

    logger.info("\n--- 3.1 Label Z-score Comparison ---")
    try:
        instruments = alignment_cfg.get("unified_instruments", "csi300")
        start_time = alignment_cfg.get("unified_start_time", "2020-01-01")
        end_time = alignment_cfg.get("unified_end_time", "2023-06-01")

        a158_raw, _ = load_alpha158_data(instruments, start_time, end_time)
        is_multi = isinstance(a158_raw.columns, pd.MultiIndex)

        label_zscore_comparison(
            aligned_df=a158_raw,
            is_multi_index_cols=is_multi,
            lgbm_params=followup_cfg.get("lgbm_params"),
            output_dir=str(figure_dir),
        )
    except Exception as e:
        logger.error(f"Label z-score comparison failed: {e}")

    logger.info("\n--- 3.2 Data Window Comparison (3yr vs 6yr) ---")
    try:
        data_window_comparison(
            aligned_df=a158_raw,
            is_multi_index_cols=is_multi,
            test_years=followup_cfg.get("test_period_months", 12) // 12,
            lgbm_params=followup_cfg.get("lgbm_params"),
            output_dir=str(figure_dir),
        )
    except Exception as e:
        logger.error(f"Data window comparison failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Alpha158 vs Alpha50 Sample Misalignment Root Cause Study"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_DIR / "config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-followup",
        action="store_true",
        help="Skip follow-up analysis (3.1 & 3.2)",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip alignment experiment (requires Qlib data)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Run only a specific step (1-6)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_cfg = config.get("output", {})
    logger = setup_logging(output_cfg.get("log_dir", str(PROJECT_DIR / "output" / "logs")))

    logger.info("Alpha158 vs Alpha50 Sample Misalignment Root Cause Study")
    logger.info(f"Config: {args.config}")
    logger.info(f"Start time: {datetime.now()}")

    config_result = None
    preprocess_result = None
    sample_gen_result = None
    alignment_result = None

    try:
        if args.step is None or args.step == 1:
            config_result = step1_config_check(config, logger)

        if args.step is None or args.step == 2:
            preprocess_result = step2_preprocessing_compare(config, logger)

        if args.step is None or args.step == 3:
            sample_gen_result = step3_sample_generation(config, logger)

        if args.step is None or args.step == 4:
            alignment_result = step4_alignment_experiment(
                config, logger, skip=args.skip_alignment
            )

        if args.step is None or args.step == 5:
            if config_result is None:
                from config_checker import ConfigChecker
                checker = ConfigChecker(str(PROJECT_DIR / "config.yaml"))
                config_result = checker.check_alignment()
            if preprocess_result is None:
                from preprocessing_compare import PreprocessingComparator
                comparator = PreprocessingComparator()
                preprocess_result = comparator.compare()
            if sample_gen_result is None:
                from sample_generation_analysis import compare_sample_generation
                sample_gen_result = compare_sample_generation()

            step5_report(
                config_result, preprocess_result, sample_gen_result,
                alignment_result, logger,
            )

        if args.step is None or args.step == 6:
            step6_followup(
                config, alignment_result, logger,
                skip=args.skip_followup,
            )

    except Exception as e:
        logger.exception(f"Study failed with error: {e}")
        raise

    logger.info(f"\nStudy completed at: {datetime.now()}")
    logger.info(f"Output directory: {PROJECT_DIR / 'output'}")
    logger.info(f"Report: {PROJECT_DIR / 'output' / 'reports' / 'root_cause_report.md'}")


if __name__ == "__main__":
    main()
