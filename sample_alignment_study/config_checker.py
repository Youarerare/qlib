import yaml
import logging
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfigDiff:
    parameter: str
    alpha158_value: Any
    alpha50_value: Any
    is_aligned: bool
    severity: str
    suggestion: str


@dataclass
class AlignmentCheckResult:
    diffs: List[ConfigDiff] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unified_config: Dict[str, Any] = field(default_factory=dict)
    is_fully_aligned: bool = True


class ConfigChecker:
    CRITICAL_PARAMS = [
        "start_time", "end_time", "instruments", "freq",
        "fit_start_time", "fit_end_time",
    ]

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.alpha158_cfg = self.config.get("alpha158", {})
        self.alpha50_cfg = self.config.get("alpha50", {})
        self.alignment_cfg = self.config.get("alignment", {})

    def check_alignment(self) -> AlignmentCheckResult:
        result = AlignmentCheckResult()
        params_to_check = [
            ("start_time", "start_time"),
            ("end_time", "end_time"),
            ("instruments", "instruments"),
            ("freq", "freq"),
            ("fit_start_time", "fit_start_time"),
            ("fit_end_time", "fit_end_time"),
        ]

        for param_name, cfg_key in params_to_check:
            a158_val = self.alpha158_cfg.get(f"default_{cfg_key}")
            a50_val = self.alpha50_cfg.get(f"default_{cfg_key}")
            unified_val = self.alignment_cfg.get(f"unified_{cfg_key}")

            diff = ConfigDiff(
                parameter=param_name,
                alpha158_value=a158_val,
                alpha50_value=a50_val,
                is_aligned=(a158_val == a50_val),
                severity="critical" if param_name in self.CRITICAL_PARAMS else "warning",
                suggestion="",
            )

            if not diff.is_aligned:
                result.is_fully_aligned = False
                if unified_val is not None:
                    diff.suggestion = (
                        f"Use unified value: {unified_val} "
                        f"(Alpha158={a158_val}, Alpha50={a50_val})"
                    )
                else:
                    diff.suggestion = (
                        f"Manually align: Alpha158={a158_val}, Alpha50={a50_val}"
                    )
                result.warnings.append(
                    f"[MISMATCH] {param_name}: Alpha158={a158_val}, Alpha50={a50_val}"
                )

            result.diffs.append(diff)

        self._check_processors(result)
        self._check_label_formula(result)
        self._check_feature_source(result)
        self._generate_unified_config(result)

        return result

    def _check_processors(self, result: AlignmentCheckResult):
        a158_infer = self.alpha158_cfg.get("infer_processors", [])
        a50_infer = self.alpha50_cfg.get("infer_processors", [])
        a158_learn = self.alpha158_cfg.get("learn_processors", [])
        a50_learn = self.alpha50_cfg.get("learn_processors", [])

        a158_infer_str = str(a158_infer)
        a50_infer_str = str(a50_infer)
        if a158_infer_str != a50_infer_str:
            result.diffs.append(ConfigDiff(
                parameter="infer_processors",
                alpha158_value=a158_infer_str,
                alpha50_value=a50_infer_str,
                is_aligned=False,
                severity="critical",
                suggestion="Alpha158 uses empty infer_processors; Alpha50 uses DropnaLabel+CSZScoreNorm on features. "
                           "Unified: use empty infer_processors for both (apply normalization in learn_processors only).",
            ))
            result.is_fully_aligned = False
            result.warnings.append(
                "[MISMATCH] infer_processors differ significantly between Alpha158 and Alpha50"
            )

        a158_learn_str = str(a158_learn)
        a50_learn_str = str(a50_learn)
        if a158_learn_str != a50_learn_str:
            result.diffs.append(ConfigDiff(
                parameter="learn_processors",
                alpha158_value=a158_learn_str,
                alpha50_value=a50_learn_str,
                is_aligned=False,
                severity="warning",
                suggestion="Both use DropnaLabel + CSZScoreNorm(label), but Alpha50 adds CSZScoreNorm(feature) "
                           "in infer_processors. Unified: use DropnaLabel + CSZScoreNorm(label) for learn.",
            ))
            result.is_fully_aligned = False
            result.warnings.append(
                "[MISMATCH] learn_processors differ between Alpha158 and Alpha50"
            )

    def _check_label_formula(self, result: AlignmentCheckResult):
        a158_label = self.alpha158_cfg.get("label_formula", "")
        a50_label = self.alpha50_cfg.get("label_formula", "")

        if a158_label != a50_label:
            result.diffs.append(ConfigDiff(
                parameter="label_formula",
                alpha158_value=a158_label,
                alpha50_value=a50_label,
                is_aligned=False,
                severity="critical",
                suggestion="Alpha158 uses Ref($close,-2)/Ref($close,-1)-1 (T+1 close to T+2 close return). "
                           "Alpha50 uses pct_change(1).shift(-1) (T close to T+1 close return). "
                           "These produce DIFFERENT labels! Unified: use Ref($close,-2)/Ref($close,-1)-1.",
            ))
            result.is_fully_aligned = False
            result.warnings.append(
                "[CRITICAL] Label formulas are DIFFERENT! This is a major source of sample count difference."
            )

    def _check_feature_source(self, result: AlignmentCheckResult):
        a158_src = self.alpha158_cfg.get("feature_source", "")
        a50_src = self.alpha50_cfg.get("feature_source", "")

        result.diffs.append(ConfigDiff(
            parameter="feature_source",
            alpha158_value=a158_src,
            alpha50_value=a50_src,
            is_aligned=False,
            severity="info",
            suggestion="Alpha158 computes features on-the-fly via Qlib expression engine. "
                       "Alpha50 loads pre-computed pickle data. This fundamental difference affects "
                       "NaN handling, date alignment, and stock pool coverage.",
        ))

    def _generate_unified_config(self, result: AlignmentCheckResult):
        unified = {
            "instruments": self.alignment_cfg.get("unified_instruments", "csi300"),
            "start_time": self.alignment_cfg.get("unified_start_time", "2020-01-01"),
            "end_time": self.alignment_cfg.get("unified_end_time", "2023-06-01"),
            "freq": self.alignment_cfg.get("unified_freq", "day"),
            "fit_start_time": self.alignment_cfg.get("unified_fit_start_time", "2020-01-01"),
            "fit_end_time": self.alignment_cfg.get("unified_fit_end_time", "2022-01-01"),
            "infer_processors": [],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            "process_type": "append",
            "label_formula": "Ref($close, -2)/Ref($close, -1) - 1",
            "min_consecutive_days": self.alignment_cfg.get("min_consecutive_days", 60),
            "drop_any_nan": self.alignment_cfg.get("drop_any_nan", True),
        }
        result.unified_config = unified

    def generate_corrected_yaml(self, result: AlignmentCheckResult, output_path: str):
        corrected = copy.deepcopy(self.config)
        for dataset_key in ["alpha158", "alpha50"]:
            for k, v in result.unified_config.items():
                if k.startswith("default_"):
                    cfg_key = k
                else:
                    cfg_key = f"default_{k}" if k in [
                        "instruments", "start_time", "end_time", "freq",
                        "fit_start_time", "fit_end_time"
                    ] else None
                if cfg_key and cfg_key in corrected[dataset_key]:
                    corrected[dataset_key][cfg_key] = v

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(corrected, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Corrected config saved to {output_path}")

    def print_check_result(self, result: AlignmentCheckResult):
        print("\n" + "=" * 80)
        print("CONFIG ALIGNMENT CHECK RESULTS")
        print("=" * 80)

        header = f"{'Parameter':<25} {'Alpha158':<25} {'Alpha50':<25} {'Aligned':<10} {'Severity':<10}"
        print(header)
        print("-" * 95)

        for diff in result.diffs:
            a158_str = str(diff.alpha158_value)[:23]
            a50_str = str(diff.alpha50_value)[:23]
            aligned_str = "YES" if diff.is_aligned else "NO"
            print(
                f"{diff.parameter:<25} {a158_str:<25} {a50_str:<25} "
                f"{aligned_str:<10} {diff.severity:<10}"
            )

        print("-" * 95)
        if result.is_fully_aligned:
            print("RESULT: All configurations are aligned!")
        else:
            print(f"RESULT: {len([d for d in result.diffs if not d.is_aligned])} misalignment(s) found!")
            print("\nWARNINGS:")
            for w in result.warnings:
                print(f"  {w}")

        print("\nSUGGESTIONS:")
        for diff in result.diffs:
            if not diff.is_aligned and diff.suggestion:
                print(f"  [{diff.parameter}] {diff.suggestion}")

        print("\nUNIFIED CONFIG:")
        for k, v in result.unified_config.items():
            print(f"  {k}: {v}")

        print("=" * 80)
