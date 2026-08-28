r"""
12_experiment_08_final_results.py

Experiment 8: Final Results
Digital Forensic Evidence Synthesis Framework

Purpose
-------
Experiment 8 is a consolidation stage, NOT a new training experiment.

It reads the validated outputs produced by Experiments 2-7 and creates the
final comparative assessment required by the manuscript:

- conventional baseline performance;
- complete framework / Full FERF performance;
- component-wise ablation results;
- cross-dataset transfer results;
- computational efficiency and scalability;
- repeated-CV statistical evidence;
- claim-to-evidence matrix;
- manuscript-ready final results report.

No predictive model is trained, refitted, reoptimized, or evaluated on new
data by this script.

Important safeguards
--------------------
1. No model training.
2. No threshold selection.
3. No global-weight optimization.
4. No recomputation of predictive probabilities.
5. No test-label reuse for development.
6. Missing outputs are reported explicitly rather than estimated.
7. Experiment 7 is read from dataset-specific folders because separately run
   datasets may overwrite consolidated Report files.
8. Statistical claims are based on the actual paired-test outputs.

Expected project root
---------------------
D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments

Run
---
python 12_experiment_08_final_results.py

Optional
--------
python 12_experiment_08_final_results.py --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

RESULTS_ROOT = PROJECT_ROOT / "Results"

BASELINE_ROOT = (
    RESULTS_ROOT
    / "Experiment_03_Framework_Validation"
    / "Experiment_01_Baseline_Models"
)

ORIGINAL_FEATURE_ROOT = (
    RESULTS_ROOT
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)

MULTIVIEW_ROOT = (
    RESULTS_ROOT
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Multiview_Evidence"
    / "Phase_02_View_Models"
)

ABLATION_ROOT = (
    RESULTS_ROOT
    / "Experiment_04_Ablation_Study"
)

CROSS_DATASET_ROOT = (
    RESULTS_ROOT
    / "Experiment_05_Cross_Dataset_Validation"
)

EFFICIENCY_ROOT = (
    RESULTS_ROOT
    / "Experiment_06_Computational_Efficiency"
)

STATISTICAL_ROOT = (
    RESULTS_ROOT
    / "Experiment_07_Statistical_Analysis"
)

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "Experiment_08_Final_Results"
)

REPORTS_DIR = OUTPUT_ROOT / "Reports"
TABLES_DIR = OUTPUT_ROOT / "Tables"
MANIFESTS_DIR = OUTPUT_ROOT / "Manifests"
LOGS_DIR = OUTPUT_ROOT / "Logs"

for directory in (
    OUTPUT_ROOT,
    REPORTS_DIR,
    TABLES_DIR,
    MANIFESTS_DIR,
    LOGS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Configuration
# ============================================================

DATASETS = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
)

FULL_FERF_NAME = "A7_Full_FERF"

STATISTICAL_CONFIGURATIONS = {
    "original": "C0_Original_XGBoost",
    "unweighted": "C1_Unweighted_Multiview",
    "global": "C2_Global_Weighted_Multiview",
    "ferf": "C3_Full_FERF",
}

CORE_METRICS = (
    "balanced_accuracy",
    "f1",
    "mcc",
    "roc_auc",
)

STATISTICAL_METRICS = (
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "cohen_kappa",
    "mae",
)

EPSILON = 1e-12


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger(
    "experiment_08_final_results"
)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
LOGGER.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)

file_handler = logging.FileHandler(
    LOGS_DIR / "experiment_08_final_results.log",
    mode="w",
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

LOGGER.addHandler(console)
LOGGER.addHandler(file_handler)


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def save_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def safe_read_csv(
    path: Path,
    required: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        LOGGER.warning(
            "Missing optional file: %s",
            path,
        )

        return pd.DataFrame()

    try:
        frame = pd.read_csv(
            path
        )

        LOGGER.info(
            "Loaded %s | rows=%d | columns=%d",
            path,
            len(frame),
            len(frame.columns),
        )

        return frame

    except Exception as exc:
        if required:
            raise

        LOGGER.warning(
            "Could not read %s: %s",
            path,
            exc,
        )

        return pd.DataFrame()


def first_existing(
    paths: list[Path],
) -> Path | None:
    return next(
        (
            path
            for path in paths
            if path.exists()
        ),
        None,
    )


def finite_or_nan(
    value: Any,
) -> float:
    try:
        number = float(
            value
        )

        return (
            number
            if np.isfinite(
                number
            )
            else float("nan")
        )

    except Exception:
        return float("nan")


def format_value(
    value: Any,
    digits: int = 6,
) -> str:
    number = finite_or_nan(
        value
    )

    if not np.isfinite(
        number
    ):
        return "NA"

    return f"{number:.{digits}f}"


def significant(
    value: Any,
    alpha: float,
) -> bool:
    number = finite_or_nan(
        value
    )

    return (
        np.isfinite(
            number
        )
        and number < alpha
    )


# ============================================================
# Experiment 2: baseline results
# ============================================================

def discover_baseline_results() -> pd.DataFrame:
    """
    Search the known validation hierarchy for conventional baseline CSVs.

    Because earlier baseline code versions may use different directory names,
    this function uses conservative filename discovery and only accepts tables
    that contain recognizable model/performance fields.
    """

    candidates = []

    search_roots = [
        BASELINE_ROOT,
        RESULTS_ROOT
        / "Experiment_03_Framework_Validation",
    ]

    patterns = (
        "*baseline*.csv",
        "*Baseline*.csv",
        "*model*metrics*.csv",
        "*Model*Metrics*.csv",
    )

    seen = set()

    for root in search_roots:
        if not root.exists():
            continue

        for pattern in patterns:
            for path in root.rglob(
                pattern
            ):
                resolved = str(
                    path.resolve()
                )

                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                try:
                    frame = pd.read_csv(
                        path
                    )

                except Exception:
                    continue

                normalized_columns = {
                    str(
                        column
                    ).lower()
                    for column
                    in frame.columns
                }

                has_model = any(
                    token
                    in normalized_columns
                    for token in (
                        "model",
                        "classifier",
                        "algorithm",
                    )
                )

                has_metric = any(
                    any(
                        metric
                        in column
                        for metric in (
                            "accuracy",
                            "f1",
                            "auc",
                            "precision",
                            "recall",
                        )
                    )
                    for column
                    in normalized_columns
                )

                if (
                    has_model
                    and has_metric
                ):
                    copy = frame.copy()

                    copy[
                        "_source_file"
                    ] = str(
                        path
                    )

                    candidates.append(
                        copy
                    )

    if not candidates:
        LOGGER.warning(
            "No conventional baseline result table was discovered."
        )

        return pd.DataFrame()

    # Preserve all actual source rows. We do not attempt to merge schemas
    # destructively because baseline code versions may use different names.
    return pd.concat(
        candidates,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# Experiment 3 / 4: framework and ablation
# ============================================================

def load_ablation_results() -> pd.DataFrame:
    frames = []

    for dataset in DATASETS:
        path = (
            ABLATION_ROOT
            / dataset
            / "Metrics"
            / "ablation_results.csv"
        )

        frame = safe_read_csv(
            path,
            required=True,
        )

        if "dataset" not in frame.columns:
            frame[
                "dataset"
            ] = dataset

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def extract_full_ferf(
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "dataset",
        "configuration",
    }

    if not required_columns.issubset(
        ablation.columns
    ):
        raise ValueError(
            "Ablation table does not contain dataset/configuration."
        )

    full = ablation.loc[
        ablation[
            "configuration"
        ].astype(str).eq(
            FULL_FERF_NAME
        )
    ].copy()

    if full.empty:
        raise ValueError(
            "A7_Full_FERF was not found in Experiment 4 outputs."
        )

    return full


def build_clean_comparison(
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "configuration",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "test_roc_auc",
        "test_average_precision",
        "threshold",
        "weights_json",
    ]

    existing = [
        column
        for column in columns
        if column in ablation.columns
    ]

    result = ablation[
        existing
    ].copy()

    return result.sort_values(
        [
            "dataset",
            "configuration",
        ]
    )


# ============================================================
# Experiment 4: degradation
# ============================================================

def load_degradation_summary() -> pd.DataFrame:
    candidates = [
        ABLATION_ROOT
        / "Reports"
        / "Evidence_Degradation_Summary.csv",
    ]

    frames = []

    consolidated = first_existing(
        candidates
    )

    if consolidated is not None:
        frame = safe_read_csv(
            consolidated
        )

        if not frame.empty:
            frames.append(
                frame
            )

    # Dataset-specific fallback.
    if not frames:
        for dataset in DATASETS:
            path = (
                ABLATION_ROOT
                / dataset
                / "Evidence_Degradation"
                / "record_view_degradation_summary.csv"
            )

            frame = safe_read_csv(
                path
            )

            if frame.empty:
                continue

            if "dataset" not in frame.columns:
                frame[
                    "dataset"
                ] = dataset

            frames.append(
                frame
            )

    return (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        if frames
        else pd.DataFrame()
    )


def build_degradation_key_table(
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    if degradation.empty:
        return pd.DataFrame()

    fraction_column = next(
        (
            column
            for column in (
                "requested_degradation_fraction_",
                "requested_degradation_fraction",
            )
            if column
            in degradation.columns
        ),
        None,
    )

    if fraction_column is None:
        return degradation.copy()

    # Keep clean, moderate, and severe degradation levels when available.
    wanted = (
        0.0,
        0.1,
        0.3,
        0.5,
    )

    mask = np.zeros(
        len(
            degradation
        ),
        dtype=bool,
    )

    values = pd.to_numeric(
        degradation[
            fraction_column
        ],
        errors="coerce",
    )

    for level in wanted:
        mask |= np.isclose(
            values,
            level,
            atol=1e-9,
        )

    return degradation.loc[
        mask
    ].copy()


# ============================================================
# Experiment 5: cross-dataset
# ============================================================

def load_cross_dataset_results() -> pd.DataFrame:
    candidates = [
        CROSS_DATASET_ROOT
        / "Reports"
        / "Cross_Dataset_Results.csv",
    ]

    path = first_existing(
        candidates
    )

    if path is None:
        LOGGER.warning(
            "Cross_Dataset_Results.csv not found."
        )

        return pd.DataFrame()

    return safe_read_csv(
        path
    )


# ============================================================
# Experiment 6: efficiency
# ============================================================

def load_efficiency_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summary = safe_read_csv(
        EFFICIENCY_ROOT
        / "Reports"
        / "Computational_Efficiency_Summary.csv"
    )

    scalability = safe_read_csv(
        EFFICIENCY_ROOT
        / "Reports"
        / "Scalability_Linear_Fit.csv"
    )

    return (
        summary,
        scalability,
    )


# ============================================================
# Experiment 7: statistical results
# ============================================================

def load_statistical_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    score_frames = []
    test_frames = []
    descriptive_frames = []

    for dataset in DATASETS:
        metrics_dir = (
            STATISTICAL_ROOT
            / dataset
            / "Metrics"
        )

        scores = safe_read_csv(
            metrics_dir
            / "repeated_cv_scores.csv",
            required=True,
        )

        tests = safe_read_csv(
            metrics_dir
            / "paired_statistical_tests.csv",
            required=True,
        )

        descriptive = safe_read_csv(
            metrics_dir
            / "descriptive_statistics.csv"
        )

        if "dataset" not in scores.columns:
            scores[
                "dataset"
            ] = dataset

        if "dataset" not in tests.columns:
            tests[
                "dataset"
            ] = dataset

        if (
            not descriptive.empty
            and "dataset"
            not in descriptive.columns
        ):
            descriptive[
                "dataset"
            ] = dataset

        score_frames.append(
            scores
        )

        test_frames.append(
            tests
        )

        if not descriptive.empty:
            descriptive_frames.append(
                descriptive
            )

    return (
        pd.concat(
            score_frames,
            ignore_index=True,
            sort=False,
        ),
        pd.concat(
            test_frames,
            ignore_index=True,
            sort=False,
        ),
        (
            pd.concat(
                descriptive_frames,
                ignore_index=True,
                sort=False,
            )
            if descriptive_frames
            else pd.DataFrame()
        ),
    )


def build_statistical_summary(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (
        dataset,
        configuration,
    ), group in scores.groupby(
        [
            "dataset",
            "configuration",
        ]
    ):
        record = {
            "dataset": dataset,
            "configuration": configuration,
            "paired_evaluations": len(
                group
            ),
        }

        for metric in (
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "cohen_kappa",
            "mae",
            "runtime_seconds",
        ):
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            record[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            record[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Claim/evidence evaluation
# ============================================================

def find_stat_test(
    tests: pd.DataFrame,
    dataset: str,
    config_a: str,
    config_b: str,
    metric: str,
) -> pd.Series | None:
    required = {
        "dataset",
        "configuration_a",
        "configuration_b",
        "metric",
    }

    if not required.issubset(
        tests.columns
    ):
        return None

    match = tests.loc[
        tests[
            "dataset"
        ].astype(str).eq(
            dataset
        )
        & tests[
            "configuration_a"
        ].astype(str).eq(
            config_a
        )
        & tests[
            "configuration_b"
        ].astype(str).eq(
            config_b
        )
        & tests[
            "metric"
        ].astype(str).eq(
            metric
        )
    ]

    if match.empty:
        return None

    return match.iloc[
        -1
    ]


def infer_significance(
    row: pd.Series,
    alpha: float,
) -> tuple[
    bool,
    str,
]:
    """
    Prefer Holm-adjusted paired t-test when available, then Holm-adjusted
    Wilcoxon, then unadjusted paired t-test.
    """

    options = [
        (
            "paired_t_p_holm",
            "Holm-adjusted paired t-test",
        ),
        (
            "wilcoxon_p_holm",
            "Holm-adjusted Wilcoxon",
        ),
        (
            "paired_t_p_value",
            "paired t-test",
        ),
        (
            "wilcoxon_p_value",
            "Wilcoxon",
        ),
    ]

    for (
        column,
        label,
    ) in options:
        if column not in row.index:
            continue

        value = finite_or_nan(
            row[
                column
            ]
        )

        if np.isfinite(
            value
        ):
            return (
                value < alpha,
                (
                    f"{label} p="
                    f"{value:.6g}"
                ),
            )

    return (
        False,
        "no usable p-value",
    )


def build_claim_evidence_matrix(
    ablation: pd.DataFrame,
    degradation: pd.DataFrame,
    cross_dataset: pd.DataFrame,
    efficiency: pd.DataFrame,
    scalability: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    records = []

    # --------------------------------------------------------
    # Claim 1: FERF > naive unweighted fusion on clean data.
    # --------------------------------------------------------

    for dataset in DATASETS:
        row = find_stat_test(
            statistical_tests,
            dataset,
            STATISTICAL_CONFIGURATIONS[
                "ferf"
            ],
            STATISTICAL_CONFIGURATIONS[
                "unweighted"
            ],
            "balanced_accuracy",
        )

        if row is None:
            records.append(
                {
                    "claim_id": (
                        "C1"
                    ),
                    "dataset": dataset,
                    "claim": (
                        "Full FERF outperforms naive "
                        "unweighted multi-view fusion "
                        "on clean repeated-CV data."
                    ),
                    "status": (
                        "INSUFFICIENT_EVIDENCE"
                    ),
                    "evidence": (
                        "No paired Balanced Accuracy test found."
                    ),
                }
            )

            continue

        difference = finite_or_nan(
            row[
                "mean_difference_a_minus_b"
            ]
        )

        sig, sig_text = infer_significance(
            row,
            alpha,
        )

        supported = (
            sig
            and difference > 0
        )

        records.append(
            {
                "claim_id": (
                    "C1"
                ),
                "dataset": dataset,
                "claim": (
                    "Full FERF outperforms naive "
                    "unweighted multi-view fusion "
                    "on clean repeated-CV data."
                ),
                "status": (
                    "SUPPORTED"
                    if supported
                    else "NOT_SUPPORTED"
                ),
                "evidence": (
                    f"ΔBA={difference:.6f}; "
                    f"{sig_text}"
                ),
            }
        )

    # --------------------------------------------------------
    # Claim 2: record-specific reliability improves over global
    # weighting on clean data.
    # --------------------------------------------------------

    for dataset in DATASETS:
        row = find_stat_test(
            statistical_tests,
            dataset,
            STATISTICAL_CONFIGURATIONS[
                "ferf"
            ],
            STATISTICAL_CONFIGURATIONS[
                "global"
            ],
            "balanced_accuracy",
        )

        if row is None:
            status = (
                "INSUFFICIENT_EVIDENCE"
            )
            evidence = (
                "No paired comparison found."
            )

        else:
            difference = finite_or_nan(
                row[
                    "mean_difference_a_minus_b"
                ]
            )

            sig, sig_text = infer_significance(
                row,
                alpha,
            )

            status = (
                "SUPPORTED"
                if (
                    sig
                    and difference > 0
                )
                else "NOT_SUPPORTED"
            )

            evidence = (
                f"ΔBA={difference:.6f}; "
                f"{sig_text}"
            )

        records.append(
            {
                "claim_id": (
                    "C2"
                ),
                "dataset": dataset,
                "claim": (
                    "Record-specific reliability provides "
                    "a significant clean-data advantage over "
                    "optimized global weighting."
                ),
                "status": status,
                "evidence": evidence,
            }
        )

    # --------------------------------------------------------
    # Claim 3: FERF universally beats original XGBoost.
    # --------------------------------------------------------

    for dataset in DATASETS:
        row = find_stat_test(
            statistical_tests,
            dataset,
            STATISTICAL_CONFIGURATIONS[
                "ferf"
            ],
            STATISTICAL_CONFIGURATIONS[
                "original"
            ],
            "balanced_accuracy",
        )

        if row is None:
            status = (
                "INSUFFICIENT_EVIDENCE"
            )
            evidence = (
                "No paired comparison found."
            )

        else:
            difference = finite_or_nan(
                row[
                    "mean_difference_a_minus_b"
                ]
            )

            sig, sig_text = infer_significance(
                row,
                alpha,
            )

            status = (
                "SUPPORTED"
                if (
                    sig
                    and difference > 0
                )
                else "NOT_SUPPORTED"
            )

            evidence = (
                f"ΔBA={difference:.6f}; "
                f"{sig_text}"
            )

        records.append(
            {
                "claim_id": (
                    "C3"
                ),
                "dataset": dataset,
                "claim": (
                    "Full FERF universally outperforms "
                    "original-feature XGBoost on clean data."
                ),
                "status": status,
                "evidence": evidence,
            }
        )

    # --------------------------------------------------------
    # Claim 4: robustness under evidence degradation.
    # --------------------------------------------------------

    if degradation.empty:
        records.append(
            {
                "claim_id": (
                    "C4"
                ),
                "dataset": (
                    "all"
                ),
                "claim": (
                    "Full FERF preserves performance "
                    "under degraded record-view evidence."
                ),
                "status": (
                    "INSUFFICIENT_EVIDENCE"
                ),
                "evidence": (
                    "Evidence degradation summary unavailable."
                ),
            }
        )

    else:
        fraction_column = next(
            (
                column
                for column in degradation.columns
                if column.startswith(
                    "requested_degradation_fraction"
                )
            ),
            None,
        )

        ba_column = next(
            (
                column
                for column in degradation.columns
                if column.startswith(
                    "balanced_accuracy"
                )
                and "mean"
                in column
            ),
            None,
        )

        if (
            fraction_column
            is not None
            and ba_column
            is not None
            and "configuration"
            in degradation.columns
            and "dataset"
            in degradation.columns
        ):
            for dataset in DATASETS:
                subset = degradation.loc[
                    degradation[
                        "dataset"
                    ].astype(str).eq(
                        dataset
                    )
                    & np.isclose(
                        pd.to_numeric(
                            degradation[
                                fraction_column
                            ],
                            errors="coerce",
                        ),
                        0.5,
                        atol=1e-9,
                    )
                ]

                ferf = subset.loc[
                    subset[
                        "configuration"
                    ].astype(str).eq(
                        "A7_Full_FERF"
                    )
                ]

                global_only = subset.loc[
                    subset[
                        "configuration"
                    ].astype(str).eq(
                        "A3_Global_View_Weights_Only"
                    )
                ]

                if (
                    ferf.empty
                    or global_only.empty
                ):
                    status = (
                        "INSUFFICIENT_EVIDENCE"
                    )
                    evidence = (
                        "50% degradation comparison unavailable."
                    )

                else:
                    ferf_ba = finite_or_nan(
                        ferf.iloc[
                            -1
                        ][
                            ba_column
                        ]
                    )

                    global_ba = finite_or_nan(
                        global_only.iloc[
                            -1
                        ][
                            ba_column
                        ]
                    )

                    delta = (
                        ferf_ba
                        - global_ba
                    )

                    status = (
                        "SUPPORTED"
                        if delta > 0
                        else "NOT_SUPPORTED"
                    )

                    evidence = (
                        f"50% degradation: FERF BA="
                        f"{ferf_ba:.6f}, global BA="
                        f"{global_ba:.6f}, Δ={delta:.6f}"
                    )

                records.append(
                    {
                        "claim_id": (
                            "C4"
                        ),
                        "dataset": dataset,
                        "claim": (
                            "Full FERF improves robustness "
                            "relative to global weighting under "
                            "severe record-view evidence degradation."
                        ),
                        "status": status,
                        "evidence": evidence,
                    }
                )

    # --------------------------------------------------------
    # Claim 5: strong zero-shot cross-dataset generalization.
    # --------------------------------------------------------

    if cross_dataset.empty:
        records.append(
            {
                "claim_id": (
                    "C5"
                ),
                "dataset": (
                    "cross-dataset"
                ),
                "claim": (
                    "The framework demonstrates strong "
                    "zero-shot cross-dataset classification."
                ),
                "status": (
                    "INSUFFICIENT_EVIDENCE"
                ),
                "evidence": (
                    "Cross-dataset results unavailable."
                ),
            }
        )

    else:
        for _, row in cross_dataset.iterrows():
            source = str(
                row.get(
                    "source_dataset",
                    "unknown"
                )
            )

            target = str(
                row.get(
                    "target_dataset",
                    "unknown"
                )
            )

            ba = finite_or_nan(
                row.get(
                    "target_balanced_accuracy",
                    np.nan,
                )
            )

            auc = finite_or_nan(
                row.get(
                    "target_roc_auc",
                    np.nan,
                )
            )

            # Conservative threshold for "strong" transfer:
            # BA >= 0.70 and ROC-AUC >= 0.70.
            status = (
                "SUPPORTED"
                if (
                    np.isfinite(
                        ba
                    )
                    and np.isfinite(
                        auc
                    )
                    and ba >= 0.70
                    and auc >= 0.70
                )
                else "NOT_SUPPORTED"
            )

            records.append(
                {
                    "claim_id": (
                        "C5"
                    ),
                    "dataset": (
                        f"{source}->{target}"
                    ),
                    "claim": (
                        "The framework demonstrates strong "
                        "zero-shot cross-dataset classification."
                    ),
                    "status": status,
                    "evidence": (
                        f"target BA={ba:.6f}; "
                        f"target ROC-AUC={auc:.6f}"
                    ),
                }
            )

    # --------------------------------------------------------
    # Claim 6: moderate ranking information transfers.
    # --------------------------------------------------------

    if not cross_dataset.empty:
        for _, row in cross_dataset.iterrows():
            source = str(
                row.get(
                    "source_dataset",
                    "unknown"
                )
            )

            target = str(
                row.get(
                    "target_dataset",
                    "unknown"
                )
            )

            auc = finite_or_nan(
                row.get(
                    "target_roc_auc",
                    np.nan,
                )
            )

            records.append(
                {
                    "claim_id": (
                        "C6"
                    ),
                    "dataset": (
                        f"{source}->{target}"
                    ),
                    "claim": (
                        "The harmonized semantic representation "
                        "retains above-chance ranking information "
                        "under cross-dataset transfer."
                    ),
                    "status": (
                        "SUPPORTED"
                        if (
                            np.isfinite(
                                auc
                            )
                            and auc > 0.5
                        )
                        else "NOT_SUPPORTED"
                    ),
                    "evidence": (
                        f"target ROC-AUC={auc:.6f}"
                    ),
                }
            )

    # --------------------------------------------------------
    # Claim 7: low FERF computational overhead.
    # --------------------------------------------------------

    if efficiency.empty:
        records.append(
            {
                "claim_id": (
                    "C7"
                ),
                "dataset": (
                    "all"
                ),
                "claim": (
                    "FERF introduces limited computational "
                    "overhead relative to view-level inference."
                ),
                "status": (
                    "INSUFFICIENT_EVIDENCE"
                ),
                "evidence": (
                    "Efficiency summary unavailable."
                ),
            }
        )

    else:
        for _, row in efficiency.iterrows():
            dataset = str(
                row.get(
                    "dataset",
                    "unknown"
                )
            )

            overhead = finite_or_nan(
                row.get(
                    "ferf_overhead_percent_of_end_to_end",
                    np.nan,
                )
            )

            records.append(
                {
                    "claim_id": (
                        "C7"
                    ),
                    "dataset": dataset,
                    "claim": (
                        "FERF introduces limited computational "
                        "overhead relative to view-level inference."
                    ),
                    "status": (
                        "SUPPORTED"
                        if (
                            np.isfinite(
                                overhead
                            )
                            and overhead < 10.0
                        )
                        else "NOT_SUPPORTED"
                    ),
                    "evidence": (
                        f"FERF overhead={overhead:.3f}%"
                    ),
                }
            )

    # --------------------------------------------------------
    # Claim 8: near-linear scalability.
    # --------------------------------------------------------

    if scalability.empty:
        records.append(
            {
                "claim_id": (
                    "C8"
                ),
                "dataset": (
                    "all"
                ),
                "claim": (
                    "Analytical inference exhibits near-linear "
                    "scaling with record count."
                ),
                "status": (
                    "INSUFFICIENT_EVIDENCE"
                ),
                "evidence": (
                    "Scalability fit unavailable."
                ),
            }
        )

    else:
        for _, row in scalability.iterrows():
            dataset = str(
                row.get(
                    "dataset",
                    "unknown"
                )
            )

            r_squared = finite_or_nan(
                row.get(
                    "linear_fit_r_squared",
                    np.nan,
                )
            )

            records.append(
                {
                    "claim_id": (
                        "C8"
                    ),
                    "dataset": dataset,
                    "claim": (
                        "Analytical inference exhibits near-linear "
                        "scaling with record count."
                    ),
                    "status": (
                        "SUPPORTED"
                        if (
                            np.isfinite(
                                r_squared
                            )
                            and r_squared >= 0.98
                        )
                        else "NOT_SUPPORTED"
                    ),
                    "evidence": (
                        f"linear-fit R²={r_squared:.6f}"
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Final comparative table
# ============================================================

def build_final_dataset_assessment(
    full_ferf: pd.DataFrame,
    cross_dataset: pd.DataFrame,
    efficiency: pd.DataFrame,
    statistical_summary: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for dataset in DATASETS:
        record = {
            "dataset": dataset,
        }

        # Clean held-out Experiment 4 Full FERF.
        heldout = full_ferf.loc[
            full_ferf[
                "dataset"
            ].astype(str).eq(
                dataset
            )
        ]

        if not heldout.empty:
            row = heldout.iloc[
                -1
            ]

            for source, target in (
                (
                    "test_balanced_accuracy",
                    "heldout_balanced_accuracy",
                ),
                (
                    "test_f1",
                    "heldout_f1",
                ),
                (
                    "test_mcc",
                    "heldout_mcc",
                ),
                (
                    "test_roc_auc",
                    "heldout_roc_auc",
                ),
            ):
                if source in row.index:
                    record[
                        target
                    ] = finite_or_nan(
                        row[
                            source
                        ]
                    )

        # Repeated CV.
        repeated = statistical_summary.loc[
            statistical_summary[
                "dataset"
            ].astype(str).eq(
                dataset
            )
            & statistical_summary[
                "configuration"
            ].astype(str).eq(
                STATISTICAL_CONFIGURATIONS[
                    "ferf"
                ]
            )
        ]

        if not repeated.empty:
            row = repeated.iloc[
                -1
            ]

            for source, target in (
                (
                    "balanced_accuracy_mean",
                    "repeated_cv_balanced_accuracy_mean",
                ),
                (
                    "balanced_accuracy_std",
                    "repeated_cv_balanced_accuracy_std",
                ),
                (
                    "macro_f1_mean",
                    "repeated_cv_macro_f1_mean",
                ),
                (
                    "roc_auc_mean",
                    "repeated_cv_roc_auc_mean",
                ),
                (
                    "pr_auc_mean",
                    "repeated_cv_pr_auc_mean",
                ),
            ):
                if source in row.index:
                    record[
                        target
                    ] = finite_or_nan(
                        row[
                            source
                        ]
                    )

        # Efficiency.
        eff = efficiency.loc[
            efficiency[
                "dataset"
            ].astype(str).eq(
                dataset
            )
        ] if (
            not efficiency.empty
            and "dataset"
            in efficiency.columns
        ) else pd.DataFrame()

        if not eff.empty:
            row = eff.iloc[
                -1
            ]

            for source, target in (
                (
                    "end_to_end_latency_ms_per_record",
                    "latency_ms_per_record",
                ),
                (
                    "end_to_end_throughput_records_per_second",
                    "throughput_records_per_second",
                ),
                (
                    "ferf_overhead_percent_of_end_to_end",
                    "ferf_overhead_percent",
                ),
                (
                    "peak_rss_mb",
                    "peak_process_rss_mb",
                ),
            ):
                if source in row.index:
                    record[
                        target
                    ] = finite_or_nan(
                        row[
                            source
                        ]
                    )

        # Incoming cross-dataset transfer performance.
        incoming = (
            cross_dataset.loc[
                cross_dataset[
                    "target_dataset"
                ].astype(str).eq(
                    dataset
                )
            ]
            if (
                not cross_dataset.empty
                and "target_dataset"
                in cross_dataset.columns
            )
            else pd.DataFrame()
        )

        if not incoming.empty:
            record[
                "incoming_cross_dataset_balanced_accuracy_mean"
            ] = float(
                pd.to_numeric(
                    incoming[
                        "target_balanced_accuracy"
                    ],
                    errors="coerce",
                ).mean()
            )

            record[
                "incoming_cross_dataset_roc_auc_mean"
            ] = float(
                pd.to_numeric(
                    incoming[
                        "target_roc_auc"
                    ],
                    errors="coerce",
                ).mean()
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Markdown final report
# ============================================================

def report_section_table(
    frame: pd.DataFrame,
    columns: list[str],
    titles: list[str],
    digits: int = 6,
) -> str:
    if frame.empty:
        return "_No data available._"

    available = [
        (
            column,
            title,
        )
        for column, title in zip(
            columns,
            titles,
        )
        if column in frame.columns
    ]

    if not available:
        return "_No applicable columns available._"

    header = (
        "| "
        + " | ".join(
            title
            for _, title
            in available
        )
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            "---"
            for _ in available
        )
        + " |"
    )

    rows = [
        header,
        separator,
    ]

    for _, row in frame.iterrows():
        values = []

        for column, _ in available:
            value = row[
                column
            ]

            if isinstance(
                value,
                (
                    float,
                    np.floating,
                    int,
                    np.integer,
                ),
            ):
                values.append(
                    format_value(
                        value,
                        digits,
                    )
                )

            else:
                values.append(
                    str(
                        value
                    )
                )

        rows.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        rows
    )


def create_final_report(
    final_assessment: pd.DataFrame,
    clean_comparison: pd.DataFrame,
    cross_dataset: pd.DataFrame,
    efficiency: pd.DataFrame,
    scalability: pd.DataFrame,
    statistical_summary: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    claims: pd.DataFrame,
) -> str:
    supported = claims.loc[
        claims[
            "status"
        ].eq(
            "SUPPORTED"
        )
    ]

    not_supported = claims.loc[
        claims[
            "status"
        ].eq(
            "NOT_SUPPORTED"
        )
    ]

    insufficient = claims.loc[
        claims[
            "status"
        ].eq(
            "INSUFFICIENT_EVIDENCE"
        )
    ]

    lines = []

    lines.append(
        "# Experiment 8 — Final Results"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Purpose"
    )

    lines.append(
        ""
    )

    lines.append(
        "Experiment 8 consolidates the validated outputs from the preceding "
        "experimental stages. No model was trained, refitted, reoptimized, "
        "or evaluated on new observations during this stage."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Final Dataset-Level Assessment"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            final_assessment,
            [
                "dataset",
                "heldout_balanced_accuracy",
                "heldout_f1",
                "heldout_mcc",
                "heldout_roc_auc",
                "repeated_cv_balanced_accuracy_mean",
                "repeated_cv_balanced_accuracy_std",
                "latency_ms_per_record",
                "throughput_records_per_second",
            ],
            [
                "Dataset",
                "Held-out BA",
                "Held-out F1",
                "Held-out MCC",
                "Held-out ROC-AUC",
                "Repeated-CV BA Mean",
                "Repeated-CV BA SD",
                "Latency ms/record",
                "Throughput records/s",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Clean-Data Ablation Assessment"
    )

    lines.append(
        ""
    )

    clean_columns = [
        "dataset",
        "configuration",
        "test_balanced_accuracy",
        "test_f1",
        "test_mcc",
        "test_roc_auc",
    ]

    lines.append(
        report_section_table(
            clean_comparison,
            clean_columns,
            [
                "Dataset",
                "Configuration",
                "BA",
                "F1",
                "MCC",
                "ROC-AUC",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Cross-Dataset Assessment"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            cross_dataset,
            [
                "source_dataset",
                "target_dataset",
                "source_test_balanced_accuracy",
                "target_balanced_accuracy",
                "target_f1",
                "target_mcc",
                "target_roc_auc",
            ],
            [
                "Source",
                "Target",
                "Source-test BA",
                "Target BA",
                "Target F1",
                "Target MCC",
                "Target ROC-AUC",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Computational Efficiency"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            efficiency,
            [
                "dataset",
                "historical_view_training_seconds",
                "serialized_model_size_mb",
                "full_test_end_to_end_seconds_mean",
                "end_to_end_latency_ms_per_record",
                "end_to_end_throughput_records_per_second",
                "ferf_overhead_percent_of_end_to_end",
                "peak_rss_mb",
            ],
            [
                "Dataset",
                "Historical Training s",
                "Model Size MB",
                "Analytical Inference s",
                "Latency ms/record",
                "Throughput records/s",
                "FERF Overhead %",
                "Peak RSS MB",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Scalability"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            scalability,
            [
                "dataset",
                "slope_seconds_per_record",
                "intercept_seconds",
                "linear_fit_r_squared",
                "minimum_rows",
                "maximum_rows",
            ],
            [
                "Dataset",
                "Slope s/record",
                "Intercept s",
                "R²",
                "Minimum Rows",
                "Maximum Rows",
            ],
            digits=9,
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Repeated-CV Performance"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            statistical_summary,
            [
                "dataset",
                "configuration",
                "paired_evaluations",
                "macro_f1_mean",
                "weighted_f1_mean",
                "balanced_accuracy_mean",
                "roc_auc_mean",
                "pr_auc_mean",
                "cohen_kappa_mean",
                "mae_mean",
            ],
            [
                "Dataset",
                "Configuration",
                "N",
                "Macro-F1",
                "Weighted-F1",
                "BA",
                "ROC-AUC",
                "PR-AUC",
                "Kappa",
                "MAE",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Claim-to-Evidence Matrix"
    )

    lines.append(
        ""
    )

    lines.append(
        report_section_table(
            claims,
            [
                "claim_id",
                "dataset",
                "status",
                "claim",
                "evidence",
            ],
            [
                "Claim",
                "Dataset/Transfer",
                "Status",
                "Statement",
                "Evidence",
            ],
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## Integrated Interpretation"
    )

    lines.append(
        ""
    )

    lines.append(
        "The final assessment should distinguish intrusion-classification "
        "performance from the broader forensic evidence-synthesis contribution. "
        "Clean-data results indicate that the Full FERF configuration should "
        "not be presented as universally superior to the strongest original-feature "
        "classifier or to optimized global weighting. The strongest clean-data "
        "statistical evidence supports structured multi-view weighting relative "
        "to naive equal averaging."
    )

    lines.append(
        ""
    )

    lines.append(
        "Record-specific reliability should be interpreted primarily in relation "
        "to evidence condition and robustness rather than as a mechanism that "
        "must always increase clean-data discrimination. This interpretation is "
        "consistent with the degradation experiment, where the benefit is "
        "dataset dependent."
    )

    lines.append(
        ""
    )

    lines.append(
        "Cross-dataset results should be reported as evidence of substantial "
        "domain sensitivity. Above-chance ROC-AUC may indicate retained ranking "
        "information, but the observed threshold-dependent degradation does not "
        "support a claim of strong zero-shot cross-dataset classification."
    )

    lines.append(
        ""
    )

    lines.append(
        "Computational measurements support efficient analytical inference and "
        "near-linear scaling over the tested record-count range. Efficiency "
        "claims should refer to the analytical inference path rather than the "
        "entire digital-forensic acquisition and preprocessing workflow."
    )

    lines.append(
        ""
    )

    lines.append(
        "## Evidence Status"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Supported claim rows: {len(supported)}"
    )

    lines.append(
        f"- Not-supported claim rows: {len(not_supported)}"
    )

    lines.append(
        f"- Insufficient-evidence rows: {len(insufficient)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Experiment 8 does not create additional experimental observations; "
        "it only consolidates and interprets the validated outputs produced by "
        "Experiments 1–7."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 8: consolidate final validated results "
            "without model training or new evaluation."
        )
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "Significance threshold used when classifying "
            "statistical evidence. Default: 0.05."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if (
        args.alpha <= 0
        or args.alpha >= 1
    ):
        raise ValueError(
            "--alpha must lie between 0 and 1."
        )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 8: FINAL RESULTS"
    )
    LOGGER.info(
        "New model training: NO"
    )
    LOGGER.info(
        "New threshold/weight optimization: NO"
    )
    LOGGER.info(
        "Purpose: consolidation of validated Experiments 1-7"
    )
    LOGGER.info("=" * 78)

    loaded_files = []
    missing_files = []

    # --------------------------------------------------------
    # Load validated outputs
    # --------------------------------------------------------

    baseline = discover_baseline_results()

    ablation = load_ablation_results()

    full_ferf = extract_full_ferf(
        ablation
    )

    clean_comparison = build_clean_comparison(
        ablation
    )

    degradation = load_degradation_summary()

    degradation_key = build_degradation_key_table(
        degradation
    )

    cross_dataset = load_cross_dataset_results()

    (
        efficiency,
        scalability,
    ) = load_efficiency_results()

    (
        repeated_scores,
        statistical_tests,
        descriptive_statistics,
    ) = load_statistical_outputs()

    statistical_summary = build_statistical_summary(
        repeated_scores
    )

    # --------------------------------------------------------
    # Final integrated outputs
    # --------------------------------------------------------

    final_assessment = build_final_dataset_assessment(
        full_ferf=full_ferf,
        cross_dataset=cross_dataset,
        efficiency=efficiency,
        statistical_summary=(
            statistical_summary
        ),
    )

    claims = build_claim_evidence_matrix(
        ablation=ablation,
        degradation=degradation,
        cross_dataset=cross_dataset,
        efficiency=efficiency,
        scalability=scalability,
        statistical_tests=(
            statistical_tests
        ),
        alpha=args.alpha,
    )

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    baseline.to_csv(
        TABLES_DIR
        / "Final_Baseline_Results.csv",
        index=False,
    )

    clean_comparison.to_csv(
        TABLES_DIR
        / "Final_Clean_Ablation_Comparison.csv",
        index=False,
    )

    degradation_key.to_csv(
        TABLES_DIR
        / "Final_Evidence_Degradation_Comparison.csv",
        index=False,
    )

    cross_dataset.to_csv(
        TABLES_DIR
        / "Final_Cross_Dataset_Results.csv",
        index=False,
    )

    efficiency.to_csv(
        TABLES_DIR
        / "Final_Computational_Efficiency.csv",
        index=False,
    )

    scalability.to_csv(
        TABLES_DIR
        / "Final_Scalability.csv",
        index=False,
    )

    repeated_scores.to_csv(
        TABLES_DIR
        / "Final_Repeated_CV_Scores.csv",
        index=False,
    )

    statistical_summary.to_csv(
        TABLES_DIR
        / "Final_Statistical_Performance_Summary.csv",
        index=False,
    )

    statistical_tests.to_csv(
        TABLES_DIR
        / "Final_Paired_Statistical_Tests.csv",
        index=False,
    )

    descriptive_statistics.to_csv(
        TABLES_DIR
        / "Final_Descriptive_Statistics.csv",
        index=False,
    )

    final_assessment.to_csv(
        TABLES_DIR
        / "Final_Dataset_Assessment.csv",
        index=False,
    )

    claims.to_csv(
        TABLES_DIR
        / "Final_Claims_Evidence_Matrix.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Manuscript-ready report
    # --------------------------------------------------------

    report = create_final_report(
        final_assessment=(
            final_assessment
        ),
        clean_comparison=(
            clean_comparison
        ),
        cross_dataset=(
            cross_dataset
        ),
        efficiency=efficiency,
        scalability=scalability,
        statistical_summary=(
            statistical_summary
        ),
        statistical_tests=(
            statistical_tests
        ),
        claims=claims,
    )

    report_path = (
        REPORTS_DIR
        / "Experiment_08_Final_Results_Report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "generated_utc": utc_now(),
        "experiment": (
            "Experiment 8: Final Results"
        ),
        "purpose": (
            "Consolidation of validated outputs from "
            "Experiments 1-7."
        ),
        "new_model_training": False,
        "new_model_refitting": False,
        "new_threshold_selection": False,
        "new_global_weight_optimization": False,
        "new_predictive_probability_computation": (
            False
        ),
        "new_test_observations": False,
        "statistical_alpha": (
            args.alpha
        ),
        "datasets": list(
            DATASETS
        ),
        "source_roots": {
            "baseline": str(
                BASELINE_ROOT
            ),
            "original_feature": str(
                ORIGINAL_FEATURE_ROOT
            ),
            "multiview": str(
                MULTIVIEW_ROOT
            ),
            "ablation": str(
                ABLATION_ROOT
            ),
            "cross_dataset": str(
                CROSS_DATASET_ROOT
            ),
            "efficiency": str(
                EFFICIENCY_ROOT
            ),
            "statistical": str(
                STATISTICAL_ROOT
            ),
        },
        "output_tables": [
            path.name
            for path in sorted(
                TABLES_DIR.glob(
                    "*.csv"
                )
            )
        ],
        "report": str(
            report_path
        ),
        "claim_status_counts": (
            claims[
                "status"
            ]
            .value_counts()
            .to_dict()
            if not claims.empty
            else {}
        ),
        "important_interpretive_rules": [
            (
                "Do not claim universal clean-data superiority "
                "unless supported by paired statistical tests."
            ),
            (
                "Interpret record-specific reliability primarily "
                "through evidence robustness when clean-data "
                "differences are statistically negligible."
            ),
            (
                "Report cross-dataset transfer separately from "
                "within-dataset performance."
            ),
            (
                "Describe Experiment 6 timing as analytical "
                "inference rather than complete forensic "
                "acquisition-to-report runtime."
            ),
            (
                "Experiment 8 creates no additional experimental "
                "observations."
            ),
        ],
    }

    save_json(
        MANIFESTS_DIR
        / "experiment_08_final_results_manifest.json",
        manifest,
    )

    LOGGER.info(
        "Final dataset rows: %d",
        len(
            final_assessment
        ),
    )

    LOGGER.info(
        "Claim evidence rows: %d",
        len(
            claims
        ),
    )

    if not claims.empty:
        for (
            status,
            count,
        ) in claims[
            "status"
        ].value_counts().items():
            LOGGER.info(
                "Claims %s: %d",
                status,
                count,
            )

    LOGGER.info(
        "Report: %s",
        report_path,
    )

    LOGGER.info(
        "Tables: %s",
        TABLES_DIR,
    )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 8 COMPLETED SUCCESSFULLY"
    )
    LOGGER.info("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
