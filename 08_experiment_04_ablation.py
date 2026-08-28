r"""
08_experiment_04_ablation.py

Experiment 4: Component-wise Ablation Study
Digital Forensic Evidence Synthesis Framework

Purpose
-------
Quantify the contribution of the principal computational components of the
multi-view FERF framework without retraining the already trained view models.

The experiment reuses:
1. Original-feature XGBoost validation/test predictions from Experiment 2.1.
2. View-level validation/test probabilities and record-view reliability
   components from Experiment 3 multi-view evidence modeling.

Ablation configurations
-----------------------
A0  Original-feature XGBoost
    - Representation ablation.
    - No semantic multi-view representation and no FERF.

A1  Best single semantic view
    - Tests whether multi-view synthesis is better than relying on the best
      individual evidence perspective.

A2  Unweighted multi-view mean
    - Equal contribution from all views.
    - No learned global view weights.
    - No record-specific reliability.

A3  Global view weights only
    - Validation-selected global view weights.
    - No record-specific reliability.

A4  FERF without integrity
    - Quality + temporal reliability only.
    - Remaining reliability coefficients are renormalized.

A5  FERF without quality
    - Integrity + temporal reliability only.
    - Remaining reliability coefficients are renormalized.

A6  FERF without temporal reliability
    - Integrity + quality reliability only.
    - Remaining reliability coefficients are renormalized.

A7  Full FERF
    - Integrity + quality + temporal reliability.
    - Validation-selected global view weights.
    - Validation-selected operating threshold.

Controlled evidence-availability degradation
---------------------------------------------
The script additionally evaluates robustness when an increasing proportion
of record-view evidence items becomes unavailable at the fusion stage.
For degraded record-view pairs:
    probability -> 0.5 (neutral evidence)
    reliability -> degradation reliability floor

This does NOT modify raw network-flow attributes and therefore should be
reported as "record-view evidence availability degradation", not as raw
feature corruption.

Experimental safeguards
-----------------------
- No predictive model is retrained by this script.
- All global view weights are selected using validation data only.
- All operating thresholds are selected using validation data only.
- Held-out test labels are used only after a configuration is fixed.
- The same saved records and splits produced by earlier experiments are reused.
- Every ablation uses the same view-level predictive outputs.
- Evidence degradation masks are reproducible and do not use labels.

Run
---
python 08_experiment_04_ablation.py --dataset all

Optional:
python 08_experiment_04_ablation.py --dataset CICIDS2017
python 08_experiment_04_ablation.py --dataset CSE-CIC-IDS2018
python 08_experiment_04_ablation.py --dataset all --weight-candidates 3000
python 08_experiment_04_ablation.py --dataset all --degradation-repeats 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

MULTIVIEW_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Multiview_Evidence"
    / "Phase_02_View_Models"
)

ORIGINAL_FEATURE_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_04_Ablation_Study"
)

REPORTS_DIR = RESULTS_ROOT / "Reports"
MANIFESTS_DIR = RESULTS_ROOT / "Manifests"
LOGS_DIR = RESULTS_ROOT / "Logs"

for directory in (
    RESULTS_ROOT,
    REPORTS_DIR,
    MANIFESTS_DIR,
    LOGS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experimental constants
# ============================================================

RANDOM_SEED = 42

INTEGRITY_WEIGHT = 0.35
QUALITY_WEIGHT = 0.45
TEMPORAL_WEIGHT = 0.20

THRESHOLDS = np.arange(
    0.20,
    0.801,
    0.01,
)

DEFAULT_WEIGHT_CANDIDATES = 1500

DEGRADATION_LEVELS = (
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)

DEFAULT_DEGRADATION_REPEATS = 5

NEUTRAL_PROBABILITY = 0.50
DEGRADED_RELIABILITY_FLOOR = 0.05

EPSILON = 1e-12


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger("experiment_04_ablation")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if LOGGER.handlers:
    LOGGER.handlers.clear()

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formatter)

_file_handler = logging.FileHandler(
    LOGS_DIR / "experiment_04_ablation.log",
    mode="w",
    encoding="utf-8",
)
_file_handler.setFormatter(_formatter)

LOGGER.addHandler(_console)
LOGGER.addHandler(_file_handler)


# ============================================================
# Result structures
# ============================================================

@dataclass
class AblationResult:
    dataset: str
    configuration: str
    description: str

    selected_view: str
    threshold: float

    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_mcc: float
    validation_roc_auc: float
    validation_average_precision: float

    test_accuracy: float
    test_balanced_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_mcc: float
    test_roc_auc: float
    test_average_precision: float

    delta_balanced_accuracy_vs_full: float
    delta_f1_vs_full: float
    delta_mcc_vs_full: float
    delta_roc_auc_vs_full: float

    weights_json: str
    reliability_components_json: str


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def calculate_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    target = np.asarray(
        target,
        dtype=np.int8,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability >= threshold
    ).astype(np.int8)

    if len(np.unique(target)) < 2:
        roc_auc = float("nan")
        average_precision = float("nan")
    else:
        roc_auc = float(
            roc_auc_score(
                target,
                probability,
            )
        )

        average_precision = float(
            average_precision_score(
                target,
                probability,
            )
        )

    return {
        "accuracy": float(
            accuracy_score(
                target,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                target,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                prediction,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                target,
                prediction,
            )
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def optimize_threshold(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[
    float,
    dict[str, float],
]:
    """
    Select the operating threshold on validation data only.

    Selection priority:
    1. Balanced accuracy
    2. F1
    3. MCC
    4. ROC-AUC
    5. Threshold closest to 0.5
    """

    best_threshold = 0.5
    best_metrics = None
    best_key = None

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            target,
            probability,
            float(threshold),
        )

        key = (
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["mcc"],
            (
                metrics["roc_auc"]
                if np.isfinite(
                    metrics["roc_auc"]
                )
                else -np.inf
            ),
            -abs(
                float(threshold) - 0.5
            ),
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_threshold = float(
                threshold
            )
            best_metrics = metrics

    assert best_metrics is not None

    return (
        best_threshold,
        best_metrics,
    )


# ============================================================
# Dataset discovery
# ============================================================

def discover_runnable_datasets() -> list[str]:
    """
    A dataset is runnable when the multiview prediction file exists.
    """

    datasets = []

    if not MULTIVIEW_ROOT.exists():
        return datasets

    for child in sorted(
        MULTIVIEW_ROOT.iterdir()
    ):
        if not child.is_dir():
            continue

        prediction_path = (
            child
            / "Predictions"
            / "view_predictions_and_reliability.csv"
        )

        if prediction_path.exists():
            datasets.append(
                child.name
            )

    return datasets


# ============================================================
# Input loading and validation
# ============================================================

def load_multiview_predictions(
    dataset: str,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    path = (
        MULTIVIEW_ROOT
        / dataset
        / "Predictions"
        / "view_predictions_and_reliability.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing multiview prediction file: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required_base = {
        "true_label",
        "split",
    }

    if not required_base.issubset(
        dataframe.columns
    ):
        raise ValueError(
            f"{dataset}: multiview predictions are missing "
            f"required columns: "
            f"{sorted(required_base - set(dataframe.columns))}"
        )

    views = [
        column.removesuffix(
            "__probability"
        )
        for column in dataframe.columns
        if column.endswith(
            "__probability"
        )
    ]

    if len(views) < 2:
        raise ValueError(
            f"{dataset}: at least two semantic views are required."
        )

    required_view_columns = set()

    for view in views:
        required_view_columns.update(
            {
                f"{view}__probability",
                f"{view}__integrity",
                f"{view}__quality",
                f"{view}__temporal",
            }
        )

    missing = (
        required_view_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{dataset}: missing reliability columns: "
            f"{sorted(missing)}"
        )

    observed_splits = set(
        dataframe["split"]
        .dropna()
        .astype(str)
    )

    required_splits = {
        "validation",
        "test",
    }

    if not required_splits.issubset(
        observed_splits
    ):
        raise ValueError(
            f"{dataset}: required validation/test splits "
            f"were not found. Observed={sorted(observed_splits)}"
        )

    return (
        dataframe,
        views,
    )


def load_original_feature_predictions(
    dataset: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None:
    """
    Return:
        validation_target
        validation_probability
        test_target
        test_probability

    Returns None if Experiment 2.1 predictions are not available.
    """

    root = (
        ORIGINAL_FEATURE_ROOT
        / dataset
        / "Predictions"
    )

    validation_path = (
        root
        / "validation_predictions.csv"
    )

    test_path = (
        root
        / "test_predictions.csv"
    )

    if (
        not validation_path.exists()
        or not test_path.exists()
    ):
        return None

    validation = pd.read_csv(
        validation_path
    )

    test = pd.read_csv(
        test_path
    )

    required = {
        "true_label",
        "attack_probability",
    }

    for name, frame in (
        ("validation", validation),
        ("test", test),
    ):
        if not required.issubset(
            frame.columns
        ):
            raise ValueError(
                f"{dataset}: original-feature {name} "
                f"predictions are missing columns "
                f"{sorted(required - set(frame.columns))}"
            )

    return (
        validation["true_label"].to_numpy(
            dtype=np.int8
        ),
        validation[
            "attack_probability"
        ].to_numpy(
            dtype=float
        ),
        test["true_label"].to_numpy(
            dtype=np.int8
        ),
        test[
            "attack_probability"
        ].to_numpy(
            dtype=float
        ),
    )


# ============================================================
# Matrix extraction
# ============================================================

def extract_split_matrices(
    dataframe: pd.DataFrame,
    views: list[str],
    split_name: str,
) -> dict[str, np.ndarray]:
    mask = dataframe[
        "split"
    ].eq(
        split_name
    ).to_numpy()

    target = dataframe.loc[
        mask,
        "true_label",
    ].to_numpy(
        dtype=np.int8
    )

    probability = np.column_stack(
        [
            dataframe.loc[
                mask,
                f"{view}__probability",
            ].to_numpy(
                dtype=float
            )
            for view in views
        ]
    )

    integrity = np.column_stack(
        [
            dataframe.loc[
                mask,
                f"{view}__integrity",
            ].to_numpy(
                dtype=float
            )
            for view in views
        ]
    )

    quality = np.column_stack(
        [
            dataframe.loc[
                mask,
                f"{view}__quality",
            ].to_numpy(
                dtype=float
            )
            for view in views
        ]
    )

    temporal = np.column_stack(
        [
            dataframe.loc[
                mask,
                f"{view}__temporal",
            ].to_numpy(
                dtype=float
            )
            for view in views
        ]
    )

    probability = np.nan_to_num(
        probability,
        nan=NEUTRAL_PROBABILITY,
        posinf=1.0,
        neginf=0.0,
    )

    integrity = np.nan_to_num(
        integrity,
        nan=0.5,
    )

    quality = np.nan_to_num(
        quality,
        nan=0.5,
    )

    temporal = np.nan_to_num(
        temporal,
        nan=0.5,
    )

    return {
        "target": target,
        "probability": np.clip(
            probability,
            0.0,
            1.0,
        ),
        "integrity": np.clip(
            integrity,
            0.0,
            1.0,
        ),
        "quality": np.clip(
            quality,
            0.0,
            1.0,
        ),
        "temporal": np.clip(
            temporal,
            0.0,
            1.0,
        ),
    }


# ============================================================
# Reliability calculation
# ============================================================

def normalized_component_weights(
    use_integrity: bool,
    use_quality: bool,
    use_temporal: bool,
) -> tuple[
    float,
    float,
    float,
]:
    weights = np.array(
        [
            (
                INTEGRITY_WEIGHT
                if use_integrity
                else 0.0
            ),
            (
                QUALITY_WEIGHT
                if use_quality
                else 0.0
            ),
            (
                TEMPORAL_WEIGHT
                if use_temporal
                else 0.0
            ),
        ],
        dtype=float,
    )

    total = weights.sum()

    if total <= 0:
        return (
            0.0,
            0.0,
            0.0,
        )

    weights /= total

    return tuple(
        float(value)
        for value in weights
    )


def build_reliability_matrix(
    matrices: dict[str, np.ndarray],
    use_integrity: bool,
    use_quality: bool,
    use_temporal: bool,
) -> tuple[
    np.ndarray,
    dict[str, float],
]:
    """
    Construct record-view reliability.

    When one reliability component is ablated, the remaining
    coefficients are renormalized to sum to one.
    """

    (
        integrity_weight,
        quality_weight,
        temporal_weight,
    ) = normalized_component_weights(
        use_integrity,
        use_quality,
        use_temporal,
    )

    if (
        integrity_weight == 0.0
        and quality_weight == 0.0
        and temporal_weight == 0.0
    ):
        reliability = np.ones_like(
            matrices["probability"],
            dtype=float,
        )
    else:
        reliability = (
            integrity_weight
            * matrices["integrity"]
            + quality_weight
            * matrices["quality"]
            + temporal_weight
            * matrices["temporal"]
        )

    reliability = np.clip(
        reliability,
        DEGRADED_RELIABILITY_FLOOR,
        1.0,
    )

    components = {
        "integrity": integrity_weight,
        "quality": quality_weight,
        "temporal": temporal_weight,
    }

    return (
        reliability,
        components,
    )


# ============================================================
# Fusion
# ============================================================

def fuse_probabilities(
    probability: np.ndarray,
    reliability: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_reliability = (
        reliability
        * weights.reshape(
            1,
            -1,
        )
    )

    denominator = np.maximum(
        weighted_reliability.sum(
            axis=1
        ),
        EPSILON,
    )

    fused = (
        weighted_reliability
        * probability
    ).sum(
        axis=1
    ) / denominator

    return np.clip(
        fused,
        0.0,
        1.0,
    )


def generate_weight_candidates(
    number_of_views: int,
    random_candidates: int,
    seed: int,
) -> np.ndarray:
    """
    Reproduce the candidate design used by optimized FERF:
    - equal weights
    - one-hot single-view controls
    - reproducible Dirichlet candidates
    """

    rng = np.random.default_rng(
        seed
    )

    random_weights = rng.dirichlet(
        np.ones(
            number_of_views
        ),
        size=max(
            random_candidates,
            1,
        ),
    )

    equal = np.full(
        (
            1,
            number_of_views,
        ),
        1.0
        / number_of_views,
    )

    one_hot = np.eye(
        number_of_views
    )

    return np.vstack(
        [
            equal,
            one_hot,
            random_weights,
        ]
    )


def optimize_fusion(
    validation_target: np.ndarray,
    validation_probability: np.ndarray,
    validation_reliability: np.ndarray,
    weight_candidates: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    dict[str, float],
]:
    """
    Select global view weights and operating threshold using
    validation data only.
    """

    best_weights = None
    best_threshold = None
    best_metrics = None
    best_key = None

    for weights in weight_candidates:
        fused = fuse_probabilities(
            validation_probability,
            validation_reliability,
            weights,
        )

        threshold, metrics = (
            optimize_threshold(
                validation_target,
                fused,
            )
        )

        key = (
            metrics[
                "balanced_accuracy"
            ],
            metrics["f1"],
            metrics["mcc"],
            (
                metrics["roc_auc"]
                if np.isfinite(
                    metrics["roc_auc"]
                )
                else -np.inf
            ),
            -abs(
                threshold - 0.5
            ),
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_weights = (
                weights.copy()
            )
            best_threshold = (
                threshold
            )
            best_metrics = (
                metrics
            )

    assert best_weights is not None
    assert best_threshold is not None
    assert best_metrics is not None

    return (
        best_weights,
        best_threshold,
        best_metrics,
    )


# ============================================================
# Individual ablation configurations
# ============================================================

def evaluate_original_feature_ablation(
    dataset: str,
) -> dict[str, Any] | None:
    data = load_original_feature_predictions(
        dataset
    )

    if data is None:
        return None

    (
        validation_target,
        validation_probability,
        test_target,
        test_probability,
    ) = data

    threshold, validation_metrics = (
        optimize_threshold(
            validation_target,
            validation_probability,
        )
    )

    test_metrics = calculate_metrics(
        test_target,
        test_probability,
        threshold,
    )

    return {
        "configuration": (
            "A0_Original_Feature_XGBoost"
        ),
        "description": (
            "Representation ablation: original "
            "network-flow feature space with XGBoost; "
            "no semantic multi-view representation and no FERF."
        ),
        "selected_view": "",
        "threshold": threshold,
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": test_metrics,
        "weights": {},
        "reliability_components": {},
        "test_probability": (
            test_probability
        ),
        "test_target": test_target,
    }


def evaluate_best_single_view(
    views: list[str],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    best = None
    best_key = None

    for index, view in enumerate(
        views
    ):
        validation_probability = (
            validation[
                "probability"
            ][:, index]
        )

        threshold, validation_metrics = (
            optimize_threshold(
                validation["target"],
                validation_probability,
            )
        )

        key = (
            validation_metrics[
                "balanced_accuracy"
            ],
            validation_metrics["f1"],
            validation_metrics["mcc"],
            validation_metrics["roc_auc"],
        )

        if (
            best_key is None
            or key > best_key
        ):
            test_probability = (
                test[
                    "probability"
                ][:, index]
            )

            best = {
                "configuration": (
                    "A1_Best_Single_View"
                ),
                "description": (
                    "Best individual semantic evidence view "
                    "selected using validation data only."
                ),
                "selected_view": view,
                "threshold": threshold,
                "validation_metrics": (
                    validation_metrics
                ),
                "test_metrics": (
                    calculate_metrics(
                        test["target"],
                        test_probability,
                        threshold,
                    )
                ),
                "weights": {
                    candidate_view: (
                        1.0
                        if candidate_view
                        == view
                        else 0.0
                    )
                    for candidate_view
                    in views
                },
                "reliability_components": {},
                "test_probability": (
                    test_probability
                ),
                "test_target": (
                    test["target"]
                ),
            }

            best_key = key

    assert best is not None

    return best


def evaluate_unweighted_mean(
    views: list[str],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    weights = np.full(
        len(views),
        1.0 / len(views),
    )

    validation_probability = (
        np.mean(
            validation[
                "probability"
            ],
            axis=1,
        )
    )

    threshold, validation_metrics = (
        optimize_threshold(
            validation["target"],
            validation_probability,
        )
    )

    test_probability = np.mean(
        test["probability"],
        axis=1,
    )

    return {
        "configuration": (
            "A2_Unweighted_Multiview_Mean"
        ),
        "description": (
            "Equal-probability averaging across all semantic "
            "views; no validation-selected global view weights "
            "and no record-specific reliability."
        ),
        "selected_view": "all",
        "threshold": threshold,
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": (
            calculate_metrics(
                test["target"],
                test_probability,
                threshold,
            )
        ),
        "weights": {
            view: float(
                weights[index]
            )
            for index, view
            in enumerate(views)
        },
        "reliability_components": {},
        "test_probability": (
            test_probability
        ),
        "test_target": (
            test["target"]
        ),
    }


def evaluate_weighted_configuration(
    configuration: str,
    description: str,
    views: list[str],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    weight_candidates: np.ndarray,
    use_integrity: bool,
    use_quality: bool,
    use_temporal: bool,
) -> dict[str, Any]:

    validation_reliability, components = (
        build_reliability_matrix(
            validation,
            use_integrity=(
                use_integrity
            ),
            use_quality=(
                use_quality
            ),
            use_temporal=(
                use_temporal
            ),
        )
    )

    test_reliability, _ = (
        build_reliability_matrix(
            test,
            use_integrity=(
                use_integrity
            ),
            use_quality=(
                use_quality
            ),
            use_temporal=(
                use_temporal
            ),
        )
    )

    (
        best_weights,
        threshold,
        validation_metrics,
    ) = optimize_fusion(
        validation_target=(
            validation["target"]
        ),
        validation_probability=(
            validation[
                "probability"
            ]
        ),
        validation_reliability=(
            validation_reliability
        ),
        weight_candidates=(
            weight_candidates
        ),
    )

    test_probability = (
        fuse_probabilities(
            test["probability"],
            test_reliability,
            best_weights,
        )
    )

    test_metrics = calculate_metrics(
        test["target"],
        test_probability,
        threshold,
    )

    return {
        "configuration": configuration,
        "description": description,
        "selected_view": "all",
        "threshold": threshold,
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": test_metrics,
        "weights": {
            view: float(
                best_weights[index]
            )
            for index, view
            in enumerate(views)
        },
        "reliability_components": (
            components
        ),
        "test_probability": (
            test_probability
        ),
        "test_target": (
            test["target"]
        ),
        "test_reliability": (
            test_reliability
        ),
    }


# ============================================================
# Controlled record-view evidence availability degradation
# ============================================================

def degrade_record_view_evidence(
    probability: np.ndarray,
    reliability: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Randomly mark a fraction of record-view evidence items unavailable.

    For an unavailable record-view pair:
    - probability becomes neutral (0.5)
    - reliability becomes the minimum reliability floor

    The mask is label-free and reproducible.
    """

    if (
        fraction < 0.0
        or fraction > 1.0
    ):
        raise ValueError(
            "Degradation fraction must be in [0, 1]."
        )

    degraded_probability = (
        probability.copy()
    )

    degraded_reliability = (
        reliability.copy()
    )

    rng = np.random.default_rng(
        seed
    )

    mask = rng.random(
        probability.shape
    ) < fraction

    degraded_probability[
        mask
    ] = NEUTRAL_PROBABILITY

    degraded_reliability[
        mask
    ] = DEGRADED_RELIABILITY_FLOOR

    return (
        degraded_probability,
        degraded_reliability,
        mask,
    )


def evaluate_degradation(
    dataset: str,
    views: list[str],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    full_configuration: dict[str, Any],
    global_configuration: dict[str, Any],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate robustness of three fusion strategies under increasing
    record-view evidence unavailability.

    Configurations:
    1. Unweighted mean
    2. Global weights only
    3. Full FERF

    Important:
    The clean-data global weights and thresholds selected on validation
    data remain fixed during degradation. No degraded test data is used
    to retune the system.
    """

    records = []

    full_weights = np.array(
        [
            full_configuration[
                "weights"
            ][view]
            for view in views
        ],
        dtype=float,
    )

    global_weights = np.array(
        [
            global_configuration[
                "weights"
            ][view]
            for view in views
        ],
        dtype=float,
    )

    full_reliability, _ = (
        build_reliability_matrix(
            test,
            use_integrity=True,
            use_quality=True,
            use_temporal=True,
        )
    )

    unit_reliability = np.ones_like(
        test["probability"],
        dtype=float,
    )

    unweighted_weights = np.full(
        len(views),
        1.0 / len(views),
    )

    # The clean-data unweighted threshold is selected once
    # from the validation set and then fixed for degradation.
    clean_validation_unweighted = (
        np.mean(
            validation[
                "probability"
            ],
            axis=1,
        )
    )

    unweighted_threshold, _ = (
        optimize_threshold(
            validation["target"],
            clean_validation_unweighted,
        )
    )

    for fraction in DEGRADATION_LEVELS:
        for repeat in range(
            repeats
        ):
            degradation_seed = (
                seed
                + int(
                    round(
                        fraction * 1000
                    )
                )
                + repeat * 10000
            )

            (
                degraded_probability,
                degraded_full_reliability,
                mask,
            ) = degrade_record_view_evidence(
                probability=(
                    test[
                        "probability"
                    ]
                ),
                reliability=(
                    full_reliability
                ),
                fraction=fraction,
                seed=degradation_seed,
            )

            # Apply exactly the same unavailable-view mask
            # to the non-reliability methods.
            degraded_unit_reliability = (
                unit_reliability.copy()
            )

            degraded_unit_reliability[
                mask
            ] = (
                DEGRADED_RELIABILITY_FLOOR
            )

            # A2: Unweighted mean.
            # Missing views are replaced with neutral probability.
            unweighted_probability = (
                fuse_probabilities(
                    degraded_probability,
                    np.ones_like(
                        degraded_unit_reliability
                    ),
                    unweighted_weights,
                )
            )

            # A3: global weights only.
            # Reliability is deliberately not used.
            global_probability = (
                fuse_probabilities(
                    degraded_probability,
                    np.ones_like(
                        degraded_unit_reliability
                    ),
                    global_weights,
                )
            )

            # A7: full FERF.
            full_probability = (
                fuse_probabilities(
                    degraded_probability,
                    degraded_full_reliability,
                    full_weights,
                )
            )

            methods = (
                (
                    "A2_Unweighted_Multiview_Mean",
                    unweighted_probability,
                    unweighted_threshold,
                ),
                (
                    "A3_Global_View_Weights_Only",
                    global_probability,
                    global_configuration[
                        "threshold"
                    ],
                ),
                (
                    "A7_Full_FERF",
                    full_probability,
                    full_configuration[
                        "threshold"
                    ],
                ),
            )

            actual_fraction = float(
                mask.mean()
            )

            for (
                configuration,
                probability,
                threshold,
            ) in methods:
                metrics = calculate_metrics(
                    test["target"],
                    probability,
                    threshold,
                )

                records.append(
                    {
                        "dataset": dataset,
                        "configuration": (
                            configuration
                        ),
                        "requested_degradation_fraction": (
                            fraction
                        ),
                        "actual_degradation_fraction": (
                            actual_fraction
                        ),
                        "repeat": repeat + 1,
                        "seed": (
                            degradation_seed
                        ),
                        "threshold_fixed_from_clean_validation": (
                            threshold
                        ),
                        **metrics,
                    }
                )

    return pd.DataFrame(
        records
    )


# ============================================================
# Result formatting
# ============================================================

def finalize_ablation_records(
    dataset: str,
    configurations: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    full = next(
        item
        for item in configurations
        if item["configuration"]
        == "A7_Full_FERF"
    )

    full_metrics = full[
        "test_metrics"
    ]

    rows = []

    for item in configurations:
        validation_metrics = item[
            "validation_metrics"
        ]

        test_metrics = item[
            "test_metrics"
        ]

        rows.append(
            AblationResult(
                dataset=dataset,
                configuration=item[
                    "configuration"
                ],
                description=item[
                    "description"
                ],
                selected_view=item[
                    "selected_view"
                ],
                threshold=float(
                    item["threshold"]
                ),

                validation_accuracy=(
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                validation_balanced_accuracy=(
                    validation_metrics[
                        "balanced_accuracy"
                    ]
                ),
                validation_precision=(
                    validation_metrics[
                        "precision"
                    ]
                ),
                validation_recall=(
                    validation_metrics[
                        "recall"
                    ]
                ),
                validation_f1=(
                    validation_metrics[
                        "f1"
                    ]
                ),
                validation_mcc=(
                    validation_metrics[
                        "mcc"
                    ]
                ),
                validation_roc_auc=(
                    validation_metrics[
                        "roc_auc"
                    ]
                ),
                validation_average_precision=(
                    validation_metrics[
                        "average_precision"
                    ]
                ),

                test_accuracy=(
                    test_metrics[
                        "accuracy"
                    ]
                ),
                test_balanced_accuracy=(
                    test_metrics[
                        "balanced_accuracy"
                    ]
                ),
                test_precision=(
                    test_metrics[
                        "precision"
                    ]
                ),
                test_recall=(
                    test_metrics[
                        "recall"
                    ]
                ),
                test_f1=(
                    test_metrics[
                        "f1"
                    ]
                ),
                test_mcc=(
                    test_metrics[
                        "mcc"
                    ]
                ),
                test_roc_auc=(
                    test_metrics[
                        "roc_auc"
                    ]
                ),
                test_average_precision=(
                    test_metrics[
                        "average_precision"
                    ]
                ),

                delta_balanced_accuracy_vs_full=(
                    test_metrics[
                        "balanced_accuracy"
                    ]
                    - full_metrics[
                        "balanced_accuracy"
                    ]
                ),
                delta_f1_vs_full=(
                    test_metrics["f1"]
                    - full_metrics["f1"]
                ),
                delta_mcc_vs_full=(
                    test_metrics["mcc"]
                    - full_metrics["mcc"]
                ),
                delta_roc_auc_vs_full=(
                    test_metrics[
                        "roc_auc"
                    ]
                    - full_metrics[
                        "roc_auc"
                    ]
                ),

                weights_json=json.dumps(
                    item["weights"],
                    sort_keys=True,
                ),
                reliability_components_json=(
                    json.dumps(
                        item[
                            "reliability_components"
                        ],
                        sort_keys=True,
                    )
                ),
            )
        )

    return pd.DataFrame(
        [
            asdict(row)
            for row in rows
        ]
    )


def build_degradation_summary(
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "average_precision",
    ]

    aggregation = {
        metric: [
            "mean",
            "std",
        ]
        for metric in metric_columns
    }

    summary = (
        degradation
        .groupby(
            [
                "dataset",
                "configuration",
                "requested_degradation_fraction",
            ],
            as_index=False,
        )
        .agg(
            aggregation
        )
    )

    summary.columns = [
        (
            "_".join(
                str(part)
                for part in column
                if part
            )
            if isinstance(
                column,
                tuple,
            )
            else column
        )
        for column in summary.columns
    ]

    return summary


# ============================================================
# Dataset execution
# ============================================================

def run_dataset(
    dataset: str,
    random_weight_candidates: int,
    degradation_repeats: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 4: ABLATION STUDY | DATASET=%s",
        dataset,
    )
    LOGGER.info("=" * 78)

    dataframe, views = (
        load_multiview_predictions(
            dataset
        )
    )

    LOGGER.info(
        "Detected %d views: %s",
        len(views),
        ", ".join(views),
    )

    validation = extract_split_matrices(
        dataframe,
        views,
        "validation",
    )

    test = extract_split_matrices(
        dataframe,
        views,
        "test",
    )

    LOGGER.info(
        "Validation rows=%d | Test rows=%d",
        len(
            validation["target"]
        ),
        len(
            test["target"]
        ),
    )

    weight_candidates = (
        generate_weight_candidates(
            number_of_views=(
                len(views)
            ),
            random_candidates=(
                random_weight_candidates
            ),
            seed=seed,
        )
    )

    LOGGER.info(
        "Global-weight search candidates=%d",
        len(
            weight_candidates
        ),
    )

    configurations = []

    # --------------------------------------------------------
    # A0 - Representation ablation
    # --------------------------------------------------------

    original = (
        evaluate_original_feature_ablation(
            dataset
        )
    )

    if original is not None:
        configurations.append(
            original
        )

        LOGGER.info(
            "A0 completed: Original-feature XGBoost"
        )
    else:
        LOGGER.warning(
            "A0 skipped: original-feature predictions not found."
        )

    # --------------------------------------------------------
    # A1 - Best single view
    # --------------------------------------------------------

    best_single = (
        evaluate_best_single_view(
            views,
            validation,
            test,
        )
    )

    configurations.append(
        best_single
    )

    LOGGER.info(
        "A1 completed: best single view=%s",
        best_single[
            "selected_view"
        ],
    )

    # --------------------------------------------------------
    # A2 - Unweighted multi-view mean
    # --------------------------------------------------------

    unweighted = (
        evaluate_unweighted_mean(
            views,
            validation,
            test,
        )
    )

    configurations.append(
        unweighted
    )

    LOGGER.info(
        "A2 completed: unweighted multi-view mean"
    )

    # --------------------------------------------------------
    # A3 - Global view weights only
    # --------------------------------------------------------

    global_only = (
        evaluate_weighted_configuration(
            configuration=(
                "A3_Global_View_Weights_Only"
            ),
            description=(
                "Validation-selected global semantic-view weights "
                "with no record-specific integrity, quality, or "
                "temporal reliability."
            ),
            views=views,
            validation=validation,
            test=test,
            weight_candidates=(
                weight_candidates
            ),
            use_integrity=False,
            use_quality=False,
            use_temporal=False,
        )
    )

    configurations.append(
        global_only
    )

    LOGGER.info(
        "A3 completed: global view weights only"
    )

    # --------------------------------------------------------
    # A4 - Full FERF minus integrity
    # --------------------------------------------------------

    no_integrity = (
        evaluate_weighted_configuration(
            configuration=(
                "A4_FERF_Without_Integrity"
            ),
            description=(
                "FERF with the acquisition-integrity component "
                "removed; quality and temporal coefficients are "
                "renormalized."
            ),
            views=views,
            validation=validation,
            test=test,
            weight_candidates=(
                weight_candidates
            ),
            use_integrity=False,
            use_quality=True,
            use_temporal=True,
        )
    )

    configurations.append(
        no_integrity
    )

    LOGGER.info(
        "A4 completed: FERF without integrity"
    )

    # --------------------------------------------------------
    # A5 - Full FERF minus quality
    # --------------------------------------------------------

    no_quality = (
        evaluate_weighted_configuration(
            configuration=(
                "A5_FERF_Without_Quality"
            ),
            description=(
                "FERF with information-quality reliability removed; "
                "integrity and temporal coefficients are renormalized."
            ),
            views=views,
            validation=validation,
            test=test,
            weight_candidates=(
                weight_candidates
            ),
            use_integrity=True,
            use_quality=False,
            use_temporal=True,
        )
    )

    configurations.append(
        no_quality
    )

    LOGGER.info(
        "A5 completed: FERF without quality"
    )

    # --------------------------------------------------------
    # A6 - Full FERF minus temporal reliability
    # --------------------------------------------------------

    no_temporal = (
        evaluate_weighted_configuration(
            configuration=(
                "A6_FERF_Without_Temporal"
            ),
            description=(
                "FERF with temporal reliability removed; integrity "
                "and quality coefficients are renormalized."
            ),
            views=views,
            validation=validation,
            test=test,
            weight_candidates=(
                weight_candidates
            ),
            use_integrity=True,
            use_quality=True,
            use_temporal=False,
        )
    )

    configurations.append(
        no_temporal
    )

    LOGGER.info(
        "A6 completed: FERF without temporal reliability"
    )

    # --------------------------------------------------------
    # A7 - Full FERF
    # --------------------------------------------------------

    full = (
        evaluate_weighted_configuration(
            configuration=(
                "A7_Full_FERF"
            ),
            description=(
                "Complete reliability-aware FERF using integrity, "
                "information quality, temporal reliability, "
                "validation-selected global view weights, and a "
                "validation-selected operating threshold."
            ),
            views=views,
            validation=validation,
            test=test,
            weight_candidates=(
                weight_candidates
            ),
            use_integrity=True,
            use_quality=True,
            use_temporal=True,
        )
    )

    configurations.append(
        full
    )

    LOGGER.info(
        "A7 completed: full FERF"
    )

    # --------------------------------------------------------
    # Save per-configuration test predictions
    # --------------------------------------------------------

    dataset_root = (
        RESULTS_ROOT / dataset
    )

    predictions_dir = (
        dataset_root
        / "Predictions"
    )

    metrics_dir = (
        dataset_root
        / "Metrics"
    )

    manifests_dir = (
        dataset_root
        / "Manifests"
    )

    degradation_dir = (
        dataset_root
        / "Evidence_Degradation"
    )

    for directory in (
        predictions_dir,
        metrics_dir,
        manifests_dir,
        degradation_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for item in configurations:
        probability = item[
            "test_probability"
        ]

        target = item[
            "test_target"
        ]

        prediction = (
            probability
            >= item["threshold"]
        ).astype(
            np.int8
        )

        pd.DataFrame(
            {
                "true_label": target,
                "probability": probability,
                "predicted_label": prediction,
            }
        ).to_csv(
            predictions_dir
            / (
                item[
                    "configuration"
                ]
                + "_test_predictions.csv"
            ),
            index=False,
        )

    # --------------------------------------------------------
    # Main ablation table
    # --------------------------------------------------------

    ablation_frame = (
        finalize_ablation_records(
            dataset,
            configurations,
        )
    )

    ablation_frame.to_csv(
        metrics_dir
        / "ablation_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Controlled evidence-availability degradation
    # --------------------------------------------------------

    LOGGER.info(
        "Running controlled evidence-availability degradation..."
    )

    degradation = (
        evaluate_degradation(
            dataset=dataset,
            views=views,
            validation=validation,
            test=test,
            full_configuration=full,
            global_configuration=(
                global_only
            ),
            repeats=(
                degradation_repeats
            ),
            seed=seed,
        )
    )

    degradation.to_csv(
        degradation_dir
        / "record_view_degradation_results.csv",
        index=False,
    )

    degradation_summary = (
        build_degradation_summary(
            degradation
        )
    )

    degradation_summary.to_csv(
        degradation_dir
        / "record_view_degradation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    save_json(
        manifests_dir
        / "experiment_04_ablation_manifest.json",
        {
            "generated_utc": utc_now(),
            "dataset": dataset,
            "source_multiview_predictions": str(
                MULTIVIEW_ROOT
                / dataset
                / "Predictions"
                / "view_predictions_and_reliability.csv"
            ),
            "source_original_feature_predictions": str(
                ORIGINAL_FEATURE_ROOT
                / dataset
                / "Predictions"
            ),
            "views": views,
            "validation_rows": len(
                validation[
                    "target"
                ]
            ),
            "test_rows": len(
                test[
                    "target"
                ]
            ),
            "random_seed": seed,
            "random_weight_candidates": (
                random_weight_candidates
            ),
            "total_weight_candidates": len(
                weight_candidates
            ),
            "threshold_search": {
                "minimum": float(
                    THRESHOLDS.min()
                ),
                "maximum": float(
                    THRESHOLDS.max()
                ),
                "step": 0.01,
                "selection_data": (
                    "validation only"
                ),
            },
            "base_reliability_coefficients": {
                "integrity": (
                    INTEGRITY_WEIGHT
                ),
                "quality": (
                    QUALITY_WEIGHT
                ),
                "temporal": (
                    TEMPORAL_WEIGHT
                ),
            },
            "ablation_rule": (
                "When a reliability component is removed, "
                "the remaining coefficients are renormalized "
                "to sum to one."
            ),
            "configurations": [
                {
                    "configuration": item[
                        "configuration"
                    ],
                    "description": item[
                        "description"
                    ],
                    "threshold": item[
                        "threshold"
                    ],
                    "weights": item[
                        "weights"
                    ],
                    "reliability_components": (
                        item[
                            "reliability_components"
                        ]
                    ),
                }
                for item in configurations
            ],
            "evidence_degradation": {
                "interpretation": (
                    "record-view evidence availability degradation"
                ),
                "raw_features_modified": False,
                "degradation_levels": list(
                    DEGRADATION_LEVELS
                ),
                "repeats": (
                    degradation_repeats
                ),
                "neutral_probability": (
                    NEUTRAL_PROBABILITY
                ),
                "degraded_reliability_floor": (
                    DEGRADED_RELIABILITY_FLOOR
                ),
                "retuning_on_degraded_test_data": False,
            },
            "test_data_policy": (
                "Test labels are not used for global-weight "
                "or operating-threshold selection."
            ),
            "model_training": (
                "No predictive models are trained or retrained "
                "by Experiment 4."
            ),
        },
    )

    LOGGER.info(
        "%s | full FERF test BA=%.6f | F1=%.6f | MCC=%.6f | AUC=%.6f",
        dataset,
        full[
            "test_metrics"
        ][
            "balanced_accuracy"
        ],
        full[
            "test_metrics"
        ]["f1"],
        full[
            "test_metrics"
        ]["mcc"],
        full[
            "test_metrics"
        ]["roc_auc"],
    )

    return (
        ablation_frame,
        degradation,
    )


# ============================================================
# Consolidated outputs
# ============================================================

def save_consolidated_outputs(
    ablation_frames: list[
        pd.DataFrame
    ],
    degradation_frames: list[
        pd.DataFrame
    ],
) -> None:

    if ablation_frames:
        ablation = pd.concat(
            ablation_frames,
            ignore_index=True,
        )

        ablation.to_csv(
            REPORTS_DIR
            / "Ablation_Results.csv",
            index=False,
        )

        delta_columns = [
            "dataset",
            "configuration",
            "test_balanced_accuracy",
            "test_f1",
            "test_mcc",
            "test_roc_auc",
            "delta_balanced_accuracy_vs_full",
            "delta_f1_vs_full",
            "delta_mcc_vs_full",
            "delta_roc_auc_vs_full",
        ]

        ablation[
            delta_columns
        ].to_csv(
            REPORTS_DIR
            / "Ablation_Deltas_vs_Full_FERF.csv",
            index=False,
        )

        ranking = (
            ablation
            .sort_values(
                by=[
                    "dataset",
                    "test_balanced_accuracy",
                    "test_f1",
                    "test_mcc",
                    "test_roc_auc",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            )
        )

        ranking.to_csv(
            REPORTS_DIR
            / "Ablation_Ranking.csv",
            index=False,
        )

    if degradation_frames:
        degradation = pd.concat(
            degradation_frames,
            ignore_index=True,
        )

        degradation.to_csv(
            REPORTS_DIR
            / "Evidence_Degradation_Results.csv",
            index=False,
        )

        summary = (
            build_degradation_summary(
                degradation
            )
        )

        summary.to_csv(
            REPORTS_DIR
            / "Evidence_Degradation_Summary.csv",
            index=False,
        )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    runnable = (
        discover_runnable_datasets()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Experiment 4: component-wise FERF ablation "
            "and record-view evidence availability degradation."
        )
    )

    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset to evaluate, or 'all'. "
            f"Currently discoverable: "
            f"{', '.join(runnable) if runnable else 'none'}"
        ),
    )

    parser.add_argument(
        "--weight-candidates",
        type=int,
        default=(
            DEFAULT_WEIGHT_CANDIDATES
        ),
        help=(
            "Number of reproducible random Dirichlet "
            "global-view-weight candidates. "
            f"Default: {DEFAULT_WEIGHT_CANDIDATES}."
        ),
    )

    parser.add_argument(
        "--degradation-repeats",
        type=int,
        default=(
            DEFAULT_DEGRADATION_REPEATS
        ),
        help=(
            "Number of random degradation masks for each "
            "degradation level. "
            f"Default: {DEFAULT_DEGRADATION_REPEATS}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            f"Random seed. Default: {RANDOM_SEED}."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    if args.weight_candidates <= 0:
        raise ValueError(
            "--weight-candidates must be positive."
        )

    if args.degradation_repeats <= 0:
        raise ValueError(
            "--degradation-repeats must be positive."
        )

    runnable = (
        discover_runnable_datasets()
    )

    if not runnable:
        LOGGER.error(
            "No dataset contains the required multiview "
            "prediction file."
        )
        return 1

    if args.dataset == "all":
        selected = runnable
    else:
        if args.dataset not in runnable:
            LOGGER.error(
                "Dataset '%s' is not runnable. "
                "Runnable datasets: %s",
                args.dataset,
                ", ".join(runnable),
            )
            return 1

        selected = [
            args.dataset
        ]

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 4: COMPONENT-WISE ABLATION STUDY"
    )
    LOGGER.info(
        "Datasets: %s",
        ", ".join(selected),
    )
    LOGGER.info(
        "No predictive model retraining will be performed."
    )
    LOGGER.info(
        "Global weights and thresholds are selected "
        "using validation data only."
    )
    LOGGER.info("=" * 78)

    start = time.perf_counter()

    ablation_frames = []
    degradation_frames = []
    failures = []

    for dataset in selected:
        try:
            (
                ablation_frame,
                degradation_frame,
            ) = run_dataset(
                dataset=dataset,
                random_weight_candidates=(
                    args.weight_candidates
                ),
                degradation_repeats=(
                    args.degradation_repeats
                ),
                seed=args.seed,
            )

            ablation_frames.append(
                ablation_frame
            )

            degradation_frames.append(
                degradation_frame
            )

        except Exception:
            failures.append(
                dataset
            )

            LOGGER.exception(
                "Experiment 4 failed for dataset=%s",
                dataset,
            )

    save_consolidated_outputs(
        ablation_frames,
        degradation_frames,
    )

    save_json(
        MANIFESTS_DIR
        / "experiment_04_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(
                __file__
            ).name,
            "experiment": (
                "Experiment 4: Ablation Study"
            ),
            "requested_dataset": (
                args.dataset
            ),
            "selected_datasets": (
                selected
            ),
            "successful_datasets": [
                str(
                    frame[
                        "dataset"
                    ].iloc[0]
                )
                for frame in ablation_frames
                if not frame.empty
            ],
            "failed_datasets": (
                failures
            ),
            "weight_candidates": (
                args.weight_candidates
            ),
            "degradation_repeats": (
                args.degradation_repeats
            ),
            "seed": args.seed,
            "elapsed_seconds": (
                time.perf_counter()
                - start
            ),
            "no_model_retraining": True,
        },
    )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Successful datasets: %d",
        len(
            ablation_frames
        ),
    )
    LOGGER.info(
        "Failed datasets: %d",
        len(
            failures
        ),
    )
    LOGGER.info(
        "Elapsed time: %.2f minutes",
        (
            time.perf_counter()
            - start
        ) / 60.0,
    )
    LOGGER.info(
        "Results directory: %s",
        RESULTS_ROOT,
    )
    LOGGER.info("=" * 78)

    if not ablation_frames:
        return 1

    if failures:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
