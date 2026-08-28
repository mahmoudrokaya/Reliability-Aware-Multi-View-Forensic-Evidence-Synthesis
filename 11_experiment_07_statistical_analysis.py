"""
11_experiment_07_statistical_analysis.py

Experiment 7: Statistical Analysis
Digital Forensic Evidence Synthesis Framework

Purpose
-------
Provide inferential validation of performance differences using genuine
repeated stratified cross-validation.

Protocol
--------
- RepeatedStratifiedKFold: 10 repetitions x 5 folds = 50 outer evaluations.
- Each outer fold holds out 20% for evaluation.
- The remaining 80% is split into:
      87.5% inner training  = 70% of the full fold population
      12.5% inner validation = 10% of the full fold population
  thereby preserving the study's 70/10/20 train/validation/test logic.
- All preprocessing, predictive-model fitting, quality-estimator fitting,
  fusion-weight selection, and operating-threshold selection are performed
  without using the outer test fold.
- The exact same outer folds are used for every compared configuration.

Configurations
--------------
C0  Original-feature XGBoost
C1  Unweighted multi-view fusion
C2  Global-weighted multi-view fusion (no record-specific reliability)
C3  Full FERF (global weights + record-specific reliability)

Metrics
-------
- Macro-F1
- Weighted-F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- Cohen's kappa
- MAE (mean absolute error between true binary label and attack probability)
- Runtime

Statistical comparisons
-----------------------
Full FERF is compared pairwise against:
- Original-feature XGBoost
- Unweighted multi-view fusion
- Global-weighted multi-view fusion

For every metric:
- paired t-test
- Wilcoxon signed-rank test
- bootstrap 95% confidence interval for the paired mean difference
- Cohen's dz paired effect size
- Holm-adjusted p-values are additionally reported for transparency

Important
---------
This experiment genuinely retrains the models inside each repeated CV fold.
It does not create pseudo-replicates by resampling fixed held-out predictions.

Default sample size
-------------------
The default is 200,000 records per dataset to make 50 repeated model-development
cycles computationally practical. The exact number used is saved in every
manifest. Use --max-rows 1000000 if a 1M-record repeated-CV experiment is
required and computational resources permit it.

Run
---
python 11_experiment_07_statistical_analysis.py --dataset all

Examples
--------
python 11_experiment_07_statistical_analysis.py --dataset CICIDS2017
python 11_experiment_07_statistical_analysis.py --dataset all --max-rows 200000
python 11_experiment_07_statistical_analysis.py --dataset all --weight-candidates 300
python 11_experiment_07_statistical_analysis.py --dataset CICIDS2017 --repeats 2 --folds 5
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ferf_common import (
    QualityEstimator,
    build_views,
    create_preprocessor,
    create_xgboost,
    integrity_scores,
    select_predictors,
    temporal_scores,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

CLEANING_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_01_Data_Preparation"
    / "Phase_02_Integrity_Verification"
    / "Step_02_Data_Cleaning"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_07_Statistical_Analysis"
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
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Configuration
# ============================================================

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
)

RANDOM_SEED = 42

DEFAULT_MAX_ROWS = 200_000
DEFAULT_REPEATS = 10
DEFAULT_FOLDS = 5
DEFAULT_ESTIMATORS = 300
DEFAULT_WEIGHT_CANDIDATES = 300
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000

INTEGRITY_WEIGHT = 0.35
QUALITY_WEIGHT = 0.45
TEMPORAL_WEIGHT = 0.20

THRESHOLDS = np.arange(
    0.20,
    0.801,
    0.01,
)

EPSILON = 1e-12

CONFIGURATION_ORDER = (
    "C0_Original_XGBoost",
    "C1_Unweighted_Multiview",
    "C2_Global_Weighted_Multiview",
    "C3_Full_FERF",
)

METRIC_COLUMNS = (
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "cohen_kappa",
    "mae",
    "runtime_seconds",
)

HIGHER_IS_BETTER = {
    "macro_f1": True,
    "weighted_f1": True,
    "balanced_accuracy": True,
    "roc_auc": True,
    "pr_auc": True,
    "cohen_kappa": True,
    "mae": False,
    "runtime_seconds": False,
}


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger(
    "experiment_07_statistical_analysis"
)
LOGGER.setLevel(
    logging.INFO
)
LOGGER.propagate = False
LOGGER.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console = logging.StreamHandler(
    sys.stdout
)
console.setFormatter(
    formatter
)

file_handler = logging.FileHandler(
    LOGS_DIR
    / "experiment_07_statistical_analysis.log",
    mode="w",
    encoding="utf-8",
)
file_handler.setFormatter(
    formatter
)

LOGGER.addHandler(
    console
)
LOGGER.addHandler(
    file_handler
)


# ============================================================
# Structures
# ============================================================

@dataclass
class FoldResult:
    dataset: str
    repetition: int
    fold: int
    evaluation_id: int
    configuration: str

    total_rows: int
    inner_train_rows: int
    inner_validation_rows: int
    outer_test_rows: int
    number_of_views: int

    threshold: float

    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    cohen_kappa: float
    mae: float

    training_seconds: float
    validation_selection_seconds: float
    inference_seconds: float
    runtime_seconds: float

    weights_json: str


# ============================================================
# Utilities
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


def format_seconds(
    seconds: float,
) -> str:
    seconds = max(
        float(
            seconds
        ),
        0.0,
    )

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = (
        seconds
        / 60.0
    )

    if minutes < 60:
        return f"{minutes:.2f}m"

    return (
        f"{minutes / 60.0:.2f}h"
    )


# ============================================================
# Dataset reconstruction
# ============================================================

def discover_parts(
    dataset: str,
) -> list[Path]:
    directory = (
        CLEANING_ROOT
        / dataset
        / "Cleaned_Data"
    )

    parquet = sorted(
        directory.glob(
            "cleaned_part_*.parquet"
        )
    )

    if parquet:
        return parquet

    return sorted(
        directory.glob(
            "cleaned_part_*.csv.gz"
        )
    )


def read_part(
    path: Path,
) -> pd.DataFrame:
    if (
        path.suffix.lower()
        == ".parquet"
    ):
        return pd.read_parquet(
            path
        )

    return pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )


def count_rows(
    path: Path,
) -> int:
    if (
        path.suffix.lower()
        == ".parquet"
    ):
        import pyarrow.parquet as pq

        return int(
            pq.ParquetFile(
                path
            ).metadata.num_rows
        )

    total = 0

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=[0],
        chunksize=250_000,
    ):
        total += len(
            chunk
        )

    return total


def allocate_rows(
    sizes: list[int],
    requested: int,
) -> list[int]:
    total = sum(
        sizes
    )

    if (
        requested <= 0
        or requested >= total
    ):
        return sizes.copy()

    exact = [
        requested
        * size
        / total
        for size in sizes
    ]

    allocation = [
        min(
            size,
            int(
                np.floor(
                    value
                )
            ),
        )
        for size, value
        in zip(
            sizes,
            exact,
        )
    ]

    remainder = (
        requested
        - sum(
            allocation
        )
    )

    order = np.argsort(
        [
            value
            - np.floor(
                value
            )
            for value in exact
        ]
    )[::-1]

    for index in order:
        if remainder <= 0:
            break

        if (
            allocation[
                index
            ]
            < sizes[
                index
            ]
        ):
            allocation[
                index
            ] += 1
            remainder -= 1

    return allocation


def load_sample(
    dataset: str,
    maximum_rows: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    parts = discover_parts(
        dataset
    )

    if not parts:
        raise FileNotFoundError(
            f"No cleaned data parts found for {dataset}."
        )

    LOGGER.info(
        "%s | counting %d cleaned parts...",
        dataset,
        len(
            parts
        ),
    )

    sizes = [
        count_rows(
            path
        )
        for path in parts
    ]

    total_available = sum(
        sizes
    )

    requested = (
        total_available
        if maximum_rows <= 0
        else min(
            maximum_rows,
            total_available,
        )
    )

    allocation = allocate_rows(
        sizes,
        requested,
    )

    frames = []

    for index, (
        path,
        part_size,
        selected,
    ) in enumerate(
        zip(
            parts,
            sizes,
            allocation,
        ),
        start=1,
    ):
        if selected <= 0:
            continue

        frame = read_part(
            path
        )

        if selected < part_size:
            frame = frame.sample(
                n=selected,
                random_state=(
                    seed
                    + index
                ),
            )

        frames.append(
            frame
        )

        LOGGER.info(
            "%s | loaded part %d/%d | selected=%d/%d",
            dataset,
            index,
            len(
                parts
            ),
            len(
                frame
            ),
            part_size,
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if (
        len(
            dataframe
        )
        > requested
    ):
        dataframe = dataframe.sample(
            n=requested,
            random_state=seed,
        ).reset_index(
            drop=True
        )

    LOGGER.info(
        "%s | deterministic statistical sample=%d/%d",
        dataset,
        len(
            dataframe
        ),
        total_available,
    )

    return (
        dataframe,
        total_available,
    )


# ============================================================
# Metrics
# ============================================================

def evaluate_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[
    str,
    float,
]:
    target = np.asarray(
        target,
        dtype=np.int8,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability
        >= threshold
    ).astype(
        np.int8
    )

    return {
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                target,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                prediction,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                target,
                probability,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                target,
                prediction,
            )
        ),
        "mae": float(
            mean_absolute_error(
                target,
                probability,
            )
        ),
    }


# ============================================================
# Vectorized threshold selection
# ============================================================

def optimize_threshold(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[
    float,
    dict[str, float],
]:
    target = np.asarray(
        target,
        dtype=np.int8,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    predictions = (
        probability[
            :,
            None,
        ]
        >= THRESHOLDS[
            None,
            :,
        ]
    )

    positive = (
        target
        == 1
    )

    negative = (
        ~positive
    )

    tp = predictions[
        positive
    ].sum(
        axis=0,
        dtype=np.int64,
    )

    fn = (
        int(
            positive.sum()
        )
        - tp
    )

    fp = predictions[
        negative
    ].sum(
        axis=0,
        dtype=np.int64,
    )

    tn = (
        int(
            negative.sum()
        )
        - fp
    )

    sensitivity = (
        tp
        / np.maximum(
            tp
            + fn,
            1,
        )
    )

    specificity = (
        tn
        / np.maximum(
            tn
            + fp,
            1,
        )
    )

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2.0

    f1 = (
        2.0
        * tp
        / np.maximum(
            2.0
            * tp
            + fp
            + fn,
            1,
        )
    )

    denominator = np.sqrt(
        np.maximum(
            (
                tp
                + fp
            )
            * (
                tp
                + fn
            )
            * (
                tn
                + fp
            )
            * (
                tn
                + fn
            ),
            1,
        )
    )

    mcc = (
        tp
        * tn
        - fp
        * fn
    ) / denominator

    best_index = max(
        range(
            len(
                THRESHOLDS
            )
        ),
        key=lambda index: (
            float(
                balanced_accuracy[
                    index
                ]
            ),
            float(
                f1[
                    index
                ]
            ),
            float(
                mcc[
                    index
                ]
            ),
            -abs(
                float(
                    THRESHOLDS[
                        index
                    ]
                )
                - 0.5
            ),
        ),
    )

    threshold = float(
        THRESHOLDS[
            best_index
        ]
    )

    return (
        threshold,
        evaluate_metrics(
            target,
            probability,
            threshold,
        ),
    )


# ============================================================
# Reliability and FERF
# ============================================================

def combine_reliability(
    integrity: np.ndarray,
    quality: np.ndarray,
    temporal: np.ndarray,
) -> np.ndarray:
    return np.clip(
        INTEGRITY_WEIGHT
        * integrity
        + QUALITY_WEIGHT
        * quality
        + TEMPORAL_WEIGHT
        * temporal,
        0.0,
        1.0,
    )


def fuse_probabilities(
    probabilities: np.ndarray,
    reliabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_reliability = (
        reliabilities
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
        * probabilities
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
    rng = np.random.default_rng(
        seed
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

    random_weights = rng.dirichlet(
        np.ones(
            number_of_views
        ),
        size=max(
            random_candidates,
            1,
        ),
    )

    return np.vstack(
        [
            equal,
            one_hot,
            random_weights,
        ]
    )


def optimize_fusion_weights(
    validation_target: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_reliabilities: np.ndarray,
    candidates: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    best_key = None
    best_weights = None
    best_threshold = None

    for weights in candidates:
        fused = fuse_probabilities(
            validation_probabilities,
            validation_reliabilities,
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
            metrics[
                "macro_f1"
            ],
            metrics[
                "cohen_kappa"
            ],
            -abs(
                threshold
                - 0.5
            ),
        )

        if (
            best_key is None
            or key
            > best_key
        ):
            best_key = key
            best_weights = (
                weights.copy()
            )
            best_threshold = (
                threshold
            )

    assert best_weights is not None
    assert best_threshold is not None

    return (
        best_weights,
        best_threshold,
    )


# ============================================================
# Fold execution
# ============================================================

def build_pipeline(
    training_features: pd.DataFrame,
    seed: int,
    estimators: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessing",
                create_preprocessor(
                    training_features
                ),
            ),
            (
                "classifier",
                create_xgboost(
                    seed=seed,
                    estimators=estimators,
                ),
            ),
        ]
    )


def run_fold(
    dataset: str,
    features: pd.DataFrame,
    target: pd.Series,
    raw: pd.DataFrame,
    views: dict[
        str,
        list[str],
    ],
    integrity_all: np.ndarray,
    temporal_all: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
    repetition: int,
    fold: int,
    evaluation_id: int,
    estimators: int,
    weight_candidates_count: int,
    seed: int,
) -> list[
    FoldResult
]:
    fold_seed = (
        seed
        + repetition
        * 10_000
        + fold
        * 100
    )

    # Preserve total 70/10/20 proportions:
    # outer train is 80%; validation is 12.5% of outer train = 10% total.
    (
        inner_train_indices,
        inner_validation_indices,
    ) = train_test_split(
        outer_train_indices,
        test_size=0.125,
        stratify=target.iloc[
            outer_train_indices
        ],
        random_state=fold_seed,
    )

    y_train = target.iloc[
        inner_train_indices
    ]

    y_validation = target.iloc[
        inner_validation_indices
    ].to_numpy(
        dtype=np.int8
    )

    y_test = target.iloc[
        outer_test_indices
    ].to_numpy(
        dtype=np.int8
    )

    LOGGER.info(
        "%s | rep=%d/%d fold=%d/%d eval=%d | "
        "train=%d validation=%d test=%d",
        dataset,
        repetition,
        DEFAULT_REPEATS,
        fold,
        DEFAULT_FOLDS,
        evaluation_id,
        len(
            inner_train_indices
        ),
        len(
            inner_validation_indices
        ),
        len(
            outer_test_indices
        ),
    )

    results = []

    # --------------------------------------------------------
    # C0 Original-feature XGBoost
    # --------------------------------------------------------

    original_start = time.perf_counter()

    original_model = build_pipeline(
        features.iloc[
            inner_train_indices
        ],
        fold_seed,
        estimators,
    )

    fit_start = time.perf_counter()

    original_model.fit(
        features.iloc[
            inner_train_indices
        ],
        y_train,
    )

    original_training = (
        time.perf_counter()
        - fit_start
    )

    validation_selection_start = (
        time.perf_counter()
    )

    original_validation_probability = (
        original_model.predict_proba(
            features.iloc[
                inner_validation_indices
            ]
        )[
            :,
            1,
        ]
    )

    (
        original_threshold,
        _,
    ) = optimize_threshold(
        y_validation,
        original_validation_probability,
    )

    original_validation_selection = (
        time.perf_counter()
        - validation_selection_start
    )

    inference_start = time.perf_counter()

    original_test_probability = (
        original_model.predict_proba(
            features.iloc[
                outer_test_indices
            ]
        )[
            :,
            1,
        ]
    )

    original_inference = (
        time.perf_counter()
        - inference_start
    )

    original_metrics = evaluate_metrics(
        y_test,
        original_test_probability,
        original_threshold,
    )

    original_runtime = (
        time.perf_counter()
        - original_start
    )

    results.append(
        FoldResult(
            dataset=dataset,
            repetition=repetition,
            fold=fold,
            evaluation_id=evaluation_id,
            configuration=(
                "C0_Original_XGBoost"
            ),
            total_rows=len(
                features
            ),
            inner_train_rows=len(
                inner_train_indices
            ),
            inner_validation_rows=len(
                inner_validation_indices
            ),
            outer_test_rows=len(
                outer_test_indices
            ),
            number_of_views=len(
                views
            ),
            threshold=(
                original_threshold
            ),
            **original_metrics,
            training_seconds=(
                original_training
            ),
            validation_selection_seconds=(
                original_validation_selection
            ),
            inference_seconds=(
                original_inference
            ),
            runtime_seconds=(
                original_runtime
            ),
            weights_json="{}",
        )
    )

    del original_model
    gc.collect()

    # --------------------------------------------------------
    # Train all view-level models once for C1-C3
    # --------------------------------------------------------

    multiview_start = time.perf_counter()
    multiview_training_seconds = 0.0

    view_names = list(
        views.keys()
    )

    validation_probability_columns = []
    test_probability_columns = []
    validation_reliability_columns = []
    test_reliability_columns = []

    for view_index, view_name in enumerate(
        view_names,
        start=1,
    ):
        columns = views[
            view_name
        ]

        view_train = features.iloc[
            inner_train_indices
        ][
            columns
        ]

        model = build_pipeline(
            view_train,
            (
                fold_seed
                + view_index
            ),
            estimators,
        )

        fit_start = time.perf_counter()

        model.fit(
            view_train,
            y_train,
        )

        multiview_training_seconds += (
            time.perf_counter()
            - fit_start
        )

        quality_estimator = (
            QualityEstimator()
            .fit(
                view_train
            )
        )

        validation_probability = (
            model.predict_proba(
                features.iloc[
                    inner_validation_indices
                ][
                    columns
                ]
            )[
                :,
                1,
            ]
        )

        test_probability = (
            model.predict_proba(
                features.iloc[
                    outer_test_indices
                ][
                    columns
                ]
            )[
                :,
                1,
            ]
        )

        validation_quality = (
            quality_estimator.transform(
                features.iloc[
                    inner_validation_indices
                ][
                    columns
                ]
            )
        )

        test_quality = (
            quality_estimator.transform(
                features.iloc[
                    outer_test_indices
                ][
                    columns
                ]
            )
        )

        validation_integrity = (
            integrity_all[
                inner_validation_indices
            ]
        )

        test_integrity = (
            integrity_all[
                outer_test_indices
            ]
        )

        if (
            view_name
            == "temporal"
        ):
            validation_temporal = (
                temporal_all[
                    inner_validation_indices
                ]
            )

            test_temporal = (
                temporal_all[
                    outer_test_indices
                ]
            )
        else:
            validation_temporal = np.ones(
                len(
                    inner_validation_indices
                ),
                dtype=float,
            )

            test_temporal = np.ones(
                len(
                    outer_test_indices
                ),
                dtype=float,
            )

        validation_reliability = (
            combine_reliability(
                validation_integrity,
                validation_quality,
                validation_temporal,
            )
        )

        test_reliability = (
            combine_reliability(
                test_integrity,
                test_quality,
                test_temporal,
            )
        )

        validation_probability_columns.append(
            validation_probability
        )

        test_probability_columns.append(
            test_probability
        )

        validation_reliability_columns.append(
            validation_reliability
        )

        test_reliability_columns.append(
            test_reliability
        )

        del model
        del quality_estimator
        gc.collect()

    validation_probabilities = (
        np.column_stack(
            validation_probability_columns
        )
    )

    test_probabilities = (
        np.column_stack(
            test_probability_columns
        )
    )

    validation_reliabilities = (
        np.column_stack(
            validation_reliability_columns
        )
    )

    test_reliabilities = (
        np.column_stack(
            test_reliability_columns
        )
    )

    # Shared weight candidates for C2/C3.
    candidates = generate_weight_candidates(
        len(
            view_names
        ),
        weight_candidates_count,
        fold_seed,
    )

    # --------------------------------------------------------
    # C1 Unweighted multi-view
    # --------------------------------------------------------

    c1_start = time.perf_counter()

    unweighted_validation_probability = (
        validation_probabilities.mean(
            axis=1
        )
    )

    (
        unweighted_threshold,
        _,
    ) = optimize_threshold(
        y_validation,
        unweighted_validation_probability,
    )

    unweighted_test_probability = (
        test_probabilities.mean(
            axis=1
        )
    )

    c1_metrics = evaluate_metrics(
        y_test,
        unweighted_test_probability,
        unweighted_threshold,
    )

    c1_selection_inference = (
        time.perf_counter()
        - c1_start
    )

    results.append(
        FoldResult(
            dataset=dataset,
            repetition=repetition,
            fold=fold,
            evaluation_id=evaluation_id,
            configuration=(
                "C1_Unweighted_Multiview"
            ),
            total_rows=len(
                features
            ),
            inner_train_rows=len(
                inner_train_indices
            ),
            inner_validation_rows=len(
                inner_validation_indices
            ),
            outer_test_rows=len(
                outer_test_indices
            ),
            number_of_views=len(
                views
            ),
            threshold=(
                unweighted_threshold
            ),
            **c1_metrics,
            training_seconds=(
                multiview_training_seconds
            ),
            validation_selection_seconds=(
                c1_selection_inference
            ),
            inference_seconds=0.0,
            runtime_seconds=(
                multiview_training_seconds
                + c1_selection_inference
            ),
            weights_json=json.dumps(
                {
                    view_name: (
                        1.0
                        / len(
                            view_names
                        )
                    )
                    for view_name
                    in view_names
                },
                sort_keys=True,
            ),
        )
    )

    # --------------------------------------------------------
    # C2 Global weights only
    # --------------------------------------------------------

    c2_start = time.perf_counter()

    unit_validation_reliability = np.ones_like(
        validation_probabilities
    )

    unit_test_reliability = np.ones_like(
        test_probabilities
    )

    (
        global_weights,
        global_threshold,
    ) = optimize_fusion_weights(
        y_validation,
        validation_probabilities,
        unit_validation_reliability,
        candidates,
    )

    global_test_probability = (
        fuse_probabilities(
            test_probabilities,
            unit_test_reliability,
            global_weights,
        )
    )

    c2_metrics = evaluate_metrics(
        y_test,
        global_test_probability,
        global_threshold,
    )

    c2_selection_inference = (
        time.perf_counter()
        - c2_start
    )

    results.append(
        FoldResult(
            dataset=dataset,
            repetition=repetition,
            fold=fold,
            evaluation_id=evaluation_id,
            configuration=(
                "C2_Global_Weighted_Multiview"
            ),
            total_rows=len(
                features
            ),
            inner_train_rows=len(
                inner_train_indices
            ),
            inner_validation_rows=len(
                inner_validation_indices
            ),
            outer_test_rows=len(
                outer_test_indices
            ),
            number_of_views=len(
                views
            ),
            threshold=(
                global_threshold
            ),
            **c2_metrics,
            training_seconds=(
                multiview_training_seconds
            ),
            validation_selection_seconds=(
                c2_selection_inference
            ),
            inference_seconds=0.0,
            runtime_seconds=(
                multiview_training_seconds
                + c2_selection_inference
            ),
            weights_json=json.dumps(
                {
                    view_name: float(
                        global_weights[
                            index
                        ]
                    )
                    for index, view_name
                    in enumerate(
                        view_names
                    )
                },
                sort_keys=True,
            ),
        )
    )

    # --------------------------------------------------------
    # C3 Full FERF
    # --------------------------------------------------------

    c3_start = time.perf_counter()

    (
        ferf_weights,
        ferf_threshold,
    ) = optimize_fusion_weights(
        y_validation,
        validation_probabilities,
        validation_reliabilities,
        candidates,
    )

    ferf_test_probability = (
        fuse_probabilities(
            test_probabilities,
            test_reliabilities,
            ferf_weights,
        )
    )

    c3_metrics = evaluate_metrics(
        y_test,
        ferf_test_probability,
        ferf_threshold,
    )

    c3_selection_inference = (
        time.perf_counter()
        - c3_start
    )

    results.append(
        FoldResult(
            dataset=dataset,
            repetition=repetition,
            fold=fold,
            evaluation_id=evaluation_id,
            configuration=(
                "C3_Full_FERF"
            ),
            total_rows=len(
                features
            ),
            inner_train_rows=len(
                inner_train_indices
            ),
            inner_validation_rows=len(
                inner_validation_indices
            ),
            outer_test_rows=len(
                outer_test_indices
            ),
            number_of_views=len(
                views
            ),
            threshold=(
                ferf_threshold
            ),
            **c3_metrics,
            training_seconds=(
                multiview_training_seconds
            ),
            validation_selection_seconds=(
                c3_selection_inference
            ),
            inference_seconds=0.0,
            runtime_seconds=(
                multiview_training_seconds
                + c3_selection_inference
            ),
            weights_json=json.dumps(
                {
                    view_name: float(
                        ferf_weights[
                            index
                        ]
                    )
                    for index, view_name
                    in enumerate(
                        view_names
                    )
                },
                sort_keys=True,
            ),
        )
    )

    multiview_elapsed = (
        time.perf_counter()
        - multiview_start
    )

    LOGGER.info(
        "%s | eval=%d completed | "
        "Original BA=%.6f | Unweighted BA=%.6f | "
        "Global BA=%.6f | FERF BA=%.6f | elapsed=%s",
        dataset,
        evaluation_id,
        results[
            -4
        ].balanced_accuracy,
        results[
            -3
        ].balanced_accuracy,
        results[
            -2
        ].balanced_accuracy,
        results[
            -1
        ].balanced_accuracy,
        format_seconds(
            original_runtime
            + multiview_elapsed
        ),
    )

    return results


# ============================================================
# Statistical analysis
# ============================================================

def bootstrap_mean_difference_ci(
    differences: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[
    float,
    float,
]:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    number = len(
        differences
    )

    bootstrap_means = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(
        iterations
    ):
        sample = rng.choice(
            differences,
            size=number,
            replace=True,
        )

        bootstrap_means[
            index
        ] = sample.mean()

    lower, upper = np.percentile(
        bootstrap_means,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(
            lower
        ),
        float(
            upper
        ),
    )


def cohens_dz(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    standard_deviation = differences.std(
        ddof=1
    )

    if (
        not np.isfinite(
            standard_deviation
        )
        or standard_deviation
        <= EPSILON
    ):
        return 0.0

    return float(
        differences.mean()
        / standard_deviation
    )


def holm_adjust(
    p_values: list[
        float
    ],
) -> list[
    float
]:
    p = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p
    )

    adjusted = np.empty_like(
        p
    )

    running_max = 0.0
    number = len(
        p
    )

    for rank, index in enumerate(
        order
    ):
        value = min(
            (
                number
                - rank
            )
            * p[
                index
            ],
            1.0,
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[
            index
        ] = min(
            running_max,
            1.0,
        )

    return [
        float(
            value
        )
        for value in adjusted
    ]


def statistical_comparisons(
    fold_frame: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        (
            "C3_Full_FERF",
            "C0_Original_XGBoost",
        ),
        (
            "C3_Full_FERF",
            "C1_Unweighted_Multiview",
        ),
        (
            "C3_Full_FERF",
            "C2_Global_Weighted_Multiview",
        ),
    ]

    records = []

    for dataset in sorted(
        fold_frame[
            "dataset"
        ].unique()
    ):
        dataset_frame = fold_frame.loc[
            fold_frame[
                "dataset"
            ].eq(
                dataset
            )
        ]

        for (
            configuration_a,
            configuration_b,
        ) in comparisons:
            a = (
                dataset_frame.loc[
                    dataset_frame[
                        "configuration"
                    ].eq(
                        configuration_a
                    )
                ]
                .sort_values(
                    "evaluation_id"
                )
            )

            b = (
                dataset_frame.loc[
                    dataset_frame[
                        "configuration"
                    ].eq(
                        configuration_b
                    )
                ]
                .sort_values(
                    "evaluation_id"
                )
            )

            if (
                len(
                    a
                )
                != len(
                    b
                )
                or not np.array_equal(
                    a[
                        "evaluation_id"
                    ].to_numpy(),
                    b[
                        "evaluation_id"
                    ].to_numpy(),
                )
            ):
                raise ValueError(
                    f"Paired observations are inconsistent for "
                    f"{dataset}: {configuration_a} vs {configuration_b}"
                )

            metric_record_indices = []

            for metric in METRIC_COLUMNS:
                values_a = a[
                    metric
                ].to_numpy(
                    dtype=float
                )

                values_b = b[
                    metric
                ].to_numpy(
                    dtype=float
                )

                differences = (
                    values_a
                    - values_b
                )

                t_result = ttest_rel(
                    values_a,
                    values_b,
                    nan_policy="omit",
                )

                try:
                    wilcoxon_result = wilcoxon(
                        values_a,
                        values_b,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )

                    wilcoxon_statistic = float(
                        wilcoxon_result.statistic
                    )

                    wilcoxon_p = float(
                        wilcoxon_result.pvalue
                    )
                except ValueError:
                    wilcoxon_statistic = 0.0
                    wilcoxon_p = 1.0

                (
                    ci_lower,
                    ci_upper,
                ) = bootstrap_mean_difference_ci(
                    differences,
                    bootstrap_iterations,
                    (
                        seed
                        + len(
                            records
                        )
                    ),
                )

                record = {
                    "dataset": dataset,
                    "configuration_a": (
                        configuration_a
                    ),
                    "configuration_b": (
                        configuration_b
                    ),
                    "metric": metric,
                    "higher_is_better": (
                        HIGHER_IS_BETTER[
                            metric
                        ]
                    ),
                    "paired_evaluations": (
                        len(
                            values_a
                        )
                    ),
                    "mean_a": float(
                        values_a.mean()
                    ),
                    "std_a": float(
                        values_a.std(
                            ddof=1
                        )
                    ),
                    "mean_b": float(
                        values_b.mean()
                    ),
                    "std_b": float(
                        values_b.std(
                            ddof=1
                        )
                    ),
                    "mean_difference_a_minus_b": float(
                        differences.mean()
                    ),
                    "bootstrap_ci_95_lower": (
                        ci_lower
                    ),
                    "bootstrap_ci_95_upper": (
                        ci_upper
                    ),
                    "paired_t_statistic": float(
                        t_result.statistic
                    ),
                    "paired_t_p_value": float(
                        t_result.pvalue
                    ),
                    "wilcoxon_statistic": (
                        wilcoxon_statistic
                    ),
                    "wilcoxon_p_value": (
                        wilcoxon_p
                    ),
                    "cohens_dz": (
                        cohens_dz(
                            differences
                        )
                    ),
                }

                records.append(
                    record
                )

                metric_record_indices.append(
                    len(
                        records
                    )
                    - 1
                )

            # Holm correction across the eight metrics for
            # each dataset/configuration comparison.
            t_values = [
                records[
                    index
                ][
                    "paired_t_p_value"
                ]
                for index in metric_record_indices
            ]

            w_values = [
                records[
                    index
                ][
                    "wilcoxon_p_value"
                ]
                for index in metric_record_indices
            ]

            adjusted_t = holm_adjust(
                t_values
            )

            adjusted_w = holm_adjust(
                w_values
            )

            for (
                index,
                t_adjusted,
                w_adjusted,
            ) in zip(
                metric_record_indices,
                adjusted_t,
                adjusted_w,
            ):
                records[
                    index
                ][
                    "paired_t_p_holm"
                ] = t_adjusted

                records[
                    index
                ][
                    "wilcoxon_p_holm"
                ] = w_adjusted

    return pd.DataFrame(
        records
    )


# ============================================================
# Dataset execution
# ============================================================

def run_dataset(
    dataset: str,
    maximum_rows: int,
    repeats: int,
    folds: int,
    estimators: int,
    weight_candidates: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset_start = time.perf_counter()

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 7 | DATASET=%s",
        dataset,
    )
    LOGGER.info("=" * 78)

    raw, total_available = load_sample(
        dataset,
        maximum_rows,
        seed,
    )

    features, target, removed = (
        select_predictors(
            raw
        )
    )

    views = build_views(
        features
    )

    if len(
        views
    ) < 2:
        raise ValueError(
            f"{dataset}: fewer than two semantic views were constructed."
        )

    LOGGER.info(
        "%s | predictors=%d | views=%d: %s",
        dataset,
        features.shape[
            1
        ],
        len(
            views
        ),
        ", ".join(
            views.keys()
        ),
    )

    source_files = (
        raw.loc[
            features.index,
            "source_file",
        ]
        if "source_file"
        in raw.columns
        else pd.Series(
            [
                "unknown"
            ]
            * len(
                features
            )
        )
    )

    integrity_all = integrity_scores(
        CLEANING_ROOT
        / dataset,
        source_files.reset_index(
            drop=True
        ),
    )

    temporal_all = temporal_scores(
        raw.loc[
            features.index
        ].reset_index(
            drop=True
        )
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )

    fold_results = []

    total_evaluations = (
        folds
        * repeats
    )

    for evaluation_id, (
        outer_train_indices,
        outer_test_indices,
    ) in enumerate(
        splitter.split(
            features,
            target,
        ),
        start=1,
    ):
        repetition = (
            (
                evaluation_id
                - 1
            )
            // folds
            + 1
        )

        fold = (
            (
                evaluation_id
                - 1
            )
            % folds
            + 1
        )

        evaluation_start = (
            time.perf_counter()
        )

        results = run_fold(
            dataset=dataset,
            features=features,
            target=target,
            raw=raw,
            views=views,
            integrity_all=(
                integrity_all
            ),
            temporal_all=(
                temporal_all
            ),
            outer_train_indices=(
                outer_train_indices
            ),
            outer_test_indices=(
                outer_test_indices
            ),
            repetition=(
                repetition
            ),
            fold=fold,
            evaluation_id=(
                evaluation_id
            ),
            estimators=(
                estimators
            ),
            weight_candidates_count=(
                weight_candidates
            ),
            seed=seed,
        )

        fold_results.extend(
            results
        )

        elapsed = (
            time.perf_counter()
            - dataset_start
        )

        average = (
            elapsed
            / evaluation_id
        )

        remaining = (
            total_evaluations
            - evaluation_id
        ) * average

        LOGGER.info(
            "%s | progress=%d/%d | %.1f%% | "
            "fold elapsed=%s | total elapsed=%s | ETA=%s",
            dataset,
            evaluation_id,
            total_evaluations,
            100.0
            * evaluation_id
            / total_evaluations,
            format_seconds(
                time.perf_counter()
                - evaluation_start
            ),
            format_seconds(
                elapsed
            ),
            format_seconds(
                remaining
            ),
        )

        gc.collect()

    fold_frame = pd.DataFrame(
        [
            asdict(
                result
            )
            for result in fold_results
        ]
    )

    statistics = statistical_comparisons(
        fold_frame,
        bootstrap_iterations,
        seed,
    )

    dataset_root = (
        RESULTS_ROOT
        / dataset
    )

    metrics_dir = (
        dataset_root
        / "Metrics"
    )

    manifests_dir = (
        dataset_root
        / "Manifests"
    )

    for directory in (
        metrics_dir,
        manifests_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    fold_frame.to_csv(
        metrics_dir
        / "repeated_cv_scores.csv",
        index=False,
    )

    statistics.to_csv(
        metrics_dir
        / "paired_statistical_tests.csv",
        index=False,
    )

    descriptive = (
        fold_frame
        .groupby(
            [
                "dataset",
                "configuration",
            ],
            as_index=False,
        )[
            list(
                METRIC_COLUMNS
            )
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
            ]
        )
    )

    descriptive.columns = [
        "_".join(
            str(
                part
            )
            for part in column
            if part
        )
        if isinstance(
            column,
            tuple,
        )
        else column
        for column in descriptive.columns
    ]

    descriptive.to_csv(
        metrics_dir
        / "descriptive_statistics.csv",
        index=False,
    )

    save_json(
        manifests_dir
        / "experiment_07_manifest.json",
        {
            "generated_utc": utc_now(),
            "dataset": dataset,
            "total_available_rows": (
                total_available
            ),
            "rows_used": len(
                features
            ),
            "removed_columns": (
                removed
            ),
            "semantic_views": (
                views
            ),
            "outer_cv": {
                "type": (
                    "RepeatedStratifiedKFold"
                ),
                "repetitions": (
                    repeats
                ),
                "folds": folds,
                "paired_evaluations": (
                    repeats
                    * folds
                ),
            },
            "inner_development_split": {
                "outer_training_fraction": (
                    0.80
                ),
                "inner_training_fraction_of_outer_train": (
                    0.875
                ),
                "inner_validation_fraction_of_outer_train": (
                    0.125
                ),
                "equivalent_total_fraction": (
                    "70% train / 10% validation / 20% outer test"
                ),
            },
            "configurations": list(
                CONFIGURATION_ORDER
            ),
            "metrics": {
                "macro_f1": (
                    "binary labels, macro average"
                ),
                "weighted_f1": (
                    "binary labels, support-weighted average"
                ),
                "balanced_accuracy": (
                    "mean class recall"
                ),
                "roc_auc": (
                    "ROC-AUC from attack probabilities"
                ),
                "pr_auc": (
                    "average precision / PR-AUC"
                ),
                "cohen_kappa": (
                    "Cohen's kappa on thresholded predictions"
                ),
                "mae": (
                    "mean absolute error between binary target "
                    "and attack probability"
                ),
                "runtime_seconds": (
                    "per-fold configuration runtime; "
                    "multi-view configurations include shared "
                    "view-model training plus their own fusion selection"
                ),
            },
            "statistical_tests": {
                "paired_t_test": (
                    "scipy.stats.ttest_rel"
                ),
                "wilcoxon": (
                    "scipy.stats.wilcoxon"
                ),
                "bootstrap_ci": (
                    f"{bootstrap_iterations} paired bootstrap iterations"
                ),
                "effect_size": (
                    "Cohen's dz = mean paired difference / "
                    "SD of paired differences"
                ),
                "additional_multiple_testing_control": (
                    "Holm-adjusted p-values across metrics "
                    "within each pairwise comparison"
                ),
            },
            "reliability_coefficients": {
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
            "xgboost_estimators": (
                estimators
            ),
            "random_global_weight_candidates_per_fold": (
                weight_candidates
            ),
            "threshold_grid": {
                "minimum": float(
                    THRESHOLDS.min()
                ),
                "maximum": float(
                    THRESHOLDS.max()
                ),
                "step": 0.01,
            },
            "random_seed": seed,
            "outer_test_used_for_selection": (
                False
            ),
            "models_retrained_each_fold": (
                True
            ),
            "dataset_elapsed_seconds": (
                time.perf_counter()
                - dataset_start
            ),
        },
    )

    LOGGER.info(
        "%s completed | evaluations=%d | elapsed=%s",
        dataset,
        total_evaluations,
        format_seconds(
            time.perf_counter()
            - dataset_start
        ),
    )

    return (
        fold_frame,
        statistics,
    )


# ============================================================
# Consolidated reporting
# ============================================================

def save_consolidated(
    score_frames: list[
        pd.DataFrame
    ],
    statistical_frames: list[
        pd.DataFrame
    ],
) -> None:
    if score_frames:
        scores = pd.concat(
            score_frames,
            ignore_index=True,
        )

        scores.to_csv(
            REPORTS_DIR
            / "Repeated_CV_Scores.csv",
            index=False,
        )

        summary_records = []

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
                "configuration": (
                    configuration
                ),
                "paired_evaluations": len(
                    group
                ),
            }

            for metric in METRIC_COLUMNS:
                values = group[
                    metric
                ].to_numpy(
                    dtype=float
                )

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

                record[
                    f"{metric}_median"
                ] = float(
                    np.median(
                        values
                    )
                )

            summary_records.append(
                record
            )

        pd.DataFrame(
            summary_records
        ).to_csv(
            REPORTS_DIR
            / "Statistical_Performance_Summary.csv",
            index=False,
        )

    if statistical_frames:
        statistical = pd.concat(
            statistical_frames,
            ignore_index=True,
        )

        statistical.to_csv(
            REPORTS_DIR
            / "Paired_Statistical_Tests.csv",
            index=False,
        )

        ci_columns = [
            "dataset",
            "configuration_a",
            "configuration_b",
            "metric",
            "mean_difference_a_minus_b",
            "bootstrap_ci_95_lower",
            "bootstrap_ci_95_upper",
        ]

        statistical[
            ci_columns
        ].to_csv(
            REPORTS_DIR
            / "Bootstrap_Confidence_Intervals.csv",
            index=False,
        )

        effect_columns = [
            "dataset",
            "configuration_a",
            "configuration_b",
            "metric",
            "cohens_dz",
        ]

        statistical[
            effect_columns
        ].to_csv(
            REPORTS_DIR
            / "Effect_Sizes.csv",
            index=False,
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 7: repeated paired statistical validation."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(
            *DATASET_NAMES,
            "all",
        ),
        default="all",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=(
            DEFAULT_MAX_ROWS
        ),
        help=(
            "Deterministic sample size per dataset. "
            f"Default: {DEFAULT_MAX_ROWS:,}. "
            "Use 0 for all available cleaned records."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=(
            DEFAULT_REPEATS
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=(
            DEFAULT_FOLDS
        ),
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=(
            DEFAULT_ESTIMATORS
        ),
    )

    parser.add_argument(
        "--weight-candidates",
        type=int,
        default=(
            DEFAULT_WEIGHT_CANDIDATES
        ),
        help=(
            "Random Dirichlet global-weight candidates per fold, "
            "in addition to equal and one-hot controls."
        ),
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=(
            DEFAULT_BOOTSTRAP_ITERATIONS
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            RANDOM_SEED
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    if args.max_rows < 0:
        raise ValueError(
            "--max-rows must be zero or positive."
        )

    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be positive."
        )

    if args.folds < 2:
        raise ValueError(
            "--folds must be at least 2."
        )

    if args.estimators <= 0:
        raise ValueError(
            "--estimators must be positive."
        )

    if args.weight_candidates <= 0:
        raise ValueError(
            "--weight-candidates must be positive."
        )

    if args.bootstrap_iterations < 1000:
        raise ValueError(
            "--bootstrap-iterations should be at least 1000."
        )

    selected = (
        list(
            DATASET_NAMES
        )
        if args.dataset
        == "all"
        else [
            args.dataset
        ]
    )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 7: STATISTICAL ANALYSIS"
    )
    LOGGER.info(
        "Datasets: %s",
        ", ".join(
            selected
        ),
    )
    LOGGER.info(
        "Repeated stratified CV: %d repetitions x %d folds = %d paired evaluations",
        args.repeats,
        args.folds,
        (
            args.repeats
            * args.folds
        ),
    )
    LOGGER.info(
        "Deterministic sample per dataset: %s",
        (
            "ALL"
            if args.max_rows
            == 0
            else f"{args.max_rows:,}"
        ),
    )
    LOGGER.info(
        "Each outer fold preserves 70/10/20 development/evaluation proportions."
    )
    LOGGER.info(
        "Models are genuinely retrained within every fold."
    )
    LOGGER.info("=" * 78)

    run_start = time.perf_counter()

    score_frames = []
    statistical_frames = []
    failures = []

    for dataset in selected:
        try:
            (
                scores,
                statistics,
            ) = run_dataset(
                dataset=dataset,
                maximum_rows=(
                    args.max_rows
                ),
                repeats=(
                    args.repeats
                ),
                folds=(
                    args.folds
                ),
                estimators=(
                    args.estimators
                ),
                weight_candidates=(
                    args.weight_candidates
                ),
                bootstrap_iterations=(
                    args.bootstrap_iterations
                ),
                seed=(
                    args.seed
                ),
            )

            score_frames.append(
                scores
            )

            statistical_frames.append(
                statistics
            )

        except Exception:
            failures.append(
                dataset
            )

            LOGGER.exception(
                "Experiment 7 failed for dataset=%s",
                dataset,
            )

    save_consolidated(
        score_frames,
        statistical_frames,
    )

    save_json(
        MANIFESTS_DIR
        / "experiment_07_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": (
                Path(
                    __file__
                ).name
            ),
            "experiment": (
                "Experiment 7: Statistical Analysis"
            ),
            "selected_datasets": (
                selected
            ),
            "successful_datasets": [
                str(
                    frame[
                        "dataset"
                    ].iloc[
                        0
                    ]
                )
                for frame in score_frames
                if not frame.empty
            ],
            "failed_datasets": (
                failures
            ),
            "repetitions": (
                args.repeats
            ),
            "folds": (
                args.folds
            ),
            "paired_evaluations_per_dataset": (
                args.repeats
                * args.folds
            ),
            "maximum_rows_per_dataset": (
                args.max_rows
            ),
            "estimators": (
                args.estimators
            ),
            "weight_candidates": (
                args.weight_candidates
            ),
            "bootstrap_iterations": (
                args.bootstrap_iterations
            ),
            "seed": (
                args.seed
            ),
            "elapsed_seconds": (
                time.perf_counter()
                - run_start
            ),
        },
    )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Successful datasets: %d",
        len(
            score_frames
        ),
    )
    LOGGER.info(
        "Failed datasets: %d",
        len(
            failures
        ),
    )
    LOGGER.info(
        "Total runtime: %s",
        format_seconds(
            time.perf_counter()
            - run_start
        ),
    )
    LOGGER.info(
        "Results directory: %s",
        RESULTS_ROOT,
    )
    LOGGER.info("=" * 78)

    if not score_frames:
        return 1

    if failures:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
