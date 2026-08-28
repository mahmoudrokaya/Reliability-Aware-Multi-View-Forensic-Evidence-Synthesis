
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from ferf_common import (
    calculate_metrics,
    save_json,
)


PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

SOURCE_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Multiview_Evidence"
    / "Phase_02_View_Models"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_03_FERF_Validation_Optimized"
)

INTEGRITY_WEIGHT = 0.35
QUALITY_WEIGHT = 0.45
TEMPORAL_WEIGHT = 0.20


def configure_logging(
    dataset: str,
) -> logging.Logger:
    output_root = RESULTS_ROOT / dataset
    logs_dir = output_root / "Logs"

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        f"ferf_{dataset}"
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler(
        sys.stdout
    )
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        logs_dir / "ferf_execution.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def generate_random_weights(
    number_of_views: int,
    candidates: int,
    seed: int,
) -> np.ndarray:
    """
    Generate reproducible simplex weights.

    The candidate set includes:
    - equal weighting;
    - one-hot single-view solutions;
    - random Dirichlet solutions.
    """

    rng = np.random.default_rng(seed)

    generated = rng.dirichlet(
        np.ones(number_of_views),
        size=max(candidates, 1),
    )

    equal = np.full(
        (1, number_of_views),
        1.0 / number_of_views,
    )

    one_hot = np.eye(number_of_views)

    return np.vstack(
        [
            equal,
            one_hot,
            generated,
        ]
    )


def fuse_probabilities(
    probabilities: np.ndarray,
    reliabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_reliability = (
        reliabilities
        * weights.reshape(1, -1)
    )

    return (
        (
            weighted_reliability
            * probabilities
        ).sum(axis=1)
        / np.maximum(
            weighted_reliability.sum(axis=1),
            1e-12,
        )
    )


def optimize_threshold_vectorized(
    target: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[
    tuple[float, float, float, float],
    float,
]:
    """
    Evaluate all thresholds without repeatedly calling sklearn.

    Returns:
    (balanced accuracy, F1, MCC, ROC-AUC), best threshold.
    """

    target = target.astype(np.int8)

    predictions = (
        probability[:, None]
        >= thresholds[None, :]
    )

    positive = target == 1
    negative = ~positive

    tp = predictions[
        positive
    ].sum(axis=0)

    fn = positive.sum() - tp

    fp = predictions[
        negative
    ].sum(axis=0)

    tn = negative.sum() - fp

    sensitivity = tp / np.maximum(
        tp + fn,
        1,
    )

    specificity = tn / np.maximum(
        tn + fp,
        1,
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    f1 = (
        2.0 * tp
        / np.maximum(
            2.0 * tp + fp + fn,
            1,
        )
    )

    mcc_denominator = np.sqrt(
        np.maximum(
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn),
            1,
        )
    )

    mcc = (
        tp * tn - fp * fn
    ) / mcc_denominator

    auc = roc_auc_score(
        target,
        probability,
    )

    best_index = max(
        range(len(thresholds)),
        key=lambda index: (
            balanced_accuracy[index],
            f1[index],
            mcc[index],
            auc,
            -abs(
                thresholds[index] - 0.5
            ),
        ),
    )

    return (
        (
            float(
                balanced_accuracy[
                    best_index
                ]
            ),
            float(f1[best_index]),
            float(mcc[best_index]),
            float(auc),
        ),
        float(thresholds[best_index]),
    )


def run(
    dataset: str,
    candidate_weights: int,
    seed: int,
    progress_every: int,
) -> None:
    logger = configure_logging(dataset)
    run_start = time.perf_counter()

    output_root = RESULTS_ROOT / dataset

    for directory_name in (
        "Reports",
        "Predictions",
        "Manifests",
        "Logs",
    ):
        (
            output_root / directory_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    source_path = (
        SOURCE_ROOT
        / dataset
        / "Predictions"
        / "view_predictions_and_reliability.csv"
    )

    logger.info("=" * 78)
    logger.info(
        "EXPERIMENT 3: OPTIMIZED FERF VALIDATION"
    )
    logger.info(
        "Dataset=%s | random candidates=%d | seed=%d",
        dataset,
        candidate_weights,
        seed,
    )
    logger.info("=" * 78)

    logger.info(
        "Loading multi-view predictions: %s",
        source_path,
    )

    dataframe = pd.read_csv(
        source_path
    )

    logger.info(
        "Loaded %d records and %d columns.",
        len(dataframe),
        len(dataframe.columns),
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
            "At least two evidence views are required."
        )

    logger.info(
        "Detected %d views: %s",
        len(views),
        ", ".join(views),
    )

    validation_mask = dataframe[
        "split"
    ].eq(
        "validation"
    ).to_numpy()

    test_mask = dataframe[
        "split"
    ].eq(
        "test"
    ).to_numpy()

    target = dataframe[
        "true_label"
    ].to_numpy(
        dtype=np.int8
    )

    probability_matrix = np.column_stack(
        [
            dataframe[
                f"{view}__probability"
            ].to_numpy(dtype=float)
            for view in views
        ]
    )

    reliability_matrix = np.column_stack(
        [
            np.clip(
                INTEGRITY_WEIGHT
                * dataframe[
                    f"{view}__integrity"
                ].to_numpy(dtype=float)
                + QUALITY_WEIGHT
                * dataframe[
                    f"{view}__quality"
                ].to_numpy(dtype=float)
                + TEMPORAL_WEIGHT
                * dataframe[
                    f"{view}__temporal"
                ].to_numpy(dtype=float),
                0.05,
                1.0,
            )
            for view in views
        ]
    )

    thresholds = np.arange(
        0.20,
        0.801,
        0.01,
    )

    weight_candidates = (
        generate_random_weights(
            number_of_views=len(views),
            candidates=candidate_weights,
            seed=seed,
        )
    )

    total_candidates = len(
        weight_candidates
    )

    logger.info(
        "Searching %d weight candidates and %d thresholds.",
        total_candidates,
        len(thresholds),
    )

    validation_target = target[
        validation_mask
    ]

    validation_probability_matrix = (
        probability_matrix[
            validation_mask
        ]
    )

    validation_reliability_matrix = (
        reliability_matrix[
            validation_mask
        ]
    )

    search_records = []
    best_key = None
    best_weights = None
    best_threshold = 0.5

    search_start = time.perf_counter()

    for index, weights in enumerate(
        weight_candidates,
        start=1,
    ):
        fused_probability = (
            fuse_probabilities(
                validation_probability_matrix,
                validation_reliability_matrix,
                weights,
            )
        )

        metric_tuple, threshold = (
            optimize_threshold_vectorized(
                validation_target,
                fused_probability,
                thresholds,
            )
        )

        balanced_accuracy, f1, mcc, auc = (
            metric_tuple
        )

        current_key = (
            balanced_accuracy,
            f1,
            mcc,
            auc,
            -abs(threshold - 0.5),
        )

        search_records.append(
            {
                "candidate": index,
                "threshold": threshold,
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "f1": f1,
                "mcc": mcc,
                "roc_auc": auc,
                **{
                    f"weight_{view}": (
                        float(weights[
                            view_index
                        ])
                    )
                    for view_index, view
                    in enumerate(views)
                },
            }
        )

        if (
            best_key is None
            or current_key > best_key
        ):
            best_key = current_key
            best_weights = weights.copy()
            best_threshold = threshold

        if (
            index == 1
            or index % progress_every == 0
            or index == total_candidates
        ):
            elapsed = (
                time.perf_counter()
                - search_start
            )

            rate = index / max(
                elapsed,
                1e-9,
            )

            remaining = (
                total_candidates - index
            ) / max(
                rate,
                1e-9,
            )

            logger.info(
                "FERF search %d/%d | %.1f%% | "
                "elapsed=%.2f min | ETA=%.2f min | "
                "best BA=%.6f",
                index,
                total_candidates,
                100.0 * index
                / total_candidates,
                elapsed / 60.0,
                remaining / 60.0,
                best_key[0],
            )

    assert best_weights is not None

    logger.info(
        "Optimization completed in %.2f minutes.",
        (
            time.perf_counter()
            - search_start
        ) / 60.0,
    )

    logger.info(
        "Best threshold=%.3f | weights=%s",
        best_threshold,
        {
            view: round(
                float(best_weights[index]),
                6,
            )
            for index, view
            in enumerate(views)
        },
    )

    logger.info(
        "Running held-out test evaluation..."
    )

    test_probability = fuse_probabilities(
        probability_matrix[test_mask],
        reliability_matrix[test_mask],
        best_weights,
    )

    test_target = target[test_mask]

    ferf_metrics = calculate_metrics(
        test_target,
        test_probability,
        best_threshold,
    )

    unweighted_probability = np.nanmean(
        probability_matrix[test_mask],
        axis=1,
    )

    unweighted_metrics = calculate_metrics(
        test_target,
        unweighted_probability,
        0.5,
    )

    single_view_records = []

    for view_index, view in enumerate(
        views
    ):
        view_metrics = calculate_metrics(
            test_target,
            probability_matrix[
                test_mask,
                view_index,
            ],
            0.5,
        )

        single_view_records.append(
            {
                "method": (
                    "Single view"
                ),
                "view": view,
                **view_metrics,
            }
        )

    best_single = max(
        single_view_records,
        key=lambda record: (
            record[
                "balanced_accuracy"
            ],
            record["f1"],
            record["mcc"],
        ),
    )

    comparison = pd.DataFrame(
        [
            best_single,
            {
                "method": (
                    "Unweighted mean"
                ),
                "view": "all",
                **unweighted_metrics,
            },
            {
                "method": "FERF",
                "view": "all",
                **ferf_metrics,
            },
        ]
    )

    logger.info(
        "FERF results | Accuracy=%.6f | "
        "Balanced Accuracy=%.6f | "
        "F1=%.6f | MCC=%.6f | ROC-AUC=%.6f",
        ferf_metrics["accuracy"],
        ferf_metrics[
            "balanced_accuracy"
        ],
        ferf_metrics["f1"],
        ferf_metrics["mcc"],
        ferf_metrics["roc_auc"],
    )

    logger.info(
        "Saving reports and predictions..."
    )

    pd.DataFrame(
        search_records
    ).sort_values(
        by=[
            "balanced_accuracy",
            "f1",
            "mcc",
            "roc_auc",
        ],
        ascending=False,
    ).to_csv(
        output_root
        / "Reports"
        / "validation_search.csv",
        index=False,
    )

    comparison.to_csv(
        output_root
        / "Reports"
        / "ferf_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "true_label": test_target,
            "ferf_probability": (
                test_probability
            ),
            "ferf_prediction": (
                test_probability
                >= best_threshold
            ).astype(np.int8),
            "unweighted_probability": (
                unweighted_probability
            ),
        }
    ).to_csv(
        output_root
        / "Predictions"
        / "ferf_test_predictions.csv",
        index=False,
    )

    save_json(
        output_root
        / "Manifests"
        / "ferf_configuration.json",
        {
            "dataset": dataset,
            "views": views,
            "weights": {
                view: float(
                    best_weights[index]
                )
                for index, view
                in enumerate(views)
            },
            "threshold": (
                best_threshold
            ),
            "reliability": {
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
            "weight_search": {
                "method": (
                    "reproducible Dirichlet random search"
                ),
                "random_candidates": (
                    candidate_weights
                ),
                "total_candidates_including_controls": (
                    total_candidates
                ),
                "seed": seed,
            },
            "threshold_search": {
                "minimum": 0.20,
                "maximum": 0.80,
                "step": 0.01,
                "implementation": (
                    "vectorized"
                ),
            },
            "selection": (
                "validation subset only"
            ),
            "scope": (
                "multi-view network-flow evidence"
            ),
        },
    )

    logger.info("=" * 78)
    logger.info(
        "Optimized FERF experiment completed successfully."
    )
    logger.info(
        "Total elapsed time: %.2f minutes.",
        (
            time.perf_counter()
            - run_start
        ) / 60.0,
    )
    logger.info(
        "Output directory: %s",
        output_root,
    )
    logger.info("=" * 78)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=(
            "CICIDS2017",
            "CSE-CIC-IDS2018",
        ),
        required=True,
    )

    parser.add_argument(
        "--weight-candidates",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run(
        dataset=arguments.dataset,
        candidate_weights=(
            arguments.weight_candidates
        ),
        seed=arguments.seed,
        progress_every=(
            arguments.progress_every
        ),
    )
