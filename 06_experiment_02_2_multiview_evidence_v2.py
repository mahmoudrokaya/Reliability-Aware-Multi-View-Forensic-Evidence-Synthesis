
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ferf_common import (
    QualityEstimator,
    build_views,
    calculate_metrics,
    create_preprocessor,
    create_xgboost,
    integrity_scores,
    save_json,
    select_predictors,
    temporal_scores,
)


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

EXPERIMENT_21_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Multiview_Evidence"
    / "Phase_02_View_Models"
)


def configure_logging(
    dataset: str,
) -> logging.Logger:
    dataset_root = RESULTS_ROOT / dataset
    logs_dir = dataset_root / "Logs"
    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        f"multiview_{dataset}"
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
        logs_dir / "multiview_execution.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def discover_parts(
    dataset: str,
) -> list[Path]:
    cleaned_dir = (
        CLEANING_ROOT
        / dataset
        / "Cleaned_Data"
    )

    parquet = sorted(
        cleaned_dir.glob(
            "cleaned_part_*.parquet"
        )
    )

    if parquet:
        return parquet

    return sorted(
        cleaned_dir.glob(
            "cleaned_part_*.csv.gz"
        )
    )


def read_part(
    path: Path,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )


def count_rows(
    path: Path,
) -> int:
    if path.suffix.lower() == ".parquet":
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
        total += len(chunk)

    return total


def allocate_rows(
    sizes: list[int],
    target: int,
) -> list[int]:
    total = sum(sizes)

    if target <= 0 or target >= total:
        return sizes.copy()

    exact = [
        target * size / total
        for size in sizes
    ]

    allocation = [
        min(
            size,
            int(np.floor(value)),
        )
        for size, value in zip(
            sizes,
            exact,
        )
    ]

    remainder = target - sum(allocation)

    order = np.argsort(
        [
            value - np.floor(value)
            for value in exact
        ]
    )[::-1]

    for index in order:
        if remainder <= 0:
            break

        if allocation[index] < sizes[index]:
            allocation[index] += 1
            remainder -= 1

    return allocation


def load_sample(
    dataset: str,
    maximum_rows: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, int]:
    parts = discover_parts(dataset)

    if not parts:
        raise FileNotFoundError(
            f"No cleaned parts found for {dataset}."
        )

    logger.info(
        "Found %d cleaned parts.",
        len(parts),
    )

    logger.info(
        "Counting records in cleaned parts..."
    )

    sizes = []

    for index, path in enumerate(
        parts,
        start=1,
    ):
        start = time.perf_counter()
        size = count_rows(path)
        sizes.append(size)

        logger.info(
            "Counted part %d/%d: %s | rows=%d | %.2f s",
            index,
            len(parts),
            path.name,
            size,
            time.perf_counter() - start,
        )

    total_available = sum(sizes)

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

    logger.info(
        "Available rows=%d | requested rows=%d",
        total_available,
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

        start = time.perf_counter()

        frame = read_part(path)

        if selected < part_size:
            frame = frame.sample(
                n=selected,
                random_state=seed + index,
            )

        frames.append(frame)

        logger.info(
            "Loaded part %d/%d: %s | selected=%d/%d | %.2f s",
            index,
            len(parts),
            path.name,
            len(frame),
            part_size,
            time.perf_counter() - start,
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if len(dataframe) > requested:
        dataframe = dataframe.sample(
            n=requested,
            random_state=seed,
        ).reset_index(drop=True)

    logger.info(
        "Sample construction completed: rows=%d",
        len(dataframe),
    )

    return dataframe, total_available


def load_split_positions(
    dataset: str,
    expected_rows: int,
) -> dict[str, np.ndarray]:
    split_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Splits"
        / "fixed_split_assignments.csv"
    )

    assignments = pd.read_csv(
        split_path
    ).sort_values(
        "row_position"
    )

    if len(assignments) != expected_rows:
        raise ValueError(
            "Sample and saved split assignments do not match: "
            f"sample={expected_rows}, "
            f"assignments={len(assignments)}."
        )

    return {
        split_name: assignments.loc[
            assignments["split"].eq(
                split_name
            ),
            "row_position",
        ].to_numpy(dtype=int)
        for split_name in (
            "train",
            "validation",
            "test",
        )
    }


def run(
    dataset: str,
    maximum_rows: int,
    seed: int,
    estimators: int,
) -> None:
    logger = configure_logging(dataset)
    run_start = time.perf_counter()

    output_root = RESULTS_ROOT / dataset

    for directory_name in (
        "Models",
        "Predictions",
        "Reports",
        "Manifests",
        "Logs",
    ):
        (
            output_root / directory_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info("=" * 78)
    logger.info(
        "EXPERIMENT 2.2: MULTI-VIEW EVIDENCE MODELING"
    )
    logger.info(
        "Dataset=%s | max_rows=%s | seed=%d | estimators=%d",
        dataset,
        (
            "ALL"
            if maximum_rows == 0
            else f"{maximum_rows:,}"
        ),
        seed,
        estimators,
    )
    logger.info("=" * 78)

    raw, total_available = load_sample(
        dataset,
        maximum_rows,
        seed,
        logger,
    )

    logger.info(
        "Selecting valid predictive features..."
    )

    features, target, removed = (
        select_predictors(raw)
    )

    logger.info(
        "Retained predictors=%d | removed columns=%d",
        features.shape[1],
        len(removed),
    )

    splits = load_split_positions(
        dataset,
        len(features),
    )

    logger.info(
        "Splits: train=%d | validation=%d | test=%d",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )

    views = build_views(features)

    logger.info(
        "Constructed %d evidence views: %s",
        len(views),
        ", ".join(views.keys()),
    )

    source_files = (
        raw.loc[
            features.index,
            "source_file",
        ]
        if "source_file" in raw.columns
        else pd.Series(
            ["unknown"] * len(features)
        )
    )

    integrity = integrity_scores(
        CLEANING_ROOT / dataset,
        source_files.reset_index(
            drop=True
        ),
    )

    temporal = temporal_scores(
        raw.loc[
            features.index
        ].reset_index(
            drop=True
        )
    )

    predictions = pd.DataFrame(
        {
            "true_label": target.reset_index(
                drop=True
            ),
            "split": "",
        }
    )

    for split_name, indices in splits.items():
        predictions.loc[
            indices,
            "split",
        ] = split_name

    report_records = []
    total_views = len(views)

    for view_index, (
        view_name,
        columns,
    ) in enumerate(
        views.items(),
        start=1,
    ):
        view_start = time.perf_counter()

        logger.info("-" * 78)
        logger.info(
            "Training view %d/%d: %s | features=%d",
            view_index,
            total_views,
            view_name,
            len(columns),
        )

        train_features = features.iloc[
            splits["train"]
        ][columns]

        model = Pipeline(
            steps=[
                (
                    "preprocessing",
                    create_preprocessor(
                        train_features
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

        fit_start = time.perf_counter()

        model.fit(
            train_features,
            target.iloc[
                splits["train"]
            ],
        )

        training_seconds = (
            time.perf_counter()
            - fit_start
        )

        logger.info(
            "View %s training completed in %.2f minutes.",
            view_name,
            training_seconds / 60.0,
        )

        logger.info(
            "Computing quality and predictions for %s...",
            view_name,
        )

        quality_estimator = (
            QualityEstimator()
            .fit(train_features)
        )

        quality = quality_estimator.transform(
            features[columns]
        )

        probability = np.full(
            len(features),
            np.nan,
            dtype=float,
        )

        for split_name in (
            "validation",
            "test",
        ):
            inference_start = (
                time.perf_counter()
            )

            indices = splits[split_name]

            probability[indices] = (
                model.predict_proba(
                    features.iloc[
                        indices
                    ][columns]
                )[:, 1]
            )

            logger.info(
                "%s inference for %s: rows=%d | %.2f s",
                split_name,
                view_name,
                len(indices),
                (
                    time.perf_counter()
                    - inference_start
                ),
            )

        predictions[
            f"{view_name}__probability"
        ] = probability

        predictions[
            f"{view_name}__integrity"
        ] = integrity

        predictions[
            f"{view_name}__quality"
        ] = quality

        predictions[
            f"{view_name}__temporal"
        ] = (
            temporal
            if view_name == "temporal"
            else np.ones(len(features))
        )

        validation_metrics = (
            calculate_metrics(
                target.iloc[
                    splits["validation"]
                ],
                probability[
                    splits["validation"]
                ],
            )
        )

        test_metrics = calculate_metrics(
            target.iloc[
                splits["test"]
            ],
            probability[
                splits["test"]
            ],
        )

        report_records.append(
            {
                "view": view_name,
                "features": len(columns),
                "training_seconds": (
                    training_seconds
                ),
                "total_view_seconds": (
                    time.perf_counter()
                    - view_start
                ),
                **{
                    f"validation_{key}": value
                    for key, value
                    in validation_metrics.items()
                },
                **{
                    f"test_{key}": value
                    for key, value
                    in test_metrics.items()
                },
            }
        )

        with (
            output_root
            / "Models"
            / f"{view_name}_pipeline.pkl"
        ).open("wb") as stream:
            pickle.dump(
                model,
                stream,
            )

        logger.info(
            "Completed view %s | "
            "validation BA=%.6f | test BA=%.6f",
            view_name,
            validation_metrics[
                "balanced_accuracy"
            ],
            test_metrics[
                "balanced_accuracy"
            ],
        )

    logger.info(
        "Saving consolidated multi-view outputs..."
    )

    predictions.to_csv(
        output_root
        / "Predictions"
        / "view_predictions_and_reliability.csv",
        index=False,
    )

    pd.DataFrame(
        report_records
    ).to_csv(
        output_root
        / "Reports"
        / "view_metrics.csv",
        index=False,
    )

    save_json(
        output_root
        / "Manifests"
        / "design.json",
        {
            "dataset": dataset,
            "rows_used": len(features),
            "total_available": (
                total_available
            ),
            "views": views,
            "removed_columns": removed,
            "scope": (
                "multi-view network-flow evidence"
            ),
            "xgboost_estimators": estimators,
            "seed": seed,
        },
    )

    logger.info("=" * 78)
    logger.info(
        "Experiment 2.2 completed successfully."
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
        "--max-rows",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=250,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run(
        dataset=arguments.dataset,
        maximum_rows=arguments.max_rows,
        seed=arguments.seed,
        estimators=arguments.estimators,
    )
