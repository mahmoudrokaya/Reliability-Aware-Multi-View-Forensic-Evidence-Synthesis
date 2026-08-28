"""
05_experiment_02_1_original_features.py

Experiment 2.1
Original Feature-Space Evaluation

Purpose
-------
Evaluate the native cleaned network-flow features using one fixed
computational reasoning model, XGBoost.

This experiment establishes the reference result against which the
proposed forensic evidence representation will be compared.

Protocol
--------
- Binary intrusion classification.
- Stratified 70% training, 10% validation, and 20% testing split.
- Random seed = 42.
- Leakage-safe preprocessing fitted only on the training subset.
- Original features only.
- No FERF.
- No evidence reliability estimation.
- No forensic evidence weighting.
- No proposed feature representation.

Outputs
-------
- Fixed split assignments.
- Dataset and feature manifests.
- Trained preprocessing and XGBoost pipeline.
- Validation and testing predictions.
- Accuracy, balanced accuracy, precision, recall, F1, MCC, and ROC-AUC.
- Confusion matrix.
- Classification report.
- ROC and precision-recall curves.
- Feature importance.
- Timing and throughput measurements.
- Independent result directory for every dataset.

Important
---------
The script reads the cleaned outputs created by:
03_phase2_step2_clean_datasets.py

The current cleaned UNSW-NB15 and BoT-IoT outputs are automatically
rejected when their cleaning status or retention ratio is unacceptable.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Project paths
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

CLEANING_SUMMARY_PATH = (
    CLEANING_ROOT
    / "Reports"
    / "Dataset_Cleaning_Summary.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)

REPORTS_DIR = RESULTS_ROOT / "Reports"
LOGS_DIR = RESULTS_ROOT / "Logs"
MANIFESTS_DIR = RESULTS_ROOT / "Manifests"

for directory in (
    RESULTS_ROOT,
    REPORTS_DIR,
    LOGS_DIR,
    MANIFESTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experimental configuration
# ============================================================

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
)

TARGET_COLUMN = "binary_label"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.20

DEFAULT_MAX_ROWS = 1_000_000
MINIMUM_RETENTION_RATIO = 0.50

NON_PREDICTIVE_COLUMNS = {
    "binary_label",
    "multiclass_label",
    "original_label",
    "source_dataset",
    "source_file",
    "source_row",
}

IDENTIFIER_PATTERNS = (
    r"(^|_)flow_id($|_)",
    r"(^|_)src_ip($|_)",
    r"(^|_)dst_ip($|_)",
    r"(^|_)source_ip($|_)",
    r"(^|_)destination_ip($|_)",
    r"(^|_)timestamp($|_)",
    r"(^|_)time_stamp($|_)",
    r"(^|_)record_id($|_)",
)


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "experiment_02_1_original_features.log"

LOGGER = logging.getLogger("experiment_02_1_original_features")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


# ============================================================
# Result structures
# ============================================================

@dataclass
class ExperimentResult:
    dataset: str
    status: str

    total_available_cleaned_rows: int
    total_rows_used: int
    training_rows: int
    validation_rows: int
    testing_rows: int

    original_column_count: int
    predictive_feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int

    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_f1: float
    validation_roc_auc: float

    test_accuracy: float
    test_balanced_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_mcc: float
    test_roc_auc: float
    test_average_precision: float

    training_seconds: float
    validation_inference_seconds: float
    test_inference_seconds: float
    test_latency_ms_per_sample: float
    test_throughput_samples_per_second: float

    result_directory: str
    remarks: str


EXPERIMENT_RESULTS: list[ExperimentResult] = []


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def normalize_column_name(column: object) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def is_identifier_column(column: str) -> bool:
    normalized = normalize_column_name(column)

    return any(
        re.search(pattern, normalized)
        for pattern in IDENTIFIER_PATTERNS
    )


def discover_cleaned_parts(dataset: str) -> list[Path]:
    cleaned_directory = (
        CLEANING_ROOT
        / dataset
        / "Cleaned_Data"
    )

    if not cleaned_directory.exists():
        return []

    parquet_parts = sorted(
        cleaned_directory.glob("cleaned_part_*.parquet")
    )

    if parquet_parts:
        return parquet_parts

    return sorted(
        cleaned_directory.glob("cleaned_part_*.csv.gz")
    )


def read_cleaned_part(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )


def count_part_rows(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(
            pq.ParquetFile(path).metadata.num_rows
        )

    row_count = 0

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=[0],
        chunksize=250_000,
    ):
        row_count += len(chunk)

    return row_count


def proportional_allocation(
    part_sizes: list[int],
    requested_rows: int,
) -> list[int]:
    total_rows = sum(part_sizes)

    if requested_rows <= 0 or requested_rows >= total_rows:
        return part_sizes.copy()

    exact_allocations = [
        requested_rows * part_size / total_rows
        for part_size in part_sizes
    ]

    allocations = [
        min(
            part_size,
            int(np.floor(exact)),
        )
        for part_size, exact in zip(
            part_sizes,
            exact_allocations,
        )
    ]

    remainder = requested_rows - sum(allocations)

    fractional_order = np.argsort(
        [
            exact - np.floor(exact)
            for exact in exact_allocations
        ]
    )[::-1]

    for part_index in fractional_order:
        if remainder <= 0:
            break

        if allocations[part_index] < part_sizes[part_index]:
            allocations[part_index] += 1
            remainder -= 1

    return allocations


# ============================================================
# Dataset readiness
# ============================================================

def load_cleaning_summary() -> pd.DataFrame:
    if not CLEANING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Cleaning summary does not exist: "
            f"{CLEANING_SUMMARY_PATH}"
        )

    summary = pd.read_csv(CLEANING_SUMMARY_PATH)
    summary["dataset"] = summary["dataset"].astype(str)

    return summary


def check_dataset_readiness(
    dataset: str,
    cleaning_summary: pd.DataFrame,
) -> tuple[bool, str]:
    matching_rows = cleaning_summary[
        cleaning_summary["dataset"] == dataset
    ]

    if matching_rows.empty:
        return False, "No cleaning summary was found."

    row = matching_rows.iloc[-1]

    cleaning_status = str(
        row.get("status", "")
    ).upper()

    if cleaning_status != "PASS":
        return (
            False,
            f"Cleaning status is {cleaning_status}.",
        )

    input_rows = int(row.get("input_rows", 0))
    output_rows = int(row.get("output_rows", 0))

    if input_rows <= 0 or output_rows <= 0:
        return False, "Cleaning produced no usable records."

    retention_ratio = output_rows / input_rows

    if retention_ratio < MINIMUM_RETENTION_RATIO:
        return (
            False,
            f"Retention ratio {retention_ratio:.4f} is below "
            f"{MINIMUM_RETENTION_RATIO:.2f}.",
        )

    cleaned_parts = discover_cleaned_parts(dataset)

    if not cleaned_parts:
        return False, "No cleaned data parts were found."

    return (
        True,
        f"Ready; cleaning retention ratio={retention_ratio:.4f}.",
    )


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int]:
    cleaned_parts = discover_cleaned_parts(dataset)

    if not cleaned_parts:
        raise FileNotFoundError(
            f"No cleaned parts were found for {dataset}."
        )

    LOGGER.info(
        "%s: counting rows in %d cleaned part(s).",
        dataset,
        len(cleaned_parts),
    )

    part_sizes = [
        count_part_rows(path)
        for path in cleaned_parts
    ]

    total_available = sum(part_sizes)

    requested_rows = (
        total_available
        if maximum_rows <= 0
        else min(maximum_rows, total_available)
    )

    allocations = proportional_allocation(
        part_sizes=part_sizes,
        requested_rows=requested_rows,
    )

    selected_frames: list[pd.DataFrame] = []

    for index, (
        path,
        part_size,
        rows_to_select,
    ) in enumerate(
        zip(
            cleaned_parts,
            part_sizes,
            allocations,
        ),
        start=1,
    ):
        if rows_to_select <= 0:
            continue

        frame = read_cleaned_part(path)

        if rows_to_select < part_size:
            frame = frame.sample(
                n=rows_to_select,
                random_state=random_seed + index,
            )

        selected_frames.append(frame)

        LOGGER.info(
            "%s | part=%d/%d | selected=%d/%d",
            dataset,
            index,
            len(cleaned_parts),
            len(frame),
            part_size,
        )

    if not selected_frames:
        raise RuntimeError(
            f"No records were loaded for {dataset}."
        )

    dataframe = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )

    if len(dataframe) > requested_rows:
        dataframe = dataframe.sample(
            n=requested_rows,
            random_state=random_seed,
        ).reset_index(drop=True)

    LOGGER.info(
        "%s: loaded %d of %d available records.",
        dataset,
        len(dataframe),
        total_available,
    )

    return dataframe, total_available


# ============================================================
# Original feature-space preparation
# ============================================================

def prepare_original_features(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    original_column_count = dataframe.shape[1]

    target = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    valid_target_mask = target.isin([0, 1])

    dataframe = dataframe.loc[
        valid_target_mask
    ].copy()

    target = target.loc[
        valid_target_mask
    ].astype("int8")

    excluded_columns = set(NON_PREDICTIVE_COLUMNS)

    for column in dataframe.columns:
        if is_identifier_column(column):
            excluded_columns.add(column)

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    features = dataframe[
        feature_columns
    ].copy()

    all_missing_columns = [
        column
        for column in features.columns
        if features[column].isna().all()
    ]

    if all_missing_columns:
        features = features.drop(
            columns=all_missing_columns
        )

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:
        features = features.drop(
            columns=constant_columns
        )

    removed_columns = sorted(
        excluded_columns
        | set(all_missing_columns)
        | set(constant_columns)
    )

    if features.empty:
        raise ValueError(
            "No predictive features remained."
        )

    retained_columns = list(features.columns)

    LOGGER.info(
        "Original columns=%d | retained predictive features=%d",
        original_column_count,
        len(retained_columns),
    )

    return (
        features,
        target,
        retained_columns,
        removed_columns,
    )


# ============================================================
# Fixed data splitting
# ============================================================

def create_fixed_split(
    features: pd.DataFrame,
    target: pd.Series,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    row_indices = np.arange(len(features))

    (
        train_indices,
        temporary_indices,
    ) = train_test_split(
        row_indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=random_seed,
        stratify=target,
    )

    temporary_target = target.iloc[
        temporary_indices
    ]

    relative_test_fraction = (
        TEST_FRACTION
        / (VALIDATION_FRACTION + TEST_FRACTION)
    )

    (
        validation_indices,
        test_indices,
    ) = train_test_split(
        temporary_indices,
        test_size=relative_test_fraction,
        random_state=random_seed,
        stratify=temporary_target,
    )

    X_train = features.iloc[
        train_indices
    ].copy()

    X_validation = features.iloc[
        validation_indices
    ].copy()

    X_test = features.iloc[
        test_indices
    ].copy()

    y_train = target.iloc[
        train_indices
    ].copy()

    y_validation = target.iloc[
        validation_indices
    ].copy()

    y_test = target.iloc[
        test_indices
    ].copy()

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_indices,
        validation_indices,
        test_indices,
    )


# ============================================================
# Preprocessing
# ============================================================

def create_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[
    ColumnTransformer,
    list[str],
    list[str],
]:
    numeric_columns = list(
        X_train.select_dtypes(
            include=[np.number, "bool"]
        ).columns
    )

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers: list[
        tuple[str, Pipeline, list[str]]
    ] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns,
    )


# ============================================================
# XGBoost model
# ============================================================

def create_xgboost_model(
    random_seed: int,
) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install it using: "
            "pip install xgboost"
        ) from exc

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_seed,
    )


# ============================================================
# Metrics and plots
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predicted_labels,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predicted_labels,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predicted_labels,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predicted_labels,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    predicted_labels: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        predicted_labels,
        labels=[0, 1],
    )

    dataframe = pd.DataFrame(
        matrix,
        index=[
            "true_benign",
            "true_attack",
        ],
        columns=[
            "predicted_benign",
            "predicted_attack",
        ],
    ).reset_index(
        names="true_class"
    )

    save_dataframe(
        dataframe,
        output_path,
    )


def plot_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    auc_value = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_value:.6f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Original Feature Representation: ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_precision_recall_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    average_precision = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        label=(
            "Average Precision = "
            f"{average_precision:.6f}"
        ),
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Original Feature Representation: "
        "Precision–Recall Curve"
    )
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importance(
    pipeline: Pipeline,
    output_directory: Path,
) -> None:
    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        return

    try:
        transformed_names = (
            preprocessor.get_feature_names_out()
        )

        importances = classifier.feature_importances_

        importance_frame = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": importances,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        save_dataframe(
            importance_frame,
            output_directory
            / "feature_importance.csv",
        )

        top_features = importance_frame.head(30)

        plt.figure(figsize=(10, 8))
        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1],
        )
        plt.xlabel("XGBoost Feature Importance")
        plt.ylabel("Feature")
        plt.title(
            "Top 30 Original-Representation Features"
        )
        plt.tight_layout()
        plt.savefig(
            output_directory
            / "feature_importance_top30.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as exc:
        LOGGER.warning(
            "Feature importance export failed: %s",
            exc,
        )


# ============================================================
# Experiment execution
# ============================================================

def run_original_feature_experiment(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
) -> None:
    experiment_start = time.perf_counter()

    dataset_root = RESULTS_ROOT / dataset

    model_directory = dataset_root / "Model"
    metrics_directory = dataset_root / "Metrics"
    predictions_directory = dataset_root / "Predictions"
    figures_directory = dataset_root / "Figures"
    splits_directory = dataset_root / "Splits"
    manifests_directory = dataset_root / "Manifests"

    for directory in (
        model_directory,
        metrics_directory,
        predictions_directory,
        figures_directory,
        splits_directory,
        manifests_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Experiment 2.1 dataset: %s",
        dataset,
    )

    dataframe, total_available = load_dataset(
        dataset=dataset,
        maximum_rows=maximum_rows,
        random_seed=random_seed,
    )

    original_column_count = dataframe.shape[1]

    (
        features,
        target,
        retained_columns,
        removed_columns,
    ) = prepare_original_features(
        dataframe
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_indices,
        validation_indices,
        test_indices,
    ) = create_fixed_split(
        features=features,
        target=target,
        random_seed=random_seed,
    )

    (
        preprocessor,
        numeric_columns,
        categorical_columns,
    ) = create_preprocessor(
        X_train
    )

    classifier = create_xgboost_model(
        random_seed=random_seed
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    split_frame = pd.DataFrame(
        {
            "row_position": np.concatenate(
                [
                    train_indices,
                    validation_indices,
                    test_indices,
                ]
            ),
            "split": (
                ["train"] * len(train_indices)
                + ["validation"]
                * len(validation_indices)
                + ["test"] * len(test_indices)
            ),
        }
    ).sort_values("row_position")

    save_dataframe(
        split_frame,
        splits_directory
        / "fixed_split_assignments.csv",
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(y_train),
                "benign": int(
                    (y_train == 0).sum()
                ),
                "attack": int(
                    (y_train == 1).sum()
                ),
                "attack_prevalence": float(
                    y_train.mean()
                ),
            },
            {
                "split": "validation",
                "rows": len(y_validation),
                "benign": int(
                    (y_validation == 0).sum()
                ),
                "attack": int(
                    (y_validation == 1).sum()
                ),
                "attack_prevalence": float(
                    y_validation.mean()
                ),
            },
            {
                "split": "test",
                "rows": len(y_test),
                "benign": int(
                    (y_test == 0).sum()
                ),
                "attack": int(
                    (y_test == 1).sum()
                ),
                "attack_prevalence": float(
                    y_test.mean()
                ),
            },
        ]
    )

    save_dataframe(
        split_summary,
        splits_directory
        / "split_summary.csv",
    )

    write_json(
        manifests_directory
        / "original_feature_manifest.json",
        {
            "dataset": dataset,
            "generated_utc": utc_now(),
            "representation": (
                "Original cleaned network-flow features"
            ),
            "target": TARGET_COLUMN,
            "retained_features": retained_columns,
            "removed_columns": removed_columns,
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "original_column_count": (
                original_column_count
            ),
            "predictive_feature_count": (
                len(retained_columns)
            ),
        },
    )

    LOGGER.info(
        "%s: fitting original-feature XGBoost model.",
        dataset,
    )

    training_start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    validation_start = time.perf_counter()

    validation_labels = pipeline.predict(
        X_validation
    )

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_inference_seconds = (
        time.perf_counter()
        - validation_start
    )

    test_start = time.perf_counter()

    test_labels = pipeline.predict(
        X_test
    )

    test_probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_inference_seconds = (
        time.perf_counter()
        - test_start
    )

    validation_metrics = calculate_metrics(
        y_true=y_validation,
        predicted_labels=validation_labels,
        probabilities=validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_true=y_test,
        predicted_labels=test_labels,
        probabilities=test_probabilities,
    )

    validation_predictions = pd.DataFrame(
        {
            "true_label": y_validation.to_numpy(),
            "predicted_label": validation_labels,
            "attack_probability": (
                validation_probabilities
            ),
        }
    )

    test_predictions = pd.DataFrame(
        {
            "true_label": y_test.to_numpy(),
            "predicted_label": test_labels,
            "attack_probability": (
                test_probabilities
            ),
        }
    )

    save_dataframe(
        validation_predictions,
        predictions_directory
        / "validation_predictions.csv",
    )

    save_dataframe(
        test_predictions,
        predictions_directory
        / "test_predictions.csv",
    )

    write_json(
        metrics_directory
        / "validation_metrics.json",
        validation_metrics,
    )

    write_json(
        metrics_directory
        / "test_metrics.json",
        test_metrics,
    )

    validation_report = classification_report(
        y_validation,
        validation_labels,
        labels=[0, 1],
        target_names=[
            "benign",
            "attack",
        ],
        output_dict=True,
        zero_division=0,
    )

    test_report = classification_report(
        y_test,
        test_labels,
        labels=[0, 1],
        target_names=[
            "benign",
            "attack",
        ],
        output_dict=True,
        zero_division=0,
    )

    write_json(
        metrics_directory
        / "validation_classification_report.json",
        validation_report,
    )

    write_json(
        metrics_directory
        / "test_classification_report.json",
        test_report,
    )

    save_confusion_matrix(
        y_true=y_test,
        predicted_labels=test_labels,
        output_path=(
            metrics_directory
            / "test_confusion_matrix.csv"
        ),
    )

    plot_roc_curve(
        y_true=y_test,
        probabilities=test_probabilities,
        output_path=(
            figures_directory
            / "test_roc_curve.png"
        ),
    )

    plot_precision_recall_curve(
        y_true=y_test,
        probabilities=test_probabilities,
        output_path=(
            figures_directory
            / "test_precision_recall_curve.png"
        ),
    )

    save_feature_importance(
        pipeline=pipeline,
        output_directory=(
            metrics_directory
        ),
    )

    with (
        model_directory
        / "original_feature_xgboost_pipeline.pkl"
    ).open("wb") as stream:
        pickle.dump(
            pipeline,
            stream,
        )

    latency_ms = (
        test_inference_seconds
        * 1000
        / max(len(X_test), 1)
    )

    throughput = (
        len(X_test)
        / max(test_inference_seconds, 1e-12)
    )

    experiment_result = ExperimentResult(
        dataset=dataset,
        status="PASS",
        total_available_cleaned_rows=(
            total_available
        ),
        total_rows_used=len(features),
        training_rows=len(X_train),
        validation_rows=len(X_validation),
        testing_rows=len(X_test),
        original_column_count=(
            original_column_count
        ),
        predictive_feature_count=(
            len(retained_columns)
        ),
        numeric_feature_count=(
            len(numeric_columns)
        ),
        categorical_feature_count=(
            len(categorical_columns)
        ),
        validation_accuracy=(
            validation_metrics["accuracy"]
        ),
        validation_balanced_accuracy=(
            validation_metrics[
                "balanced_accuracy"
            ]
        ),
        validation_f1=(
            validation_metrics["f1"]
        ),
        validation_roc_auc=(
            validation_metrics["roc_auc"]
        ),
        test_accuracy=(
            test_metrics["accuracy"]
        ),
        test_balanced_accuracy=(
            test_metrics[
                "balanced_accuracy"
            ]
        ),
        test_precision=(
            test_metrics["precision"]
        ),
        test_recall=(
            test_metrics["recall"]
        ),
        test_f1=(
            test_metrics["f1"]
        ),
        test_mcc=(
            test_metrics["mcc"]
        ),
        test_roc_auc=(
            test_metrics["roc_auc"]
        ),
        test_average_precision=(
            test_metrics[
                "average_precision"
            ]
        ),
        training_seconds=round(
            training_seconds,
            6,
        ),
        validation_inference_seconds=round(
            validation_inference_seconds,
            6,
        ),
        test_inference_seconds=round(
            test_inference_seconds,
            6,
        ),
        test_latency_ms_per_sample=round(
            latency_ms,
            9,
        ),
        test_throughput_samples_per_second=round(
            throughput,
            3,
        ),
        result_directory=str(
            dataset_root
        ),
        remarks=(
            "Original feature representation evaluated "
            "using a fixed leakage-safe XGBoost pipeline."
        ),
    )

    EXPERIMENT_RESULTS.append(
        experiment_result
    )

    write_json(
        dataset_root
        / "experiment_result.json",
        asdict(experiment_result),
    )

    LOGGER.info(
        "%s | accuracy=%.6f | balanced_accuracy=%.6f | "
        "F1=%.6f | MCC=%.6f | ROC-AUC=%.6f",
        dataset,
        experiment_result.test_accuracy,
        experiment_result.test_balanced_accuracy,
        experiment_result.test_f1,
        experiment_result.test_mcc,
        experiment_result.test_roc_auc,
    )

    LOGGER.info(
        "%s completed in %.2f seconds.",
        dataset,
        time.perf_counter()
        - experiment_start,
    )

    del dataframe
    del features
    del target
    del X_train
    del X_validation
    del X_test
    del y_train
    del y_validation
    del y_test
    del pipeline

    gc.collect()


# ============================================================
# Consolidated outputs
# ============================================================

def save_consolidated_results() -> None:
    result_records = [
        asdict(result)
        for result in EXPERIMENT_RESULTS
    ]

    result_frame = pd.DataFrame(
        result_records
    )

    save_dataframe(
        result_frame,
        REPORTS_DIR
        / "Original_Feature_Evaluation_Results.csv",
    )

    write_json(
        REPORTS_DIR
        / "Original_Feature_Evaluation_Results.json",
        {
            "generated_utc": utc_now(),
            "results": result_records,
        },
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2.1: evaluate original network-flow "
            "features using a fixed XGBoost classifier."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASET_NAMES, "all"),
        default="all",
        help="Dataset to evaluate. Default: all.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "Maximum records per dataset. "
            "Use 0 for all cleaned records. "
            f"Default: {DEFAULT_MAX_ROWS:,}."
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

    if args.max_rows < 0:
        raise ValueError(
            "--max-rows must be zero or positive."
        )

    selected_datasets = (
        DATASET_NAMES
        if args.dataset == "all"
        else (args.dataset,)
    )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "EXPERIMENT 2.1: ORIGINAL FEATURE-SPACE EVALUATION"
    )
    LOGGER.info(
        "Datasets: %s",
        ", ".join(selected_datasets),
    )
    LOGGER.info(
        "Maximum rows per dataset: %s",
        (
            "ALL"
            if args.max_rows == 0
            else f"{args.max_rows:,}"
        ),
    )
    LOGGER.info(
        "Random seed: %d",
        args.seed,
    )
    LOGGER.info(
        "Split: 70%% train, 10%% validation, 20%% test"
    )
    LOGGER.info("=" * 78)

    cleaning_summary = load_cleaning_summary()

    readiness_records: list[
        dict[str, object]
    ] = []

    runnable_datasets: list[str] = []

    for dataset in selected_datasets:
        ready, reason = check_dataset_readiness(
            dataset=dataset,
            cleaning_summary=cleaning_summary,
        )

        readiness_records.append(
            {
                "dataset": dataset,
                "ready": ready,
                "reason": reason,
            }
        )

        if ready:
            runnable_datasets.append(dataset)

            LOGGER.info(
                "%s | READY | %s",
                dataset,
                reason,
            )
        else:
            LOGGER.warning(
                "%s | SKIPPED | %s",
                dataset,
                reason,
            )

    save_dataframe(
        pd.DataFrame(
            readiness_records
        ),
        REPORTS_DIR
        / "Dataset_Readiness.csv",
    )

    write_json(
        MANIFESTS_DIR
        / "experiment_02_1_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "experiment": (
                "2.1 Original Feature-Space Evaluation"
            ),
            "representation": (
                "Original cleaned network-flow features"
            ),
            "classifier": "XGBoost",
            "selected_datasets": list(
                selected_datasets
            ),
            "runnable_datasets": (
                runnable_datasets
            ),
            "maximum_rows_per_dataset": (
                args.max_rows
            ),
            "random_seed": args.seed,
            "train_fraction": (
                TRAIN_FRACTION
            ),
            "validation_fraction": (
                VALIDATION_FRACTION
            ),
            "test_fraction": TEST_FRACTION,
            "minimum_retention_ratio": (
                MINIMUM_RETENTION_RATIO
            ),
            "ferf_enabled": False,
            "evidence_representation_enabled": (
                False
            ),
            "evidence_weighting_enabled": False,
            "reliability_estimation_enabled": (
                False
            ),
        },
    )

    if not runnable_datasets:
        LOGGER.error(
            "No dataset passed the readiness checks."
        )
        return 1

    failed_datasets: list[str] = []

    for dataset in runnable_datasets:
        try:
            run_original_feature_experiment(
                dataset=dataset,
                maximum_rows=args.max_rows,
                random_seed=args.seed,
            )
        except Exception:
            failed_datasets.append(dataset)

            LOGGER.exception(
                "Experiment 2.1 failed for %s.",
                dataset,
            )

    save_consolidated_results()

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Successful dataset runs: %d",
        len(EXPERIMENT_RESULTS),
    )
    LOGGER.info(
        "Failed dataset runs: %d",
        len(failed_datasets),
    )
    LOGGER.info(
        "Results directory: %s",
        RESULTS_ROOT,
    )
    LOGGER.info("=" * 78)

    if not EXPERIMENT_RESULTS:
        return 1

    if failed_datasets:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())