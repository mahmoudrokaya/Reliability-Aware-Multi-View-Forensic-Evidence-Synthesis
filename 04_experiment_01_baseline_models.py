"""
04_experiment_01_baseline_models.py

Experiment 1: Baseline Model Evaluation

Evaluates:
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The script uses cleaned outputs from:
Experiment_01_Data_Preparation/
Phase_02_Integrity_Verification/
Step_02_Data_Cleaning/

Main safeguards
---------------
1. Reads only standardized cleaned parts.
2. Verifies cleaning retention before model development.
3. Skips incomplete or severely reduced datasets.
4. Uses stratified 70/10/20 train-validation-test splitting.
5. Fits preprocessing only on training data.
6. Removes metadata, labels, IP addresses, timestamps, and identifiers
   from predictive features.
7. Saves every model's results in an independent folder.
8. Generates dataset-level and consolidated result tables.

Development versus final execution
----------------------------------
Default:
    At most 1,000,000 rows per dataset.

Final complete experiment:
    --max-rows 0

Use the complete experiment only after correcting the cleaning policy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import pickle
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore", category=FutureWarning)


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

CLEANING_SUMMARY_PATH = (
    CLEANING_ROOT
    / "Reports"
    / "Dataset_Cleaning_Summary.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_02_Baseline_Models"
    / "Experiment_01_Baseline_Classification"
)

LOGS_DIR = RESULTS_ROOT / "Logs"
REPORTS_DIR = RESULTS_ROOT / "Reports"
MANIFESTS_DIR = RESULTS_ROOT / "Manifests"

for directory in (
    RESULTS_ROOT,
    LOGS_DIR,
    REPORTS_DIR,
    MANIFESTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment configuration
# ============================================================

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
)

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20

MINIMUM_RETENTION_RATIO = 0.50

TARGET_COLUMN = "binary_label"

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
    r"(^|_)id($|_)",
)


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "experiment_01_baselines.log"

LOGGER = logging.getLogger("experiment_01_baselines")
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
# Result structure
# ============================================================

@dataclass
class ModelResult:
    dataset: str
    model: str
    status: str

    total_rows_used: int
    training_rows: int
    validation_rows: int
    testing_rows: int
    feature_count_before_encoding: int

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    matthews_correlation_coefficient: float
    roc_auc: float

    training_seconds: float
    inference_seconds: float
    inference_ms_per_sample: float

    result_directory: str
    remarks: str


RESULTS: list[ModelResult] = []


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
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


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
    cleaned_dir = (
        CLEANING_ROOT
        / dataset
        / "Cleaned_Data"
    )

    if not cleaned_dir.exists():
        return []

    parts = sorted(cleaned_dir.glob("cleaned_part_*.parquet"))

    if parts:
        return parts

    return sorted(cleaned_dir.glob("cleaned_part_*.csv.gz"))


def load_cleaning_summary() -> pd.DataFrame:
    if not CLEANING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Cleaning summary was not found: {CLEANING_SUMMARY_PATH}"
        )

    summary = pd.read_csv(CLEANING_SUMMARY_PATH)
    summary["dataset"] = summary["dataset"].astype(str)

    return summary


def evaluate_dataset_readiness(
    dataset: str,
    cleaning_summary: pd.DataFrame,
) -> tuple[bool, str]:
    matching = cleaning_summary[
        cleaning_summary["dataset"] == dataset
    ]

    if matching.empty:
        return False, "No cleaning summary was found."

    row = matching.iloc[-1]

    status = str(row.get("status", "")).upper()

    if status != "PASS":
        return False, f"Cleaning status is {status}."

    input_rows = int(row.get("input_rows", 0))
    output_rows = int(row.get("output_rows", 0))

    if input_rows <= 0:
        return False, "The cleaning summary reports no input records."

    retention_ratio = output_rows / input_rows

    if retention_ratio < MINIMUM_RETENTION_RATIO:
        return (
            False,
            f"Retention ratio {retention_ratio:.4f} is below "
            f"the required {MINIMUM_RETENTION_RATIO:.2f}.",
        )

    parts = discover_cleaned_parts(dataset)

    if not parts:
        return False, "No cleaned data parts were found."

    return (
        True,
        f"Dataset passed readiness checks; "
        f"retention ratio={retention_ratio:.4f}.",
    )


# ============================================================
# Dataset loading
# ============================================================

def read_cleaned_part(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )


def proportional_part_allocation(
    part_sizes: list[int],
    target_rows: int,
) -> list[int]:
    total_rows = sum(part_sizes)

    if target_rows <= 0 or target_rows >= total_rows:
        return part_sizes.copy()

    exact = [
        target_rows * size / total_rows
        for size in part_sizes
    ]

    allocation = [
        min(size, int(np.floor(value)))
        for size, value in zip(part_sizes, exact)
    ]

    remainder = target_rows - sum(allocation)

    order = np.argsort(
        [
            value - np.floor(value)
            for value in exact
        ]
    )[::-1]

    for index in order:
        if remainder <= 0:
            break

        if allocation[index] < part_sizes[index]:
            allocation[index] += 1
            remainder -= 1

    return allocation


def count_part_rows(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows

    count = 0

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=[0],
        chunksize=250_000,
    ):
        count += len(chunk)

    return count


def load_dataset(
    dataset: str,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    parts = discover_cleaned_parts(dataset)

    if not parts:
        raise FileNotFoundError(
            f"No cleaned parts found for {dataset}."
        )

    LOGGER.info(
        "%s: counting rows in %d cleaned part(s).",
        dataset,
        len(parts),
    )

    part_sizes = [
        count_part_rows(path)
        for path in parts
    ]

    total_available = sum(part_sizes)

    target_rows = (
        total_available
        if max_rows <= 0
        else min(max_rows, total_available)
    )

    allocation = proportional_part_allocation(
        part_sizes=part_sizes,
        target_rows=target_rows,
    )

    frames: list[pd.DataFrame] = []

    for part_index, (path, rows_to_take, part_size) in enumerate(
        zip(parts, allocation, part_sizes),
        start=1,
    ):
        if rows_to_take <= 0:
            continue

        frame = read_cleaned_part(path)

        if rows_to_take < part_size:
            frame = frame.sample(
                n=rows_to_take,
                random_state=random_seed + part_index,
            )

        frames.append(frame)

        LOGGER.info(
            "%s | part=%d/%d | selected=%d/%d",
            dataset,
            part_index,
            len(parts),
            len(frame),
            part_size,
        )

    if not frames:
        raise RuntimeError(
            f"No rows were loaded for {dataset}."
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if len(dataframe) > target_rows:
        dataframe = dataframe.sample(
            n=target_rows,
            random_state=random_seed,
        ).reset_index(drop=True)

    LOGGER.info(
        "%s: loaded %d of %d available cleaned records.",
        dataset,
        len(dataframe),
        total_available,
    )

    return dataframe


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    y = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    valid_target = y.isin([0, 1])

    dataframe = dataframe.loc[valid_target].copy()
    y = y.loc[valid_target].astype("int8")

    excluded = set(NON_PREDICTIVE_COLUMNS)

    for column in dataframe.columns:
        if is_identifier_column(column):
            excluded.add(column)

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    X = dataframe[feature_columns].copy()

    all_missing_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if all_missing_columns:
        X = X.drop(columns=all_missing_columns)

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        X = X.drop(columns=constant_columns)

    removed_columns = sorted(
        excluded
        | set(all_missing_columns)
        | set(constant_columns)
    )

    if X.empty:
        raise ValueError(
            "No predictive features remained after exclusions."
        )

    return X, y, removed_columns


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=random_seed,
        stratify=y,
    )

    relative_test_size = (
        TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=random_seed,
            stratify=y_temp,
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def create_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
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
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=10,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []

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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


# ============================================================
# Models
# ============================================================

def build_models(
    random_seed: int,
    include_optional: bool,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Decision_Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_seed,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_seed,
        ),
    }

    if not include_optional:
        return models

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_seed,
        )
    except ImportError:
        LOGGER.warning(
            "XGBoost is unavailable. Install with: pip install xgboost"
        )

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
            verbosity=-1,
        )
    except ImportError:
        LOGGER.warning(
            "LightGBM is unavailable. Install with: pip install lightgbm"
        )

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=8,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:
        LOGGER.warning(
            "CatBoost is unavailable. Install with: pip install catboost"
        )

    return models


# ============================================================
# Evaluation
# ============================================================

def safe_roc_auc(
    y_true: pd.Series,
    probability: np.ndarray | None,
) -> float:
    if probability is None:
        return float("nan")

    try:
        return float(
            roc_auc_score(y_true, probability)
        )
    except ValueError:
        return float("nan")


def obtain_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)

        if probabilities.ndim == 2:
            return probabilities[:, 1]

        return probabilities

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)

        return 1.0 / (1.0 + np.exp(-scores))

    return None


def run_model(
    dataset: str,
    model_name: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    model_result_dir = (
        RESULTS_ROOT
        / dataset
        / model_name
    )

    metrics_dir = model_result_dir / "Metrics"
    predictions_dir = model_result_dir / "Predictions"
    model_dir = model_result_dir / "Model"

    for directory in (
        metrics_dir,
        predictions_dir,
        model_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", estimator),
        ]
    )

    LOGGER.info(
        "%s | %s | training started.",
        dataset,
        model_name,
    )

    training_start = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_seconds = (
        time.perf_counter() - training_start
    )

    validation_predictions = pipeline.predict(
        X_validation
    )

    validation_probabilities = obtain_probabilities(
        pipeline,
        X_validation,
    )

    validation_metrics = {
        "accuracy": accuracy_score(
            y_validation,
            validation_predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation,
            validation_predictions,
        ),
        "f1": f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        ),
        "roc_auc": safe_roc_auc(
            y_validation,
            validation_probabilities,
        ),
    }

    inference_start = time.perf_counter()

    test_predictions = pipeline.predict(X_test)

    test_probabilities = obtain_probabilities(
        pipeline,
        X_test,
    )

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_predictions,
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_test,
        test_predictions,
    )

    auc = safe_roc_auc(
        y_test,
        test_probabilities,
    )

    confusion = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["benign", "attack"],
        output_dict=True,
        zero_division=0,
    )

    prediction_frame = pd.DataFrame(
        {
            "true_label": y_test.to_numpy(),
            "predicted_label": test_predictions,
            "attack_probability": (
                test_probabilities
                if test_probabilities is not None
                else np.nan
            ),
        }
    )

    save_dataframe(
        prediction_frame,
        predictions_dir / "test_predictions.csv",
    )

    save_dataframe(
        pd.DataFrame(
            confusion,
            index=["true_benign", "true_attack"],
            columns=[
                "predicted_benign",
                "predicted_attack",
            ],
        ).reset_index(names="true_class"),
        metrics_dir / "confusion_matrix.csv",
    )

    write_json(
        metrics_dir / "classification_report.json",
        report,
    )

    write_json(
        metrics_dir / "validation_metrics.json",
        validation_metrics,
    )

    test_metrics = {
        "dataset": dataset,
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "matthews_correlation_coefficient": mcc,
        "roc_auc": auc,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": (
            inference_seconds
            * 1000
            / max(len(X_test), 1)
        ),
    }

    write_json(
        metrics_dir / "test_metrics.json",
        test_metrics,
    )

    with (
        model_dir / "trained_pipeline.pkl"
    ).open("wb") as stream:
        pickle.dump(pipeline, stream)

    RESULTS.append(
        ModelResult(
            dataset=dataset,
            model=model_name,
            status="PASS",
            total_rows_used=(
                len(X_train)
                + len(X_validation)
                + len(X_test)
            ),
            training_rows=len(X_train),
            validation_rows=len(X_validation),
            testing_rows=len(X_test),
            feature_count_before_encoding=X_train.shape[1],
            accuracy=float(accuracy),
            balanced_accuracy=float(
                balanced_accuracy
            ),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            matthews_correlation_coefficient=float(
                mcc
            ),
            roc_auc=float(auc),
            training_seconds=round(
                training_seconds,
                6,
            ),
            inference_seconds=round(
                inference_seconds,
                6,
            ),
            inference_ms_per_sample=round(
                inference_seconds
                * 1000
                / max(len(X_test), 1),
                9,
            ),
            result_directory=str(
                model_result_dir
            ),
            remarks=(
                "Binary classification using a "
                "leakage-safe preprocessing pipeline."
            ),
        )
    )

    LOGGER.info(
        "%s | %s | accuracy=%.6f | "
        "balanced_accuracy=%.6f | F1=%.6f | AUC=%.6f",
        dataset,
        model_name,
        accuracy,
        balanced_accuracy,
        f1,
        auc,
    )

    del pipeline
    gc.collect()


# ============================================================
# Dataset experiment
# ============================================================

def run_dataset_experiment(
    dataset: str,
    max_rows: int,
    random_seed: int,
    include_optional_models: bool,
) -> None:
    dataset_start = time.perf_counter()

    LOGGER.info("=" * 78)
    LOGGER.info("Dataset experiment: %s", dataset)

    dataframe = load_dataset(
        dataset=dataset,
        max_rows=max_rows,
        random_seed=random_seed,
    )

    X, y, removed_columns = prepare_features(
        dataframe
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(
        X=X,
        y=y,
        random_seed=random_seed,
    )

    dataset_dir = RESULTS_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_summary = pd.DataFrame(
        [
            {
                "split": "training",
                "rows": len(y_train),
                "benign": int((y_train == 0).sum()),
                "attack": int((y_train == 1).sum()),
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
            },
            {
                "split": "testing",
                "rows": len(y_test),
                "benign": int((y_test == 0).sum()),
                "attack": int((y_test == 1).sum()),
            },
        ]
    )

    save_dataframe(
        split_summary,
        dataset_dir / "dataset_split_summary.csv",
    )

    write_json(
        dataset_dir / "feature_manifest.json",
        {
            "dataset": dataset,
            "generated_utc": utc_now(),
            "predictive_features": list(X.columns),
            "removed_columns": removed_columns,
            "number_of_predictive_features": X.shape[1],
        },
    )

    models = build_models(
        random_seed=random_seed,
        include_optional=include_optional_models,
    )

    for model_name, estimator in models.items():
        try:
            preprocessor = create_preprocessor(
                X_train
            )

            run_model(
                dataset=dataset,
                model_name=model_name,
                estimator=estimator,
                preprocessor=preprocessor,
                X_train=X_train,
                X_validation=X_validation,
                X_test=X_test,
                y_train=y_train,
                y_validation=y_validation,
                y_test=y_test,
            )
        except Exception as exc:
            LOGGER.exception(
                "%s | %s failed.",
                dataset,
                model_name,
            )

            RESULTS.append(
                ModelResult(
                    dataset=dataset,
                    model=model_name,
                    status="FAILED",
                    total_rows_used=len(dataframe),
                    training_rows=len(X_train),
                    validation_rows=len(
                        X_validation
                    ),
                    testing_rows=len(X_test),
                    feature_count_before_encoding=(
                        X_train.shape[1]
                    ),
                    accuracy=float("nan"),
                    balanced_accuracy=float("nan"),
                    precision=float("nan"),
                    recall=float("nan"),
                    f1_score=float("nan"),
                    matthews_correlation_coefficient=(
                        float("nan")
                    ),
                    roc_auc=float("nan"),
                    training_seconds=float("nan"),
                    inference_seconds=float("nan"),
                    inference_ms_per_sample=float(
                        "nan"
                    ),
                    result_directory=str(
                        RESULTS_ROOT
                        / dataset
                        / model_name
                    ),
                    remarks=str(exc),
                )
            )

    LOGGER.info(
        "%s completed in %.2f seconds.",
        dataset,
        time.perf_counter() - dataset_start,
    )

    del dataframe
    del X
    del y
    del X_train
    del X_validation
    del X_test
    del y_train
    del y_validation
    del y_test

    gc.collect()


# ============================================================
# Consolidated reporting
# ============================================================

def save_consolidated_results() -> None:
    records = [
        asdict(result)
        for result in RESULTS
    ]

    result_frame = pd.DataFrame(records)

    save_dataframe(
        result_frame,
        REPORTS_DIR / "Baseline_Model_Results.csv",
    )

    write_json(
        REPORTS_DIR / "Baseline_Model_Results.json",
        {
            "generated_utc": utc_now(),
            "results": records,
        },
    )

    successful = result_frame[
        result_frame["status"] == "PASS"
    ].copy()

    if not successful.empty:
        successful = successful.sort_values(
            by=[
                "dataset",
                "roc_auc",
                "f1_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        save_dataframe(
            successful,
            REPORTS_DIR
            / "Baseline_Model_Ranking.csv",
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1: leakage-safe baseline model "
            "evaluation on cleaned intrusion datasets."
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
        default=1_000_000,
        help=(
            "Maximum records used per dataset. "
            "Use 0 for all records. Default: 1,000,000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}.",
    )

    parser.add_argument(
        "--basic-models-only",
        action="store_true",
        help=(
            "Evaluate only Decision Tree and Random Forest."
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
            "--max-rows must be zero or a positive integer."
        )

    selected_datasets = (
        DATASET_NAMES
        if args.dataset == "all"
        else (args.dataset,)
    )

    LOGGER.info("=" * 78)
    LOGGER.info("EXPERIMENT 1: BASELINE MODEL EVALUATION")
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
        "Split: training=70%%, validation=10%%, testing=20%%"
    )
    LOGGER.info("=" * 78)

    cleaning_summary = load_cleaning_summary()

    readiness_records: list[dict[str, Any]] = []

    runnable_datasets: list[str] = []

    for dataset in selected_datasets:
        ready, reason = evaluate_dataset_readiness(
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
        pd.DataFrame(readiness_records),
        REPORTS_DIR / "Dataset_Readiness.csv",
    )

    write_json(
        MANIFESTS_DIR
        / "experiment_01_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "selected_datasets": list(
                selected_datasets
            ),
            "runnable_datasets": runnable_datasets,
            "maximum_rows_per_dataset": (
                args.max_rows
            ),
            "random_seed": args.seed,
            "training_fraction": TRAIN_SIZE,
            "validation_fraction": (
                VALIDATION_SIZE
            ),
            "testing_fraction": TEST_SIZE,
            "minimum_retention_ratio": (
                MINIMUM_RETENTION_RATIO
            ),
        },
    )

    if not runnable_datasets:
        LOGGER.error(
            "No dataset passed the readiness checks."
        )
        return 1

    for dataset in runnable_datasets:
        try:
            run_dataset_experiment(
                dataset=dataset,
                max_rows=args.max_rows,
                random_seed=args.seed,
                include_optional_models=(
                    not args.basic_models_only
                ),
            )
        except Exception:
            LOGGER.exception(
                "Dataset experiment failed: %s",
                dataset,
            )

    save_consolidated_results()

    successful_results = [
        result
        for result in RESULTS
        if result.status == "PASS"
    ]

    failed_results = [
        result
        for result in RESULTS
        if result.status == "FAILED"
    ]

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Successful model runs: %d",
        len(successful_results),
    )
    LOGGER.info(
        "Failed model runs: %d",
        len(failed_results),
    )
    LOGGER.info(
        "Results directory: %s",
        RESULTS_ROOT,
    )
    LOGGER.info("=" * 78)

    if not successful_results:
        return 1

    if failed_results:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())