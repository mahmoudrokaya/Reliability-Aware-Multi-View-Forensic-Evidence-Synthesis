"""
06_experiment_02_2_forensic_evidence_representation.py

Experiment 2.2
Forensic Evidence Representation Evaluation

Purpose
-------
Construct and evaluate the proposed forensic evidence representation
using exactly the same records and train/validation/test assignments
used in Experiment 2.1.

Comparison
----------
Experiment 2.1:
    Original cleaned network-flow representation + XGBoost

Experiment 2.2:
    Original flow attributes
    + robust deviation evidence
    + evidence-quality descriptors
    + reliability-weighted evidence
    + the same XGBoost configuration

Leakage safeguards
------------------
1. The sampled records are reconstructed deterministically.
2. Saved split assignments from Experiment 2.1 are reused.
3. All medians, IQR values, clipping thresholds, category vocabularies,
   imputers, and scalers are fitted using training data only.
4. Labels and source metadata are excluded from predictors.
5. The test subset is used only once for final evaluation.

Current dataset safeguards
--------------------------
- CICIDS2017 and CSE-CIC-IDS2018 are evaluated when ready.
- UNSW-NB15 is rejected while its cleaning retention ratio is invalid.
- BoT-IoT is rejected while its cleaning status is not PASS.

Outputs
-------
For every dataset:
- forensic evidence feature manifest;
- evidence-quality summaries;
- transformed evidence samples;
- trained XGBoost pipeline;
- validation and test predictions;
- confusion matrix;
- ROC and precision-recall curves;
- feature importance;
- paired comparison with Experiment 2.1;
- independent result folders.
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

from sklearn.base import BaseEstimator, TransformerMixin
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

EXPERIMENT_21_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)

EXPERIMENT_21_RESULTS_PATH = (
    EXPERIMENT_21_ROOT
    / "Reports"
    / "Original_Feature_Evaluation_Results.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_02_Forensic_Evidence_Representation"
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
# Experiment configuration
# ============================================================

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
)

TARGET_COLUMN = "binary_label"

RANDOM_SEED = 42
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

EPSILON = 1e-8
ROBUST_DEVIATION_CLIP = 10.0
OUTLIER_THRESHOLD = 3.0

# The original numeric variables are retained alongside evidence
# descriptors to test whether evidence construction adds useful
# information without discarding the native representation.
INCLUDE_ORIGINAL_NUMERIC_FEATURES = True
INCLUDE_ORIGINAL_CATEGORICAL_FEATURES = True


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "experiment_02_2_forensic_evidence.log"

LOGGER = logging.getLogger("experiment_02_2_forensic_evidence")
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
class EvidenceExperimentResult:
    dataset: str
    status: str

    total_available_cleaned_rows: int
    total_rows_used: int
    training_rows: int
    validation_rows: int
    testing_rows: int

    original_predictive_features: int
    numeric_input_features: int
    categorical_input_features: int
    constructed_evidence_features: int

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
    test_inference_seconds: float
    test_latency_ms_per_sample: float
    test_throughput_samples_per_second: float

    original_accuracy: float
    delta_accuracy: float
    original_balanced_accuracy: float
    delta_balanced_accuracy: float
    original_f1: float
    delta_f1: float
    original_mcc: float
    delta_mcc: float
    original_roc_auc: float
    delta_roc_auc: float

    result_directory: str
    remarks: str


EXPERIMENT_RESULTS: list[EvidenceExperimentResult] = []


# ============================================================
# General utilities
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
        requested_rows * size / total_rows
        for size in part_sizes
    ]

    allocations = [
        min(size, int(np.floor(exact)))
        for size, exact in zip(
            part_sizes,
            exact_allocations,
        )
    ]

    remainder = requested_rows - sum(allocations)

    order = np.argsort(
        [
            exact - np.floor(exact)
            for exact in exact_allocations
        ]
    )[::-1]

    for index in order:
        if remainder <= 0:
            break

        if allocations[index] < part_sizes[index]:
            allocations[index] += 1
            remainder -= 1

    return allocations


# ============================================================
# Dataset readiness and loading
# ============================================================

def load_cleaning_summary() -> pd.DataFrame:
    if not CLEANING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Cleaning summary not found: {CLEANING_SUMMARY_PATH}"
        )

    summary = pd.read_csv(CLEANING_SUMMARY_PATH)
    summary["dataset"] = summary["dataset"].astype(str)
    return summary


def check_dataset_readiness(
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

    if input_rows <= 0 or output_rows <= 0:
        return False, "No usable cleaned records were produced."

    retention_ratio = output_rows / input_rows

    if retention_ratio < MINIMUM_RETENTION_RATIO:
        return (
            False,
            f"Retention ratio {retention_ratio:.4f} is below "
            f"{MINIMUM_RETENTION_RATIO:.2f}.",
        )

    if not discover_cleaned_parts(dataset):
        return False, "No cleaned parts were found."

    split_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Splits"
        / "fixed_split_assignments.csv"
    )

    if not split_path.exists():
        return (
            False,
            "Experiment 2.1 split assignments were not found.",
        )

    return (
        True,
        f"Ready; retention ratio={retention_ratio:.4f}.",
    )


def load_dataset(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int]:
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

    requested_rows = (
        total_available
        if maximum_rows <= 0
        else min(maximum_rows, total_available)
    )

    allocations = proportional_allocation(
        part_sizes,
        requested_rows,
    )

    frames: list[pd.DataFrame] = []

    for index, (
        path,
        part_size,
        selected_rows,
    ) in enumerate(
        zip(parts, part_sizes, allocations),
        start=1,
    ):
        if selected_rows <= 0:
            continue

        frame = read_cleaned_part(path)

        if selected_rows < part_size:
            frame = frame.sample(
                n=selected_rows,
                random_state=random_seed + index,
            )

        frames.append(frame)

        LOGGER.info(
            "%s | part=%d/%d | selected=%d/%d",
            dataset,
            index,
            len(parts),
            len(frame),
            part_size,
        )

    if not frames:
        raise RuntimeError(
            f"No records loaded for {dataset}."
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if len(dataframe) > requested_rows:
        dataframe = dataframe.sample(
            n=requested_rows,
            random_state=random_seed,
        ).reset_index(drop=True)

    LOGGER.info(
        "%s: loaded %d of %d records.",
        dataset,
        len(dataframe),
        total_available,
    )

    return dataframe, total_available


# ============================================================
# Original feature selection
# ============================================================

def prepare_original_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    target = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    valid_mask = target.isin([0, 1])

    dataframe = dataframe.loc[valid_mask].copy()
    target = target.loc[valid_mask].astype("int8")

    excluded = set(NON_PREDICTIVE_COLUMNS)

    for column in dataframe.columns:
        if is_identifier_column(column):
            excluded.add(column)

    features = dataframe[
        [
            column
            for column in dataframe.columns
            if column not in excluded
        ]
    ].copy()

    all_missing = [
        column
        for column in features.columns
        if features[column].isna().all()
    ]

    if all_missing:
        features = features.drop(columns=all_missing)

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        features = features.drop(
            columns=constant_columns
        )

    if features.empty:
        raise ValueError(
            "No predictive features remained."
        )

    removed = sorted(
        excluded
        | set(all_missing)
        | set(constant_columns)
    )

    return features, target, removed


# ============================================================
# Reuse Experiment 2.1 split assignments
# ============================================================

def load_fixed_split_assignments(
    dataset: str,
    expected_rows: int,
) -> pd.DataFrame:
    split_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Splits"
        / "fixed_split_assignments.csv"
    )

    assignments = pd.read_csv(split_path)

    required_columns = {
        "row_position",
        "split",
    }

    if not required_columns.issubset(
        assignments.columns
    ):
        raise ValueError(
            f"Invalid split assignment file: {split_path}"
        )

    assignments["row_position"] = pd.to_numeric(
        assignments["row_position"],
        errors="raise",
    ).astype(int)

    if len(assignments) != expected_rows:
        raise ValueError(
            "The reconstructed sample does not match Experiment 2.1. "
            f"Expected {expected_rows} split assignments but found "
            f"{len(assignments)}."
        )

    expected_positions = set(range(expected_rows))
    actual_positions = set(
        assignments["row_position"].tolist()
    )

    if actual_positions != expected_positions:
        raise ValueError(
            "Split row positions are incomplete or inconsistent."
        )

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        assignments["split"].astype(str)
    )

    if observed_splits != valid_splits:
        raise ValueError(
            f"Unexpected split labels: {observed_splits}"
        )

    return assignments.sort_values(
        "row_position"
    ).reset_index(drop=True)


def apply_saved_splits(
    features: pd.DataFrame,
    target: pd.Series,
    assignments: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    split_by_position = assignments.set_index(
        "row_position"
    )["split"]

    train_positions = split_by_position[
        split_by_position == "train"
    ].index.to_numpy()

    validation_positions = split_by_position[
        split_by_position == "validation"
    ].index.to_numpy()

    test_positions = split_by_position[
        split_by_position == "test"
    ].index.to_numpy()

    return (
        features.iloc[train_positions].copy(),
        features.iloc[validation_positions].copy(),
        features.iloc[test_positions].copy(),
        target.iloc[train_positions].copy(),
        target.iloc[validation_positions].copy(),
        target.iloc[test_positions].copy(),
    )


# ============================================================
# Forensic evidence transformer
# ============================================================

class ForensicEvidenceTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Construct dataset-independent forensic evidence descriptors.

    Numeric evidence generated for each original numeric feature:
    ------------------------------------------------------------
    1. robust_dev:
       Signed deviation from the training median divided by training IQR.

    2. absolute_dev:
       Magnitude of the robust deviation.

    3. anomaly_support:
       Smooth support that the observation is atypical:
           |d| / (1 + |d|)

    4. reliability:
       Reliability of the value based on availability and bounded
       deviation:
           availability * exp(-|d| / clip)

    5. weighted_evidence:
       anomaly_support * reliability

    Record-level quality evidence:
    ------------------------------
    - missing_fraction
    - observed_fraction
    - extreme_fraction
    - mean_absolute_deviation
    - maximum_absolute_deviation
    - mean_anomaly_support
    - mean_reliability
    - minimum_reliability
    - mean_weighted_evidence
    - maximum_weighted_evidence
    - evidence_dispersion
    - reliable_evidence_fraction
    - low_reliability_fraction

    All reference statistics are fitted from training records only.
    """

    def __init__(
        self,
        include_original_numeric: bool = True,
        deviation_clip: float = ROBUST_DEVIATION_CLIP,
        outlier_threshold: float = OUTLIER_THRESHOLD,
        epsilon: float = EPSILON,
    ) -> None:
        self.include_original_numeric = (
            include_original_numeric
        )
        self.deviation_clip = deviation_clip
        self.outlier_threshold = outlier_threshold
        self.epsilon = epsilon

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "ForensicEvidenceTransformer":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.input_features_ = list(X.columns)

        numeric = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        self.medians_ = numeric.median(axis=0)
        self.q1_ = numeric.quantile(0.25)
        self.q3_ = numeric.quantile(0.75)

        iqr = self.q3_ - self.q1_

        # Avoid zero denominators. Zero-IQR variables use a robust
        # fallback based on median absolute deviation.
        mad = (
            numeric
            .sub(self.medians_, axis=1)
            .abs()
            .median(axis=0)
        )

        robust_scale = iqr.copy()

        zero_iqr = (
            robust_scale.isna()
            | (robust_scale.abs() <= self.epsilon)
        )

        robust_scale.loc[zero_iqr] = (
            1.4826 * mad.loc[zero_iqr]
        )

        still_zero = (
            robust_scale.isna()
            | (robust_scale.abs() <= self.epsilon)
        )

        robust_scale.loc[still_zero] = 1.0

        self.robust_scale_ = robust_scale

        self.feature_names_out_ = (
            self._build_output_names()
        )

        return self

    def _build_output_names(self) -> list[str]:
        output_names: list[str] = []

        if self.include_original_numeric:
            output_names.extend(
                f"original__{column}"
                for column in self.input_features_
            )

        for column in self.input_features_:
            output_names.extend(
                [
                    f"robust_dev__{column}",
                    f"absolute_dev__{column}",
                    f"anomaly_support__{column}",
                    f"reliability__{column}",
                    f"weighted_evidence__{column}",
                    f"missing_indicator__{column}",
                ]
            )

        output_names.extend(
            [
                "quality__missing_fraction",
                "quality__observed_fraction",
                "quality__extreme_fraction",
                "quality__mean_absolute_deviation",
                "quality__maximum_absolute_deviation",
                "quality__mean_anomaly_support",
                "quality__mean_reliability",
                "quality__minimum_reliability",
                "quality__mean_weighted_evidence",
                "quality__maximum_weighted_evidence",
                "quality__evidence_dispersion",
                "quality__reliable_evidence_fraction",
                "quality__low_reliability_fraction",
            ]
        )

        return output_names

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.input_features_,
            )

        X = X[self.input_features_]

        numeric = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        missing = numeric.isna().astype(float)
        observed = 1.0 - missing

        imputed = numeric.fillna(self.medians_)

        deviation = (
            imputed.sub(self.medians_, axis=1)
            .div(self.robust_scale_, axis=1)
        )

        deviation = deviation.clip(
            lower=-self.deviation_clip,
            upper=self.deviation_clip,
        )

        absolute_deviation = deviation.abs()

        anomaly_support = (
            absolute_deviation
            / (1.0 + absolute_deviation)
        )

        reliability = (
            observed
            * np.exp(
                -absolute_deviation
                / self.deviation_clip
            )
        )

        weighted_evidence = (
            anomaly_support * reliability
        )

        extreme_indicator = (
            absolute_deviation
            >= self.outlier_threshold
        ).astype(float)

        reliable_indicator = (
            reliability >= 0.80
        ).astype(float)

        low_reliability_indicator = (
            reliability < 0.50
        ).astype(float)

        record_quality = pd.DataFrame(
            {
                "quality__missing_fraction": (
                    missing.mean(axis=1)
                ),
                "quality__observed_fraction": (
                    observed.mean(axis=1)
                ),
                "quality__extreme_fraction": (
                    extreme_indicator.mean(axis=1)
                ),
                "quality__mean_absolute_deviation": (
                    absolute_deviation.mean(axis=1)
                ),
                "quality__maximum_absolute_deviation": (
                    absolute_deviation.max(axis=1)
                ),
                "quality__mean_anomaly_support": (
                    anomaly_support.mean(axis=1)
                ),
                "quality__mean_reliability": (
                    reliability.mean(axis=1)
                ),
                "quality__minimum_reliability": (
                    reliability.min(axis=1)
                ),
                "quality__mean_weighted_evidence": (
                    weighted_evidence.mean(axis=1)
                ),
                "quality__maximum_weighted_evidence": (
                    weighted_evidence.max(axis=1)
                ),
                "quality__evidence_dispersion": (
                    weighted_evidence.std(
                        axis=1,
                        ddof=0,
                    )
                ),
                "quality__reliable_evidence_fraction": (
                    reliable_indicator.mean(axis=1)
                ),
                "quality__low_reliability_fraction": (
                    low_reliability_indicator.mean(
                        axis=1
                    )
                ),
            },
            index=X.index,
        )

        frames: list[pd.DataFrame] = []

        if self.include_original_numeric:
            original = imputed.copy()
            original.columns = [
                f"original__{column}"
                for column in original.columns
            ]
            frames.append(original)

        deviation.columns = [
            f"robust_dev__{column}"
            for column in deviation.columns
        ]

        absolute_deviation.columns = [
            f"absolute_dev__{column}"
            for column in absolute_deviation.columns
        ]

        anomaly_support.columns = [
            f"anomaly_support__{column}"
            for column in anomaly_support.columns
        ]

        reliability.columns = [
            f"reliability__{column}"
            for column in reliability.columns
        ]

        weighted_evidence.columns = [
            f"weighted_evidence__{column}"
            for column in weighted_evidence.columns
        ]

        missing.columns = [
            f"missing_indicator__{column}"
            for column in missing.columns
        ]

        frames.extend(
            [
                deviation,
                absolute_deviation,
                anomaly_support,
                reliability,
                weighted_evidence,
                missing,
                record_quality,
            ]
        )

        transformed = pd.concat(
            frames,
            axis=1,
        )

        return transformed.to_numpy(
            dtype=np.float32,
        )

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        return np.asarray(
            self.feature_names_out_,
            dtype=object,
        )


# ============================================================
# Evidence preprocessing pipeline
# ============================================================

def create_evidence_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[
    ColumnTransformer,
    list[str],
    list[str],
    int,
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

    evidence_transformer = (
        ForensicEvidenceTransformer(
            include_original_numeric=(
                INCLUDE_ORIGINAL_NUMERIC_FEATURES
            ),
        )
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "evidence",
                evidence_transformer,
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
                "forensic_numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if (
        categorical_columns
        and INCLUDE_ORIGINAL_CATEGORICAL_FEATURES
    ):
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

    evidence_feature_count = (
        len(numeric_columns)
        * (
            7
            if INCLUDE_ORIGINAL_NUMERIC_FEATURES
            else 6
        )
        + 13
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns,
        evidence_feature_count,
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
            "Install XGBoost using: pip install xgboost"
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
# Metrics and figures
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
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
    predictions: np.ndarray,
    path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    frame = pd.DataFrame(
        matrix,
        index=[
            "true_benign",
            "true_attack",
        ],
        columns=[
            "predicted_benign",
            "predicted_attack",
        ],
    ).reset_index(names="true_class")

    save_dataframe(frame, path)


def plot_roc(
    y_true: pd.Series,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.6f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Forensic Evidence Representation: ROC Curve"
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_precision_recall(
    y_true: pd.Series,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    ap = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {ap:.6f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Forensic Evidence Representation: "
        "Precision–Recall Curve"
    )
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importance(
    pipeline: Pipeline,
    output_directory: Path,
) -> None:
    classifier = pipeline.named_steps[
        "classifier"
    ]

    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        return

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )

        importances = classifier.feature_importances_

        importance_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        save_dataframe(
            importance_frame,
            output_directory
            / "evidence_feature_importance.csv",
        )

        top = importance_frame.head(30)

        plt.figure(figsize=(10, 8))
        plt.barh(
            top["feature"][::-1],
            top["importance"][::-1],
        )
        plt.xlabel("XGBoost Feature Importance")
        plt.ylabel("Evidence Feature")
        plt.title(
            "Top 30 Forensic Evidence Features"
        )
        plt.tight_layout()
        plt.savefig(
            output_directory
            / "evidence_feature_importance_top30.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as exc:
        LOGGER.warning(
            "Evidence feature importance export failed: %s",
            exc,
        )


# ============================================================
# Original-result lookup
# ============================================================

def load_original_results() -> pd.DataFrame:
    if not EXPERIMENT_21_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Experiment 2.1 consolidated result file was not found: "
            f"{EXPERIMENT_21_RESULTS_PATH}"
        )

    frame = pd.read_csv(
        EXPERIMENT_21_RESULTS_PATH
    )

    frame["dataset"] = frame[
        "dataset"
    ].astype(str)

    return frame


def original_metric_values(
    dataset: str,
    original_results: pd.DataFrame,
) -> dict[str, float]:
    matching = original_results[
        original_results["dataset"] == dataset
    ]

    if matching.empty:
        raise ValueError(
            f"No Experiment 2.1 result found for {dataset}."
        )

    row = matching.iloc[-1]

    return {
        "accuracy": float(
            row["test_accuracy"]
        ),
        "balanced_accuracy": float(
            row["test_balanced_accuracy"]
        ),
        "f1": float(row["test_f1"]),
        "mcc": float(row["test_mcc"]),
        "roc_auc": float(
            row["test_roc_auc"]
        ),
    }


# ============================================================
# Evidence quality export
# ============================================================

def export_evidence_quality_sample(
    pipeline: Pipeline,
    X: pd.DataFrame,
    output_path: Path,
    maximum_rows: int = 10_000,
) -> None:
    """
    Export a manageable sample of the transformed evidence matrix.

    The full transformed matrix is not persisted because it can be
    substantially larger than the source dataset.
    """

    sample = X.head(
        min(maximum_rows, len(X))
    )

    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    transformed = preprocessor.transform(
        sample
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    names = preprocessor.get_feature_names_out()

    evidence_frame = pd.DataFrame(
        transformed,
        columns=names,
    )

    quality_columns = [
        column
        for column in evidence_frame.columns
        if "quality__" in column
    ]

    if quality_columns:
        save_dataframe(
            evidence_frame[quality_columns],
            output_path,
        )


# ============================================================
# Dataset execution
# ============================================================

def run_dataset_experiment(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
    original_results: pd.DataFrame,
) -> None:
    dataset_start = time.perf_counter()

    dataset_root = RESULTS_ROOT / dataset

    model_dir = dataset_root / "Model"
    metrics_dir = dataset_root / "Metrics"
    predictions_dir = dataset_root / "Predictions"
    figures_dir = dataset_root / "Figures"
    evidence_dir = dataset_root / "Evidence"
    manifests_dir = dataset_root / "Manifests"

    for directory in (
        model_dir,
        metrics_dir,
        predictions_dir,
        figures_dir,
        evidence_dir,
        manifests_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Experiment 2.2 dataset: %s",
        dataset,
    )

    dataframe, total_available = load_dataset(
        dataset=dataset,
        maximum_rows=maximum_rows,
        random_seed=random_seed,
    )

    (
        features,
        target,
        removed_columns,
    ) = prepare_original_features(
        dataframe
    )

    assignments = load_fixed_split_assignments(
        dataset=dataset,
        expected_rows=len(features),
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = apply_saved_splits(
        features=features,
        target=target,
        assignments=assignments,
    )

    (
        evidence_preprocessor,
        numeric_columns,
        categorical_columns,
        estimated_evidence_features,
    ) = create_evidence_preprocessor(
        X_train
    )

    classifier = create_xgboost_model(
        random_seed=random_seed
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                evidence_preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    write_json(
        manifests_dir
        / "forensic_evidence_manifest.json",
        {
            "dataset": dataset,
            "generated_utc": utc_now(),
            "representation": (
                "Original flow features plus robust deviation, "
                "quality, reliability, and reliability-weighted "
                "forensic evidence descriptors"
            ),
            "original_predictive_features": list(
                features.columns
            ),
            "removed_columns": removed_columns,
            "numeric_features": numeric_columns,
            "categorical_features": (
                categorical_columns
            ),
            "estimated_numeric_evidence_features": (
                estimated_evidence_features
            ),
            "deviation_clip": (
                ROBUST_DEVIATION_CLIP
            ),
            "outlier_threshold": (
                OUTLIER_THRESHOLD
            ),
            "include_original_numeric": (
                INCLUDE_ORIGINAL_NUMERIC_FEATURES
            ),
            "include_original_categorical": (
                INCLUDE_ORIGINAL_CATEGORICAL_FEATURES
            ),
            "split_source": str(
                EXPERIMENT_21_ROOT
                / dataset
                / "Splits"
                / "fixed_split_assignments.csv"
            ),
        },
    )

    LOGGER.info(
        "%s: fitting forensic-evidence XGBoost model.",
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

    validation_predictions = pipeline.predict(
        X_validation
    )

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    test_start = time.perf_counter()

    test_predictions = pipeline.predict(
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
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    save_dataframe(
        pd.DataFrame(
            {
                "true_label": (
                    y_validation.to_numpy()
                ),
                "predicted_label": (
                    validation_predictions
                ),
                "attack_probability": (
                    validation_probabilities
                ),
            }
        ),
        predictions_dir
        / "validation_predictions.csv",
    )

    save_dataframe(
        pd.DataFrame(
            {
                "true_label": y_test.to_numpy(),
                "predicted_label": (
                    test_predictions
                ),
                "attack_probability": (
                    test_probabilities
                ),
            }
        ),
        predictions_dir
        / "test_predictions.csv",
    )

    write_json(
        metrics_dir
        / "validation_metrics.json",
        validation_metrics,
    )

    write_json(
        metrics_dir
        / "test_metrics.json",
        test_metrics,
    )

    write_json(
        metrics_dir
        / "test_classification_report.json",
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "benign",
                "attack",
            ],
            output_dict=True,
            zero_division=0,
        ),
    )

    save_confusion_matrix(
        y_test,
        test_predictions,
        metrics_dir
        / "test_confusion_matrix.csv",
    )

    plot_roc(
        y_test,
        test_probabilities,
        figures_dir
        / "test_roc_curve.png",
    )

    plot_precision_recall(
        y_test,
        test_probabilities,
        figures_dir
        / "test_precision_recall_curve.png",
    )

    save_feature_importance(
        pipeline,
        metrics_dir,
    )

    export_evidence_quality_sample(
        pipeline=pipeline,
        X=X_test,
        output_path=(
            evidence_dir
            / "test_evidence_quality_sample.csv"
        ),
    )

    with (
        model_dir
        / "forensic_evidence_xgboost_pipeline.pkl"
    ).open("wb") as stream:
        pickle.dump(
            pipeline,
            stream,
        )

    original = original_metric_values(
        dataset,
        original_results,
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

    result = EvidenceExperimentResult(
        dataset=dataset,
        status="PASS",
        total_available_cleaned_rows=(
            total_available
        ),
        total_rows_used=len(features),
        training_rows=len(X_train),
        validation_rows=len(X_validation),
        testing_rows=len(X_test),
        original_predictive_features=(
            features.shape[1]
        ),
        numeric_input_features=(
            len(numeric_columns)
        ),
        categorical_input_features=(
            len(categorical_columns)
        ),
        constructed_evidence_features=(
            estimated_evidence_features
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
        original_accuracy=(
            original["accuracy"]
        ),
        delta_accuracy=(
            test_metrics["accuracy"]
            - original["accuracy"]
        ),
        original_balanced_accuracy=(
            original["balanced_accuracy"]
        ),
        delta_balanced_accuracy=(
            test_metrics["balanced_accuracy"]
            - original["balanced_accuracy"]
        ),
        original_f1=original["f1"],
        delta_f1=(
            test_metrics["f1"]
            - original["f1"]
        ),
        original_mcc=original["mcc"],
        delta_mcc=(
            test_metrics["mcc"]
            - original["mcc"]
        ),
        original_roc_auc=(
            original["roc_auc"]
        ),
        delta_roc_auc=(
            test_metrics["roc_auc"]
            - original["roc_auc"]
        ),
        result_directory=str(
            dataset_root
        ),
        remarks=(
            "Paired comparison against Experiment 2.1 "
            "using identical records, splits, XGBoost "
            "hyperparameters, and target definition."
        ),
    )

    EXPERIMENT_RESULTS.append(result)

    write_json(
        dataset_root
        / "experiment_result.json",
        asdict(result),
    )

    comparison_frame = pd.DataFrame(
        [
            {
                "metric": "Accuracy",
                "original_features": (
                    original["accuracy"]
                ),
                "forensic_evidence": (
                    result.test_accuracy
                ),
                "difference": (
                    result.delta_accuracy
                ),
            },
            {
                "metric": "Balanced Accuracy",
                "original_features": (
                    original[
                        "balanced_accuracy"
                    ]
                ),
                "forensic_evidence": (
                    result.test_balanced_accuracy
                ),
                "difference": (
                    result.delta_balanced_accuracy
                ),
            },
            {
                "metric": "F1-score",
                "original_features": (
                    original["f1"]
                ),
                "forensic_evidence": (
                    result.test_f1
                ),
                "difference": (
                    result.delta_f1
                ),
            },
            {
                "metric": "MCC",
                "original_features": (
                    original["mcc"]
                ),
                "forensic_evidence": (
                    result.test_mcc
                ),
                "difference": (
                    result.delta_mcc
                ),
            },
            {
                "metric": "ROC-AUC",
                "original_features": (
                    original["roc_auc"]
                ),
                "forensic_evidence": (
                    result.test_roc_auc
                ),
                "difference": (
                    result.delta_roc_auc
                ),
            },
        ]
    )

    save_dataframe(
        comparison_frame,
        metrics_dir
        / "paired_representation_comparison.csv",
    )

    LOGGER.info(
        "%s | evidence accuracy=%.6f | "
        "balanced_accuracy=%.6f | F1=%.6f | "
        "MCC=%.6f | ROC-AUC=%.6f",
        dataset,
        result.test_accuracy,
        result.test_balanced_accuracy,
        result.test_f1,
        result.test_mcc,
        result.test_roc_auc,
    )

    LOGGER.info(
        "%s | deltas: Accuracy=%+.6f | "
        "Balanced Accuracy=%+.6f | F1=%+.6f | "
        "MCC=%+.6f | ROC-AUC=%+.6f",
        dataset,
        result.delta_accuracy,
        result.delta_balanced_accuracy,
        result.delta_f1,
        result.delta_mcc,
        result.delta_roc_auc,
    )

    LOGGER.info(
        "%s completed in %.2f seconds.",
        dataset,
        time.perf_counter()
        - dataset_start,
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
# Consolidated reports
# ============================================================

def save_consolidated_results() -> None:
    records = [
        asdict(result)
        for result in EXPERIMENT_RESULTS
    ]

    frame = pd.DataFrame(records)

    save_dataframe(
        frame,
        REPORTS_DIR
        / "Forensic_Evidence_Representation_Results.csv",
    )

    write_json(
        REPORTS_DIR
        / "Forensic_Evidence_Representation_Results.json",
        {
            "generated_utc": utc_now(),
            "results": records,
        },
    )

    if not frame.empty:
        comparison_columns = [
            "dataset",
            "original_accuracy",
            "test_accuracy",
            "delta_accuracy",
            "original_balanced_accuracy",
            "test_balanced_accuracy",
            "delta_balanced_accuracy",
            "original_f1",
            "test_f1",
            "delta_f1",
            "original_mcc",
            "test_mcc",
            "delta_mcc",
            "original_roc_auc",
            "test_roc_auc",
            "delta_roc_auc",
        ]

        save_dataframe(
            frame[comparison_columns],
            REPORTS_DIR
            / "Original_vs_Evidence_Comparison.csv",
        )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2.2: construct and evaluate "
            "the forensic evidence representation."
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
            "Maximum records per dataset. This must match "
            "Experiment 2.1. Use 0 for all records. "
            f"Default: {DEFAULT_MAX_ROWS:,}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            "Random seed used in Experiment 2.1. "
            f"Default: {RANDOM_SEED}."
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
        "EXPERIMENT 2.2: FORENSIC EVIDENCE REPRESENTATION"
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
        "Split source: Experiment 2.1 saved assignments"
    )
    LOGGER.info("=" * 78)

    cleaning_summary = load_cleaning_summary()
    original_results = load_original_results()

    readiness_records: list[
        dict[str, object]
    ] = []

    runnable_datasets: list[str] = []

    for dataset in selected_datasets:
        ready, reason = check_dataset_readiness(
            dataset,
            cleaning_summary,
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
        REPORTS_DIR
        / "Dataset_Readiness.csv",
    )

    write_json(
        MANIFESTS_DIR
        / "experiment_02_2_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "experiment": (
                "2.2 Forensic Evidence Representation"
            ),
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
            "classifier": "XGBoost",
            "paired_with_experiment": (
                "2.1 Original Feature-Space Evaluation"
            ),
            "original_numeric_features_retained": (
                INCLUDE_ORIGINAL_NUMERIC_FEATURES
            ),
            "original_categorical_features_retained": (
                INCLUDE_ORIGINAL_CATEGORICAL_FEATURES
            ),
            "evidence_components": [
                "robust signed deviation",
                "absolute deviation",
                "anomaly support",
                "evidence reliability",
                "reliability-weighted evidence",
                "missingness indicators",
                "record-level quality descriptors",
            ],
            "leakage_safeguard": (
                "All evidence reference statistics are "
                "estimated using training records only."
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
            run_dataset_experiment(
                dataset=dataset,
                maximum_rows=args.max_rows,
                random_seed=args.seed,
                original_results=original_results,
            )
        except Exception:
            failed_datasets.append(dataset)

            LOGGER.exception(
                "Experiment 2.2 failed for %s.",
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