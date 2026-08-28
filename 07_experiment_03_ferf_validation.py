"""
07_experiment_03_ferf_validation.py

Experiment 3
Forensic Evidence Reliability Fusion (FERF) Validation

This corrected version defines ForensicEvidenceTransformer in the
__main__ module before loading the pipeline saved by Experiment 2.2.

That definition is required because the saved pickle references:

    __main__.ForensicEvidenceTransformer

Experimental branches
---------------------
1. Original-feature XGBoost model from Experiment 2.1.
2. Forensic-evidence XGBoost model from Experiment 2.2.
3. Quality-conditioned XGBoost branch trained in this experiment.
4. Unweighted probability fusion.
5. Proposed reliability-weighted FERF.

Experimental safeguards
-----------------------
- The same deterministic sample used in Experiments 2.1 and 2.2.
- The saved train/validation/test assignments from Experiment 2.1.
- FERF weights selected using validation data only.
- Decision threshold selected using validation data only.
- Final test evaluation performed once.
- No test labels used during model or fusion configuration.
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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import StandardScaler


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

EXPERIMENT_22_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_02_Forensic_Evidence_Representation"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_03_Framework_Validation"
    / "Experiment_03_FERF_Validation"
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
DEFAULT_MAX_ROWS = 1_000_000
MINIMUM_RETENTION_RATIO = 0.50

EPSILON = 1e-8
ROBUST_DEVIATION_CLIP = 10.0
OUTLIER_THRESHOLD = 3.0

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

# The grid is evaluated on the validation subset only.
WEIGHT_VALUES = np.arange(0.0, 1.01, 0.10)
THRESHOLD_VALUES = np.arange(0.20, 0.801, 0.01)


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "experiment_03_ferf_validation.log"

LOGGER = logging.getLogger("experiment_03_ferf_validation")
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
# Option 2 fix
# Custom transformer required by the Experiment 2.2 pickle
# ============================================================

class ForensicEvidenceTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Reproduces the transformer used by Experiment 2.2.

    This class must be defined at module level before pickle.load()
    is called because the saved pipeline references:

        __main__.ForensicEvidenceTransformer
    """

    def __init__(
        self,
        include_original_numeric: bool = True,
        deviation_clip: float = ROBUST_DEVIATION_CLIP,
        outlier_threshold: float = OUTLIER_THRESHOLD,
        epsilon: float = EPSILON,
    ) -> None:
        self.include_original_numeric = include_original_numeric
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
        self.feature_names_out_ = self._build_output_names()

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
            imputed
            .sub(self.medians_, axis=1)
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
                    low_reliability_indicator.mean(axis=1)
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
# Quality-conditioned evidence transformer
# ============================================================

class QualityFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Constructs record-level evidence-quality descriptors.

    No target labels are used when constructing the descriptors.
    """

    def __init__(
        self,
        deviation_clip: float = ROBUST_DEVIATION_CLIP,
        epsilon: float = EPSILON,
    ) -> None:
        self.deviation_clip = deviation_clip
        self.epsilon = epsilon

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "QualityFeatureTransformer":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_ = list(X.columns)

        numeric = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        self.medians_ = numeric.median(axis=0)

        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        scale = q3 - q1

        mad = (
            numeric
            .sub(self.medians_, axis=1)
            .abs()
            .median(axis=0)
        )

        invalid_scale = (
            scale.isna()
            | (scale.abs() <= self.epsilon)
        )

        scale.loc[invalid_scale] = (
            1.4826 * mad.loc[invalid_scale]
        )

        invalid_scale = (
            scale.isna()
            | (scale.abs() <= self.epsilon)
        )

        scale.loc[invalid_scale] = 1.0

        self.scale_ = scale

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.columns_,
            )

        numeric = X[self.columns_].apply(
            pd.to_numeric,
            errors="coerce",
        )

        missing = numeric.isna().astype(float)
        observed = 1.0 - missing

        imputed = numeric.fillna(self.medians_)

        deviation = (
            imputed
            .sub(self.medians_, axis=1)
            .div(self.scale_, axis=1)
            .clip(
                lower=-self.deviation_clip,
                upper=self.deviation_clip,
            )
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

        output = pd.DataFrame(
            {
                "quality__missing_fraction": (
                    missing.mean(axis=1)
                ),
                "quality__observed_fraction": (
                    observed.mean(axis=1)
                ),
                "quality__mean_absolute_deviation": (
                    absolute_deviation.mean(axis=1)
                ),
                "quality__maximum_absolute_deviation": (
                    absolute_deviation.max(axis=1)
                ),
                "quality__extreme_fraction": (
                    absolute_deviation
                    .ge(OUTLIER_THRESHOLD)
                    .mean(axis=1)
                ),
                "quality__mean_anomaly_support": (
                    anomaly_support.mean(axis=1)
                ),
                "quality__maximum_anomaly_support": (
                    anomaly_support.max(axis=1)
                ),
                "quality__mean_reliability": (
                    reliability.mean(axis=1)
                ),
                "quality__minimum_reliability": (
                    reliability.min(axis=1)
                ),
                "quality__reliable_fraction": (
                    reliability.ge(0.80).mean(axis=1)
                ),
                "quality__low_reliability_fraction": (
                    reliability.lt(0.50).mean(axis=1)
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
            },
            index=X.index,
        )

        return output.to_numpy(
            dtype=np.float32,
        )

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        return np.asarray(
            [
                "quality__missing_fraction",
                "quality__observed_fraction",
                "quality__mean_absolute_deviation",
                "quality__maximum_absolute_deviation",
                "quality__extreme_fraction",
                "quality__mean_anomaly_support",
                "quality__maximum_anomaly_support",
                "quality__mean_reliability",
                "quality__minimum_reliability",
                "quality__reliable_fraction",
                "quality__low_reliability_fraction",
                "quality__mean_weighted_evidence",
                "quality__maximum_weighted_evidence",
                "quality__evidence_dispersion",
            ],
            dtype=object,
        )


# ============================================================
# Result structure
# ============================================================

@dataclass
class FerfResult:
    dataset: str
    status: str

    total_available_rows: int
    total_rows_used: int
    training_rows: int
    validation_rows: int
    testing_rows: int

    original_weight: float
    evidence_weight: float
    quality_weight: float
    optimized_threshold: float

    original_accuracy: float
    original_balanced_accuracy: float
    original_f1: float
    original_mcc: float
    original_roc_auc: float

    evidence_accuracy: float
    evidence_balanced_accuracy: float
    evidence_f1: float
    evidence_mcc: float
    evidence_roc_auc: float

    quality_accuracy: float
    quality_balanced_accuracy: float
    quality_f1: float
    quality_mcc: float
    quality_roc_auc: float

    unweighted_accuracy: float
    unweighted_balanced_accuracy: float
    unweighted_f1: float
    unweighted_mcc: float
    unweighted_roc_auc: float

    ferf_accuracy: float
    ferf_balanced_accuracy: float
    ferf_precision: float
    ferf_recall: float
    ferf_f1: float
    ferf_mcc: float
    ferf_roc_auc: float
    ferf_average_precision: float

    delta_accuracy_vs_original: float
    delta_balanced_accuracy_vs_original: float
    delta_f1_vs_original: float
    delta_mcc_vs_original: float
    delta_roc_auc_vs_original: float

    quality_training_seconds: float
    total_test_inference_seconds: float
    latency_ms_per_sample: float
    throughput_samples_per_second: float

    result_directory: str
    remarks: str


FERF_RESULTS: list[FerfResult] = []


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def normalize_column_name(
    column: object,
) -> str:
    text = str(column).strip().lower()
    text = re.sub(
        r"[^a-z0-9_]+",
        "_",
        text,
    )
    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def is_identifier_column(
    column: str,
) -> bool:
    normalized = normalize_column_name(column)

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in IDENTIFIER_PATTERNS
    )


def confidence_from_probability(
    probability: np.ndarray,
) -> np.ndarray:
    return np.clip(
        2.0 * np.abs(probability - 0.5),
        0.0,
        1.0,
    )


# ============================================================
# Dataset discovery and loading
# ============================================================

def discover_cleaned_parts(
    dataset: str,
) -> list[Path]:
    cleaned_directory = (
        CLEANING_ROOT
        / dataset
        / "Cleaned_Data"
    )

    if not cleaned_directory.exists():
        return []

    parquet_parts = sorted(
        cleaned_directory.glob(
            "cleaned_part_*.parquet"
        )
    )

    if parquet_parts:
        return parquet_parts

    return sorted(
        cleaned_directory.glob(
            "cleaned_part_*.csv.gz"
        )
    )


def read_cleaned_part(
    path: Path,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )


def count_part_rows(
    path: Path,
) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(
            pq.ParquetFile(
                path
            ).metadata.num_rows
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

    if (
        requested_rows <= 0
        or requested_rows >= total_rows
    ):
        return part_sizes.copy()

    exact_allocations = [
        requested_rows * size / total_rows
        for size in part_sizes
    ]

    allocations = [
        min(
            size,
            int(np.floor(exact)),
        )
        for size, exact in zip(
            part_sizes,
            exact_allocations,
        )
    ]

    remainder = (
        requested_rows
        - sum(allocations)
    )

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


def load_dataset(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int]:
    parts = discover_cleaned_parts(dataset)

    if not parts:
        raise FileNotFoundError(
            f"No cleaned parts were found for {dataset}."
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
        else min(
            maximum_rows,
            total_available,
        )
    )

    allocations = proportional_allocation(
        part_sizes=part_sizes,
        requested_rows=requested_rows,
    )

    frames: list[pd.DataFrame] = []

    for index, (
        path,
        part_size,
        selected_rows,
    ) in enumerate(
        zip(
            parts,
            part_sizes,
            allocations,
        ),
        start=1,
    ):
        if selected_rows <= 0:
            continue

        frame = read_cleaned_part(path)

        if selected_rows < part_size:
            frame = frame.sample(
                n=selected_rows,
                random_state=(
                    random_seed + index
                ),
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
            f"No records were loaded for {dataset}."
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
        "%s: loaded %d of %d available records.",
        dataset,
        len(dataframe),
        total_available,
    )

    return dataframe, total_available


# ============================================================
# Readiness verification
# ============================================================

def load_cleaning_summary() -> pd.DataFrame:
    if not CLEANING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Cleaning summary was not found: "
            f"{CLEANING_SUMMARY_PATH}"
        )

    summary = pd.read_csv(
        CLEANING_SUMMARY_PATH
    )

    summary["dataset"] = (
        summary["dataset"].astype(str)
    )

    return summary


def get_model_paths(
    dataset: str,
) -> tuple[Path, Path]:
    original_model_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Model"
        / "original_feature_xgboost_pipeline.pkl"
    )

    evidence_model_path = (
        EXPERIMENT_22_ROOT
        / dataset
        / "Model"
        / "forensic_evidence_xgboost_pipeline.pkl"
    )

    return (
        original_model_path,
        evidence_model_path,
    )


def check_dataset_readiness(
    dataset: str,
    cleaning_summary: pd.DataFrame,
) -> tuple[bool, str]:
    matching = cleaning_summary[
        cleaning_summary["dataset"] == dataset
    ]

    if matching.empty:
        return (
            False,
            "No cleaning summary was found.",
        )

    row = matching.iloc[-1]

    status = str(
        row.get("status", "")
    ).upper()

    if status != "PASS":
        return (
            False,
            f"Cleaning status is {status}.",
        )

    input_rows = int(
        row.get("input_rows", 0)
    )

    output_rows = int(
        row.get("output_rows", 0)
    )

    if input_rows <= 0 or output_rows <= 0:
        return (
            False,
            "No usable records were produced.",
        )

    retention_ratio = (
        output_rows / input_rows
    )

    if (
        retention_ratio
        < MINIMUM_RETENTION_RATIO
    ):
        return (
            False,
            f"Retention ratio {retention_ratio:.4f} "
            f"is below {MINIMUM_RETENTION_RATIO:.2f}.",
        )

    split_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Splits"
        / "fixed_split_assignments.csv"
    )

    if not split_path.exists():
        return (
            False,
            "Experiment 2.1 split assignments are missing.",
        )

    (
        original_model_path,
        evidence_model_path,
    ) = get_model_paths(dataset)

    if not original_model_path.exists():
        return (
            False,
            "Experiment 2.1 original model is missing.",
        )

    if not evidence_model_path.exists():
        return (
            False,
            "Experiment 2.2 evidence model is missing.",
        )

    return (
        True,
        f"Ready; retention ratio={retention_ratio:.4f}.",
    )


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    target = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    valid_mask = target.isin([0, 1])

    dataframe = dataframe.loc[
        valid_mask
    ].copy()

    target = target.loc[
        valid_mask
    ].astype("int8")

    excluded_columns = set(
        NON_PREDICTIVE_COLUMNS
    )

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

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(
            dropna=True
        ) <= 1
    ]

    columns_to_remove = sorted(
        set(
            all_missing_columns
            + constant_columns
        )
    )

    if columns_to_remove:
        features = features.drop(
            columns=columns_to_remove
        )

    if features.empty:
        raise ValueError(
            "No predictive features remained."
        )

    return features, target


# ============================================================
# Fixed split reuse
# ============================================================

def load_split_assignments(
    dataset: str,
    expected_rows: int,
) -> pd.DataFrame:
    split_path = (
        EXPERIMENT_21_ROOT
        / dataset
        / "Splits"
        / "fixed_split_assignments.csv"
    )

    assignments = pd.read_csv(
        split_path
    )

    required_columns = {
        "row_position",
        "split",
    }

    if not required_columns.issubset(
        assignments.columns
    ):
        raise ValueError(
            "The split-assignment file is invalid."
        )

    assignments["row_position"] = pd.to_numeric(
        assignments["row_position"],
        errors="raise",
    ).astype(int)

    if len(assignments) != expected_rows:
        raise ValueError(
            "The reconstructed sample does not match "
            "Experiment 2.1. "
            f"Rows={expected_rows}, "
            f"assignments={len(assignments)}."
        )

    expected_positions = set(
        range(expected_rows)
    )

    observed_positions = set(
        assignments[
            "row_position"
        ].tolist()
    )

    if (
        observed_positions
        != expected_positions
    ):
        raise ValueError(
            "Split row positions are inconsistent."
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
    split_by_position = (
        assignments
        .set_index("row_position")["split"]
    )

    train_positions = (
        split_by_position[
            split_by_position == "train"
        ]
        .index
        .to_numpy()
    )

    validation_positions = (
        split_by_position[
            split_by_position == "validation"
        ]
        .index
        .to_numpy()
    )

    test_positions = (
        split_by_position[
            split_by_position == "test"
        ]
        .index
        .to_numpy()
    )

    return (
        features.iloc[
            train_positions
        ].copy(),
        features.iloc[
            validation_positions
        ].copy(),
        features.iloc[
            test_positions
        ].copy(),
        target.iloc[
            train_positions
        ].copy(),
        target.iloc[
            validation_positions
        ].copy(),
        target.iloc[
            test_positions
        ].copy(),
    )


# ============================================================
# Quality-conditioned branch
# ============================================================

def create_quality_model(
    X_train: pd.DataFrame,
    random_seed: int,
) -> Pipeline:
    numeric_columns = list(
        X_train.select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        ).columns
    )

    if not numeric_columns:
        raise ValueError(
            "No numeric features are available "
            "for the quality branch."
        )

    quality_pipeline = Pipeline(
        steps=[
            (
                "quality",
                QualityFeatureTransformer(),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "quality_branch",
                quality_pipeline,
                numeric_columns,
            ),
        ],
        remainder="drop",
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install it using: "
            "pip install xgboost"
        ) from exc

    classifier = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_seed,
    )

    return Pipeline(
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


# ============================================================
# Input-quality extraction
# ============================================================

def calculate_input_quality(
    quality_pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    preprocessing = quality_pipeline.named_steps[
        "preprocessing"
    ]

    transformed = preprocessing.transform(X)

    if hasattr(
        transformed,
        "toarray",
    ):
        transformed = transformed.toarray()

    names = (
        preprocessing.get_feature_names_out()
    )

    quality_frame = pd.DataFrame(
        transformed,
        columns=names,
    )

    def locate(
        suffix: str,
    ) -> np.ndarray:
        matching_columns = [
            column
            for column in quality_frame.columns
            if column.endswith(suffix)
        ]

        if not matching_columns:
            raise ValueError(
                "Required quality descriptor "
                f"was not found: {suffix}"
            )

        return (
            quality_frame[
                matching_columns[0]
            ]
            .to_numpy()
        )

    observed_fraction = locate(
        "quality__observed_fraction"
    )

    mean_reliability = locate(
        "quality__mean_reliability"
    )

    reliable_fraction = locate(
        "quality__reliable_fraction"
    )

    quality = (
        0.50 * observed_fraction
        + 0.30 * mean_reliability
        + 0.20 * reliable_fraction
    )

    return np.clip(
        quality,
        0.0,
        1.0,
    )


# ============================================================
# Reliability calculation
# ============================================================

def calculate_branch_reliabilities(
    original_probability: np.ndarray,
    evidence_probability: np.ndarray,
    quality_probability: np.ndarray,
    input_quality: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    original_confidence = (
        confidence_from_probability(
            original_probability
        )
    )

    evidence_confidence = (
        confidence_from_probability(
            evidence_probability
        )
    )

    quality_confidence = (
        confidence_from_probability(
            quality_probability
        )
    )

    original_evidence_agreement = (
        1.0
        - np.abs(
            original_probability
            - evidence_probability
        )
    )

    evidence_quality_agreement = (
        1.0
        - np.abs(
            evidence_probability
            - quality_probability
        )
    )

    original_reliability = np.clip(
        0.55 * original_confidence
        + 0.25 * input_quality
        + 0.20 * original_evidence_agreement,
        0.05,
        1.0,
    )

    evidence_reliability = np.clip(
        0.50 * evidence_confidence
        + 0.30 * input_quality
        + 0.20 * original_evidence_agreement,
        0.05,
        1.0,
    )

    quality_reliability = np.clip(
        0.45 * quality_confidence
        + 0.35 * input_quality
        + 0.20 * evidence_quality_agreement,
        0.05,
        1.0,
    )

    return (
        original_reliability,
        evidence_reliability,
        quality_reliability,
    )


# ============================================================
# FERF fusion
# ============================================================

def ferf_probability(
    probabilities: np.ndarray,
    reliabilities: np.ndarray,
    global_weights: np.ndarray,
) -> np.ndarray:
    weighted_reliabilities = (
        reliabilities
        * global_weights.reshape(
            1,
            -1,
        )
    )

    numerator = np.sum(
        weighted_reliabilities
        * probabilities,
        axis=1,
    )

    denominator = np.sum(
        weighted_reliabilities,
        axis=1,
    )

    return numerator / np.maximum(
        denominator,
        EPSILON,
    )


def generate_weight_candidates() -> list[np.ndarray]:
    candidates: list[np.ndarray] = []

    for original_weight in WEIGHT_VALUES:
        for evidence_weight in WEIGHT_VALUES:
            quality_weight = (
                1.0
                - original_weight
                - evidence_weight
            )

            if quality_weight < -1e-9:
                continue

            quality_weight = max(
                0.0,
                round(
                    quality_weight,
                    10,
                ),
            )

            candidate = np.asarray(
                [
                    original_weight,
                    evidence_weight,
                    quality_weight,
                ],
                dtype=float,
            )

            if np.isclose(
                candidate.sum(),
                1.0,
            ):
                candidates.append(candidate)

    return candidates


def optimize_ferf(
    validation_target: pd.Series,
    validation_probabilities: np.ndarray,
    validation_reliabilities: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    pd.DataFrame,
]:
    search_records: list[
        dict[str, float]
    ] = []

    best_key: tuple[
        float,
        ...
    ] | None = None

    best_weights: np.ndarray | None = None
    best_threshold = 0.50

    for weights in generate_weight_candidates():
        fused_probability = ferf_probability(
            probabilities=(
                validation_probabilities
            ),
            reliabilities=(
                validation_reliabilities
            ),
            global_weights=weights,
        )

        auc = roc_auc_score(
            validation_target,
            fused_probability,
        )

        for threshold in THRESHOLD_VALUES:
            predictions = (
                fused_probability
                >= threshold
            ).astype(np.int8)

            balanced_accuracy = (
                balanced_accuracy_score(
                    validation_target,
                    predictions,
                )
            )

            f1 = f1_score(
                validation_target,
                predictions,
                zero_division=0,
            )

            mcc = matthews_corrcoef(
                validation_target,
                predictions,
            )

            search_records.append(
                {
                    "original_weight": (
                        float(weights[0])
                    ),
                    "evidence_weight": (
                        float(weights[1])
                    ),
                    "quality_weight": (
                        float(weights[2])
                    ),
                    "threshold": (
                        float(threshold)
                    ),
                    "balanced_accuracy": (
                        float(
                            balanced_accuracy
                        )
                    ),
                    "f1": float(f1),
                    "mcc": float(mcc),
                    "roc_auc": float(auc),
                }
            )

            selection_key = (
                balanced_accuracy,
                f1,
                mcc,
                auc,
                -abs(threshold - 0.50),
            )

            if (
                best_key is None
                or selection_key > best_key
            ):
                best_key = selection_key
                best_weights = weights.copy()
                best_threshold = float(
                    threshold
                )

    if best_weights is None:
        raise RuntimeError(
            "FERF optimization did not produce "
            "a valid configuration."
        )

    search_frame = pd.DataFrame(
        search_records
    ).sort_values(
        by=[
            "balanced_accuracy",
            "f1",
            "mcc",
            "roc_auc",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return (
        best_weights,
        best_threshold,
        search_frame,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
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
        "roc_auc": float(
            roc_auc_score(
                target,
                probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target,
                probability,
            )
        ),
    }


def save_confusion_matrix(
    target: pd.Series,
    prediction: np.ndarray,
    path: Path,
) -> None:
    matrix = confusion_matrix(
        target,
        prediction,
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
    ).reset_index(
        names="true_class"
    )

    save_dataframe(
        frame,
        path,
    )


# ============================================================
# Figures
# ============================================================

def plot_roc_curves(
    target: pd.Series,
    probability_map: dict[
        str,
        np.ndarray,
    ],
    path: Path,
) -> None:
    plt.figure(
        figsize=(8, 7)
    )

    for name, probability in (
        probability_map.items()
    ):
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                target,
                probability,
            )
        )

        auc = roc_auc_score(
            target,
            probability,
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{name}: {auc:.6f}"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "FERF and Component ROC Curves"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_precision_recall_curves(
    target: pd.Series,
    probability_map: dict[
        str,
        np.ndarray,
    ],
    path: Path,
) -> None:
    plt.figure(
        figsize=(8, 7)
    )

    for name, probability in (
        probability_map.items()
    ):
        precision, recall, _ = (
            precision_recall_curve(
                target,
                probability,
            )
        )

        average_precision = (
            average_precision_score(
                target,
                probability,
            )
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{name}: "
                f"{average_precision:.6f}"
            ),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "FERF and Component "
        "Precision–Recall Curves"
    )

    plt.legend(
        loc="lower left"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Pickle loading
# ============================================================

def load_pickle(
    path: Path,
) -> Any:
    """
    ForensicEvidenceTransformer is already defined above, allowing
    pickle to resolve __main__.ForensicEvidenceTransformer.
    """

    with path.open(
        "rb"
    ) as stream:
        return pickle.load(stream)


# ============================================================
# Experiment execution
# ============================================================

def run_ferf_experiment(
    dataset: str,
    maximum_rows: int,
    random_seed: int,
) -> None:
    dataset_start = time.perf_counter()

    dataset_root = (
        RESULTS_ROOT / dataset
    )

    metrics_dir = (
        dataset_root / "Metrics"
    )

    predictions_dir = (
        dataset_root / "Predictions"
    )

    figures_dir = (
        dataset_root / "Figures"
    )

    model_dir = (
        dataset_root / "Model"
    )

    reliability_dir = (
        dataset_root / "Reliability"
    )

    manifests_dir = (
        dataset_root / "Manifests"
    )

    for directory in (
        metrics_dir,
        predictions_dir,
        figures_dir,
        model_dir,
        reliability_dir,
        manifests_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Experiment 3 FERF dataset: %s",
        dataset,
    )

    dataframe, total_available = load_dataset(
        dataset=dataset,
        maximum_rows=maximum_rows,
        random_seed=random_seed,
    )

    features, target = prepare_features(
        dataframe
    )

    assignments = load_split_assignments(
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
        original_model_path,
        evidence_model_path,
    ) = get_model_paths(dataset)

    LOGGER.info(
        "%s: loading Experiment 2.1 model.",
        dataset,
    )

    original_model = load_pickle(
        original_model_path
    )

    LOGGER.info(
        "%s: loading Experiment 2.2 evidence model.",
        dataset,
    )

    evidence_model = load_pickle(
        evidence_model_path
    )

    LOGGER.info(
        "%s: Experiment 2.2 model loaded successfully.",
        dataset,
    )

    quality_model = create_quality_model(
        X_train=X_train,
        random_seed=random_seed,
    )

    LOGGER.info(
        "%s: training quality-conditioned branch.",
        dataset,
    )

    quality_training_start = (
        time.perf_counter()
    )

    quality_model.fit(
        X_train,
        y_train,
    )

    quality_training_seconds = (
        time.perf_counter()
        - quality_training_start
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_original_probability = (
        original_model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_evidence_probability = (
        evidence_model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_quality_probability = (
        quality_model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_input_quality = (
        calculate_input_quality(
            quality_pipeline=quality_model,
            X=X_validation,
        )
    )

    (
        validation_original_reliability,
        validation_evidence_reliability,
        validation_quality_reliability,
    ) = calculate_branch_reliabilities(
        original_probability=(
            validation_original_probability
        ),
        evidence_probability=(
            validation_evidence_probability
        ),
        quality_probability=(
            validation_quality_probability
        ),
        input_quality=(
            validation_input_quality
        ),
    )

    validation_probability_matrix = (
        np.column_stack(
            [
                validation_original_probability,
                validation_evidence_probability,
                validation_quality_probability,
            ]
        )
    )

    validation_reliability_matrix = (
        np.column_stack(
            [
                validation_original_reliability,
                validation_evidence_reliability,
                validation_quality_reliability,
            ]
        )
    )

    LOGGER.info(
        "%s: optimizing FERF weights and threshold "
        "on validation records only.",
        dataset,
    )

    (
        optimized_weights,
        optimized_threshold,
        search_frame,
    ) = optimize_ferf(
        validation_target=y_validation,
        validation_probabilities=(
            validation_probability_matrix
        ),
        validation_reliabilities=(
            validation_reliability_matrix
        ),
    )

    save_dataframe(
        search_frame,
        metrics_dir
        / "validation_weight_threshold_search.csv",
    )

    validation_ferf_probability = (
        ferf_probability(
            probabilities=(
                validation_probability_matrix
            ),
            reliabilities=(
                validation_reliability_matrix
            ),
            global_weights=(
                optimized_weights
            ),
        )
    )

    validation_ferf_prediction = (
        validation_ferf_probability
        >= optimized_threshold
    ).astype(np.int8)

    validation_ferf_metrics = (
        calculate_metrics(
            target=y_validation,
            prediction=(
                validation_ferf_prediction
            ),
            probability=(
                validation_ferf_probability
            ),
        )
    )

    # --------------------------------------------------------
    # Held-out test inference
    # --------------------------------------------------------

    LOGGER.info(
        "%s: beginning held-out test inference.",
        dataset,
    )

    inference_start = (
        time.perf_counter()
    )

    test_original_probability = (
        original_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_evidence_probability = (
        evidence_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_quality_probability = (
        quality_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_input_quality = (
        calculate_input_quality(
            quality_pipeline=quality_model,
            X=X_test,
        )
    )

    (
        test_original_reliability,
        test_evidence_reliability,
        test_quality_reliability,
    ) = calculate_branch_reliabilities(
        original_probability=(
            test_original_probability
        ),
        evidence_probability=(
            test_evidence_probability
        ),
        quality_probability=(
            test_quality_probability
        ),
        input_quality=(
            test_input_quality
        ),
    )

    test_probability_matrix = (
        np.column_stack(
            [
                test_original_probability,
                test_evidence_probability,
                test_quality_probability,
            ]
        )
    )

    test_reliability_matrix = (
        np.column_stack(
            [
                test_original_reliability,
                test_evidence_reliability,
                test_quality_reliability,
            ]
        )
    )

    ferf_test_probability = (
        ferf_probability(
            probabilities=(
                test_probability_matrix
            ),
            reliabilities=(
                test_reliability_matrix
            ),
            global_weights=(
                optimized_weights
            ),
        )
    )

    ferf_test_prediction = (
        ferf_test_probability
        >= optimized_threshold
    ).astype(np.int8)

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # Reference methods
    # --------------------------------------------------------

    original_prediction = (
        test_original_probability
        >= 0.50
    ).astype(np.int8)

    evidence_prediction = (
        test_evidence_probability
        >= 0.50
    ).astype(np.int8)

    quality_prediction = (
        test_quality_probability
        >= 0.50
    ).astype(np.int8)

    unweighted_probability = (
        test_probability_matrix.mean(
            axis=1
        )
    )

    unweighted_prediction = (
        unweighted_probability
        >= 0.50
    ).astype(np.int8)

    original_metrics = calculate_metrics(
        target=y_test,
        prediction=original_prediction,
        probability=(
            test_original_probability
        ),
    )

    evidence_metrics = calculate_metrics(
        target=y_test,
        prediction=evidence_prediction,
        probability=(
            test_evidence_probability
        ),
    )

    quality_metrics = calculate_metrics(
        target=y_test,
        prediction=quality_prediction,
        probability=(
            test_quality_probability
        ),
    )

    unweighted_metrics = (
        calculate_metrics(
            target=y_test,
            prediction=(
                unweighted_prediction
            ),
            probability=(
                unweighted_probability
            ),
        )
    )

    ferf_metrics = calculate_metrics(
        target=y_test,
        prediction=(
            ferf_test_prediction
        ),
        probability=(
            ferf_test_probability
        ),
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_frame = pd.DataFrame(
        {
            "true_label": (
                y_test.to_numpy()
            ),
            "original_probability": (
                test_original_probability
            ),
            "original_prediction": (
                original_prediction
            ),
            "original_reliability": (
                test_original_reliability
            ),
            "evidence_probability": (
                test_evidence_probability
            ),
            "evidence_prediction": (
                evidence_prediction
            ),
            "evidence_reliability": (
                test_evidence_reliability
            ),
            "quality_probability": (
                test_quality_probability
            ),
            "quality_prediction": (
                quality_prediction
            ),
            "quality_reliability": (
                test_quality_reliability
            ),
            "input_quality": (
                test_input_quality
            ),
            "unweighted_probability": (
                unweighted_probability
            ),
            "unweighted_prediction": (
                unweighted_prediction
            ),
            "ferf_probability": (
                ferf_test_probability
            ),
            "ferf_prediction": (
                ferf_test_prediction
            ),
            "branch_probability_std": (
                test_probability_matrix.std(
                    axis=1,
                    ddof=0,
                )
            ),
            "branch_prediction_agreement": (
                (
                    original_prediction
                    == evidence_prediction
                )
                & (
                    evidence_prediction
                    == quality_prediction
                )
            ).astype(np.int8),
        }
    )

    save_dataframe(
        prediction_frame,
        predictions_dir
        / "ferf_test_predictions.csv",
    )

    validation_prediction_frame = (
        pd.DataFrame(
            {
                "true_label": (
                    y_validation.to_numpy()
                ),
                "ferf_probability": (
                    validation_ferf_probability
                ),
                "ferf_prediction": (
                    validation_ferf_prediction
                ),
            }
        )
    )

    save_dataframe(
        validation_prediction_frame,
        predictions_dir
        / "ferf_validation_predictions.csv",
    )

    # --------------------------------------------------------
    # Reliability analysis
    # --------------------------------------------------------

    reliability_columns = [
        "original_reliability",
        "evidence_reliability",
        "quality_reliability",
        "input_quality",
        "branch_probability_std",
        "branch_prediction_agreement",
    ]

    reliability_summary = (
        prediction_frame[
            reliability_columns
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .transpose()
        .reset_index(
            names="measure"
        )
    )

    save_dataframe(
        reliability_summary,
        reliability_dir
        / "reliability_summary.csv",
    )

    prediction_frame[
        "ferf_correct"
    ] = (
        prediction_frame[
            "ferf_prediction"
        ]
        == prediction_frame[
            "true_label"
        ]
    ).astype(np.int8)

    agreement_summary = (
        prediction_frame
        .groupby(
            "branch_prediction_agreement",
            as_index=False,
        )
        .agg(
            records=(
                "true_label",
                "size",
            ),
            ferf_accuracy=(
                "ferf_correct",
                "mean",
            ),
            mean_input_quality=(
                "input_quality",
                "mean",
            ),
            mean_probability_dispersion=(
                "branch_probability_std",
                "mean",
            ),
        )
    )

    save_dataframe(
        agreement_summary,
        reliability_dir
        / "agreement_disagreement_analysis.csv",
    )

    # --------------------------------------------------------
    # Metrics and comparison
    # --------------------------------------------------------

    method_metrics = pd.DataFrame(
        [
            {
                "method": (
                    "Original Features"
                ),
                **original_metrics,
            },
            {
                "method": (
                    "Forensic Evidence"
                ),
                **evidence_metrics,
            },
            {
                "method": (
                    "Quality Branch"
                ),
                **quality_metrics,
            },
            {
                "method": (
                    "Unweighted Average"
                ),
                **unweighted_metrics,
            },
            {
                "method": "FERF",
                **ferf_metrics,
            },
        ]
    )

    save_dataframe(
        method_metrics,
        metrics_dir
        / "branch_and_fusion_metrics.csv",
    )

    write_json(
        metrics_dir
        / "validation_ferf_metrics.json",
        validation_ferf_metrics,
    )

    write_json(
        metrics_dir
        / "test_ferf_metrics.json",
        ferf_metrics,
    )

    write_json(
        metrics_dir
        / "optimized_ferf_configuration.json",
        {
            "selected_using": (
                "validation subset only"
            ),
            "original_weight": float(
                optimized_weights[0]
            ),
            "evidence_weight": float(
                optimized_weights[1]
            ),
            "quality_weight": float(
                optimized_weights[2]
            ),
            "decision_threshold": float(
                optimized_threshold
            ),
            "validation_metrics": (
                validation_ferf_metrics
            ),
        },
    )

    write_json(
        metrics_dir
        / "ferf_test_classification_report.json",
        classification_report(
            y_test,
            ferf_test_prediction,
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
        target=y_test,
        prediction=ferf_test_prediction,
        path=(
            metrics_dir
            / "ferf_test_confusion_matrix.csv"
        ),
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    probability_map = {
        "Original": (
            test_original_probability
        ),
        "Evidence": (
            test_evidence_probability
        ),
        "Quality": (
            test_quality_probability
        ),
        "Unweighted": (
            unweighted_probability
        ),
        "FERF": (
            ferf_test_probability
        ),
    }

    plot_roc_curves(
        target=y_test,
        probability_map=(
            probability_map
        ),
        path=(
            figures_dir
            / "ferf_component_roc_curves.png"
        ),
    )

    plot_precision_recall_curves(
        target=y_test,
        probability_map=(
            probability_map
        ),
        path=(
            figures_dir
            / "ferf_component_precision_recall_curves.png"
        ),
    )

    # --------------------------------------------------------
    # Save quality branch
    # --------------------------------------------------------

    with (
        model_dir
        / "quality_conditioned_pipeline.pkl"
    ).open("wb") as stream:
        pickle.dump(
            quality_model,
            stream,
        )

    write_json(
        manifests_dir
        / "ferf_manifest.json",
        {
            "dataset": dataset,
            "generated_utc": utc_now(),
            "sample_size": len(features),
            "split_source": str(
                EXPERIMENT_21_ROOT
                / dataset
                / "Splits"
                / "fixed_split_assignments.csv"
            ),
            "original_model": str(
                original_model_path
            ),
            "evidence_model": str(
                evidence_model_path
            ),
            "quality_model": str(
                model_dir
                / "quality_conditioned_pipeline.pkl"
            ),
            "branches": [
                "original-feature XGBoost",
                "forensic-evidence XGBoost",
                "quality-conditioned XGBoost",
            ],
            "fusion_formula": (
                "sum(global_weight * record_reliability "
                "* probability) / "
                "sum(global_weight * record_reliability)"
            ),
            "reliability_inputs": [
                "branch confidence",
                "input quality",
                "cross-branch probability agreement",
            ],
            "weight_selection": (
                "validation-only grid search"
            ),
            "threshold_selection": (
                "validation-only grid search"
            ),
            "held_out_test_use": (
                "single final evaluation"
            ),
        },
    )

    latency_ms = (
        inference_seconds
        * 1000
        / max(
            len(X_test),
            1,
        )
    )

    throughput = (
        len(X_test)
        / max(
            inference_seconds,
            EPSILON,
        )
    )

    result = FerfResult(
        dataset=dataset,
        status="PASS",

        total_available_rows=(
            total_available
        ),
        total_rows_used=len(features),
        training_rows=len(X_train),
        validation_rows=(
            len(X_validation)
        ),
        testing_rows=len(X_test),

        original_weight=float(
            optimized_weights[0]
        ),
        evidence_weight=float(
            optimized_weights[1]
        ),
        quality_weight=float(
            optimized_weights[2]
        ),
        optimized_threshold=float(
            optimized_threshold
        ),

        original_accuracy=(
            original_metrics[
                "accuracy"
            ]
        ),
        original_balanced_accuracy=(
            original_metrics[
                "balanced_accuracy"
            ]
        ),
        original_f1=(
            original_metrics["f1"]
        ),
        original_mcc=(
            original_metrics["mcc"]
        ),
        original_roc_auc=(
            original_metrics[
                "roc_auc"
            ]
        ),

        evidence_accuracy=(
            evidence_metrics[
                "accuracy"
            ]
        ),
        evidence_balanced_accuracy=(
            evidence_metrics[
                "balanced_accuracy"
            ]
        ),
        evidence_f1=(
            evidence_metrics["f1"]
        ),
        evidence_mcc=(
            evidence_metrics["mcc"]
        ),
        evidence_roc_auc=(
            evidence_metrics[
                "roc_auc"
            ]
        ),

        quality_accuracy=(
            quality_metrics[
                "accuracy"
            ]
        ),
        quality_balanced_accuracy=(
            quality_metrics[
                "balanced_accuracy"
            ]
        ),
        quality_f1=(
            quality_metrics["f1"]
        ),
        quality_mcc=(
            quality_metrics["mcc"]
        ),
        quality_roc_auc=(
            quality_metrics[
                "roc_auc"
            ]
        ),

        unweighted_accuracy=(
            unweighted_metrics[
                "accuracy"
            ]
        ),
        unweighted_balanced_accuracy=(
            unweighted_metrics[
                "balanced_accuracy"
            ]
        ),
        unweighted_f1=(
            unweighted_metrics["f1"]
        ),
        unweighted_mcc=(
            unweighted_metrics["mcc"]
        ),
        unweighted_roc_auc=(
            unweighted_metrics[
                "roc_auc"
            ]
        ),

        ferf_accuracy=(
            ferf_metrics["accuracy"]
        ),
        ferf_balanced_accuracy=(
            ferf_metrics[
                "balanced_accuracy"
            ]
        ),
        ferf_precision=(
            ferf_metrics[
                "precision"
            ]
        ),
        ferf_recall=(
            ferf_metrics["recall"]
        ),
        ferf_f1=(
            ferf_metrics["f1"]
        ),
        ferf_mcc=(
            ferf_metrics["mcc"]
        ),
        ferf_roc_auc=(
            ferf_metrics["roc_auc"]
        ),
        ferf_average_precision=(
            ferf_metrics[
                "average_precision"
            ]
        ),

        delta_accuracy_vs_original=(
            ferf_metrics["accuracy"]
            - original_metrics[
                "accuracy"
            ]
        ),
        delta_balanced_accuracy_vs_original=(
            ferf_metrics[
                "balanced_accuracy"
            ]
            - original_metrics[
                "balanced_accuracy"
            ]
        ),
        delta_f1_vs_original=(
            ferf_metrics["f1"]
            - original_metrics["f1"]
        ),
        delta_mcc_vs_original=(
            ferf_metrics["mcc"]
            - original_metrics["mcc"]
        ),
        delta_roc_auc_vs_original=(
            ferf_metrics["roc_auc"]
            - original_metrics[
                "roc_auc"
            ]
        ),

        quality_training_seconds=round(
            quality_training_seconds,
            6,
        ),
        total_test_inference_seconds=round(
            inference_seconds,
            6,
        ),
        latency_ms_per_sample=round(
            latency_ms,
            9,
        ),
        throughput_samples_per_second=round(
            throughput,
            3,
        ),

        result_directory=str(
            dataset_root
        ),
        remarks=(
            "FERF weights and decision threshold were "
            "selected using validation records only. "
            "The Experiment 2.2 custom transformer was "
            "defined locally to support pickle loading."
        ),
    )

    FERF_RESULTS.append(result)

    write_json(
        dataset_root
        / "ferf_experiment_result.json",
        asdict(result),
    )

    LOGGER.info(
        "%s | weights=(%.2f, %.2f, %.2f) | "
        "threshold=%.2f",
        dataset,
        result.original_weight,
        result.evidence_weight,
        result.quality_weight,
        result.optimized_threshold,
    )

    LOGGER.info(
        "%s | FERF accuracy=%.6f | "
        "balanced_accuracy=%.6f | "
        "F1=%.6f | MCC=%.6f | "
        "ROC-AUC=%.6f",
        dataset,
        result.ferf_accuracy,
        result.ferf_balanced_accuracy,
        result.ferf_f1,
        result.ferf_mcc,
        result.ferf_roc_auc,
    )

    LOGGER.info(
        "%s | deltas vs original: "
        "Accuracy=%+.6f | "
        "Balanced Accuracy=%+.6f | "
        "F1=%+.6f | MCC=%+.6f | "
        "ROC-AUC=%+.6f",
        dataset,
        result.delta_accuracy_vs_original,
        result.delta_balanced_accuracy_vs_original,
        result.delta_f1_vs_original,
        result.delta_mcc_vs_original,
        result.delta_roc_auc_vs_original,
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
    del original_model
    del evidence_model
    del quality_model

    gc.collect()


# ============================================================
# Consolidated reporting
# ============================================================

def save_consolidated_results() -> None:
    records = [
        asdict(result)
        for result in FERF_RESULTS
    ]

    frame = pd.DataFrame(
        records
    )

    save_dataframe(
        frame,
        REPORTS_DIR
        / "FERF_Validation_Results.csv",
    )

    write_json(
        REPORTS_DIR
        / "FERF_Validation_Results.json",
        {
            "generated_utc": utc_now(),
            "results": records,
        },
    )

    if frame.empty:
        return

    comparison_columns = [
        "dataset",

        "original_accuracy",
        "evidence_accuracy",
        "quality_accuracy",
        "unweighted_accuracy",
        "ferf_accuracy",
        "delta_accuracy_vs_original",

        "original_balanced_accuracy",
        "evidence_balanced_accuracy",
        "quality_balanced_accuracy",
        "unweighted_balanced_accuracy",
        "ferf_balanced_accuracy",
        "delta_balanced_accuracy_vs_original",

        "original_f1",
        "evidence_f1",
        "quality_f1",
        "unweighted_f1",
        "ferf_f1",
        "delta_f1_vs_original",

        "original_mcc",
        "evidence_mcc",
        "quality_mcc",
        "unweighted_mcc",
        "ferf_mcc",
        "delta_mcc_vs_original",

        "original_roc_auc",
        "evidence_roc_auc",
        "quality_roc_auc",
        "unweighted_roc_auc",
        "ferf_roc_auc",
        "delta_roc_auc_vs_original",
    ]

    save_dataframe(
        frame[
            comparison_columns
        ],
        REPORTS_DIR
        / "FERF_Method_Comparison.csv",
    )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: validate the Forensic "
            "Evidence Reliability Fusion mechanism."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(
            *DATASET_NAMES,
            "all",
        ),
        default="all",
        help=(
            "Dataset to evaluate. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "Maximum records per dataset. "
            "This must match Experiments 2.1 "
            "and 2.2. Use 0 for all records. "
            f"Default: {DEFAULT_MAX_ROWS:,}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            "Random seed used in Experiments "
            "2.1 and 2.2. "
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
        "EXPERIMENT 3: FORENSIC EVIDENCE "
        "RELIABILITY FUSION"
    )
    LOGGER.info(
        "Datasets: %s",
        ", ".join(
            selected_datasets
        ),
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
        "Split source: Experiment 2.1 "
        "saved assignments"
    )
    LOGGER.info(
        "FERF configuration source: "
        "validation subset only"
    )
    LOGGER.info(
        "Pickle compatibility fix: "
        "ForensicEvidenceTransformer "
        "defined in __main__"
    )
    LOGGER.info("=" * 78)

    cleaning_summary = (
        load_cleaning_summary()
    )

    readiness_records: list[
        dict[str, object]
    ] = []

    runnable_datasets: list[str] = []

    for dataset in selected_datasets:
        ready, reason = (
            check_dataset_readiness(
                dataset=dataset,
                cleaning_summary=(
                    cleaning_summary
                ),
            )
        )

        readiness_records.append(
            {
                "dataset": dataset,
                "ready": ready,
                "reason": reason,
            }
        )

        if ready:
            runnable_datasets.append(
                dataset
            )

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
        / "experiment_03_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "experiment": (
                "Experiment 3: FERF Validation"
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
            "branches": [
                "original-feature XGBoost",
                "forensic-evidence XGBoost",
                "quality-conditioned XGBoost",
            ],
            "fusion": (
                "record-specific reliability-weighted "
                "probability fusion"
            ),
            "weight_selection": (
                "validation-only grid search"
            ),
            "threshold_selection": (
                "validation-only grid search"
            ),
            "pickle_fix": (
                "ForensicEvidenceTransformer defined "
                "locally before loading Experiment 2.2 model"
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
            run_ferf_experiment(
                dataset=dataset,
                maximum_rows=(
                    args.max_rows
                ),
                random_seed=args.seed,
            )

        except Exception:
            failed_datasets.append(
                dataset
            )

            LOGGER.exception(
                "FERF validation failed "
                "for %s.",
                dataset,
            )

    save_consolidated_results()

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Successful dataset runs: %d",
        len(FERF_RESULTS),
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

    if not FERF_RESULTS:
        return 1

    if failed_datasets:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())