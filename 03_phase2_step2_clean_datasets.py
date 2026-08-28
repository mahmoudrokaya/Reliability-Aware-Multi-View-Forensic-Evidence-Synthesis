"""
03_phase2_step2_clean_datasets.py

Experiment 01 - Data Preparation
Phase 02 - Integrity Verification and Cleaning
Step 02 - Dataset Cleaning and Standardization

Supported datasets
------------------
1. CICIDS2017
2. CSE-CIC-IDS2018
3. UNSW-NB15
4. BoT-IoT

Main operations
---------------
- Uses canonical dataset locations only.
- Avoids duplicate dataset copies.
- Reads large CSV files in chunks.
- Standardizes column names.
- Removes empty rows and columns.
- Replaces positive/negative infinity with missing values.
- Removes malformed and exact duplicate rows.
- Strips spaces and invalid characters from labels.
- Produces unified binary and multiclass target columns.
- Preserves source-file and source-row metadata.
- Saves each dataset in an independent output directory.
- Generates cleaning, schema, and label-distribution reports.
- Skips BoT-IoT safely when the correct tabular dataset is unavailable.

The script does not normalize features. Scaling must be fitted only on
training data in a later leakage-safe preprocessing stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

DATA_ROOT = PROJECT_ROOT / "Data"

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Experiment_01_Data_Preparation"
    / "Phase_02_Integrity_Verification"
    / "Step_02_Data_Cleaning"
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
# Canonical dataset locations
# ============================================================

DATASET_CONFIG = {
    "CICIDS2017": {
        "input_dir": (
            DATA_ROOT
            / "CICIDS2017"
            / "Raw"
            / "CSVs"
            / "MachineLearningCVE"
        ),
        "file_patterns": ("*.csv",),
        "header_mode": "existing",
        "label_candidates": (
            "label",
            "class",
            "attack",
            "attack_cat",
            "category",
        ),
    },
    "CSE-CIC-IDS2018": {
        "input_dir": (
            DATA_ROOT
            / "CSE-CIC-IDS2018"
            / "Raw"
            / "Processed Traffic Data for ML Algorithms"
        ),
        "file_patterns": ("*.csv",),
        "header_mode": "existing",
        "label_candidates": (
            "label",
            "class",
            "attack",
            "attack_cat",
            "category",
        ),
    },
    "UNSW-NB15": {
        "input_dir": (
            DATA_ROOT
            / "UNSW-NB15"
            / "Raw"
            / "CSV Files"
        ),
        "file_patterns": (
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv",
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv",
        ),
        "header_mode": "unsw_feature_file",
        "feature_file_candidates": (
            "NUSW-NB15_features.csv",
            "UNSW-NB15_features.csv",
        ),
        "label_candidates": (
            "attack_cat",
            "label",
            "class",
            "category",
        ),
    },
    "BoT-IoT": {
        # Change this path only if the official BoT-IoT tabular files
        # are stored elsewhere.
        "input_dir": (
            DATA_ROOT
            / "BoT-IoT"
            / "Raw"
            / "Dataset"
        ),
        "alternative_input_dirs": (
            DATA_ROOT
            / "BoT-IoT"
            / "Raw"
            / "__Bot-IoT_Dataset"
            / "Dataset",
            DATA_ROOT
            / "BoT-IoT"
            / "Raw"
            / "CSV Files",
        ),
        "file_patterns": (
            "UNSW_2018_IoT_Botnet*.csv",
            "*.csv",
        ),
        "header_mode": "existing",
        "label_candidates": (
            "label",
            "attack",
            "category",
            "subcategory",
            "attack_cat",
            "class",
        ),
    },
}


DATASET_NAMES = tuple(DATASET_CONFIG.keys())


# ============================================================
# Cleaning configuration
# ============================================================

DEFAULT_CHUNK_SIZE = 200_000

MISSING_TEXT_VALUES = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "?",
    "-",
    "--",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}

BENIGN_LABELS = {
    "benign",
    "normal",
    "normaltraffic",
    "normal traffic",
    "0",
    "false",
    "background",
    "non attack",
    "non-attack",
    "no attack",
}

IDENTIFIER_COLUMN_PATTERNS = (
    r"^flow_id$",
    r"^src_ip$",
    r"^dst_ip$",
    r"^source_ip$",
    r"^destination_ip$",
)

PROTECTED_COLUMNS = {
    "binary_label",
    "multiclass_label",
    "original_label",
    "source_dataset",
    "source_file",
    "source_row",
}


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "step_02_data_cleaning.log"

LOGGER = logging.getLogger("step_02_data_cleaning")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
        mode="w",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


# ============================================================
# Data structures
# ============================================================

@dataclass
class FileCleaningSummary:
    dataset: str
    source_file: str
    input_rows: int
    output_rows: int
    empty_rows_removed: int
    duplicate_rows_removed: int
    invalid_label_rows_removed: int
    rows_with_missing_values: int
    infinite_values_replaced: int
    malformed_rows_skipped: int
    input_columns: int
    output_columns: int
    elapsed_seconds: float
    status: str
    remarks: str


@dataclass
class DatasetCleaningSummary:
    dataset: str
    status: str
    source_files: int
    input_rows: int
    output_rows: int
    removed_rows: int
    duplicate_rows_removed: int
    invalid_label_rows_removed: int
    infinite_values_replaced: int
    output_parts: int
    output_columns: int
    benign_rows: int
    attack_rows: int
    elapsed_seconds: float
    output_directory: str
    remarks: str


FILE_SUMMARIES: list[FileCleaningSummary] = []
DATASET_SUMMARIES: list[DatasetCleaningSummary] = []


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def normalize_column_name(name: object) -> str:
    text = str(name).strip()
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\s/\\\-().:%]+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def make_unique_columns(columns: Iterable[object]) -> list[str]:
    counts: Counter[str] = Counter()
    unique_columns: list[str] = []

    for raw_name in columns:
        normalized = normalize_column_name(raw_name)

        if not normalized:
            normalized = "unnamed_column"

        counts[normalized] += 1

        if counts[normalized] == 1:
            unique_columns.append(normalized)
        else:
            unique_columns.append(
                f"{normalized}_{counts[normalized]}"
            )

    return unique_columns


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    text = text.replace("\ufeff", "")
    text = text.replace("�", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" _-")

    if not text or text in MISSING_TEXT_VALUES:
        return None

    label_replacements = {
        "ddos attack-hoic": "ddos_hoic",
        "ddos attack-loic-udp": "ddos_loic_udp",
        "ddos attacks-loic-http": "ddos_loic_http",
        "do s": "dos",
        "portscan": "port_scan",
        "web attack brute force": "web_attack_brute_force",
        "web attack xss": "web_attack_xss",
        "web attack sql injection": "web_attack_sql_injection",
        "infilteration": "infiltration",
        "backdoor": "backdoors",
        "normal": "benign",
    }

    text = label_replacements.get(text, text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or None


def binary_from_label(label: object) -> int | None:
    normalized = normalize_label(label)

    if normalized is None:
        return None

    benign_normalized = {
        normalize_label(item)
        for item in BENIGN_LABELS
    }

    return 0 if normalized in benign_normalized else 1


def detect_label_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    normalized_candidates = {
        normalize_column_name(candidate)
        for candidate in candidates
    }

    for column in columns:
        if normalize_column_name(column) in normalized_candidates:
            return column

    # Prefer common variants if exact candidates were not found.
    fallback_patterns = (
        "label",
        "attack_cat",
        "attack_category",
        "category",
        "class",
    )

    for fallback in fallback_patterns:
        if fallback in columns:
            return fallback

    return None


def resolve_dataset_input_dir(dataset: str) -> Path | None:
    config = DATASET_CONFIG[dataset]
    candidates = [config["input_dir"]]

    candidates.extend(
        config.get("alternative_input_dirs", ())
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def discover_source_files(
    dataset: str,
    input_dir: Path,
) -> list[Path]:
    config = DATASET_CONFIG[dataset]
    found: dict[str, Path] = {}

    for pattern in config["file_patterns"]:
        for file_path in input_dir.rglob(pattern):
            if not file_path.is_file():
                continue

            lower_name = file_path.name.lower()

            if lower_name.endswith("_error.txt"):
                continue

            if "feature" in lower_name and dataset != "BoT-IoT":
                continue

            if "list_events" in lower_name:
                continue

            if "training-set" in lower_name:
                continue

            if "testing-set" in lower_name:
                continue

            found[str(file_path.resolve()).lower()] = file_path

    files = sorted(
        found.values(),
        key=lambda path: str(path).lower(),
    )

    # Protect against mistakenly using UNSW-NB15 as BoT-IoT.
    if dataset == "BoT-IoT":
        files = [
            file_path
            for file_path in files
            if (
                "iot_botnet" in file_path.name.lower()
                or "bot-iot" in str(file_path).lower()
                or "bot_iot" in str(file_path).lower()
            )
        ]

    return files


def calculate_file_sha256(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# UNSW-NB15 schema loading
# ============================================================

def find_unsw_feature_file(input_dir: Path) -> Path | None:
    config = DATASET_CONFIG["UNSW-NB15"]

    for expected_name in config["feature_file_candidates"]:
        matches = list(input_dir.rglob(expected_name))

        if matches:
            return matches[0]

    return None


def load_unsw_feature_names(input_dir: Path) -> list[str]:
    feature_file = find_unsw_feature_file(input_dir)

    if feature_file is None:
        raise FileNotFoundError(
            "UNSW-NB15 feature-description file was not found."
        )

    feature_df = pd.read_csv(
        feature_file,
        encoding_errors="replace",
    )

    feature_df.columns = make_unique_columns(feature_df.columns)

    possible_name_columns = (
        "name",
        "feature",
        "feature_name",
        "features",
    )

    name_column = next(
        (
            column
            for column in possible_name_columns
            if column in feature_df.columns
        ),
        None,
    )

    if name_column is None:
        # The feature name is commonly the second column.
        if feature_df.shape[1] < 2:
            raise ValueError(
                "Unable to identify UNSW-NB15 feature names."
            )

        name_column = feature_df.columns[1]

    feature_names = [
        normalize_column_name(value)
        for value in feature_df[name_column].tolist()
        if pd.notna(value)
    ]

    feature_names = [
        name
        for name in feature_names
        if name
    ]

    if not feature_names:
        raise ValueError(
            "No valid UNSW-NB15 feature names were loaded."
        )

    LOGGER.info(
        "Loaded %d UNSW-NB15 feature names from %s.",
        len(feature_names),
        feature_file,
    )

    return make_unique_columns(feature_names)


# ============================================================
# CSV reading
# ============================================================

def count_malformed_rows_estimate(
    file_path: Path,
) -> int:
    """
    Pandas skips malformed lines when on_bad_lines='skip'.
    Exact skipped-line counting is parser-dependent; this function
    currently returns zero and the limitation is reported explicitly.
    """
    return 0


def read_csv_chunks(
    dataset: str,
    file_path: Path,
    chunk_size: int,
    unsw_columns: list[str] | None,
) -> Iterator[pd.DataFrame]:
    common_options = {
        "chunksize": chunk_size,
        "low_memory": False,
        "encoding_errors": "replace",
        "on_bad_lines": "skip",
    }

    if dataset == "UNSW-NB15":
        if not unsw_columns:
            raise ValueError(
                "UNSW-NB15 requires feature names."
            )

        yield from pd.read_csv(
            file_path,
            header=None,
            names=unsw_columns,
            **common_options,
        )
    else:
        try:
            yield from pd.read_csv(
                file_path,
                **common_options,
            )
        except UnicodeDecodeError:
            yield from pd.read_csv(
                file_path,
                encoding="latin-1",
                **common_options,
            )


# ============================================================
# Global deduplication
# ============================================================

class GlobalDuplicateTracker:
    """
    Disk-backed duplicate tracker.

    Row hashes are stored in SQLite so exact duplicate rows can be
    removed across chunks and across files without keeping all hashes
    in RAM.
    """

    def __init__(
        self,
        database_path: Path,
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self.connection: sqlite3.Connection | None = None

        if not enabled:
            return

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if database_path.exists():
            database_path.unlink()

        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            """
            CREATE TABLE row_hashes (
                row_hash TEXT PRIMARY KEY
            )
            """
        )
        self.connection.commit()

    def keep_unique(
        self,
        dataframe: pd.DataFrame,
        hash_columns: list[str],
    ) -> tuple[pd.DataFrame, int]:
        if dataframe.empty:
            return dataframe, 0

        if not self.enabled:
            before = len(dataframe)
            dataframe = dataframe.drop_duplicates(
                subset=hash_columns,
                keep="first",
            )
            return dataframe, before - len(dataframe)

        assert self.connection is not None

        row_hashes = pd.util.hash_pandas_object(
            dataframe[hash_columns],
            index=False,
        ).astype(str)

        keep_mask: list[bool] = []
        duplicate_count = 0

        cursor = self.connection.cursor()

        for row_hash in row_hashes:
            try:
                cursor.execute(
                    "INSERT INTO row_hashes(row_hash) VALUES (?)",
                    (row_hash,),
                )
                keep_mask.append(True)
            except sqlite3.IntegrityError:
                keep_mask.append(False)
                duplicate_count += 1

        self.connection.commit()

        return dataframe.loc[keep_mask].copy(), duplicate_count

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()


# ============================================================
# Chunk cleaning
# ============================================================

def clean_string_columns(dataframe: pd.DataFrame) -> None:
    string_columns = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in string_columns:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.strip()
        )

        normalized_missing = (
            dataframe[column]
            .str.lower()
            .isin(MISSING_TEXT_VALUES)
        )

        dataframe.loc[
            normalized_missing,
            column,
        ] = pd.NA


def coerce_numeric_like_columns(
    dataframe: pd.DataFrame,
    protected_columns: set[str],
    threshold: float = 0.95,
) -> None:
    """
    Converts an object column to numeric only when at least 95% of its
    non-missing values can be parsed numerically.
    """

    for column in dataframe.select_dtypes(
        include=["object", "string"]
    ).columns:
        if column in protected_columns:
            continue

        non_missing = dataframe[column].dropna()

        if non_missing.empty:
            continue

        converted = pd.to_numeric(
            non_missing,
            errors="coerce",
        )

        numeric_ratio = converted.notna().mean()

        if numeric_ratio >= threshold:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )


def clean_chunk(
    dataset: str,
    chunk: pd.DataFrame,
    source_file: Path,
    source_row_offset: int,
    label_candidates: Iterable[str],
    duplicate_tracker: GlobalDuplicateTracker,
) -> tuple[pd.DataFrame, dict[str, int], str]:
    metrics = {
        "input_rows": len(chunk),
        "empty_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "invalid_label_rows_removed": 0,
        "rows_with_missing_values": 0,
        "infinite_values_replaced": 0,
    }

    chunk.columns = make_unique_columns(chunk.columns)

    # Remove completely unnamed index-like columns.
    removable_unnamed = [
        column
        for column in chunk.columns
        if column.startswith("unnamed")
        and chunk[column].isna().all()
    ]

    if removable_unnamed:
        chunk = chunk.drop(columns=removable_unnamed)

    clean_string_columns(chunk)
    coerce_numeric_like_columns(
        chunk,
        protected_columns=PROTECTED_COLUMNS,
    )

    numeric_columns = chunk.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) > 0:
        infinite_mask = np.isinf(
            chunk[numeric_columns].to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )

        metrics["infinite_values_replaced"] = int(
            infinite_mask.sum()
        )

        chunk[numeric_columns] = chunk[
            numeric_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    empty_row_mask = chunk.isna().all(axis=1)
    metrics["empty_rows_removed"] = int(
        empty_row_mask.sum()
    )
    chunk = chunk.loc[~empty_row_mask].copy()

    label_column = detect_label_column(
        chunk.columns,
        label_candidates,
    )

    if label_column is None:
        raise ValueError(
            f"No label column was detected in {source_file.name}. "
            f"Columns: {list(chunk.columns)}"
        )

    chunk["original_label"] = (
        chunk[label_column]
        .astype("string")
        .str.strip()
    )

    chunk["multiclass_label"] = (
        chunk["original_label"]
        .map(normalize_label)
    )

    chunk["binary_label"] = (
        chunk["original_label"]
        .map(binary_from_label)
    )

    invalid_label_mask = (
        chunk["multiclass_label"].isna()
        | chunk["binary_label"].isna()
    )

    metrics["invalid_label_rows_removed"] = int(
        invalid_label_mask.sum()
    )

    chunk = chunk.loc[
        ~invalid_label_mask
    ].copy()

    chunk["binary_label"] = (
        chunk["binary_label"].astype("int8")
    )

    chunk["source_dataset"] = dataset
    chunk["source_file"] = source_file.name

    chunk["source_row"] = np.arange(
        source_row_offset,
        source_row_offset + len(chunk),
        dtype=np.int64,
    )

    # Remove the original dataset label column only when it is not one
    # of the standardized output label columns.
    if label_column not in PROTECTED_COLUMNS:
        chunk = chunk.drop(columns=[label_column])

    metrics["rows_with_missing_values"] = int(
        chunk.isna().any(axis=1).sum()
    )

    hash_columns = [
        column
        for column in chunk.columns
        if column not in {
            "source_dataset",
            "source_file",
            "source_row",
        }
    ]

    chunk, removed_duplicates = (
        duplicate_tracker.keep_unique(
            dataframe=chunk,
            hash_columns=hash_columns,
        )
    )

    metrics["duplicate_rows_removed"] = (
        removed_duplicates
    )

    # Ensure metadata appears first.
    ordered_first = [
        "source_dataset",
        "source_file",
        "source_row",
        "original_label",
        "multiclass_label",
        "binary_label",
    ]

    remaining_columns = [
        column
        for column in chunk.columns
        if column not in ordered_first
    ]

    chunk = chunk[
        ordered_first + remaining_columns
    ]

    return chunk, metrics, label_column


# ============================================================
# Output writing
# ============================================================

def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_cleaned_part(
    dataframe: pd.DataFrame,
    output_dir: Path,
    part_number: int,
    output_format: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        output_path = (
            output_dir
            / f"cleaned_part_{part_number:05d}.parquet"
        )

        dataframe.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

        return output_path

    output_path = (
        output_dir
        / f"cleaned_part_{part_number:05d}.csv.gz"
    )

    dataframe.to_csv(
        output_path,
        index=False,
        compression="gzip",
        encoding="utf-8",
    )

    return output_path


# ============================================================
# Dataset processing
# ============================================================

def clean_dataset(
    dataset: str,
    chunk_size: int,
    output_format: str,
    global_dedup: bool,
) -> None:
    dataset_start = time.perf_counter()

    LOGGER.info("=" * 78)
    LOGGER.info("Cleaning dataset: %s", dataset)

    input_dir = resolve_dataset_input_dir(dataset)

    dataset_output_root = RESULTS_ROOT / dataset
    cleaned_data_dir = dataset_output_root / "Cleaned_Data"
    dataset_report_dir = dataset_output_root / "Reports"
    dataset_manifest_dir = dataset_output_root / "Manifests"
    rejected_dir = dataset_output_root / "Rejected"

    for directory in (
        cleaned_data_dir,
        dataset_report_dir,
        dataset_manifest_dir,
        rejected_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Remove stale cleaned parts from previous runs.
    for stale_file in cleaned_data_dir.glob("cleaned_part_*"):
        stale_file.unlink()

    if input_dir is None:
        message = (
            "No valid canonical input directory was found. "
            "Dataset was skipped."
        )

        LOGGER.warning("%s: %s", dataset, message)

        DATASET_SUMMARIES.append(
            DatasetCleaningSummary(
                dataset=dataset,
                status="SKIPPED",
                source_files=0,
                input_rows=0,
                output_rows=0,
                removed_rows=0,
                duplicate_rows_removed=0,
                invalid_label_rows_removed=0,
                infinite_values_replaced=0,
                output_parts=0,
                output_columns=0,
                benign_rows=0,
                attack_rows=0,
                elapsed_seconds=round(
                    time.perf_counter() - dataset_start,
                    3,
                ),
                output_directory=str(cleaned_data_dir),
                remarks=message,
            )
        )
        return

    source_files = discover_source_files(
        dataset=dataset,
        input_dir=input_dir,
    )

    if not source_files:
        message = (
            f"No valid experiment CSV files were found under {input_dir}. "
            "Dataset was skipped."
        )

        LOGGER.warning("%s: %s", dataset, message)

        DATASET_SUMMARIES.append(
            DatasetCleaningSummary(
                dataset=dataset,
                status="SKIPPED",
                source_files=0,
                input_rows=0,
                output_rows=0,
                removed_rows=0,
                duplicate_rows_removed=0,
                invalid_label_rows_removed=0,
                infinite_values_replaced=0,
                output_parts=0,
                output_columns=0,
                benign_rows=0,
                attack_rows=0,
                elapsed_seconds=round(
                    time.perf_counter() - dataset_start,
                    3,
                ),
                output_directory=str(cleaned_data_dir),
                remarks=message,
            )
        )
        return

    LOGGER.info(
        "%s: using canonical input directory: %s",
        dataset,
        input_dir,
    )

    LOGGER.info(
        "%s: discovered %d source file(s).",
        dataset,
        len(source_files),
    )

    unsw_columns: list[str] | None = None

    if dataset == "UNSW-NB15":
        unsw_columns = load_unsw_feature_names(input_dir)

    duplicate_tracker = GlobalDuplicateTracker(
        database_path=(
            dataset_manifest_dir
            / "global_row_hashes.sqlite"
        ),
        enabled=global_dedup,
    )

    total_input_rows = 0
    total_output_rows = 0
    total_duplicates_removed = 0
    total_invalid_labels_removed = 0
    total_infinite_replaced = 0
    total_empty_removed = 0
    output_part_number = 0
    final_output_columns = 0
    label_counter: Counter[str] = Counter()
    binary_counter: Counter[int] = Counter()
    source_manifest: list[dict[str, object]] = []

    try:
        for file_index, source_file in enumerate(
            source_files,
            start=1,
        ):
            file_start = time.perf_counter()

            LOGGER.info(
                "%s: processing file %d/%d: %s",
                dataset,
                file_index,
                len(source_files),
                source_file.name,
            )

            source_manifest.append(
                {
                    "source_file": str(source_file.resolve()),
                    "size_bytes": source_file.stat().st_size,
                    "sha256": calculate_file_sha256(
                        source_file
                    ),
                }
            )

            file_input_rows = 0
            file_output_rows = 0
            file_empty_removed = 0
            file_duplicates_removed = 0
            file_invalid_labels_removed = 0
            file_missing_rows = 0
            file_infinite_replaced = 0
            source_row_offset = 0
            detected_label_column = ""

            malformed_rows_skipped = (
                count_malformed_rows_estimate(
                    source_file
                )
            )

            for chunk_index, chunk in enumerate(
                read_csv_chunks(
                    dataset=dataset,
                    file_path=source_file,
                    chunk_size=chunk_size,
                    unsw_columns=unsw_columns,
                ),
                start=1,
            ):
                original_chunk_rows = len(chunk)

                cleaned_chunk, metrics, label_column = (
                    clean_chunk(
                        dataset=dataset,
                        chunk=chunk,
                        source_file=source_file,
                        source_row_offset=source_row_offset,
                        label_candidates=DATASET_CONFIG[
                            dataset
                        ]["label_candidates"],
                        duplicate_tracker=duplicate_tracker,
                    )
                )

                detected_label_column = label_column
                source_row_offset += original_chunk_rows

                file_input_rows += metrics["input_rows"]
                file_output_rows += len(cleaned_chunk)
                file_empty_removed += metrics[
                    "empty_rows_removed"
                ]
                file_duplicates_removed += metrics[
                    "duplicate_rows_removed"
                ]
                file_invalid_labels_removed += metrics[
                    "invalid_label_rows_removed"
                ]
                file_missing_rows += metrics[
                    "rows_with_missing_values"
                ]
                file_infinite_replaced += metrics[
                    "infinite_values_replaced"
                ]

                if cleaned_chunk.empty:
                    continue

                label_counter.update(
                    cleaned_chunk[
                        "multiclass_label"
                    ].astype(str)
                )

                binary_counter.update(
                    cleaned_chunk[
                        "binary_label"
                    ].astype(int)
                )

                output_part_number += 1
                final_output_columns = len(
                    cleaned_chunk.columns
                )

                output_path = save_cleaned_part(
                    dataframe=cleaned_chunk,
                    output_dir=cleaned_data_dir,
                    part_number=output_part_number,
                    output_format=output_format,
                )

                LOGGER.info(
                    "%s | %s | chunk=%d | input=%d | output=%d | %s",
                    dataset,
                    source_file.name,
                    chunk_index,
                    metrics["input_rows"],
                    len(cleaned_chunk),
                    output_path.name,
                )

            file_elapsed = (
                time.perf_counter() - file_start
            )

            file_removed = (
                file_input_rows - file_output_rows
            )

            FILE_SUMMARIES.append(
                FileCleaningSummary(
                    dataset=dataset,
                    source_file=str(source_file),
                    input_rows=file_input_rows,
                    output_rows=file_output_rows,
                    empty_rows_removed=file_empty_removed,
                    duplicate_rows_removed=(
                        file_duplicates_removed
                    ),
                    invalid_label_rows_removed=(
                        file_invalid_labels_removed
                    ),
                    rows_with_missing_values=(
                        file_missing_rows
                    ),
                    infinite_values_replaced=(
                        file_infinite_replaced
                    ),
                    malformed_rows_skipped=(
                        malformed_rows_skipped
                    ),
                    input_columns=(
                        len(unsw_columns)
                        if dataset == "UNSW-NB15"
                        and unsw_columns
                        else final_output_columns
                    ),
                    output_columns=final_output_columns,
                    elapsed_seconds=round(
                        file_elapsed,
                        3,
                    ),
                    status="PASS",
                    remarks=(
                        f"Detected label column: "
                        f"{detected_label_column}. "
                        f"Removed rows: {file_removed}."
                    ),
                )
            )

            total_input_rows += file_input_rows
            total_output_rows += file_output_rows
            total_empty_removed += file_empty_removed
            total_duplicates_removed += (
                file_duplicates_removed
            )
            total_invalid_labels_removed += (
                file_invalid_labels_removed
            )
            total_infinite_replaced += (
                file_infinite_replaced
            )

        dataset_elapsed = (
            time.perf_counter() - dataset_start
        )

        label_distribution_path = (
            dataset_report_dir
            / "Label_Distribution.csv"
        )

        with label_distribution_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "multiclass_label",
                    "count",
                ]
            )

            for label, count in sorted(
                label_counter.items()
            ):
                writer.writerow([label, count])

        binary_distribution_path = (
            dataset_report_dir
            / "Binary_Label_Distribution.csv"
        )

        with binary_distribution_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "binary_label",
                    "meaning",
                    "count",
                ]
            )

            for binary_value in (0, 1):
                writer.writerow(
                    [
                        binary_value,
                        (
                            "benign"
                            if binary_value == 0
                            else "attack"
                        ),
                        binary_counter.get(
                            binary_value,
                            0,
                        ),
                    ]
                )

        write_json(
            dataset_manifest_dir
            / "source_file_manifest.json",
            {
                "dataset": dataset,
                "input_directory": str(input_dir),
                "generated_utc": utc_now(),
                "source_files": source_manifest,
            },
        )

        removed_rows = (
            total_input_rows - total_output_rows
        )

        DATASET_SUMMARIES.append(
            DatasetCleaningSummary(
                dataset=dataset,
                status="PASS",
                source_files=len(source_files),
                input_rows=total_input_rows,
                output_rows=total_output_rows,
                removed_rows=removed_rows,
                duplicate_rows_removed=(
                    total_duplicates_removed
                ),
                invalid_label_rows_removed=(
                    total_invalid_labels_removed
                ),
                infinite_values_replaced=(
                    total_infinite_replaced
                ),
                output_parts=output_part_number,
                output_columns=final_output_columns,
                benign_rows=binary_counter.get(0, 0),
                attack_rows=binary_counter.get(1, 0),
                elapsed_seconds=round(
                    dataset_elapsed,
                    3,
                ),
                output_directory=str(
                    cleaned_data_dir
                ),
                remarks=(
                    f"Empty rows removed: "
                    f"{total_empty_removed}. "
                    f"Output format: {output_format}. "
                    f"Global deduplication: "
                    f"{global_dedup}."
                ),
            )
        )

        LOGGER.info(
            "%s completed | input=%d | output=%d | removed=%d",
            dataset,
            total_input_rows,
            total_output_rows,
            removed_rows,
        )

    except Exception as exc:
        LOGGER.exception(
            "%s cleaning failed.",
            dataset,
        )

        DATASET_SUMMARIES.append(
            DatasetCleaningSummary(
                dataset=dataset,
                status="FAILED",
                source_files=len(source_files),
                input_rows=total_input_rows,
                output_rows=total_output_rows,
                removed_rows=(
                    total_input_rows
                    - total_output_rows
                ),
                duplicate_rows_removed=(
                    total_duplicates_removed
                ),
                invalid_label_rows_removed=(
                    total_invalid_labels_removed
                ),
                infinite_values_replaced=(
                    total_infinite_replaced
                ),
                output_parts=output_part_number,
                output_columns=final_output_columns,
                benign_rows=binary_counter.get(0, 0),
                attack_rows=binary_counter.get(1, 0),
                elapsed_seconds=round(
                    time.perf_counter()
                    - dataset_start,
                    3,
                ),
                output_directory=str(
                    cleaned_data_dir
                ),
                remarks=str(exc),
            )
        )

    finally:
        duplicate_tracker.close()


# ============================================================
# Consolidated reports
# ============================================================

def save_dataclass_csv(
    path: Path,
    records: Iterable[object],
    dataclass_type: type,
) -> None:
    fieldnames = list(
        dataclass_type.__dataclass_fields__.keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def save_reports(
    selected_datasets: Iterable[str],
    chunk_size: int,
    output_format: str,
    global_dedup: bool,
) -> None:
    save_dataclass_csv(
        REPORTS_DIR / "File_Cleaning_Summary.csv",
        FILE_SUMMARIES,
        FileCleaningSummary,
    )

    save_dataclass_csv(
        REPORTS_DIR / "Dataset_Cleaning_Summary.csv",
        DATASET_SUMMARIES,
        DatasetCleaningSummary,
    )

    write_json(
        REPORTS_DIR / "Dataset_Cleaning_Summary.json",
        {
            "generated_utc": utc_now(),
            "summaries": [
                asdict(summary)
                for summary in DATASET_SUMMARIES
            ],
        },
    )

    write_json(
        MANIFESTS_DIR / "step_02_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "selected_datasets": list(
                selected_datasets
            ),
            "chunk_size": chunk_size,
            "output_format": output_format,
            "global_deduplication": global_dedup,
            "data_root": str(DATA_ROOT),
            "results_root": str(RESULTS_ROOT),
            "important_note": (
                "No feature scaling was applied. "
                "Scaling must be fitted on training data only."
            ),
        },
    )


def print_summary() -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("STEP 2 DATA CLEANING SUMMARY")

    for summary in DATASET_SUMMARIES:
        LOGGER.info(
            "%-18s | %-7s | input=%12d | output=%12d | "
            "removed=%10d | parts=%4d",
            summary.dataset,
            summary.status,
            summary.input_rows,
            summary.output_rows,
            summary.removed_rows,
            summary.output_parts,
        )

    LOGGER.info("=" * 78)


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and standardize network intrusion datasets "
            "using chunked, traceable processing."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASET_NAMES, "all"),
        default="all",
        help="Dataset to clean. Default: all.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Rows read per chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE:,}."
        ),
    )

    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv.gz"),
        default=(
            "parquet"
            if parquet_available()
            else "csv.gz"
        ),
        help=(
            "Cleaned output format. Parquet requires pyarrow. "
            "Default: parquet when available, otherwise csv.gz."
        ),
    )

    parser.add_argument(
        "--no-global-dedup",
        action="store_true",
        help=(
            "Disable cross-file and cross-chunk duplicate removal. "
            "Within-chunk duplicates will still be removed."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than zero."
        )

    if (
        args.output_format == "parquet"
        and not parquet_available()
    ):
        LOGGER.error(
            "Parquet output requires pyarrow. Install it using: "
            "pip install pyarrow"
        )
        return 1

    selected_datasets = (
        DATASET_NAMES
        if args.dataset == "all"
        else (args.dataset,)
    )

    global_dedup = not args.no_global_dedup

    LOGGER.info("=" * 78)
    LOGGER.info("EXPERIMENT 01 - PHASE 02 - STEP 2")
    LOGGER.info("DATASET CLEANING AND STANDARDIZATION")
    LOGGER.info(
        "Datasets: %s",
        ", ".join(selected_datasets),
    )
    LOGGER.info(
        "Chunk size: %s",
        f"{args.chunk_size:,}",
    )
    LOGGER.info(
        "Output format: %s",
        args.output_format,
    )
    LOGGER.info(
        "Global duplicate removal: %s",
        global_dedup,
    )
    LOGGER.info(
        "Data root: %s",
        DATA_ROOT,
    )
    LOGGER.info("=" * 78)

    for dataset in selected_datasets:
        clean_dataset(
            dataset=dataset,
            chunk_size=args.chunk_size,
            output_format=args.output_format,
            global_dedup=global_dedup,
        )

    save_reports(
        selected_datasets=selected_datasets,
        chunk_size=args.chunk_size,
        output_format=args.output_format,
        global_dedup=global_dedup,
    )

    print_summary()

    failed = [
        summary.dataset
        for summary in DATASET_SUMMARIES
        if summary.status == "FAILED"
    ]

    skipped = [
        summary.dataset
        for summary in DATASET_SUMMARIES
        if summary.status == "SKIPPED"
    ]

    if failed:
        LOGGER.error(
            "Cleaning failed for: %s",
            ", ".join(failed),
        )
        return 1

    if skipped:
        LOGGER.warning(
            "Cleaning completed with skipped datasets: %s",
            ", ".join(skipped),
        )
        return 2

    LOGGER.info(
        "Step 2 completed successfully for all selected datasets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())