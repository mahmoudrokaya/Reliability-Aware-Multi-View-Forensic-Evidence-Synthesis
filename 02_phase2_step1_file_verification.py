"""
02_phase2_step1_file_verification.py

Experiment 01 - Data Preparation
Phase 02 - Integrity Verification
Step 1 - Dataset File and Folder Verification

This script verifies:

1. Dataset directories.
2. Required subdirectories.
3. Expected dataset files.
4. Empty files.
5. Duplicate files based on SHA-256.
6. Duplicate filenames.
7. Unexpected file types.
8. Dataset-level completeness.
9. Dataset storage size.

No dataset content is modified.

Exit codes
----------
0: All datasets passed.
1: One or more datasets failed.
2: No failures, but one or more warnings were detected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


# ============================================================
# Project paths
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
    / "Step_01_File_Verification"
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
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "step_01_file_verification.log"

LOGGER = logging.getLogger("step_01_file_verification")
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
# Dataset definitions
# ============================================================

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
)

REQUIRED_STANDARD_DIRECTORIES = (
    "Raw",
    "Extracted",
    "Processed",
    "Metadata",
    "Logs",
)

CICIDS2017_EXPECTED_CSV_FILES = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
)

CSE_CIC_IDS2018_EXPECTED_FILES = (
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
)

UNSW_NB15_EXPECTED_FILES = (
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "NUSW-NB15_features.csv",
    "UNSW-NB15_LIST_EVENTS.csv",
)

UNSW_NB15_OPTIONAL_FILES = (
    "NUSW-NB15_GT.csv",
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
)

BOT_IOT_REQUIRED_GROUPS = {
    "Argus files": {
        "patterns": ("*.argus",),
        "minimum_count": 1,
    },
    "Argus CSV files": {
        "patterns": ("*.csv",),
        "path_contains": "CSV of Argus",
        "minimum_count": 1,
    },
    "BRO/Zeek logs": {
        "patterns": ("*.log",),
        "minimum_count": 1,
    },
    "Reports": {
        "patterns": ("*.pdf",),
        "path_contains": "Reports",
        "minimum_count": 1,
    },
}

ALLOWED_EXTENSIONS = {
    ".csv",
    ".zip",
    ".gz",
    ".argus",
    ".log",
    ".txt",
    ".json",
    ".md5",
    ".sha256",
    ".pcap",
    ".pdf",
    ".xlsx",
    ".xls",
    ".md",
    ".exe",
}

IGNORED_FILE_NAMES = {
    "DOWNLOAD_INSTRUCTIONS.txt",
}

IGNORED_SUFFIX_PATTERNS = (
    "_Error.txt",
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class VerificationRecord:
    dataset: str
    check_category: str
    item: str
    expected: str
    exists: bool
    count: int
    size_bytes: int
    size_mib: float
    status: str
    remarks: str


@dataclass
class DatasetSummary:
    dataset: str
    status: str
    dataset_directory_exists: bool
    required_directories: int
    directories_found: int
    expected_items: int
    items_found: int
    missing_items: int
    empty_files: int
    duplicate_name_groups: int
    duplicate_hash_groups: int
    unexpected_files: int
    ignored_error_placeholders: int
    total_files: int
    total_size_bytes: int
    total_size_gib: float
    remarks: str


VERIFICATION_RECORDS: list[VerificationRecord] = []
DATASET_SUMMARIES: list[DatasetSummary] = []


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PiB"


def calculate_sha256(
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
        )


def is_ignored_error_placeholder(file_path: Path) -> bool:
    return any(
        file_path.name.endswith(suffix)
        for suffix in IGNORED_SUFFIX_PATTERNS
    )


def collect_dataset_files(dataset_root: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in dataset_root.rglob("*")
        if file_path.is_file()
        and file_path.name not in IGNORED_FILE_NAMES
    )


def find_files_by_name(
    dataset_root: Path,
    expected_name: str,
) -> list[Path]:
    expected_lower = expected_name.lower()

    return [
        file_path
        for file_path in dataset_root.rglob("*")
        if file_path.is_file()
        and file_path.name.lower() == expected_lower
    ]


def find_matching_files(
    dataset_root: Path,
    patterns: Sequence[str],
    path_contains: str | None = None,
) -> list[Path]:
    matches: dict[str, Path] = {}

    for pattern in patterns:
        for file_path in dataset_root.rglob(pattern):
            if not file_path.is_file():
                continue

            if is_ignored_error_placeholder(file_path):
                continue

            if path_contains:
                normalized_path = str(file_path).lower()
                if path_contains.lower() not in normalized_path:
                    continue

            matches[str(file_path.resolve()).lower()] = file_path

    return sorted(matches.values())


def add_record(
    dataset: str,
    check_category: str,
    item: str,
    expected: str,
    exists: bool,
    count: int = 0,
    size_bytes: int = 0,
    status: str = "PASS",
    remarks: str = "",
) -> None:
    VERIFICATION_RECORDS.append(
        VerificationRecord(
            dataset=dataset,
            check_category=check_category,
            item=item,
            expected=expected,
            exists=exists,
            count=count,
            size_bytes=size_bytes,
            size_mib=round(size_bytes / (1024**2), 6),
            status=status,
            remarks=remarks,
        )
    )


# ============================================================
# Folder verification
# ============================================================

def verify_standard_directories(
    dataset: str,
    dataset_root: Path,
) -> tuple[int, list[str]]:
    found = 0
    missing: list[str] = []

    for directory_name in REQUIRED_STANDARD_DIRECTORIES:
        directory_path = dataset_root / directory_name
        exists = directory_path.exists() and directory_path.is_dir()

        if exists:
            found += 1
            status = "PASS"
            remarks = "Required directory exists."
        else:
            missing.append(directory_name)
            status = "FAIL"
            remarks = "Required directory is missing."

        add_record(
            dataset=dataset,
            check_category="Required directory",
            item=directory_name,
            expected="Directory",
            exists=exists,
            count=1 if exists else 0,
            status=status,
            remarks=remarks,
        )

    return found, missing


# ============================================================
# Expected file verification
# ============================================================

def verify_expected_files(
    dataset: str,
    dataset_root: Path,
    expected_files: Sequence[str],
) -> tuple[int, list[str]]:
    found_count = 0
    missing_files: list[str] = []

    for expected_name in expected_files:
        matches = find_files_by_name(
            dataset_root=dataset_root,
            expected_name=expected_name,
        )

        exists = bool(matches)
        total_size = sum(path.stat().st_size for path in matches)

        if not exists:
            status = "FAIL"
            remarks = "Expected file was not found."
            missing_files.append(expected_name)
        elif len(matches) == 1:
            status = "PASS"
            remarks = f"Found at: {matches[0]}"
            found_count += 1
        else:
            status = "WARNING"
            remarks = (
                f"Found {len(matches)} copies: "
                + " | ".join(str(path) for path in matches)
            )
            found_count += 1

        add_record(
            dataset=dataset,
            check_category="Expected file",
            item=expected_name,
            expected="One non-empty official file",
            exists=exists,
            count=len(matches),
            size_bytes=total_size,
            status=status,
            remarks=remarks,
        )

    return found_count, missing_files


def verify_cicids2017(
    dataset_root: Path,
) -> tuple[int, int, list[str]]:
    """
    CICIDS2017 is considered complete when the eight expected
    machine-learning CSV files are present.

    Files can be under MachineLearningCVE, TrafficLabelling_, or
    another extracted official folder.
    """

    found_count, missing_files = verify_expected_files(
        dataset="CICIDS2017",
        dataset_root=dataset_root,
        expected_files=CICIDS2017_EXPECTED_CSV_FILES,
    )

    return (
        len(CICIDS2017_EXPECTED_CSV_FILES),
        found_count,
        missing_files,
    )


def verify_cse_cic_ids2018(
    dataset_root: Path,
) -> tuple[int, int, list[str]]:
    found_count, missing_files = verify_expected_files(
        dataset="CSE-CIC-IDS2018",
        dataset_root=dataset_root,
        expected_files=CSE_CIC_IDS2018_EXPECTED_FILES,
    )

    return (
        len(CSE_CIC_IDS2018_EXPECTED_FILES),
        found_count,
        missing_files,
    )


def verify_unsw_nb15(
    dataset_root: Path,
) -> tuple[int, int, list[str]]:
    found_count, missing_files = verify_expected_files(
        dataset="UNSW-NB15",
        dataset_root=dataset_root,
        expected_files=UNSW_NB15_EXPECTED_FILES,
    )

    for optional_name in UNSW_NB15_OPTIONAL_FILES:
        matches = find_files_by_name(
            dataset_root=dataset_root,
            expected_name=optional_name,
        )

        add_record(
            dataset="UNSW-NB15",
            check_category="Optional file",
            item=optional_name,
            expected="Optional",
            exists=bool(matches),
            count=len(matches),
            size_bytes=sum(path.stat().st_size for path in matches),
            status="PASS" if matches else "WARNING",
            remarks=(
                "Optional file is available."
                if matches
                else "Optional file was not found."
            ),
        )

    return (
        len(UNSW_NB15_EXPECTED_FILES),
        found_count,
        missing_files,
    )


def verify_bot_iot(
    dataset_root: Path,
) -> tuple[int, int, list[str]]:
    expected_items = len(BOT_IOT_REQUIRED_GROUPS)
    found_items = 0
    missing_items: list[str] = []

    for group_name, definition in BOT_IOT_REQUIRED_GROUPS.items():
        patterns = definition["patterns"]
        minimum_count = int(definition["minimum_count"])
        path_contains = definition.get("path_contains")

        matches = find_matching_files(
            dataset_root=dataset_root,
            patterns=patterns,
            path_contains=path_contains,
        )

        exists = len(matches) >= minimum_count
        total_size = sum(path.stat().st_size for path in matches)

        if exists:
            found_items += 1
            status = "PASS"
            remarks = (
                f"Found {len(matches)} valid files. "
                f"Total size: {human_size(total_size)}."
            )
        else:
            status = "FAIL"
            missing_items.append(group_name)
            remarks = (
                f"Required minimum count is {minimum_count}; "
                f"found {len(matches)}."
            )

        add_record(
            dataset="BoT-IoT",
            check_category="Required data group",
            item=group_name,
            expected=f"At least {minimum_count} valid file(s)",
            exists=exists,
            count=len(matches),
            size_bytes=total_size,
            status=status,
            remarks=remarks,
        )

    return expected_items, found_items, missing_items


# ============================================================
# File integrity checks
# ============================================================

def detect_empty_files(
    dataset: str,
    files: Iterable[Path],
) -> list[Path]:
    empty_files = [
        file_path
        for file_path in files
        if file_path.stat().st_size == 0
        and not is_ignored_error_placeholder(file_path)
    ]

    for file_path in empty_files:
        add_record(
            dataset=dataset,
            check_category="Empty file",
            item=str(file_path),
            expected="File size > 0 bytes",
            exists=True,
            count=1,
            size_bytes=0,
            status="FAIL",
            remarks="The file exists but is empty.",
        )

    return empty_files


def detect_duplicate_names(
    dataset: str,
    files: Iterable[Path],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for file_path in files:
        if is_ignored_error_placeholder(file_path):
            continue

        grouped[file_path.name.lower()].append(file_path)

    duplicates = {
        name: paths
        for name, paths in grouped.items()
        if len(paths) > 1
    }

    for file_name, paths in duplicates.items():
        add_record(
            dataset=dataset,
            check_category="Duplicate filename",
            item=file_name,
            expected="One intended copy",
            exists=True,
            count=len(paths),
            size_bytes=sum(path.stat().st_size for path in paths),
            status="WARNING",
            remarks=" | ".join(str(path) for path in paths),
        )

    return duplicates


def detect_duplicate_hashes(
    dataset: str,
    files: Iterable[Path],
    compute_hashes: bool,
) -> dict[str, list[Path]]:
    if not compute_hashes:
        add_record(
            dataset=dataset,
            check_category="Duplicate content",
            item="SHA-256 duplicate analysis",
            expected="Computed unless --skip-hash is used",
            exists=False,
            count=0,
            status="WARNING",
            remarks="SHA-256 duplicate analysis was skipped.",
        )
        return {}

    candidates = [
        file_path
        for file_path in files
        if file_path.stat().st_size > 0
        and not is_ignored_error_placeholder(file_path)
        and file_path.suffix.lower() in {
            ".csv",
            ".zip",
            ".argus",
            ".log",
            ".pdf",
            ".xlsx",
            ".pcap",
        }
    ]

    grouped_by_size: dict[int, list[Path]] = defaultdict(list)

    for file_path in candidates:
        grouped_by_size[file_path.stat().st_size].append(file_path)

    grouped_by_hash: dict[str, list[Path]] = defaultdict(list)

    for size_bytes, same_size_files in grouped_by_size.items():
        if len(same_size_files) < 2:
            continue

        LOGGER.info(
            "%s: hashing %d same-size files of %s.",
            dataset,
            len(same_size_files),
            human_size(size_bytes),
        )

        for file_path in same_size_files:
            digest = calculate_sha256(file_path)
            grouped_by_hash[digest].append(file_path)

    duplicates = {
        digest: paths
        for digest, paths in grouped_by_hash.items()
        if len(paths) > 1
    }

    for digest, paths in duplicates.items():
        add_record(
            dataset=dataset,
            check_category="Duplicate content",
            item=digest,
            expected="Unique file content",
            exists=True,
            count=len(paths),
            size_bytes=sum(path.stat().st_size for path in paths),
            status="WARNING",
            remarks=" | ".join(str(path) for path in paths),
        )

    return duplicates


def detect_unexpected_extensions(
    dataset: str,
    files: Iterable[Path],
) -> list[Path]:
    unexpected: list[Path] = []

    for file_path in files:
        if is_ignored_error_placeholder(file_path):
            continue

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            unexpected.append(file_path)

            add_record(
                dataset=dataset,
                check_category="Unexpected extension",
                item=str(file_path),
                expected="Recognized dataset or metadata file type",
                exists=True,
                count=1,
                size_bytes=file_path.stat().st_size,
                status="WARNING",
                remarks=f"Extension: {file_path.suffix}",
            )

    return unexpected


def count_error_placeholders(files: Iterable[Path]) -> int:
    return sum(
        1
        for file_path in files
        if is_ignored_error_placeholder(file_path)
    )


# ============================================================
# Dataset verification orchestration
# ============================================================

def verify_dataset(
    dataset: str,
    compute_hashes: bool,
) -> None:
    LOGGER.info("-" * 78)
    LOGGER.info("Checking dataset: %s", dataset)

    dataset_root = DATA_ROOT / dataset
    dataset_exists = dataset_root.exists() and dataset_root.is_dir()

    if not dataset_exists:
        add_record(
            dataset=dataset,
            check_category="Dataset directory",
            item=str(dataset_root),
            expected="Existing directory",
            exists=False,
            status="FAIL",
            remarks="Dataset directory does not exist.",
        )

        DATASET_SUMMARIES.append(
            DatasetSummary(
                dataset=dataset,
                status="FAILED",
                dataset_directory_exists=False,
                required_directories=len(REQUIRED_STANDARD_DIRECTORIES),
                directories_found=0,
                expected_items=0,
                items_found=0,
                missing_items=1,
                empty_files=0,
                duplicate_name_groups=0,
                duplicate_hash_groups=0,
                unexpected_files=0,
                ignored_error_placeholders=0,
                total_files=0,
                total_size_bytes=0,
                total_size_gib=0.0,
                remarks="Dataset directory is missing.",
            )
        )
        return

    add_record(
        dataset=dataset,
        check_category="Dataset directory",
        item=str(dataset_root),
        expected="Existing directory",
        exists=True,
        count=1,
        status="PASS",
        remarks="Dataset directory exists.",
    )

    directories_found, missing_directories = verify_standard_directories(
        dataset=dataset,
        dataset_root=dataset_root,
    )

    if dataset == "CICIDS2017":
        expected_items, items_found, missing_items = verify_cicids2017(
            dataset_root
        )
    elif dataset == "CSE-CIC-IDS2018":
        expected_items, items_found, missing_items = (
            verify_cse_cic_ids2018(dataset_root)
        )
    elif dataset == "UNSW-NB15":
        expected_items, items_found, missing_items = verify_unsw_nb15(
            dataset_root
        )
    elif dataset == "BoT-IoT":
        expected_items, items_found, missing_items = verify_bot_iot(
            dataset_root
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    files = collect_dataset_files(dataset_root)
    total_size = sum(file_path.stat().st_size for file_path in files)

    empty_files = detect_empty_files(
        dataset=dataset,
        files=files,
    )

    duplicate_names = detect_duplicate_names(
        dataset=dataset,
        files=files,
    )

    duplicate_hashes = detect_duplicate_hashes(
        dataset=dataset,
        files=files,
        compute_hashes=compute_hashes,
    )

    unexpected_files = detect_unexpected_extensions(
        dataset=dataset,
        files=files,
    )

    error_placeholders = count_error_placeholders(files)

    failure_conditions = (
        bool(missing_directories)
        or bool(missing_items)
        or bool(empty_files)
    )

    warning_conditions = (
        bool(duplicate_names)
        or bool(duplicate_hashes)
        or bool(unexpected_files)
        or error_placeholders > 0
    )

    if failure_conditions:
        final_status = "FAILED"
    elif warning_conditions:
        final_status = "WARNING"
    else:
        final_status = "PASS"

    remarks_parts: list[str] = []

    if missing_directories:
        remarks_parts.append(
            "Missing directories: " + ", ".join(missing_directories)
        )

    if missing_items:
        remarks_parts.append(
            "Missing expected items: " + ", ".join(missing_items)
        )

    if duplicate_hashes:
        remarks_parts.append(
            f"{len(duplicate_hashes)} duplicate-content group(s) found."
        )

    if error_placeholders:
        remarks_parts.append(
            f"{error_placeholders} *_Error.txt placeholder file(s) ignored."
        )

    if not remarks_parts:
        remarks_parts.append("All required checks passed.")

    DATASET_SUMMARIES.append(
        DatasetSummary(
            dataset=dataset,
            status=final_status,
            dataset_directory_exists=True,
            required_directories=len(REQUIRED_STANDARD_DIRECTORIES),
            directories_found=directories_found,
            expected_items=expected_items,
            items_found=items_found,
            missing_items=len(missing_items),
            empty_files=len(empty_files),
            duplicate_name_groups=len(duplicate_names),
            duplicate_hash_groups=len(duplicate_hashes),
            unexpected_files=len(unexpected_files),
            ignored_error_placeholders=error_placeholders,
            total_files=len(files),
            total_size_bytes=total_size,
            total_size_gib=round(total_size / (1024**3), 6),
            remarks=" ".join(remarks_parts),
        )
    )

    LOGGER.info(
        "%s | status=%s | files=%d | size=%s",
        dataset,
        final_status,
        len(files),
        human_size(total_size),
    )


# ============================================================
# Output generation
# ============================================================

def save_verification_records() -> None:
    csv_path = REPORTS_DIR / "Integrity_Report.csv"
    json_path = REPORTS_DIR / "Integrity_Report.json"

    fieldnames = list(
        VerificationRecord.__dataclass_fields__.keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in VERIFICATION_RECORDS:
            writer.writerow(asdict(record))

    write_json(
        json_path,
        {
            "generated_utc": utc_now(),
            "project_root": str(PROJECT_ROOT),
            "records": [
                asdict(record)
                for record in VERIFICATION_RECORDS
            ],
        },
    )

    LOGGER.info("Integrity report CSV: %s", csv_path)
    LOGGER.info("Integrity report JSON: %s", json_path)


def save_dataset_summary() -> None:
    csv_path = REPORTS_DIR / "Dataset_Verification_Summary.csv"
    json_path = REPORTS_DIR / "Dataset_Verification_Summary.json"

    fieldnames = list(
        DatasetSummary.__dataclass_fields__.keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in DATASET_SUMMARIES:
            writer.writerow(asdict(summary))

    write_json(
        json_path,
        {
            "generated_utc": utc_now(),
            "summaries": [
                asdict(summary)
                for summary in DATASET_SUMMARIES
            ],
        },
    )

    LOGGER.info("Dataset summary CSV: %s", csv_path)
    LOGGER.info("Dataset summary JSON: %s", json_path)


def save_run_manifest(
    selected_datasets: Sequence[str],
    compute_hashes: bool,
) -> None:
    path = MANIFESTS_DIR / "step_01_run_manifest.json"

    write_json(
        path,
        {
            "generated_utc": utc_now(),
            "script": Path(__file__).name,
            "selected_datasets": list(selected_datasets),
            "compute_duplicate_hashes": compute_hashes,
            "data_root": str(DATA_ROOT),
            "results_root": str(RESULTS_ROOT),
        },
    )


def print_summary() -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("STEP 1 DATASET FILE VERIFICATION SUMMARY")

    for summary in DATASET_SUMMARIES:
        LOGGER.info(
            "%-18s | %-7s | files=%4d | size=%8.3f GiB | "
            "missing=%d | duplicate_hash_groups=%d",
            summary.dataset,
            summary.status,
            summary.total_files,
            summary.total_size_gib,
            summary.missing_items,
            summary.duplicate_hash_groups,
        )

    LOGGER.info("=" * 78)


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify dataset folders and expected files without "
            "modifying any dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASET_NAMES, "all"),
        default="all",
        help="Dataset to verify. Default: all.",
    )

    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help=(
            "Skip SHA-256 duplicate-content detection. "
            "Use this for a faster first run."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    selected_datasets = (
        DATASET_NAMES
        if args.dataset == "all"
        else (args.dataset,)
    )

    compute_hashes = not args.skip_hash

    LOGGER.info("=" * 78)
    LOGGER.info("EXPERIMENT 01 - PHASE 02 - STEP 1")
    LOGGER.info("DATASET FILE AND FOLDER VERIFICATION")
    LOGGER.info("Datasets: %s", ", ".join(selected_datasets))
    LOGGER.info("Duplicate SHA-256 analysis: %s", compute_hashes)
    LOGGER.info("Data root: %s", DATA_ROOT)
    LOGGER.info("=" * 78)

    save_run_manifest(
        selected_datasets=selected_datasets,
        compute_hashes=compute_hashes,
    )

    for dataset in selected_datasets:
        try:
            verify_dataset(
                dataset=dataset,
                compute_hashes=compute_hashes,
            )
        except Exception:
            LOGGER.exception(
                "Unexpected verification failure for %s.",
                dataset,
            )

            DATASET_SUMMARIES.append(
                DatasetSummary(
                    dataset=dataset,
                    status="FAILED",
                    dataset_directory_exists=(
                        DATA_ROOT / dataset
                    ).exists(),
                    required_directories=len(
                        REQUIRED_STANDARD_DIRECTORIES
                    ),
                    directories_found=0,
                    expected_items=0,
                    items_found=0,
                    missing_items=1,
                    empty_files=0,
                    duplicate_name_groups=0,
                    duplicate_hash_groups=0,
                    unexpected_files=0,
                    ignored_error_placeholders=0,
                    total_files=0,
                    total_size_bytes=0,
                    total_size_gib=0.0,
                    remarks=(
                        "Unexpected exception. Review the execution log."
                    ),
                )
            )

    save_verification_records()
    save_dataset_summary()
    print_summary()

    failed = [
        summary.dataset
        for summary in DATASET_SUMMARIES
        if summary.status == "FAILED"
    ]

    warned = [
        summary.dataset
        for summary in DATASET_SUMMARIES
        if summary.status == "WARNING"
    ]

    if failed:
        LOGGER.error(
            "Step 1 failed for: %s",
            ", ".join(failed),
        )
        return 1

    if warned:
        LOGGER.warning(
            "Step 1 completed with warnings for: %s",
            ", ".join(warned),
        )
        return 2

    LOGGER.info("Step 1 completed successfully for all datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())