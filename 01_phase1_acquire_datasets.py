"""
01_phase1_acquire_datasets.py

Phase 1: Official dataset acquisition and provenance recording.

This script:

1. Creates a standardized directory structure for all datasets.
2. Automatically downloads the processed CSE-CIC-IDS2018 CSV files
   from the official public AWS S3 bucket.
3. Detects manually downloaded official files for:
      - CICIDS2017
      - UNSW-NB15
      - BoT-IoT
4. Creates official manual-download instructions when required files
   are absent.
5. Computes SHA-256 hashes for acquired files.
6. Records file paths, sizes, sources, timestamps, and acquisition status.
7. Produces CSV and JSON manifests.
8. Produces a dataset-level acquisition summary.
9. Returns:
      0 = all selected datasets complete
      1 = technical failure
      2 = partial completion; one or more datasets still missing

This phase does not clean, merge, transform, sample, or split data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

DATA_ROOT = PROJECT_ROOT / "Data"
CODE_ROOT = PROJECT_ROOT / "Code"
RESULTS_ROOT = PROJECT_ROOT / "Results"

PHASE_ROOT = (
    RESULTS_ROOT
    / "Experiment_01_Data_Preparation"
    / "Phase_01_Acquisition"
)

LOGS_DIR = PHASE_ROOT / "Logs"
MANIFESTS_DIR = PHASE_ROOT / "Manifests"
REPORTS_DIR = PHASE_ROOT / "Reports"

DATASET_NAMES = (
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
)


# ============================================================
# Official dataset sources
# ============================================================

OFFICIAL_SOURCES = {
    "CICIDS2017": {
        "landing_page": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "required_content": (
            "MachineLearningCSV.zip or the extracted machine-learning CSV files"
        ),
    },
    "CSE-CIC-IDS2018": {
        "landing_page": "https://registry.opendata.aws/cse-cic-ids2018/",
        "s3_prefix": (
            "s3://cse-cic-ids2018/"
            "Processed Traffic Data for ML Algorithms/"
        ),
        "required_content": (
            "Processed Traffic Data for ML Algorithms/*.csv"
        ),
    },
    "UNSW-NB15": {
        "landing_page": (
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        ),
        "required_content": "Official UNSW-NB15 CSV feature files",
    },
    "BoT-IoT": {
        "landing_page": (
            "https://research.unsw.edu.au/projects/bot-iot-dataset"
        ),
        "required_content": (
            "Official BoT-IoT extracted CSV or flow-feature files"
        ),
    },
}


# ============================================================
# Expected dataset files
# ============================================================

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

UNSW_NB15_CORE_FILES = (
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "UNSW-NB15_features.csv",
    "UNSW-NB15_LIST_EVENTS.csv",
)

UNSW_NB15_OPTIONAL_FILES = (
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
)

CICIDS2017_ARCHIVE_NAME = "MachineLearningCSV.zip"


# ============================================================
# Directory preparation
# ============================================================

for directory in (
    DATA_ROOT,
    CODE_ROOT,
    RESULTS_ROOT,
    PHASE_ROOT,
    LOGS_DIR,
    MANIFESTS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================

LOG_FILE = LOGS_DIR / "phase_01_acquisition.log"

LOGGER = logging.getLogger("phase_01_acquisition")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


# ============================================================
# Data structures
# ============================================================

@dataclass
class FileRecord:
    dataset: str
    file_name: str
    local_path: str
    relative_path: str
    source: str
    status: str
    size_bytes: int
    size_mib: float
    size_gib: float
    sha256: str
    acquisition_utc: str
    notes: str


@dataclass
class DatasetStatus:
    dataset: str
    complete: bool
    number_of_files: int
    total_bytes: int
    total_gib: float
    missing_files: list[str]
    statuses: list[str]
    official_source: str
    notes: str


FILE_RECORDS: list[FileRecord] = []
DATASET_STATUSES: dict[str, DatasetStatus] = {}


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def human_size(size_bytes: int) -> str:
    """Convert bytes into a readable binary storage unit."""

    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def prepare_dataset_directories(dataset: str) -> dict[str, Path]:
    """Create the standard directory hierarchy for one dataset."""

    base = DATA_ROOT / dataset

    directories = {
        "base": base,
        "raw": base / "Raw",
        "extracted": base / "Extracted",
        "processed": base / "Processed",
        "metadata": base / "Metadata",
        "logs": base / "Logs",
        "manual": base / "Manual_Download",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def find_aws_cli() -> str:
    """Locate the AWS CLI executable on Windows."""

    discovered = shutil.which("aws")

    if discovered:
        return discovered

    common_paths = (
        Path(r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"),
        Path(r"C:\Program Files\Amazon\AWSCLIV2\aws.EXE"),
    )

    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "AWS CLI was not found. Confirm that `aws --version` works "
        "in the current PowerShell session."
    )


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Calculate a SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file using UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
        )


def find_files(
    directories: Iterable[Path],
    patterns: Sequence[str],
) -> list[Path]:
    """Find unique files matching one or more glob patterns."""

    found: dict[str, Path] = {}

    for directory in directories:
        if not directory.exists():
            continue

        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if not file_path.is_file():
                    continue

                if file_path.name.endswith(".part"):
                    continue

                key = str(file_path.resolve()).lower()
                found[key] = file_path

    return sorted(
        found.values(),
        key=lambda path: str(path).lower(),
    )


def find_csv_files(directories: Iterable[Path]) -> list[Path]:
    """Find CSV files recursively."""

    return find_files(
        directories=directories,
        patterns=("*.csv", "*.CSV"),
    )


def record_file(
    dataset: str,
    file_path: Path,
    source: str,
    status: str,
    compute_hash: bool,
    notes: str = "",
) -> None:
    """Add an acquired file to the phase manifest."""

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot record missing file: {file_path}"
        )

    resolved_path = file_path.resolve()
    size_bytes = resolved_path.stat().st_size

    LOGGER.info(
        "Recording file: %s (%s)",
        resolved_path.name,
        human_size(size_bytes),
    )

    if compute_hash:
        LOGGER.info(
            "Computing SHA-256: %s",
            resolved_path.name,
        )
        sha256 = calculate_sha256(resolved_path)
    else:
        sha256 = "NOT_COMPUTED"

    try:
        relative_path = str(
            resolved_path.relative_to(PROJECT_ROOT.resolve())
        )
    except ValueError:
        relative_path = str(resolved_path)

    FILE_RECORDS.append(
        FileRecord(
            dataset=dataset,
            file_name=resolved_path.name,
            local_path=str(resolved_path),
            relative_path=relative_path,
            source=source,
            status=status,
            size_bytes=size_bytes,
            size_mib=round(size_bytes / (1024**2), 6),
            size_gib=round(size_bytes / (1024**3), 6),
            sha256=sha256,
            acquisition_utc=utc_now(),
            notes=notes,
        )
    )


def write_manual_instructions(
    dataset: str,
    directories: dict[str, Path],
    required_files: Iterable[str],
    additional_notes: str,
) -> Path:
    """Create dataset-specific official download instructions."""

    source = OFFICIAL_SOURCES[dataset]
    instruction_path = (
        directories["manual"]
        / "DOWNLOAD_INSTRUCTIONS.txt"
    )

    required_text = "\n".join(
        f"  - {required_file}"
        for required_file in required_files
    )

    instructions = f"""
DATASET
{dataset}

OFFICIAL ACCESS PAGE
{source["landing_page"]}

REQUIRED CONTENT
{source["required_content"]}

FILES REQUIRED FOR THE CURRENT STUDY
{required_text}

RAW DOWNLOAD DIRECTORY
{directories["raw"]}

EXTRACTED DATA DIRECTORY
{directories["extracted"]}

PROCEDURE
1. Open the official access page.
2. Download the official machine-learning, flow-feature, or CSV files.
3. Do not download the PCAP collection unless packet-level analysis is planned.
4. Store original archives or downloaded files in the Raw directory.
5. Store extracted CSV files in the Extracted directory.
6. Rerun this script so the files are inventoried and hashed.
7. Do not rename official files before the acquisition manifest is generated.

ADDITIONAL NOTES
{additional_notes}

Generated UTC
{utc_now()}
""".strip()

    instruction_path.write_text(
        instructions,
        encoding="utf-8",
    )

    LOGGER.warning(
        "%s is incomplete. Download instructions: %s",
        dataset,
        instruction_path,
    )

    return instruction_path


def names_lower(files: Iterable[Path]) -> set[str]:
    """Return lowercase file names."""

    return {file_path.name.lower() for file_path in files}


def expected_missing(
    discovered_files: Iterable[Path],
    expected_names: Iterable[str],
) -> list[str]:
    """Identify expected files that have not been discovered."""

    discovered = names_lower(discovered_files)

    return [
        name
        for name in expected_names
        if name.lower() not in discovered
    ]


def set_dataset_status(
    dataset: str,
    complete: bool,
    files: Iterable[Path],
    missing_files: Iterable[str],
    statuses: Iterable[str],
    notes: str,
) -> None:
    """Store the dataset-level acquisition state."""

    file_list = list(files)
    total_bytes = sum(
        file_path.stat().st_size
        for file_path in file_list
        if file_path.exists()
    )

    DATASET_STATUSES[dataset] = DatasetStatus(
        dataset=dataset,
        complete=complete,
        number_of_files=len(file_list),
        total_bytes=total_bytes,
        total_gib=round(total_bytes / (1024**3), 6),
        missing_files=list(missing_files),
        statuses=sorted(set(statuses)),
        official_source=OFFICIAL_SOURCES[dataset][
            "landing_page"
        ],
        notes=notes,
    )


# ============================================================
# CICIDS2017
# ============================================================

def inspect_cicids2017(compute_hash: bool) -> None:
    """Inventory official CICIDS2017 machine-learning files."""

    dataset = "CICIDS2017"
    directories = prepare_dataset_directories(dataset)

    archives = find_files(
        directories=(directories["raw"],),
        patterns=(CICIDS2017_ARCHIVE_NAME,),
    )

    csv_files = find_csv_files(
        (
            directories["raw"],
            directories["extracted"],
        )
    )

    all_files = archives + csv_files

    if not all_files:
        write_manual_instructions(
            dataset=dataset,
            directories=directories,
            required_files=(
                CICIDS2017_ARCHIVE_NAME,
                "or all extracted MachineLearningCSV CSV files",
            ),
            additional_notes=(
                "Select the MachineLearningCSV package from the "
                "official CICIDS2017 page."
            ),
        )

        set_dataset_status(
            dataset=dataset,
            complete=False,
            files=[],
            missing_files=[
                CICIDS2017_ARCHIVE_NAME
            ],
            statuses=["NOT_ACQUIRED"],
            notes="Official CICIDS2017 files are not present.",
        )
        return

    for archive in archives:
        record_file(
            dataset=dataset,
            file_path=archive,
            source=OFFICIAL_SOURCES[dataset]["landing_page"],
            status="ACQUIRED_OFFICIAL_ARCHIVE",
            compute_hash=compute_hash,
            notes="Official CICIDS2017 MachineLearningCSV archive.",
        )

    for file_path in csv_files:
        record_file(
            dataset=dataset,
            file_path=file_path,
            source=OFFICIAL_SOURCES[dataset]["landing_page"],
            status="ACQUIRED_OFFICIAL_CSV",
            compute_hash=compute_hash,
            notes="Official CICIDS2017 flow-feature CSV.",
        )

    complete = bool(csv_files or archives)

    set_dataset_status(
        dataset=dataset,
        complete=complete,
        files=all_files,
        missing_files=[],
        statuses=[
            "ACQUIRED_OFFICIAL_ARCHIVE"
            if archives
            else "",
            "ACQUIRED_OFFICIAL_CSV"
            if csv_files
            else "",
        ],
        notes=(
            "CICIDS2017 acquisition detected. Archive extraction "
            "and structural validation will be handled in Phase 2."
        ),
    )


# ============================================================
# CSE-CIC-IDS2018
# ============================================================

def acquire_cse_cic_ids2018(
    compute_hash: bool,
    dry_run: bool,
) -> None:
    """Download and inventory official CSE-CIC-IDS2018 CSV files."""

    dataset = "CSE-CIC-IDS2018"
    directories = prepare_dataset_directories(dataset)

    aws_executable = find_aws_cli()
    official_prefix = OFFICIAL_SOURCES[dataset]["s3_prefix"]

    download_target = (
        directories["raw"]
        / "Processed Traffic Data for ML Algorithms"
    )
    download_target.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        aws_executable,
        "s3",
        "sync",
        official_prefix,
        str(download_target),
        "--no-sign-request",
        "--region",
        "ca-central-1",
        "--exclude",
        "*",
        "--include",
        "*.csv",
        "--only-show-errors",
    ]

    if dry_run:
        command.append("--dryrun")

    LOGGER.info("Official source: %s", official_prefix)
    LOGGER.info("Destination: %s", download_target)
    LOGGER.info(
        "Command: %s",
        subprocess.list2cmdline(command),
    )

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    aws_log_path = (
        directories["logs"]
        / "aws_sync_output.txt"
    )

    aws_log_path.write_text(
        "\n".join(
            (
                f"Executed UTC: {utc_now()}",
                "",
                "COMMAND",
                subprocess.list2cmdline(command),
                "",
                "RETURN CODE",
                str(completed.returncode),
                "",
                "STDOUT",
                completed.stdout,
                "",
                "STDERR",
                completed.stderr,
            )
        ),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "AWS synchronization failed with return code "
            f"{completed.returncode}. See {aws_log_path}"
        )

    if dry_run:
        LOGGER.info(
            "CSE-CIC-IDS2018 dry run completed."
        )

        set_dataset_status(
            dataset=dataset,
            complete=False,
            files=[],
            missing_files=list(
                CSE_CIC_IDS2018_EXPECTED_FILES
            ),
            statuses=["DRY_RUN"],
            notes="No files were downloaded during the dry run.",
        )
        return

    csv_files = find_csv_files((download_target,))

    missing_files = expected_missing(
        csv_files,
        CSE_CIC_IDS2018_EXPECTED_FILES,
    )

    total_bytes = sum(
        file_path.stat().st_size
        for file_path in csv_files
    )

    LOGGER.info(
        "%s acquisition found %d CSV files totaling %s.",
        dataset,
        len(csv_files),
        human_size(total_bytes),
    )

    for file_path in csv_files:
        record_file(
            dataset=dataset,
            file_path=file_path,
            source=official_prefix,
            status="ACQUIRED_OFFICIAL_AWS",
            compute_hash=compute_hash,
            notes=(
                "Processed flow-feature CSV downloaded from "
                "the official public AWS bucket."
            ),
        )

    complete = (
        len(csv_files)
        >= len(CSE_CIC_IDS2018_EXPECTED_FILES)
        and not missing_files
    )

    set_dataset_status(
        dataset=dataset,
        complete=complete,
        files=csv_files,
        missing_files=missing_files,
        statuses=[
            "ACQUIRED_OFFICIAL_AWS"
            if csv_files
            else "NOT_ACQUIRED"
        ],
        notes=(
            "All expected official CSV files were found."
            if complete
            else "One or more expected CSE-CIC-IDS2018 files are missing."
        ),
    )


# ============================================================
# UNSW-NB15
# ============================================================

def inspect_unsw_nb15(compute_hash: bool) -> None:
    """Inventory official UNSW-NB15 files."""

    dataset = "UNSW-NB15"
    directories = prepare_dataset_directories(dataset)

    csv_files = find_csv_files(
        (
            directories["raw"],
            directories["extracted"],
        )
    )

    missing_core = expected_missing(
        csv_files,
        UNSW_NB15_CORE_FILES,
    )

    if not csv_files or missing_core:
        write_manual_instructions(
            dataset=dataset,
            directories=directories,
            required_files=(
                *UNSW_NB15_CORE_FILES,
                *(
                    f"{name} (optional)"
                    for name in UNSW_NB15_OPTIONAL_FILES
                ),
            ),
            additional_notes=(
                "The four original CSV partitions and the feature "
                "description file are required. The predefined training "
                "and testing files are optional but recommended."
            ),
        )

    status = (
        "ACQUIRED_COMPLETE"
        if csv_files and not missing_core
        else "ACQUIRED_PARTIAL"
        if csv_files
        else "NOT_ACQUIRED"
    )

    for file_path in csv_files:
        record_file(
            dataset=dataset,
            file_path=file_path,
            source=OFFICIAL_SOURCES[dataset]["landing_page"],
            status=status,
            compute_hash=compute_hash,
            notes="Official UNSW-NB15 CSV file.",
        )

    set_dataset_status(
        dataset=dataset,
        complete=bool(csv_files and not missing_core),
        files=csv_files,
        missing_files=missing_core,
        statuses=[status],
        notes=(
            "All required UNSW-NB15 files are present."
            if csv_files and not missing_core
            else "UNSW-NB15 acquisition is incomplete."
        ),
    )


# ============================================================
# BoT-IoT
# ============================================================

def inspect_bot_iot(compute_hash: bool) -> None:
    """Inventory official BoT-IoT CSV or flow-feature files."""

    dataset = "BoT-IoT"
    directories = prepare_dataset_directories(dataset)

    flow_files = find_files(
        directories=(
            directories["raw"],
            directories["extracted"],
        ),
        patterns=(
            "*.csv",
            "*.CSV",
            "*.zip",
            "*.ZIP",
            "*.gz",
            "*.GZ",
        ),
    )

    if not flow_files:
        write_manual_instructions(
            dataset=dataset,
            directories=directories,
            required_files=(
                "BoT-IoT extracted CSV or flow-feature files",
                "dataset feature-description or documentation files",
            ),
            additional_notes=(
                "Use the flow-feature representation for the current "
                "experiments. The complete PCAP collection is unnecessary."
            ),
        )

        status = "NOT_ACQUIRED"
    else:
        status = "ACQUIRED_OFFICIAL_FLOW_FILES"

    for file_path in flow_files:
        record_file(
            dataset=dataset,
            file_path=file_path,
            source=OFFICIAL_SOURCES[dataset]["landing_page"],
            status=status,
            compute_hash=compute_hash,
            notes="Official BoT-IoT flow-feature or archive file.",
        )

    set_dataset_status(
        dataset=dataset,
        complete=bool(flow_files),
        files=flow_files,
        missing_files=[] if flow_files else [
            "BoT-IoT flow-feature files"
        ],
        statuses=[status],
        notes=(
            "BoT-IoT flow-feature files were detected."
            if flow_files
            else "BoT-IoT files have not been acquired."
        ),
    )


# ============================================================
# Output generation
# ============================================================

def save_file_manifests() -> None:
    """Save file-level acquisition manifests."""

    csv_path = (
        MANIFESTS_DIR
        / "phase_01_file_manifest.csv"
    )

    json_path = (
        MANIFESTS_DIR
        / "phase_01_file_manifest.json"
    )

    fieldnames = list(
        FileRecord.__dataclass_fields__.keys()
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

        for record in FILE_RECORDS:
            writer.writerow(asdict(record))

    write_json(
        json_path,
        {
            "generated_utc": utc_now(),
            "project_root": str(PROJECT_ROOT),
            "records": [
                asdict(record)
                for record in FILE_RECORDS
            ],
        },
    )

    LOGGER.info("CSV manifest: %s", csv_path)
    LOGGER.info("JSON manifest: %s", json_path)


def save_dataset_summary(
    selected_datasets: Iterable[str],
) -> None:
    """Save dataset-level status reports."""

    selected_statuses = {
        dataset: asdict(DATASET_STATUSES[dataset])
        for dataset in selected_datasets
        if dataset in DATASET_STATUSES
    }

    json_path = (
        REPORTS_DIR
        / "phase_01_acquisition_summary.json"
    )

    csv_path = (
        REPORTS_DIR
        / "phase_01_acquisition_summary.csv"
    )

    write_json(
        json_path,
        {
            "generated_utc": utc_now(),
            "datasets": selected_statuses,
        },
    )

    csv_fieldnames = (
        "dataset",
        "complete",
        "number_of_files",
        "total_bytes",
        "total_gib",
        "missing_files",
        "statuses",
        "official_source",
        "notes",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=csv_fieldnames,
        )

        writer.writeheader()

        for dataset in selected_datasets:
            status = DATASET_STATUSES.get(dataset)

            if status is None:
                continue

            row = asdict(status)
            row["missing_files"] = " | ".join(
                status.missing_files
            )
            row["statuses"] = " | ".join(
                status.statuses
            )

            writer.writerow(row)

    LOGGER.info(
        "Acquisition summary JSON: %s",
        json_path,
    )

    LOGGER.info(
        "Acquisition summary CSV: %s",
        csv_path,
    )


def print_console_summary(
    selected_datasets: Iterable[str],
) -> None:
    """Print a concise dataset-level status summary."""

    LOGGER.info("-" * 76)
    LOGGER.info("DATASET ACQUISITION SUMMARY")

    for dataset in selected_datasets:
        status = DATASET_STATUSES.get(dataset)

        if status is None:
            LOGGER.error(
                "%s: no status was produced.",
                dataset,
            )
            continue

        state = (
            "COMPLETE"
            if status.complete
            else "INCOMPLETE"
        )

        LOGGER.info(
            "%s | %s | files=%d | size=%.3f GiB",
            dataset,
            state,
            status.number_of_files,
            status.total_gib,
        )

        if status.missing_files:
            LOGGER.warning(
                "%s missing: %s",
                dataset,
                ", ".join(status.missing_files),
            )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: acquire and inventory official network "
            "intrusion datasets."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASET_NAMES, "all"),
        default="all",
        help="Dataset to acquire or inspect. Default: all.",
    )

    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help=(
            "Skip SHA-256 generation for a faster inventory. "
            "Not recommended for the final acquisition record."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview CSE-CIC-IDS2018 AWS synchronization "
            "without downloading files."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main execution
# ============================================================

def main() -> int:
    """Run Phase 1 acquisition."""

    args = parse_arguments()

    selected_datasets = (
        DATASET_NAMES
        if args.dataset == "all"
        else (args.dataset,)
    )

    compute_hash = not args.skip_hash

    LOGGER.info("=" * 76)
    LOGGER.info("PHASE 1: OFFICIAL DATASET ACQUISITION")
    LOGGER.info(
        "Selected datasets: %s",
        ", ".join(selected_datasets),
    )
    LOGGER.info(
        "SHA-256 enabled: %s",
        compute_hash,
    )
    LOGGER.info(
        "Dry run: %s",
        args.dry_run,
    )
    LOGGER.info("=" * 76)

    procedures: dict[str, Callable[[], None]] = {
        "CICIDS2017": lambda: inspect_cicids2017(
            compute_hash=compute_hash
        ),
        "CSE-CIC-IDS2018": lambda: acquire_cse_cic_ids2018(
            compute_hash=compute_hash,
            dry_run=args.dry_run,
        ),
        "UNSW-NB15": lambda: inspect_unsw_nb15(
            compute_hash=compute_hash
        ),
        "BoT-IoT": lambda: inspect_bot_iot(
            compute_hash=compute_hash
        ),
    }

    technical_failures: list[str] = []

    for dataset in selected_datasets:
        LOGGER.info("-" * 76)
        LOGGER.info(
            "Processing dataset: %s",
            dataset,
        )

        try:
            procedures[dataset]()
        except Exception:
            technical_failures.append(dataset)

            LOGGER.exception(
                "Technical failure while processing %s",
                dataset,
            )

            set_dataset_status(
                dataset=dataset,
                complete=False,
                files=[],
                missing_files=["Unknown due to technical failure"],
                statuses=["TECHNICAL_FAILURE"],
                notes=(
                    "The dataset procedure raised an exception. "
                    "Review the acquisition log."
                ),
            )

    save_file_manifests()
    save_dataset_summary(selected_datasets)
    print_console_summary(selected_datasets)

    complete_datasets = [
        dataset
        for dataset in selected_datasets
        if DATASET_STATUSES.get(dataset)
        and DATASET_STATUSES[dataset].complete
    ]

    incomplete_datasets = [
        dataset
        for dataset in selected_datasets
        if not DATASET_STATUSES.get(dataset)
        or not DATASET_STATUSES[dataset].complete
    ]

    LOGGER.info("=" * 76)

    if technical_failures:
        LOGGER.error(
            "Phase 1 completed with technical failures: %s",
            ", ".join(technical_failures),
        )

        LOGGER.info(
            "Completed datasets: %s",
            ", ".join(complete_datasets)
            if complete_datasets
            else "None",
        )

        LOGGER.info(
            "Results directory: %s",
            PHASE_ROOT,
        )

        LOGGER.info("=" * 76)
        return 1

    if incomplete_datasets:
        LOGGER.warning(
            "Phase 1 completed partially."
        )

        LOGGER.warning(
            "Datasets awaiting acquisition or completion: %s",
            ", ".join(incomplete_datasets),
        )

        LOGGER.info(
            "Completed datasets: %s",
            ", ".join(complete_datasets)
            if complete_datasets
            else "None",
        )

        LOGGER.info(
            "Results directory: %s",
            PHASE_ROOT,
        )

        LOGGER.info("=" * 76)
        return 2

    LOGGER.info(
        "Phase 1 completed successfully for all selected datasets."
    )

    LOGGER.info(
        "Completed datasets: %s",
        ", ".join(complete_datasets),
    )

    LOGGER.info(
        "Results directory: %s",
        PHASE_ROOT,
    )

    LOGGER.info("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())