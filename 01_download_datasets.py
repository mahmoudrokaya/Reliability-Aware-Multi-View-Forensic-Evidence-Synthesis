"""
01_download_datasets.py

Downloads and organizes the flow-based files required for:
1. CICIDS2017
2. CSE-CIC-IDS2018
3. UNSW-NB15
4. BoT-IoT

The script:
- creates Raw, Extracted, Processed, Metadata, and Logs folders;
- downloads directly accessible official files;
- uses AWS CLI for CSE-CIC-IDS2018;
- detects datasets requiring manual download;
- records SHA-256 hashes and download status;
- avoids downloading PCAP files where possible.

Run:
    python 01_download_datasets.py

Optional:
    python 01_download_datasets.py --dataset CICIDS2017
    python 01_download_datasets.py --dataset CSE-CIC-IDS2018
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
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments"
)

DATA_ROOT = PROJECT_ROOT / "Data"
CODE_ROOT = PROJECT_ROOT / "Code"
RESULTS_ROOT = PROJECT_ROOT / "Results"

DOWNLOAD_LOG_DIR = RESULTS_ROOT / "Experiment_01_Data_Preparation" / "Download_Logs"
DOWNLOAD_LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ("CICIDS2017", "CSE-CIC-IDS2018", "UNSW-NB15", "BoT-IoT")


# ============================================================
# Official sources
# ============================================================

# CICIDS2017 official downloadable ML archive.
CICIDS2017_URL = (
    "https://www.unb.ca/cic/datasets/"
    "ids-2017.html"
)

# The direct archive URL can occasionally change at the provider.
# This URL is attempted first. If unavailable, the script creates
# manual download instructions containing the official landing page.
CICIDS2017_DIRECT_URL = (
    "https://www.unb.ca/cic/datasets/"
    "MachineLearningCSV.zip"
)

CSE_CIC_IDS2018_S3 = "s3://cse-cic-ids2018/"

UNSW_NB15_OFFICIAL_PAGE = (
    "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
)

BOT_IOT_OFFICIAL_PAGE = (
    "https://research.unsw.edu.au/projects/bot-iot-dataset"
)


# ============================================================
# Logging
# ============================================================

LOG_FILE = DOWNLOAD_LOG_DIR / "dataset_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

LOGGER = logging.getLogger("dataset_downloader")


# ============================================================
# Metadata record
# ============================================================

@dataclass
class DownloadRecord:
    dataset: str
    status: str
    source: str
    local_path: str
    size_bytes: int | None
    sha256: str | None
    timestamp_utc: str
    notes: str = ""


DOWNLOAD_RECORDS: list[DownloadRecord] = []


# ============================================================
# Folder preparation
# ============================================================

def prepare_dataset_folders(dataset: str) -> dict[str, Path]:
    """Create a consistent directory structure for one dataset."""

    base = DATA_ROOT / dataset

    folders = {
        "base": base,
        "raw": base / "Raw",
        "extracted": base / "Extracted",
        "processed": base / "Processed",
        "metadata": base / "Metadata",
        "logs": base / "Logs",
        "manual": base / "Manual_Download",
    }

    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)

    return folders


# ============================================================
# Utility functions
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def save_download_records() -> None:
    """Save the cumulative download manifest as JSON and CSV."""

    json_path = DOWNLOAD_LOG_DIR / "download_manifest.json"
    csv_path = DOWNLOAD_LOG_DIR / "download_manifest.csv"

    write_json(json_path, [asdict(record) for record in DOWNLOAD_RECORDS])

    fieldnames = list(DownloadRecord.__dataclass_fields__.keys())

    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for record in DOWNLOAD_RECORDS:
            writer.writerow(asdict(record))


def record_download(
    dataset: str,
    status: str,
    source: str,
    local_path: Path,
    notes: str = "",
) -> None:
    """Create one manifest record."""

    if local_path.is_file():
        size_bytes = local_path.stat().st_size
        sha256 = calculate_sha256(local_path)
    else:
        size_bytes = None
        sha256 = None

    DOWNLOAD_RECORDS.append(
        DownloadRecord(
            dataset=dataset,
            status=status,
            source=source,
            local_path=str(local_path),
            size_bytes=size_bytes,
            sha256=sha256,
            timestamp_utc=utc_now(),
            notes=notes,
        )
    )

    save_download_records()


def download_file(
    url: str,
    destination: Path,
    dataset: str,
    timeout: int = 60,
) -> bool:
    """
    Download a file with progress reporting.

    Existing non-empty files are retained to allow resumable workflows.
    """

    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info(
            "%s already exists: %s (%s)",
            dataset,
            destination,
            human_size(destination.stat().st_size),
        )

        record_download(
            dataset=dataset,
            status="existing",
            source=url,
            local_path=destination,
            notes="Existing file retained.",
        )
        return True

    temporary_path = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "Digital-Forensics-Research-Downloader/1.0"
            )
        },
    )

    try:
        LOGGER.info("Downloading %s from %s", dataset, url)

        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            start_time = time.time()

            destination.parent.mkdir(parents=True, exist_ok=True)

            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    elapsed = max(time.time() - start_time, 0.001)
                    speed = downloaded / elapsed

                    if total_size:
                        percentage = downloaded * 100 / total_size
                        message = (
                            f"\r{dataset}: {percentage:6.2f}% | "
                            f"{human_size(downloaded)} / "
                            f"{human_size(total_size)} | "
                            f"{human_size(int(speed))}/s"
                        )
                    else:
                        message = (
                            f"\r{dataset}: {human_size(downloaded)} | "
                            f"{human_size(int(speed))}/s"
                        )

                    print(message, end="", flush=True)

        print()
        temporary_path.replace(destination)

        LOGGER.info(
            "Downloaded %s: %s (%s)",
            dataset,
            destination,
            human_size(destination.stat().st_size),
        )

        record_download(
            dataset=dataset,
            status="downloaded",
            source=url,
            local_path=destination,
        )
        return True

    except Exception as exc:
        LOGGER.error("%s download failed: %s", dataset, exc)

        if temporary_path.exists():
            temporary_path.unlink()

        record_download(
            dataset=dataset,
            status="failed",
            source=url,
            local_path=destination,
            notes=str(exc),
        )
        return False


def extract_zip(archive: Path, destination: Path, dataset: str) -> bool:
    """Extract a ZIP archive safely."""

    if not archive.exists():
        LOGGER.error("Cannot extract missing archive: %s", archive)
        return False

    marker = destination / ".extraction_complete"

    if marker.exists():
        LOGGER.info("%s archive was already extracted.", dataset)
        return True

    try:
        LOGGER.info("Extracting %s to %s", archive, destination)
        destination.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive, "r") as zip_stream:
            for member in zip_stream.infolist():
                member_path = destination / member.filename

                # Protect against ZIP path traversal.
                if not member_path.resolve().is_relative_to(
                    destination.resolve()
                ):
                    raise RuntimeError(
                        f"Unsafe archive path detected: {member.filename}"
                    )

            zip_stream.extractall(destination)

        marker.write_text(
            f"Extracted from {archive.name} at {utc_now()}",
            encoding="utf-8",
        )

        LOGGER.info("%s extraction completed.", dataset)
        return True

    except Exception as exc:
        LOGGER.error("%s extraction failed: %s", dataset, exc)
        return False


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def write_manual_instructions(
    dataset: str,
    official_page: str,
    folders: dict[str, Path],
    expected_files: Iterable[str],
    additional_text: str = "",
) -> Path:
    """Create instructions for datasets that cannot be fetched directly."""

    instructions_path = folders["manual"] / "DOWNLOAD_INSTRUCTIONS.txt"

    expected = "\n".join(f"  - {name}" for name in expected_files)

    text = f"""
Dataset: {dataset}
Official access page:
{official_page}

Required experimental data:
{expected}

Instructions:
1. Open the official access page in a web browser.
2. Follow the provider's download link or access conditions.
3. Download only the CSV or flow-feature files required for machine learning.
4. Do not download PCAP files unless packet-level experiments are planned.
5. Place downloaded archives in:
   {folders["raw"]}

6. Extract files into:
   {folders["extracted"]}

7. Run:
   python 02_validate_datasets.py

{additional_text}

Generated: {utc_now()}
""".strip()

    instructions_path.write_text(text, encoding="utf-8")

    LOGGER.warning(
        "%s requires manual confirmation or provider-mediated download. "
        "Instructions: %s",
        dataset,
        instructions_path,
    )

    record_download(
        dataset=dataset,
        status="manual_download_required",
        source=official_page,
        local_path=instructions_path,
        notes="Use the official provider page and download flow/CSV files.",
    )

    return instructions_path


# ============================================================
# Dataset-specific download procedures
# ============================================================

def download_cicids2017() -> None:
    dataset = "CICIDS2017"
    folders = prepare_dataset_folders(dataset)

    archive = folders["raw"] / "MachineLearningCSV.zip"

    success = download_file(
        url=CICIDS2017_DIRECT_URL,
        destination=archive,
        dataset=dataset,
    )

    # Some provider configurations block direct programmatic access.
    # In that case, provide a verified manual path rather than using
    # an unofficial mirror.
    if not success or not zipfile.is_zipfile(archive):
        if archive.exists() and not zipfile.is_zipfile(archive):
            LOGGER.warning(
                "The downloaded CICIDS2017 file is not a valid ZIP archive."
            )
            archive.unlink()

        write_manual_instructions(
            dataset=dataset,
            official_page=CICIDS2017_URL,
            folders=folders,
            expected_files=[
                "MachineLearningCSV.zip",
                "or the extracted MachineLearningCSV/*.csv files",
            ],
            additional_text=(
                "On the official CICIDS2017 page, select the "
                "MachineLearningCSV.zip package."
            ),
        )
        return

    extract_zip(
        archive=archive,
        destination=folders["extracted"],
        dataset=dataset,
    )


def download_cse_cic_ids2018() -> None:
    dataset = "CSE-CIC-IDS2018"
    folders = prepare_dataset_folders(dataset)

    if not command_exists("aws"):
        write_manual_instructions(
            dataset=dataset,
            official_page=(
                "https://registry.opendata.aws/cse-cic-ids2018/"
            ),
            folders=folders,
            expected_files=[
                "Processed Traffic Data for ML Algorithms/*.csv"
            ],
            additional_text=(
                "Install AWS CLI, then rerun this script. "
                "The script downloads only CSV machine-learning files "
                "and excludes PCAP and host-log directories."
            ),
        )
        return

    destination = folders["raw"]

    command = [
        "aws",
        "s3",
        "sync",
        CSE_CIC_IDS2018_S3,
        str(destination),
        "--no-sign-request",
        "--region",
        "ca-central-1",
        "--exclude",
        "*",
        "--include",
        "*.csv",
    ]

    LOGGER.info("Running AWS download command:")
    LOGGER.info(" ".join(command))
    LOGGER.warning(
        "CSE-CIC-IDS2018 contains many large files. "
        "Only files ending in .csv will be synchronized."
    )

    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )

        aws_log = folders["logs"] / "aws_sync_output.txt"
        aws_log.write_text(
            (
                "COMMAND\n"
                + " ".join(command)
                + "\n\nSTDOUT\n"
                + completed.stdout
                + "\n\nSTDERR\n"
                + completed.stderr
            ),
            encoding="utf-8",
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"AWS CLI returned exit code {completed.returncode}. "
                f"See {aws_log}"
            )

        csv_files = list(destination.rglob("*.csv"))

        if not csv_files:
            raise RuntimeError(
                "AWS sync completed but no CSV files were found."
            )

        LOGGER.info(
            "%s: downloaded %d CSV files.",
            dataset,
            len(csv_files),
        )

        record_download(
            dataset=dataset,
            status="downloaded",
            source=CSE_CIC_IDS2018_S3,
            local_path=destination,
            notes=f"{len(csv_files)} CSV files synchronized through AWS CLI.",
        )

    except Exception as exc:
        LOGGER.error("%s download failed: %s", dataset, exc)

        write_manual_instructions(
            dataset=dataset,
            official_page=(
                "https://registry.opendata.aws/cse-cic-ids2018/"
            ),
            folders=folders,
            expected_files=[
                "Processed Traffic Data for ML Algorithms/*.csv"
            ],
            additional_text=f"AWS CLI error: {exc}",
        )


def prepare_unsw_nb15_download() -> None:
    dataset = "UNSW-NB15"
    folders = prepare_dataset_folders(dataset)

    write_manual_instructions(
        dataset=dataset,
        official_page=UNSW_NB15_OFFICIAL_PAGE,
        folders=folders,
        expected_files=[
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv",
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv",
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv",
            "UNSW-NB15_features.csv",
            "UNSW-NB15_LIST_EVENTS.csv",
        ],
        additional_text=(
            "For the primary experiments, the four original CSV partitions "
            "and the feature-description file are preferred. The predefined "
            "training and testing files may be retained for comparison."
        ),
    )


def prepare_bot_iot_download() -> None:
    dataset = "BoT-IoT"
    folders = prepare_dataset_folders(dataset)

    write_manual_instructions(
        dataset=dataset,
        official_page=BOT_IOT_OFFICIAL_PAGE,
        folders=folders,
        expected_files=[
            "BoT-IoT CSV or Argus flow-feature files",
            "dataset feature-description/documentation files",
        ],
        additional_text=(
            "Download the CSV/flow-feature representation needed for "
            "machine-learning experiments. The complete PCAP collection "
            "is unnecessary for the planned flow-based framework."
        ),
    )


# ============================================================
# Project-level metadata
# ============================================================

def write_dataset_registry() -> None:
    """Record official access pages and expected local directories."""

    registry = {
        "generated_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "datasets": {
            "CICIDS2017": {
                "official_page": CICIDS2017_URL,
                "local_directory": str(DATA_ROOT / "CICIDS2017"),
                "required_format": "MachineLearningCSV flow files",
            },
            "CSE-CIC-IDS2018": {
                "official_page": (
                    "https://registry.opendata.aws/cse-cic-ids2018/"
                ),
                "official_s3_bucket": CSE_CIC_IDS2018_S3,
                "local_directory": str(DATA_ROOT / "CSE-CIC-IDS2018"),
                "required_format": "CSV flow-feature files",
            },
            "UNSW-NB15": {
                "official_page": UNSW_NB15_OFFICIAL_PAGE,
                "local_directory": str(DATA_ROOT / "UNSW-NB15"),
                "required_format": "Original CSV feature files",
            },
            "BoT-IoT": {
                "official_page": BOT_IOT_OFFICIAL_PAGE,
                "local_directory": str(DATA_ROOT / "BoT-IoT"),
                "required_format": "CSV/Argus flow-feature files",
            },
        },
    }

    write_json(
        DOWNLOAD_LOG_DIR / "official_dataset_registry.json",
        registry,
    )


# ============================================================
# Main
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize network intrusion datasets."
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="Dataset to prepare. Default: all.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    write_dataset_registry()

    selected = DATASETS if args.dataset == "all" else (args.dataset,)

    procedures = {
        "CICIDS2017": download_cicids2017,
        "CSE-CIC-IDS2018": download_cse_cic_ids2018,
        "UNSW-NB15": prepare_unsw_nb15_download,
        "BoT-IoT": prepare_bot_iot_download,
    }

    LOGGER.info("=" * 72)
    LOGGER.info("Dataset acquisition started")
    LOGGER.info("Selected datasets: %s", ", ".join(selected))
    LOGGER.info("=" * 72)

    for dataset in selected:
        LOGGER.info("-" * 72)
        LOGGER.info("Preparing %s", dataset)

        try:
            procedures[dataset]()
        except Exception:
            LOGGER.exception("Unexpected error while preparing %s", dataset)

    save_download_records()

    LOGGER.info("=" * 72)
    LOGGER.info("Dataset acquisition stage completed.")
    LOGGER.info("Manifest: %s", DOWNLOAD_LOG_DIR / "download_manifest.csv")
    LOGGER.info("Log: %s", LOG_FILE)
    LOGGER.info("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())