"""
Download the official CSE-CIC-IDS2018 dataset
Author: Mahmoud Rokaya

Official Source:
https://registry.opendata.aws/cse-cic-ids2018/

The script downloads ONLY the processed CSV files
required for machine learning experiments.
"""

import subprocess
import shutil
import sys
from pathlib import Path
from datetime import datetime

# =====================================================
# Configuration
# =====================================================

AWS_REGION = "ca-central-1"

S3_SOURCE = (
    "s3://cse-cic-ids2018/"
    "Processed Traffic Data for ML Algorithms/"
)

DESTINATION = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments\Data\CSE-CIC-IDS2018\Raw"
)

LOG_DIR = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments\Results\Experiment_01_Data_Preparation"
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "CSE-CIC-IDS2018_Download.log"

# =====================================================
# Utility
# =====================================================

def log(message):

    print(message)

    with open(LOG_FILE, "a", encoding="utf8") as f:
        f.write(message + "\n")


def aws_exists():

    return shutil.which("aws") is not None


# =====================================================
# Main
# =====================================================

log("=" * 70)
log("CSE-CIC-IDS2018 Official Downloader")
log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 70)

if not aws_exists():

    log("ERROR")
    log("AWS CLI is not installed.")
    log("Install it from:")
    log("https://aws.amazon.com/cli/")
    sys.exit()

# -----------------------------------------------------

log("\nChecking access to official bucket...\n")

command = [
    "aws",
    "s3",
    "ls",
    S3_SOURCE,
    "--no-sign-request",
    "--region",
    AWS_REGION,
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True
)

if result.returncode != 0:

    log(result.stderr)

    raise RuntimeError(
        "Cannot access official bucket."
    )

log(result.stdout)

# -----------------------------------------------------

DESTINATION.mkdir(
    parents=True,
    exist_ok=True
)

log("\nStarting synchronization...\n")

command = [

    "aws",

    "s3",

    "sync",

    S3_SOURCE,

    str(DESTINATION),

    "--no-sign-request",

    "--region",

    AWS_REGION,

    "--exclude",

    "*",

    "--include",

    "*.csv"

]

process = subprocess.Popen(command)

process.wait()

if process.returncode != 0:

    raise RuntimeError(
        "Download failed."
    )

# -----------------------------------------------------

total_size = 0
num_files = 0

for f in DESTINATION.rglob("*.csv"):

    num_files += 1
    total_size += f.stat().st_size

log("\nDownload Finished Successfully")
log(f"CSV Files : {num_files}")
log(f"Total Size: {total_size/1024/1024/1024:.2f} GB")

log("=" * 70)