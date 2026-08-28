"""
==========================================================
Project Preparation
Digital Forensics Framework for Network Intrusions
==========================================================
"""

from pathlib import Path
import os

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments"
)

DATA_DIR = PROJECT_ROOT / "Data"
CODE_DIR = PROJECT_ROOT / "Code"
RESULTS_DIR = PROJECT_ROOT / "Results"

# ==========================================================
# Dataset folders
# ==========================================================

datasets = [
    "CICIDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT"
]

# ==========================================================
# Experiment folders
# ==========================================================

experiments = [

    "Experiment_01_Data_Preparation",

    "Experiment_02_Baseline_Models",

    "Experiment_03_Framework_Validation",

    "Experiment_04_Ablation_Study",

    "Experiment_05_Cross_Dataset_Validation",

    "Experiment_06_Computational_Efficiency",

    "Experiment_07_Statistical_Analysis",

    "Experiment_08_Final_Results"
]

# ==========================================================
# Create directories
# ==========================================================

for folder in [DATA_DIR, CODE_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Dataset folders

for ds in datasets:

    (DATA_DIR / ds).mkdir(exist_ok=True)

# Results folders

for exp in experiments:

    (RESULTS_DIR / exp).mkdir(exist_ok=True)

# Code folders

(CODE_DIR / "Utilities").mkdir(exist_ok=True)
(CODE_DIR / "Experiments").mkdir(exist_ok=True)
(CODE_DIR / "Visualization").mkdir(exist_ok=True)

print("="*60)
print("Project folders created successfully.")
print("="*60)

print("\nDatasets")

for ds in datasets:
    print(" -", DATA_DIR / ds)

print("\nExperiments")

for exp in experiments:
    print(" -", RESULTS_DIR / exp)