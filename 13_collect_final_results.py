from pathlib import Path
import shutil


# ============================================================
# Paths
# ============================================================

EXPERIMENTS_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)

RESULTS_ROOT = EXPERIMENTS_ROOT / "Results"

EXP08_ROOT = RESULTS_ROOT / "Experiment_08_Final_Results"

TABLES_ROOT = EXP08_ROOT / "Tables"
REPORTS_ROOT = EXP08_ROOT / "Reports"


# ============================================================
# Files to collect
# ============================================================

FILES_TO_COPY = [
    TABLES_ROOT / "Final_Baseline_Results.csv",
    TABLES_ROOT / "Final_Clean_Ablation_Comparison.csv",
    TABLES_ROOT / "Final_Evidence_Degradation_Comparison.csv",
    TABLES_ROOT / "Final_Cross_Dataset_Results.csv",
    TABLES_ROOT / "Final_Computational_Efficiency.csv",
    TABLES_ROOT / "Final_Scalability.csv",
    TABLES_ROOT / "Final_Statistical_Performance_Summary.csv",
    TABLES_ROOT / "Final_Paired_Statistical_Tests.csv",
    TABLES_ROOT / "Final_Dataset_Assessment.csv",
    TABLES_ROOT / "Final_Claims_Evidence_Matrix.csv",
    REPORTS_ROOT / "Experiment_08_Final_Results_Report.md",
]


# ============================================================
# Copy files
# ============================================================

def main():
    print("=" * 78)
    print("COLLECTING FINAL EXPERIMENTAL RESULTS")
    print("=" * 78)

    copied = 0
    missing = 0

    for source in FILES_TO_COPY:

        if not source.exists():
            print(f"[MISSING] {source}")
            missing += 1
            continue

        destination = RESULTS_ROOT / source.name

        shutil.copy2(
            source,
            destination,
        )

        print(
            f"[COPIED] {source.name}"
        )

        copied += 1

    print("=" * 78)
    print(f"Copied files : {copied}")
    print(f"Missing files: {missing}")
    print(f"Destination  : {RESULTS_ROOT}")
    print("=" * 78)

    if missing == 0:
        print("FINAL RESULTS COLLECTION COMPLETED SUCCESSFULLY")
    else:
        print(
            "COLLECTION COMPLETED, BUT SOME FILES WERE MISSING."
        )


if __name__ == "__main__":
    main()