"""
Explore Dataset Folder Structure

This script recursively explores a dataset directory and generates:

1. Console tree
2. CSV inventory
3. JSON inventory
4. Folder statistics
5. File statistics

Author: Mahmoud Rokaya
"""

from pathlib import Path
import json
import csv
from datetime import datetime

# ============================================================
# CHANGE THIS PATH
# ============================================================

ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments\Data"
)

OUTPUT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments\Results\Experiment_01_Data_Preparation\Dataset_Exploration"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

TREE_FILE = OUTPUT / "folder_tree.txt"
CSV_FILE = OUTPUT / "file_inventory.csv"
JSON_FILE = OUTPUT / "file_inventory.json"

# ============================================================

inventory = []

folder_count = 0
file_count = 0
total_size = 0

tree_lines = []

print("=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)

for current in sorted(ROOT.rglob("*")):

    depth = len(current.relative_to(ROOT).parts)

    indent = "    " * (depth - 1)

    if current.is_dir():

        folder_count += 1

        line = f"{indent}📁 {current.name}"

        print(line)

        tree_lines.append(line)

    else:

        file_count += 1

        size = current.stat().st_size

        total_size += size

        line = f"{indent}📄 {current.name}"

        print(line)

        tree_lines.append(line)

        inventory.append({

            "relative_path": str(current.relative_to(ROOT)),

            "filename": current.name,

            "extension": current.suffix,

            "size_bytes": size,

            "size_MB": round(size / 1024 / 1024, 3),

        })

# ============================================================
# Save tree
# ============================================================

with open(TREE_FILE, "w", encoding="utf8") as f:

    f.write("\n".join(tree_lines))

# ============================================================
# CSV inventory
# ============================================================

with open(CSV_FILE, "w", newline="", encoding="utf8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=inventory[0].keys() if inventory else []
    )

    if inventory:

        writer.writeheader()

        writer.writerows(inventory)

# ============================================================
# JSON inventory
# ============================================================

summary = {

    "generated":

        datetime.now().isoformat(),

    "root":

        str(ROOT),

    "folders":

        folder_count,

    "files":

        file_count,

    "total_size_GB":

        round(total_size / 1024 / 1024 / 1024, 3),

    "inventory":

        inventory

}

with open(JSON_FILE, "w", encoding="utf8") as f:

    json.dump(summary, f, indent=4)

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("Folders :", folder_count)
print("Files   :", file_count)
print("Size GB :", round(total_size / 1024 / 1024 / 1024, 3))

print("\nResults written to")

print(TREE_FILE)
print(CSV_FILE)
print(JSON_FILE)