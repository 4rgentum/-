"""Stage 3.1 stub: instructions for obtaining UNSW-NB15.

The UNSW-NB15 dataset is hosted by UNSW Canberra Cyber under their licensing
terms. This script does not download the data automatically; it prints
instructions and verifies that expected files are present.

References:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_FILES = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
]

INSTRUCTIONS = """
UNSW-NB15 download instructions
================================
1. Open the official portal:
       https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Accept the dataset terms of use.
3. Download the cleaned CSV partition (UNSW_NB15_training-set.csv,
   UNSW_NB15_testing-set.csv).
4. Place both files into:
       data/raw/UNSW-NB15/
5. Re-run this script to verify the layout.
"""


def main() -> int:
    target = Path("data/raw/UNSW-NB15")
    target.mkdir(parents=True, exist_ok=True)
    missing = [name for name in EXPECTED_FILES if not (target / name).is_file()]
    if missing:
        print(INSTRUCTIONS)
        print(f"[unsw] missing files in {target}: {missing}")
        return 1
    print(f"[unsw] all expected files present in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
