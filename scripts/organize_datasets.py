#!/usr/bin/env python3
"""
Helper script to organize MVTec and VisA datasets if they're mixed in the same directory
"""

import os
import shutil
from pathlib import Path
import argparse

# Dataset categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]


def organize_datasets(data_dir, create_subfolders=True):
    """
    Organize MVTec and VisA datasets into separate folders.

    Args:
        data_dir: Path to directory containing mixed datasets
        create_subfolders: If True, create MVTec and VisA subfolders
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return

    # Check what we have
    all_items = [d.name for d in data_dir.iterdir() if d.is_dir()]

    existing_mvtec = [cat for cat in MVTEC_CATEGORIES if cat in all_items]
    existing_visa = [cat for cat in VISA_CATEGORIES if cat in all_items]

    print(f"Found {len(existing_mvtec)} MVTec categories")
    print(f"Found {len(existing_visa)} VisA categories")

    if len(existing_mvtec) == 0 and len(existing_visa) == 0:
        print("No dataset categories found")
        return

    if create_subfolders:
        # Create separate folders for each dataset
        mvtec_dir = data_dir / "MVTec"
        visa_dir = data_dir / "VisA"

        # Move MVTec categories
        if existing_mvtec and not mvtec_dir.exists():
            print(f"\nCreating MVTec folder at {mvtec_dir}")
            mvtec_dir.mkdir(exist_ok=True)

            for cat in existing_mvtec:
                source = data_dir / cat
                target = mvtec_dir / cat
                if not target.exists():
                    print(f"  Moving {cat} to MVTec/")
                    shutil.move(str(source), str(target))
                else:
                    print(f"  {cat} already in MVTec/")

        # Move VisA categories
        if existing_visa and not visa_dir.exists():
            print(f"\nCreating VisA folder at {visa_dir}")
            visa_dir.mkdir(exist_ok=True)

            for cat in existing_visa:
                source = data_dir / cat
                target = visa_dir / cat
                if not target.exists():
                    print(f"  Moving {cat} to VisA/")
                    shutil.move(str(source), str(target))
                else:
                    print(f"  {cat} already in VisA/")

        print("\nDatasets organized successfully!")
        if existing_mvtec:
            print(f"MVTec: {mvtec_dir}")
        if existing_visa:
            print(f"VisA: {visa_dir}")

    else:
        # Just report the current state
        print("\nCurrent structure (mixed):")
        print(f"Data directory: {data_dir}")
        print(f"  MVTec categories: {', '.join(existing_mvtec)}")
        print(f"  VisA categories: {', '.join(existing_visa)}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize MVTec and VisA datasets"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to directory containing the datasets"
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Create separate MVTec and VisA subfolders"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current structure, don't move anything"
    )

    args = parser.parse_args()

    if args.check_only:
        organize_datasets(args.data_dir, create_subfolders=False)
    else:
        organize_datasets(args.data_dir, create_subfolders=args.separate)


if __name__ == "__main__":
    main()