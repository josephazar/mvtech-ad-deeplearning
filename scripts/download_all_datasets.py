#!/usr/bin/env python3
"""
Dataset Downloader for Industrial Anomaly Detection
Downloads publicly available datasets mentioned in the manuscript:
- MVTec-AD (publicly available)
- VisA (publicly available)

Note: BTAD, MPDD, WFDD, and VAD require special permissions or manual download
"""

import os
import sys
import tarfile
import zipfile
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm
import hashlib
import subprocess

# Dataset Information - Only publicly available datasets
DATASETS_INFO = {
    "mvtec": {
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "categories": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                      "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                      "transistor", "wood", "zipper"],
        "folder_name": "mvtec_anomaly_detection",
        "format": "tar.xz"
    },
    "visa": {
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "categories": ["candle", "capsules", "cashew", "chewinggum", "fryum",
                      "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
        "folder_name": "VisA",
        "format": "tar"
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)




def extract_archive(archive_path, extract_to, archive_format):
    """Extract archive based on format."""
    print(f"Extracting {archive_path}...")

    # Use local temp directory if extracting to Google Drive
    use_temp = "/content/drive" in str(extract_to)
    if use_temp:
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="dataset_extract_")
        actual_extract_to = Path(temp_dir)
        print(f"Using temporary directory for extraction: {temp_dir}")
    else:
        actual_extract_to = extract_to

    try:
        if archive_format == "tar.xz":
            with tarfile.open(archive_path, 'r:xz') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting"):
                    tar.extract(member, path=actual_extract_to)
        elif archive_format == "tar":
            with tarfile.open(archive_path, 'r') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting"):
                    tar.extract(member, path=actual_extract_to)
        elif archive_format == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                    zip_ref.extract(member, actual_extract_to)
        else:
            raise ValueError(f"Unknown archive format: {archive_format}")

        # If we used temp directory, move to final location
        if use_temp:
            print(f"Moving extracted files to {extract_to}...")
            import shutil

            # Find the actual dataset folder in temp directory
            extracted_items = list(actual_extract_to.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                # Single folder extracted, move its contents
                source_dir = extracted_items[0]
                target_dir = extract_to / source_dir.name

                if target_dir.exists():
                    print(f"Removing existing {target_dir}")
                    shutil.rmtree(target_dir)

                print(f"Moving {source_dir} to {target_dir}")
                shutil.move(str(source_dir), str(target_dir))
            else:
                # Multiple items, move all
                for item in extracted_items:
                    target = extract_to / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))

            # Clean up temp directory
            shutil.rmtree(temp_dir)
            print("Temporary directory cleaned up")

    except Exception as e:
        print(f"Error during extraction: {e}")
        if use_temp and 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def download_mvtec(data_dir, force_download=False):
    """Download MVTec-AD dataset."""
    dataset_info = DATASETS_INFO["mvtec"]
    dataset_path = data_dir / dataset_info["folder_name"]

    if dataset_path.exists() and not force_download:
        print(f"MVTec-AD already exists at {dataset_path}")
        return dataset_path

    archive_path = data_dir / f"mvtec.tar.xz"

    if not archive_path.exists() or force_download:
        print("Downloading MVTec-AD dataset...")
        download_file(dataset_info["url"], str(archive_path), "MVTec-AD")

    extract_archive(archive_path, data_dir, dataset_info["format"])

    # Clean up
    if archive_path.exists():
        archive_path.unlink()

    return dataset_path


def download_visa(data_dir, force_download=False):
    """Download VisA dataset."""
    dataset_info = DATASETS_INFO["visa"]
    dataset_path = data_dir / dataset_info["folder_name"]

    if dataset_path.exists() and not force_download:
        print(f"VisA already exists at {dataset_path}")
        return dataset_path

    archive_path = data_dir / "visa.tar"

    if not archive_path.exists() or force_download:
        print("Downloading VisA dataset...")
        if dataset_info["url"]:
            download_file(dataset_info["url"], str(archive_path), "VisA")
        else:
            print("Alternative: Using wget to download VisA...")
            subprocess.run([
                "wget", "-O", str(archive_path),
                "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
            ])

    extract_archive(archive_path, data_dir, dataset_info["format"])

    # Clean up
    if archive_path.exists():
        archive_path.unlink()

    return dataset_path




def download_all_datasets(data_dir, datasets_to_download=None, force_download=False):
    """
    Download all publicly available datasets.

    Args:
        data_dir: Directory to save datasets
        datasets_to_download: List of dataset names to download (None for all)
        force_download: Force re-download even if exists

    Returns:
        Dictionary of dataset paths
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if datasets_to_download is None:
        datasets_to_download = list(DATASETS_INFO.keys())

    dataset_paths = {}

    for dataset_name in datasets_to_download:
        if dataset_name not in DATASETS_INFO:
            print(f"Skipping {dataset_name} - requires special permissions or manual download")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} Dataset")
        print(f"{'='*60}")

        if dataset_name == "mvtec":
            dataset_paths["mvtec"] = download_mvtec(data_dir, force_download)
        elif dataset_name == "visa":
            dataset_paths["visa"] = download_visa(data_dir, force_download)
        else:
            print(f"Unknown dataset: {dataset_name}")

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, path in dataset_paths.items():
        if path:
            status = "✓ Downloaded" if path.exists() else "✗ Not Found"
            print(f"{name.upper():10} -> {status:15} {path}")
        else:
            print(f"{name.upper():10} -> Manual download required")

    return dataset_paths


def setup_colab_environment(use_drive=True):
    """Setup environment for Google Colab.

    Args:
        use_drive: If True, try to use Google Drive. If False, use local storage.
    """
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab environment")

        # Mount Google Drive if requested
        if use_drive:
            from google.colab import drive
            drive_mounted = False
            try:
                drive.mount('/content/drive')
                drive_mounted = True
                print("Google Drive mounted at /content/drive")

                # Set data directory in Drive
                data_dir = "/content/drive/MyDrive/anomaly_detection_data"
                print(f"Will save datasets to Google Drive: {data_dir}")
            except Exception as e:
                print(f"Could not mount Google Drive: {e}")
                print("Falling back to local storage.")
                data_dir = "/content/anomaly_detection_data"
        else:
            print("Using local Colab storage (faster but temporary)")
            data_dir = "/content/anomaly_detection_data"
            drive_mounted = False

        return data_dir

    except ImportError:
        print("Running in local environment")
        return "./datasets"


def main():
    parser = argparse.ArgumentParser(
        description="Download industrial anomaly detection datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save datasets (default: auto-detect)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mvtec", "visa", "all"],
        default=["all"],
        help="Datasets to download (only publicly available)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local storage in Colab instead of Google Drive (faster but temporary)"
    )

    args = parser.parse_args()

    # Auto-detect environment and set data directory
    if args.data_dir is None:
        args.data_dir = setup_colab_environment(use_drive=not args.use_local)

    print(f"Data directory: {args.data_dir}")

    # Determine which datasets to download
    if "all" in args.datasets:
        datasets_to_download = None  # Download all
    else:
        datasets_to_download = args.datasets

    # Download datasets
    dataset_paths = download_all_datasets(
        args.data_dir,
        datasets_to_download,
        args.force
    )

    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("Downloaded publicly available datasets:")
    print("- MVTec-AD: 15 categories")
    print("- VisA: 12 categories")
    print("\nNote: BTAD, MPDD, WFDD, and VAD require special permissions")
    print("You can now use these dataset paths in your evaluation scripts.")

    return dataset_paths


if __name__ == "__main__":
    dataset_paths = main()