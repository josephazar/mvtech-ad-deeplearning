#!/usr/bin/env python3
"""
MVTec-AD Dataset Downloader
Downloads and organizes the MVTec Anomaly Detection dataset.
Compatible with Google Colab and local environments.
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm
import hashlib

# MVTec-AD dataset information
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
MVTEC_MD5 = "c1e2f0e1a0b94eb8c183f0e445c4e0d0"


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


def verify_md5(file_path, expected_md5):
    """Verify MD5 checksum of downloaded file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5


def extract_tar_xz(tar_path, extract_to, desc="Extracting"):
    """Extract tar.xz file with progress bar."""
    with tarfile.open(tar_path, 'r:xz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=desc):
            tar.extract(member, path=extract_to)


def download_mvtec_ad(data_dir="./datasets", force_download=False):
    """
    Download and extract MVTec-AD dataset.

    Args:
        data_dir: Directory to save the dataset
        force_download: Force re-download even if dataset exists

    Returns:
        Path to extracted dataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    mvtec_dir = data_dir / "mvtec_anomaly_detection"
    tar_file = data_dir / "mvtec_anomaly_detection.tar.xz"

    # Check if dataset already exists
    if mvtec_dir.exists() and not force_download:
        print(f"Dataset already exists at {mvtec_dir}")
        # Verify it has all categories
        existing_categories = [d.name for d in mvtec_dir.iterdir() if d.is_dir()]
        if set(MVTEC_CATEGORIES).issubset(set(existing_categories)):
            print("All categories present. Skipping download.")
            return mvtec_dir
        else:
            print("Some categories missing. Re-downloading...")

    # Download dataset
    if not tar_file.exists() or force_download:
        print(f"Downloading MVTec-AD dataset from {MVTEC_URL}")
        print(f"This may take a while (dataset size ~4.9 GB)...")

        try:
            download_file(MVTEC_URL, str(tar_file), "Downloading MVTec-AD")
        except Exception as e:
            print(f"Error downloading from primary URL: {e}")
            print("Trying alternative download method...")

            # Alternative: wget command for Colab
            import subprocess
            result = subprocess.run(
                ["wget", "-O", str(tar_file), MVTEC_URL],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Download failed. Please download manually from:")
                print("https://www.mvtec.com/company/research/datasets/mvtec-ad")
                sys.exit(1)

        print("Download complete. Verifying checksum...")
        if not verify_md5(tar_file, MVTEC_MD5):
            print("Warning: MD5 checksum does not match. File may be corrupted.")

    # Extract dataset
    if not mvtec_dir.exists() or force_download:
        print(f"Extracting dataset to {data_dir}")
        extract_tar_xz(tar_file, data_dir, "Extracting MVTec-AD")
        print("Extraction complete.")

        # Clean up tar file to save space
        if tar_file.exists():
            print("Removing tar file to save space...")
            tar_file.unlink()

    # Verify extraction
    extracted_categories = [d.name for d in mvtec_dir.iterdir() if d.is_dir()]
    if not set(MVTEC_CATEGORIES).issubset(set(extracted_categories)):
        missing = set(MVTEC_CATEGORIES) - set(extracted_categories)
        print(f"Warning: Missing categories after extraction: {missing}")
    else:
        print(f"Successfully downloaded and extracted all {len(MVTEC_CATEGORIES)} categories")

    # Print dataset structure
    print("\nDataset structure:")
    for category in MVTEC_CATEGORIES:
        cat_path = mvtec_dir / category
        if cat_path.exists():
            train_path = cat_path / "train"
            test_path = cat_path / "test"
            gt_path = cat_path / "ground_truth"

            n_train = len(list(train_path.rglob("*.png"))) if train_path.exists() else 0
            n_test = len(list(test_path.rglob("*.png"))) if test_path.exists() else 0

            print(f"  {category}: {n_train} train, {n_test} test images")

    return mvtec_dir


def setup_colab_environment():
    """Setup environment for Google Colab."""
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab environment")

        # Mount Google Drive if needed
        from google.colab import drive
        drive_mounted = False
        try:
            drive.mount('/content/drive')
            drive_mounted = True
            print("Google Drive mounted at /content/drive")
        except:
            print("Could not mount Google Drive. Using local storage.")

        # Set default data directory
        if drive_mounted:
            data_dir = "/content/drive/MyDrive/mvtec_data"
        else:
            data_dir = "/content/mvtec_data"

        return data_dir

    except ImportError:
        print("Running in local environment")
        return "./datasets"


def main():
    parser = argparse.ArgumentParser(description="Download MVTec-AD dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save dataset (default: auto-detect based on environment)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )

    args = parser.parse_args()

    # Auto-detect environment and set data directory
    if args.data_dir is None:
        args.data_dir = setup_colab_environment()

    print(f"Data directory: {args.data_dir}")

    # Download dataset
    dataset_path = download_mvtec_ad(args.data_dir, args.force)

    print(f"\nDataset ready at: {dataset_path}")
    print("\nYou can now use this path in your training/evaluation scripts:")
    print(f"  --data-path {dataset_path}")

    return dataset_path


if __name__ == "__main__":
    dataset_path = main()