#!/usr/bin/env python3
"""Cross-platform Data Ingestion & Provisioning Routine for MIMIC-IV Demo.

Senior Staff Healthcare DataOps & Analytics Engineering
Detects, provisions, decompresses, and validates raw clinical tables in the
Medallion 01_raw storage layer.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataOps.Ingestion")

TARGET_FILES = ["admissions.csv", "transfers.csv", "services.csv"]


def get_downloads_dir() -> Path:
    """Resolve the platform-agnostic user Downloads folder."""
    return Path.home() / "Downloads"


def provision_raw_dataset(
    source_dir: Path | None = None,
    target_dir: Path | None = None,
) -> bool:
    """Locate, transfer, decompress, and validate MIMIC-IV raw files.

    Args:
        source_dir: Directory to search for downloaded files (defaults to ~/Downloads).
        target_dir: Directory where raw files must reside (defaults to project data/01_raw).

    Returns:
        bool: True if all files exist and are non-empty, False otherwise.
    """
    project_root = Path(__file__).resolve().parent
    source = source_dir if source_dir is not None else get_downloads_dir()
    destination = target_dir if target_dir is not None else (project_root / "data" / "01_raw")

    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing MIMIC-IV raw clinical dataset provisioning")
    logger.info(f"Source search path: {source}")
    logger.info(f"Target raw path:    {destination}")

    for target_csv in TARGET_FILES:
        base_name = target_csv.replace(".csv", "")
        dest_csv_path = destination / target_csv
        dest_gz_path = destination / f"{target_csv}.gz"

        # Check if already present and valid in target
        if dest_csv_path.exists() and dest_csv_path.stat().st_size > 0:
            logger.info(f"[EXISTS] {target_csv} already present in {destination} ({dest_csv_path.stat().st_size:,} bytes)")
            continue

        # Look for sources in source_dir: .csv.gz or .csv
        src_gz = source / f"{target_csv}.gz"
        src_csv = source / target_csv

        if src_gz.exists():
            logger.info(f"[FOUND] Gzip archive detected: {src_gz} ({src_gz.stat().st_size:,} bytes)")
            logger.info(f"[EXTRACT] Decompressing {src_gz.name} -> {dest_csv_path}")
            with gzip.open(src_gz, "rb") as f_in:
                with open(dest_csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif src_csv.exists():
            logger.info(f"[FOUND] Uncompressed CSV detected: {src_csv} ({src_csv.stat().st_size:,} bytes)")
            logger.info(f"[COPY] Copying {src_csv.name} -> {dest_csv_path}")
            shutil.copy2(src_csv, dest_csv_path)
        elif dest_gz_path.exists():
            logger.info(f"[EXTRACT] Found existing gz in target: {dest_gz_path.name} -> {dest_csv_path}")
            with gzip.open(dest_gz_path, "rb") as f_in:
                with open(dest_csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            logger.error(
                f"[MISSING] Unable to locate '{target_csv}' or '{target_csv}.gz' in {source} or {destination}"
            )
            return False

    # Validation Phase
    logger.info("Executing post-ingestion validation checks on Medallion 01_raw layer...")
    all_valid = True
    for target_csv in TARGET_FILES:
        csv_file = destination / target_csv
        if not csv_file.exists():
            logger.error(f"[VALIDATION FAILED] File missing: {csv_file}")
            all_valid = False
            continue

        size = csv_file.stat().st_size
        if size == 0:
            logger.error(f"[VALIDATION FAILED] File is empty (0 bytes): {csv_file}")
            all_valid = False
            continue

        # Quick line count check
        with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
            line_count = sum(1 for _ in f)

        logger.info(
            f"[VALIDATION PASSED] {target_csv:<16} | Size: {size:>9,} bytes | Total Lines: {line_count:>6,}"
        )

    return all_valid


if __name__ == "__main__":
    if "--ci" in sys.argv or "-c" in sys.argv:
        from scripts.generate_ci_fixtures import generate_ci_datasets
        logger.info("Executing CI synthetic data provisioning...")
        generate_ci_datasets()
        logger.info("CI synthetic dataset successfully generated into data/01_raw!")
        sys.exit(0)

    success = provision_raw_dataset()
    if not success:
        logger.error("Dataset provisioning failed. Verify download location and file integrity.")
        sys.exit(1)
    logger.info("MIMIC-IV raw dataset successfully provisioned into 01_raw!")
    sys.exit(0)
