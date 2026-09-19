#!/usr/bin/env python3
"""Clinical Flow Medallion Pipeline Execution Runner.

Senior Staff Healthcare DataOps & Analytics Engineering
Executes the end-to-end Medallion data engineering pipeline across:
  - 01_raw -> 02_intermediate (Silver Cleansing)
  - 02_intermediate -> 03_primary (Gold Patient Flow Synthesis)
  - 03_primary -> 04_feature (Surge Capacity & Turnover Analytics)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Add src to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical_flow.pipelines.data_engineering.nodes import (
    clean_admissions,
    clean_transfers,
    clean_services,
    create_patient_flow,
    compute_bed_surge_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataOps.PipelineRunner")


def load_parameters() -> dict:
    """Load parameter configuration from conf/base/parameters.yml."""
    param_file = PROJECT_ROOT / "conf" / "base" / "parameters.yml"
    if param_file.exists():
        with open(param_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning("parameters.yml not found, using default configuration")
    return {"surge_multiplier": 1.25, "bed_turnover_lead_hours": 4}


def run_pipeline() -> None:
    """Execute the full clinical flow data engineering pipeline."""
    start_time = time.time()
    logger.info("==================================================================")
    logger.info("Starting Clinical Flow Medallion Pipeline Execution")
    logger.info("==================================================================")

    params = load_parameters()
    surge_multiplier = params.get("surge_multiplier", 1.25)
    bed_turnover_lead_hours = params.get("bed_turnover_lead_hours", 4)
    logger.info(
        f"Active Governance Parameters: surge_multiplier={surge_multiplier}, "
        f"bed_turnover_lead_hours={bed_turnover_lead_hours}"
    )

    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "01_raw"
    int_dir = data_dir / "02_intermediate"
    prm_dir = data_dir / "03_primary"
    feat_dir = data_dir / "04_feature"

    for d in [raw_dir, int_dir, prm_dir, feat_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Bronze / 01_raw datasets
    adm_path = raw_dir / "admissions.csv"
    trf_path = raw_dir / "transfers.csv"
    srv_path = raw_dir / "services.csv"

    for p in [adm_path, trf_path, srv_path]:
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing required raw file: {p}. Run 'python setup_data.py' first."
            )

    logger.info(f"Loading raw admissions from: {adm_path.name}")
    raw_admissions = pd.read_csv(adm_path)
    logger.info(f"Loading raw transfers from:  {trf_path.name}")
    raw_transfers = pd.read_csv(trf_path)
    logger.info(f"Loading raw services from:   {srv_path.name}")
    raw_services = pd.read_csv(srv_path)

    # 2. Bronze -> Silver (02_intermediate)
    logger.info("--- [STAGE 1/3] Bronze -> Silver Cleansing & Standardization ---")
    int_admissions = clean_admissions(raw_admissions)
    int_transfers = clean_transfers(raw_transfers)
    int_services = clean_services(raw_services)

    int_adm_out = int_dir / "int_admissions.parquet"
    int_trf_out = int_dir / "int_transfers.parquet"
    int_srv_out = int_dir / "int_services.parquet"

    int_admissions.to_parquet(int_adm_out, index=False)
    int_transfers.to_parquet(int_trf_out, index=False)
    int_services.to_parquet(int_srv_out, index=False)
    logger.info(f"Persisted Silver: {int_adm_out.name} ({len(int_admissions):,} rows)")
    logger.info(f"Persisted Silver: {int_trf_out.name} ({len(int_transfers):,} rows)")
    logger.info(f"Persisted Silver: {int_srv_out.name} ({len(int_services):,} rows)")

    # 3. Silver -> Gold (03_primary)
    logger.info("--- [STAGE 2/3] Silver -> Gold Patient Flow Synthesis ---")
    prm_patient_flow = create_patient_flow(int_admissions, int_transfers, int_services)
    prm_out = prm_dir / "prm_patient_flow.parquet"
    prm_patient_flow.to_parquet(prm_out, index=False)
    logger.info(f"Persisted Gold:   {prm_out.name} ({len(prm_patient_flow):,} admissions)")

    # 4. Gold -> Platinum (04_feature)
    logger.info("--- [STAGE 3/3] Gold -> Feature Operational Surge Modeling ---")
    feat_bed_surge = compute_bed_surge_metrics(
        prm_patient_flow,
        surge_multiplier=surge_multiplier,
        bed_turnover_lead_hours=bed_turnover_lead_hours,
    )
    feat_out = feat_dir / "feat_bed_surge_metrics.parquet"
    feat_bed_surge.to_parquet(feat_out, index=False)
    logger.info(f"Persisted Feature: {feat_out.name} ({len(feat_bed_surge):,} feature cohorts)")

    elapsed = time.time() - start_time
    logger.info("==================================================================")
    logger.info(f"Pipeline executed successfully in {elapsed:.2f}s!")
    logger.info("==================================================================")


if __name__ == "__main__":
    run_pipeline()
