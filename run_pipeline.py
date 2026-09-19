#!/usr/bin/env python3
"""Clinical Flow Medallion Pipeline Execution Runner.

Senior Staff Healthcare DataOps & Analytics Engineering
Phase 2: Discrete-Event Bed Capacity & Surge Simulator + Power BI Analytical Mart Export
Executes the end-to-end Medallion data engineering pipeline across:
  - 01_raw -> 02_intermediate (HIPAA Safe Harbor De-identification & Cleansing)
  - 02_intermediate -> 03_primary (Gold Patient Flow Trajectory Modeling)
  - 03_primary -> 04_feature (Stochastic Bed Surge Simulation & Power BI Export)
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
    validate_and_ingest_bronze_to_silver,
    build_patient_flow_trajectories,
    simulate_department_surge_capacity,
    DEFAULT_SALT,
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
    return {
        "surge_multiplier": 1.25,
        "bed_turnover_lead_hours": 4,
        "standard_target_occupancy": 0.85,
        "hipaa_salt": DEFAULT_SALT,
    }


def run_pipeline() -> None:
    """Execute the full clinical flow data engineering pipeline."""
    start_time = time.time()
    logger.info("==================================================================")
    logger.info("Starting Clinical Flow Medallion Pipeline Execution (Phase 2)")
    logger.info("==================================================================")

    params = load_parameters()
    surge_multiplier = float(params.get("surge_multiplier", 1.25))
    bed_turnover_lead_hours = int(params.get("bed_turnover_lead_hours", 4))
    standard_target_occupancy = float(params.get("standard_target_occupancy", 0.85))
    hipaa_salt = str(params.get("hipaa_salt", DEFAULT_SALT))
    logger.info(
        f"Active Governance Parameters: surge_multiplier={surge_multiplier}, "
        f"bed_turnover_lead_hours={bed_turnover_lead_hours}, target_occupancy={standard_target_occupancy}, "
        f"hipaa_salt='{hipaa_salt[:12]}...'"
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

    # 2. Bronze -> Silver (HIPAA Safe Harbor De-identification & Cleansing)
    logger.info("--- [STAGE 1/3] Node 1: Bronze -> Silver De-Identification & Cleansing ---")
    int_admissions, int_transfers, int_services = validate_and_ingest_bronze_to_silver(
        admissions=adm_path,
        transfers=trf_path,
        services=srv_path,
        salt=hipaa_salt,
        output_dir=int_dir,
    )

    sample_sub_hash = int_admissions["subject_id"].iloc[0]
    sample_hadm_hash = int_admissions["hadm_id"].iloc[0]
    logger.info(
        f"Verification: SHA-256 Subject Hash: {sample_sub_hash} (len={len(sample_sub_hash)})"
    )
    logger.info(
        f"Verification: SHA-256 Hadm Hash:    {sample_hadm_hash} (len={len(sample_hadm_hash)})"
    )
    logger.info(
        f"Verification: Intermediate Row Counts: int_admissions={len(int_admissions):,}, "
        f"int_transfers={len(int_transfers):,}, int_services={len(int_services):,}"
    )

    # 3. Silver -> Gold (Patient Flow Trajectory Synthesis)
    logger.info("--- [STAGE 2/3] Node 2: Silver -> Gold Patient Flow Trajectory Modeling ---")
    prm_out_path = prm_dir / "prm_patient_flow.parquet"
    prm_patient_flow = build_patient_flow_trajectories(
        int_admissions=int_admissions,
        int_transfers=int_transfers,
        int_services=int_services,
        output_path=prm_out_path,
    )

    logger.info(
        f"Verification: Gold Primary File Created: {prm_out_path.name} "
        f"({prm_out_path.stat().st_size:,} bytes, {len(prm_patient_flow):,} rows, "
        f"{len(prm_patient_flow.columns)} columns)"
    )

    # 4. Gold -> Feature (Discrete-Event Surge Simulation & Power BI Export)
    logger.info("--- [STAGE 3/3] Node 3: Discrete-Event Bed Capacity & Surge Simulation ---")
    feat_out_path = feat_dir / "feat_bed_surge_metrics.parquet"
    csv_out_path = feat_dir / "powerbi_executive_capacity_report.csv"
    json_out_path = PROJECT_ROOT / "web" / "public" / "simulation_baseline.json"

    feat_bed_surge = simulate_department_surge_capacity(
        patient_flow=prm_patient_flow,
        surge_multiplier=surge_multiplier,
        bed_turnover_lead_hours=bed_turnover_lead_hours,
        standard_target_occupancy=standard_target_occupancy,
        output_parquet_path=feat_out_path,
        output_csv_path=csv_out_path,
        output_json_path=json_out_path,
    )

    # Clean Tabular Summary Display
    print("\n" + "=" * 105)
    print(f"{'DISCRETE-EVENT BED CAPACITY & SURGE SIMULATION SUMMARY (LEAD WINDOW = 4H)':^105}")
    print("=" * 105)
    header = f"{'Care Unit':<42} | {'Inflow (4h)':>11} | {'Discharges (4h)':>15} | {'Net Deficit':>11} | {'Risk Tier':<20}"
    print(header)
    print("-" * 105)

    for _, row in feat_bed_surge.iterrows():
        care_unit = str(row['care_unit'])[:40]
        inflow = f"{row['projected_inflow_4h']:.2f}"
        disch = f"{row['expected_discharges_4h']:.2f}"
        deficit = f"{row['surge_deficit']:+.2f}"
        tier = str(row['operational_risk_tier'])
        print(f"{care_unit:<42} | {inflow:>11} | {disch:>15} | {deficit:>11} | {tier:<20}")

    print("=" * 105 + "\n")

    logger.info(
        f"Verification: Feature Parquet Mart Created: {feat_out_path.name} "
        f"({feat_out_path.stat().st_size:,} bytes, {len(feat_bed_surge):,} care units)"
    )
    logger.info(
        f"Verification: Power BI Executive Report Exported: {csv_out_path.name} "
        f"({csv_out_path.stat().st_size:,} bytes, {len(feat_bed_surge):,} rows, "
        f"cols={list(feat_bed_surge.columns)})"
    )

    elapsed = time.time() - start_time
    logger.info("==================================================================")
    logger.info(f"Phase 2 Pipeline executed successfully in {elapsed:.2f}s!")
    logger.info("==================================================================")


if __name__ == "__main__":
    run_pipeline()
