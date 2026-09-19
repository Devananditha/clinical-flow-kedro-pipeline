#!/usr/bin/env python3
"""Synthetic Clinical Data Provisioning for CI / GitHub Actions.

Senior Staff Healthcare DataOps & DevOps Engineering
Generates lightweight, syntactically valid, and referentially coherent
mock MIMIC-IV datasets (admissions, transfers, services) to seed 01_raw
in ephemeral runner environments where real PHI/MIMIC datasets are git-ignored.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataOps.CIFixtures")


def generate_ci_datasets(output_dir: Path | str | None = None) -> tuple[Path, Path, Path]:
    """Generate minimal, schema-accurate mock MIMIC-IV CSVs for CI pipeline validation.

    Args:
        output_dir: Destination directory (defaults to project data/01_raw).

    Returns:
        Tuple of Path objects for (admissions.csv, transfers.csv, services.csv).
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        dest = project_root / "data" / "01_raw"
    else:
        dest = Path(output_dir)

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic clinical datasets for CI runner in: %s", dest)

    # 1. Synthetic Admissions Table
    admissions_data = [
        {
            "subject_id": 10001,
            "hadm_id": 20001,
            "admittime": "2100-01-01 08:00:00",
            "dischtime": "2100-01-05 14:00:00",
            "deathtime": None,
            "admission_type": "EMERGENCY",
            "admit_provider_id": "P01ABC",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME",
            "insurance": "Medicare",
            "language": "ENGLISH",
            "marital_status": "MARRIED",
            "race": "WHITE",
            "edregtime": "2100-01-01 06:15:00",
            "edouttime": "2100-01-01 08:30:00",
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 10002,
            "hadm_id": 20002,
            "admittime": "2100-01-01 10:00:00",
            "dischtime": "2100-01-04 12:00:00",
            "deathtime": None,
            "admission_type": "URGENT",
            "admit_provider_id": "P02XYZ",
            "admission_location": "TRANSFER FROM HOSPITAL",
            "discharge_location": "SNF",
            "insurance": "Medicaid",
            "language": "ENGLISH",
            "marital_status": "SINGLE",
            "race": "BLACK/AFRICAN AMERICAN",
            "edregtime": None,
            "edouttime": None,
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 10003,
            "hadm_id": 20003,
            "admittime": "2100-01-02 07:30:00",
            "dischtime": "2100-01-06 17:00:00",
            "deathtime": None,
            "admission_type": "EW EMER.",
            "admit_provider_id": "P03MED",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME HEALTH CARE",
            "insurance": "Other",
            "language": "SPANISH",
            "marital_status": "DIVORCED",
            "race": "HISPANIC/LATINO",
            "edregtime": "2100-01-02 05:00:00",
            "edouttime": "2100-01-02 08:00:00",
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 10004,
            "hadm_id": 20004,
            "admittime": "2100-01-02 12:00:00",
            "dischtime": "2100-01-03 16:00:00",
            "deathtime": None,
            "admission_type": "ELECTIVE",
            "admit_provider_id": "P04SUR",
            "admission_location": "PHYSICIAN REFERRAL",
            "discharge_location": "HOME",
            "insurance": "Medicare",
            "language": "ENGLISH",
            "marital_status": "MARRIED",
            "race": "WHITE",
            "edregtime": None,
            "edouttime": None,
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 10005,
            "hadm_id": 20005,
            "admittime": "2100-01-03 14:00:00",
            "dischtime": "2100-01-08 11:30:00",
            "deathtime": None,
            "admission_type": "EMERGENCY",
            "admit_provider_id": "P05ICU",
            "admission_location": "EMERGENCY ROOM",
            "discharge_location": "REHAB",
            "insurance": "Medicare",
            "language": "ENGLISH",
            "marital_status": "WIDOWED",
            "race": "ASIAN",
            "edregtime": "2100-01-03 11:00:00",
            "edouttime": "2100-01-03 15:00:00",
            "hospital_expire_flag": 0,
        },
        {
            "subject_id": 10006,
            "hadm_id": 20006,
            "admittime": "2100-01-04 09:15:00",
            "dischtime": "2100-01-07 15:00:00",
            "deathtime": None,
            "admission_type": "URGENT",
            "admit_provider_id": "P06CAR",
            "admission_location": "TRANSFER FROM HOSPITAL",
            "discharge_location": "HOME",
            "insurance": "Other",
            "language": "ENGLISH",
            "marital_status": "SINGLE",
            "race": "WHITE",
            "edregtime": None,
            "edouttime": None,
            "hospital_expire_flag": 0,
        },
    ]
    df_adm = pd.DataFrame(admissions_data)

    # 2. Synthetic Transfers Table (Multistep Longitudinal Trajectories)
    transfers_data = [
        # Patient 10001: ED -> MICU -> MEDICINE
        {
            "subject_id": 10001,
            "hadm_id": 20001,
            "transfer_id": 30001,
            "eventtype": "admit",
            "careunit": "Emergency Department",
            "intime": "2100-01-01 08:00:00",
            "outtime": "2100-01-01 14:00:00",
        },
        {
            "subject_id": 10001,
            "hadm_id": 20001,
            "transfer_id": 30002,
            "eventtype": "transfer",
            "careunit": "Medical Intensive Care Unit (MICU)",
            "intime": "2100-01-01 14:00:00",
            "outtime": "2100-01-03 10:00:00",
        },
        {
            "subject_id": 10001,
            "hadm_id": 20001,
            "transfer_id": 30003,
            "eventtype": "transfer",
            "careunit": "Medicine",
            "intime": "2100-01-03 10:00:00",
            "outtime": "2100-01-05 14:00:00",
        },
        # Patient 10002: MEDICINE -> DISCHARGE LOUNGE
        {
            "subject_id": 10002,
            "hadm_id": 20002,
            "transfer_id": 30004,
            "eventtype": "admit",
            "careunit": "Medicine",
            "intime": "2100-01-01 10:00:00",
            "outtime": "2100-01-04 09:00:00",
        },
        {
            "subject_id": 10002,
            "hadm_id": 20002,
            "transfer_id": 30005,
            "eventtype": "transfer",
            "careunit": "Discharge Lounge",
            "intime": "2100-01-04 09:00:00",
            "outtime": "2100-01-04 12:00:00",
        },
        # Patient 10003: ED -> SICU -> SURGERY
        {
            "subject_id": 10003,
            "hadm_id": 20003,
            "transfer_id": 30006,
            "eventtype": "admit",
            "careunit": "Emergency Department",
            "intime": "2100-01-02 07:30:00",
            "outtime": "2100-01-02 12:30:00",
        },
        {
            "subject_id": 10003,
            "hadm_id": 20003,
            "transfer_id": 30007,
            "eventtype": "transfer",
            "careunit": "Surgical Intensive Care Unit (SICU)",
            "intime": "2100-01-02 12:30:00",
            "outtime": "2100-01-04 16:00:00",
        },
        {
            "subject_id": 10003,
            "hadm_id": 20003,
            "transfer_id": 30008,
            "eventtype": "transfer",
            "careunit": "Surgery/Trauma",
            "intime": "2100-01-04 16:00:00",
            "outtime": "2100-01-06 17:00:00",
        },
        # Patient 10004: PACU -> SURGERY
        {
            "subject_id": 10004,
            "hadm_id": 20004,
            "transfer_id": 30009,
            "eventtype": "admit",
            "careunit": "PACU",
            "intime": "2100-01-02 12:00:00",
            "outtime": "2100-01-02 18:00:00",
        },
        {
            "subject_id": 10004,
            "hadm_id": 20004,
            "transfer_id": 30010,
            "eventtype": "transfer",
            "careunit": "Surgery/Trauma",
            "intime": "2100-01-02 18:00:00",
            "outtime": "2100-01-03 16:00:00",
        },
        # Patient 10005: ED -> CVICU -> MEDICINE/CARDIOLOGY
        {
            "subject_id": 10005,
            "hadm_id": 20005,
            "transfer_id": 30011,
            "eventtype": "admit",
            "careunit": "Emergency Department",
            "intime": "2100-01-03 14:00:00",
            "outtime": "2100-01-03 19:00:00",
        },
        {
            "subject_id": 10005,
            "hadm_id": 20005,
            "transfer_id": 30012,
            "eventtype": "transfer",
            "careunit": "Cardiac Vascular Intensive Care Unit (CVICU)",
            "intime": "2100-01-03 19:00:00",
            "outtime": "2100-01-06 12:00:00",
        },
        {
            "subject_id": 10005,
            "hadm_id": 20005,
            "transfer_id": 30013,
            "eventtype": "transfer",
            "careunit": "Medicine/Cardiology",
            "intime": "2100-01-06 12:00:00",
            "outtime": "2100-01-08 11:30:00",
        },
        # Patient 10006: MEDICINE/CARDIOLOGY
        {
            "subject_id": 10006,
            "hadm_id": 20006,
            "transfer_id": 30014,
            "eventtype": "admit",
            "careunit": "Medicine/Cardiology",
            "intime": "2100-01-04 09:15:00",
            "outtime": "2100-01-07 15:00:00",
        },
    ]
    df_trf = pd.DataFrame(transfers_data)

    # 3. Synthetic Services Table
    services_data = [
        {"subject_id": 10001, "hadm_id": 20001, "transfertime": "2100-01-01 08:00:00", "prev_service": None, "curr_service": "MED"},
        {"subject_id": 10001, "hadm_id": 20001, "transfertime": "2100-01-01 14:00:00", "prev_service": "MED", "curr_service": "CMED"},
        {"subject_id": 10002, "hadm_id": 20002, "transfertime": "2100-01-01 10:00:00", "prev_service": None, "curr_service": "MED"},
        {"subject_id": 10003, "hadm_id": 20003, "transfertime": "2100-01-02 07:30:00", "prev_service": None, "curr_service": "SURG"},
        {"subject_id": 10004, "hadm_id": 20004, "transfertime": "2100-01-02 12:00:00", "prev_service": None, "curr_service": "SURG"},
        {"subject_id": 10005, "hadm_id": 20005, "transfertime": "2100-01-03 14:00:00", "prev_service": None, "curr_service": "CARDIAC"},
        {"subject_id": 10006, "hadm_id": 20006, "transfertime": "2100-01-04 09:15:00", "prev_service": None, "curr_service": "CARDIAC"},
    ]
    df_srv = pd.DataFrame(services_data)

    adm_file = dest / "admissions.csv"
    trf_file = dest / "transfers.csv"
    srv_file = dest / "services.csv"

    df_adm.to_csv(adm_file, index=False)
    df_trf.to_csv(trf_file, index=False)
    df_srv.to_csv(srv_file, index=False)

    logger.info("Successfully created: %s (%d records, %d bytes)", adm_file.name, len(df_adm), adm_file.stat().st_size)
    logger.info("Successfully created: %s (%d records, %d bytes)", trf_file.name, len(df_trf), trf_file.stat().st_size)
    logger.info("Successfully created: %s (%d records, %d bytes)", srv_file.name, len(df_srv), srv_file.stat().st_size)

    return adm_file, trf_file, srv_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV clinical datasets for CI runner.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory to write admissions.csv, transfers.csv, and services.csv",
    )
    args = parser.parse_args()
    generate_ci_datasets(args.output_dir)


if __name__ == "__main__":
    main()
