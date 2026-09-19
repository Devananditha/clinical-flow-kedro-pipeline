"""Unit and Clinical Data Governance Tests for Medallion Flow.

Senior Staff Healthcare DataOps & Analytics Engineering
Validates clinical boundary conditions, timestamp sanity, identifier integrity,
and surge capacity calculation rules.
"""

import pandas as pd
import pytest

from clinical_flow.pipelines.data_engineering.nodes import (
    clean_admissions,
    clean_transfers,
    clean_services,
    create_patient_flow,
    compute_bed_surge_metrics,
)


def test_clean_admissions_governance(sample_raw_admissions: pd.DataFrame):
    """Verify that admissions cleaning enforces key integrity and chronological sanity."""
    cleaned = clean_admissions(sample_raw_admissions)

    # 1. Null subject_id / hadm_id dropped (Row 4 has null subject_id)
    assert not cleaned["subject_id"].isna().any()
    assert not cleaned["hadm_id"].isna().any()

    # 2. Chronological anomaly dropped (Row 3 had dischtime before admittime)
    assert 2003 not in cleaned["hadm_id"].values

    # 3. All retained LOS hours must be strictly non-negative
    assert (cleaned["los_hours"] >= 0).all()

    # 4. Emergency classification
    assert cleaned.loc[cleaned["hadm_id"] == 2001, "is_emergency"].iloc[0] == True
    assert cleaned.loc[cleaned["hadm_id"] == 2002, "is_emergency"].iloc[0] == False


def test_clean_transfers_careunit_duration(sample_raw_transfers: pd.DataFrame):
    """Verify transfer data cleaning, missing hadm_id pruning, and ICU identification."""
    cleaned = clean_transfers(sample_raw_transfers)

    # Missing hadm_id dropped (transfer_id 3004)
    assert 3004 not in cleaned["transfer_id"].values
    assert len(cleaned) == 3

    # MICU identified as ICU
    micu_row = cleaned[cleaned["careunit"].str.contains("MICU")]
    assert micu_row["is_icu"].iloc[0] == True

    # Floor stay not flagged as ICU
    floor_row = cleaned[cleaned["careunit"] == "MEDICINE"]
    assert floor_row["is_icu"].iloc[0] == False

    # Stay hours non-negative
    assert (cleaned["careunit_stay_hours"] >= 0).all()


def test_clean_services_standardization(sample_raw_services: pd.DataFrame):
    """Verify service table cleaning and uppercase standardization."""
    cleaned = clean_services(sample_raw_services)
    assert (cleaned["curr_service"] == cleaned["curr_service"].str.upper()).all()
    assert len(cleaned) == 3


def test_create_patient_flow_synthesis(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
):
    """Verify synthesis of patient flow trajectory across Silver tables into Gold."""
    clean_adm = clean_admissions(sample_raw_admissions)
    clean_trf = clean_transfers(sample_raw_transfers)
    clean_srv = clean_services(sample_raw_services)

    flow = create_patient_flow(clean_adm, clean_trf, clean_srv)

    # Hadm 2001 visited ED and MICU -> 2 transfers, had_icu_stay True
    pt_2001 = flow[flow["hadm_id"] == 2001].iloc[0]
    assert pt_2001["total_transfers"] == 2
    assert pt_2001["had_icu_stay"] == True
    assert pt_2001["admitting_service"] == "MED"

    # Hadm 2002 visited Medicine -> 1 transfer, had_icu_stay False
    pt_2002 = flow[flow["hadm_id"] == 2002].iloc[0]
    assert pt_2002["total_transfers"] == 1
    assert pt_2002["had_icu_stay"] == False


def test_compute_bed_surge_metrics(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
):
    """Verify mathematical correctness of surge multiplier and bed turnover buffer."""
    clean_adm = clean_admissions(sample_raw_admissions)
    clean_trf = clean_transfers(sample_raw_transfers)
    clean_srv = clean_services(sample_raw_services)
    flow = create_patient_flow(clean_adm, clean_trf, clean_srv)

    surge_multiplier = 1.25
    turnover_lead_hours = 4

    metrics = compute_bed_surge_metrics(
        flow,
        surge_multiplier=surge_multiplier,
        bed_turnover_lead_hours=turnover_lead_hours,
    )

    assert not metrics.empty
    assert (metrics["surge_multiplier_applied"] == surge_multiplier).all()
    assert (metrics["turnover_buffer_applied_hours"] == turnover_lead_hours).all()
    # Total surge capacity hours must exceed effective bed hours by exactly surge multiplier
    for _, row in metrics.iterrows():
        expected_surge = row["total_effective_bed_hours"] * surge_multiplier
        assert pytest.approx(row["total_surge_capacity_hours"], rel=1e-4) == expected_surge
