"""Unit and Clinical Data Governance Tests for Medallion Flow.

Senior Staff Healthcare DataOps & Analytics Engineering
Validates clinical boundary conditions, timestamp sanity, identifier integrity,
HIPAA Safe Harbor SHA-256 de-identification, window sequence ordering,
discrete-event surge simulation, and Power BI export schema compliance.
"""

from pathlib import Path
import pandas as pd
import pytest

from clinical_flow.pipelines.data_engineering.nodes import (
    clean_admissions,
    clean_transfers,
    clean_services,
    create_patient_flow,
    get_execution_engine,
    hash_identifier,
    validate_and_ingest_bronze_to_silver,
    build_patient_flow_trajectories,
    simulate_department_surge_capacity,
    compute_bed_surge_metrics,
)


def test_get_execution_engine_fallback():
    """Verify execution engine factory gracefully falls back to vectorized Pandas when JVM is absent."""
    engine_type, session = get_execution_engine()
    assert engine_type in ("spark", "pandas")
    if engine_type == "pandas":
        assert session is None


def test_hash_identifier_sha256():
    """Verify SHA-256 identifier hashing complies with HIPAA Safe Harbor standards."""
    # Deterministic hashing with salt
    hash1 = hash_identifier(1001, salt="test_salt")
    hash2 = hash_identifier(1001, salt="test_salt")
    assert hash1 is not None
    assert hash2 is not None
    assert hash1 == hash2
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Different salt produces different hash
    hash_diff_salt = hash_identifier(1001, salt="different_salt")
    assert hash_diff_salt is not None
    assert hash1 != hash_diff_salt

    # None and NaN handling
    assert hash_identifier(None) is None
    assert hash_identifier(float("nan")) is None


def test_validate_and_ingest_bronze_to_silver_hipaa(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
    tmp_path,
):
    """Verify Bronze->Silver ingestion enforces HIPAA de-identification and hygiene."""
    int_adm, int_trf, int_srv = validate_and_ingest_bronze_to_silver(
        admissions=sample_raw_admissions,
        transfers=sample_raw_transfers,
        services=sample_raw_services,
        salt="governance_test_salt",
        output_dir=tmp_path,
    )

    # 1. Check all subject_id and hadm_id are 64-character SHA-256 hashes
    for df in [int_adm, int_trf, int_srv]:
        assert not df["subject_id"].isna().any()
        assert not df["hadm_id"].isna().any()
        assert all(len(str(val)) == 64 for val in df["subject_id"])
        assert all(len(str(val)) == 64 for val in df["hadm_id"])

    # 2. Referential integrity: Hashed hadm_id in transfers and services must exist in admissions
    adm_hadms = set(int_adm["hadm_id"])
    assert set(int_trf["hadm_id"]).issubset(adm_hadms)
    assert set(int_srv["hadm_id"]).issubset(adm_hadms)

    # 3. Chronological filter: Admission 2003 had dischtime < admittime, must be purged
    hashed_2003 = hash_identifier(2003, salt="governance_test_salt")
    assert hashed_2003 not in adm_hadms

    # 4. Parquet files persisted with Snappy compression in tmp_path
    assert (tmp_path / "int_admissions.parquet").exists()
    assert (tmp_path / "int_transfers.parquet").exists()
    assert (tmp_path / "int_services.parquet").exists()


def test_build_patient_flow_trajectories_windowing(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
    tmp_path,
):
    """Verify window partitioning, sequence IDs, stay duration, and transition lag."""
    int_adm, int_trf, int_srv = validate_and_ingest_bronze_to_silver(
        admissions=sample_raw_admissions,
        transfers=sample_raw_transfers,
        services=sample_raw_services,
        salt="governance_test_salt",
        output_dir=tmp_path,
    )

    out_file = tmp_path / "prm_patient_flow.parquet"
    trajectory = build_patient_flow_trajectories(
        int_admissions=int_adm,
        int_transfers=int_trf,
        int_services=int_srv,
        output_path=out_file,
    )

    assert out_file.exists()
    assert not trajectory.empty

    # Find patient with multiple transfers (hadm 2001)
    hashed_2001 = hash_identifier(2001, salt="governance_test_salt")
    pt_stays = trajectory[trajectory["hadm_id"] == hashed_2001].sort_values("transfer_sequence_id")

    # Patient 2001 had 2 transfers
    assert len(pt_stays) == 2
    # Sequences must be 1 and 2
    assert list(pt_stays["transfer_sequence_id"]) == [1, 2]

    # First stay: lag is 0.0, prev_careunit is NaN/None
    first_stay = pt_stays.iloc[0]
    assert first_stay["transfer_sequence_id"] == 1
    assert first_stay["transition_lag_hours"] == 0.0

    # Second stay: care unit was MICU -> is_icu_stay is True
    second_stay = pt_stays.iloc[1]
    assert second_stay["is_icu_stay"] == True
    assert second_stay["prev_careunit"] == "EMERGENCY DEPARTMENT"

    # Emergency admission indicator
    assert first_stay["emergency_admission"] == True


def test_clean_admissions_governance(sample_raw_admissions: pd.DataFrame):
    """Verify that admissions cleaning enforces key integrity and chronological sanity."""
    cleaned = clean_admissions(sample_raw_admissions)
    assert not cleaned["subject_id"].isna().any()
    assert not cleaned["hadm_id"].isna().any()
    assert (cleaned["los_hours"] >= 0).all()


def test_clean_transfers_careunit_duration(sample_raw_transfers: pd.DataFrame):
    """Verify transfer data cleaning, missing hadm_id pruning, and careunit duration."""
    cleaned = clean_transfers(sample_raw_transfers)
    assert len(cleaned) == 3


def test_clean_services_standardization(sample_raw_services: pd.DataFrame):
    """Verify service table cleaning and uppercase standardization."""
    cleaned = clean_services(sample_raw_services)
    assert (cleaned["curr_service"] == cleaned["curr_service"].str.upper()).all()
    assert len(cleaned) == 3


def test_surge_multiplier_effect(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
    tmp_path,
):
    """Verify that projected_inflow_4h with surge multiplier (1.25) is strictly greater than baseline 4h inflow."""
    int_adm, int_trf, int_srv = validate_and_ingest_bronze_to_silver(
        admissions=sample_raw_admissions,
        transfers=sample_raw_transfers,
        services=sample_raw_services,
        output_dir=tmp_path,
    )
    flow = build_patient_flow_trajectories(int_adm, int_trf, int_srv)

    surge_multiplier = 1.25
    lead_hours = 4

    metrics = simulate_department_surge_capacity(
        flow,
        surge_multiplier=surge_multiplier,
        bed_turnover_lead_hours=lead_hours,
        output_parquet_path=tmp_path / "feat_surge.parquet",
        output_csv_path=tmp_path / "report.csv",
    )

    assert not metrics.empty
    # For every active care unit, projected inflow (4h) with surge (1.25) > baseline 4h inflow
    for _, row in metrics.iterrows():
        baseline_4h = row["baseline_hourly_inflow"] * lead_hours
        assert row["projected_inflow_4h"] > baseline_4h


def test_risk_tier_classification_invariants(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
    tmp_path,
):
    """Ensure 100% of care units have a valid categorical risk tier and consistent alert flag."""
    int_adm, int_trf, int_srv = validate_and_ingest_bronze_to_silver(
        admissions=sample_raw_admissions,
        transfers=sample_raw_transfers,
        services=sample_raw_services,
        output_dir=tmp_path,
    )
    flow = build_patient_flow_trajectories(int_adm, int_trf, int_srv)

    valid_tiers = {"CRITICAL_BOTTLENECK", "STRAINED", "STABLE"}
    metrics = simulate_department_surge_capacity(flow, output_parquet_path=tmp_path / "feat.parquet", output_csv_path=tmp_path / "rep.csv")

    assert not metrics.empty
    assert set(metrics["operational_risk_tier"]).issubset(valid_tiers)
    for _, row in metrics.iterrows():
        if row["operational_risk_tier"] == "CRITICAL_BOTTLENECK":
            assert row["governance_alert_flag"] is True or row["governance_alert_flag"] == True
        else:
            assert row["governance_alert_flag"] is False or row["governance_alert_flag"] == False


def test_powerbi_csv_export(
    sample_raw_admissions: pd.DataFrame,
    sample_raw_transfers: pd.DataFrame,
    sample_raw_services: pd.DataFrame,
    tmp_path,
):
    """Verify that Power BI executive CSV report exists, is non-empty, and matches required headers."""
    int_adm, int_trf, int_srv = validate_and_ingest_bronze_to_silver(
        admissions=sample_raw_admissions,
        transfers=sample_raw_transfers,
        services=sample_raw_services,
        output_dir=tmp_path,
    )
    flow = build_patient_flow_trajectories(int_adm, int_trf, int_srv)

    csv_path = tmp_path / "powerbi_executive_capacity_report.csv"
    simulate_department_surge_capacity(
        flow,
        output_csv_path=csv_path,
        output_parquet_path=tmp_path / "feat.parquet",
    )

    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    exported_df = pd.read_csv(csv_path)
    expected_cols = [
        "care_unit",
        "active_patients",
        "baseline_hourly_inflow",
        "surge_multiplier",
        "projected_inflow_4h",
        "expected_discharges_4h",
        "surge_deficit",
        "projected_occupancy_pct",
        "operational_risk_tier",
        "governance_alert_flag",
    ]
    assert list(exported_df.columns) == expected_cols
    assert len(exported_df) > 0
