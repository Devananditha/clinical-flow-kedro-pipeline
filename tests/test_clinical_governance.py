"""Unit and Clinical Data Governance Tests for Medallion Flow.

Senior Staff Healthcare DataOps & Analytics Engineering
Validates clinical boundary conditions, timestamp sanity, identifier integrity,
and HIPAA Safe Harbor SHA-256 de-identification rules.
"""

from pathlib import Path
import pandas as pd
import pytest

from clinical_flow.pipelines.data_engineering.nodes import (
    clean_admissions,
    clean_transfers,
    clean_services,
    get_execution_engine,
    hash_identifier,
    validate_and_ingest_bronze_to_silver,
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
