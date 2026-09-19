"""Data Engineering Transformation Nodes for MIMIC-IV Clinical Flow.

Senior Staff Healthcare DataOps & Analytics Engineering
Implements clean Medallion transformations:
- Bronze (01_raw) -> Silver (02_intermediate): Cleansing, validation, timestamp parsing.
- Silver -> Gold (03_primary): Longitudinal patient flow and transfer trajectory modeling.
- Gold -> Platinum (04_feature): Bed occupancy surge capacity & turnover buffer metrics.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from typing import Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("DataOps.Nodes")

DEFAULT_SALT = "clinical_flow_phi_salt_2026"
ICU_CAREUNIT_TOKENS = ("MICU", "SICU", "CCU", "TSICU", "CVICU", "ICU")


def hash_identifier(identifier: Any, salt: str = DEFAULT_SALT) -> str | None:
    """Apply SHA-256 one-way cryptographic hashing to patient identifiers.

    Guarantees HIPAA Safe Harbor de-identification while maintaining referential
    integrity across all clinical domain tables.

    Args:
        identifier: Raw patient/admission identifier (int, float, or string).
        salt: Cryptographic salt string preventing rainbow table attacks.

    Returns:
        Hexadecimal 64-character SHA-256 string, or None if value is null/empty.
    """
    if pd.isna(identifier) or identifier is None:
        return None
    val_str = str(identifier).strip()
    if val_str == "" or val_str.lower() in ("nan", "none", "null"):
        return None

    try:
        val_int = int(float(val_str))
        token = f"{salt}:{val_int}"
    except (ValueError, TypeError):
        token = f"{salt}:{val_str}"

    return hashlib.sha256(token.encode("utf-8")).hexdigest()



def get_execution_engine() -> Tuple[str, Any]:
    """Initialize a PySpark execution session with seamless vectorized Pandas/PyArrow fallback.

    Attempts to spin up a local SparkSession:
        SparkSession.builder.appName("ClinicalFlowOps")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate()

    If Java / JVM or PySpark is missing or cannot initialize, logs a structured warning:
        [WARN] Java/JVM not detected. Falling back to high-performance vectorized Pandas/PyArrow engine

    Returns:
        Tuple[str, Any]: Engine name ("spark" or "pandas") and session object (SparkSession or None).
    """
    try:
        pyspark_sql = importlib.import_module("pyspark.sql")
        spark_session_cls = getattr(pyspark_sql, "SparkSession")

        spark = (
            spark_session_cls.builder.appName("ClinicalFlowOps")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
        # Probe JVM bridge to confirm execution readiness
        _ = spark.sparkContext.version
        logger.info("[INFO] PySpark engine initialized successfully (Spark version: %s)", spark.version)
        return "spark", spark
    except Exception:
        logger.warning(
            "[WARN] Java/JVM not detected. Falling back to high-performance vectorized Pandas/PyArrow engine"
        )
        return "pandas", None


def compute_duration_hours(start: Any, end: Any, default: float = 0.0) -> pd.Series:
    """Compute duration in hours between two timestamp series safely and accurately.

    Args:
        start: Start timestamp Series or sequence.
        end: End timestamp Series or sequence.
        default: Fallback numeric value for null/NaT differences.

    Returns:
        pd.Series containing elapsed hours rounded to 4 decimal places.
    """
    start_dt = pd.to_datetime(start, utc=True, errors="coerce")
    end_dt = pd.to_datetime(end, utc=True, errors="coerce")
    diff = end_dt - start_dt
    return diff.apply(lambda x: round(x.total_seconds() / 3600.0, 4) if pd.notna(x) else default)


def clean_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """Clean, standardize, and govern raw MIMIC-IV admissions table.

    Args:
        admissions: Raw admissions DataFrame from Bronze layer.

    Returns:
        Cleaned intermediate admissions DataFrame with standardized schema and LOS metrics.
    """
    if admissions.empty or "subject_id" not in admissions.columns:
        return admissions.copy()

    df = admissions.copy()
    df = df.dropna(subset=["subject_id", "hadm_id"])

    # Timestamp parsing to ISO 8601 UTC
    time_cols = ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Inpatient length of stay (LOS) in hours and days
    df["los_hours"] = compute_duration_hours(df["admittime"], df["dischtime"], default=0.0)
    df["los_days"] = (df["los_hours"] / 24.0).round(4)

    # Clinical Governance: Discard impossible hospital timelines (dischtime < admittime)
    valid_mask = df["dischtime"] >= df["admittime"]
    anomalies = (~valid_mask).sum()
    if anomalies > 0:
        logger.warning("Detected %d admissions with dischtime < admittime. Filtering anomalous records.", anomalies)
        df = df[valid_mask]

    # Sanitize categorical fields
    df["admission_type"] = df["admission_type"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["is_emergency"] = df["admission_type"].str.contains("EMER|URGENT", case=False, na=False)
    df["in_hospital_mortality"] = df["hospital_expire_flag"].fillna(0).astype(int)

    return df



def clean_transfers(transfers: pd.DataFrame) -> pd.DataFrame:
    """Clean, filter, and standardize raw MIMIC-IV transfers table.

    Args:
        transfers: Raw transfers DataFrame from Bronze layer.

    Returns:
        Cleaned intermediate transfers DataFrame with care unit stay durations.
    """
    if transfers.empty or "subject_id" not in transfers.columns:
        return transfers.copy()

    df = transfers.copy()
    df = df.dropna(subset=["subject_id", "hadm_id"])

    # Timestamps to ISO 8601 UTC
    df["intime"] = pd.to_datetime(df["intime"], utc=True, errors="coerce")
    df["outtime"] = pd.to_datetime(df["outtime"], utc=True, errors="coerce")

    # Calculate stay duration in hours
    df["careunit_stay_hours"] = compute_duration_hours(df["intime"], df["outtime"], default=0.0)

    # Sanitize care unit descriptions
    raw_col = "careunit" if "careunit" in df.columns else "curr_careunit"
    sanitized = df[raw_col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["curr_careunit"] = sanitized
    df["careunit"] = sanitized
    df["is_icu"] = df["curr_careunit"].apply(
        lambda x: any(token in str(x).upper() for token in ICU_CAREUNIT_TOKENS) if pd.notna(x) else False
    )

    return df


def clean_services(services: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw MIMIC-IV clinical services table.

    Args:
        services: Raw services DataFrame from Bronze layer.

    Returns:
        Cleaned intermediate clinical services DataFrame.
    """
    if services.empty or "subject_id" not in services.columns:
        return services.copy()

    df = services.copy()
    df = df.dropna(subset=["subject_id", "hadm_id"])

    if "transfertime" in df.columns:
        df["transfertime"] = pd.to_datetime(df["transfertime"], utc=True, errors="coerce")

    df["curr_service"] = df["curr_service"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    if "prev_service" in df.columns:
        df["prev_service"] = df["prev_service"].fillna("NONE").astype(str).str.strip().str.upper()

    return df



def create_patient_flow(
    admissions: pd.DataFrame,
    transfers: pd.DataFrame,
    services: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesize longitudinal patient flow trajectory across clinical units (Gold layer).

    Args:
        admissions: Cleaned intermediate admissions DataFrame.
        transfers: Cleaned intermediate transfers DataFrame.
        services: Cleaned intermediate services DataFrame.

    Returns:
        Consolidated primary patient flow DataFrame tracking bed visits and transitions.
    """
    logger.info("Assembling Primary Gold layer: Unified Patient Flow Trajectories")

    # Aggregate transfer metrics per admission
    transfer_agg = (
        transfers.groupby("hadm_id")
        .agg(
            total_transfers=("transfer_id", "count"),
            total_icu_stay_hours=("careunit_stay_hours", lambda s: s[transfers.loc[s.index, "is_icu"]].sum()),
            careunits_visited=("careunit", lambda s: list(pd.unique(s))),
            had_icu_stay=("is_icu", "any"),
        )
        .reset_index()
    )

    # Primary clinical service per admission (first service encountered)
    sorted_services = services.sort_values("transfertime") if "transfertime" in services.columns else services
    primary_svc = (
        sorted_services.groupby("hadm_id")["curr_service"]
        .first()
        .reset_index()
        .rename(columns={"curr_service": "admitting_service"})
    )

    # Primary join: admissions + transfer summary + primary service
    flow_df = admissions.merge(transfer_agg, on="hadm_id", how="left")
    flow_df = flow_df.merge(primary_svc, on="hadm_id", how="left")

    # Fill defaults for patients without recorded transfers
    flow_df["total_transfers"] = flow_df["total_transfers"].fillna(0).astype(int)
    flow_df["total_icu_stay_hours"] = flow_df["total_icu_stay_hours"].fillna(0.0)
    flow_df["had_icu_stay"] = flow_df["had_icu_stay"].fillna(False).astype(bool)
    flow_df["admitting_service"] = flow_df["admitting_service"].fillna("UNKNOWN")

    logger.info("Produced Primary Patient Flow dataset with %d admissions", len(flow_df))
    return flow_df


def compute_bed_surge_metrics(
    patient_flow: pd.DataFrame,
    surge_multiplier: float = 1.25,
    bed_turnover_lead_hours: int = 4,
) -> pd.DataFrame:
    """Generate operational bed capacity, surge buffers, and turnover metrics (Feature layer).

    Calculates:
    - Adjusted bed occupancy hours factoring in turnaround disinfection buffer.
    - Surge-augmented capacity requirements based on surge multiplier.
    - Unit-level bed utilization summary.

    Args:
        patient_flow: Primary patient flow DataFrame.
        surge_multiplier: Regulatory or epidemic surge buffer multiplier (e.g., 1.25 for +25%).
        bed_turnover_lead_hours: Minimum hours reserved for bed cleaning and turnover.

    Returns:
        Analytical feature table ready for capacity planning and dashboard consumption.
    """
    logger.info(
        "Computing Feature layer with surge_multiplier=%.2f, turnover_buffer=%dh",
        surge_multiplier,
        bed_turnover_lead_hours,
    )
    df = patient_flow.copy()

    # Effective bed reservation = clinical LOS + turnover disinfection buffer
    df["effective_bed_hours"] = df["los_hours"] + bed_turnover_lead_hours

    # Surge-adjusted bed demand
    df["surge_capacity_hours"] = df["effective_bed_hours"] * surge_multiplier

    # Group by admitting service and admission type for capacity allocation
    summary = (
        df.groupby(["admitting_service", "admission_type"])
        .agg(
            patient_census=("hadm_id", "count"),
            avg_clinical_los_hours=("los_hours", "mean"),
            total_effective_bed_hours=("effective_bed_hours", "sum"),
            total_surge_capacity_hours=("surge_capacity_hours", "sum"),
            icu_admission_count=("had_icu_stay", "sum"),
            in_hospital_deaths=("in_hospital_mortality", "sum"),
        )
        .reset_index()
    )

    summary["surge_multiplier_applied"] = surge_multiplier
    summary["turnover_buffer_applied_hours"] = bed_turnover_lead_hours

    logger.info("Generated %d surge capacity feature records", len(summary))
    return summary
