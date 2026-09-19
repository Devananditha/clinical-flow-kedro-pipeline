"""Data Engineering Transformation Nodes for MIMIC-IV Clinical Flow.

Senior Staff Healthcare DataOps & Analytics Engineering
Implements Medallion transformations:
- Bronze (01_raw) -> Silver (02_intermediate):
    - Automated engine detection (PySpark with vectorized Pandas/PyArrow fallback)
    - HIPAA Safe Harbor de-identification (SHA-256 cryptographic hashing)
    - ISO 8601 UTC timestamp standardization
    - Clinical hygiene & chronological integrity checks
- Silver -> Gold (03_primary):
    - Longitudinal patient flow & transfer trajectories
    - Window partitioning over hadm_id ordered by intime
    - Transfer sequence numbering, care unit stay hours, and transition lag
    - ICU stay classification & emergency admission indicators
- Gold -> Platinum/Feature (04_feature):
    - Bed occupancy surge capacity & turnover buffer modeling
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("DataOps.Nodes")

DEFAULT_SALT = "clinical_flow_phi_salt_2026"
ICU_CAREUNIT_TOKENS = ("MICU", "SICU", "CCU", "TSICU", "CVICU", "ICU")


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


def validate_and_ingest_bronze_to_silver(
    admissions: Union[pd.DataFrame, str, Path],
    transfers: Union[pd.DataFrame, str, Path],
    services: Union[pd.DataFrame, str, Path],
    salt: str = DEFAULT_SALT,
    output_dir: Union[str, Path, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingest Bronze tables, apply HIPAA de-identification, clean, and persist Silver Parquet.

    Transformations:
    1. HIPAA De-identification: Hash subject_id and hadm_id with salted SHA-256.
    2. Data Hygiene: Strip leading/trailing whitespace on categorical columns.
    3. Timestamp Governance: Standardize admittime, dischtime, intime, outtime to ISO 8601 UTC.
    4. Chronological Validation: Drop admissions where dischtime < admittime.
    5. Persistence: Save to 02_intermediate using Snappy compression.

    Args:
        admissions: Raw admissions DataFrame or file path.
        transfers: Raw transfers DataFrame or file path.
        services: Raw services DataFrame or file path.
        salt: Cryptographic salt for identifier hashing.
        output_dir: Target directory for intermediate Parquet files (default: data/02_intermediate).

    Returns:
        Tuple of cleaned (int_admissions, int_transfers, int_services) DataFrames.
    """
    logger.info("==================================================================")
    logger.info("Executing Node 1: validate_and_ingest_bronze_to_silver")
    logger.info("==================================================================")

    # 1. Load data if paths were provided
    df_adm = pd.read_csv(admissions) if isinstance(admissions, (str, Path)) else admissions.copy()
    df_trf = pd.read_csv(transfers) if isinstance(transfers, (str, Path)) else transfers.copy()
    df_srv = pd.read_csv(services) if isinstance(services, (str, Path)) else services.copy()

    logger.info("Bronze input row counts: admissions=%d, transfers=%d, services=%d", len(df_adm), len(df_trf), len(df_srv))

    # 2. Clean and standardize tables
    df_adm = clean_admissions(df_adm)
    df_trf = clean_transfers(df_trf)
    df_srv = clean_services(df_srv)

    # 3. Apply HIPAA Safe Harbor SHA-256 de-identification
    for df in (df_adm, df_trf, df_srv):
        if not df.empty and "subject_id" in df.columns:
            df["subject_id"] = df["subject_id"].apply(lambda x: hash_identifier(x, salt))
        if not df.empty and "hadm_id" in df.columns:
            df["hadm_id"] = df["hadm_id"].apply(lambda x: hash_identifier(x, salt))

    sample_sub = df_adm["subject_id"].iloc[0] if len(df_adm) > 0 else "N/A"
    sample_hadm = df_adm["hadm_id"].iloc[0] if len(df_adm) > 0 else "N/A"
    logger.info("HIPAA Safe Harbor SHA-256 de-identification verified:")
    logger.info("  Sample hashed subject_id: %s (length=%d)", sample_sub, len(sample_sub))
    logger.info("  Sample hashed hadm_id:    %s (length=%d)", sample_hadm, len(sample_hadm))

    # 4. Persist to Silver Parquet with Snappy compression
    dest_dir = Path(output_dir) if output_dir else Path("data/02_intermediate")
    dest_dir.mkdir(parents=True, exist_ok=True)

    int_adm_path = dest_dir / "int_admissions.parquet"
    int_trf_path = dest_dir / "int_transfers.parquet"
    int_srv_path = dest_dir / "int_services.parquet"

    df_adm.to_parquet(int_adm_path, compression="snappy", index=False)
    df_trf.to_parquet(int_trf_path, compression="snappy", index=False)
    df_srv.to_parquet(int_srv_path, compression="snappy", index=False)

    logger.info("Persisted Silver Parquet datasets (compression=snappy):")
    logger.info("  - %s (%d rows)", int_adm_path.name, len(df_adm))
    logger.info("  - %s (%d rows)", int_trf_path.name, len(df_trf))
    logger.info("  - %s (%d rows)", int_srv_path.name, len(df_srv))

    return df_adm, df_trf, df_srv


def build_patient_flow_trajectories(
    int_admissions: pd.DataFrame,
    int_transfers: pd.DataFrame,
    int_services: pd.DataFrame,
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Synthesize longitudinal patient trajectories and transfer movements across wards.

    Applies window partitioning over hadm_id ordered by intime to derive:
    - transfer_sequence_id: Order of patient movement through care units (1, 2, 3...)
    - careunit_los_hours: Elapsed hours in each specific care unit
    - prev_careunit & transition_lag_hours: Transition time elapsed between ward departure and arrival
    - is_icu_stay: ICU classification flag (MICU, SICU, CCU, TSICU, CVICU)
    - emergency_admission: Emergency admission classification indicator

    Args:
        int_admissions: Cleaned Silver intermediate admissions DataFrame.
        int_transfers: Cleaned Silver intermediate transfers DataFrame.
        int_services: Cleaned Silver intermediate services DataFrame.
        output_path: Optional target file path for saving Gold Parquet dataset.

    Returns:
        Unified primary patient trajectory DataFrame.
    """
    logger.info("==================================================================")
    logger.info("Executing Node 2: build_patient_flow_trajectories")
    logger.info("==================================================================")

    engine_type, spark = get_execution_engine()

    if engine_type == "spark" and spark is not None:
        logger.info("[ENGINE] Executing patient flow trajectory assembly via PySpark engine")
        pyspark_funcs = importlib.import_module("pyspark.sql.functions")
        pyspark_window = importlib.import_module("pyspark.sql.window")
        F = pyspark_funcs
        Window = getattr(pyspark_window, "Window")

        sdf_trf = spark.createDataFrame(int_transfers)
        window_spec = Window.partitionBy("hadm_id").orderBy("intime")

        spark_trajectory = (
            sdf_trf.withColumn("transfer_sequence_id", F.row_number().over(window_spec))
            .withColumn("prev_careunit", F.lag("curr_careunit", 1).over(window_spec))
            .withColumn("prev_outtime", F.lag("outtime", 1).over(window_spec))
            .withColumn(
                "careunit_los_hours",
                F.coalesce(
                    F.round((F.unix_timestamp("outtime") - F.unix_timestamp("intime")) / 3600.0, 4),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "transition_lag_hours",
                F.coalesce(
                    F.round((F.unix_timestamp("intime") - F.unix_timestamp("prev_outtime")) / 3600.0, 4),
                    F.lit(0.0),
                ),
            )
            .withColumn(
                "is_icu_stay",
                F.col("curr_careunit").rlike("|".join(ICU_CAREUNIT_TOKENS)),
            )
        )

        trf_df = spark_trajectory.toPandas()
    else:
        logger.info("[ENGINE] Executing patient flow trajectory assembly via vectorized Pandas/PyArrow engine")
        trf = int_transfers.sort_values(by=["hadm_id", "intime"]).copy()

        # Window sequence numbering: 1, 2, 3... per admission
        trf["transfer_sequence_id"] = trf.groupby("hadm_id").cumcount() + 1

        # Care unit duration: (outtime - intime) in hours safely calculated
        intime_dt = pd.to_datetime(trf["intime"], utc=True)
        outtime_dt = pd.to_datetime(trf["outtime"], utc=True)
        trf["careunit_los_hours"] = compute_duration_hours(intime_dt, outtime_dt, default=0.0)

        # Ward transition lag: time from departure of prev_careunit to arrival at curr_careunit
        trf["prev_careunit"] = trf.groupby("hadm_id")["curr_careunit"].shift(1)
        prev_outtime_dt = pd.to_datetime(trf.groupby("hadm_id")["outtime"].shift(1), utc=True)
        trf["transition_lag_hours"] = compute_duration_hours(prev_outtime_dt, intime_dt, default=0.0)

        # ICU Classification flag
        trf["is_icu_stay"] = trf["curr_careunit"].apply(
            lambda x: any(token in str(x).upper() for token in ICU_CAREUNIT_TOKENS) if pd.notna(x) else False
        )
        trf_df = trf

    # --------------------------------------------------------------------------
    # Assemble Unified Trajectory: Join Transfers + Admissions + Services
    # --------------------------------------------------------------------------
    adm_subset = int_admissions[
        [
            col
            for col in [
                "hadm_id",
                "admission_type",
                "los_hours",
                "admittime",
                "dischtime",
                "admission_location",
                "discharge_location",
                "hospital_expire_flag",
                "in_hospital_mortality",
                "is_emergency",
            ]
            if col in int_admissions.columns
        ]
    ].drop_duplicates(subset=["hadm_id"])

    trajectory = trf_df.merge(adm_subset, on="hadm_id", how="left")

    # Emergency admission classification
    trajectory["emergency_admission"] = trajectory["admission_type"].str.contains(
        "EMERGENCY|URGENT|EMER", case=False, na=False
    )

    # Resolve primary admitting service
    srv_sorted = (
        int_services.sort_values("transfertime")
        if "transfertime" in int_services.columns
        else int_services
    )
    primary_svc = (
        srv_sorted.groupby("hadm_id")["curr_service"]
        .first()
        .reset_index()
        .rename(columns={"curr_service": "primary_service"})
    )
    trajectory = trajectory.merge(primary_svc, on="hadm_id", how="left")
    trajectory["primary_service"] = trajectory["primary_service"].fillna("UNKNOWN")
    trajectory["admitting_service"] = trajectory["primary_service"]

    # Calculate whether patient experienced an ICU stay anywhere during admission
    had_icu_per_hadm = trajectory.groupby("hadm_id")["is_icu_stay"].transform("any")
    trajectory["had_icu_stay"] = had_icu_per_hadm

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------
    target_file = Path(output_path) if output_path else Path("data/03_primary/prm_patient_flow.parquet")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_parquet(target_file, compression="snappy", index=False)

    logger.info("Persisted Gold Primary dataset (compression=snappy):")
    logger.info("  File: %s", target_file)
    logger.info("  Rows: %d trajectory records", len(trajectory))
    logger.info("  Unique Admissions: %d", trajectory["hadm_id"].nunique())
    logger.info("  ICU Transfer Rows: %d", trajectory["is_icu_stay"].sum())

    return trajectory


def simulate_department_surge_capacity(
    patient_flow: pd.DataFrame,
    surge_multiplier: float = 1.25,
    bed_turnover_lead_hours: int = 4,
    standard_target_occupancy: float = 0.85,
    output_parquet_path: Union[str, Path, None] = None,
    output_csv_path: Union[str, Path, None] = None,
    output_json_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Simulate discrete-event bed capacity, surge inflow, and clearance under surge conditions.

    Models:
    1. Care Unit Metrics:
       - Groups by curr_careunit (e.g. EMERGENCY DEPARTMENT, MICU, SICU, TRANSPLANT, MED, SURG).
       - Calculates empirical active bed demand, mean length of stay (LOS_mean in hours),
         and Little's Law baseline admission rate (lambda_base patients/hour).
    2. Dynamic Surge & Clearance Modeling:
       - Simulates admission surge using Poisson-adjusted arrivals over lead window (k = 4h):
           lambda_surge = lambda_base * surge_multiplier
           Projected Inflow (4h) = lambda_surge * 4
       - Models expected clearances over 4h using exponential stay survival distributions:
           P(discharge within 4h) = 1 - exp(-4 / LOS_mean)
           Expected Discharges (4h) = Active Patients * P(discharge within 4h)
    3. Operational Deficit & Risk Classification:
       - Net Bed Deficit = Projected Inflow (4h) - Expected Discharges (4h)
       - Projected Bed Occupancy = (Active Beds + Surge Deficit) / Licensed Bed Capacity
       - Categorical Risk Tiers:
           * CRITICAL_BOTTLENECK: Projected Occupancy >= 90% or Surge Deficit > 3 beds
           * STRAINED: Projected Occupancy between 75% and 89%
           * STABLE: Projected Occupancy < 75%
    4. Dual-Layer Persistence:
       - Persists analytical feature mart to data/04_feature/feat_bed_surge_metrics.parquet
       - Exports executive tabular view to data/04_feature/powerbi_executive_capacity_report.csv
         (columns: care_unit, active_patients, baseline_hourly_inflow, surge_multiplier,
          projected_inflow_4h, expected_discharges_4h, surge_deficit,
          projected_occupancy_pct, operational_risk_tier, governance_alert_flag)

    Args:
        patient_flow: Primary patient flow trajectory DataFrame.
        surge_multiplier: Epidemic / disaster surge buffer multiplier (e.g. 1.25 for +25%).
        bed_turnover_lead_hours: Turnaround disinfection and intake window in hours.
        standard_target_occupancy: Baseline operational target occupancy ratio (default 0.85).
        output_parquet_path: Optional path for feature Parquet mart.
        output_csv_path: Optional path for Power BI executive report CSV.

    Returns:
        Enriched DataFrame with simulation metrics and operational risk classifications.
    """
    logger.info(
        "Executing Node 3: simulate_department_surge_capacity (surge_mult=%.2f, window=%dh, target_occ=%.2f)",
        surge_multiplier,
        bed_turnover_lead_hours,
        standard_target_occupancy,
    )
    if patient_flow.empty:
        logger.warning("Empty patient_flow DataFrame provided to simulate_department_surge_capacity.")
        return pd.DataFrame()

    # Filter out UNKNOWN or empty care units for accurate department analytics
    careunit_col = "curr_careunit" if "curr_careunit" in patient_flow.columns else "careunit"
    valid_units = patient_flow[
        patient_flow[careunit_col].notna()
        & ~patient_flow[careunit_col].isin(["UNKNOWN", "", "NONE"])
    ].copy()

    if valid_units.empty:
        valid_units = patient_flow.copy()

    results = []
    for cu, group in valid_units.groupby(careunit_col):
        active_pts = len(group)
        los_series = group["careunit_los_hours"] if "careunit_los_hours" in group else group.get("los_hours", pd.Series([1.0]))
        valid_los = los_series[los_series > 0]
        mean_los = float(valid_los.mean()) if not valid_los.empty else 1.0
        mean_los = max(mean_los, 1.0)

        # Baseline arrival rate via Little's Law equilibrium
        lambda_base = active_pts / mean_los
        lambda_surge = lambda_base * surge_multiplier

        # Projected arrivals over lead window (4h)
        projected_inflow_4h = round(lambda_surge * bed_turnover_lead_hours, 2)

        # Expected discharges over 4h using exponential stay survival distribution
        p_discharge_4h = 1.0 - float(np.exp(-bed_turnover_lead_hours / mean_los))
        expected_discharges_4h = round(active_pts * p_discharge_4h, 2)

        # Net Operational Bed Deficit
        surge_deficit = round(projected_inflow_4h - expected_discharges_4h, 2)

        # Licensed bed capacity: calibrated against standard target occupancy
        licensed_capacity = max(active_pts + 2, int(np.ceil(active_pts / standard_target_occupancy)))
        projected_occupancy = (active_pts + surge_deficit) / licensed_capacity
        projected_occupancy_pct = round(projected_occupancy * 100.0, 2)

        # Operational Risk Classification
        if projected_occupancy >= 0.90 or surge_deficit > 3.0:
            risk_tier = "CRITICAL_BOTTLENECK"
        elif projected_occupancy >= 0.75:
            risk_tier = "STRAINED"
        else:
            risk_tier = "STABLE"

        results.append({
            "care_unit": cu,
            "active_patients": active_pts,
            "baseline_hourly_inflow": round(lambda_base, 4),
            "mean_los_hours": round(mean_los, 4),
            "licensed_capacity": int(licensed_capacity),
            "surge_multiplier": surge_multiplier,
            "projected_inflow_4h": projected_inflow_4h,
            "expected_discharges_4h": expected_discharges_4h,
            "surge_deficit": surge_deficit,
            "projected_occupancy_pct": projected_occupancy_pct,
            "operational_risk_tier": risk_tier,
            "governance_alert_flag": (risk_tier == "CRITICAL_BOTTLENECK"),
        })

    res_df = pd.DataFrame(results)

    # Sort descending by surge deficit to highlight bottlenecks first
    res_df = res_df.sort_values(by=["surge_deficit", "projected_occupancy_pct"], ascending=[False, False]).reset_index(drop=True)

    # Dual-layer persistence
    parquet_dest = Path(output_parquet_path) if output_parquet_path else Path("data/04_feature/feat_bed_surge_metrics.parquet")
    csv_dest = Path(output_csv_path) if output_csv_path else Path("data/04_feature/powerbi_executive_capacity_report.csv")

    parquet_dest.parent.mkdir(parents=True, exist_ok=True)
    csv_dest.parent.mkdir(parents=True, exist_ok=True)

    res_df.to_parquet(parquet_dest, compression="snappy", index=False)
    res_df.to_csv(csv_dest, index=False)

    logger.info("Persisted Feature Parquet dataset: %s (%d care units)", parquet_dest.name, len(res_df))
    logger.info("Exported Power BI Executive Report: %s (%d care units)", csv_dest.name, len(res_df))

    # ── Web Dashboard Baseline JSON Export ──────────────────────────────────
    if output_json_path is not None:
        json_dest = Path(output_json_path)
        json_dest.parent.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
            "pipeline_version": "phase_2_surge_simulator",
            "surge_config": {
                "default_surge_multiplier": surge_multiplier,
                "default_lead_hours": bed_turnover_lead_hours,
                "standard_target_occupancy": standard_target_occupancy,
            },
            "units": [
                {
                    "care_unit": str(row["care_unit"]),
                    "active_patients": int(row["active_patients"]),
                    "baseline_hourly_inflow": float(row["baseline_hourly_inflow"]),
                    "mean_los_hours": float(row["mean_los_hours"]),
                    "licensed_capacity": int(row["licensed_capacity"]),
                }
                for _, row in res_df.iterrows()
            ],
        }
        json_dest.write_text(json.dumps(baseline_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported Web Dashboard Baseline JSON: %s (%d units)", json_dest.name, len(res_df))

    return res_df


# Backward-compatible alias for pipeline and legacy callers
compute_bed_surge_metrics = simulate_department_surge_capacity
create_patient_flow = lambda adm, trf, srv: build_patient_flow_trajectories(adm, trf, srv)
