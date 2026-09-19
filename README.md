# Clinical Flow Kedro Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Kedro](https://img.shields.io/badge/Kedro-~0.19.6-orange.svg)](https://kedro.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion%20Lakehouse-success.svg)](#medallion-data-architecture)
[![HIPAA](https://img.shields.io/badge/HIPAA-Safe%20Harbor%20SHA--256-brightgreen.svg)](#hipaa-safe-harbor-de-identification)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

An enterprise-grade Healthcare DataOps & Analytics Engineering pipeline built with **Kedro**, **PySpark / Vectorized Pandas**, and **MIMIC-IV** clinical data. It ingests, standardizes, de-identifies, and models inpatient hospital trajectories to calculate patient length-of-stay (LOS), care unit transitions, and discrete-event bed surge capacity for hospital executive dashboards.

---

## 1. Medallion Data Architecture

```
clinical-flow-kedro-pipeline/
├── .gitignore                          # HIPAA compliance rules & data exclusions
├── README.md                           # Enterprise engineering documentation
├── requirements.txt                    # Pinned core dependencies
├── pyproject.toml                      # Modern packaging & pytest configuration
├── run_pipeline.py                     # Orchestration runner with executive reporting
├── setup_data.py                       # Automated cross-platform data provisioning
├── conf/
│   └── base/
│       ├── catalog.yml                 # Kedro dataset catalog for Medallion layers
│       └── parameters.yml              # Governance & surge simulation parameters
├── data/
│   ├── 01_raw/                         # Bronze: Raw CSV tables (admissions, transfers, services)
│   ├── 02_intermediate/                # Silver: Salted SHA-256 de-identified Parquet tables
│   ├── 03_primary/                     # Gold: Windowed longitudinal patient trajectories
│   └── 04_feature/                     # Platinum/Feature: Surge mart & Power BI executive report
├── src/
│   └── clinical_flow/
│       ├── __init__.py
│       └── pipelines/
│           ├── __init__.py
│           └── data_engineering/
│               ├── __init__.py
│               ├── nodes.py            # Transformation nodes & simulation math
│               └── pipeline.py         # Kedro DAG definition with resilient shims
└── tests/
    ├── __init__.py
    ├── conftest.py                     # Synthetic clinical fixtures
    └── test_clinical_governance.py     # 10 clinical boundary & surge simulation tests
```

### Medallion Layer Flow

| Layer | File Path | Format | Description |
| :--- | :--- | :--- | :--- |
| **01_raw (Bronze)** | `data/01_raw/` | `.csv` | Raw MIMIC-IV clinical tables (`admissions`, `transfers`, `services`). |
| **02_intermediate (Silver)** | `data/02_intermediate/` | `.parquet` (Snappy) | Salted SHA-256 de-identified, ISO 8601 UTC timestamps, chronological anomalies filtered. |
| **03_primary (Gold)** | `data/03_primary/` | `.parquet` (Snappy) | Unified longitudinal patient trajectories with window-partitioned sequence IDs, careunit LOS, and ICU indicators. |
| **04_feature (Platinum)** | `data/04_feature/` | `.parquet` (Snappy) | Discrete-event department surge metrics, Little's Law arrivals, clearances, and net deficit. |
| **Power BI Mart** | `data/04_feature/` | `.csv` | Executive capacity planning tabular export formatted for direct Power BI ingestion. |

---

## 2. Core Engineering Capabilities

### Resilient Multi-Engine Factory (`get_execution_engine`)
- Dynamically probes for a configured PySpark environment (`local[*]`, 2GB driver memory).
- If Java / JVM is not detected, automatically and gracefully falls back to a high-performance vectorized Pandas/PyArrow engine without pipeline disruption.

### HIPAA Safe Harbor De-Identification (`hash_identifier`)
- Replaces protected health information (PHI) identifiers (`subject_id`, `hadm_id`) with 64-character hexadecimal SHA-256 one-way cryptographic hashes.
- Salts tokens with `hipaa_salt` from `conf/base/parameters.yml` to prevent dictionary and rainbow-table attacks while strictly preserving cross-table referential integrity.

### Windowed Patient Flow Trajectories (`build_patient_flow_trajectories`)
- Partitions transfers over `hadm_id` ordered by `intime`.
- Calculates:
  - `transfer_sequence_id`: Sequential movement integer (1, 2, 3...).
  - `careunit_los_hours`: Stay duration in elapsed hours for each ward visit.
  - `prev_careunit` & `transition_lag_hours`: Lead time elapsed between ward departure and admission.
  - `is_icu_stay` & `had_icu_stay`: Identification of intensive care stays (`MICU`, `SICU`, `CCU`, `TSICU`, `CVICU`).
  - `emergency_admission`: Classification based on admission type categories.

### Discrete-Event Bed Capacity & Surge Simulator (`simulate_department_surge_capacity`)
1. **Little's Law Baseline Equilibrium**:
   $$\lambda_{\text{base}} = \frac{N}{\text{LOS}_{\text{mean}}}$$
2. **Surge Arrival Rate (4-Hour Lead Window)**:
   $$\lambda_{\text{surge}} = \lambda_{\text{base}} \times \text{surge\_multiplier}$$
   $$\text{Projected Inflow}_{4\text{h}} = \lambda_{\text{surge}} \times 4$$
3. **Exponential Stay Survival Clearance**:
   $$P(\text{discharge} \le 4\text{h}) = 1 - e^{-\frac{4}{\text{LOS}_{\text{mean}}}}$$
   $$\text{Expected Discharges}_{4\text{h}} = N \times P(\text{discharge} \le 4\text{h})$$
4. **Operational Bed Deficit & Occupancy**:
   $$\text{Net Deficit} = \text{Projected Inflow}_{4\text{h}} - \text{Expected Discharges}_{4\text{h}}$$
   $$\text{Projected Occupancy \%} = \frac{N + \text{Net Deficit}}{\text{Licensed Capacity}} \times 100$$
5. **Operational Risk Tier Matrix**:
   - `CRITICAL_BOTTLENECK`: Projected Occupancy $\ge 90\%$ or Net Deficit $> 3.0$ beds (`governance_alert_flag = True`).
   - `STRAINED`: Projected Occupancy between $75\%$ and $89\%$.
   - `STABLE`: Projected Occupancy $< 75\%$.

---

## 3. Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Devananditha/clinical-flow-kedro-pipeline.git
cd clinical-flow-kedro-pipeline

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Provisioning
```bash
python setup_data.py
```

### 3. Run Pipeline End-to-End
```bash
python run_pipeline.py
```

**Console Output:**
```
=========================================================================================================
                DISCRETE-EVENT BED CAPACITY & SURGE SIMULATION SUMMARY (LEAD WINDOW = 4H)                
=========================================================================================================
Care Unit                                  | Inflow (4h) | Discharges (4h) | Net Deficit | Risk Tier           
---------------------------------------------------------------------------------------------------------
EMERGENCY DEPARTMENT                       |      106.27 |           67.92 |      +38.35 | CRITICAL_BOTTLENECK 
EMERGENCY DEPARTMENT OBSERVATION           |       53.46 |           20.98 |      +32.48 | CRITICAL_BOTTLENECK 
PACU                                       |       20.99 |           12.23 |       +8.76 | CRITICAL_BOTTLENECK 
DISCHARGE LOUNGE                           |       19.50 |           12.66 |       +6.84 | CRITICAL_BOTTLENECK 
MEDICINE                                   |        4.87 |            3.80 |       +1.07 | STRAINED            
CARDIAC VASCULAR INTENSIVE CARE UNIT (CV   |        3.99 |            3.04 |       +0.95 | STRAINED            
...
=========================================================================================================
```

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 4. Power BI Analytical Mart Schema

The exported executive report `data/04_feature/powerbi_executive_capacity_report.csv` contains:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `care_unit` | String | Department or care unit name. |
| `active_patients` | Integer | Current patient census count. |
| `baseline_hourly_inflow` | Float | Little's Law baseline admission rate ($\lambda_{\text{base}}$). |
| `surge_multiplier` | Float | Active surge contingency factor (e.g. `1.25`). |
| `projected_inflow_4h` | Float | Expected incoming admissions over 4-hour lead window. |
| `expected_discharges_4h` | Float | Model-projected discharges via exponential clearance. |
| `surge_deficit` | Float | Net bed deficit (inflow minus clearances). |
| `projected_occupancy_pct` | Float | Operational bed occupancy percentage. |
| `operational_risk_tier` | String | Categorical risk: `CRITICAL_BOTTLENECK`, `STRAINED`, `STABLE`. |
| `governance_alert_flag` | Boolean | Executive alert flag (`True` for `CRITICAL_BOTTLENECK`). |
