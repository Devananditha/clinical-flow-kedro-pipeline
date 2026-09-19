# Clinical Flow Kedro Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Kedro](https://img.shields.io/badge/Kedro-~0.19.6-orange.svg)](https://kedro.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion%20Lakehouse-success.svg)](#medallion-data-architecture)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

An enterprise-grade Healthcare DataOps & Analytics Engineering pipeline built on **Kedro** and **MIMIC-IV Demo** clinical data. It ingests, standardizes, governs, and synthesizes inpatient hospital trajectories to calculate patient length-of-stay (LOS), bed occupancy, turnaround disinfection buffers, and epidemic surge capacity modeling.

---

## 1. Architecture Overview

This project implements a **Medallion Architecture (Bronze $\to$ Silver $\to$ Gold $\to$ Feature/Platinum)** for clinical flow governance:

```
clinical-flow-kedro-pipeline/
├── .gitignore                      # Healthcare compliance rules & data exclusions
├── README.md                       # Project documentation & DataOps guides
├── requirements.txt                # Pinned dependencies (Kedro, PySpark, Pandas, etc.)
├── run_pipeline.py                 # Pipeline execution runner
├── setup_data.py                   # Automated cross-platform data provisioning
├── conf/
│   └── base/
│       ├── catalog.yml             # Kedro dataset catalog for Medallion layers
│       └── parameters.yml          # Governance parameters (surge_multiplier, turnover buffer)
├── data/
│   ├── 01_raw/                     # Bronze: Raw CSV/GZ tables (admissions, transfers, services)
│   ├── 02_intermediate/            # Silver: Cleaned, schema-validated Parquet tables
│   ├── 03_primary/                 # Gold: Unified patient trajectory and care-unit stays
│   └── 04_feature/                 # Feature: Bed surge metrics & occupancy capacity tables
├── src/
│   └── clinical_flow/
│       ├── __init__.py
│       └── pipelines/
│           ├── __init__.py
│           └── data_engineering/
│               ├── __init__.py
│               ├── nodes.py        # Transformation logic & clinical quality checks
│               └── pipeline.py     # Kedro DAG definition
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Synthetic clinical fixtures
    └── test_clinical_governance.py # Clinical boundary & surge calculation tests
```

### Medallion Layer Flow

| Layer | Path | Formats | Description |
| :--- | :--- | :--- | :--- |
| **01_raw (Bronze)** | `data/01_raw/` | `.csv` | Raw MIMIC-IV demo tables (`admissions`, `transfers`, `services`). |
| **02_intermediate (Silver)** | `data/02_intermediate/` | `.parquet` | Type-cast, date-parsed, anomaly-filtered clinical tables. Negative LOS records purged. |
| **03_primary (Gold)** | `data/03_primary/` | `.parquet` | Consolidated longitudinal patient journey tracking care unit visits, ICU stays, and admitting service lines. |
| **04_feature (Platinum)** | `data/04_feature/` | `.parquet` | Capacity planning features incorporating `surge_multiplier` (1.25) and `bed_turnover_lead_hours` (4h). |

---

## 2. Quickstart & Setup

### Prerequisites
- Python 3.10 or 3.11
- Git

### 1. Environment Installation
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Automated Dataset Ingestion & Provisioning
The included cross-platform routine automatically detects `admissions.csv.gz`, `transfers.csv.gz`, and `services.csv.gz` inside your `~/Downloads/` directory, extracts them into `data/01_raw/`, and validates file integrity:

```bash
python setup_data.py
```

### 3. Run Pipeline
Execute the complete Medallion pipeline from Bronze to Feature:
```bash
python run_pipeline.py
```

### 4. Run Test Suite
Validate data governance rules, timestamp monotonicity, and surge math:
```bash
pytest tests/ -v
```

---

## 3. Clinical Data Governance Rules

1. **Patient Identifier Integrity**: Records missing `subject_id` or `hadm_id` are rejected from Silver and downstream analytics.
2. **Chronological Monotonicity**: Admissions where `dischtime < admittime` (negative LOS) are flagged and filtered as anomalies.
3. **Bed Turnover Buffer**: Each inpatient stay reserves a configurable lead buffer (default: `4 hours`) to account for terminal disinfection and bed turnover.
4. **Surge Capacity Factor**: Surge capacity calculations apply a standard multiplier (default: `1.25`, representing a +25% bed contingency buffer) over effective occupied bed hours.

---

## 4. Git Deployment

To initialize and push this repository to GitHub:

```bash
# 1. Initialize git and switch to main branch
git init
git checkout -b main

# 2. Add remote origin
git remote add origin https://github.com/Devananditha/clinical-flow-kedro-pipeline.git

# 3. Stage source, configuration, and documentation (data files excluded by .gitignore)
git add .

# 4. Commit and push
git commit -m "feat: bootstrap clinical-flow-kedro-pipeline with medallion data architecture"
git push -u origin main
```
