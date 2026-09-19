import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src/ is on sys.path across all environments
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_raw_admissions() -> pd.DataFrame:
    """Fixture providing mock raw admissions data adhering to MIMIC-IV schema."""
    return pd.DataFrame(
        {
            "subject_id": [1001, 1002, 1003, None],
            "hadm_id": [2001, 2002, 2003, 2004],
            "admittime": [
                "2100-01-01 08:00:00",
                "2100-01-02 10:00:00",
                "2100-01-05 12:00:00",  # Anomalous: dischtime before admittime
                "2100-01-03 09:00:00",
            ],
            "dischtime": [
                "2100-01-04 18:00:00",
                "2100-01-03 14:00:00",
                "2100-01-04 12:00:00",  # Prior to admittime
                "2100-01-06 11:00:00",
            ],
            "deathtime": [None, None, None, None],
            "admission_type": ["EMERGENCY", "ELECTIVE", "URGENT", "EW EMER."],
            "admission_location": ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER", "EMERGENCY ROOM"],
            "discharge_location": ["HOME", "HOME HEALTH CARE", "REHAB", "HOME"],
            "insurance": ["Medicare", "Other", "Medicaid", "Other"],
            "language": ["ENGLISH", "ENGLISH", "SPANISH", "ENGLISH"],
            "marital_status": ["MARRIED", "SINGLE", "DIVORCED", "SINGLE"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "WHITE"],
            "hospital_expire_flag": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def sample_raw_transfers() -> pd.DataFrame:
    """Fixture providing mock raw transfers data adhering to MIMIC-IV schema."""
    return pd.DataFrame(
        {
            "subject_id": [1001, 1001, 1002, 1005],
            "hadm_id": [2001, 2001, 2002, None],  # One with None hadm_id
            "transfer_id": [3001, 3002, 3003, 3004],
            "eventtype": ["admit", "transfer", "admit", "discharge"],
            "careunit": ["Emergency Department", "Medical Intensive Care Unit (MICU)", "Medicine", "Floor"],
            "intime": [
                "2100-01-01 08:00:00",
                "2100-01-01 14:00:00",
                "2100-01-02 10:00:00",
                "2100-01-03 09:00:00",
            ],
            "outtime": [
                "2100-01-01 14:00:00",
                "2100-01-04 18:00:00",
                "2100-01-03 14:00:00",
                "2100-01-03 17:00:00",
            ],
        }
    )


@pytest.fixture
def sample_raw_services() -> pd.DataFrame:
    """Fixture providing mock raw services data adhering to MIMIC-IV schema."""
    return pd.DataFrame(
        {
            "subject_id": [1001, 1001, 1002],
            "hadm_id": [2001, 2001, 2002],
            "transfertime": [
                "2100-01-01 08:00:00",
                "2100-01-02 12:00:00",
                "2100-01-02 10:00:00",
            ],
            "curr_service": ["MED", "CMED", "SURG"],
            "prev_service": [None, "MED", None],
        }
    )
