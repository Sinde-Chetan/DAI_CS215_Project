"""Shared helpers: paths, data loading and plotting style.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "data plots"
RAW_FILE = DATA_DIR / "raw_student_data.csv"
CLEAN_FILE = DATA_DIR / "cleaned_student_data.csv"

STRESS_MAP = {"low": 1, "moderate": 2, "medium": 2, "high": 3}


HOUR_COLS = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
]
REFERENCE_COL = "Social_Hours_Per_Day"
PREDICTORS_FULL = [c for c in HOUR_COLS if c != REFERENCE_COL] + ["Stress_Level_Num"]

SHORT = {
    "Study_Hours_Per_Day": "Study",
    "Sleep_Hours_Per_Day": "Sleep",
    "Physical_Activity_Hours_Per_Day": "Physical activity",
    "Social_Hours_Per_Day": "Social",
    "Extracurricular_Hours_Per_Day": "Extracurricular",
    "Stress_Level_Num": "Stress (1-3)",
    "GPA": "GPA",
}


def encode_stress(series: pd.Series) -> pd.Series:
    """Map Low/Moderate/High (any case/whitespace) to 1/2/3."""
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(STRESS_MAP)
    bad = sorted(cleaned[mapped.isna()].unique())
    if bad:
        raise ValueError(f"Unmapped Stress_Level values: {bad}")
    return mapped.astype(int)


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset (run data_preprocessing.py first)."""
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"{CLEAN_FILE} not found - run scripts/data_preprocessing.py first."
        )
    df = pd.read_csv(CLEAN_FILE)
    df.columns = df.columns.str.strip()
    if "Stress_Level_Num" not in df.columns:
        df["Stress_Level_Num"] = encode_stress(df["Stress_Level"])
    return df


def setup_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True,
        "grid.linestyle": "--", "grid.alpha": 0.4, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def savefig(plt, name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path.relative_to(BASE_DIR)}")
