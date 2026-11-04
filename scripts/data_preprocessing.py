""" Data cleaning.
raw_student_data.csv  ->  cleaned_student_data.csv
"""
from pathlib import Path

import pandas as pd

from utils import HOUR_COLS, RAW_FILE, CLEAN_FILE, RESULTS_DIR, STRESS_MAP, encode_stress

REQUIRED = ["Student_ID", *HOUR_COLS, "GPA", "Stress_Level"]
NUMERIC = [*HOUR_COLS, "GPA"]


def clean_student_data(input_file=RAW_FILE, output_file=CLEAN_FILE):
    input_file, output_file = Path(input_file), Path(output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    log = []
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}; found {df.columns.tolist()}")
    log.append(f"Rows read: {len(df)}")

    n = len(df)
    df = df.dropna(how="all")
    log.append(f"Completely empty rows removed: {n - len(df)}")
    n = len(df)
    df = df.drop_duplicates()
    log.append(f"Exact duplicate rows removed: {n - len(df)}")
    n = len(df)
    df = df.drop_duplicates(subset="Student_ID")
    log.append(f"Duplicate Student_IDs removed: {n - len(df)}")

    # numeric columns: coerce, then median-impute (only if something is missing)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].median())
        log.append(f"{col}: {n_missing} missing -> median imputed")

    # categorical stress level: blank -> NaN -> mode, then standardise labels
    df["Stress_Level"] = df["Stress_Level"].replace(r"^\s*$", pd.NA, regex=True)
    n_missing = int(df["Stress_Level"].isna().sum())
    if n_missing:
        df["Stress_Level"] = df["Stress_Level"].fillna(df["Stress_Level"].mode().iloc[0])
    log.append(f"Stress_Level: {n_missing} missing -> mode imputed")
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.capitalize()
    df["Stress_Level"] = df["Stress_Level"].replace({"Medium": "Moderate"})
    df["Stress_Level_Num"] = encode_stress(df["Stress_Level"])

    # validation (reported, not silently "fixed")
    problems = []
    if not df["GPA"].between(0, 4).all():
        problems.append("GPA outside 0-4")
    if (df[HOUR_COLS] < 0).any().any():
        problems.append("negative hours")
    if (df[HOUR_COLS].sum(axis=1) > 24 + 1e-6).any():
        problems.append("daily hours > 24")
    log.append("Range checks: " + ("; ".join(problems) if problems else "all passed"))
    sums_to_24 = bool(((df[HOUR_COLS].sum(axis=1) - 24).abs() < 1e-6).all())
    log.append(f"All five hour columns sum to exactly 24 h: {sums_to_24} "
               "(=> perfect collinearity; Social hours is dropped from regressions)")
    log.append("Screen-time column present: " + str(any("screen" in c.lower() for c in df.columns)))
    log.append(f"Final rows: {len(df)}, columns: {len(df.columns)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preprocessing_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    print("\n".join(log))
    print(f"Saved cleaned data to {output_file}")
    return df


if __name__ == "__main__":
    clean_student_data()
