from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw_student_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_student_data.csv"


def clean_student_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Check that all expected columns exist
    required_columns = [
        "Student_ID",
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
        "Stress_Level",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in the input CSV: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Remove exact duplicate records
    df.drop_duplicates(inplace=True)

    # Convert numerical columns to numeric values
    numeric_columns = [
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Replace missing values with the column median,
        # provided that at least one valid value exists.
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)

    # Handle missing stress levels
    df["Stress_Level"] = df["Stress_Level"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    stress_mode = df["Stress_Level"].mode()

    if not stress_mode.empty:
        df["Stress_Level"] = df["Stress_Level"].fillna(
            stress_mode.iloc[0]
        )

    # Create the output directory if it does not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the cleaned dataset
    df.to_csv(output_file, index=False)

    # Display the results
    print("Data preprocessing completed successfully!")
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())

    if df.isnull().values.any():
        print(
            "\nWarning: Some missing values remain. "
            "Check columns containing entirely invalid or missing data."
        )
    else:
        print("\nNo missing values remain.")

    return df


if __name__ == "__main__":
    clean_student_data()
