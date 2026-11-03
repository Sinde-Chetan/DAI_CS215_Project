"""Run the whole pipeline in order:  python run_all.py
Regenerates data/cleaned_student_data.csv and everything in results/."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
STEPS = ["data_preprocessing.py", "eda.py", "correlation_map.py",
         "regression.py", "analysis_validation.py"]

for step in STEPS:
    print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}")
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True, cwd=SCRIPTS)
print("\nAll steps finished - see results/")
