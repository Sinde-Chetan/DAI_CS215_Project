import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "cleaned_student_data.csv"
OUTPUT_FILE = BASE_DIR / "notebooks" / "correlation_heatmap.png"

# Load cleaned data
df = pd.read_csv(INPUT_FILE)

# Summary Statistics
print("Summary Statistics:")
print(df.describe())

# Select numerical columns
columns = [
    "GPA",
    "Sleep_Hours_Per_Day",
    "Study_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]

# Correlation Matrix
correlation = df[columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
)

plt.title("Correlation Matrix: Lifestyle Factors & GPA")
plt.tight_layout()

# Save heatmap
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nHeatmap saved to: {OUTPUT_FILE}")