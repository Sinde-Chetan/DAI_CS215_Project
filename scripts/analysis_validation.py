from pathlib import Path

import math

import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "cleaned_student_data.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "validation_metrics.txt"

STRESS_MAPPING = {"low": 1, "moderate": 2, "medium": 2, "high": 3}
N_BOOT = 2000
SEED = 42


def load_data():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Works for 'object' (older pandas) AND the new 'str' dtype (pandas 3.x)
    if not pd.api.types.is_numeric_dtype(df["Stress_Level"]):
        cleaned = df["Stress_Level"].astype(str).str.strip().str.lower()
        mapped = cleaned.map(STRESS_MAPPING)
        unmapped = sorted(cleaned[mapped.isna()].unique())
        if unmapped:
            print(f"WARNING: unmapped Stress_Level values: {unmapped}")
        df["Stress_Level"] = mapped

    cols = [
        "GPA", "Study_Hours_Per_Day", "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day", "Social_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day", "Stress_Level",
    ]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    print("NaN count per column before dropna:")
    print(df[cols].isna().sum().to_string())
    df = df.dropna(subset=cols).reset_index(drop=True)
    hours = [c for c in cols if c.endswith("_Hours_Per_Day")]
    if ((df[hours].sum(axis=1) - 24).abs() < 1e-6).all():
        print("NOTE: the 5 hour columns sum to 24 -> one is dropped from the full model.")
    print(f"Rows kept: {len(df)} / {n_before}")

    if df.empty:
        raise ValueError("No rows left after cleaning - see NaN counts above.")
    return df


def mediation_effects(sleep, stress, gpa):
    """Returns a, b, direct (c'), indirect (a*b), total (c, from Y ~ X)."""
    a = sm.OLS(stress, sm.add_constant(sleep)).fit().params.iloc[1]
    m_b = sm.OLS(gpa, sm.add_constant(pd.concat([sleep, stress], axis=1))).fit()
    b = m_b.params[stress.name]
    direct = m_b.params[sleep.name]
    total = sm.OLS(gpa, sm.add_constant(sleep)).fit().params.iloc[1]
    return a, b, direct, a * b, total


def bootstrap_indirect(df, n_boot=N_BOOT, seed=SEED):
    vals = []
    for i in range(n_boot):
        s = df.sample(n=len(df), replace=True, random_state=seed + i)
        vals.append(
            mediation_effects(s["Sleep_Hours_Per_Day"], s["Stress_Level"], s["GPA"])[3]
        )
    vals = pd.Series(vals)
    return vals.quantile(0.025), vals.quantile(0.975)


def run_analysis_and_validation():
    df = load_data()
    y = df["GPA"]

    base_feats = ["Study_Hours_Per_Day", "Sleep_Hours_Per_Day"]
    # The five *_Hours_Per_Day columns sum to exactly 24 for every student, so
    # using all of them plus an intercept is perfectly collinear (singular
    # matrix). Social hours is dropped as the reference category.
    full_feats = [
        "Study_Hours_Per_Day", "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day", "Stress_Level",
    ]

    model_base = sm.OLS(y, sm.add_constant(df[base_feats])).fit()
    model_full = sm.OLS(y, sm.add_constant(df[full_feats])).fit()

    # Nested-model F-test (base is a subset of full)
    f_stat, f_p, f_df = model_full.compare_f_test(model_base)

    # Mediation: Sleep -> Stress -> GPA
    a, b, direct, indirect, total = mediation_effects(
        df["Sleep_Hours_Per_Day"], df["Stress_Level"], df["GPA"]
    )
    ci_lo, ci_hi = bootstrap_indirect(df)

    # Sobel test
    m_a = sm.OLS(df["Stress_Level"], sm.add_constant(df["Sleep_Hours_Per_Day"])).fit()
    m_b = sm.OLS(y, sm.add_constant(df[["Sleep_Hours_Per_Day", "Stress_Level"]])).fit()
    se_a = m_a.bse["Sleep_Hours_Per_Day"]
    se_b = m_b.bse["Stress_Level"]
    sobel_z = (a * b) / math.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    # Two-sided p from the standard normal CDF, via math.erfc
    sobel_p = math.erfc(abs(sobel_z) / math.sqrt(2))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Observations used: {len(df)}\n\n")
        f.write("=== MODEL COMPARISONS ===\n")
        f.write("Base Model (Study, Sleep) vs Full Multivariable Model\n\n")
        f.write(f"Base Model AIC: {model_base.aic:.2f}\n")
        f.write(f"Full Model AIC: {model_full.aic:.2f} (lower is better)\n\n")
        f.write(f"Base Model BIC: {model_base.bic:.2f}\n")
        f.write(f"Full Model BIC: {model_full.bic:.2f} (lower is better)\n\n")
        f.write(f"Base Model Adj R-squared: {model_base.rsquared_adj:.4f}\n")
        f.write(f"Full Model Adj R-squared: {model_full.rsquared_adj:.4f} (higher is better)\n\n")
        f.write(f"Nested F-test (Full vs Base): F = {f_stat:.2f}, "
                f"p = {f_p:.3g}, df diff = {int(f_df)}\n\n")

        f.write("=== MEDIATION ANALYSIS: Sleep -> Stress -> GPA ===\n\n")
        f.write(f"Path A (Sleep -> Stress):                  {a:.4f}\n")
        f.write(f"Path B (Stress -> GPA, controlling Sleep): {b:.4f}\n")
        f.write(f"Direct Effect  (c', Sleep -> GPA | Stress): {direct:.4f}\n")
        f.write(f"Indirect Effect (a*b):                      {indirect:.4f}\n")
        f.write(f"Total Effect   (c, Sleep -> GPA alone):     {total:.4f}\n")
        f.write(f"Check: direct + indirect = {direct + indirect:.4f}\n\n")
        f.write(f"Bootstrap 95% CI for indirect effect ({N_BOOT} resamples): "
                f"[{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"Sobel test: z = {sobel_z:.3f}, p = {sobel_p:.3g}\n")
        f.write("Indirect effect is significant if the bootstrap CI excludes 0.\n\n")

        f.write("Interpretation:\n")
        f.write("- Lower AIC/BIC, higher Adj R-squared and a significant F-test "
                "mean the extra lifestyle factors improve the model.\n")
        f.write("- The indirect effect is how much of Sleep's impact on GPA "
                "operates through Stress.\n")
        f.write("- Full-model hour coefficients are relative to the dropped "
                "Social_Hours_Per_Day (the 5 hour columns sum to 24).\n")
        f.write("- Stress_Level is coded Low=1, Moderate=2, High=3 (ordinal "
                "treated as numeric).\n")

    print(f"Saved to {OUTPUT_FILE}")
    print(OUTPUT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    run_analysis_and_validation()