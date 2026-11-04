
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import RESULTS_DIR, load_clean, savefig, setup_style


LABEL = {"Study": "Study hours/day", "Sleep": "Sleep hours/day", "Stress": "Stress level (1-3)"}


def scatter_with_fit(plt, df, col, model, color, fname):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    jitter = np.random.default_rng(0).uniform(-.08, .08, len(df)) if col == "Stress_Level_Num" else 0
    ax.scatter(df[col] + jitter, df["GPA"], s=14, alpha=.35, color=color, edgecolor="none",
               label="Students")
    xs = np.array([df[col].min(), df[col].max()])
    ax.plot(xs, model.params["Intercept"] + model.params[col] * xs, color="#e74c3c", lw=2.5,
            label="OLS line")
    ax.text(.04, .95, f"slope = {model.params[col]:+.4f}\nR² = {model.rsquared:.3f}\n"
            f"p = {model.pvalues[col]:.3g}", transform=ax.transAxes, va="top", fontsize=12,
            family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=.9))
    ax.set_xlabel(LABEL[col]); ax.set_ylabel("GPA")
    ax.set_title(f"{LABEL[col]} vs GPA (stand-alone model)", fontweight="bold")
    ax.legend(loc="lower right")
    savefig(plt, fname)


def main():
    plt = setup_style()
    df = load_clean()
    df = df.rename(columns={"Study_Hours_Per_Day": "Study", "Sleep_Hours_Per_Day": "Sleep",
                            "Physical_Activity_Hours_Per_Day": "Phys",
                            "Extracurricular_Hours_Per_Day": "Extra",
                            "Social_Hours_Per_Day": "Social"})
    df["Stress"] = df["Stress_Level_Num"]

    models = {
        "M1 Study only": smf.ols("GPA ~ Study", df).fit(),
        "M2 Sleep only": smf.ols("GPA ~ Sleep", df).fit(),
        "M3 Stress only": smf.ols("GPA ~ Stress", df).fit(),
        "M4 Study + Sleep": smf.ols("GPA ~ Study + Sleep", df).fit(),
        "M5 Study x Sleep": smf.ols("GPA ~ Study * Sleep", df).fit(),
        "M6 Full (Social = reference)": smf.ols(
            "GPA ~ Study + Sleep + Phys + Extra + Stress", df).fit(),
    }
    full = models["M6 Full (Social = reference)"]
    assert np.linalg.matrix_rank(full.model.exog) == full.model.exog.shape[1], "singular design!"

    
    with open(RESULTS_DIR / "regression_results_summary.txt", "w", encoding="utf-8") as f:
        for name, m in models.items():
            f.write(f"=== {name} ===\n{m.summary().as_text()}\n\n")
        f.write("Note: the five *_Hours_Per_Day columns sum to 24, so Social hours is the\n"
                "dropped reference; hour coefficients = effect of moving one hour FROM social\n"
                "time TO that activity. Stress is the ordinal code Low=1, Moderate=2, High=3.\n")
    rows = []
    for name, m in models.items():
        ci = m.conf_int()
        for term in m.params.index:
            rows.append({"model": name, "term": term, "coef": m.params[term],
                         "se": m.bse[term], "p": m.pvalues[term],
                         "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                         "R2": m.rsquared, "adj_R2": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic})
    pd.DataFrame(rows).round(5).to_csv(RESULTS_DIR / "regression_table.csv", index=False)
    print(pd.DataFrame({n: {"R2": m.rsquared, "adjR2": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic}
                        for n, m in models.items()}).T.round(4).to_string())
    print("\nFull model coefficients:\n", full.summary().tables[1])


    scatter_with_fit(plt, df, "Study", models["M1 Study only"], "#2980b9", "study_vs_gpa_model.png")
    scatter_with_fit(plt, df, "Sleep", models["M2 Sleep only"], "#d35400", "sleep_vs_gpa_model.png")
    scatter_with_fit(plt, df, "Stress", models["M3 Stress only"], "#196f3d", "stress_vs_gpa_model.png")

    
    labels = {"Study": "Study", "Sleep": "Sleep", "Phys": "Physical activity",
              "Extra": "Extracurricular", "Stress": "Stress (per level)"}
    par = full.params.drop("Intercept"); ci = full.conf_int().drop("Intercept")
    par.index = [labels[i] for i in par.index]; ci.index = par.index
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in par]
    ax.barh(par.index, par.values, xerr=[par - ci[0], ci[1] - par], color=colors,
            edgecolor="black", capsize=5, alpha=.9)
    ax.axvline(0, color="black", lw=1.2); ax.invert_yaxis()
    ax.set_xlabel("Change in GPA per +1 hour (per +1 level for stress), Social hours = reference")
    ax.set_title("Full model: coefficients with 95% CI", fontweight="bold")
    savefig(plt, "regression_coefficients.png")


    cols = ["Study", "Sleep", "Phys", "Extra", "Stress"]
    z = (df[cols + ["GPA"]] - df[cols + ["GPA"]].mean()) / df[cols + ["GPA"]].std()
    zm = sm.OLS(z["GPA"], sm.add_constant(z[cols])).fit()
    beta = zm.params.drop("const"); beta.index = [labels[i] for i in beta.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(beta.index, beta.values, color="#2e86ab", edgecolor="black")
    ax.axvline(0, color="black", lw=1.2); ax.invert_yaxis()
    ax.set_xlabel("Standardised coefficient (SD of GPA per SD of predictor)")
    ax.set_title("Relative importance of predictors", fontweight="bold")
    savefig(plt, "standardized_coefficients.png")

    print("\nStandardised coefficients:\n", beta.round(3).to_string())


if __name__ == "__main__":
    main()
