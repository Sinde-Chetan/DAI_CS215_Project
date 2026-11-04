# Sleep, Study Habits & Academic Performance

**CS215 (Data Analysis and Interpretation) · Open Project · Team 26 · IIT Bombay**

How do daily study hours, sleep, physical activity, social time, extracurricular time and stress relate to student GPA, and which habit matters most? We analyse 2,000 students from the Kaggle *Student Lifestyle* dataset using regression, interaction and mediation analysis, with full model diagnostics.

## Key findings

| Hypothesis | Result |
|---|---|
| **H1** Study and sleep both positively relate to GPA | **Half supported.** Study: +0.154 GPA per hour (p < 10⁻¹⁶, R² = 0.54). Sleep: no relationship (r = 0.00). |
| **H2** More social time / less physical activity lowers GPA (controlling for study) | **Not supported.** Raw associations vanish once study hours are controlled. |
| **H3** Study × Sleep interaction | **Not supported at 5%** (coef = +0.004, p = 0.072; adj. R² rises by only 0.0005). |
| **H4** Multi-factor model beats study hours alone | **Not supported.** Nested F-test p = 0.158; cross-validated R² is 0.537 for both. |


**Bottom line:** study hours are the story. Everything else is either unrelated to GPA or a by-product of study time (e.g. stress is largely a function of study load).

> **Note on scope.** Our original proposal centred on *screen time*, but the dataset has no screen-time variable. Those hypotheses (as written in `statement.pdf`) could not be tested and were replaced with ones on the variables that exist. See the final report for details.

## Data

- **Source:** Kaggle *Student Lifestyle Dataset*
- **Size:** 2,000 students × 8 columns: student ID, study hours, extracurricular hours, sleep hours, social hours, physical-activity hours, GPA (2.24–4.00) and stress level (Low / Moderate / High).
- **Cleaning:** the data had no missing values, duplicates or out-of-range entries. The pipeline standardises stress labels, adds a numeric stress code (Low = 1, Moderate = 2, High = 3) and validates ranges.
- **Structural quirk:** the five hours variables sum to exactly 24 for every student, so they are perfectly collinear with the intercept. All regressions drop one (social hours is the reference), so a coefficient means "one more hour of this activity *instead of* social time".

## Repository structure

```
.
├── data/          # raw and cleaned student lifestyle CSVs
├── notebooks/     # executed analysis notebook
├── scripts/       # preprocessing, EDA, modelling and plotting scripts
├── statement.pdf  # original project statement (includes screen-time hypotheses, later revised)
└── .gitignore
```

Scripts referenced in the report:

| Script | Purpose |
|---|---|
| `data_preprocessing.py` | Cleans and validates the raw CSV, exports the cleaned dataset |
| `correlation_map.py` | Pearson/Spearman correlation heatmaps |
| `run_all.py` | Regenerates every result and figure in the report |

## Getting started

```bash
git clone https://github.com/Sinde-Chetan/DAI_CS215_Project.git
cd DAI_CS215_Project

python -m venv .venv
source .venv/bin/activate       
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn

python scripts/run_all.py        
```

Or open the executed notebook in `notebooks/` with Jupyter.

## Methods

- **Models:** ordinary least squares: study only (M1), study + sleep (M4), study × sleep (M5), full model without social hours (M6).
- **Model comparison:** adjusted R², AIC, BIC, repeated 5-fold cross-validated R², nested F-test.
- **Predictor importance:** standardised coefficients and LMG variance decomposition.


## Limitations

- **Screen time was not measured**, so the original research question was not answered with this dataset.
- **Sleep quality and timing are not recorded**, and the sleep range is narrow (5–10 h). The absence of an association does not mean sleep is unimportant.
- **Association, not causation.** Stress in particular is plausibly a consequence of study load rather than a cause of GPA.

## Team

| Member | Roll number | Responsibility |
|---|---|---|
| Chetan | 25b0942 | Data cleaning and exploratory analysis |
| Karthik | 25b0976 | Statistical analysis and multivariable regression |
| Mukesh | 25b1009 | Statistical analysis and multivariable regression |
| Rishanth | 25b1044 | Predictor importance, interaction terms and model validation |

**Mentor:** Namish Bansal

## References

- Hale, L., & Guan, S. (2015). Screen time and sleep among school-aged children and adolescents. *Sleep Medicine Reviews.* https://doi.org/10.1016/j.smrv.2014.07.007
- Kaggle: Student Lifestyle Dataset
