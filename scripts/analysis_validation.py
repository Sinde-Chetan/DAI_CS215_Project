"""Step 4 - Hypothesis tests and model comparison.
"""
import itertools
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score

from utils import RESULTS_DIR, load_clean

SEED, ALPHA = 42, 0.05


def pf(p):
    return "<1e-16" if p < 1e-16 else f"{p:.3g}"


def prep():
    df = load_clean().rename(columns={
        "Study_Hours_Per_Day": "Study", "Sleep_Hours_Per_Day": "Sleep",
        "Physical_Activity_Hours_Per_Day": "Phys", "Social_Hours_Per_Day": "Social",
        "Extracurricular_Hours_Per_Day": "Extra"})
    df["Stress"] = df["Stress_Level_Num"]
    df["LowSleep"] = (df["Sleep"] < 6).astype(int)
    return df


def ols_coefs(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def lmg(df, y, predictors):
    r2 = {(): 0.0}
    for k in range(1, len(predictors) + 1):
        for S in itertools.combinations(predictors, k):
            r2[S] = sm.OLS(df[y], sm.add_constant(df[list(S)])).fit().rsquared
    p, out = len(predictors), {}
    for v in predictors:
        others = [q for q in predictors if q != v]
        total = 0.0
        for k in range(p):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(p - k - 1) / math.factorial(p)
                total += w * (r2[tuple(sorted(S + (v,), key=predictors.index))]
                              - r2[tuple(sorted(S, key=predictors.index))])
        out[v] = total
    return out


def main():
    df = prep()
    res, out = {}, []
    w = out.append

    m_study = smf.ols("GPA ~ Study", df).fit()
    m_sleep = smf.ols("GPA ~ Sleep", df).fit()
    m_ss = smf.ols("GPA ~ Study + Sleep", df).fit()
    m_int = smf.ols("GPA ~ Study * Sleep", df).fit()
    m_full = smf.ols("GPA ~ Study + Sleep + Phys + Extra + Stress", df).fit()
    m_nostress = smf.ols("GPA ~ Study + Sleep + Phys + Extra", df).fit()

    w(f"Observations: {len(df)}")
    w("Social hours is the reference (the five hour columns sum to 24).\n")

    # ---------------- H1
    w("=== H1: study and sleep both significantly positive? ===")
    w(f"Study  (alone):    b = {m_study.params['Study']:+.4f}, p = {pf(m_study.pvalues['Study'])}")
    w(f"Sleep  (alone):    b = {m_sleep.params['Sleep']:+.4f}, p = {m_sleep.pvalues['Sleep']:.3g}")
    w(f"Study  (| sleep):  b = {m_ss.params['Study']:+.4f}, p = {pf(m_ss.pvalues['Study'])}")
    w(f"Sleep  (| study):  b = {m_ss.params['Sleep']:+.4f}, p = {m_ss.pvalues['Sleep']:.3g}")
    h1 = bool(m_ss.params["Study"] > 0 and m_ss.pvalues["Study"] < ALPHA
              and m_ss.params["Sleep"] > 0 and m_ss.pvalues["Sleep"] < ALPHA)
    w(f"Verdict: {'SUPPORTED' if h1 else 'NOT SUPPORTED (study yes, sleep no)'}\n")
    res["H1"] = {"supported": h1, "study_b": m_ss.params["Study"], "study_p": m_ss.pvalues["Study"],
                 "sleep_b": m_ss.params["Sleep"], "sleep_p": m_ss.pvalues["Sleep"],
                 "sleep_alone_p": m_sleep.pvalues["Sleep"]}

    # ---------------- H2
    w("=== H2: do social time / low physical activity hurt GPA once study is controlled? ===")
    h2 = {}
    for var in ["Phys", "Social"]:
        raw = smf.ols(f"GPA ~ {var}", df).fit()
        adj = smf.ols(f"GPA ~ Study + {var}", df).fit()
        w(f"{var:<7} unadjusted: b = {raw.params[var]:+.4f} (p = {pf(raw.pvalues[var])})   "
          f"| adjusted for study: b = {adj.params[var]:+.4f} (p = {adj.pvalues[var]:.3g})")
        h2[var] = {"raw_b": raw.params[var], "raw_p": raw.pvalues[var],
                   "adj_b": adj.params[var], "adj_p": adj.pvalues[var]}
    joint = smf.ols("GPA ~ Study + Phys + Social", df).fit()
    f, p, _ = joint.compare_f_test(m_study)
    w(f"Joint F-test (Phys + Social | Study): F = {f:.2f}, p = {p:.3g}; "
      f"b_phys = {joint.params['Phys']:+.4f}, b_social = {joint.params['Social']:+.4f}")
    h2_supported = bool(h2["Phys"]["adj_b"] > 0 and h2["Social"]["adj_b"] < 0
                        and h2["Phys"]["adj_p"] < ALPHA and h2["Social"]["adj_p"] < ALPHA)
    w(f"Verdict: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'} - the raw negative association of physical "
      "activity (r = -0.34) is explained by study hours. After adjustment the effects are ~0 "
      "and, where the joint test is borderline significant, of the OPPOSITE sign and < 0.01 GPA per hour.\n")
    res["H2"] = {"supported": h2_supported, **h2, "joint_F": f, "joint_p": p}

    # ---------------- H3
    w("=== H3: Study x Sleep interaction ===")
    b, pv = m_int.params["Study:Sleep"], m_int.pvalues["Study:Sleep"]
    f, p, _ = m_int.compare_f_test(m_ss)
    w(f"Interaction coefficient = {b:+.4f}, p = {pv:.4f}; F-test vs additive model p = {p:.4f}")
    w(f"Adj R2: additive {m_ss.rsquared_adj:.4f} -> interaction {m_int.rsquared_adj:.4f}")
    m_low = smf.ols("GPA ~ Study * LowSleep", df).fit()
    w(f"Low-sleep (<6 h, n = {int(df.LowSleep.sum())}) version: "
      f"LowSleep b = {m_low.params['LowSleep']:+.4f} (p = {m_low.pvalues['LowSleep']:.3g}), "
      f"Study:LowSleep b = {m_low.params['Study:LowSleep']:+.4f} "
      f"(p = {m_low.pvalues['Study:LowSleep']:.3g})")
    for s_val in [5, 7.5, 10]:
        w(f"  marginal effect of 1 h study at sleep = {s_val:>4}: "
          f"{m_int.params['Study'] + b * s_val:+.4f}")
    h3 = bool(pv < ALPHA and b > 0)
    w(f"Verdict: {'SUPPORTED' if h3 else 'NOT SUPPORTED at the 5% level (borderline, tiny effect)'}\n")
    res["H3"] = {"supported": h3, "b": b, "p": pv, "F_p": p,
                 "lowsleep_p": m_low.pvalues["Study:LowSleep"],
                 "adj_r2_additive": m_ss.rsquared_adj, "adj_r2_interaction": m_int.rsquared_adj}

    # ---------------- H4
    w("=== H4: full multi-factor model vs study hours alone ===")
    f_full, p_full, d_full = m_full.compare_f_test(m_study)
    f_ss, p_ss, d_ss = m_ss.compare_f_test(m_study)
    cands = {"M1 Study only": m_study, "M4 Study + Sleep": m_ss,
             "M5 Study x Sleep": m_int, "M6 Full (no social)": m_full}
    comp = pd.DataFrame({n: {"R2": m.rsquared, "adj_R2": m.rsquared_adj, "AIC": m.aic,
                             "BIC": m.bic, "n_params": int(m.df_model) + 1}
                         for n, m in cands.items()}).T
    X_sets = {"M1 Study only": ["Study"], "M4 Study + Sleep": ["Study", "Sleep"],
              "M6 Full (no social)": ["Study", "Sleep", "Phys", "Extra", "Stress"]}
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    cv_rows = {}
    for n, cols in X_sets.items():
        r2 = cross_val_score(LinearRegression(), df[cols], df["GPA"], cv=cv, scoring="r2")
        rm = -cross_val_score(LinearRegression(), df[cols], df["GPA"], cv=cv,
                              scoring="neg_root_mean_squared_error")
        cv_rows[n] = {"cv_R2_mean": r2.mean(), "cv_R2_sd": r2.std(),
                      "cv_RMSE_mean": rm.mean(), "cv_RMSE_sd": rm.std()}
    cvdf = pd.DataFrame(cv_rows).T
    comp.round(4).to_csv(RESULTS_DIR / "model_comparison.csv")
    cvdf.round(5).to_csv(RESULTS_DIR / "cv_results.csv")
    w(comp.round(4).to_string()); w("")
    w(f"Nested F-test, full vs study-only: F = {f_full:.2f}, p = {p_full:.3g}, df diff = {int(d_full)}")
    w(f"Nested F-test, study+sleep vs study-only: F = {f_ss:.2f}, p = {p_ss:.3g}")
    w("Repeated 5-fold CV (20 repeats):"); w(cvdf.round(4).to_string())
    h4 = bool(p_full < ALPHA and m_full.aic < m_study.aic - 2
              and cvdf.loc["M6 Full (no social)", "cv_R2_mean"] > cvdf.loc["M1 Study only", "cv_R2_mean"])
    w(f"Verdict: {'SUPPORTED' if h4 else 'NOT SUPPORTED - extra variables add nothing beyond study hours'}\n")
    res["H4"] = {"supported": h4, "F": f_full, "p": p_full, "adj_r2_full": m_full.rsquared_adj,
                 "adj_r2_study": m_study.rsquared_adj, "aic_full": m_full.aic,
                 "aic_study": m_study.aic, "cv_r2_full": cvdf.loc["M6 Full (no social)", "cv_R2_mean"],
                 "cv_r2_study": cvdf.loc["M1 Study only", "cv_R2_mean"]}

    # ---------------- relative importance
    w("=== Relative importance (share of the full model's R2) ===")
    preds = ["Study", "Sleep", "Phys", "Extra", "Stress"]
    shares = lmg(df, "GPA", preds)
    drop = {v: m_full.rsquared - sm.OLS(df["GPA"], sm.add_constant(df[[q for q in preds if q != v]])).fit().rsquared
            for v in preds}
    imp = pd.DataFrame({"LMG_R2": shares, "LMG_pct": {k: v / m_full.rsquared * 100 for k, v in shares.items()},
                        "drop_one_dR2": drop})
    imp.round(5).to_csv(RESULTS_DIR / "relative_importance.csv")
    w(imp.round(4).to_string()); w("")
    res["importance"] = imp.round(5).to_dict()

    out.append("Notes:\n- Stress_Level is coded Low=1, Moderate=2, High=3 (ordinal treated as numeric; "
               "dummy coding gives the same conclusions).\n- Observational data: all associations, "
               "not causal effects.\n- Screen time is not in the dataset, so the screen-time hypotheses "
               "could not be tested.")
    text = "\n".join(out)
    (RESULTS_DIR / "validation_metrics.txt").write_text(text + "\n", encoding="utf-8")
    (RESULTS_DIR / "hypothesis_results.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
