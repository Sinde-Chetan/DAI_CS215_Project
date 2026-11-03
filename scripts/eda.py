"""Step 2b - Exploratory data analysis (owner: Chetan).

Outputs (results/): eda_summary.txt, summary_statistics.csv,
distributions.png, gpa_by_stress.png, gpa_vs_each_factor.png
"""
import numpy as np
import pandas as pd

from utils import HOUR_COLS, RESULTS_DIR, SHORT, load_clean, savefig, setup_style


def main():
    plt = setup_style()
    df = load_clean()
    num = ["GPA", *HOUR_COLS, "Stress_Level_Num"]

    stats = df[num].describe().T
    stats["skew"] = df[num].skew()
    stats.round(3).rename(index=SHORT).to_csv(RESULTS_DIR / "summary_statistics.csv")

    lines = [f"Students: {len(df)}", "", "Summary statistics:",
             stats.round(2).rename(index=SHORT).to_string(), "",
             "Stress level counts:", df["Stress_Level"].value_counts().to_string(), "",
             "Mean by stress level:",
             df.groupby("Stress_Level")[["GPA", *HOUR_COLS]].mean().round(2)
               .rename(columns=SHORT).to_string(), ""]
    lines.append(f"Max |sum of 5 hour columns - 24|: "
                 f"{(df[HOUR_COLS].sum(axis=1) - 24).abs().max():.2e}  -> hours are compositional")
    lines.append(f"Students at the GPA ceiling (4.00): {(df['GPA'] >= 4).sum()}")
    lines.append(f"Students sleeping < 6 h: {(df['Sleep_Hours_Per_Day'] < 6).sum()} "
                 f"({(df['Sleep_Hours_Per_Day'] < 6).mean():.1%})")
    (RESULTS_DIR / "eda_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 1. distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), ["GPA", *HOUR_COLS]):
        ax.hist(df[col], bins=25, color="#2e86ab", edgecolor="white")
        ax.set_title(SHORT[col]); ax.set_ylabel("students")
    ax = axes.ravel()[6]
    df["Stress_Level"].value_counts().reindex(["Low", "Moderate", "High"]).plot.bar(
        ax=ax, color="#c0392b", edgecolor="white", rot=0)
    ax.set_title("Stress level")
    axes.ravel()[7].axis("off")
    fig.suptitle("Distributions of all variables", fontweight="bold")
    savefig(plt, "distributions.png")

    # 2. GPA and study hours by stress level
    order = ["Low", "Moderate", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["GPA", "Study_Hours_Per_Day"]):
        data = [df.loc[df["Stress_Level"] == s, col] for s in order]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks([1, 2, 3], order)
        for patch, c in zip(bp["boxes"], ["#82c0a1", "#f4d35e", "#ee6c4d"]):
            patch.set_facecolor(c)
        ax.set_title(f"{SHORT[col]} by stress level"); ax.set_xlabel("Stress level")
    savefig(plt, "gpa_by_stress.png")

    # 3. GPA against each factor, with OLS line
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), [*HOUR_COLS, "Stress_Level_Num"]):
        x = df[col] + (np.random.default_rng(0).uniform(-.08, .08, len(df))
                       if col == "Stress_Level_Num" else 0)
        ax.scatter(x, df["GPA"], s=8, alpha=.3, color="#2e86ab")
        slope, icpt = np.polyfit(df[col], df["GPA"], 1)
        xs = np.array([df[col].min(), df[col].max()])
        ax.plot(xs, icpt + slope * xs, color="#e74c3c", lw=2.2)
        r = df[col].corr(df["GPA"])
        ax.set_title(f"{SHORT[col]}  (r = {r:+.2f})"); ax.set_ylabel("GPA")
    fig.suptitle("Simple (unadjusted) relationship of each factor with GPA", fontweight="bold")
    savefig(plt, "gpa_vs_each_factor.png")


if __name__ == "__main__":
    main()
