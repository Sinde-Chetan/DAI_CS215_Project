""" Correlation heatmap (Pearson + Spearman) of all numeric variables."""
import pandas as pd
import seaborn as sns

from utils import HOUR_COLS, SHORT, load_clean, savefig, setup_style, RESULTS_DIR

COLUMNS = ["GPA", *HOUR_COLS, "Stress_Level_Num"]


def main():
    plt = setup_style()
    df = load_clean()
    pearson = df[COLUMNS].corr(method="pearson").rename(index=SHORT, columns=SHORT)
    spearman = df[COLUMNS].corr(method="spearman").rename(index=SHORT, columns=SHORT)
    pearson.round(3).to_csv(RESULTS_DIR / "correlation_matrix_pearson.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, mat, name in zip(axes, [pearson, spearman], ["Pearson", "Spearman"]):
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    center=0, square=True, cbar=False, ax=ax)
        ax.set_title(f"{name} correlation")
        ax.grid(False)
    fig.suptitle("Correlation matrix: lifestyle factors and GPA", fontweight="bold")
    savefig(plt, "correlation_heatmap.png")
    print(pearson["GPA"].drop("GPA").sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
