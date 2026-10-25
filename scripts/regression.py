import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def plot_standalone_regression(df, x_col, y_col, model, title, dot_color, filename):
    plt.figure(figsize=(8, 6))
    
    # Scattering plot 
    plt.scatter(df[x_col], df[y_col], alpha=0.35, color=dot_color, edgecolor='none', s=40, label='Student Data')
    
    # Calculate OLS trendline points 
    intercept = model.params['const']
    slope = model.params[x_col]
    x_min, x_max = df[x_col].min(), df[x_col].max()
    plt.plot([x_min, x_max], [intercept + slope * x_min, intercept + slope * x_max], 
             color='#e74c3c', linewidth=2.5, label='OLS Trendline')
    
    # Extract R-squared and p-value for the text box annotation
    #r2 = model.rsquared
    #p_val = model.pvalues[x_col]
    #p_text = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
    
    stats_text = f"Slope = {slope:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#145a32')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=15,
                   verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Formatting labels and title
    clean_title = title.replace('_', ' ').title()
    plt.xlabel(x_col.replace('_', ' '), fontsize=11, fontweight='semibold', labelpad=8)
    plt.ylabel(y_col, fontsize=11, fontweight='semibold', labelpad=8)
    plt.title(f"{clean_title} vs. GPA (Standalone Model)", fontsize=13, fontweight='bold', pad=15)
    
    # Legend and background grid styling
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()


# Loading cleaned dataset
df = pd.read_csv('../data/cleaned_student_data.csv')
print(df.head())

# Converting 'Stress_Level' text to numbers
stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)

# Model 1: Study Hours vs. GPA
X_study = sm.add_constant(df['Study_Hours_Per_Day'])
y = df['GPA']
model_study = sm.OLS(y, X_study).fit()
print("--- Study Hours Only Model ---")
print(model_study.summary())


# 1. Generate Plot
plot_standalone_regression(
    df=df, 
    x_col='Study_Hours_Per_Day', 
    y_col='GPA', 
    model=model_study, 
    title='Study Hours', 
    dot_color='#2980b9',
    filename='../notebooks/study_vs_gpa_model.png'
)


# Model 2: Sleep Hours vs. GPA
X_sleep = sm.add_constant(df['Sleep_Hours_Per_Day'])
model_sleep = sm.OLS(y, X_sleep).fit()
print("\n--- Sleep Hours Only Model ---")
print(model_sleep.summary())


# 2. Generate Plot
plot_standalone_regression(
    df=df, 
    x_col='Sleep_Hours_Per_Day', 
    y_col='GPA', 
    model=model_sleep, 
    title='Sleep Hours', 
    dot_color='#d35400',
    filename='../notebooks/sleep_vs_gpa_model.png'
)


# Model 3: Stress Level vs. GPA
X_stress = sm.add_constant(df['Stress_Level'])
model_stress = sm.OLS(y, X_stress).fit()
print("--- Stress Level Only Model ---")
print(model_stress.summary())

# 3. Generate Plot
plot_standalone_regression(
    df=df, 
    x_col='Stress_Level', 
    y_col='GPA', 
    model=model_stress, 
    title='Stress Level', 
    dot_color='#196f3d',
    filename='../notebooks/stress_vs_gpa_model.png'
)


# Selecting independent variables (features)
features = [
    'Study_Hours_Per_Day', 
    'Sleep_Hours_Per_Day', 
    'Physical_Activity_Hours_Per_Day', 
    'Social_Hours_Per_Day', 
    'Extracurricular_Hours_Per_Day', 
    'Stress_Level'
]

for col in features + ['GPA']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X_multi = sm.add_constant(df[features])
model_multi = sm.OLS(y, X_multi).fit()

print("\n--- Multivariable Regression Model ---")
print(model_multi.summary())


# 1. Coefficient Plot
params = model_multi.params.drop('const')
conf = model_multi.conf_int().drop('const')

params.index = params.index.str.replace('_', ' ').str.title()
conf.index = conf.index.str.replace('_', ' ').str.title()

errors = conf[1] - params

plt.figure(figsize=(10, 6))

# Color-code bars: Green for positive, Red for negative
bar_colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in params]

ax = params.plot(kind='barh', xerr=errors, capsize=5, color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.8)

# Center vertical line at 0
plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)


max_val = max(abs(params.min()), abs(params.max())) + max(errors) + 0.05
plt.xlim(-max_val, max_val)

plt.title('Impact of Lifestyle Factors on GPA\n(Multivariable Regression Coefficients with 95% Confidence Intervals)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Change in GPA per Unit Increase', fontsize=10, labelpad=10)
plt.ylabel('Lifestyle Predictors', fontsize=10, labelpad=10)

plt.grid(axis='x', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('../notebooks/regression_coefficients.png', dpi=300)
#plt.show()


# 2. Actual vs. Predicted Plot
# Generating predictions from multivariable model
predictions = model_multi.predict(X_multi)

plt.figure(figsize=(8, 6))

# Scatter plot of Actual vs. Predicted GPA
plt.scatter(df['GPA'], predictions, alpha=0.4, color='#2980b9', edgecolor='none', s=40, label='Student Predictions')

# 1. Plotting the ideal fit line (y = x)
min_val = min(df['GPA'].min(), predictions.min())
max_val = max(df['GPA'].max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#6e2c00', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')

# 2. Fitting and plotting the actual OLS best-fit regression line for the predictions
X_fit = sm.add_constant(df['GPA'])
fit_model = sm.OLS(predictions, X_fit).fit()
fit_intercept = fit_model.params['const']
fit_slope = fit_model.params['GPA']

x_line = [df['GPA'].min(), df['GPA'].max()]
y_line = [fit_intercept + fit_slope * x_line[0], fit_intercept + fit_slope * x_line[1]]
plt.plot(x_line, y_line, color='#e74c3c', linestyle='-', linewidth=2, label='Best-Fit Regression Line')

# Extracting R-squared from multivariable model
r2_multi = model_multi.rsquared

# Adding a professional info box with key statistics
stats_text = f"R² = {r2_multi:.4f}\nModel: Multivariable OLS"
props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#145a32') 
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=15,
               verticalalignment='top', bbox=props, fontfamily='monospace')

# Formatting labels and title
plt.xlabel('Actual GPA', fontsize=11, fontweight='semibold', labelpad=8)
plt.ylabel('Predicted GPA', fontsize=11, fontweight='semibold', labelpad=8)
plt.title('Actual vs. Predicted GPA (Multivariable Model)', fontsize=13, fontweight='bold', pad=15)

# Legend and background grid styling
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_axisbelow(True) 

plt.tight_layout()
plt.savefig('../notebooks/actual_vs_predicted_gpa.png', dpi=300)
#plt.show()


# Saving all model summaries to a single text file in notebooks folder
summary_path = '../notebooks/regression_results_summary.txt'

with open(summary_path, 'w') as f:
    f.write("=== STUDY HOURS OLS MODEL ===\n")
    f.write(model_study.summary().as_text() + "\n\n")
    
    f.write("=== SLEEP HOURS OLS MODEL ===\n")
    f.write(model_sleep.summary().as_text() + "\n\n")
    
    f.write("=== STRESS LEVEL OLS MODEL ===\n")
    f.write(model_stress.summary().as_text() + "\n\n")
    
    f.write("=== MULTIVARIABLE OLS MODEL ===\n")
    f.write(model_multi.summary().as_text() + "\n")

print(f"All regression summaries successfully saved to {summary_path}")










