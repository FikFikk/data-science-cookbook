"""
Conformal Prediction Tutorial: Guaranteed Uncertainty Estimation in Python
==========================================================================

Implementasi lengkap Conformal Prediction dari awal (from scratch) menggunakan
scikit-learn, numpy, pandas, dan matplotlib.
Topik:
1. Split Conformal Classification (dengan Non-conformity Score)
2. Softmax Split Conformal Score
3. Split Conformal Regression (Prediction Intervals)
4. Evaluasi Coverage Rate dan Avg Set Size / Interval Width

Dependencies:
    pip install numpy pandas scikit-learn matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=== CONFORMAL PREDICTION IMPLEMENTATION FROM SCRATCH ===")

# ==========================================
# 1. CONFORMAL CLASSIFICATION
# ==========================================
print("\n[Fase 1] Conformal Classification (Predictive Sets with Coverage Guarantee)")

# Generate synthetic multi-class dataset
X_cls, y_cls = make_classification(
    n_samples=2000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2, 
    n_classes=3, 
    n_clusters_per_class=1, 
    random_state=42
)

# Split into Train (50%), Calibration (25%), Test (25%)
X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(X_cls, y_cls, test_size=0.5, random_state=42)
X_cal_c, X_test_c, y_cal_c, y_test_c = train_test_split(X_temp_c, y_temp_c, test_size=0.5, random_state=42)

print(f"Dataset Classification split: Train={len(X_train_c)}, Calib={len(X_cal_c)}, Test={len(X_test_c)}")

# Train base classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)

# Evaluate base model standard accuracy
test_preds = clf.predict(X_test_c)
print(f"Base Classifier Standard Test Accuracy: {accuracy_score(y_test_c, test_preds):.4f}")

# Calculate Non-conformity scores on Calibration Set
# Non-conformity score S_i = 1 - P(Y = y_i | X_i)
cal_probs = clf.predict_proba(X_cal_c)
# Extract true class probability for each calibration sample
cal_true_probs = cal_probs[np.arange(len(y_cal_c)), y_cal_c]
cal_scores_c = 1.0 - cal_true_probs

# Function to compute conformal threshold q_hat
def get_conformal_quantile(scores, alpha):
    n = len(scores)
    # Finite-sample correction formula: q = ceil((n+1)(1-alpha)) / n quantile
    level = np.ceil((n + 1) * (1 - alpha)) / n
    # Use quantile method with higher method to ensure exact coverage
    q_hat = np.quantile(scores, np.clip(level, 0, 1), method='higher')
    return q_hat

# Desired error rate alpha = 0.10 (Target Coverage = 90%)
alpha_c = 0.10
q_hat_c = get_conformal_quantile(cal_scores_c, alpha_c)
print(f"Target Coverage: {(1-alpha_c)*100:.1f}%, Quantile Threshold (q_hat): {q_hat_c:.4f}")

# Generate prediction sets for Test Set
test_probs = clf.predict_proba(X_test_c)
# A class k is included in prediction set if 1 - P(Y=k|X) <= q_hat  ==> P(Y=k|X) >= 1 - q_hat
prediction_sets_c = [np.where(probs >= (1.0 - q_hat_c))[0] for probs in test_probs]

# Evaluate Classification Coverage & Average Set Size
covered_c = [y_test_c[i] in prediction_sets_c[i] for i in range(len(y_test_c))]
coverage_rate_c = np.mean(covered_c)
avg_set_size = np.mean([len(s) for s in prediction_sets_c])

print(f"Empirical Test Coverage Rate: {coverage_rate_c * 100:.2f}%")
print(f"Average Prediction Set Size: {avg_set_size:.2f} classes")

# Visualizing distribution of set sizes
set_sizes = [len(s) for s in prediction_sets_c]
plt.figure(figsize=(8, 5))
plt.hist(set_sizes, bins=np.arange(0.5, 4.5, 1), rwidth=0.8, color='skyblue', edgecolor='black')
plt.title(f"Distribusi Ukuran Prediction Set (Target Coverage: {(1-alpha_c)*100:.0f}%)", fontsize=12)
plt.xlabel("Ukuran Prediction Set (Jumlah Kelas)")
plt.ylabel("Frekuensi Samples")
plt.xticks([1, 2, 3])
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/conformal-prediction/classification_set_sizes.png')
plt.close()
print("Saved plot: classification_set_sizes.png")


# ==========================================
# 2. CONFORMAL REGRESSION
# ==========================================
print("\n[Fase 2] Conformal Regression (Predictive Intervals with Guarantee)")

# Generate non-linear regression data with heteroscedastic noise
X_reg = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1 + 0.1 * np.abs(X_reg.ravel()), size=1000)

X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(X_reg, y_reg, test_size=0.5, random_state=42)
X_cal_r, X_test_r, y_cal_r, y_test_r = train_test_split(X_temp_r, y_temp_r, test_size=0.5, random_state=42)

# Train Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)

# Residuals on Calibration set: R_i = |y_i - y_hat_i|
cal_preds_r = reg.predict(X_cal_r)
cal_residuals_r = np.abs(y_cal_r - cal_preds_r)

# Compute quantile threshold for regression
alpha_r = 0.05 # 95% Target Coverage
q_hat_r = get_conformal_quantile(cal_residuals_r, alpha_r)
print(f"Regression Target Coverage: {(1-alpha_r)*100:.1f}%, Residual Threshold (q_hat): {q_hat_r:.4f}")

# Construct prediction intervals for Test set
test_preds_r = reg.predict(X_test_r)
lower_bound = test_preds_r - q_hat_r
upper_bound = test_preds_r + q_hat_r

# Evaluate Regression Coverage
covered_r = (y_test_r >= lower_bound) & (y_test_r <= upper_bound)
coverage_rate_r = np.mean(covered_r)
avg_width = np.mean(upper_bound - lower_bound)

print(f"Empirical Regression Coverage Rate: {coverage_rate_r * 100:.2f}%")
print(f"Average Interval Width: {avg_width:.4f}")

# Plotting Regression Predictions and Conformal Bands
sort_idx = np.argsort(X_test_r.ravel())
X_test_sorted = X_test_r.ravel()[sort_idx]
y_test_sorted = y_test_r[sort_idx]
test_preds_sorted = test_preds_r[sort_idx]
lower_sorted = lower_bound[sort_idx]
upper_sorted = upper_bound[sort_idx]

plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted, y_test_sorted, color='gray', alpha=0.5, label='Actual Test Points', s=20)
plt.plot(X_test_sorted, test_preds_sorted, color='blue', lw=2, label='Point Prediction (Random Forest)')
plt.fill_between(X_test_sorted, lower_sorted, upper_sorted, color='orange', alpha=0.3, label=f'{(1-alpha_r)*100:.0f}% Conformal Interval')
plt.title(f"Split Conformal Regression (Empirical Coverage: {coverage_rate_r*100:.1f}%)", fontsize=14)
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.legend()
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/conformal-prediction/regression_intervals.png')
plt.close()
print("Saved plot: regression_intervals.png")

# Save summary results
df_summary = pd.DataFrame([
    {
        "Task": "Classification",
        "Target Coverage": f"{(1-alpha_c)*100:.1f}%",
        "Empirical Coverage": f"{coverage_rate_c*100:.2f}%",
        "Metric": f"Avg Set Size: {avg_set_size:.2f}"
    },
    {
        "Task": "Regression",
        "Target Coverage": f"{(1-alpha_r)*100:.1f}%",
        "Empirical Coverage": f"{coverage_rate_r*100:.2f}%",
        "Metric": f"Avg Width: {avg_width:.4f}"
    }
])
print("\n=== SUMMARY RESULTS ===")
print(df_summary.to_string(index=False))

print("\nScript completed successfully!")
