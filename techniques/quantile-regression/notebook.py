"""
Quantile Regression — Prediksi Interval dan Estimasi Risiko

Tutorial lengkap implementasi quantile regression untuk prediction intervals,
risk assessment, dan analisis distribusi penuh.

Author: Hermes Agent
Date: 2026-06-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("QUANTILE REGRESSION TUTORIAL")
print("="*70)

# ============================================================================
# BAGIAN 1: QUANTILE LOSS FUNCTION
# ============================================================================

def quantile_loss(y_true, y_pred, quantile):
    """
    Compute quantile loss (pinball loss)
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    quantile : float
        Quantile level (0 < quantile < 1)
        
    Returns:
    --------
    float : Average quantile loss
    """
    residual = y_true - y_pred
    loss = np.where(residual >= 0, 
                    quantile * residual,
                    (quantile - 1) * residual)
    return np.mean(loss)


def check_coverage(y_true, y_pred, quantile):
    """
    Check empirical coverage of quantile prediction
    
    For quantile τ, should have approximately τ proportion of 
    actual values below prediction.
    
    Returns:
    --------
    float : Proportion of actuals <= prediction
    """
    coverage = np.mean(y_true <= y_pred)
    return coverage


def check_crossing(predictions_dict):
    """
    Check for quantile crossing violations
    
    Predictions should be monotone: Q_τ1 ≤ Q_τ2 for τ1 < τ2
    
    Returns:
    --------
    int : Number of crossing violations
    """
    sorted_quantiles = sorted(predictions_dict.keys())
    crossings = 0
    
    for i in range(len(sorted_quantiles) - 1):
        q_low = predictions_dict[sorted_quantiles[i]]
        q_high = predictions_dict[sorted_quantiles[i+1]]
        crossings += np.sum(q_low > q_high)
    
    return crossings


print("\n" + "="*70)
print("QUANTILE LOSS DEMONSTRATION")
print("="*70)

# Demo: asymmetric penalty
y_true = 100
y_pred_under = 80  # underprediction
y_pred_over = 120  # overprediction

print(f"\nTrue value: {y_true}")
print(f"Underprediction: {y_pred_under} (error = -20)")
print(f"Overprediction: {y_pred_over} (error = +20)")

for q in [0.1, 0.5, 0.9]:
    loss_under = quantile_loss(np.array([y_true]), np.array([y_pred_under]), q)
    loss_over = quantile_loss(np.array([y_true]), np.array([y_pred_over]), q)
    print(f"\nQuantile {q}:")
    print(f"  Loss (underprediction): {loss_under:.2f}")
    print(f"  Loss (overprediction):  {loss_over:.2f}")
    print(f"  Penalty ratio: {loss_under/loss_over:.2f}x")

# ============================================================================
# BAGIAN 2: LOAD DAN EXPLORE DATA
# ============================================================================

print("\n" + "="*70)
print("DATASET: CALIFORNIA HOUSING")
print("="*70)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"\nDataset shape: {df.shape}")
print(f"Target: MedHouseVal (median house value dalam $100k)")
print(f"\nFeatures:")
for col in df.columns[:-1]:
    print(f"  - {col}")

print(f"\nTarget statistics:")
print(df['MedHouseVal'].describe())

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution')
axes[0].axvline(df['MedHouseVal'].median(), color='red', 
                linestyle='--', label='Median')
axes[0].legend()

# Box plot
axes[1].boxplot(df['MedHouseVal'], vert=True)
axes[1].set_ylabel('Median House Value ($100k)')
axes[1].set_title('Target Distribution (Box Plot)')
axes[1].grid(axis='y')

plt.tight_layout()
plt.savefig('techniques/quantile-regression/target_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: target_distribution.png")

# ============================================================================
# BAGIAN 3: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ============================================================================
# BAGIAN 4: TRAIN QUANTILE REGRESSION MODELS
# ============================================================================

print("\n" + "="*70)
print("TRAINING QUANTILE REGRESSION MODELS")
print("="*70)

# Define quantiles
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
models = {}
train_losses = {}
test_losses = {}

print(f"\nTraining models for quantiles: {quantiles}")
print("-" * 70)

for q in quantiles:
    print(f"\nQuantile {q:.2f}:")
    
    # Initialize LightGBM with quantile objective
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,  # quantile parameter
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute losses
    train_loss = quantile_loss(y_train, y_train_pred, q)
    test_loss = quantile_loss(y_test, y_test_pred, q)
    
    # Check coverage
    train_cov = check_coverage(y_train, y_train_pred, q)
    test_cov = check_coverage(y_test, y_test_pred, q)
    
    # Store
    models[q] = model
    train_losses[q] = train_loss
    test_losses[q] = test_loss
    
    print(f"  Train loss: {train_loss:.4f} | Coverage: {train_cov:.3f} (target: {q:.3f})")
    print(f"  Test loss:  {test_loss:.4f} | Coverage: {test_cov:.3f} (target: {q:.3f})")
    print(f"  Iterations: {model.best_iteration_}")

print("\n✓ All models trained successfully!")

# ============================================================================
# BAGIAN 5: GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

# Generate predictions for all quantiles
predictions = {}
for q, model in models.items():
    predictions[q] = model.predict(X_test)

# Create results DataFrame
results_df = pd.DataFrame({
    'y_true': y_test.values,
    'q_0.05': predictions[0.05],
    'q_0.25': predictions[0.25],
    'q_0.50': predictions[0.5],
    'q_0.75': predictions[0.75],
    'q_0.95': predictions[0.95]
})

# Compute prediction intervals
results_df['PI_90'] = results_df['q_0.95'] - results_df['q_0.05']  # 90% interval
results_df['PI_50'] = results_df['q_0.75'] - results_df['q_0.25']  # 50% interval (IQR)

# Check if actual falls in 90% interval
results_df['in_90_PI'] = (
    (results_df['y_true'] >= results_df['q_0.05']) & 
    (results_df['y_true'] <= results_df['q_0.95'])
)

print("\nPrediction Results (first 10 samples):")
print(results_df.head(10).to_string())

print(f"\n90% Prediction Interval Statistics:")
print(f"  Average width: {results_df['PI_90'].mean():.3f}")
print(f"  Empirical coverage: {results_df['in_90_PI'].mean():.3f} (target: 0.90)")

# Check quantile crossing
crossings = check_crossing(predictions)
print(f"\nQuantile crossing violations: {crossings} / {len(y_test)} samples ({100*crossings/len(y_test):.2f}%)")

# ============================================================================
# BAGIAN 6: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATIONS")
print("="*70)

# Plot 1: Prediction Intervals
fig, ax = plt.subplots(figsize=(14, 6))

# Sort by median prediction for better visualization
sorted_idx = np.argsort(results_df['q_0.50'].values)
x_axis = np.arange(len(sorted_idx))

# Plot intervals
ax.fill_between(x_axis, 
                results_df['q_0.05'].values[sorted_idx],
                results_df['q_0.95'].values[sorted_idx],
                alpha=0.2, color='blue', label='90% PI')
ax.fill_between(x_axis,
                results_df['q_0.25'].values[sorted_idx],
                results_df['q_0.75'].values[sorted_idx],
                alpha=0.3, color='blue', label='50% PI (IQR)')

# Plot predictions
ax.plot(x_axis, results_df['q_0.50'].values[sorted_idx], 
        'b-', linewidth=1.5, label='Median Prediction')

# Plot actuals (subsample untuk clarity)
sample_every = 50
ax.scatter(x_axis[::sample_every], 
          results_df['y_true'].values[sorted_idx][::sample_every],
          c='red', s=10, alpha=0.6, label='Actual (sampled)', zorder=5)

ax.set_xlabel('Sample Index (sorted by median prediction)')
ax.set_ylabel('House Value ($100k)')
ax.set_title('Quantile Regression: Prediction Intervals')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('techniques/quantile-regression/prediction_intervals.png', dpi=150, bbox_inches='tight')
print("✓ Saved: prediction_intervals.png")

# Plot 2: Interval Width Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of interval widths
axes[0].hist(results_df['PI_90'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('90% PI Width')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Prediction Interval Widths')
axes[0].axvline(results_df['PI_90'].mean(), color='red', 
                linestyle='--', label=f'Mean: {results_df["PI_90"].mean():.2f}')
axes[0].legend()

# Interval width vs median prediction (heteroscedasticity check)
axes[1].scatter(results_df['q_0.50'], results_df['PI_90'], 
                alpha=0.3, s=10)
axes[1].set_xlabel('Median Prediction ($100k)')
axes[1].set_ylabel('90% PI Width')
axes[1].set_title('Interval Width vs Prediction Level (Heteroscedasticity)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('techniques/quantile-regression/interval_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: interval_analysis.png")

# Plot 3: Coverage Calibration
fig, ax = plt.subplots(figsize=(8, 6))

coverages = []
for q in quantiles:
    cov = check_coverage(y_test, predictions[q], q)
    coverages.append(cov)

ax.plot(quantiles, coverages, 'bo-', markersize=8, linewidth=2, label='Empirical Coverage')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
ax.set_xlabel('Target Quantile')
ax.set_ylabel('Empirical Coverage')
ax.set_title('Quantile Calibration Plot')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Annotate points
for q, cov in zip(quantiles, coverages):
    ax.annotate(f'{cov:.3f}', (q, cov), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('techniques/quantile-regression/calibration_plot.png', dpi=150, bbox_inches='tight')
print("✓ Saved: calibration_plot.png")

# Plot 4: Feature Importance (Median Model)
fig, ax = plt.subplots(figsize=(10, 6))

median_model = models[0.5]
importance = median_model.feature_importances_
feature_names = X.columns

# Sort by importance
sorted_idx = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_idx]
sorted_features = feature_names[sorted_idx]

ax.barh(range(len(sorted_features)), sorted_importance, color='steelblue')
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features)
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance (Median Quantile Model)')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('techniques/quantile-regression/feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_importance.png")

# ============================================================================
# BAGIAN 7: EVALUATION METRICS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EVALUATION METRICS SUMMARY")
print("="*70)

eval_df = pd.DataFrame({
    'Quantile': quantiles,
    'Test_Loss': [test_losses[q] for q in quantiles],
    'Coverage': [check_coverage(y_test, predictions[q], q) for q in quantiles],
    'Coverage_Error': [abs(check_coverage(y_test, predictions[q], q) - q) for q in quantiles]
})

print("\n" + eval_df.to_string(index=False))

# Prediction interval metrics
pi_90_coverage = results_df['in_90_PI'].mean()
pi_90_width_mean = results_df['PI_90'].mean()
pi_90_width_std = results_df['PI_90'].std()

print(f"\n90% Prediction Interval Metrics:")
print(f"  Empirical coverage: {pi_90_coverage:.4f} (target: 0.9000)")
print(f"  Coverage error: {abs(pi_90_coverage - 0.9):.4f}")
print(f"  Mean width: {pi_90_width_mean:.3f}")
print(f"  Std width: {pi_90_width_std:.3f}")
print(f"  Width/Range ratio: {pi_90_width_mean / (y_test.max() - y_test.min()):.3f}")

# Median prediction accuracy (for comparison)
mae_median = mean_absolute_error(y_test, predictions[0.5])
rmse_median = np.sqrt(mean_squared_error(y_test, predictions[0.5]))

print(f"\nMedian Prediction Accuracy:")
print(f"  MAE: {mae_median:.4f}")
print(f"  RMSE: {rmse_median:.4f}")

# ============================================================================
# BAGIAN 8: REAL-WORLD USE CASE DEMO
# ============================================================================

print("\n" + "="*70)
print("REAL-WORLD USE CASE DEMO")
print("="*70)

# Pick a sample house
sample_idx = 42
sample_features = X_test.iloc[sample_idx:sample_idx+1]
actual_value = y_test.iloc[sample_idx]

print(f"\nSample House Features:")
print(sample_features.T.to_string())

# Generate predictions
sample_preds = {}
for q in quantiles:
    sample_preds[q] = models[q].predict(sample_features)[0]

print(f"\nActual Value: ${actual_value:.2f} (hundred thousand)")
print(f"\nQuantile Predictions:")
for q in quantiles:
    print(f"  Q_{q:.2f}: ${sample_preds[q]:>6.2f}")

# Interpretation
print(f"\n📊 Interpretation:")
print(f"  • Median estimate: ${sample_preds[0.5]:.2f}")
print(f"  • 50% PI: [${sample_preds[0.25]:.2f}, ${sample_preds[0.75]:.2f}]")
print(f"  • 90% PI: [${sample_preds[0.05]:.2f}, ${sample_preds[0.95]:.2f}]")
print(f"  • Uncertainty (PI width): ${sample_preds[0.95] - sample_preds[0.05]:.2f}")

if actual_value >= sample_preds[0.05] and actual_value <= sample_preds[0.95]:
    print(f"  ✓ Actual value falls within 90% PI")
else:
    print(f"  ✗ Actual value outside 90% PI (rare event!)")

# ============================================================================
# BAGIAN 9: COMPARISON WITH OLS
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: QUANTILE REGRESSION vs OLS")
print("="*70)

# Train OLS model (using LightGBM with MSE objective)
ols_model = lgb.LGBMRegressor(
    objective='regression',  # MSE objective
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

ols_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

y_pred_ols = ols_model.predict(X_test)

# Compare median quantile vs OLS
mae_ols = mean_absolute_error(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))

print(f"\nOLS (Mean Prediction):")
print(f"  MAE:  {mae_ols:.4f}")
print(f"  RMSE: {rmse_ols:.4f}")

print(f"\nMedian Quantile Regression:")
print(f"  MAE:  {mae_median:.4f}")
print(f"  RMSE: {rmse_median:.4f}")

print(f"\nKey Differences:")
print(f"  • OLS optimizes MSE → predicts conditional mean")
print(f"  • Median QR optimizes MAE → predicts conditional median")
print(f"  • Median is robust to outliers; mean is not")
print(f"  • QR provides full distribution; OLS only point estimate")

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by OLS prediction
sorted_idx = np.argsort(y_pred_ols)
x_axis = np.arange(len(sorted_idx))

ax.plot(x_axis, y_test.values[sorted_idx], 'k.', markersize=2, 
        alpha=0.5, label='Actual')
ax.plot(x_axis, y_pred_ols[sorted_idx], 'r-', linewidth=2, 
        label='OLS (Mean)')
ax.plot(x_axis, predictions[0.5][sorted_idx], 'b-', linewidth=2, 
        label='Quantile (Median)')

ax.set_xlabel('Sample Index (sorted by OLS prediction)')
ax.set_ylabel('House Value ($100k)')
ax.set_title('OLS vs Median Quantile Regression')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('techniques/quantile-regression/ols_vs_quantile.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: ols_vs_quantile.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TUTORIAL COMPLETE!")
print("="*70)

print(f"""
✅ Summary:
  • Trained {len(quantiles)} quantile regression models
  • Generated prediction intervals with {pi_90_coverage:.1%} coverage
  • Average 90% PI width: ${pi_90_width_mean:.2f} (hundred thousand)
  • Quantile crossing violations: {100*crossings/len(y_test):.2f}%
  • Models well-calibrated (coverage ≈ target quantiles)

📁 Generated Files:
  • target_distribution.png
  • prediction_intervals.png
  • interval_analysis.png
  • calibration_plot.png
  • feature_importance.png
  • ols_vs_quantile.png

🎯 Key Takeaways:
  1. Quantile regression provides distributional information
  2. Prediction intervals quantify uncertainty
  3. Robust to outliers (median regression)
  4. Useful for risk assessment and decision-making
  5. Asymmetric loss enables flexible risk preferences

📚 Next Steps:
  • Explore conformal prediction for distribution-free intervals
  • Try different models (neural networks, quantile forests)
  • Apply to your own datasets
  • Experiment with extreme quantiles for tail risk
""")

print("="*70)
