"""
SHAP: Explainable AI dengan Shapley Values
Tutorial Lengkap dengan Adult Income Dataset

Dataset: UCI Adult Income (predict income >50K)
Model: LightGBM Classifier
Explainer: TreeExplainer (exact Shapley values)

Dependencies:
    pip install shap lightgbm scikit-learn pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import lightgbm as lgb
import shap

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("SHAP Tutorial: Explainable AI untuk Income Prediction")
print("=" * 80)

# ============================================================================
# BAGIAN 1: LOAD DAN PREPROCESSING DATA
# ============================================================================

print("\n[1] LOADING DATASET...")

# Adult Income dataset dari UCI
# Features: age, workclass, education, occupation, etc.
# Target: income >50K or <=50K

# Load dataset menggunakan SHAP built-in
X, y = shap.datasets.adult()

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(X.head())

print(f"\nTarget distribution:")
target_series = pd.Series(y)
print(target_series.value_counts())
print(f"Proportion >50K: {y.mean():.2%}")

# ============================================================================
# BAGIAN 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[2] EXPLORATORY DATA ANALYSIS...")

# Check missing values
print(f"\nMissing values per column:")
print(X.isnull().sum())

# Descriptive statistics untuk numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical features: {numerical_features}")
print(X[numerical_features].describe())

# Categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical features: {categorical_features}")

# ============================================================================
# BAGIAN 3: FEATURE ENGINEERING
# ============================================================================

print("\n[3] FEATURE ENGINEERING...")

# Encode categorical variables
# Note: SHAP's Adult dataset sudah pre-encoded, tapi kita tunjukkan prosesnya

# Untuk categorical features, kita bisa pakai:
# 1. Label Encoding (untuk tree-based models — OK)
# 2. One-Hot Encoding (untuk linear models)

# Kita pakai Label Encoding karena model kita XGBoost (tree-based)
X_encoded = X.copy()

for col in categorical_features:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

print(f"Encoded dataset shape: {X_encoded.shape}")
print(f"Sample encoded data:")
print(X_encoded.head())

# Convert target ke binary (0/1)
y_binary = (y == True).astype(int)

# ============================================================================
# BAGIAN 4: TRAIN-TEST SPLIT
# ============================================================================

print("\n[4] SPLITTING DATA...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_binary, 
    test_size=0.2, 
    random_state=42,
    stratify=y_binary  # Preserve class distribution
)

print(f"Training set: {X_train.shape}, positive class: {y_train.mean():.2%}")
print(f"Test set: {X_test.shape}, positive class: {y_test.mean():.2%}")

# ============================================================================
# BAGIAN 5: TRAIN MODEL
# ============================================================================

print("\n[5] TRAINING LIGHTGBM MODEL...")

# LightGBM dengan parameter tuned
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1  # Suppress warnings
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ============================================================================
# BAGIAN 6: EVALUATE MODEL
# ============================================================================

print("\n[6] EVALUATING MODEL PERFORMANCE...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance dari LightGBM (baseline untuk comparison)
print("\n[LightGBM Built-in] Top 10 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print(feature_importance)

# ============================================================================
# BAGIAN 7: SHAP EXPLAINER
# ============================================================================

print("\n[7] CREATING SHAP EXPLAINER...")

# TreeExplainer untuk tree-based models (XGBoost, Random Forest, LightGBM)
# TreeExplainer memberikan EXACT Shapley values dengan polynomial time complexity!
explainer = shap.TreeExplainer(model)

print(f"Explainer created: {type(explainer)}")

# LightGBM returns array for expected_value, get the positive class value
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    if len(expected_value) > 1:
        expected_value = expected_value[1]  # Index 1 = positive class
    else:
        expected_value = expected_value[0]  # Single value
print(f"Expected value (base value): {expected_value:.4f}")

# Compute SHAP values untuk test set
# Note: Untuk dataset besar, sample dulu (e.g., 1000 instances)
n_explain = min(1000, len(X_test))
X_test_sample = X_test.iloc[:n_explain]

print(f"\nComputing SHAP values for {n_explain} test instances...")
shap_values = explainer.shap_values(X_test_sample)

# LightGBM returns list of arrays for binary classification [class_0, class_1]
# We want class 1 (positive class, >50K)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"SHAP values shape: {shap_values.shape}")
print(f"SHAP values computed successfully!")

# ============================================================================
# BAGIAN 8: VISUALISASI SHAP — GLOBAL EXPLANATIONS
# ============================================================================

print("\n[8] GLOBAL EXPLANATIONS...")

# 8.1 Summary Plot (bar) — Feature Importance Ranking
print("\n[8.1] SHAP Summary Plot (Bar) — Global Feature Importance")
print("      Feature importance berdasarkan mean absolute SHAP value")

plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_feature_importance.png', dpi=150, bbox_inches='tight')
print("      ✓ Saved: shap_feature_importance.png")
plt.close()

# 8.2 Summary Plot (beeswarm) — Distribution of Impact
print("\n[8.2] SHAP Summary Plot (Beeswarm) — Impact Distribution")
print("      Setiap dot = 1 instance, color = feature value (merah=high, biru=low)")
print("      X-axis = SHAP value (kontribusi ke prediksi)")

plt.figure()
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP Summary Plot — Feature Impact Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
print("      ✓ Saved: shap_summary_beeswarm.png")
plt.close()

# Interpretasi Summary Plot
print("\n      INTERPRETASI SUMMARY PLOT:")
print("      - Feature di top = paling important (highest mean |SHAP value|)")
print("      - Spread horizontal = variability of impact")
print("      - Color pattern:")
print("        * Merah (high value) di kanan = positive correlation")
print("        * Biru (low value) di kanan = negative correlation")

# ============================================================================
# BAGIAN 9: VISUALISASI SHAP — LOCAL EXPLANATIONS
# ============================================================================

print("\n[9] LOCAL EXPLANATIONS — Individual Predictions")

# Pilih beberapa instances untuk explain
# Instance 1: High income prediction
y_test_sample = y_test.iloc[:n_explain] if hasattr(y_test, 'iloc') else y_test[:n_explain]
high_income_idx = np.where(y_test_sample == 1)[0][0]
# Instance 2: Low income prediction
low_income_idx = np.where(y_test_sample == 0)[0][0]

actual_high = y_test.iloc[high_income_idx] if hasattr(y_test, 'iloc') else y_test[high_income_idx]
actual_low = y_test.iloc[low_income_idx] if hasattr(y_test, 'iloc') else y_test[low_income_idx]

print(f"\nExplaining 2 instances:")
print(f"  Instance A (index {high_income_idx}): Actual = {actual_high} (>50K)")
print(f"  Instance B (index {low_income_idx}): Actual = {actual_low} (<=50K)")

# 9.1 Waterfall Plot — Instance A (High Income)
print("\n[9.1] Waterfall Plot untuk Instance A (High Income)")
print("      Menunjukkan bagaimana base value (expected) berubah step-by-step")

shap_explanation_A = shap.Explanation(
    values=shap_values[high_income_idx],
    base_values=expected_value,
    data=X_test_sample.iloc[high_income_idx],
    feature_names=X_test_sample.columns.tolist()
)

plt.figure()
shap.waterfall_plot(shap_explanation_A, show=False)
plt.title(f"Waterfall Plot — Instance A (High Income Prediction)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_waterfall_high_income.png', dpi=150, bbox_inches='tight')
print("      ✓ Saved: shap_waterfall_high_income.png")
plt.close()

# 9.2 Waterfall Plot — Instance B (Low Income)
print("\n[9.2] Waterfall Plot untuk Instance B (Low Income)")

shap_explanation_B = shap.Explanation(
    values=shap_values[low_income_idx],
    base_values=expected_value,
    data=X_test_sample.iloc[low_income_idx],
    feature_names=X_test_sample.columns.tolist()
)

plt.figure()
shap.waterfall_plot(shap_explanation_B, show=False)
plt.title(f"Waterfall Plot — Instance B (Low Income Prediction)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_waterfall_low_income.png', dpi=150, bbox_inches='tight')
print("      ✓ Saved: shap_waterfall_low_income.png")
plt.close()

# 9.3 Force Plot — Instance A
print("\n[9.3] Force Plot untuk Instance A")
print("      Horizontal visualization: features pushing prediction left (negative) or right (positive)")

shap.force_plot(
    expected_value,
    shap_values[high_income_idx],
    X_test_sample.iloc[high_income_idx],
    matplotlib=True,
    show=False
)
plt.title(f"Force Plot — Instance A (High Income)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_force_high_income.png', dpi=150, bbox_inches='tight')
print("      ✓ Saved: shap_force_high_income.png")
plt.close()

# ============================================================================
# BAGIAN 10: DEPENDENCE PLOTS — Feature Interactions
# ============================================================================

print("\n[10] DEPENDENCE PLOTS — Feature Relationships")

# Pilih top features untuk analyze
top_features = feature_importance['feature'].head(3).tolist()

for i, feature in enumerate(top_features):
    print(f"\n[10.{i+1}] Dependence Plot: {feature}")
    print(f"      X-axis = {feature} value, Y-axis = SHAP value")
    print(f"      Color = auto-detected interaction feature")
    
    plt.figure()
    shap.dependence_plot(
        feature,
        shap_values,
        X_test_sample,
        interaction_index='auto',
        show=False
    )
    plt.title(f"SHAP Dependence Plot — {feature}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    safe_filename = feature.replace(' ', '_').replace('/', '_')
    plt.savefig(f'/root/data-science-cookbook/techniques/shap-explainable-ai/shap_dependence_{safe_filename}.png', dpi=150, bbox_inches='tight')
    print(f"      ✓ Saved: shap_dependence_{safe_filename}.png")
    plt.close()

# ============================================================================
# BAGIAN 11: DETAILED ANALYSIS — Instance Breakdown
# ============================================================================

print("\n[11] DETAILED INSTANCE ANALYSIS")

def explain_instance(idx, X_data, shap_vals, y_true, y_pred_proba, model_expected_value):
    """
    Print detailed explanation untuk 1 instance
    """
    print(f"\n{'='*80}")
    print(f"INSTANCE #{idx} ANALYSIS")
    print(f"{'='*80}")
    
    # Feature values
    instance = X_data.iloc[idx]
    print(f"\nFeature Values:")
    for col in instance.index:
        print(f"  {col:20s}: {instance[col]}")
    
    # Prediction
    pred_proba = y_pred_proba[idx]
    actual = y_true[idx] if idx < len(y_true) else "N/A"
    print(f"\nPrediction:")
    print(f"  Predicted probability (>50K): {pred_proba:.4f}")
    print(f"  Predicted class: {'> 50K' if pred_proba > 0.5 else '<= 50K'}")
    print(f"  Actual class: {'> 50K' if actual == 1 else '<= 50K' if actual == 0 else 'N/A'}")
    
    # SHAP breakdown
    shap_instance = shap_vals[idx]
    print(f"\nSHAP Value Breakdown:")
    print(f"  Base value (expected): {model_expected_value:.4f}")
    
    # Sort features by absolute SHAP value
    feature_contributions = pd.DataFrame({
        'feature': X_data.columns,
        'value': instance.values,
        'shap_value': shap_instance
    }).sort_values('shap_value', key=abs, ascending=False)
    
    print(f"\n  Top 10 Contributing Features:")
    print(f"  {'Feature':<20s} {'Value':<10s} {'SHAP':<10s} {'Impact':<10s}")
    print(f"  {'-'*60}")
    
    for _, row in feature_contributions.head(10).iterrows():
        impact = "↑ positive" if row['shap_value'] > 0 else "↓ negative"
        print(f"  {row['feature']:<20s} {str(row['value']):<10s} {row['shap_value']:>9.4f}  {impact}")
    
    # Sum check (Efficiency property)
    total_shap = shap_instance.sum()
    final_prediction = model_expected_value + total_shap
    print(f"\n  Sum of SHAP values: {total_shap:.4f}")
    print(f"  Base + Sum: {model_expected_value:.4f} + {total_shap:.4f} = {final_prediction:.4f}")
    print(f"  Model output (log-odds): {final_prediction:.4f}")
    print(f"  Convert to probability: {1/(1+np.exp(-final_prediction)):.4f}")
    
    print(f"\n  ✓ SHAP Efficiency property satisfied: Σ(SHAP) = f(x) - E[f(X)]")

# Explain Instance A (High Income)
explain_instance(
    high_income_idx,
    X_test_sample,
    shap_values,
    y_test,
    y_pred_proba,
    expected_value
)

# Explain Instance B (Low Income)
explain_instance(
    low_income_idx,
    X_test_sample,
    shap_values,
    y_test,
    y_pred_proba,
    expected_value
)

# ============================================================================
# BAGIAN 12: COMPARISON — SHAP vs Built-in Feature Importance
# ============================================================================

print("\n[12] COMPARISON: SHAP vs LightGBM Feature Importance")

# SHAP feature importance (mean absolute SHAP value)
shap_importance = pd.DataFrame({
    'feature': X_test_sample.columns,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

# LightGBM feature importance
lgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'lgb_importance': model.feature_importances_
}).sort_values('lgb_importance', ascending=False)

# Merge
comparison = shap_importance.merge(lgb_importance, on='feature')
comparison['rank_shap'] = comparison['shap_importance'].rank(ascending=False)
comparison['rank_lgb'] = comparison['lgb_importance'].rank(ascending=False)
comparison['rank_diff'] = (comparison['rank_shap'] - comparison['rank_lgb']).abs()

print("\nTop 15 Features Comparison:")
print(comparison.head(15).to_string(index=False))

print("\n\nBiggest Ranking Differences:")
print(comparison.nlargest(5, 'rank_diff')[['feature', 'rank_shap', 'rank_lgb', 'rank_diff']].to_string(index=False))

print("\n\nINSIGHT:")
print("  - LightGBM feature_importances_ based on split count/gain")
print("  - SHAP based on Shapley values (game theory, consistent attribution)")
print("  - Ranking differences reveal features with interaction effects")
print("  - SHAP is more reliable for feature importance interpretation")

# ============================================================================
# BAGIAN 13: ACTIONABLE INSIGHTS
# ============================================================================

print("\n[13] ACTIONABLE INSIGHTS dari SHAP Analysis")

print("\n1. GLOBAL INSIGHTS (dari Summary Plot):")
print("   - Top predictors untuk high income:")

top_5_features = shap_importance.head(5)
for idx, row in top_5_features.iterrows():
    feature = row['feature']
    importance = row['shap_importance']
    print(f"     • {feature}: mean |SHAP| = {importance:.4f}")

print("\n   - Features ini yang paling berpengaruh terhadap prediksi")
print("   - Focus on these untuk feature engineering / data collection")

print("\n2. LOCAL INSIGHTS (dari Instance Analysis):")
print("   - Untuk high earners (>50K):")
print("     • Typical pattern: high education, senior position, long work hours")
print("     • Key differentiators visible in waterfall plot")

print("\n   - Untuk low earners (<=50K):")
print("     • Pattern: lower education, junior/service jobs, shorter work hours")
print("     • Individual explanations help understand edge cases")

print("\n3. FAIRNESS CHECK:")
print("   - Perhatikan features seperti 'Sex', 'Race' jika ada")
print("   - High SHAP value untuk protected attributes → potential bias")
print("   - Use SHAP untuk audit model fairness sebelum deployment")

print("\n4. MODEL DEBUGGING:")
print("   - SHAP dependence plots reveal non-linear relationships")
print("   - Unexpected patterns → data quality issues atau spurious correlations")
print("   - Interaction effects → consider feature crosses")

print("\n5. BUSINESS VALUE:")
print("   - Transparent predictions → user trust")
print("   - Actionable feedback (\"Increase education level to improve loan approval\")")
print("   - Regulatory compliance (GDPR right to explanation)")

# ============================================================================
# BAGIAN 14: SAVE RESULTS
# ============================================================================

print("\n[14] SAVING RESULTS...")

# Save comparison dataframe
comparison.to_csv('/root/data-science-cookbook/techniques/shap-explainable-ai/feature_importance_comparison.csv', index=False)
print("✓ Saved: feature_importance_comparison.csv")

# Save top SHAP values for first 20 instances
shap_df = pd.DataFrame(
    shap_values[:20],
    columns=X_test_sample.columns
)
shap_df['prediction'] = y_pred_proba[:20]
shap_df['actual'] = y_test[:20] if not hasattr(y_test, 'iloc') else y_test.iloc[:20].values
shap_df.to_csv('/root/data-science-cookbook/techniques/shap-explainable-ai/shap_values_sample.csv', index=False)
print("✓ Saved: shap_values_sample.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TUTORIAL COMPLETED SUCCESSFULLY!")
print("="*80)

print("\nGenerated Files:")
print("  1. shap_feature_importance.png — Global feature importance (bar)")
print("  2. shap_summary_beeswarm.png — Impact distribution (beeswarm)")
print("  3. shap_waterfall_high_income.png — Local explanation (high income)")
print("  4. shap_waterfall_low_income.png — Local explanation (low income)")
print("  5. shap_force_high_income.png — Force plot (high income)")
print("  6. shap_dependence_*.png — Feature relationships")
print("  7. feature_importance_comparison.csv — SHAP vs XGBoost comparison")
print("  8. shap_values_sample.csv — SHAP values untuk 20 instances")

print("\nKey Learnings:")
print("  ✓ SHAP provides theoretically sound feature attributions (Shapley values)")
print("  ✓ TreeExplainer gives exact SHAP values efficiently for tree models")
print("  ✓ Global explanations (summary plot) + local explanations (waterfall)")
print("  ✓ SHAP reveals feature interactions via dependence plots")
print("  ✓ Critical tool untuk explainable AI, debugging, fairness audits")

print("\nNext Steps:")
print("  1. Experiment dengan different models (RandomForest, LightGBM)")
print("  2. Try KernelExplainer untuk neural networks")
print("  3. Analyze SHAP interaction values untuk pairwise effects")
print("  4. Integrate SHAP ke production ML pipeline")
print("  5. Read original paper: Lundberg & Lee (2017) NIPS")

print("\n" + "="*80)
print("Happy explaining! 🎯")
print("="*80)
