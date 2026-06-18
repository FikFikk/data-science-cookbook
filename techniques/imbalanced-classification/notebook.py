"""
Imbalanced Classification: Complete Implementation
==================================================

Tutorial lengkap untuk menangani dataset tidak seimbang dengan berbagai teknik
resampling, cost-sensitive learning, dan evaluation metrics.

Dataset: Credit Card Fraud Detection (highly imbalanced)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("IMBALANCED CLASSIFICATION: COMPLETE TUTORIAL")
print("=" * 80)

# ============================================================================
# PART 1: GENERATE SYNTHETIC IMBALANCED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: DATA GENERATION")
print("=" * 80)

# Generate imbalanced dataset
# Simulasi credit card fraud: 98% normal, 2% fraud
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.98, 0.02],
    flip_y=0.01,
    random_state=42
)

print(f"\n[DATA] Dataset Info:")
print(f"   Total samples: {len(y)}")
print(f"   Features: {X.shape[1]}")
print(f"   Class 0 (Normal): {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"   Class 1 (Fraud): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
print(f"   Imbalance Ratio: {np.sum(y==0)/np.sum(y==1):.1f}:1")

# ============================================================================
# PART 2: TRAIN-TEST SPLIT (STRATIFIED!)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: STRATIFIED TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n[OK] Train set:")
print(f"   Total: {len(y_train)}")
print(f"   Class 0: {np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
print(f"   Class 1: {np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")

print(f"\n[OK] Test set:")
print(f"   Total: {len(y_test)}")
print(f"   Class 0: {np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)")
print(f"   Class 1: {np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PART 3: BASELINE MODEL (NO TREATMENT)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: BASELINE MODEL (Tanpa Penanganan Imbalance)")
print("=" * 80)

baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

print("\n[BASELINE] Results:")
print("\nConfusion Matrix:")
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print(cm_baseline)
print(f"\n   TN: {cm_baseline[0,0]}, FP: {cm_baseline[0,1]}")
print(f"   FN: {cm_baseline[1,0]}, TP: {cm_baseline[1,1]}")

print("\nMetrics:")
print(f"   Accuracy: {baseline_model.score(X_test_scaled, y_test):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"   Recall: {recall_score(y_test, y_pred_baseline):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_baseline):.4f}")
print(f"   ROC-AUC: {auc(*roc_curve(y_test, y_proba_baseline)[:2][::-1]):.4f}")
print(f"   PR-AUC: {average_precision_score(y_test, y_proba_baseline):.4f}")
print(f"   MCC: {matthews_corrcoef(y_test, y_pred_baseline):.4f}")

print("\n[WARNING] Accuracy tinggi tapi Recall rendah!")
print("   Model tidak belajar mendeteksi fraud dengan baik.")

# ============================================================================
# PART 4: TECHNIQUE 1 - RANDOM OVERSAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: RANDOM OVERSAMPLING")
print("=" * 80)

def random_oversample(X, y, ratio=1.0):
    """Random oversampling: duplikasi minority class"""
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]
    
    n_majority = len(y_majority)
    n_target = int(n_majority * ratio)
    
    indices = np.random.choice(len(y_minority), n_target, replace=True)
    X_minority_oversampled = X_minority[indices]
    y_minority_oversampled = y_minority[indices]
    
    X_resampled = np.vstack([X_majority, X_minority_oversampled])
    y_resampled = np.hstack([y_majority, y_minority_oversampled])
    
    return X_resampled, y_resampled

X_train_ros, y_train_ros = random_oversample(X_train_scaled, y_train, ratio=0.5)

print(f"\n[RESAMPLE] After Random Oversampling (ratio=0.5):")
print(f"   Total: {len(y_train_ros)}")
print(f"   Class 0: {np.sum(y_train_ros==0)} ({np.sum(y_train_ros==0)/len(y_train_ros)*100:.1f}%)")
print(f"   Class 1: {np.sum(y_train_ros==1)} ({np.sum(y_train_ros==1)/len(y_train_ros)*100:.1f}%)")

model_ros = LogisticRegression(random_state=42, max_iter=1000)
model_ros.fit(X_train_ros, y_train_ros)
y_pred_ros = model_ros.predict(X_test_scaled)
y_proba_ros = model_ros.predict_proba(X_test_scaled)[:, 1]

print("\n[RESULTS] Random Oversampling:")
cm_ros = confusion_matrix(y_test, y_pred_ros)
print("\nConfusion Matrix:")
print(cm_ros)

print("\nMetrics:")
print(f"   Precision: {precision_score(y_test, y_pred_ros):.4f}")
print(f"   Recall: {recall_score(y_test, y_pred_ros):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_ros):.4f}")
print(f"   PR-AUC: {average_precision_score(y_test, y_proba_ros):.4f}")

print("\n[OK] Recall meningkat! Model lebih sensitif terhadap fraud.")

# ============================================================================
# PART 5: TECHNIQUE 2 - SMOTE
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: SMOTE (Synthetic Minority Over-sampling Technique)")
print("=" * 80)

def smote(X, y, k=5, ratio=1.0):
    """SMOTE: Generate synthetic samples dengan k-nearest neighbors"""
    from sklearn.neighbors import NearestNeighbors
    
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]
    
    n_majority = len(y_majority)
    n_minority = len(y_minority)
    n_synthetic = int(n_majority * ratio) - n_minority
    
    if n_synthetic <= 0:
        return X, y
    
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_minority)
    
    synthetic_samples = []
    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_minority)
        sample = X_minority[idx]
        neighbors_indices = nn.kneighbors([sample], return_distance=False)[0][1:]
        neighbor_idx = np.random.choice(neighbors_indices)
        neighbor = X_minority[neighbor_idx]
        lambda_val = np.random.random()
        synthetic = sample + lambda_val * (neighbor - sample)
        synthetic_samples.append(synthetic)
    
    synthetic_samples = np.array(synthetic_samples)
    X_resampled = np.vstack([X_majority, X_minority, synthetic_samples])
    y_resampled = np.hstack([y_majority, y_minority, np.ones(len(synthetic_samples))])
    
    return X_resampled, y_resampled

X_train_smote, y_train_smote = smote(X_train_scaled, y_train, k=5, ratio=0.5)

print(f"\n[RESAMPLE] After SMOTE (ratio=0.5, k=5):")
print(f"   Total: {len(y_train_smote)}")
print(f"   Class 0: {np.sum(y_train_smote==0)} ({np.sum(y_train_smote==0)/len(y_train_smote)*100:.1f}%)")
print(f"   Class 1: {np.sum(y_train_smote==1)} ({np.sum(y_train_smote==1)/len(y_train_smote)*100:.1f}%)")

model_smote = LogisticRegression(random_state=42, max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test_scaled)
y_proba_smote = model_smote.predict_proba(X_test_scaled)[:, 1]

print("\n[RESULTS] SMOTE:")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print("\nConfusion Matrix:")
print(cm_smote)

print("\nMetrics:")
print(f"   Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"   Recall: {recall_score(y_test, y_pred_smote):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_smote):.4f}")
print(f"   PR-AUC: {average_precision_score(y_test, y_proba_smote):.4f}")

print("\n[OK] SMOTE menghasilkan synthetic samples yang lebih diverse.")

# ============================================================================
# PART 6: TECHNIQUE 3 - CLASS WEIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: CLASS WEIGHTS (Cost-Sensitive Learning)")
print("=" * 80)

model_weighted = LogisticRegression(
    class_weight='balanced', 
    random_state=42, 
    max_iter=1000
)
model_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = model_weighted.predict(X_test_scaled)
y_proba_weighted = model_weighted.predict_proba(X_test_scaled)[:, 1]

print("\n[RESULTS] Class Weights:")
cm_weighted = confusion_matrix(y_test, y_pred_weighted)
print("\nConfusion Matrix:")
print(cm_weighted)

print("\nMetrics:")
print(f"   Precision: {precision_score(y_test, y_pred_weighted):.4f}")
print(f"   Recall: {recall_score(y_test, y_pred_weighted):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_weighted):.4f}")
print(f"   PR-AUC: {average_precision_score(y_test, y_proba_weighted):.4f}")

print("\n[OK] Class weights: Simple dan efektif tanpa resampling!")

# ============================================================================
# PART 7: TECHNIQUE 4 - RANDOM FOREST WITH BALANCED SAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: RANDOM FOREST WITH BALANCED SAMPLING")
print("=" * 80)

model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
y_proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]

print("\n[RESULTS] Random Forest (Balanced):")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(cm_rf)

print("\nMetrics:")
print(f"   Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"   Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"   PR-AUC: {average_precision_score(y_test, y_proba_rf):.4f}")
print(f"   MCC: {matthews_corrcoef(y_test, y_pred_rf):.4f}")

print("\n[OK] Random Forest: Powerful ensemble untuk imbalanced data!")

# ============================================================================
# PART 8: THRESHOLD TUNING
# ============================================================================
print("\n" + "=" * 80)
print("PART 8: THRESHOLD TUNING")
print("=" * 80)

print("\n[INFO] Mencari threshold optimal untuk maximize F1-score...")

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []
recall_scores = []
precision_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_proba_rf >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n[OPTIMAL] Threshold: {optimal_threshold:.2f}")
print(f"   F1-Score: {f1_scores[optimal_idx]:.4f}")
print(f"   Recall: {recall_scores[optimal_idx]:.4f}")
print(f"   Precision: {precision_scores[optimal_idx]:.4f}")

y_pred_optimal = (y_proba_rf >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

print("\nConfusion Matrix (Optimal Threshold):")
print(cm_optimal)
print(f"   TN: {cm_optimal[0,0]}, FP: {cm_optimal[0,1]}")
print(f"   FN: {cm_optimal[1,0]}, TP: {cm_optimal[1,1]}")

# ============================================================================
# PART 9: COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PART 9: COMPARISON SUMMARY")
print("=" * 80)

comparison_data = {
    'Method': [
        'Baseline (No Treatment)',
        'Random Oversampling',
        'SMOTE',
        'Class Weights',
        'Random Forest (Balanced)',
        'RF + Optimal Threshold'
    ],
    'Precision': [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_ros),
        precision_score(y_test, y_pred_smote),
        precision_score(y_test, y_pred_weighted),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_optimal)
    ],
    'Recall': [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_ros),
        recall_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_weighted),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_optimal)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_ros),
        f1_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_weighted),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_optimal)
    ],
    'MCC': [
        matthews_corrcoef(y_test, y_pred_baseline),
        matthews_corrcoef(y_test, y_pred_ros),
        matthews_corrcoef(y_test, y_pred_smote),
        matthews_corrcoef(y_test, y_pred_weighted),
        matthews_corrcoef(y_test, y_pred_rf),
        matthews_corrcoef(y_test, y_pred_optimal)
    ]
}

df_comparison = pd.DataFrame(comparison_data)

print("\n[SUMMARY] Performance Comparison:")
print(df_comparison.to_string(index=False))

best_f1_idx = df_comparison['F1-Score'].idxmax()
best_method = df_comparison.loc[best_f1_idx, 'Method']
best_f1 = df_comparison.loc[best_f1_idx, 'F1-Score']

print(f"\n[BEST] Method: {best_method}")
print(f"   F1-Score: {best_f1:.4f}")

# ============================================================================
# PART 10: KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 80)
print("PART 10: KEY TAKEAWAYS")
print("=" * 80)

print("""
1. BASELINE TRAP: Accuracy tinggi (98%) misleading! Recall rendah = gagal detect fraud.

2. RESAMPLING WORKS:
   - Random Oversampling: Simple, tapi risk overfitting
   - SMOTE: Lebih robust, generate synthetic samples di feature space
   - Undersampling: Cepat tapi loss informasi

3. COST-SENSITIVE: Class weights simple dan efektif tanpa ubah data.

4. ENSEMBLE POWER: Random Forest + balanced sampling = kombinasi powerful.

5. THRESHOLD TUNING: Jangan stuck di 0.5! Optimize threshold sesuai business need.

6. METRICS MATTER: 
   - Precision: Minimize false alarms
   - Recall: Catch all fraud (safety-critical)
   - F1: Balance keduanya
   - MCC: Robust untuk imbalanced data

7. NO SILVER BULLET: Experiment berbagai teknik, pilih yang terbaik untuk use case.

8. BUSINESS CONTEXT: Cost of FP vs FN determine metric priority.
   - Medical: Maximize Recall (jangan miss penyakit)
   - Fraud: Balance Precision-Recall (limited investigation resource)
   - Marketing: Maximize Precision (fokus high-quality leads)

9. ALWAYS STRATIFY: Train-test split dan cross-validation harus stratified.

10. TEST DISTRIBUTION: Jangan resample test set! Test harus reflect real-world.
""")

print("=" * 80)
print("TUTORIAL SELESAI!")
print("=" * 80)
print("\nLihat README.md untuk teori lengkap dan mathematical intuition.")
print("Experiment dengan parameter berbeda untuk deeper understanding!")
