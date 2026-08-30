"""
Bayesian Optimization: Implementasi Lengkap
============================================

Tutorial praktis Bayesian Optimization untuk hyperparameter tuning.

CATATAN: Script ini memerlukan dependencies berikut:
  pip install --break-system-packages scikit-optimize scikit-learn matplotlib numpy pandas

Untuk menjalankan:
  python3 notebook.py

Author: Hermes Agent
Date: 22 Juni 2026
"""

import sys

print("="*70)
print("BAYESIAN OPTIMIZATION TUTORIAL")
print("="*70)
print("\nMengecek dependencies...")

# Check dependencies
required_packages = {
    'numpy': 'numpy',
    'sklearn': 'scikit-learn', 
    'matplotlib': 'matplotlib',
    'skopt': 'scikit-optimize'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - BELUM TERINSTALL")
        missing_packages.append(package)

if missing_packages:
    print("\n" + "="*70)
    print("DEPENDENCIES BELUM LENGKAP")
    print("="*70)
    print("\nInstall dependencies dengan command berikut:")
    print(f"\npip install --break-system-packages {' '.join(missing_packages)}")
    print("\nSetelah install, jalankan ulang script ini:")
    print("python3 notebook.py")
    print("="*70)
    sys.exit(1)

# Import semua dependencies
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skopt import gp_minimize, BayesSearchCV, dummy_minimize, dump, load
from skopt.plots import plot_convergence, plot_gaussian_process
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

print("\n✓ Semua dependencies tersedia!\n")

# ============================================================================
# PART 1: Optimasi Fungsi 1D - Visualisasi Konsep
# ============================================================================

print("="*70)
print("PART 1: Optimasi Fungsi 1D - Visualisasi Konsep")
print("="*70)

# Definisikan black-box function
def objective_1d(x):
    """Fungsi non-convex dengan multiple local minima"""
    return (x[0] ** 2) - 10 * np.cos(2 * np.pi * x[0]) + 10

# Search space
space_1d = [(-5.0, 5.0)]

# Jalankan Bayesian Optimization
print("\nMenjalankan Bayesian Optimization untuk fungsi 1D...")
result_1d = gp_minimize(
    objective_1d,
    space_1d,
    n_calls=20,
    random_state=42,
    acq_func='EI',  # Expected Improvement
    n_initial_points=5
)

print(f"✓ Optimum ditemukan di x = {result_1d.x[0]:.4f}")
print(f"✓ Nilai fungsi = {result_1d.fun:.4f}")
print(f"✓ Jumlah evaluasi = {len(result_1d.func_vals)}")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fungsi dan titik evaluasi
x_plot = np.linspace(-5, 5, 200)
y_plot = [objective_1d([x]) for x in x_plot]

axes[0].plot(x_plot, y_plot, 'b-', label='True Function', linewidth=2)
axes[0].plot(result_1d.x_iters, result_1d.func_vals, 'ro', 
             label='Evaluated Points', markersize=8, alpha=0.6)
axes[0].plot(result_1d.x[0], result_1d.fun, 'g*', 
             label='Best Point', markersize=20)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('f(x)', fontsize=12)
axes[0].set_title('Function Optimization', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Convergence
convergence_trace = np.minimum.accumulate(result_1d.func_vals)
axes[1].plot(convergence_trace, 'b-', linewidth=2)
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Best Value Found', fontsize=12)
axes[1].set_title('Convergence Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1d_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Visualisasi disimpan ke: 1d_optimization.png")
plt.close()

# ============================================================================
# PART 2: Hyperparameter Tuning - Random Forest
# ============================================================================

print("\n" + "="*70)
print("PART 2: Hyperparameter Tuning - Random Forest")
print("="*70)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDataset: Breast Cancer Wisconsin")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

print("\n--- Baseline: Random Forest dengan default parameters ---")
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train_scaled, y_train)
baseline_score = rf_baseline.score(X_test_scaled, y_test)
print(f"Baseline Test Accuracy: {baseline_score:.4f}")

print("\n--- Bayesian Optimization untuk Random Forest ---")

# Define search space
search_space_rf = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.3, 1.0)
}

# Bayesian Optimization
print("Menjalankan Bayesian Optimization (50 iterations)...")
opt_rf = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_space_rf,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

opt_rf.fit(X_train_scaled, y_train)

print("\n✓ Optimization selesai!")
print("\nBest Parameters:")
for param, value in opt_rf.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score: {opt_rf.best_score_:.4f}")

# Test set evaluation
test_score = opt_rf.score(X_test_scaled, y_test)
print(f"Test Accuracy (optimized): {test_score:.4f}")
improvement = test_score - baseline_score
print(f"Improvement: {improvement:.4f} ({improvement/baseline_score*100:.2f}%)")

# Detailed classification report
y_pred = opt_rf.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Visualisasi convergence
cv_results = opt_rf.cv_results_
scores = -cv_results['mean_test_score']
best_scores = np.minimum.accumulate(scores)

plt.figure(figsize=(10, 6))
plt.plot(scores, 'o-', alpha=0.6, label='CV Score', linewidth=2)
plt.plot(best_scores, 'r-', linewidth=2, label='Best Score')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('Random Forest Bayesian Optimization Convergence', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_convergence.png', dpi=150, bbox_inches='tight')
print("\n✓ Convergence plot disimpan ke: rf_convergence.png")
plt.close()

# ============================================================================
# PART 3: Custom Objective Function - Gradient Boosting
# ============================================================================

print("\n" + "="*70)
print("PART 3: Custom Objective Function - Gradient Boosting")
print("="*70)

# Define search space
space_gb = [
    Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
    Integer(3, 10, name='max_depth'),
    Integer(50, 300, name='n_estimators'),
    Real(0.5, 1.0, name='subsample')
]

# Custom objective function
@use_named_args(space_gb)
def objective_gb(**params):
    """
    Objective function untuk Gradient Boosting.
    Return negative accuracy karena gp_minimize mencari minimum.
    """
    model = GradientBoostingClassifier(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        random_state=42
    )
    
    # 3-fold CV untuk speed
    score = cross_val_score(model, X_train_scaled, y_train, 
                            cv=3, n_jobs=-1, scoring='accuracy').mean()
    return -score

print("\nMenjalankan Bayesian Optimization untuk Gradient Boosting...")
print("(40 iterations, ini mungkin memakan waktu beberapa menit)\n")

result_gb = gp_minimize(
    objective_gb,
    space_gb,
    n_calls=40,
    n_initial_points=10,
    random_state=42,
    verbose=False
)

print("✓ Optimization selesai!")
print("\nBest Parameters:")
param_names = ['learning_rate', 'max_depth', 'n_estimators', 'subsample']
best_params_gb = dict(zip(param_names, result_gb.x))
for param, value in best_params_gb.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Accuracy: {-result_gb.fun:.4f}")

# Train final model
final_model_gb = GradientBoostingClassifier(
    learning_rate=best_params_gb['learning_rate'],
    max_depth=int(best_params_gb['max_depth']),
    n_estimators=int(best_params_gb['n_estimators']),
    subsample=best_params_gb['subsample'],
    random_state=42
)

final_model_gb.fit(X_train_scaled, y_train)
test_score_gb = final_model_gb.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_score_gb:.4f}")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Convergence
convergence = np.minimum.accumulate(-np.array(result_gb.func_vals))
axes[0].plot(convergence, 'b-', linewidth=2)
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Best Accuracy', fontsize=12)
axes[0].set_title('Gradient Boosting Optimization', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Feature importance
feature_importance = final_model_gb.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
top_features = [data.feature_names[i] for i in top_features_idx]
top_importance = feature_importance[top_features_idx]

axes[1].barh(range(len(top_features)), top_importance, color='steelblue')
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features, fontsize=10)
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('gb_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Results plot disimpan ke: gb_results.png")
plt.close()

# ============================================================================
# PART 4: Comparison - BO vs Random Search
# ============================================================================

print("\n" + "="*70)
print("PART 4: Perbandingan Bayesian Optimization vs Random Search")
print("="*70)

# Synthetic dataset yang lebih challenging
X_synth, y_synth = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_synth, y_synth, test_size=0.2, random_state=42
)

# Simplified search space
space_comp = [
    Integer(50, 200, name='n_estimators'),
    Integer(3, 15, name='max_depth'),
    Real(0.3, 1.0, name='max_features')
]

@use_named_args(space_comp)
def objective_comp(**params):
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        random_state=42,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train_s, y_train_s, cv=3, n_jobs=-1).mean()
    return -score

print("\nMenjalankan perbandingan (30 iterations each)...\n")

# Bayesian Optimization
print("1. Bayesian Optimization...")
result_bo = gp_minimize(
    objective_comp,
    space_comp,
    n_calls=30,
    n_initial_points=5,
    random_state=42,
    verbose=False
)

# Random Search
print("2. Random Search...")
result_rs = dummy_minimize(
    objective_comp,
    space_comp,
    n_calls=30,
    random_state=42
)

print("\n" + "="*50)
print("HASIL PERBANDINGAN")
print("="*50)
print(f"Bayesian Optimization - Best Score: {-result_bo.fun:.4f}")
print(f"Random Search - Best Score: {-result_rs.fun:.4f}")
improvement_pct = ((-result_bo.fun + result_rs.fun) / -result_rs.fun) * 100
print(f"Improvement: {(-result_bo.fun + result_rs.fun):.4f} ({improvement_pct:.2f}%)")
print("="*50)

# Visualisasi perbandingan
plt.figure(figsize=(12, 6))

bo_convergence = np.minimum.accumulate(-np.array(result_bo.func_vals))
rs_convergence = np.minimum.accumulate(-np.array(result_rs.func_vals))

plt.plot(bo_convergence, 'b-', linewidth=2, label='Bayesian Optimization', marker='o')
plt.plot(rs_convergence, 'r--', linewidth=2, label='Random Search', marker='s')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Best Accuracy', fontsize=12)
plt.title('Bayesian Optimization vs Random Search', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Comparison plot disimpan ke: comparison.png")
plt.close()

# ============================================================================
# PART 5: Saving and Resuming
# ============================================================================

print("\n" + "="*70)
print("PART 5: Saving and Resuming Optimization")
print("="*70)

# Save result
dump(result_gb, 'optimization_checkpoint.pkl')
print("\n✓ Optimization result disimpan ke: optimization_checkpoint.pkl")

# Load
loaded_result = load('optimization_checkpoint.pkl')
print(f"✓ Loaded result: Best score = {-loaded_result.fun:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Bayesian Optimization telah berhasil didemonstrasikan untuk:

1. ✓ Optimasi fungsi 1D dengan visualisasi GP
2. ✓ Hyperparameter tuning Random Forest (50 iterations)
3. ✓ Custom objective function untuk Gradient Boosting
4. ✓ Perbandingan dengan Random Search
5. ✓ Saving dan loading optimization results

KEY TAKEAWAYS:
- Bayesian Optimization jauh lebih sample-efficient dibanding Random Search
- Untuk hyperparameter tuning dengan budget terbatas, BO adalah pilihan terbaik
- Expected Improvement (EI) adalah acquisition function yang robust
- Selalu validasi dengan cross-validation, bukan test set
- Monitor convergence untuk tahu kapan berhenti

File yang dihasilkan:
- 1d_optimization.png: Visualisasi optimasi fungsi 1D
- rf_convergence.png: Convergence Random Forest tuning
- gb_results.png: Hasil Gradient Boosting + feature importance
- comparison.png: BO vs Random Search
- optimization_checkpoint.pkl: Saved optimization result
""")

print("="*70)
print("Tutorial selesai! 🎉")
print("="*70)
