"""
Active Learning Simulation: Uncertainty Sampling vs Committee vs Random Sampling
=============================================================================

Tutorial implementasi complete Active Learning menggunakan python dan framework modAL.
Kita menyimulasikan budget pelabelan data yang terbatas secara interaktif.

Dependencies:
    pip install numpy pandas matplotlib seaborn scikit-learn modAL-python
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import sklearn.utils
import sklearn.utils.validation
# Patch check_X_y to handle force_all_finite argument which was deprecated/removed in scikit-learn >= 1.6
orig_check_X_y = sklearn.utils.validation.check_X_y
def patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return orig_check_X_y(*args, **kwargs)
sklearn.utils.validation.check_X_y = patched_check_X_y
sklearn.utils.check_X_y = patched_check_X_y

# Framework modAL untuk Active Learning
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling, margin_sampling
from modAL.disagreement import vote_entropy_sampling

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("MUTUAL ACTIVE LEARNING COMPARISON SIMULATOR")
print("="*80)

# Setup path output
OUTPUT_DIR = "/root/data-science-cookbook/techniques/active-learning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. GENERATE DATASET & PREPROCESSING
# ----------------------------------------------------------------------------
print("\n[1] Menyiapkan dataset sintetis dengan noise...")
X_raw, y_raw = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=8,
    n_redundant=4,
    n_classes=2,
    weights=[0.55, 0.45],
    class_sep=0.85,
    random_state=42
)

# Standardisasi
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw)

# Pisahkan hold-out test set (200 sampel) dan unlabeled pool (1000 sampel)
n_test = 200
X_test, y_test = X_raw[:n_test], y_raw[:n_test]
X_pool_orig, y_pool_orig = X_raw[n_test:], y_raw[n_test:]

print(f"    Ukuran Pool Awal  : {len(X_pool_orig)} sampel")
print(f"    Ukuran Data Uji   : {len(X_test)} sampel")

# Inisialisasi label set kecil (15 sampel pertama)
n_initial = 15
np.random.seed(42)
initial_idx = np.random.choice(range(len(X_pool_orig)), size=n_initial, replace=False)

X_initial = X_pool_orig[initial_idx]
y_initial = y_pool_orig[initial_idx]

# Pool tanpa data inisial awal
X_pool_clean = np.delete(X_pool_orig, initial_idx, axis=0)
y_pool_clean = np.delete(y_pool_orig, initial_idx)

# Parameter Simulasi Active Learning
N_QUERIES = 40  # Jumlah langkah anotasi manual
BATCH_SIZE = 5  # Setiap langkah, 5 sampel dilabeli oleh evaluator

# ----------------------------------------------------------------------------
# 2. DEFINISI STRATEGI QUERY & MODEL
# ----------------------------------------------------------------------------
# Base classifier yang digunakan: Random Forest
base_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

# Kita akan membandingkan 4 strategi:
# A. Random Sampling (Passive Learning)
# B. Entropy Sampling (Uncertainty Active Learning)
# C. Margin Sampling (Uncertainty Active Learning)
# D. Query By Committee (QBC) dengan Ensemble RandomForest

# ----------------------------------------------------------------------------
# SIMULASI A: RANDOM SAMPLING (PASSIVE LEARNING)
# ----------------------------------------------------------------------------
print("\n[2] Menjalankan simulasi Passive Learning (Random Sampling)...")
X_train_rand = deepcopy(X_initial)
y_train_rand = deepcopy(y_initial)
X_pool_rand = deepcopy(X_pool_clean)
y_pool_rand = deepcopy(y_pool_clean)

# Evaluasi baseline awal
clf_rand = deepcopy(base_clf)
clf_rand.fit(X_train_rand, y_train_rand)
history_random = [accuracy_score(y_test, clf_rand.predict(X_test))]

for i in range(N_QUERIES):
    # Pilih secara acak dari pool rand
    query_idx = np.random.choice(range(len(X_pool_rand)), size=BATCH_SIZE, replace=False)
    
    # Kueri data baru
    X_new = X_pool_rand[query_idx]
    y_new = y_pool_rand[query_idx]
    
    # Gabungkan ke dalam data latih
    X_train_rand = np.vstack([X_train_rand, X_new])
    y_train_rand = np.hstack([y_train_rand, y_new])
    
    # Hapus dari pool rand
    X_pool_rand = np.delete(X_pool_rand, query_idx, axis=0)
    y_pool_rand = np.delete(y_pool_rand, query_idx)
    
    # Latih ulang model
    clf_rand = deepcopy(base_clf)
    clf_rand.fit(X_train_rand, y_train_rand)
    
    # Catat performa
    acc = accuracy_score(y_test, clf_rand.predict(X_test))
    history_random.append(acc)

# ----------------------------------------------------------------------------
# SIMULASI B: ENTROPY SAMPLING (ACTIVE LEARNING)
# ----------------------------------------------------------------------------
print("[3] Menjalankan simulasi Active Learning - Entropy Sampling...")
X_train_ent = deepcopy(X_initial)
y_train_ent = deepcopy(y_initial)
X_pool_ent = deepcopy(X_pool_clean)
y_pool_ent = deepcopy(y_pool_clean)

# Inisialisasi ActiveLearner
learner_ent = ActiveLearner(
    estimator=deepcopy(base_clf),
    query_strategy=entropy_sampling,
    X_training=X_train_ent,
    y_training=y_train_ent
)
history_entropy = [accuracy_score(y_test, learner_ent.predict(X_test))]

for i in range(N_QUERIES):
    # Pilih sampel berdasarkan ketidakpastian entropy tertinggi
    query_idx, query_inst = learner_ent.query(X_pool_ent, n_instances=BATCH_SIZE)
    
    # Input label baru dari Oracle
    y_new = y_pool_ent[query_idx]
    
    # Ajarkan model sample baru tersebut
    learner_ent.teach(X_pool_ent[query_idx], y_new)
    
    # Hapus dari pool
    X_pool_ent = np.delete(X_pool_ent, query_idx, axis=0)
    y_pool_ent = np.delete(y_pool_ent, query_idx)
    
    # Catat performa
    acc = accuracy_score(y_test, learner_ent.predict(X_test))
    history_entropy.append(acc)

# ----------------------------------------------------------------------------
# SIMULASI C: MARGIN SAMPLING (ACTIVE LEARNING)
# ----------------------------------------------------------------------------
print("[4] Menjalankan simulasi Active Learning - Margin Sampling...")
X_train_marg = deepcopy(X_initial)
y_train_marg = deepcopy(y_initial)
X_pool_marg = deepcopy(X_pool_clean)
y_pool_marg = deepcopy(y_pool_clean)

learner_marg = ActiveLearner(
    estimator=deepcopy(base_clf),
    query_strategy=margin_sampling,
    X_training=X_train_marg,
    y_training=y_train_marg
)
history_margin = [accuracy_score(y_test, learner_marg.predict(X_test))]

for i in range(N_QUERIES):
    # Pilih sampel berdasarkan margin kelas terkecil
    query_idx, query_inst = learner_marg.query(X_pool_marg, n_instances=BATCH_SIZE)
    
    y_new = y_pool_marg[query_idx]
    learner_marg.teach(X_pool_marg[query_idx], y_new)
    
    X_pool_marg = np.delete(X_pool_marg, query_idx, axis=0)
    y_pool_marg = np.delete(y_pool_marg, query_idx)
    
    acc = accuracy_score(y_test, learner_marg.predict(X_test))
    history_margin.append(acc)

# ----------------------------------------------------------------------------
# SIMULASI D: QUERY-BY-COMMITTEE (QBC)
# ----------------------------------------------------------------------------
print("[5] Menjalankan simulasi Active Learning - Query-By-Committee...")
X_train_qbc = deepcopy(X_initial)
y_train_qbc = deepcopy(y_initial)
X_pool_qbc = deepcopy(X_pool_clean)
y_pool_qbc = deepcopy(y_pool_clean)

# Anggota komite: 3 random forest classifier dengan random state berbeda
learner_1 = ActiveLearner(estimator=RandomForestClassifier(n_estimators=30, max_depth=5, random_state=1), X_training=X_train_qbc, y_training=y_train_qbc)
learner_2 = ActiveLearner(estimator=RandomForestClassifier(n_estimators=30, max_depth=5, random_state=2), X_training=X_train_qbc, y_training=y_train_qbc)
learner_3 = ActiveLearner(estimator=RandomForestClassifier(n_estimators=30, max_depth=5, random_state=3), X_training=X_train_qbc, y_training=y_train_qbc)

committee = Committee(
    learner_list=[learner_1, learner_2, learner_3],
    query_strategy=vote_entropy_sampling
)
history_qbc = [accuracy_score(y_test, committee.predict(X_test))]

for i in range(N_QUERIES):
    # Pilih sampel berdasarkan perbedaan suara komite terbesar
    query_idx, query_inst = committee.query(X_pool_qbc, n_instances=BATCH_SIZE)
    
    y_new = y_pool_qbc[query_idx]
    committee.teach(X_pool_qbc[query_idx], y_new)
    
    X_pool_qbc = np.delete(X_pool_qbc, query_idx, axis=0)
    y_pool_qbc = np.delete(y_pool_qbc, query_idx)
    
    acc = accuracy_score(y_test, committee.predict(X_test))
    history_qbc.append(acc)

# ----------------------------------------------------------------------------
# 3. ANALISIS HASIL & VISUALISASI
# ----------------------------------------------------------------------------
print("\n[6] Menganalisis dan mengeplot hasil pembelajaran...")

# Buat list budget data berlabel di setiap langkah
x_axis = [n_initial + i * BATCH_SIZE for i in range(N_QUERIES + 1)]

plt.figure(figsize=(10, 6))
plt.plot(x_axis, history_random, label="Passive Learning (Random Sampling)", color="gray", linestyle="--", marker="o", markersize=4)
plt.plot(x_axis, history_entropy, label="Active Learning (Entropy)", color="blue", marker="s", markersize=4)
plt.plot(x_axis, history_margin, label="Active Learning (Margin)", color="green", marker="^", markersize=4)
plt.plot(x_axis, history_qbc, label="Active Learning (Committee/QBC)", color="purple", marker="d", markersize=4)

plt.title("Komparasi Efisiensi Active Learning vs Passive Learning", fontsize=14, fontweight="bold")
plt.xlabel("Jumlah Data Berlabel", fontsize=12)
plt.ylabel("Akurasi Model", fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)

# Simpan visualisasi plot komparasi
plot_path = os.path.join(OUTPUT_DIR, "active_learning_curve.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=180)
plt.close()
print(f"    ✓ Plot evaluasi tersimpan di: {plot_path}")

# Tampilkan ringkasan metrik akhir ke CSV/DataFrame pendukung
milestones = [0, 10, 20, 30, 40]
print("\nRingkasan Kenaikan Akurasi di beberapa Milestones:")
print(f"{'Data Berlabel':<15} | {'Random Acc':<12} | {'Entropy Acc':<12} | {'Margin Acc':<12} | {'QBC Acc':<12}")
print("-" * 75)

for m in milestones:
    lbl_count = n_initial + m * BATCH_SIZE
    print(f"{lbl_count:<15} | {history_random[m]:.4f}     | {history_entropy[m]:.4f}      | {history_margin[m]:.4f}     | {history_qbc[m]:.4f}")

# Simpan ringkasan tabel data
df_out = pd.DataFrame({
    "Labels_Count": x_axis,
    "Passive_Random": history_random,
    "Active_Entropy": history_entropy,
    "Active_Margin": history_margin,
    "Active_Committee": history_qbc
})
csv_path = os.path.join(OUTPUT_DIR, "active_learning_metrics.csv")
df_out.to_csv(csv_path, index=False)
print(f"\n    ✓ Data metrik CSV tersimpan di: {csv_path}")

# Analisis Laporan Klasifikasi Akhir untuk Entropy Learner
from sklearn.metrics import classification_report
print("\nLaporan Klasifikasi Akhir model Active Learning (Entropy):")
print(classification_report(y_test, learner_ent.predict(X_test)))

print("="*80)
print("SIMULASI SUKSES!")
print("="*80)
