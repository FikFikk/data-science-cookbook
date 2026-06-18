# Imbalanced Classification: Teknik Menangani Dataset Tidak Seimbang

## Ringkasan
Tutorial lengkap untuk menangani masalah klasifikasi dengan dataset tidak seimbang (imbalanced dataset) - salah satu tantangan paling umum dalam machine learning di dunia nyata seperti fraud detection, medical diagnosis, dan anomaly detection.

## Masalah Imbalanced Dataset

### Apa itu Imbalanced Dataset?
Dataset tidak seimbang terjadi ketika distribusi kelas target sangat tidak merata. Contoh:
- **Fraud Detection**: 99.5% transaksi normal, 0.5% fraud
- **Medical Diagnosis**: 95% pasien sehat, 5% sakit
- **Churn Prediction**: 85% customer bertahan, 15% churn
- **Spam Detection**: 90% email normal, 10% spam

### Mengapa Ini Masalah?
Model ML cenderung bias terhadap kelas mayoritas karena:
1. **Optimasi accuracy yang menyesatkan**: Model yang memprediksi semua kelas mayoritas bisa dapat accuracy 95% tapi tidak berguna
2. **Gradient lebih kuat dari kelas mayoritas**: Model "belajar" lebih banyak dari kelas mayoritas
3. **Decision boundary suboptimal**: Batas keputusan condong ke kelas mayoritas

Contoh kasus ekstrim:
```
Dataset: 990 kelas Normal, 10 kelas Fraud
Model bodoh yang selalu prediksi "Normal" → Accuracy = 99%!
Tapi Recall untuk Fraud = 0% (tidak mendeteksi fraud sama sekali)
```

## Teknik-Teknik Penanganan

### 1. Resampling Techniques

#### A. Oversampling (Menambah Minority Class)

**Random Oversampling**
- Duplikasi sample minority secara random
- Pros: Sederhana, cepat
- Cons: Overfitting (model hafal exact duplicates)

**SMOTE (Synthetic Minority Over-sampling Technique)**
- Generate synthetic samples dengan interpolasi k-nearest neighbors
- Cara kerja:
  1. Pilih sample minority random
  2. Cari k tetangga terdekat (default k=5)
  3. Pilih salah satu tetangga secara random
  4. Buat synthetic sample di antara keduanya: `x_new = x + λ * (x_neighbor - x)`, λ ∈ [0,1]
- Pros: Mengurangi overfitting, generalisasi lebih baik
- Cons: Bisa generate noise di boundary overlap

**ADASYN (Adaptive Synthetic Sampling)**
- Seperti SMOTE tapi fokus pada sample yang "sulit dipelajari"
- Generate lebih banyak synthetic samples di region dengan density rendah
- Pros: Lebih adaptif terhadap distribusi data
- Cons: Lebih kompleks, bisa overfit pada outliers

**Borderline-SMOTE**
- Hanya oversample minority samples yang dekat dengan decision boundary
- Identifikasi "borderline" samples: minority samples yang tetangganya mayoritas adalah kelas lawan
- Pros: Lebih efisien, fokus pada region kritis
- Cons: Butuh tuning parameter m (jumlah tetangga untuk identifikasi borderline)

#### B. Undersampling (Mengurangi Majority Class)

**Random Undersampling**
- Hapus sample mayoritas secara random
- Pros: Cepat, kurangi computational cost
- Cons: Loss informasi, bisa hapus sample penting

**Tomek Links**
- Hapus sample mayoritas yang merupakan "Tomek Link"
- Tomek Link: pasangan sample dari kelas berbeda yang saling nearest neighbor
- Pros: Bersihkan boundary, kurangi overlap
- Cons: Penghapusan minimal, masih perlu teknik lain

**Edited Nearest Neighbors (ENN)**
- Hapus sample yang misclassified oleh k-nearest neighbors
- Pros: Bersihkan noise
- Cons: Bisa hapus terlalu banyak data

**NearMiss**
- Pilih sample mayoritas yang dekat dengan minority
- Varian: NearMiss-1, NearMiss-2, NearMiss-3
- Pros: Jaga informasi penting di boundary
- Cons: Bisa kehilangan struktur global

#### C. Hybrid Methods

**SMOTE + Tomek Links**
- Oversample dengan SMOTE, lalu bersihkan dengan Tomek Links
- Pros: Balance antara generasi synthetic dan pembersihan boundary
- Cons: Lebih lambat

**SMOTE + ENN**
- Oversample dengan SMOTE, lalu bersihkan dengan ENN
- Pros: Kurangi noise dari SMOTE
- Cons: Computational cost lebih tinggi

### 2. Cost-Sensitive Learning

Berikan **penalty lebih besar** untuk misclassification pada minority class.

**Class Weights**
```python
# Sklearn otomatis hitung weight
class_weight = 'balanced'  # weight_i = n_samples / (n_classes * n_samples_i)

# Atau manual
class_weight = {0: 1, 1: 10}  # Penalty 10x untuk misclassify kelas 1
```

**Mathematical Intuition:**
Loss function standar: `L = Σ loss(y_true, y_pred)`
Cost-sensitive: `L = Σ w_i * loss(y_true_i, y_pred_i)`

Dengan class weight, model "peduli" lebih terhadap minority class.

### 3. Algorithm-Level Techniques

**Ensemble Methods yang Robust:**
- **Balanced Random Forest**: Setiap tree dilatih pada balanced bootstrap sample
- **EasyEnsemble**: Multiple balanced subsets → multiple classifiers → voting
- **BalancedBaggingClassifier**: Bootstrap dengan resampling pada setiap bag
- **RUSBoostClassifier**: Kombinasi Random Undersampling + Boosting

**Tree-based dengan class_weight:**
- Random Forest, XGBoost, LightGBM, CatBoost mendukung `scale_pos_weight` atau `class_weight`

### 4. Threshold Moving

Model menghasilkan probability, lalu threshold (default 0.5) untuk binary classification.
Dengan imbalanced data, **adjust threshold** untuk optimize metric yang relevan.

```python
# Default: predict class 1 if P(class=1) > 0.5
# Imbalanced: predict class 1 if P(class=1) > 0.3 (lebih sensitif terhadap minority)
```

Gunakan **Precision-Recall Curve** atau **ROC Curve** untuk pilih threshold optimal.

## Evaluation Metrics yang Tepat

**JANGAN gunakan Accuracy!** Gunakan metrics yang informatif untuk imbalanced data:

### 1. Confusion Matrix
```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP
```

### 2. Precision, Recall, F1-Score

**Precision**: Dari semua yang diprediksi positif, berapa yang benar?
```
Precision = TP / (TP + FP)
```
Penting ketika **False Positive mahal** (misal: spam filter, jangan block email penting)

**Recall (Sensitivity, True Positive Rate)**: Dari semua yang sebenarnya positif, berapa yang terdeteksi?
```
Recall = TP / (TP + FN)
```
Penting ketika **False Negative mahal** (misal: medical diagnosis, jangan miss penyakit)

**F1-Score**: Harmonic mean antara Precision dan Recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Balance antara Precision dan Recall.

**F-beta Score**: Generalisasi F1 dengan weight berbeda
```
F_beta = (1 + beta²) * (Precision * Recall) / (beta² * Precision + Recall)
```
- `beta < 1`: prioritas Precision
- `beta > 1`: prioritas Recall
- `beta = 2`: Recall 2x lebih penting (F2-score)

### 3. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

ROC curve: plot TPR (Recall) vs FPR di berbagai threshold.
```
TPR = TP / (TP + FN)  # True Positive Rate = Recall
FPR = FP / (FP + TN)  # False Positive Rate
```

**AUC (Area Under Curve)**: Probabilitas model rank random positive sample lebih tinggi dari random negative sample.
- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect classifier
- AUC = 0.0: Perfect tapi inverted (flip predictions!)

**Catatan**: ROC-AUC bisa optimistic pada imbalanced data (karena TN besar). Gunakan dengan hati-hati.

### 4. Precision-Recall AUC

Lebih informatif untuk imbalanced data dibanding ROC-AUC.
Plot Precision vs Recall di berbagai threshold.

**Average Precision (AP)**: Area under Precision-Recall curve
```python
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_proba)
```

### 5. Matthews Correlation Coefficient (MCC)

Metric balance yang consider semua elemen confusion matrix.
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
Range: [-1, 1]
- MCC = 1: Perfect prediction
- MCC = 0: Random prediction
- MCC = -1: Total disagreement

MCC robust terhadap imbalanced data dan sering digunakan di biomedical research.

### 6. Cohen's Kappa

Mengukur agreement antara predicted dan actual, adjusted untuk chance.
```
Kappa = (p_o - p_e) / (1 - p_e)
```
- `p_o`: observed agreement (accuracy)
- `p_e`: expected agreement by chance

Range: [-1, 1], interpretasi mirip MCC.

## Strategi Pemilihan Teknik

### Decision Tree untuk Pilih Teknik:

```
1. Ukuran dataset?
   - Besar (>100K samples) → Undersampling atau Hybrid
   - Kecil (<10K samples) → Oversampling (SMOTE/ADASYN)
   - Medium → Hybrid atau Cost-Sensitive

2. Imbalance ratio?
   - Moderat (1:10 s/d 1:100) → SMOTE + class_weight
   - Ekstrim (1:100 s/d 1:1000) → SMOTE + ENN + class_weight + threshold moving
   - Sangat ekstrim (>1:1000) → Anomaly detection approach (One-Class SVM, Isolation Forest)

3. Computational resource?
   - Terbatas → Random Undersampling + class_weight
   - Cukup → SMOTE + ensemble methods
   - Banyak → Hybrid + hyperparameter tuning + threshold optimization

4. Domain constraint?
   - Medical/Safety-critical → Maximize Recall (minimize False Negatives)
   - Fraud/Spam → Balance Precision-Recall (minimize wasted investigation)
   - Marketing → Maximize Precision (fokus pada high-quality leads)
```

### Kombinasi yang Proven:

**Setup 1: Balanced Approach**
```python
# Preprocessing
SMOTE(sampling_strategy=0.5)  # Tidak perlu 1:1, cukup kurangi imbalance

# Model
RandomForestClassifier(class_weight='balanced')

# Evaluation
Fokus pada F1-score, Precision-Recall AUC
```

**Setup 2: Recall-First (Medical, Safety)**
```python
# Preprocessing
SMOTE(sampling_strategy=0.8) + Tomek Links

# Model
LGBMClassifier(scale_pos_weight=ratio, is_unbalance=True)

# Threshold
Tuning threshold untuk maximize Recall dengan Precision minimum acceptable

# Evaluation
F2-score (Recall 2x penting), Sensitivity
```

**Setup 3: Precision-First (Fraud, High-cost FP)**
```python
# Preprocessing
Borderline-SMOTE atau ADASYN

# Model
XGBoost(scale_pos_weight=ratio)

# Threshold
Tuning threshold untuk maximize Precision dengan Recall minimum acceptable

# Evaluation
Precision, F0.5-score (Precision 2x penting)
```

## Tips dan Pitfalls

### ✅ Best Practices

1. **Stratified Split**: Selalu gunakan `stratify=y` di train_test_split untuk jaga proporsi kelas
2. **Resample hanya Training Set**: JANGAN resample test set (test set harus reflect real-world distribution)
3. **Cross-validation dengan Stratified**: Gunakan `StratifiedKFold`
4. **Pipeline**: Wrap resampling dalam pipeline untuk avoid data leakage
5. **Try Multiple Techniques**: Tidak ada silver bullet, experiment dan compare
6. **Domain Knowledge**: Pilih metric berdasarkan business impact, bukan technical preference
7. **Interpretability**: Monitor confusion matrix, jangan hanya lihat single metric

### ❌ Common Pitfalls

1. **Resampling sebelum split**: Data leakage! Synthetic samples dari train masuk ke test
2. **Hanya lihat Accuracy**: Misleading, bisa 99% tapi tidak detect minority class
3. **Over-optimistic dengan ROC-AUC**: Gunakan Precision-Recall AUC untuk imbalanced data
4. **Oversample terlalu agresif**: 1:1 ratio tidak selalu optimal, bisa overfitting
5. **Lupa adjust threshold**: Default 0.5 tidak optimal untuk imbalanced data
6. **Ignore computational cost**: SMOTE pada dataset besar bisa lambat
7. **Copy-paste tanpa understand**: Pahami trade-off setiap teknik

### 🔧 Debugging Tips

**Jika Recall rendah:**
- Model tidak belajar minority class → increase oversampling atau class_weight
- Threshold terlalu tinggi → lower threshold
- Model terlalu simple → increase complexity atau ensemble

**Jika Precision rendah:**
- Terlalu banyak False Positives → kurangi oversampling atau increase threshold
- Noise dari synthetic samples → gunakan SMOTE + ENN atau Tomek Links
- Model terlalu complex → regularization

**Jika F1 rendah tapi ROC-AUC tinggi:**
- Model bisa separate classes tapi threshold tidak optimal → threshold tuning
- ROC-AUC misleading karena imbalanced → fokus pada PR-AUC

## Real-World Applications

### 1. Fraud Detection (Financial)
- **Imbalance**: 0.1% - 0.5% fraud
- **Metric priority**: Precision (minimize false alarms) + Recall (catch fraud)
- **Technique**: SMOTE + XGBoost(scale_pos_weight) + threshold tuning
- **Business impact**: Balance antara catch fraud dan tidak ganggu legitimate users

### 2. Medical Diagnosis (Cancer Detection)
- **Imbalance**: 1% - 5% positive cases
- **Metric priority**: Recall (minimize miss diagnosis), acceptable lower Precision
- **Technique**: ADASYN + Ensemble + optimize for high Recall
- **Business impact**: False negative (miss cancer) lebih berbahaya dari false positive (further testing)

### 3. Churn Prediction (Telecom/SaaS)
- **Imbalance**: 10% - 20% churn
- **Metric priority**: Balance F1, tapi prioritas Precision (fokus retention effort)
- **Technique**: SMOTE(sampling_strategy=0.6) + LightGBM + class_weight
- **Business impact**: Retention budget terbatas, fokus pada high-risk customers

### 4. Anomaly Detection (Cybersecurity)
- **Imbalance**: <0.01% attacks
- **Metric priority**: Recall extremely high, Precision as high as possible
- **Technique**: Isolation Forest atau One-Class SVM (treat as anomaly detection)
- **Business impact**: Miss attack = catastrophic, tapi terlalu banyak alert = alert fatigue

### 5. Predictive Maintenance (Manufacturing)
- **Imbalance**: 2% - 5% failures
- **Metric priority**: Recall (predict failure before happen), minimize downtime
- **Technique**: Borderline-SMOTE + Random Forest(balanced) + threshold tuning
- **Business impact**: Unplanned downtime sangat mahal, false positive = unnecessary maintenance (murah)

## Mathematical Intuition Deep Dive

### SMOTE: Geometri Interpolasi

SMOTE membuat synthetic sample dengan interpolasi linear di feature space:

```
x_new = x_i + λ * (x_nn - x_i), λ ~ U(0,1)
```

**Intuisi Geometris:**
- `x_i`: sample minority original
- `x_nn`: salah satu k-nearest neighbor dari kelas yang sama
- `λ`: random weight untuk variasi

Jika `x_i = [2, 3]` dan `x_nn = [4, 5]`, synthetic bisa di `[3, 4]` (λ=0.5).

**Mengapa ini work?**
1. **Manifold assumption**: Data dari kelas yang sama cluster di region tertentu dalam feature space
2. **Interpolasi = plausible samples**: Titik di antara dua sample dari kelas sama likely juga dari kelas sama
3. **Smooth decision boundary**: Menambah density di region minority memaksa model belajar boundary yang lebih general

**Limitasi:**
- Jika kelas overlap, SMOTE generate synthetic di region ambiguous → noise
- Curse of dimensionality: Di high-dim, "nearest neighbor" bisa jauh → interpolasi tidak meaningful

### Class Weight: Loss Function Modification

Tanpa class weight:
```
L = (1/n) * Σ loss(y_i, ŷ_i)
```

Dengan class weight:
```
L = (1/n) * Σ w_i * loss(y_i, ŷ_i)

w_i = n / (n_classes * n_samples_class_i)
```

Untuk imbalanced dataset (n_majority >> n_minority):
```
w_majority ≈ small (contributes less)
w_minority ≈ large (contributes more)
```

**Intuisi:**
Model gradient descent minimize loss. Dengan weight besar di minority, model "hurt" lebih banyak ketika salah predict minority → forced to learn minority pattern.

**Trade-off:**
Weight terlalu besar → overfitting pada minority (model terlalu sensitif).

### Threshold Moving: Bayes Optimal Decision

Binary classification dengan probability output `p = P(y=1|x)`.

Decision rule: predict 1 if `p > threshold`.

**Default threshold = 0.5** optimal hanya jika:
1. Classes balanced
2. Misclassification cost sama

Untuk imbalanced, optimal threshold:
```
threshold_optimal = C(FP) / (C(FP) + C(FN))
```
- `C(FP)`: cost of False Positive
- `C(FN)`: cost of False Negative

Jika `C(FN) >> C(FP)` (misal: medical), threshold harus **rendah** (predict positive lebih agresif).

**Cara cari threshold optimal:**
1. Hitung predictions probability pada validation set
2. Loop threshold dari 0 to 1 (step 0.01)
3. Hitung metric (F1, F-beta, atau custom cost) pada setiap threshold
4. Pilih threshold yang maximize metric

## Referensi & Resources

### Papers

1. **SMOTE: Synthetic Minority Over-sampling Technique**
   - Chawla et al., 2002
   - Original SMOTE paper
   - [DOI: 10.1613/jair.953](https://doi.org/10.1613/jair.953)

2. **ADASYN: Adaptive Synthetic Sampling Approach**
   - He et al., 2008
   - Improved SMOTE dengan adaptive density
   - [DOI: 10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969)

3. **Borderline-SMOTE: A Variant of SMOTE**
   - Han et al., 2005
   - Fokus oversample pada borderline samples
   - [DOI: 10.1007/11538059_91](https://doi.org/10.1007/11538059_91)

4. **Learning from Imbalanced Data**
   - He & Garcia, 2009
   - Comprehensive survey
   - [DOI: 10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)

5. **The Relationship Between Precision-Recall and ROC Curves**
   - Davis & Goadrich, 2006
   - Kenapa PR curve lebih baik untuk imbalanced data
   - [ACM Digital Library](https://dl.acm.org/doi/10.1145/1143844.1143874)

### Libraries

- **imbalanced-learn**: https://imbalanced-learn.org/
  - SMOTE, ADASYN, dan semua resampling techniques
  - Compatible dengan scikit-learn pipeline
  
- **scikit-learn**: https://scikit-learn.org/
  - `class_weight` parameter
  - Ensemble methods dengan balanced sampling
  
- **XGBoost, LightGBM, CatBoost**:
  - `scale_pos_weight`, `is_unbalance` parameters
  - Tree-based yang handle imbalanced well

### Tutorials & Courses

1. **Imbalanced Learn Documentation**
   - https://imbalanced-learn.org/stable/user_guide.html
   - Complete guide dengan examples

2. **Google ML Course - Classification**
   - https://developers.google.com/machine-learning/crash-course/classification
   - Precision, Recall, ROC, AUC explained

3. **Kaggle - Credit Card Fraud Detection**
   - https://www.kaggle.com/mlg-ulb/creditcardfraud
   - Real imbalanced dataset untuk practice

### Books

1. **Imbalanced Learning: Foundations, Algorithms, and Applications**
   - He & Ma (editors), 2013
   - Comprehensive academic treatment

2. **Learning from Imbalanced Data Sets**
   - Alberto Fernández et al., 2018
   - Practical guide dengan case studies

## Kesimpulan

Imbalanced classification adalah masalah fundamental dalam ML dengan impact besar di real-world applications. Tidak ada solusi universal - pemilihan teknik tergantung pada:

1. **Dataset characteristics**: size, imbalance ratio, feature space
2. **Domain constraints**: cost of FP vs FN, interpretability requirements
3. **Computational resources**: training time, memory constraints
4. **Business objectives**: optimize untuk metric yang align dengan business value

**Golden Rules:**
- Stratified split, resample hanya training
- Gunakan metrics yang relevan (bukan accuracy!)
- Experiment dengan multiple techniques
- Understand trade-offs, bukan copy-paste
- Validate pada real-world distribution

Dengan pemahaman solid tentang trade-offs dan mathematical intuition, kamu bisa tackle imbalanced classification dengan confidence.

---

**Lihat `notebook.py` untuk complete implementation dengan real dataset!**
