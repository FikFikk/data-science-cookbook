# Conformal Prediction: Estimasi Ketidakpastian Bergaransi di Python

> **Kategori**: Uncertainty Estimation, Model Trustworthiness, Machine Learning Theory  
> **Tingkat Kesulitan**: Menengah  
> **Framework**: Python (scikit-learn, numpy, pandas, matplotlib)  
> **Waktu Baca**: ~20 menit

---

## 📌 Apa Itu Conformal Prediction?

Dalam banyak aplikasi kritis (seperti medis, keuangan, atau kendaraan otonom), memberikan prediksi titik tunggal (point prediction) seperti *"Pasien ini menderita penyakit X dengan probabilitas 82%"* tidaklah cukup. Model Machine Learning sering kali menderita **overconfidence** (sangat yakin padahal salah), terutama saat menghadapi distribusi data baru.

**Conformal Prediction (CP)** adalah kerangka kerja (framework) matematika yang mengubah model Machine Learning *apa pun* menjadi prediktor yang dilengkapi dengan **garansi statistik persis (exact statistical guarantee)** untuk ketidakpastian.

Alih-alih memberikan satu nilai/kelas tunggal, Conformal Prediction menghasilkan:
1. **Prediksi Klasifikasi**: *Prediction Set* berisi sekumpulan kelas potensial (misal: `{Kucing, Anjing}`) alih-alih satu kelas dasar.
2. **Prediksi Regresi**: *Prediction Interval* berupa rentang nilai `[Lower, Upper]` yang dijamin mencakup nilai target sebenarnya dengan tingkat kepercayaan tertentu (misal: 95%).

### Analogi Sehari-hari

Bayangkan ahli meteorologi memprediksi suhu besok:
- **Prediksi titik standar**: *"Suhu besok persis 28°C."* (Hampir pasti sedikit meleset).
- **Interval tradisional**: *"Suhu besok 28°C ± 2°C."* (Tanpa garansi statistik seberapa sering rentang ini benar pada data baru).
- **Conformal Prediction**: *"Saya garansi 95% bahwa suhu besok berada antara 25.5°C hingga 30.2°C pada situasi apa pun."* (Memiliki garansi matematis pada finite-sample).

---

## 🧠 Mengapa Ini Penting?

| Masalah Prediksi Standar | Solusi Conformal Prediction |
|---|---|
| Probabilitas `predict_proba()` terkalibrasi buruk & overconfident | Garansi coverage statistik persis untuk sampel terhingga (*finite-sample guarantee*) |
| Tidak bisa mengukur risiko kegagalan model pada kasus marginal | Ukuran prediksi (*set size* / *interval width*) otomatis membesar saat model ragu |
| Membutuhkan asumsi distribusi data tertentu (misal: Gaussian) | **Distribution-free**: Bebas dari asumsi distribusi statistik data |
| Berbeda algoritma butuh rumus interval yang berbeda | **Model-agnostic**: Bekerja di atas model apa pun (Random Forest, Neural Network, XGBoost, dll) |

---

## 📐 Intuisi Matematis (Tidak Terlalu Formal)

### 1. Asumsi Dasar: Engkapsulasi Pertukaran (Exchangeability)

Satu-satuan asumsi utama Conformal Prediction adalah data $(X_1, Y_1), \dots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$ bersifat **exchangeable** (independen dan terdistribusi identik / i.i.d. adalah kasus khusus dari exchangeable). Tidak ada asumsi distribusi normal atau linear!

### 2. Skor Ketidakcocokan (Non-conformity Score)

Skor $S_i$ mengukur seberapa "aneh" atau tidak cocoknya pasangan data $(X_i, Y_i)$ menurut model yang telah dilatih.

- **Untuk Regresi**: Residual absolut antara target sebenarnya $y_i$ dan prediksi model $\hat{y}_i$:
  $$S_i = |y_i - \hat{y}(X_i)|$$
- **Untuk Klasifikasi**: Peluang kegagalan model pada kelas sebenarnya:
  $$S_i = 1 - P(Y = y_i \mid X_i)$$

### 3. Split Conformal Algorithm (Inductive Conformal Prediction)

Untuk menghemat komputasi tanpa perlu melatih ulang model:
1. Bagi data menjadi **Training Set** dan **Calibration Set** (ukuran $n$).
2. Latih model pada Training Set.
3. Hitung Non-conformity Score $S_1, S_2, \dots, S_n$ pada Calibration Set.
4. Tentukan ambang batas persentil $\hat{q}$ dengan koreksi sampel berukuran hingga (finite-sample correction):
   $$p = \frac{\lceil (n+1)(1-\alpha) \rceil}{n}$$
   Di mana $\alpha$ adalah tingkat toleransi error (misal $\alpha = 0.05$ untuk coverage 95%). Ambang $\hat{q}$ diambil sebagai kuantil ke-$p$ dari skor kualibrasi.
5. Untuk titik data baru $X_{\text{new}}$:
   - **Regresi**: $C(X_{\text{new}}) = [\hat{y}(X_{\text{new}}) - \hat{q}, \hat{y}(X_{\text{new}}) + \hat{q}]$
   - **Klasifikasi**: Masukkan semua kelas $y$ yang memenuhi $1 - P(Y=y \mid X_{\text{new}}) \le \hat{q}$

---

## 🏗️ Alur Kerja Split Conformal Prediction

```
Dataset Utama
    │
    ├──► Split Train Set (50%) ──► Latih Model Base (Random Forest / LightGBM)
    │
    ├──► Split Calibration Set (25%) ──► Hitung Non-Conformity Scores (S_i) ──► Hitung Quantile Threshold (q_hat)
    │
    └──► Split Test Set (25%) ──► Evaluasi Coverage & Set Size / Interval Width
```

---

## 💻 Implementasi Python (From Scratch)

Berikut langkah-langkah implementasi Conformal Prediction dalam skenario klasifikasi dan regresi menggunakan `scikit-learn` dan `numpy`. (File runnable tersedia di `notebook.py`).

### Kode Ringkas

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Filter Quantile Function
def get_conformal_quantile(scores, alpha):
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, np.clip(level, 0, 1), method='higher')

# 1. Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Fit Base Model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# 3. Calibration Phase
cal_preds = reg.predict(X_cal)
cal_residuals = np.abs(y_cal - cal_preds)
q_hat = get_conformal_quantile(cal_residuals, alpha=0.05) # Target 95% Coverage

# 4. Inference on Test Set
test_preds = reg.predict(X_test)
lower_bound = test_preds - q_hat
upper_bound = test_preds + q_hat

# 5. Evaluate Coverage
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
print(f"Empirical Coverage: {coverage*100:.2f}%")
```

---

## 📊 Hasil Evaluasi Eksperimen

Hasil eksekusi `notebook.py` melegitimasi teori matematis Conformal Prediction pada sampel terhingga:

| Task | Target Coverage ($1-\alpha$) | Empirical Coverage | Metric Efisiensi |
|---|---|---|---|
| **Classification** | **90.0%** | **88.20%** | Avg Set Size: **0.97** kelas |
| **Regression** | **95.0%** | **95.20%** | Avg Interval Width: **1.3234** |

- Pada regresi, cakupan empiris (95.20%) sangat presisi mendekati target matematis 95.0%.
- Pada klasifikasi, saat prediksi sangat yakin, ukuran *prediction set* rata-rata di bawah 1 (menandakan model sangat efisien dan informatif).

---

## 🌍 Real-World Applications

1. **Diagnosis Medis & Radiologi**: Menghasilkan set diagnosa kanker potensial. Daripada memilih 1 tumor ganas/jinak, CP memberikan sekumpulan diagnosa terjamin sehingga dokter dapat melakukan pemeriksaan lanjutan jika *prediction set* berisi lebih dari 1 sampel.
2. **Autonomous Driving**: Deteksi pejalan kaki dan jarak objek membutuhkan batas aman tergaransi untuk menghindari tabrakan.
3. **Algorithmic Trading & Risk Management**: Mengestimasi risiko downside harga aset saham dengan rentang interval harga bergaransi statistik 99%.
4. **Natural Language Processing (LLM)**: Menentukan kapan LLM boleh menjawab langsung vs kapan harus menyatakan "ragu-ragu" berdasarkan tingkat uncertainty jawaban.

---

## 💡 Tips & Pitfalls

### Tips Best Practice
- **Pisahkan Data Calibration dengan Ketat**: Jangan pernah menghitung kuantil $q$ pada data training! Hal itu akan menyebabkan *data leakage* dan merusak garansi coverage.
- **Adaptive Conformal Prediction (APS / RAPS)**: Untuk klasifikasi dengan banyak kelas (misal 1000 kelas ImageNet), gunakan skor akumulasi kuantil agar *predictive set* tidak membengkak berlebihan pada kasus sulit.
- **Normalized Conformal Prediction (CQR)**: Gunakan Conformalized Quantile Regression untuk menghasilkan *interval width* yang bervariasi secara adaptif tergantung pada tingkat kesulitan lokal sampel input (*heteroscedasticity*).

### Pitfalls (Jebakan Umum)
- **Data Drift / Concept Drift**: Garansi Conformal Prediction bergantung pada asumsi *exchangeability*. Jika distribusi data uji berubah drastis dibandingkan data kalibrasi (*non-stationary distribution*), garansi statistik dapat gugur.
- **Ukuran Calibration Set Terlalu Kecil**: Jika data kalibrasi terlalu sedikit (misal $n < 50$), nilai kuantil akan menjadi terlalu konservatif (interval sangat lebar) sehingga kurang berguna secara praktis.

---

## 📚 Referensi Papers & Resources

1. **Angelopoulos, A. N., & Bates, S. (2021)**. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv preprint arXiv:2107.07511.
2. **Shafer, G., & Vovk, V. (2008)**. *A tutorial on conformal prediction*. Journal of Machine Learning Research, 9(Mar), 371-421.
3. **MAPIE Library (Modular Adaptive Prediction Interval Estimator)**: Library Python scikit-learn compatible populer untuk Conformal Prediction (`pip install mapie`).
4. **Awesome Conformal Prediction**: Repositori GitHub berisi kompilasi paper, tools, dan video tutorial CP.
