# Quantile Regression — Prediksi Interval dan Estimasi Risiko

## 📋 Daftar Isi
- [Pengenalan](#pengenalan)
- [Intuisi Matematis](#intuisi-matematis)
- [Implementasi Step-by-Step](#implementasi-step-by-step)
- [Dataset dan Preprocessing](#dataset-dan-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Real-World Applications](#real-world-applications)
- [Tips dan Pitfalls](#tips-dan-pitfalls)
- [Referensi](#referensi)

---

## Pengenalan

**Quantile Regression** adalah teknik regresi yang memprediksi kuantil tertentu dari distribusi target, bukan hanya mean (rata-rata) seperti pada regresi biasa.

### Mengapa Quantile Regression?

Regresi biasa (OLS - Ordinary Least Squares) memprediksi **nilai ekspektasi** (mean). Masalahnya:
- Sensitif terhadap outliers
- Tidak memberi informasi tentang variabilitas atau ketidakpastian
- Asumsi distribusi normal sering dilanggar di dunia nyata

**Quantile Regression** memberikan:
- **Prediction intervals** yang lebih informatif
- **Robust terhadap outliers** — median regression (quantile 0.5) tidak terpengaruh ekstrem
- **Analisis distribusi penuh** — lihat bagaimana predictor mempengaruhi seluruh distribusi, bukan hanya rata-rata
- **Risk assessment** — prediksi skenario terburuk (quantile rendah) dan terbaik (quantile tinggi)

### Use Cases Praktis

1. **Finance**: Estimasi Value at Risk (VaR) — kerugian maksimum pada confidence level tertentu
2. **Healthcare**: Prediksi growth chart anak — percentile 5th, 50th, 95th
3. **E-commerce**: Estimasi delivery time — prediksi waktu tercepat, median, dan worst-case
4. **Energy**: Forecasting load listrik dengan uncertainty bounds
5. **Real Estate**: Prediksi harga properti dengan confidence intervals

---

## Intuisi Matematis

### Regresi Biasa (OLS)
Minimisasi **Mean Squared Error (MSE)**:

```
L_MSE = Σ (y_i - ŷ_i)²
```

Solusi: **conditional mean** E[Y|X]

### Quantile Regression
Minimisasi **Quantile Loss** (juga disebut Pinball Loss):

```
L_τ(y, ŷ) = {
    τ · |y - ŷ|        jika y ≥ ŷ  (underprediction)
    (1-τ) · |y - ŷ|    jika y < ŷ  (overprediction)
}
```

Di mana **τ ∈ (0,1)** adalah quantile yang diinginkan:
- τ = 0.5 → **median regression** (robust terhadap outliers)
- τ = 0.1 → prediksi **10th percentile** (skenario pesimis)
- τ = 0.9 → prediksi **90th percentile** (skenario optimis)

### Asymmetric Penalty

Quantile loss memberikan **penalti asimetris**:
- Untuk τ = 0.9, underprediction dikenai penalti 0.9, overprediction hanya 0.1
- Model "didorong" untuk prediksi lebih tinggi agar 90% data aktual di bawah prediksi
- Sebaliknya untuk τ = 0.1

**Contoh Numerik**:
```
y_true = 100
y_pred = 80  (underprediction by 20)

τ = 0.9:
L_0.9 = 0.9 × 20 = 18  (penalti besar!)

τ = 0.1:
L_0.1 = 0.1 × 20 = 2   (penalti kecil)
```

Model belajar: untuk τ tinggi, lebih baik overshoot daripada undershoot.

### Interpretasi Kuantil

Untuk τ = 0.9:
- "90% dari observasi aktual akan berada **di bawah** prediksi ini"
- Atau: "Ada 10% probabilitas nilai aktual melebihi prediksi"

Dengan melatih model untuk beberapa τ (misal: 0.1, 0.5, 0.9), kita dapat:
- **Interval prediksi**: [Q_0.1, Q_0.9] adalah 80% prediction interval
- **Analisis distribusi**: lihat bagaimana spread berubah terhadap fitur

---

## Implementasi Step-by-Step

### 1. Setup dan Import

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing

# Quantile loss function
def quantile_loss(y_true, y_pred, quantile):
    """
    Compute quantile loss (pinball loss)
    """
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
```

### 2. Load dan Prepare Dataset

Gunakan California Housing dataset — prediksi median house value dengan intervals.

```python
# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset shape: {df.shape}")
print(f"Target variable: MedHouseVal (dalam $100k)")
print(df.describe())

# Train-test split
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: standardize features untuk neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Train Model untuk Multiple Quantiles

Latih model terpisah untuk setiap kuantil:

```python
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
models = {}

for q in quantiles:
    print(f"\nTraining quantile {q}...")
    
    # LightGBM dengan quantile objective
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,  # quantile parameter
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    models[q] = model
    
    # Evaluate
    y_pred = model.predict(X_test)
    loss = quantile_loss(y_test, y_pred, q)
    print(f"Quantile {q} - Test Loss: {loss:.4f}")
```

### 4. Generate Predictions

```python
# Predict semua quantiles pada test set
predictions = {}
for q, model in models.items():
    predictions[q] = model.predict(X_test)

# Create DataFrame untuk analisis
results_df = pd.DataFrame({
    'y_true': y_test.values,
    'q_0.05': predictions[0.05],
    'q_0.25': predictions[0.25],
    'q_0.50': predictions[0.5],
    'q_0.75': predictions[0.75],
    'q_0.95': predictions[0.95]
})

# Hitung prediction interval widths
results_df['PI_90'] = results_df['q_0.95'] - results_df['q_0.05']  # 90% interval
results_df['PI_50'] = results_df['q_0.75'] - results_df['q_0.25']  # 50% interval (IQR)

print(results_df.head(10))
```

---

## Dataset dan Preprocessing

### California Housing Dataset
- **8 fitur**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: MedHouseVal (median house value dalam $100k)
- **20,640 observasi** dari California census 1990

### Preprocessing Steps
1. **Train-test split**: 80-20 stratifikasi
2. **Feature scaling** (optional untuk tree-based): StandardScaler untuk neural networks
3. **No missing values** — dataset sudah bersih
4. **No categorical encoding** — semua fitur numerik

### Alternatif Dataset
- **Bike Sharing**: prediksi demand dengan uncertainty (weather impact)
- **Stock Prices**: financial forecasting dengan risk bounds
- **Medical Data**: patient outcomes dengan percentiles
- **Energy Load**: electricity demand forecasting

---

## Evaluation Metrics

### 1. Quantile Loss (Pinball Loss)
Metrik utama — lower is better:
```python
def evaluate_quantile_model(y_true, y_pred, quantile):
    residual = y_true - y_pred
    loss = np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
    return loss
```

### 2. Coverage (Empirical Quantile)
Validasi apakah prediksi quantile τ benar-benar mengcover τ × 100% data:

```python
def check_coverage(y_true, y_pred, quantile):
    """
    Untuk quantile τ, hitung berapa % data aktual ≤ prediksi
    Idealnya = τ
    """
    coverage = np.mean(y_true <= y_pred)
    return coverage

# Contoh
for q in quantiles:
    y_pred_q = predictions[q]
    cov = check_coverage(y_test, y_pred_q, q)
    print(f"Quantile {q}: Coverage = {cov:.3f} (target = {q:.3f})")
```

**Interpretasi**: 
- Coverage ≈ τ → model well-calibrated
- Coverage < τ → model overpredicting (terlalu optimis)
- Coverage > τ → model underpredicting (terlalu pesimis)

### 3. Interval Width Analysis
Evaluasi kualitas prediction intervals:

```python
# 90% prediction interval
PI_90_width = np.mean(results_df['PI_90'])
print(f"Average 90% PI width: {PI_90_width:.2f}")

# Interval coverage
in_interval = (y_test >= predictions[0.05]) & (y_test <= predictions[0.95])
interval_coverage = np.mean(in_interval)
print(f"90% PI coverage: {interval_coverage:.3f} (target = 0.90)")
```

### 4. Quantile Crossing Check
Pastikan prediksi tidak "crossing" (Q_0.1 ≤ Q_0.5 ≤ Q_0.9):

```python
def check_crossing(predictions_dict):
    """
    Verifikasi monotonicity: Q_τ1 ≤ Q_τ2 untuk τ1 < τ2
    """
    sorted_quantiles = sorted(predictions_dict.keys())
    crossings = 0
    
    for i in range(len(sorted_quantiles) - 1):
        q_low = predictions_dict[sorted_quantiles[i]]
        q_high = predictions_dict[sorted_quantiles[i+1]]
        crossings += np.sum(q_low > q_high)
    
    return crossings

crossings = check_crossing(predictions)
print(f"Quantile crossings: {crossings} / {len(y_test)} samples")
```

**Catatan**: Model independen per-quantile dapat menghasilkan crossing. Solusi: gunakan model monotone atau post-process.

---

## Real-World Applications

### 1. Financial Risk Management
**Value at Risk (VaR)** — estimasi kerugian maksimum pada confidence level tertentu:

```python
# Portfolio returns prediction
# Train quantile model pada historical returns

# Predict 5th percentile (VaR 95%)
var_95 = models[0.05].predict(current_features)
print(f"VaR 95%: Ada 5% probabilitas kerugian melebihi ${var_95:.2f}")
```

**Conditional Value at Risk (CVaR)**: mean dari tail distribution di bawah VaR.

### 2. Healthcare Growth Charts
Pediatric growth monitoring menggunakan percentile curves:

```python
# Features: age, gender, ethnicity, nutrition
# Target: height / weight

# Predict percentile curves
percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]

# Plot growth chart
ages = np.linspace(0, 18, 100)
for p in percentiles:
    height_p = models[p].predict(ages.reshape(-1, 1))
    plt.plot(ages, height_p, label=f"{int(p*100)}th percentile")
```

### 3. E-commerce Delivery Time Estimation
Berikan customer realistic delivery window:

```python
# Features: distance, weather, traffic, package_size, carrier
# Target: delivery_time (hours)

# Predict delivery window
delivery_fast = models[0.1].predict(order_features)  # 10% fastest
delivery_median = models[0.5].predict(order_features)
delivery_slow = models[0.9].predict(order_features)  # 10% slowest

print(f"Estimated delivery: {delivery_median:.1f} hours")
print(f"Window: {delivery_fast:.1f} - {delivery_slow:.1f} hours")
```

### 4. Energy Load Forecasting
Grid operators perlu worst-case scenario untuk capacity planning:

```python
# Features: temperature, time_of_day, day_of_week, holiday
# Target: electricity_load (MW)

# High quantile untuk capacity planning
peak_load_95 = models[0.95].predict(tomorrow_features)
print(f"Persiapkan kapasitas untuk: {peak_load_95:.0f} MW")
```

### 5. Real Estate Price Ranges
Berikan buyer/seller realistic price range:

```python
# Features: location, sqft, bedrooms, age, amenities
# Target: sale_price

# Predict price range
price_low = models[0.1].predict(property_features)   # Pessimistic
price_mid = models[0.5].predict(property_features)   # Median
price_high = models[0.9].predict(property_features)  # Optimistic

print(f"Estimated price: ${price_mid:.0f}k")
print(f"Range: ${price_low:.0f}k - ${price_high:.0f}k")
```

---

## Tips dan Pitfalls

### ✅ Best Practices

1. **Pilih Quantiles yang Relevan**
   - Untuk prediction intervals: {0.05, 0.5, 0.95} atau {0.1, 0.5, 0.9}
   - Untuk analisis distribusi penuh: {0.1, 0.25, 0.5, 0.75, 0.9}
   - Untuk risk management: fokus pada tail (0.01, 0.05, 0.95, 0.99)

2. **Gunakan Algorithms yang Tepat**
   - **Tree-based** (LightGBM, XGBoost, RandomForest): robust, handle non-linearity, built-in quantile support
   - **Linear Quantile Regression**: interpretable, cepat untuk dataset besar
   - **Neural Networks**: flexible tapi butuh careful tuning (custom loss function)

3. **Validate Coverage**
   - Cek empirical coverage ≈ target quantile
   - Gunakan calibration plots untuk diagnose miscalibration
   - Test pada out-of-sample data, bukan training set

4. **Handle Quantile Crossing**
   - Train model secara monotone (e.g., monotone constraints di LightGBM)
   - Post-process: sort predictions untuk enforce monotonicity
   - Atau gunakan simultaneous quantile regression

5. **Feature Engineering Sama Pentingnya**
   - Quantile regression bukan magic — garbage in, garbage out
   - Domain knowledge untuk capture variability sources
   - Interaction terms untuk heteroscedastic relationships

### ⚠️ Common Pitfalls

1. **Crossing Quantiles**
   - Problem: Q_0.9 < Q_0.5 untuk beberapa samples (non-monotone)
   - Solusi: enforce monotonicity constraints atau retrain dengan regularization

2. **Extreme Quantiles Tidak Stabil**
   - τ = 0.01 atau 0.99 sulit diestimasi (data jarang di tail)
   - Butuh dataset lebih besar atau smoothing techniques
   - Trade-off: coverage vs stability

3. **Misinterpretasi Coverage**
   - Coverage = 0.9 bukan berarti "90% confidence"
   - Itu frequentist interpretation: "90% data points fall below this prediction"
   - Berbeda dengan Bayesian credible intervals

4. **Overfitting pada Extreme Quantiles**
   - Model bisa memorize outliers di training set
   - Regularization lebih penting untuk extreme quantiles
   - Validate dengan holdout set, bukan CV saja

5. **Assuming Independence**
   - Model per-quantile independent → bisa crossing
   - Alternative: joint quantile regression atau conditional distributions

6. **Ignoring Heteroscedasticity**
   - Jika variance konstan (homoscedastic), quantile intervals akan sama lebar
   - Real-world data often heteroscedastic → interval width vary with features
   - Feature engineering untuk capture variance drivers

### 🔧 Debugging Tips

**Problem**: Coverage terlalu rendah (e.g., 0.85 untuk quantile 0.9)
- **Diagnosis**: Model overpredicting (terlalu optimis)
- **Fix**: Decrease learning rate, increase regularization, check data leakage

**Problem**: Coverage terlalu tinggi (e.g., 0.95 untuk quantile 0.9)
- **Diagnosis**: Model underpredicting (terlalu pesimis)
- **Fix**: Increase model complexity, check for missing features

**Problem**: Interval terlalu lebar (tidak informatif)
- **Diagnosis**: Model uncertainty tinggi, missing predictive features
- **Fix**: Feature engineering, collect more relevant data, ensemble methods

**Problem**: Interval terlalu sempit (unrealistic)
- **Diagnosis**: Overfitting, aleatoric uncertainty tidak tertangkap
- **Fix**: Regularization, dropout (untuk NN), validate on test set

---

## Referensi

### Papers
1. **Koenker, R., & Bassett, G. (1978)**. "Econometrica: Regression Quantiles"  
   Paper original quantile regression — foundational work

2. **Meinshausen, N. (2006)**. "Quantile Regression Forests"  
   Journal of Machine Learning Research — extension ke tree-based methods

3. **Takeuchi, I., Le, Q. V., Sears, T. D., & Smola, A. J. (2006)**. "Nonparametric Quantile Estimation"  
   JMLR — theoretical properties dan convergence

4. **Rodrigues, F., & Pereira, F. (2020)**. "Beyond Expectation: Deep Joint Mean and Quantile Regression for Spatiotemporal Problems"  
   IEEE Transactions — aplikasi modern dengan deep learning

### Books
- **Quantile Regression** by Roger Koenker (2005) — textbook komprehensif
- **Advances in Quantile Regression** edited by Koenker et al. (2013)

### Libraries & Tools
- **scikit-learn**: `QuantileRegressor` (linear quantile regression)
- **LightGBM / XGBoost**: native quantile objective
- **statsmodels**: `QuantReg` untuk statistical inference
- **PyTorch / TensorFlow**: custom quantile loss untuk neural networks
- **MAPIE**: conformal prediction library (complement quantile regression)

### Online Resources
- [Quantile Regression - Penn State STAT 501](https://online.stat.psu.edu/stat501/)
- [Scikit-learn: Quantile Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html)
- [Towards Data Science: Quantile Regression Tutorial](https://towardsdatascience.com/)

### Related Techniques
- **Conformal Prediction**: distribution-free prediction intervals (ada di repo ini!)
- **Bayesian Methods**: probabilistic predictions dengan uncertainty quantification
- **Quantile Regression Forests**: random forest extension untuk quantile estimation
- **Expectile Regression**: alternative asymmetric loss function

---

## Summary

**Quantile Regression** adalah teknik powerful untuk:
- ✅ Prediction intervals yang informatif
- ✅ Robust terhadap outliers (median regression)
- ✅ Risk assessment dan worst-case planning
- ✅ Analisis distribusi penuh (beyond mean)
- ✅ Heteroscedasticity handling

**Kapan Gunakan**:
- Butuh uncertainty quantification, bukan point prediction saja
- Data memiliki outliers atau skewed distribution
- Risk-sensitive applications (finance, healthcare, operations)
- Compliance requirements (e.g., reporting confidence intervals)

**Kapan Tidak**:
- Data sangat sedikit (< 100 samples) — extreme quantiles tidak reliable
- Hanya butuh mean prediction dan dataset well-behaved (Gaussian)
- Computational constraints (perlu train multiple models)

**Next Steps**: Eksplorasi [Conformal Prediction](../conformal-prediction/) untuk alternative approach yang distribution-free dengan finite-sample guarantees!
