# Time Series Forecasting dengan Prophet

## Pengantar

Prophet adalah library forecasting yang dikembangkan oleh Facebook (Meta) untuk prediksi time series yang robust dan mudah digunakan. Prophet dirancang khusus untuk data bisnis yang memiliki pola musiman kuat dan beberapa seasonal periods (harian, mingguan, tahunan).

## Use Cases

- **Demand Forecasting**: Prediksi penjualan produk, traffic website, atau resource usage
- **Capacity Planning**: Perencanaan infrastruktur berdasarkan proyeksi pertumbuhan
- **Anomaly Detection**: Deteksi outlier dengan membandingkan nilai aktual vs prediksi
- **Budget Planning**: Proyeksi revenue atau cost untuk periode mendatang
- **Workforce Planning**: Prediksi kebutuhan staffing berdasarkan seasonal pattern

## Keunggulan Prophet

1. **Intuitive Parameters**: Parameter yang mudah dipahami oleh non-expert
2. **Automatic Seasonality Detection**: Deteksi otomatis pola harian, mingguan, tahunan
3. **Holiday Effects**: Built-in support untuk hari libur dan event khusus
4. **Robust to Missing Data**: Handle missing values dan outliers dengan baik
5. **Fast Fitting**: Training yang cepat bahkan untuk dataset besar
6. **Uncertainty Intervals**: Confidence intervals untuk mengukur ketidakpastian prediksi

## Mathematical Intuition

Prophet menggunakan **additive model** yang mendekomposisi time series menjadi tiga komponen utama:

```
y(t) = g(t) + s(t) + h(t) + εₜ
```

Dimana:
- **g(t)**: Trend (pertumbuhan non-periodik)
- **s(t)**: Seasonality (perubahan periodik - mingguan, bulanan, tahunan)
- **h(t)**: Holiday effects (pengaruh hari libur/event)
- **εₜ**: Error term (noise yang tidak dapat dijelaskan model)

### 1. Trend Component g(t)

Prophet mendukung dua jenis trend:

**a) Linear Trend (default untuk data tanpa saturation)**
```
g(t) = (k + a(t)ᵀδ) · t + (m + a(t)ᵀγ)
```
- k: growth rate
- δ: rate adjustments (changepoints)
- m: offset parameter

**b) Logistic Growth (untuk data dengan saturating maximum)**
```
g(t) = C(t) / (1 + exp(-(k + a(t)ᵀδ)(t - (m + a(t)ᵀγ))))
```
- C(t): carrying capacity (maximum achievable value)

**Changepoints**: Prophet secara otomatis mendeteksi titik-titik dimana trend berubah drastis menggunakan regularized regression.

### 2. Seasonality Component s(t)

Prophet menggunakan **Fourier series** untuk model seasonality yang fleksibel:

```
s(t) = Σ [aₙ · cos(2πnt/P) + bₙ · sin(2πnt/P)]
```

Dimana:
- P: period (365.25 untuk yearly, 7 untuk weekly)
- N: number of Fourier terms (semakin besar, semakin fleksibel pattern-nya)
- aₙ, bₙ: koefisien yang di-learn dari data

Fourier series memungkinkan model menangkap pola musiman yang kompleks dan tidak sempurna sinus/cosinus.

### 3. Holiday Effects h(t)

```
h(t) = Z(t)κ
```

Dimana:
- Z(t): matrix of holiday indicators
- κ: vector of holiday effects (di-learn dari data)

Prophet bisa model window di sekitar holiday (misalnya 2 hari sebelum/sesudah Natal).

### 4. Fitting Process

Prophet menggunakan **Stan** (probabilistic programming language) untuk Bayesian inference:

1. **Prior distributions**: Set prior yang reasonable untuk semua parameters
2. **MCMC sampling**: Sample dari posterior distribution menggunakan L-BFGS
3. **MAP estimation**: Find maximum a posteriori estimate
4. **Uncertainty intervals**: Generate dari posterior samples

## Implementation Step-by-Step

### 1. Instalasi

```bash
pip install prophet pandas matplotlib numpy
```

### 2. Data Preparation

Prophet membutuhkan DataFrame dengan dua kolom:
- `ds`: timestamp (YYYY-MM-DD atau datetime)
- `y`: nilai yang ingin diprediksi

### 3. Basic Forecasting

```python
from prophet import Prophet
import pandas as pd

# Create model
model = Prophet()

# Fit model
model.fit(df)

# Make future dataframe
future = model.make_future_dataframe(periods=365)  # 365 days ahead

# Predict
forecast = model.predict(future)

# Plot
model.plot(forecast)
model.plot_components(forecast)
```

### 4. Hyperparameter Tuning

**Key Parameters:**

- **changepoint_prior_scale** (default=0.05): Fleksibilitas trend changes
  - Lebih besar = lebih fleksibel, risk overfitting
  - Lebih kecil = lebih smooth, risk underfitting
  
- **seasonality_prior_scale** (default=10): Kekuatan seasonality
  - Lebih besar = seasonality lebih kuat
  
- **seasonality_mode**: 'additive' vs 'multiplicative'
  - Additive: seasonality amplitude konstan
  - Multiplicative: seasonality amplitude grows with trend

- **changepoint_range** (default=0.8): Proporsi data untuk detect changepoints
  - 0.8 = changepoints hanya di 80% data pertama

**Example dengan tuning:**

```python
model = Prophet(
    changepoint_prior_scale=0.1,      # lebih conservative
    seasonality_prior_scale=15,        # seasonality lebih kuat
    seasonality_mode='multiplicative', # untuk data dengan growth
    changepoint_range=0.9,
    daily_seasonality=False,           # disable jika tidak perlu
    weekly_seasonality=True,
    yearly_seasonality=True
)
```

### 5. Adding Custom Seasonality

```python
# Tambah seasonality custom (misalnya monthly)
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)
```

### 6. Adding Holiday Effects

```python
# Define holidays
holidays = pd.DataFrame({
    'holiday': 'lunar_new_year',
    'ds': pd.to_datetime(['2023-01-22', '2024-02-10', '2025-01-29']),
    'lower_window': -1,  # 1 hari sebelum
    'upper_window': 2,   # 2 hari sesudah
})

model = Prophet(holidays=holidays)
```

### 7. Adding Regressors (External Variables)

```python
# Tambah variabel eksternal (misalnya promo, temperature)
model.add_regressor('promo_active')
model.add_regressor('temperature')

# Data harus punya kolom promo_active dan temperature
model.fit(df)

# Future dataframe juga harus punya regressors
future = model.make_future_dataframe(periods=30)
future['promo_active'] = 0  # asumsi no promo
future['temperature'] = 25  # asumsi constant
```

## Evaluation Metrics

### 1. Time Series Cross-Validation

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Cross-validation
df_cv = cross_validation(
    model, 
    initial='730 days',    # minimal training data
    period='180 days',     # spacing between cutoff dates
    horizon='365 days'     # forecast horizon
)

# Calculate metrics
df_metrics = performance_metrics(df_cv)
print(df_metrics[['horizon', 'mape', 'rmse', 'mae']].head())
```

### 2. Key Metrics

**MAE (Mean Absolute Error)**
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```
- Unit yang sama dengan data asli
- Mudah diinterpretasikan

**RMSE (Root Mean Squared Error)**
```
RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]
```
- Penalti lebih besar untuk error besar
- Sensitif terhadap outliers

**MAPE (Mean Absolute Percentage Error)**
```
MAPE = (100/n) Σ |yᵢ - ŷᵢ| / |yᵢ|
```
- Persentase error, mudah dibandingkan antar dataset
- Tidak defined jika yᵢ = 0

**Coverage**: Proporsi actual values yang masuk dalam prediction interval

### 3. Visualisasi Evaluasi

```python
from prophet.plot import plot_cross_validation_metric

# Plot MAPE over forecast horizon
plot_cross_validation_metric(df_cv, metric='mape')
```

## Real-World Applications

### 1. E-commerce Demand Forecasting

**Scenario**: Toko online ingin prediksi penjualan untuk inventory planning

**Data characteristics**:
- Strong weekly seasonality (weekend peak)
- Yearly seasonality (liburan akhir tahun)
- Holiday effects (Black Friday, Harbolnas)
- Growth trend

**Prophet configuration**:
```python
model = Prophet(
    seasonality_mode='multiplicative',  # sales grow over time
    changepoint_prior_scale=0.05
)
model.add_country_holidays(country_name='ID')  # Indonesia holidays
model.add_regressor('marketing_spend')
```

### 2. Website Traffic Prediction

**Scenario**: Platform digital ingin prediksi traffic untuk capacity planning

**Data characteristics**:
- Daily seasonality (peak hours)
- Weekly seasonality (weekday vs weekend)
- Seasonal campaigns

**Prophet configuration**:
```python
model = Prophet(
    daily_seasonality=10,  # strong daily pattern
    weekly_seasonality=True,
    changepoint_prior_scale=0.2  # allow flexible trends
)
```

### 3. Energy Consumption Forecasting

**Scenario**: Utility company ingin prediksi demand listrik

**Data characteristics**:
- Temperature dependency
- Time-of-day patterns
- Seasonal variations

**Prophet configuration**:
```python
model = Prophet()
model.add_regressor('temperature')
model.add_regressor('humidity')
model.add_seasonality(name='hourly', period=1/24, fourier_order=8)
```

## Tips & Best Practices

### ✅ Do's

1. **Explore Data First**: Plot time series, identifikasi outliers, missing values, dan pola
2. **Start Simple**: Mulai dengan default parameters, tuning nanti jika perlu
3. **Use Domain Knowledge**: Tambahkan holidays dan regressors yang relevan
4. **Cross-Validate**: Selalu validate dengan time series CV, bukan random split
5. **Check Residuals**: Analisa residuals untuk identifikasi pattern yang missed
6. **Multiple Horizons**: Test performance di berbagai forecast horizons
7. **Monitor Uncertainty**: Perhatikan prediction intervals, bukan hanya point estimates

### ❌ Don'ts

1. **Don't Use Random Train-Test Split**: Time series butuh chronological split
2. **Don't Ignore Outliers**: Handle outliers dulu sebelum modeling (bisa corrupt trend)
3. **Don't Overfit on Recent Data**: Changepoint_range terlalu tinggi bisa overfit
4. **Don't Forget Regressors in Future**: Future dataframe harus punya semua regressors
5. **Don't Trust Long-Term Forecasts Blindly**: Uncertainty meningkat eksponensial
6. **Don't Mix Seasonality Modes**: Pilih additive atau multiplicative, konsisten
7. **Don't Ignore Capacity**: Untuk growth yang saturating, gunakan logistic growth

## Common Pitfalls

### 1. Overfitting Recent Trends

**Problem**: Model terlalu mengikuti fluktuasi terkini

**Solution**:
```python
# Kurangi changepoint_prior_scale
model = Prophet(changepoint_prior_scale=0.001)

# Atau batasi changepoint detection
model = Prophet(changepoint_range=0.8)
```

### 2. Underfitting Seasonality

**Problem**: Seasonal patterns tidak tertangkap dengan baik

**Solution**:
```python
# Increase Fourier order
model.add_seasonality(
    name='monthly', 
    period=30.5, 
    fourier_order=10  # increase from 5
)

# Atau increase seasonality_prior_scale
model = Prophet(seasonality_prior_scale=20)
```

### 3. Outlier Contamination

**Problem**: Outliers merusak trend estimation

**Solution**:
```python
# Remove outliers sebelum fitting
from scipy import stats
z_scores = np.abs(stats.zscore(df['y']))
df_clean = df[z_scores < 3]

# Atau gunakan robust regression (tidak built-in, custom implementation)
```

### 4. Missing Future Regressors

**Problem**: Error karena future dataframe tidak punya regressor columns

**Solution**:
```python
# Always populate regressors in future
future = model.make_future_dataframe(periods=30)
future = future.merge(df[['ds', 'regressor_col']], on='ds', how='left')
future['regressor_col'].fillna(method='ffill', inplace=True)
```

## Kapan TIDAK Menggunakan Prophet

1. **Hourly/Minute-Level Data dengan banyak noise**: Prophet lebih cocok daily/weekly aggregation
2. **Short Time Series** (< 2 seasonal cycles): Tidak cukup data untuk learn seasonality
3. **Non-Stationary Extreme**: Data dengan trend/seasonality yang berubah drastis setiap periode
4. **Multivariate Dependencies**: Jika perlu model interaksi kompleks antar variabel (gunakan VAR, LSTM)
5. **Real-Time Requirements**: Prophet fitting bisa lambat untuk very large datasets

## Alternatif Methods

- **ARIMA/SARIMA**: Traditional statistical method, good untuk univariate time series
- **LSTM/GRU**: Deep learning untuk sequential data dengan complex patterns
- **XGBoost Time Series**: Feature engineering + gradient boosting
- **Exponential Smoothing**: Simple dan cepat, cocok untuk short-term forecasts
- **VAR (Vector Autoregression)**: Untuk multivariate time series dengan dependencies

## Referensi

### Papers & Articles

1. **Taylor, S. J., & Letham, B. (2018)**. "Forecasting at Scale". *The American Statistician*.
   - Paper original Prophet dari Facebook Research
   - DOI: 10.1080/00031305.2017.1380080

2. **Harvey, A. C. (1990)**. "Forecasting, Structural Time Series Models and the Kalman Filter". *Cambridge University Press*.
   - Fundamental theory di balik time series decomposition

3. **Hyndman, R. J., & Athanasopoulos, G. (2021)**. "Forecasting: Principles and Practice" (3rd ed).
   - Free online textbook: https://otexts.com/fpp3/
   - Comprehensive guide untuk time series forecasting

### Code & Tutorials

- **Official Prophet Docs**: https://facebook.github.io/prophet/
- **Prophet GitHub**: https://github.com/facebook/prophet
- **Quick Start Guide**: https://facebook.github.io/prophet/docs/quick_start.html

### Datasets untuk Practice

- **Kaggle Store Sales**: https://www.kaggle.com/c/store-sales-time-series-forecasting
- **Wikipedia Page Views**: https://www.kaggle.com/c/web-traffic-time-series-forecasting
- **Energy Consumption**: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

### Communities

- **Stack Overflow Tag**: `facebook-prophet`
- **Prophet Google Group**: https://groups.google.com/forum/#!forum/prophet-user-group

## Summary

Prophet adalah tool yang powerful dan user-friendly untuk time series forecasting, terutama untuk data bisnis dengan seasonal patterns yang kuat. Kunci sukses menggunakan Prophet:

1. **Understand your data**: Seasonality, trends, outliers
2. **Start simple, iterate**: Default parameters sering sudah cukup bagus
3. **Validate properly**: Time series cross-validation, multiple horizons
4. **Leverage domain knowledge**: Holidays, regressors, capacity limits
5. **Interpret uncertainty**: Prediction intervals as important as point estimates

Dengan pendekatan yang tepat, Prophet bisa deliver production-ready forecasts dengan effort yang minimal.
