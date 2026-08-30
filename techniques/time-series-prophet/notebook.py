"""
Time Series Forecasting dengan Prophet
=======================================

Tutorial lengkap implementasi Prophet untuk forecasting data time series.
"""

# %% [markdown]
# # Time Series Forecasting dengan Prophet
# 
# Notebook ini mendemonstrasikan penggunaan Prophet untuk forecasting time series
# dengan berbagai skenario: basic forecasting, hyperparameter tuning, holidays,
# dan external regressors.

# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Libraries imported successfully")

# %% [markdown]
# ## 1. Generate Synthetic Time Series Data
# 
# Kita akan generate data dengan karakteristik:
# - Trend pertumbuhan linear
# - Weekly seasonality (peak di weekend)
# - Yearly seasonality (peak di akhir tahun)
# - Random noise
# - Beberapa outliers

# %% Generate data
np.random.seed(42)

# Date range: 3 tahun data
dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Components
trend = np.linspace(100, 200, n)  # Linear growth dari 100 ke 200

# Weekly seasonality (peak weekend)
weekly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 7)

# Yearly seasonality (peak akhir tahun)
yearly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25 - np.pi/2)

# Noise
noise = np.random.normal(0, 10, n)

# Combine all components
y = trend + weekly_seasonality + yearly_seasonality + noise

# Add some outliers
outlier_indices = np.random.choice(n, size=10, replace=False)
y[outlier_indices] += np.random.choice([-50, 50], size=10)

# Create DataFrame
df = pd.DataFrame({
    'ds': dates,
    'y': y
})

print(f"Dataset generated: {len(df)} days dari {df['ds'].min().date()} hingga {df['ds'].max().date()}")
print(f"Range nilai: {df['y'].min():.2f} - {df['y'].max():.2f}")
print(f"\nFirst 5 rows:")
print(df.head())

# %% Visualize generated data
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full time series
axes[0].plot(df['ds'], df['y'], linewidth=1, alpha=0.8)
axes[0].set_title('Generated Time Series Data', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

# Decomposition components (approximate)
axes[1].plot(dates, trend, label='Trend', linewidth=2)
axes[1].plot(dates, weekly_seasonality, label='Weekly Seasonality', linewidth=2, alpha=0.7)
axes[1].plot(dates, yearly_seasonality, label='Yearly Seasonality', linewidth=2, alpha=0.7)
axes[1].set_title('Time Series Components', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_data.png', dpi=100, bbox_inches='tight')
print("\n✓ Data visualization saved: time_series_data.png")

# %% [markdown]
# ## 2. Basic Prophet Model
# 
# Mulai dengan model Prophet sederhana tanpa tuning.

# %% Install Prophet (jika belum)
import subprocess
import sys

try:
    from prophet import Prophet
    print("✓ Prophet already installed")
except ImportError:
    print("Installing Prophet...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "prophet"])
    from prophet import Prophet
    print("✓ Prophet installed successfully")

# %% Train basic model
print("Training basic Prophet model...")

# Split data: 80% train, 20% test
train_size = int(len(df) * 0.8)
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

print(f"Train size: {len(train_df)} days")
print(f"Test size: {len(test_df)} days")

# Create and fit model
model_basic = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model_basic.fit(train_df)
print("✓ Model trained")

# %% Make predictions
# Create future dataframe (includes train + test period)
future_basic = model_basic.make_future_dataframe(periods=len(test_df))
forecast_basic = model_basic.predict(future_basic)

print(f"Forecast generated for {len(forecast_basic)} days")
print("\nForecast columns:")
print(forecast_basic.columns.tolist())

# %% Visualize basic forecast
fig = model_basic.plot(forecast_basic, figsize=(14, 6))
plt.title('Prophet Basic Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('forecast_basic.png', dpi=100, bbox_inches='tight')
print("✓ Basic forecast plot saved: forecast_basic.png")

# %% Visualize components
fig = model_basic.plot_components(forecast_basic, figsize=(14, 10))
plt.tight_layout()
plt.savefig('forecast_components.png', dpi=100, bbox_inches='tight')
print("✓ Components plot saved: forecast_components.png")

# %% [markdown]
# ## 3. Evaluate Model Performance

# %% Calculate metrics on test set
test_forecast = forecast_basic.iloc[train_size:].copy()
test_actual = test_df['y'].values
test_pred = test_forecast['yhat'].values

# Calculate metrics
mae = np.mean(np.abs(test_actual - test_pred))
rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

# Calculate coverage (% of actuals within prediction interval)
lower = test_forecast['yhat_lower'].values
upper = test_forecast['yhat_upper'].values
coverage = np.mean((test_actual >= lower) & (test_actual <= upper)) * 100

print("=" * 50)
print("BASIC MODEL PERFORMANCE")
print("=" * 50)
print(f"MAE (Mean Absolute Error):       {mae:.2f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
print(f"MAPE (Mean Absolute % Error):    {mape:.2f}%")
print(f"Prediction Interval Coverage:    {coverage:.1f}%")
print("=" * 50)

# %% Visualize predictions vs actuals
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Full forecast
axes[0].plot(train_df['ds'], train_df['y'], label='Train Data', linewidth=1, alpha=0.6)
axes[0].plot(test_df['ds'], test_actual, label='Test Data (Actual)', linewidth=2, color='red')
axes[0].plot(test_df['ds'], test_pred, label='Forecast', linewidth=2, color='green', linestyle='--')
axes[0].fill_between(test_df['ds'], lower, upper, alpha=0.2, color='green', label='Prediction Interval')
axes[0].set_title('Forecast vs Actual', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = test_actual - test_pred
axes[1].scatter(range(len(residuals)), residuals, alpha=0.6)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Test Sample Index')
axes[1].set_ylabel('Residual')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecast_evaluation.png', dpi=100, bbox_inches='tight')
print("✓ Evaluation plot saved: forecast_evaluation.png")

# %% [markdown]
# ## 4. Hyperparameter Tuning
# 
# Tuning changepoint_prior_scale dan seasonality_prior_scale untuk
# meningkatkan performa.

# %% Grid search untuk hyperparameter tuning
print("\nHyperparameter tuning...")

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

best_score = float('inf')
best_params = {}

results = []

for cp_scale in param_grid['changepoint_prior_scale']:
    for seas_scale in param_grid['seasonality_prior_scale']:
        # Train model
        model = Prophet(
            changepoint_prior_scale=cp_scale,
            seasonality_prior_scale=seas_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        model.fit(train_df)
        
        # Predict
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        
        # Evaluate
        test_pred = forecast.iloc[train_size:]['yhat'].values
        mae = np.mean(np.abs(test_actual - test_pred))
        
        results.append({
            'changepoint_prior_scale': cp_scale,
            'seasonality_prior_scale': seas_scale,
            'mae': mae
        })
        
        if mae < best_score:
            best_score = mae
            best_params = {
                'changepoint_prior_scale': cp_scale,
                'seasonality_prior_scale': seas_scale
            }
        
        print(f"  cp={cp_scale:.3f}, seas={seas_scale:.2f} -> MAE={mae:.2f}")

print("\n" + "=" * 50)
print("BEST PARAMETERS")
print("=" * 50)
print(f"changepoint_prior_scale: {best_params['changepoint_prior_scale']}")
print(f"seasonality_prior_scale: {best_params['seasonality_prior_scale']}")
print(f"Best MAE: {best_score:.2f}")
print("=" * 50)

# %% Train final tuned model
model_tuned = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model_tuned.fit(train_df)

future_tuned = model_tuned.make_future_dataframe(periods=len(test_df))
forecast_tuned = model_tuned.predict(future_tuned)

# Evaluate tuned model
test_pred_tuned = forecast_tuned.iloc[train_size:]['yhat'].values
mae_tuned = np.mean(np.abs(test_actual - test_pred_tuned))
rmse_tuned = np.sqrt(np.mean((test_actual - test_pred_tuned) ** 2))
mape_tuned = np.mean(np.abs((test_actual - test_pred_tuned) / test_actual)) * 100

print("\n" + "=" * 50)
print("TUNED MODEL PERFORMANCE")
print("=" * 50)
print(f"MAE:   {mae_tuned:.2f}  (Basic: {mae:.2f}, Improvement: {((mae - mae_tuned) / mae * 100):.1f}%)")
print(f"RMSE:  {rmse_tuned:.2f}  (Basic: {rmse:.2f})")
print(f"MAPE:  {mape_tuned:.2f}%  (Basic: {mape:.2f}%)")
print("=" * 50)

# %% [markdown]
# ## 5. Adding Holidays
# 
# Tambahkan efek holiday untuk Indonesia.

# %% Define Indonesian holidays
holidays = pd.DataFrame({
    'holiday': 'lebaran',
    'ds': pd.to_datetime([
        '2021-05-13', '2021-05-14',
        '2022-05-02', '2022-05-03',
        '2023-04-22', '2023-04-23'
    ]),
    'lower_window': -2,  # 2 hari sebelum
    'upper_window': 1,   # 1 hari sesudah
})

# Add more holidays
holidays = pd.concat([
    holidays,
    pd.DataFrame({
        'holiday': 'natal',
        'ds': pd.to_datetime(['2021-12-25', '2022-12-25', '2023-12-25']),
        'lower_window': -1,
        'upper_window': 1,
    }),
    pd.DataFrame({
        'holiday': 'tahun_baru',
        'ds': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
        'lower_window': 0,
        'upper_window': 1,
    })
])

print("Holidays defined:")
print(holidays)

# %% Train model with holidays
model_holidays = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model_holidays.fit(train_df)

future_holidays = model_holidays.make_future_dataframe(periods=len(test_df))
forecast_holidays = model_holidays.predict(future_holidays)

print("✓ Model with holidays trained")

# %% [markdown]
# ## 6. Adding External Regressors
# 
# Tambahkan variabel eksternal seperti marketing spend atau temperature.

# %% Generate external regressor data
# Simulate marketing campaign data
np.random.seed(42)
marketing_spend = np.zeros(len(df))

# Campaign periods (random)
campaign_dates = [
    ('2021-06-01', '2021-06-15'),
    ('2021-11-20', '2021-12-10'),
    ('2022-06-01', '2022-06-15'),
    ('2022-11-20', '2022-12-10'),
    ('2023-06-01', '2023-06-15'),
    ('2023-11-20', '2023-12-10'),
]

for start, end in campaign_dates:
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    marketing_spend[mask] = np.random.uniform(1000, 5000, mask.sum())

df['marketing_spend'] = marketing_spend

print("External regressor added: marketing_spend")
print(f"Campaign days: {(marketing_spend > 0).sum()}")
print(f"Total spend: ${marketing_spend.sum():,.0f}")

# %% Visualize marketing spend
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(df['ds'], df['y'], linewidth=1)
axes[0].set_title('Time Series with Marketing Campaigns', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

axes[1].bar(df['ds'], df['marketing_spend'], width=1, alpha=0.7, color='orange')
axes[1].set_title('Marketing Spend', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Spend ($)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('marketing_campaigns.png', dpi=100, bbox_inches='tight')
print("✓ Marketing campaigns plot saved: marketing_campaigns.png")

# %% Train model with regressor
train_df_reg = df[:train_size].copy()
test_df_reg = df[train_size:].copy()

model_regressor = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add regressor
model_regressor.add_regressor('marketing_spend')

model_regressor.fit(train_df_reg)

# Make future dataframe with regressor
future_reg = model_regressor.make_future_dataframe(periods=len(test_df))
future_reg = future_reg.merge(df[['ds', 'marketing_spend']], on='ds', how='left')
future_reg['marketing_spend'].fillna(0, inplace=True)

forecast_reg = model_regressor.predict(future_reg)

print("✓ Model with regressor trained")

# Evaluate
test_pred_reg = forecast_reg.iloc[train_size:]['yhat'].values
mae_reg = np.mean(np.abs(test_actual - test_pred_reg))
rmse_reg = np.sqrt(np.mean((test_actual - test_pred_reg) ** 2))

print("\n" + "=" * 50)
print("MODEL WITH REGRESSOR PERFORMANCE")
print("=" * 50)
print(f"MAE:   {mae_reg:.2f}  (Tuned: {mae_tuned:.2f})")
print(f"RMSE:  {rmse_reg:.2f}  (Tuned: {rmse_tuned:.2f})")
print("=" * 50)

# %% [markdown]
# ## 7. Cross-Validation
# 
# Time series cross-validation untuk evaluasi yang lebih robust.

# %% Perform cross-validation
print("\nPerforming cross-validation...")
print("(This may take a few minutes...)")

try:
    from prophet.diagnostics import cross_validation, performance_metrics
    
    df_cv = cross_validation(
        model_tuned,
        initial='365 days',    # minimal 1 tahun training
        period='90 days',      # evaluate setiap 3 bulan
        horizon='180 days'     # forecast 6 bulan ke depan
    )
    
    df_metrics = performance_metrics(df_cv)
    
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION METRICS")
    print("=" * 50)
    print(df_metrics[['horizon', 'mape', 'mae', 'rmse']].head(10))
    print("=" * 50)
    
    # Plot CV metrics
    from prophet.plot import plot_cross_validation_metric
    
    fig = plot_cross_validation_metric(df_cv, metric='mape', figsize=(10, 6))
    plt.title('Cross-Validation: MAPE over Forecast Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cv_mape.png', dpi=100, bbox_inches='tight')
    print("✓ CV MAPE plot saved: cv_mape.png")
    
except Exception as e:
    print(f"⚠ Cross-validation skipped: {str(e)}")
    print("(Mungkin data terlalu pendek untuk CV dengan parameter ini)")

# %% [markdown]
# ## 8. Future Forecast
# 
# Predict untuk 90 hari ke depan setelah test period.

# %% Make future predictions
print("\nGenerating 90-day future forecast...")

# Retrain on full dataset
model_final = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model_final.fit(df)

# Predict 90 days ahead
future_final = model_final.make_future_dataframe(periods=90)
forecast_final = model_final.predict(future_final)

# Extract future predictions
future_predictions = forecast_final.iloc[len(df):].copy()

print(f"Future forecast: {len(future_predictions)} days")
print(f"Forecast range: {future_predictions['ds'].min().date()} to {future_predictions['ds'].max().date()}")
print(f"\nSample predictions:")
print(future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10))

# %% Visualize future forecast
fig, ax = plt.subplots(figsize=(14, 6))

# Historical data
ax.plot(df['ds'], df['y'], label='Historical Data', linewidth=1, alpha=0.7)

# Future forecast
ax.plot(future_predictions['ds'], future_predictions['yhat'], 
        label='Future Forecast', linewidth=2, color='red', linestyle='--')

# Prediction intervals
ax.fill_between(future_predictions['ds'], 
                future_predictions['yhat_lower'], 
                future_predictions['yhat_upper'],
                alpha=0.3, color='red', label='Prediction Interval')

ax.axvline(x=df['ds'].max(), color='green', linestyle=':', linewidth=2, label='Forecast Start')

ax.set_title('90-Day Future Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('future_forecast.png', dpi=100, bbox_inches='tight')
print("✓ Future forecast plot saved: future_forecast.png")

# %% [markdown]
# ## 9. Summary & Key Takeaways

# %% Print summary
print("\n" + "=" * 70)
print(" " * 20 + "SUMMARY & KEY TAKEAWAYS")
print("=" * 70)
print(f"""
Dataset:
  • {len(df)} days of historical data
  • Date range: {df['ds'].min().date()} to {df['ds'].max().date()}
  • Train/Test split: {len(train_df)}/{len(test_df)} days

Model Performance:
  • Basic Model MAE:  {mae:.2f}
  • Tuned Model MAE:  {mae_tuned:.2f} (improvement: {((mae - mae_tuned) / mae * 100):.1f}%)
  • Best Parameters:
    - changepoint_prior_scale: {best_params['changepoint_prior_scale']}
    - seasonality_prior_scale: {best_params['seasonality_prior_scale']}

Extensions:
  ✓ Holidays: Added Indonesian holidays (Lebaran, Natal, Tahun Baru)
  ✓ Regressors: Added marketing_spend as external variable
  ✓ Cross-Validation: Time series CV untuk robust evaluation

Future Forecast:
  • 90 days ahead dari {df['ds'].max().date()}
  • Expected range: {future_predictions['yhat'].min():.1f} - {future_predictions['yhat'].max():.1f}

Key Insights:
  1. Prophet effectively captures weekly dan yearly seasonality
  2. Hyperparameter tuning improved MAE by {((mae - mae_tuned) / mae * 100):.1f}%
  3. Uncertainty intervals grow wider untuk long-term forecasts
  4. External regressors bisa improve accuracy jika ada causal relationship

Next Steps:
  • Monitor forecast accuracy dengan incoming data
  • Retrain model secara berkala (weekly/monthly)
  • Experiment dengan additional regressors
  • Consider ensemble dengan models lain (ARIMA, XGBoost)
""")
print("=" * 70)

print("\n✓ All visualizations saved:")
print("  • time_series_data.png")
print("  • forecast_basic.png")
print("  • forecast_components.png")
print("  • forecast_evaluation.png")
print("  • marketing_campaigns.png")
print("  • future_forecast.png")
if 'df_cv' in locals():
    print("  • cv_mape.png")

print("\n🎉 Tutorial completed successfully!")
