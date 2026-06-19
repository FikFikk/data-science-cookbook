"""
Gradient Boosting Demo: Perbandingan XGBoost vs LightGBM vs CatBoost

Dataset: California Housing (Regression)
Goal: Predict median house value
Comparison: Performance, speed, dan feature importance
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GRADIENT BOOSTING COMPARISON: XGBoost vs LightGBM vs CatBoost")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1] Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

print(f"    Dataset shape: {X.shape}")
print(f"    Target (median house value): min=${y.min():.2f}, max=${y.max():.2f}, mean=${y.mean():.2f}")
print(f"    Features: {list(X.columns)}")
print()

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
print("[2] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {X_train.shape[0]} samples")
print(f"    Test:  {X_test.shape[0]} samples")
print()

# ============================================================================
# 3. XGBOOST
# ============================================================================
print("[3] Training XGBoost...")
try:
    import xgboost as xgb
    
    start_time = time.time()
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    train_time_xgb = time.time() - start_time
    
    # Predictions
    y_pred_xgb = model_xgb.predict(X_test)
    
    # Metrics
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print(f"    ✓ Training time: {train_time_xgb:.2f}s")
    print(f"    ✓ RMSE: {rmse_xgb:.4f}")
    print(f"    ✓ MAE:  {mae_xgb:.4f}")
    print(f"    ✓ R²:   {r2_xgb:.4f}")
    
    xgb_available = True
    
except ImportError:
    print("    ✗ XGBoost not installed (pip install xgboost)")
    xgb_available = False
    model_xgb = None
    rmse_xgb = mae_xgb = r2_xgb = train_time_xgb = None

print()

# ============================================================================
# 4. LIGHTGBM
# ============================================================================
print("[4] Training LightGBM...")
try:
    import lightgbm as lgb
    
    start_time = time.time()
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    train_time_lgb = time.time() - start_time
    
    # Predictions
    y_pred_lgb = model_lgb.predict(X_test)
    
    # Metrics
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    
    print(f"    ✓ Training time: {train_time_lgb:.2f}s")
    print(f"    ✓ RMSE: {rmse_lgb:.4f}")
    print(f"    ✓ MAE:  {mae_lgb:.4f}")
    print(f"    ✓ R²:   {r2_lgb:.4f}")
    
    lgb_available = True
    
except ImportError:
    print("    ✗ LightGBM not installed (pip install lightgbm)")
    lgb_available = False
    model_lgb = None
    rmse_lgb = mae_lgb = r2_lgb = train_time_lgb = None

print()

# ============================================================================
# 5. CATBOOST
# ============================================================================
print("[5] Training CatBoost...")
try:
    from catboost import CatBoostRegressor
    
    start_time = time.time()
    
    model_cat = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=False
    )
    
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    train_time_cat = time.time() - start_time
    
    # Predictions
    y_pred_cat = model_cat.predict(X_test)
    
    # Metrics
    rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
    mae_cat = mean_absolute_error(y_test, y_pred_cat)
    r2_cat = r2_score(y_test, y_pred_cat)
    
    print(f"    ✓ Training time: {train_time_cat:.2f}s")
    print(f"    ✓ RMSE: {rmse_cat:.4f}")
    print(f"    ✓ MAE:  {mae_cat:.4f}")
    print(f"    ✓ R²:   {r2_cat:.4f}")
    
    cat_available = True
    
except ImportError:
    print("    ✗ CatBoost not installed (pip install catboost)")
    cat_available = False
    model_cat = None
    rmse_cat = mae_cat = r2_cat = train_time_cat = None

print()

# ============================================================================
# 6. COMPARISON SUMMARY
# ============================================================================
print("="*80)
print("COMPARISON SUMMARY")
print("="*80)
print()

if any([xgb_available, lgb_available, cat_available]):
    # Create comparison DataFrame
    results = []
    
    if xgb_available:
        results.append({
            'Model': 'XGBoost',
            'RMSE': rmse_xgb,
            'MAE': mae_xgb,
            'R²': r2_xgb,
            'Training Time (s)': train_time_xgb
        })
    
    if lgb_available:
        results.append({
            'Model': 'LightGBM',
            'RMSE': rmse_lgb,
            'MAE': mae_lgb,
            'R²': r2_lgb,
            'Training Time (s)': train_time_lgb
        })
    
    if cat_available:
        results.append({
            'Model': 'CatBoost',
            'RMSE': rmse_cat,
            'MAE': mae_cat,
            'R²': r2_cat,
            'Training Time (s)': train_time_cat
        })
    
    df_results = pd.DataFrame(results)
    
    # Find best model for each metric
    best_rmse = df_results.loc[df_results['RMSE'].idxmin(), 'Model']
    best_r2 = df_results.loc[df_results['R²'].idxmax(), 'Model']
    best_speed = df_results.loc[df_results['Training Time (s)'].idxmin(), 'Model']
    
    print(df_results.to_string(index=False))
    print()
    print(f"🏆 Best RMSE (accuracy):  {best_rmse}")
    print(f"🏆 Best R² (variance):    {best_r2}")
    print(f"🏆 Fastest training:      {best_speed}")
    print()
    
    # ========================================================================
    # 7. FEATURE IMPORTANCE
    # ========================================================================
    print("="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    print()
    
    if xgb_available and model_xgb is not None:
        print("[XGBoost] Top 5 Most Important Features:")
        importance_xgb = pd.DataFrame({
            'feature': X.columns,
            'importance': model_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_xgb.head(5).iterrows():
            print(f"    {row['feature']:20s}: {row['importance']:.4f}")
        print()
    
    if lgb_available and model_lgb is not None:
        print("[LightGBM] Top 5 Most Important Features:")
        importance_lgb = pd.DataFrame({
            'feature': X.columns,
            'importance': model_lgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_lgb.head(5).iterrows():
            print(f"    {row['feature']:20s}: {row['importance']:.4f}")
        print()
    
    if cat_available and model_cat is not None:
        print("[CatBoost] Top 5 Most Important Features:")
        importance_cat = pd.DataFrame({
            'feature': X.columns,
            'importance': model_cat.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_cat.head(5).iterrows():
            print(f"    {row['feature']:20s}: {row['importance']:.4f}")
        print()
    
    # ========================================================================
    # 8. PREDICTION EXAMPLES
    # ========================================================================
    print("="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    print()
    
    # Show first 5 test samples
    sample_idx = [0, 1, 2, 3, 4]
    
    print("Sample | Actual | XGBoost | LightGBM | CatBoost")
    print("-" * 60)
    
    for i in sample_idx:
        actual = y_test.iloc[i]
        
        pred_xgb = y_pred_xgb[i] if xgb_available else 0
        pred_lgb = y_pred_lgb[i] if lgb_available else 0
        pred_cat = y_pred_cat[i] if cat_available else 0
        
        print(f"{i:6d} | {actual:6.3f} | {pred_xgb:7.3f} | {pred_lgb:8.3f} | {pred_cat:8.3f}")
    
    print()

else:
    print("❌ No gradient boosting library available!")
    print("   Install at least one: pip install xgboost lightgbm catboost")
    print()

# ============================================================================
# 9. KEY TAKEAWAYS
# ============================================================================
print("="*80)
print("KEY TAKEAWAYS")
print("="*80)
print()
print("1. Gradient Boosting sangat powerful untuk data tabular")
print("2. LightGBM typically fastest, XGBoost most mature, CatBoost easiest to tune")
print("3. Try all three dan ensemble untuk best results")
print("4. Feature engineering masih penting meskipun algorithm powerful")
print("5. Always use early stopping untuk prevent overfitting")
print()
print("Next steps:")
print("- Hyperparameter tuning dengan Optuna")
print("- Feature engineering (interaction, domain knowledge)")
print("- Ensemble (stack/blend ketiga models)")
print("- SHAP analysis untuk interpretability")
print()
print("="*80)
