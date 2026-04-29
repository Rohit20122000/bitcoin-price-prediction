"""
================================================================================
BITCOIN PRICE PREDICTION PROJECT - STEP 4: BUILDING PREDICTION MODELS
================================================================================
Author: Rohit Chaudhary
MSc Data Analytics Dissertation Project

WHAT WE'RE DOING IN THIS STEP:
------------------------------
1. ARIMA - Traditional time series model
2. Prophet - Facebook's forecasting tool
3. Random Forest - Ensemble machine learning
4. XGBoost - Gradient boosting (often wins competitions!)

We'll train each model, make predictions, and compare their performance.
================================================================================
"""

# =============================================================================
# IMPORTING LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

print("=" * 70)
print("STEP 4: BUILDING PREDICTION MODELS")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n📥 Loading feature-engineered data...")
df = pd.read_csv('bitcoin_features.csv', index_col='Date', parse_dates=True)
print(f"✅ Loaded {len(df)} days of data with {len(df.columns)} features")

# =============================================================================
# PREPARE DATA FOR ML MODELS
# =============================================================================
"""
DATA SPLIT EXPLANATION:
-----------------------
- Training Set (80%): Model learns patterns from this data
- Test Set (20%): We evaluate how well model predicts UNSEEN data

WHY THIS MATTERS:
- If model performs well on training but bad on test = OVERFITTING
- We want model to generalize to new, unseen data
"""

print("\n" + "=" * 70)
print("PREPARING DATA FOR MODELS")
print("=" * 70)

# Select features for ML models
feature_cols = [
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_7', 'Close_Lag_14',
    'SMA_7', 'SMA_21', 'SMA_50', 'EMA_7', 'EMA_21',
    'RSI_14', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'Volume', 'Volume_Lag_1',
    'Daily_Return', 'Return_Lag_1',
    'Momentum_7', 'Momentum_14',
    'Volatility_7', 'Volatility_30',
    'Day_of_Week', 'Month'
]

# Make sure all features exist
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df['Target_Price'].copy()

# Remove any remaining NaN values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

# Split data - 80% train, 20% test
# IMPORTANT: For time series, we don't shuffle! We keep time order
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n📊 Data Split:")
print(f"   • Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   • Testing samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"   • Features used:    {len(feature_cols)}")

# Scale features for ML models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results for comparison
results = {}

# =============================================================================
# MODEL 1: ARIMA (AutoRegressive Integrated Moving Average)
# =============================================================================
"""
WHAT IS ARIMA?
--------------
- AR (AutoRegressive): Uses past values to predict future
- I (Integrated): Differencing to make data stationary
- MA (Moving Average): Uses past forecast errors

PARAMETERS:
- p: Number of AR terms (lag order)
- d: Degree of differencing
- q: Number of MA terms

ARIMA(5,1,0) means:
- p=5: Use last 5 days
- d=1: Difference once to make stationary
- q=0: No MA terms
"""

print("\n" + "=" * 70)
print("MODEL 1: ARIMA (AutoRegressive Integrated Moving Average)")
print("=" * 70)

# Prepare data for ARIMA
close_prices = df['Close'].dropna()
train_size = int(len(close_prices) * 0.8)
train_arima = close_prices[:train_size]
test_arima = close_prices[train_size:]

print(f"\n🔧 Training ARIMA(5,1,0) model...")
print(f"   • Training on {len(train_arima)} data points")

# Fit ARIMA model
try:
    model_arima = ARIMA(train_arima, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    
    # Make predictions
    arima_predictions = model_arima_fit.forecast(steps=len(test_arima))
    
    # Calculate metrics
    arima_mae = mean_absolute_error(test_arima, arima_predictions)
    arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_predictions))
    arima_mape = np.mean(np.abs((test_arima.values - arima_predictions.values) / test_arima.values)) * 100
    
    results['ARIMA'] = {
        'MAE': arima_mae,
        'RMSE': arima_rmse,
        'MAPE': arima_mape,
        'predictions': arima_predictions,
        'actual': test_arima
    }
    
    print(f"\n✅ ARIMA Results:")
    print(f"   • MAE:  ${arima_mae:,.2f}")
    print(f"   • RMSE: ${arima_rmse:,.2f}")
    print(f"   • MAPE: {arima_mape:.2f}%")
    
except Exception as e:
    print(f"⚠️ ARIMA Error: {e}")
    results['ARIMA'] = None

# =============================================================================
# MODEL 2: PROPHET (Facebook's Forecasting Tool)
# =============================================================================
"""
WHAT IS PROPHET?
----------------
- Developed by Facebook for forecasting at scale
- Handles seasonality automatically
- Good for data with strong seasonal patterns
- Very easy to use!

WHY PROPHET FOR CRYPTO?
-----------------------
- Can capture weekly/monthly patterns
- Handles missing data well
- Provides uncertainty intervals
"""

print("\n" + "=" * 70)
print("MODEL 2: PROPHET (Facebook's Forecasting Tool)")
print("=" * 70)

# Prepare data for Prophet (needs specific column names: ds, y)
prophet_df = df[['Close']].reset_index()
prophet_df.columns = ['ds', 'y']
prophet_df = prophet_df.dropna()

train_prophet = prophet_df[:train_size]
test_prophet = prophet_df[train_size:]

print(f"\n🔧 Training Prophet model...")
print(f"   • Training on {len(train_prophet)} data points")

try:
    # Initialize and train Prophet
    model_prophet = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model_prophet.fit(train_prophet)
    
    # Create future dataframe for predictions
    future = test_prophet[['ds']]
    prophet_forecast = model_prophet.predict(future)
    prophet_predictions = prophet_forecast['yhat'].values
    
    # Calculate metrics
    prophet_mae = mean_absolute_error(test_prophet['y'], prophet_predictions)
    prophet_rmse = np.sqrt(mean_squared_error(test_prophet['y'], prophet_predictions))
    prophet_mape = np.mean(np.abs((test_prophet['y'].values - prophet_predictions) / test_prophet['y'].values)) * 100
    
    results['Prophet'] = {
        'MAE': prophet_mae,
        'RMSE': prophet_rmse,
        'MAPE': prophet_mape,
        'predictions': prophet_predictions,
        'actual': test_prophet['y'].values
    }
    
    print(f"\n✅ Prophet Results:")
    print(f"   • MAE:  ${prophet_mae:,.2f}")
    print(f"   • RMSE: ${prophet_rmse:,.2f}")
    print(f"   • MAPE: {prophet_mape:.2f}%")
    
except Exception as e:
    print(f"⚠️ Prophet Error: {e}")
    results['Prophet'] = None

# =============================================================================
# MODEL 3: RANDOM FOREST
# =============================================================================
"""
WHAT IS RANDOM FOREST?
----------------------
- Ensemble of many Decision Trees
- Each tree learns from a random subset of data
- Final prediction = average of all trees

WHY IT'S GOOD:
- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance

PARAMETERS:
- n_estimators: Number of trees (more = better, but slower)
- max_depth: How deep each tree can grow
- random_state: For reproducibility
"""

print("\n" + "=" * 70)
print("MODEL 3: RANDOM FOREST")
print("=" * 70)

print(f"\n🔧 Training Random Forest with 100 trees...")
print(f"   • Training on {len(X_train)} samples with {len(feature_cols)} features")

try:
    # Initialize and train Random Forest
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model_rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_predictions = model_rf.predict(X_test_scaled)
    
    # Calculate metrics
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_mape = np.mean(np.abs((y_test.values - rf_predictions) / y_test.values)) * 100
    rf_r2 = r2_score(y_test, rf_predictions)
    
    results['Random Forest'] = {
        'MAE': rf_mae,
        'RMSE': rf_rmse,
        'MAPE': rf_mape,
        'R2': rf_r2,
        'predictions': rf_predictions,
        'actual': y_test.values
    }
    
    print(f"\n✅ Random Forest Results:")
    print(f"   • MAE:  ${rf_mae:,.2f}")
    print(f"   • RMSE: ${rf_rmse:,.2f}")
    print(f"   • MAPE: {rf_mape:.2f}%")
    print(f"   • R²:   {rf_r2:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   📊 Top 5 Most Important Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"      • {row['feature']}: {row['importance']*100:.1f}%")
    
except Exception as e:
    print(f"⚠️ Random Forest Error: {e}")
    results['Random Forest'] = None

# =============================================================================
# MODEL 4: XGBOOST (Extreme Gradient Boosting)
# =============================================================================
"""
WHAT IS XGBOOST?
----------------
- Gradient Boosting: Trees are built sequentially
- Each new tree corrects errors from previous trees
- XGBoost = Optimized, fast implementation

WHY XGBOOST IS POWERFUL:
- Often wins ML competitions
- Handles missing values
- Built-in regularization (prevents overfitting)
- Very fast due to parallel processing

PARAMETERS:
- n_estimators: Number of boosting rounds
- learning_rate: How much each tree contributes (smaller = more conservative)
- max_depth: Depth of each tree
"""

print("\n" + "=" * 70)
print("MODEL 4: XGBOOST (Extreme Gradient Boosting)")
print("=" * 70)

print(f"\n🔧 Training XGBoost with 100 boosting rounds...")
print(f"   • Training on {len(X_train)} samples with {len(feature_cols)} features")

try:
    # Initialize and train XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_xgb.fit(X_train_scaled, y_train)
    
    # Make predictions
    xgb_predictions = model_xgb.predict(X_test_scaled)
    
    # Calculate metrics
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_mape = np.mean(np.abs((y_test.values - xgb_predictions) / y_test.values)) * 100
    xgb_r2 = r2_score(y_test, xgb_predictions)
    
    results['XGBoost'] = {
        'MAE': xgb_mae,
        'RMSE': xgb_rmse,
        'MAPE': xgb_mape,
        'R2': xgb_r2,
        'predictions': xgb_predictions,
        'actual': y_test.values
    }
    
    print(f"\n✅ XGBoost Results:")
    print(f"   • MAE:  ${xgb_mae:,.2f}")
    print(f"   • RMSE: ${xgb_rmse:,.2f}")
    print(f"   • MAPE: {xgb_mape:.2f}%")
    print(f"   • R²:   {xgb_r2:.4f}")
    
    # XGBoost Feature Importance
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   📊 Top 5 Most Important Features:")
    for i, row in xgb_importance.head(5).iterrows():
        print(f"      • {row['feature']}: {row['importance']*100:.1f}%")
    
except Exception as e:
    print(f"⚠️ XGBoost Error: {e}")
    results['XGBoost'] = None

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# Create comparison table
comparison_data = []
for model_name, model_results in results.items():
    if model_results is not None:
        comparison_data.append({
            'Model': model_name,
            'MAE ($)': f"${model_results['MAE']:,.2f}",
            'RMSE ($)': f"${model_results['RMSE']:,.2f}",
            'MAPE (%)': f"{model_results['MAPE']:.2f}%"
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 Model Performance Comparison:")
print("-" * 60)
print(comparison_df.to_string(index=False))
print("-" * 60)

# Find best model
best_model = min(results.items(), key=lambda x: x[1]['MAE'] if x[1] else float('inf'))
print(f"\n🏆 BEST MODEL: {best_model[0]} (Lowest MAE: ${best_model[1]['MAE']:,.2f})")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bitcoin Price Prediction - Model Comparison', fontsize=16, fontweight='bold')

# Get test dates for x-axis
test_dates = df.index[train_size:train_size + len(y_test)]

# Plot each model
plot_data = [
    ('ARIMA', 0, 0, 'blue'),
    ('Prophet', 0, 1, 'green'),
    ('Random Forest', 1, 0, 'orange'),
    ('XGBoost', 1, 1, 'red')
]

for model_name, row, col, color in plot_data:
    ax = axes[row, col]
    
    if results.get(model_name) is not None:
        actual = results[model_name]['actual']
        predicted = results[model_name]['predictions']
        
        # Handle different lengths
        plot_len = min(len(actual), len(predicted), len(test_dates))
        
        ax.plot(range(plot_len), actual[:plot_len], label='Actual', color='black', linewidth=1.5)
        ax.plot(range(plot_len), predicted[:plot_len], label='Predicted', color=color, linewidth=1.5, linestyle='--')
        
        ax.set_title(f'{model_name}\nMAE: ${results[model_name]["MAE"]:,.0f} | MAPE: {results[model_name]["MAPE"]:.1f}%', 
                     fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='upper left')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    else:
        ax.text(0.5, 0.5, f'{model_name}\nNot Available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(model_name)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: model_comparison.png")

# Create Model Performance Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))

models = []
maes = []
colors = ['blue', 'green', 'orange', 'red']

for model_name, model_results in results.items():
    if model_results is not None:
        models.append(model_name)
        maes.append(model_results['MAE'])

bars = ax.bar(models, maes, color=colors[:len(models)], edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
            f'${mae:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Mean Absolute Error (USD)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Model Performance Comparison - Mean Absolute Error (Lower is Better)', 
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Highlight best model
best_idx = maes.index(min(maes))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(4)

plt.tight_layout()
plt.savefig('model_performance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: model_performance_bar.png")

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'Date': test_dates[:len(y_test)],
    'Actual': y_test.values
})

for model_name, model_results in results.items():
    if model_results is not None:
        pred = model_results['predictions']
        predictions_df[f'{model_name}_Predicted'] = pred[:len(y_test)]

predictions_df.to_csv('model_predictions.csv', index=False)
print("💾 Saved: model_predictions.csv")

print("\n" + "=" * 70)
print("✅ STEP 4 COMPLETE!")
print("=" * 70)
print("""
WHAT YOU LEARNED:
-----------------
1. ARIMA - Traditional time series, good for trend capture
2. Prophet - Facebook's tool, handles seasonality well
3. Random Forest - Ensemble of trees, robust predictions
4. XGBoost - Gradient boosting, often best performance

KEY METRICS EXPLAINED:
----------------------
• MAE (Mean Absolute Error): Average prediction error in dollars
• RMSE (Root Mean Squared Error): Penalizes large errors more
• MAPE (Mean Absolute Percentage Error): Error as percentage
• R² (R-Squared): How much variance the model explains (0-1)

LOWER IS BETTER for MAE, RMSE, MAPE
HIGHER IS BETTER for R²

NEXT STEP:
----------
Step 5: Final Summary & GitHub README
- Create professional documentation
- Package everything for your portfolio
""")
