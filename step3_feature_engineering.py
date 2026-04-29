"""
================================================================================
BITCOIN PRICE PREDICTION PROJECT - STEP 3: FEATURE ENGINEERING
================================================================================
Author: Rohit Chaudhary
MSc Data Analytics Dissertation Project

WHAT WE'RE DOING IN THIS STEP:
------------------------------
1. Create Moving Averages (SMA, EMA)
2. Create RSI (Relative Strength Index)
3. Create MACD (Moving Average Convergence Divergence)
4. Create Lag Features (for time series models)
5. Prepare data for ML models

WHY FEATURE ENGINEERING?
------------------------
- Raw price data alone isn't enough
- Technical indicators capture PATTERNS in the data
- ML models can learn from these patterns
- Better features = Better predictions
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

print("=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n📥 Loading processed Bitcoin data...")
df = pd.read_csv('bitcoin_data_processed.csv', index_col='Date', parse_dates=True)
print(f"✅ Loaded {len(df)} days of data")

# Keep original columns
original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# =============================================================================
# 1. MOVING AVERAGES (SMA & EMA)
# =============================================================================
"""
WHAT ARE MOVING AVERAGES?
-------------------------
- SMA (Simple Moving Average): Average of last N days
- EMA (Exponential Moving Average): Weighted average (recent days matter more)

WHY THEY MATTER:
----------------
- Smooth out daily noise
- Show the "trend" direction
- When price crosses MA = potential signal

FORMULA:
SMA(7) = (Price_today + Price_yesterday + ... + Price_7_days_ago) / 7
"""

print("\n" + "=" * 60)
print("1. CREATING MOVING AVERAGES")
print("=" * 60)

# Simple Moving Averages
df['SMA_7'] = df['Close'].rolling(window=7).mean()    # 1 week
df['SMA_21'] = df['Close'].rolling(window=21).mean()  # 3 weeks
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # ~2 months

# Exponential Moving Averages (more weight on recent prices)
df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

print("   ✓ SMA_7 (7-day Simple Moving Average)")
print("   ✓ SMA_21 (21-day Simple Moving Average)")
print("   ✓ SMA_50 (50-day Simple Moving Average)")
print("   ✓ EMA_7 (7-day Exponential Moving Average)")
print("   ✓ EMA_21 (21-day Exponential Moving Average)")

# Price relative to moving averages (normalized)
df['Price_to_SMA7'] = df['Close'] / df['SMA_7']
df['Price_to_SMA21'] = df['Close'] / df['SMA_21']

print("\n   💡 WHY RATIOS?")
print("   Price_to_SMA > 1 means price is ABOVE average (bullish)")
print("   Price_to_SMA < 1 means price is BELOW average (bearish)")

# =============================================================================
# 2. RSI (RELATIVE STRENGTH INDEX)
# =============================================================================
"""
WHAT IS RSI?
------------
- Measures how "overbought" or "oversold" an asset is
- Ranges from 0 to 100
- RSI > 70 = Overbought (might go down)
- RSI < 30 = Oversold (might go up)

FORMULA:
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
"""

print("\n" + "=" * 60)
print("2. CREATING RSI (RELATIVE STRENGTH INDEX)")
print("=" * 60)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

df['RSI_14'] = calculate_rsi(df['Close'], period=14)

print("   ✓ RSI_14 (14-day Relative Strength Index)")
print(f"\n   📊 Current RSI: {df['RSI_14'].iloc[-1]:.1f}")

if df['RSI_14'].iloc[-1] > 70:
    print("   ⚠️ RSI > 70: Market may be OVERBOUGHT")
elif df['RSI_14'].iloc[-1] < 30:
    print("   ⚠️ RSI < 30: Market may be OVERSOLD")
else:
    print("   ✓ RSI in neutral zone (30-70)")

# =============================================================================
# 3. MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
# =============================================================================
"""
WHAT IS MACD?
-------------
- Shows relationship between two moving averages
- MACD Line = EMA_12 - EMA_26
- Signal Line = EMA_9 of MACD Line
- Histogram = MACD Line - Signal Line

TRADING SIGNALS:
- MACD crosses above Signal = Bullish
- MACD crosses below Signal = Bearish
"""

print("\n" + "=" * 60)
print("3. CREATING MACD")
print("=" * 60)

# Calculate MACD components
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

print("   ✓ MACD (12-26 EMA difference)")
print("   ✓ MACD_Signal (9-day EMA of MACD)")
print("   ✓ MACD_Histogram (MACD - Signal)")

# =============================================================================
# 4. BOLLINGER BANDS
# =============================================================================
"""
WHAT ARE BOLLINGER BANDS?
-------------------------
- Upper Band = SMA + (2 * Standard Deviation)
- Lower Band = SMA - (2 * Standard Deviation)
- Measures volatility

INTERPRETATION:
- Price near Upper Band = Potentially overbought
- Price near Lower Band = Potentially oversold
- Bands expanding = High volatility
- Bands contracting = Low volatility
"""

print("\n" + "=" * 60)
print("4. CREATING BOLLINGER BANDS")
print("=" * 60)

period = 20
df['BB_Middle'] = df['Close'].rolling(window=period).mean()
df['BB_Std'] = df['Close'].rolling(window=period).std()
df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']  # Normalized width

print("   ✓ BB_Upper (Upper Bollinger Band)")
print("   ✓ BB_Lower (Lower Bollinger Band)")
print("   ✓ BB_Width (Band Width - volatility measure)")

# =============================================================================
# 5. LAG FEATURES (FOR TIME SERIES)
# =============================================================================
"""
WHAT ARE LAG FEATURES?
----------------------
- Previous day's values
- Allow model to "look back" in time
- Essential for time series prediction

EXAMPLE:
If we want to predict today's price, we use:
- Yesterday's price (lag_1)
- Day before yesterday's price (lag_2)
- Price from 7 days ago (lag_7)
"""

print("\n" + "=" * 60)
print("5. CREATING LAG FEATURES")
print("=" * 60)

# Price lags
for lag in [1, 2, 3, 7, 14]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    print(f"   ✓ Close_Lag_{lag} (Close price from {lag} day(s) ago)")

# Return lags
df['Return_Lag_1'] = df['Daily_Return'].shift(1)
df['Return_Lag_7'] = df['Daily_Return'].shift(7)
print("   ✓ Return_Lag_1 (Yesterday's return)")
print("   ✓ Return_Lag_7 (Return from 7 days ago)")

# Volume lags
df['Volume_Lag_1'] = df['Volume'].shift(1)
print("   ✓ Volume_Lag_1 (Yesterday's volume)")

# =============================================================================
# 6. ADDITIONAL FEATURES
# =============================================================================
"""
OTHER USEFUL FEATURES:
----------------------
- Day of week (weekends might behave differently)
- Month (seasonal patterns)
- Rolling statistics (mean, std over different windows)
"""

print("\n" + "=" * 60)
print("6. CREATING ADDITIONAL FEATURES")
print("=" * 60)

# Time-based features
df['Day_of_Week'] = df.index.dayofweek
df['Month'] = df.index.month
df['Quarter'] = df.index.quarter

# Rolling statistics
df['Rolling_Mean_7'] = df['Close'].rolling(window=7).mean()
df['Rolling_Std_7'] = df['Close'].rolling(window=7).std()
df['Rolling_Mean_30'] = df['Close'].rolling(window=30).mean()
df['Rolling_Std_30'] = df['Close'].rolling(window=30).std()

# Price momentum
df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
df['Momentum_14'] = df['Close'] - df['Close'].shift(14)

# Volatility
df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()
df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()

print("   ✓ Day_of_Week, Month, Quarter")
print("   ✓ Rolling_Mean_7, Rolling_Std_7")
print("   ✓ Rolling_Mean_30, Rolling_Std_30")
print("   ✓ Momentum_7, Momentum_14")
print("   ✓ Volatility_7, Volatility_30")

# =============================================================================
# 7. CREATE TARGET VARIABLE
# =============================================================================
"""
WHAT WE'RE PREDICTING:
----------------------
Option 1: Next day's price (regression)
Option 2: Price direction (up/down) (classification)

We'll create both targets so we can try different approaches.
"""

print("\n" + "=" * 60)
print("7. CREATING TARGET VARIABLES")
print("=" * 60)

# Target 1: Next day's close price (for regression)
df['Target_Price'] = df['Close'].shift(-1)

# Target 2: Next day's direction (for classification)
df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

print("   ✓ Target_Price (Tomorrow's close price - REGRESSION)")
print("   ✓ Target_Direction (1 if up, 0 if down - CLASSIFICATION)")

# =============================================================================
# 8. HANDLE MISSING VALUES
# =============================================================================

print("\n" + "=" * 60)
print("8. HANDLING MISSING VALUES")
print("=" * 60)

# Count missing values before
missing_before = df.isnull().sum().sum()
print(f"\n   Missing values before cleaning: {missing_before}")

# Remove rows with NaN (mostly at the beginning due to rolling calculations)
df_clean = df.dropna()
print(f"   Rows removed: {len(df) - len(df_clean)}")
print(f"   Remaining rows: {len(df_clean)}")

# =============================================================================
# 9. SUMMARY OF FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("9. FEATURE SUMMARY")
print("=" * 60)

# Categorize features
price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
ma_features = ['SMA_7', 'SMA_21', 'SMA_50', 'EMA_7', 'EMA_21', 'Price_to_SMA7', 'Price_to_SMA21']
momentum_features = ['RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Momentum_7', 'Momentum_14']
volatility_features = ['BB_Upper', 'BB_Lower', 'BB_Width', 'Volatility_7', 'Volatility_30']
lag_features = [col for col in df_clean.columns if 'Lag' in col]
time_features = ['Day_of_Week', 'Month', 'Quarter']

print(f"\n📊 Total Features Created: {len(df_clean.columns) - 2}")  # -2 for targets
print(f"\n   • Price Features: {len(price_features)}")
print(f"   • Moving Average Features: {len(ma_features)}")
print(f"   • Momentum Features: {len(momentum_features)}")
print(f"   • Volatility Features: {len(volatility_features)}")
print(f"   • Lag Features: {len(lag_features)}")
print(f"   • Time Features: {len(time_features)}")

# =============================================================================
# 10. VISUALIZE SOME FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("10. CREATING FEATURE VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Price with Moving Averages
ax1 = axes[0]
ax1.plot(df_clean.index[-200:], df_clean['Close'].iloc[-200:], label='Close', color='blue', linewidth=1.5)
ax1.plot(df_clean.index[-200:], df_clean['SMA_7'].iloc[-200:], label='SMA 7', color='orange', linewidth=1)
ax1.plot(df_clean.index[-200:], df_clean['SMA_21'].iloc[-200:], label='SMA 21', color='green', linewidth=1)
ax1.plot(df_clean.index[-200:], df_clean['SMA_50'].iloc[-200:], label='SMA 50', color='red', linewidth=1)
ax1.fill_between(df_clean.index[-200:], df_clean['BB_Upper'].iloc[-200:], 
                  df_clean['BB_Lower'].iloc[-200:], alpha=0.2, color='gray', label='Bollinger Bands')
ax1.set_title('Bitcoin Price with Moving Averages & Bollinger Bands (Last 200 Days)', fontweight='bold')
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 2: RSI
ax2 = axes[1]
ax2.plot(df_clean.index[-200:], df_clean['RSI_14'].iloc[-200:], color='purple', linewidth=1.5)
ax2.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
ax2.fill_between(df_clean.index[-200:], 30, 70, alpha=0.1, color='gray')
ax2.set_title('RSI (Relative Strength Index)', fontweight='bold')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left')

# Plot 3: MACD
ax3 = axes[2]
ax3.plot(df_clean.index[-200:], df_clean['MACD'].iloc[-200:], label='MACD', color='blue', linewidth=1.5)
ax3.plot(df_clean.index[-200:], df_clean['MACD_Signal'].iloc[-200:], label='Signal', color='orange', linewidth=1)
ax3.bar(df_clean.index[-200:], df_clean['MACD_Histogram'].iloc[-200:], 
        color=['green' if x > 0 else 'red' for x in df_clean['MACD_Histogram'].iloc[-200:]], 
        alpha=0.5, width=1)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
ax3.set_ylabel('MACD')
ax3.legend(loc='upper left')

plt.tight_layout()
plt.savefig('feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: feature_engineering.png")

# =============================================================================
# 11. SAVE PROCESSED DATA
# =============================================================================

df_clean.to_csv('bitcoin_features.csv')
print("\n" + "=" * 60)
print("💾 Feature data saved to 'bitcoin_features.csv'")
print("=" * 60)

print("\n" + "=" * 60)
print("✅ STEP 3 COMPLETE!")
print("=" * 60)
print("""
WHAT YOU LEARNED:
-----------------
1. Moving Averages (SMA, EMA) - Show price trends
2. RSI - Shows if market is overbought/oversold
3. MACD - Momentum indicator for buy/sell signals
4. Bollinger Bands - Volatility measurement
5. Lag Features - Allow models to "look back" in time
6. Target Variables - What we're predicting

TOTAL FEATURES CREATED: 40+
---------------------------
These features capture:
• Trend (Moving Averages)
• Momentum (RSI, MACD)
• Volatility (Bollinger Bands, Rolling Std)
• Historical Patterns (Lag Features)

NEXT STEP:
----------
Step 4: Building Prediction Models
- ARIMA (Time Series)
- Prophet (Facebook's Forecasting)
- Random Forest (ML)
- XGBoost (Gradient Boosting)
""")
