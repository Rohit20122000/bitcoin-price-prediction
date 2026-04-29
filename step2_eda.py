"""
================================================================================
BITCOIN PRICE PREDICTION PROJECT - STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
Author: Rohit Chaudhary
MSc Data Analytics Dissertation Project

WHAT WE'RE DOING IN THIS STEP:
------------------------------
1. Visualize Bitcoin price trends over time
2. Analyze daily returns and volatility
3. Create correlation heatmap
4. Understand data distribution

WHY EDA IS IMPORTANT?
---------------------
- Helps us understand patterns in the data
- Identifies relationships between variables
- Guides feature engineering decisions
- Required before building any ML model
================================================================================
"""

# =============================================================================
# IMPORTING LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n📥 Loading Bitcoin data...")
btc_data = pd.read_csv('bitcoin_data.csv', index_col='Date', parse_dates=True)
print(f"✅ Loaded {len(btc_data)} days of data")

# =============================================================================
# 1. BITCOIN PRICE TREND OVER TIME
# =============================================================================
"""
WHAT THIS SHOWS:
----------------
- Overall price movement from 2019-2024
- Bull markets (uptrends) and bear markets (downtrends)
- The volatility we mentioned earlier is visible here
"""

print("\n" + "=" * 60)
print("1. CREATING PRICE TREND VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bitcoin Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Close Price Over Time
ax1 = axes[0, 0]
ax1.plot(btc_data.index, btc_data['Close'], color='#2E86AB', linewidth=1)
ax1.fill_between(btc_data.index, btc_data['Close'], alpha=0.3, color='#2E86AB')
ax1.set_title('Bitcoin Close Price (2019-2024)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Add annotations for key events
max_price = btc_data['Close'].max()
max_date = btc_data['Close'].idxmax()
ax1.annotate(f'Peak: ${max_price:,.0f}', xy=(max_date, max_price), 
             xytext=(max_date, max_price*1.1),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='red'))

print("   ✓ Price trend chart created")

# =============================================================================
# 2. TRADING VOLUME ANALYSIS
# =============================================================================
"""
WHAT THIS SHOWS:
----------------
- Trading activity over time
- Volume spikes often correlate with price movements
- Higher volume = more market activity/interest
"""

# Plot 2: Volume Over Time
ax2 = axes[0, 1]
ax2.bar(btc_data.index, btc_data['Volume'], color='#A23B72', alpha=0.7, width=1)
ax2.set_title('Daily Trading Volume', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Volume (USD)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B'))

print("   ✓ Volume analysis chart created")

# =============================================================================
# 3. DAILY RETURNS DISTRIBUTION
# =============================================================================
"""
WHAT THIS SHOWS:
----------------
- Distribution of daily price changes
- Normal distribution = easier to predict
- Fat tails = extreme events happen more often (risky!)

HOW TO CALCULATE:
Daily Return = (Today's Price - Yesterday's Price) / Yesterday's Price
"""

# Calculate daily returns
btc_data['Daily_Return'] = btc_data['Close'].pct_change() * 100  # Percentage

# Plot 3: Returns Distribution
ax3 = axes[1, 0]
btc_data['Daily_Return'].hist(bins=50, ax=ax3, color='#F18F01', edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
ax3.axvline(x=btc_data['Daily_Return'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Mean: {btc_data["Daily_Return"].mean():.2f}%')
ax3.set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.legend()

print("   ✓ Returns distribution chart created")

# =============================================================================
# 4. PRICE CANDLESTICK-STYLE (HIGH-LOW RANGE)
# =============================================================================
"""
WHAT THIS SHOWS:
----------------
- Daily price range (High - Low)
- Larger ranges = more volatility that day
"""

# Calculate daily range
btc_data['Daily_Range'] = (btc_data['High'] - btc_data['Low']) / btc_data['Close'] * 100

# Plot 4: Rolling Volatility
rolling_vol = btc_data['Daily_Return'].rolling(window=30).std()
ax4 = axes[1, 1]
ax4.plot(btc_data.index, rolling_vol, color='#C73E1D', linewidth=1)
ax4.fill_between(btc_data.index, rolling_vol, alpha=0.3, color='#C73E1D')
ax4.set_title('30-Day Rolling Volatility', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Volatility (Std Dev %)')

print("   ✓ Volatility analysis chart created")

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n💾 Saved: eda_analysis.png")

# =============================================================================
# 5. CORRELATION HEATMAP
# =============================================================================
"""
WHAT THIS SHOWS:
----------------
- Relationships between different variables
- Values close to 1: Strong positive correlation (move together)
- Values close to -1: Strong negative correlation (move opposite)
- Values close to 0: No relationship

WHY IT MATTERS:
---------------
- Helps identify which features are useful for prediction
- Highly correlated features might be redundant
- This is what your dissertation discussed!
"""

print("\n" + "=" * 60)
print("2. CREATING CORRELATION HEATMAP")
print("=" * 60)

# Create additional features for correlation analysis
btc_data['Price_Change'] = btc_data['Close'].diff()
btc_data['Volume_Change'] = btc_data['Volume'].pct_change() * 100
btc_data['High_Low_Diff'] = btc_data['High'] - btc_data['Low']
btc_data['Open_Close_Diff'] = btc_data['Close'] - btc_data['Open']

# Select columns for correlation
corr_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Daily_Range']
correlation_matrix = btc_data[corr_columns].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Upper triangle mask
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdYlBu_r',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            mask=mask,
            ax=ax,
            annot_kws={'size': 11})
ax.set_title('Correlation Matrix - Bitcoin Features', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: correlation_heatmap.png")

# Print correlation insights
print("\n📊 KEY CORRELATION INSIGHTS:")
print("-" * 40)
print(f"   • Open-Close correlation:    {correlation_matrix.loc['Open', 'Close']:.3f}")
print(f"   • High-Low correlation:      {correlation_matrix.loc['High', 'Low']:.3f}")
print(f"   • Volume-Close correlation:  {correlation_matrix.loc['Volume', 'Close']:.3f}")
print(f"   • Returns-Volume correlation: {correlation_matrix.loc['Daily_Return', 'Volume']:.3f}")

print("\n   💡 INTERPRETATION:")
print("   - Open, High, Low, Close are highly correlated (0.99+)")
print("     This is expected - they're all price measurements!")
print("   - Volume has lower correlation with price")
print("     This means volume adds NEW information for prediction")

# =============================================================================
# 6. STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("3. STATISTICAL SUMMARY")
print("=" * 60)

print("\n📈 Daily Returns Statistics:")
print(f"   • Mean daily return:     {btc_data['Daily_Return'].mean():.3f}%")
print(f"   • Std deviation:         {btc_data['Daily_Return'].std():.3f}%")
print(f"   • Worst day:             {btc_data['Daily_Return'].min():.2f}%")
print(f"   • Best day:              {btc_data['Daily_Return'].max():.2f}%")
print(f"   • Skewness:              {btc_data['Daily_Return'].skew():.3f}")

# Count positive vs negative days
positive_days = (btc_data['Daily_Return'] > 0).sum()
negative_days = (btc_data['Daily_Return'] < 0).sum()
total_days = len(btc_data['Daily_Return'].dropna())

print(f"\n📅 Market Direction:")
print(f"   • Positive days: {positive_days} ({positive_days/total_days*100:.1f}%)")
print(f"   • Negative days: {negative_days} ({negative_days/total_days*100:.1f}%)")

# =============================================================================
# 7. YEARLY PERFORMANCE SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("4. YEARLY PERFORMANCE SUMMARY")
print("=" * 60)

btc_data['Year'] = btc_data.index.year
yearly_stats = btc_data.groupby('Year').agg({
    'Close': ['first', 'last', 'min', 'max'],
    'Daily_Return': 'mean',
    'Volume': 'mean'
}).round(2)

print("\n📊 Bitcoin Performance by Year:")
print("-" * 60)
for year in btc_data['Year'].unique():
    year_data = btc_data[btc_data['Year'] == year]
    start_price = year_data['Close'].iloc[0]
    end_price = year_data['Close'].iloc[-1]
    yearly_return = ((end_price - start_price) / start_price) * 100
    
    trend = "📈" if yearly_return > 0 else "📉"
    print(f"   {year}: ${start_price:>10,.0f} → ${end_price:>10,.0f}  ({yearly_return:>+7.1f}%) {trend}")

# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

btc_data.to_csv('bitcoin_data_processed.csv')
print("\n" + "=" * 60)
print("💾 Processed data saved to 'bitcoin_data_processed.csv'")
print("=" * 60)

print("\n" + "=" * 60)
print("✅ STEP 2 COMPLETE!")
print("=" * 60)
print("""
WHAT YOU LEARNED:
-----------------
1. Price Trend Analysis - Bitcoin went through bull/bear cycles
2. Volume Analysis - Higher volume often means big price moves
3. Returns Distribution - Bitcoin has fat tails (extreme events)
4. Correlation Matrix - Price features are correlated, Volume adds info
5. Yearly Performance - Bitcoin can go up 300%+ or down 60%+ in a year!

KEY INSIGHTS FOR MODELING:
--------------------------
• High volatility = Challenging to predict
• Strong correlation between OHLC = May need only Close price
• Volume provides additional signal
• Need models that handle non-linear patterns

NEXT STEP:
----------
Step 3: Feature Engineering
- Create technical indicators (Moving Averages, RSI, MACD)
- Build lag features for time series
- Prepare data for ML models
""")
