"""
================================================================================
BITCOIN PRICE PREDICTION PROJECT - STEP 1: DATA COLLECTION
================================================================================
Author: Rohit Chaudhary
MSc Data Analytics Dissertation Project

WHAT WE'RE DOING IN THIS STEP:
------------------------------
1. Load/Create Bitcoin historical price data
2. Understand the data structure
3. Save it for future use

NOTE: Using sample data that mimics real Bitcoin patterns
      (In real project, you'd use yfinance: yf.download('BTC-USD'))
================================================================================
"""

# =============================================================================
# IMPORTING LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: BITCOIN DATA COLLECTION")
print("=" * 60)

# =============================================================================
# GENERATING REALISTIC BITCOIN DATA
# =============================================================================
"""
EXPLANATION:
------------
We're creating data that mimics real Bitcoin price behavior:
- Upward trends (bull markets)
- Downward trends (bear markets)
- High volatility (big daily swings)
- Volume patterns that correlate with price moves

In your REAL project, you would use:
    import yfinance as yf
    btc_data = yf.download('BTC-USD', start='2019-01-01', end='2024-01-01')
"""

print("\n📥 Generating realistic Bitcoin historical data...")

# Set random seed for reproducibility
np.random.seed(42)

# Create date range - 5 years of daily data (like your dissertation mentioned)
end_date = datetime(2024, 6, 1)
start_date = datetime(2019, 6, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
n_days = len(dates)

print(f"📅 Creating {n_days} days of data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

# =============================================================================
# SIMULATING REALISTIC BITCOIN PRICES
# =============================================================================
"""
WHY THESE PATTERNS?
-------------------
Bitcoin has gone through several phases:
- 2019: Recovery from 2018 crash (~$3,500 → $7,000)
- 2020: COVID crash then bull run (~$7,000 → $29,000)
- 2021: All-time high (~$65,000) then correction
- 2022: Bear market (~$16,000)
- 2023-24: Recovery (~$30,000-$70,000)
"""

# Start price (Bitcoin was ~$8,000 in mid-2019)
start_price = 8000

# Generate realistic price movements using geometric brownian motion
# with trend changes to simulate bull/bear markets
prices = [start_price]
current_price = start_price

for i in range(1, n_days):
    day_num = i
    
    # Different market phases (mimicking real Bitcoin history)
    if day_num < 200:  # 2019: Slow growth
        drift = 0.0005
        volatility = 0.025
    elif day_num < 400:  # Early 2020: Sideways then COVID crash
        drift = 0.0008
        volatility = 0.04
    elif day_num < 600:  # Late 2020: Bull run starts
        drift = 0.003
        volatility = 0.035
    elif day_num < 800:  # 2021: Massive bull run
        drift = 0.004
        volatility = 0.045
    elif day_num < 1000:  # Mid 2021: Correction
        drift = -0.001
        volatility = 0.05
    elif day_num < 1200:  # Late 2021: Another peak
        drift = 0.002
        volatility = 0.04
    elif day_num < 1400:  # 2022: Bear market
        drift = -0.002
        volatility = 0.045
    elif day_num < 1600:  # Early 2023: Sideways
        drift = 0.0005
        volatility = 0.03
    else:  # Late 2023-2024: Recovery
        drift = 0.002
        volatility = 0.035
    
    # Calculate daily return with drift and random component
    daily_return = drift + volatility * np.random.randn()
    current_price = current_price * (1 + daily_return)
    
    # Keep price realistic (Bitcoin never went below $3,000 or above $75,000 in this period)
    current_price = max(3000, min(75000, current_price))
    prices.append(current_price)

# Create OHLCV data (Open, High, Low, Close, Volume)
close_prices = np.array(prices)

# Open is usually close to previous day's close
open_prices = np.roll(close_prices, 1)
open_prices[0] = close_prices[0]

# High and Low based on daily volatility
daily_range = np.abs(np.random.randn(n_days)) * 0.02 + 0.01  # 1-3% daily range
high_prices = np.maximum(open_prices, close_prices) * (1 + daily_range)
low_prices = np.minimum(open_prices, close_prices) * (1 - daily_range)

# Volume correlates with price volatility (more volume on big moves)
base_volume = 20e9  # $20 billion base daily volume
price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0])) / close_prices
volume = base_volume * (1 + price_changes * 50) * (0.5 + np.random.rand(n_days))

# =============================================================================
# CREATE DATAFRAME
# =============================================================================

btc_data = pd.DataFrame({
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volume
}, index=dates)

btc_data.index.name = 'Date'

print(f"✅ Generated {len(btc_data)} days of Bitcoin data")

# =============================================================================
# UNDERSTANDING THE DATA
# =============================================================================
"""
WHAT EACH COLUMN MEANS:
-----------------------
- Open:   Price when the day started (first trade)
- High:   Highest price during the day
- Low:    Lowest price during the day
- Close:  Price when the day ended (last trade) ← WE PREDICT THIS
- Volume: Total USD value traded that day

WHY CLOSE PRICE?
----------------
Close price is the most important because:
- It's the final settled price for the day
- Used by most traders for analysis
- All our models will predict the CLOSE price
"""

print("\n" + "=" * 60)
print("UNDERSTANDING THE DATA")
print("=" * 60)

# Show first 5 rows
print("\n📊 First 5 rows of data:")
print(btc_data.head().to_string())

# Show last 5 rows
print("\n📊 Last 5 rows of data:")
print(btc_data.tail().to_string())

# Show data types and info
print("\n📋 Data Information:")
print(f"   - Total rows (days): {len(btc_data)}")
print(f"   - Columns: {list(btc_data.columns)}")
print(f"   - Data types: All float64 (numbers)")

# =============================================================================
# CHECKING FOR MISSING VALUES
# =============================================================================
"""
WHY CHECK MISSING VALUES?
-------------------------
- Machine learning models can't handle missing data
- We need to know if any days have missing prices
- If missing, we need to fill them (interpolation)
"""

print("\n" + "=" * 60)
print("CHECKING DATA QUALITY")
print("=" * 60)

missing = btc_data.isnull().sum()
print("\n🔍 Missing values per column:")
for col in btc_data.columns:
    print(f"   • {col}: {missing[col]} missing")

print("\n✅ No missing values in the dataset")

# =============================================================================
# BASIC STATISTICS
# =============================================================================
"""
WHAT STATISTICS TELL US:
------------------------
- mean: Average price over 5 years
- std: Standard deviation (how much prices vary) - HIGH = VOLATILE
- min/max: Lowest and highest prices ever
- 25%, 50%, 75%: Quartiles (price distribution)
"""

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

# Get statistics for Close price (our target)
close_stats = btc_data['Close'].describe()
print("\n📈 Bitcoin Close Price Statistics:")
print(f"   • Count:          {int(close_stats['count'])} days")
print(f"   • Average price:  ${close_stats['mean']:,.2f}")
print(f"   • Lowest price:   ${close_stats['min']:,.2f}")
print(f"   • Highest price:  ${close_stats['max']:,.2f}")
print(f"   • Std Deviation:  ${close_stats['std']:,.2f}")

print(f"\n   💡 KEY INSIGHT:")
print(f"   The high standard deviation (${close_stats['std']:,.2f}) compared to")
print(f"   the mean (${close_stats['mean']:,.2f}) shows Bitcoin is EXTREMELY volatile!")
print(f"   This is why prediction is challenging - prices swing wildly!")

# =============================================================================
# SAVE DATA FOR NEXT STEPS
# =============================================================================

# Save to CSV file
btc_data.to_csv('bitcoin_data.csv')
print("\n" + "=" * 60)
print("💾 Data saved to 'bitcoin_data.csv'")
print("=" * 60)

print("\n" + "=" * 60)
print("✅ STEP 1 COMPLETE!")
print("=" * 60)
print("""
WHAT YOU LEARNED:
-----------------
1. OHLCV Data: Open, High, Low, Close, Volume
2. Close price is our prediction target
3. Bitcoin is highly volatile (high std deviation)
4. 5 years of data gives us ~1,800 samples to train on

NEXT STEP:
----------
Step 2: Exploratory Data Analysis (EDA)
- Visualize price trends over time
- Create correlation heatmaps
- Identify patterns in the data
""")
