import yfinance as yf
import sqlite3
import pandas as pd
import time

# Database setup
db_name = "stocks.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER)''')
conn.commit()

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1m")
        if data.empty:
            print(f"No data found for {symbol}. Skipping...")
            return None  # Skip if no data is available
        latest = data.iloc[-1]  # Get the most recent price data
        return {
            "symbol": symbol,
            "open": latest["Open"],
            "high": latest["High"],
            "low": latest["Low"],
            "close": latest["Close"],
            "volume": latest["Volume"]
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to store data in SQLite
def store_data(symbol):
    stock_data = fetch_stock_data(symbol)
    if stock_data:  # Only store if data is available
        cursor.execute('''INSERT INTO stock_data (symbol, open, high, low, 
                        close, volume) VALUES (?, ?, ?, ?, ?, ?)''',
                       (stock_data["symbol"], stock_data["open"],
                        stock_data["high"], stock_data["low"],
                        stock_data["close"], stock_data["volume"]))
        conn.commit()
        print(f"Stored data for {symbol}")

# Function to analyze stock data
def analyze_stock(symbol):
    df = pd.read_sql_query("SELECT * FROM stock_data WHERE symbol=? ORDER BY timestamp DESC LIMIT 100",
                           conn, params=(symbol,))
    print(df)

# Example Usage
symbol = "GOOGL"  # Change this to any stock symbol
for _ in range(5):  # Fetch data 5 times with a 1-minute gap
    store_data(symbol)
    time.sleep(60)  # Wait for 1 minute before fetching again

analyze_stock(symbol)
import pandas as pd

df = pd.read_sql_query("SELECT * FROM stock_data", conn)
print(df)

# Close database connection
conn.close()
