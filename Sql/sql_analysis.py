import sqlite3
import pandas as pd
import os

# Create output folder if it doesn't exist
os.makedirs('Outputs', exist_ok=True)
os.makedirs('dashboard', exist_ok=True)

# Connect to your SQLite database
conn = sqlite3.connect('Data/flights.db')

# Quick test — count total rows
total = pd.read_sql("SELECT COUNT(*) as total FROM flights", conn)
print(f"Connected! Total flights: {total['total'][0]:,}")

print("="*50)
print("FLIGHT DELAY SQL ANALYSIS")
print("="*50)