# cleaning/clean_and_engineer.py
# Loads raw data, cleans it, engineers new features,
# saves flights_clean.csv for EDA and ML

import pandas as pd
import numpy as np
import sqlite3
import os

# Load data from your SQLite database into a DataFrame
print("Loading data from database...")
conn = sqlite3.connect('Data/flights.db')
df   = pd.read_sql("SELECT * FROM flights", conn)
conn.close()

print(f"Loaded: {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")


# ── STEP 2: Find missing values ───────────────────────────
# isnull() marks every missing cell as True
# .sum() counts how many True values per column
# We only show columns that actually have missing values

print("\\n--- Missing Value Report ---")
missing = df.isnull().sum()
missing = missing[missing > 0]    # only show columns with missing values

if len(missing) == 0:
    print("No missing values found!")
else:
    for col ,count in missing.items():
        pct = count / len(df) * 100    
        print(f"  {col:25s}: {count:>6,} missing ({pct:.1f}%)")

print(f"\nTotal rows before cleaning {len(df):,}")


# ── STEP 3: Remove cancelled flights ──────────────────────
# cancelled = 1 means the flight never flew
# We only want flights that actually departed and arrived
# .copy() creates a new separate DataFrame — never modify original

before = len(df)
df_clean = df[df['cancelled'] == 0].copy()
removed = before - len(df_clean)

print(f"\\n--- Step 3: Remove Cancelled Flights ---")
print(f"Rows before : {before:,}")
print(f"Rows removed: {removed:,} (cancelled flights)")
print(f"Rows after: {len(df_clean):,}")

# Verify no more missing values in key columns
print(f"Null arr_delay after removal: {df_clean['arr_delay'].isnull().sum()}")


# ── STEP 4: Fix data types ────────────────────────────────
# date column: convert from text to proper datetime
# is_delayed:  convert from float (1.0) to int (1)
# cancelled:   convert to int
# diverted:    convert to int

print("\\n--- Step 4: Fix Data Types ---")

df_clean['date'] = pd.to_datetime(df_clean['date'])
df_clean['is_delayed']= df_clean['is_delayed'].astype(int)
df_clean['cancelled']  = df_clean['cancelled'].astype(int)
df_clean['diverted'] = df_clean['diverted'].astype(int)

# Verify the types are now correct
print(df_clean[['date','is_delayed','cancelled','diverted']].dtypes)
print("Data types fixed!")

# ── STEP 5: Handle outliers ───────────────────────────────
# Check current min and max before clipping
print("\\n--- Step 5: Handle Outliers ---")
print("Before clipping:")
print(f" dep_delay : min={df_clean['dep_delay'].min():.0f}, max: {df_clean['dep_delay'].max():.0f}")
print(f" arr_delay : min={df_clean['arr_delay'].min():.0f}, max = {df_clean['arr_delay'].max():.0f}")

# Clip delays to realistic range
# -60 = 1 hour early (most early arrivals are within 60 mins)
# 600 = 10 hours late (anything beyond is likely a data error)
df_clean['dep_delay']= df_clean['dep_delay'].clip(-60,600)
df_clean['arr_delay']= df_clean['arr_delay'].clip(-60,600)

print("After clipping:")
print(f"  dep_delay: min={df_clean['dep_delay'].min():.0f}, max={df_clean['dep_delay'].max():.0f}")
print(f"  arr_delay: min={df_clean['arr_delay'].min():.0f}, max={df_clean['arr_delay'].max():.0f}")
print("Outliers handled!")

# ── STEP 6: Feature Engineering ─────────────────────────
# Creating 6 new columns from existing data
# We do NOT delete old columns — we ADD new ones alongside them

print("\n--- Step 6: Feature Engineering ---")

# NEW COLUMN 1: is_weekend
# .isin([6,7]) checks if value is 6 or 7 — returns True/False
# .astype(int) converts True→1 and False→0
df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
print("  + is_weekend     : 1=weekend, 0=weekday")


# NEW COLUMN 2: season
# We write a small function, then .apply() runs it
# on EVERY row automatically — like Excel formula dragged down
def get_season(m):
    if m in [12, 1, 2]:  return 'Winter'
    if m in [3, 4, 5]:   return 'Spring'
    if m in [6, 7, 8]:   return 'Summer'
    return 'Fall'

df_clean['season'] = df_clean['month'].apply(get_season)
print("  + season         : Winter/Spring/Summer/Fall")

# NEW COLUMN 3: time_of_day
# Same idea — function takes hour, returns label
def get_time_of_day(h):
    if  0 <= h <  6: return 'Night'
    if  6 <= h < 12: return 'Morning'
    if 12 <= h < 18: return 'Afternoon'
    return 'Evening'

df_clean['time_of_day'] = df_clean['hour'].apply(get_time_of_day)
print("  + time_of_day    : Night/Morning/Afternoon/Evening")

# NEW COLUMN 4: is_peak_hour
# lambda is a mini one-line function
# if hour is 6-9 OR 16-20 → 1, everything else → 0
df_clean['is_long_haul']= (df_clean['distance']>1500).astype(int)
print("  + is_long_haul   : 1=over 1500 miles, 0=short flight")

# NEW COLUMN 6: year
# .dt is how you access date parts in pandas
# .dt.year extracts just the year number from the date column
df_clean['year']= df_clean['date'].dt.year
print("  +year   : 2022, 2023 or 2024")

print(f"\n  Total columns now: {len(df_clean.columns)}")




# ── STEP 7: Label Encoding ───────────────────────────────
# Convert text columns → numbers for the ML model
# We KEEP original text columns — needed for charts later
# We CREATE new _enc columns alongside the originals

from sklearn.preprocessing import LabelEncoder

print("\n--- Step 7: Label Encoding ---")

# Create one LabelEncoder object — we reuse it for each column
# .fit_transform() does two things:
#   fit       = learns all unique values ("Delta", "Spirit" etc)
#   transform = converts them all to numbers (3, 8 etc)
le = LabelEncoder()

# Encode airline names → numbers
# Also print the mapping so you know Delta=3, Spirit=8 etc
df_clean['airline_enc'] = le.fit_transform(df_clean['airline'])
airline_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  airline_enc  : {airline_map}")

# Encode origin airport codes → numbers
df_clean['origin_enc'] = le.fit_transform(df_clean['origin'])
print(f"  origin_enc   : {df_clean['origin_enc'].nunique()} airports encoded (0 to {df_clean['origin_enc'].max()})")

# Encode destination airport codes → numbers
df_clean['dest_enc'] = le.fit_transform(df_clean['destination'])
print(f"  dest_enc     : {df_clean['dest_enc'].nunique()} airports encoded (0 to {df_clean['dest_enc'].max()})")

# Encode season → numbers
# Fall=0, Spring=1, Summer=2, Winter=3 (alphabetical)
df_clean['season_enc'] = le.fit_transform(df_clean['season'])
season_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  season_enc   : {season_map}")

# Encode time_of_day → numbers
df_clean['tod_enc'] = le.fit_transform(df_clean['time_of_day'])
tod_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  tod_enc      : {tod_map}")

print(f"\n  Total columns now: {len(df_clean.columns)}")


# ── STEP 8: Final Check and Save ────────────────────────
# Before saving, verify everything looks right

print("\n--- Step 8: Final Check and Save ---")

# How many rows and columns do we have now?
print(f" shape:  {df_clean.shape[0]:,}rows * {df_clean.shape[1]}columns")

# What % of flights are delayed? Should be around 33%
print(f" delay_rate: {df_clean['is_delayed'].mean()*100:.1f}%")

# Does the date range cover all 3 years?
print(f" Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

# Are all 10 airlines still present?
print(f" Airlines: {sorted(df_clean['airline'].unique())}")

# NULL CHECK — this MUST show 0
# If it shows anything other than 0, tell me immediately
total_nulls = df_clean.isnull().sum()
print(f"  Null values: {total_nulls}  <-- THIS MUST BE 0")

# Print every single column name and its data type
# This is your proof that everything was created correctly
print(f"\n  All {len(df_clean.columns)} columns:")
for col in df_clean.columns:
    print(f"    {col:25s}  {str(df_clean[col].dtype)}")



# Save the clean file — this is what every phase after this uses
os.makedirs('data', exist_ok=True)
df_clean.to_csv('Data/flights_clean.csv', index=False)
print(f"\n  Saved: data/flights_clean.csv")
