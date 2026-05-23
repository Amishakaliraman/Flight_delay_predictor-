# eda/numpy_analysis.py
# Statistical analysis using NumPy

import pandas as pd
import numpy as np

df  = pd.read_csv('data/flights_clean.csv')
arr = df['arr_delay'].dropna().values
dep = df['dep_delay'].dropna().values

print("="*55)
print("NumPy Statistical Analysis — Flight Delays")
print("="*55)

print(f"\nArrival Delay Statistics:")
print(f"  Mean delay    : {np.mean(arr):.2f} minutes")
print(f"  Median delay  : {np.median(arr):.2f} minutes")
print(f"  Std deviation : {np.std(arr):.2f} minutes")
print(f"  Min delay     : {np.min(arr):.2f} minutes")
print(f"  Max delay     : {np.max(arr):.2f} minutes")
print(f"  25th pct      : {np.percentile(arr,25):.2f} minutes")
print(f"  75th pct      : {np.percentile(arr,75):.2f} minutes")
print(f"  95th pct      : {np.percentile(arr,95):.2f} minutes")
print(f"  99th pct      : {np.percentile(arr,99):.2f} minutes")

print(f"\nCorrelations with arr_delay:")
for col in ['dep_delay','distance','hour','month','is_weekend']:
    x = df[col].dropna().values
    n = min(len(x), len(arr))
    c = np.corrcoef(x[:n], arr[:n])[0,1]
    print(f"  {col:20s} : {c:.4f}")

print(f"\nDelay rate by time of day:")
for label, mask in [
    ('Morning  (6-11am) ', (df['hour']>=6) &(df['hour']<12)),
    ('Afternoon(12-5pm) ', (df['hour']>=12)&(df['hour']<18)),
    ('Evening  (6-10pm) ', (df['hour']>=18)&(df['hour']<22)),
    ('Night    (10pm-6am)', (df['hour']>=22)|(df['hour']<6))
]:
    rate = df.loc[mask,'is_delayed'].mean()*100
    print(f"  {label}: {rate:.1f}%")

print(f"\nTop 3 worst airlines by delay rate:")
worst = df.groupby('airline')['is_delayed'].mean()*100
worst = worst.sort_values(ascending=False).head(3)
for airline, rate in worst.items():
    print(f"  {airline:12s}: {rate:.1f}%")

print(f"\nTop 3 best airlines by delay rate:")
best = df.groupby('airline')['is_delayed'].mean()*100
best = best.sort_values().head(3)
for airline, rate in best.items():
    print(f"  {airline:12s}: {rate:.1f}%")

print("\nNumPy analysis complete!")