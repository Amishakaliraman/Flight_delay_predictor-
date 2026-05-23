# eda/eda_visualizations.py
# Makes 8 charts and saves them as PNG files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Create output/charts folder if it doesn't exist
os.makedirs('Outputs/charts', exist_ok=True)
print(plt.style.available)

# Set a clean visual style for all charts
plt.style.use('ggplot')


# Load the clean data from Phase 3
print("Loading cleaned data...")
df = pd.read_csv('Data/flights_clean.csv')
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")


# ── CHART 1: Delay rate by airline ──────────────────────
# groupby('airline') groups all rows by airline name
# ['is_delayed'].mean() calculates average of 0s and 1s = % delayed
# * 100 converts 0.33 to 33%
# .sort_values() sorts from lowest to highest

print("\nMaking Chart 1: Delay rate by airline...")
airline_delay = (df.groupby('airline')['is_delayed']
                   .mean()
                   .sort_values()
                   * 100)


# Color palette — 10 colors for 10 airlines
Colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(airline_delay)))


fig, ax = plt.subplots(figsize=(10, 6), facecolor = "#FCFBFE")
ax.set_facecolor("#FDF2F2")

# Draw horizontal bars — one per airline
bars = ax.barh(airline_delay.index, airline_delay.values, color= Colors)

# Add red dashed line showing the overall average
ax.axvline(airline_delay.mean(), color='red', ls='--', lw=1.5,
           label=f'Average: {airline_delay.mean():.1f}%')

# Add the % number at end of each bar
max_val = airline_delay.max()
for bar, val in zip(bars, airline_delay.values):
    ax.text(val + max_val*0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=11, color= '#374151')

ax.set_xlim(0, max_val * 1.15)
ax.set_xlabel('Delay Rate (%)', fontsize=12)
ax.set_title('Flight Delay Rate by Airline', fontsize=14, fontweight='bold', color= '#111827')
ax.legend()
plt.tight_layout()

# Save the chart as PNG — this is your portfolio screenshot
plt.savefig('Outputs/charts/01_delay_by_airline.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Outputs/charts/01_delay_by_airline.png")

# ── CHART 2: Delay rate by hour of day ──────────────────

print("Making Chart 2: Delay rate by hour...")

hourly = df.groupby('hour')['is_delayed'].mean() * 100

fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FCFBFE')
ax.set_facecolor("#FAF4F4")


# marker='o' puts a dot at each hour point
# fill_between adds the light blue shaded area below the line
ax.plot(hourly.index, hourly.values, marker='o',
        color="#0D41B0", linewidth=2.5, markersize=6)
ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#84BBF9")

# Red dashed line = daily average
ax.axhline(hourly.mean(), color="#DF6963", ls='--', lw=1.2,
           label=f'Daily avg: {hourly.mean():.1f}%')

ax.set_xlabel('Hour of Day (0 = midnight, 12 = noon, 23 = 11pm)', fontsize=12, color= '#6B7280')
ax.set_ylabel('Delay Rate (%)', fontsize=12, color= '#6B7280')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color("#EBE7EC")

ax.grid(color= '#EBE7EC', linestyle= '-', linewidth= 0.5)
ax.tick_params(colors= '#374151')
ax.set_title('Delay Rate by Hour of Day', fontsize=14, fontweight='bold', color= '#111827')
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig('Outputs/charts/02_delay_by_hour.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: Outputs/charts/02_delay_by_hour.png")


# ── CHART 3: Monthly delay trend ────────────────────────
# .agg() calculates multiple things at once for each group

print("Making Chart 3: Monthly trend...")

monthly = df.groupby('month').agg(
    delay_rate=('is_delayed', 'mean'),
    avg_delay =('arr_delay',  'mean'),
    total     =('flight_id',  'count')
).reset_index()
monthly['delay_rate'] *= 100

fig, ax1 = plt.subplots(figsize=(12, 5),facecolor='#FCFBFE')
ax.set_facecolor('#FDF2F2')

ax2 = ax1.twinx()  # creates second Y axis on the right

# Left axis: grey bars for flight count
ax1.bar(monthly['month'], monthly['total']/1000,
        alpha=0.3, color="#AFC2E9", label='No. of Flights (thousands)')

# Left axis: red line for delay rate %
ax1.plot(monthly['month'], monthly['delay_rate'], 'o-',
         color="#EC685E", lw=2.5, label='Delay Rate %', markersize=5)

# Right axis: blue dashed line for avg delay minutes
ax2.plot(monthly['month'], monthly['avg_delay'], marker='s', linestyle='--',
         color="#4279F0", lw=2, label='Avg Delay (min)', markersize=4)

ax1.set_xlabel('Month', fontsize=12, color='#6B7280', labelpad=10)
ax1.set_ylabel('Delay Rate (%) / Flights (K)', fontsize=11, color= '#EF4444', labelpad=10)
ax2.set_ylabel('Avg Delay (minutes)', fontsize=11, color= "#427BF7", labelpad=10)
ax1.set_title('Monthly Delay Trends', fontsize=14, fontweight='bold', color= '#111827', pad=20)
ax1.set_xticks(range(1,13))
ax1.tick_params(axis='x', pad=5)
ax1.tick_params(axis='y', pad=5)  
ax2.tick_params(axis='y', pad=5)
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec'])

ax1.spines[['top','right']].set_visible(False)
ax2.spines[['top']].set_visible(False)

ax1.grid(color= '#E5E7EB', linewidth=0.5)

# Combine legends from both axes into one box
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper center', bbox_to_anchor=(0.5,1.3),ncol=3,  frameon= False)
plt.tight_layout(pad=2)
plt.savefig('Outputs/charts/03_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: Outputs/charts/03_monthly_trend.png")


# ── CHART 4: Heatmap hour x day of week ─────────────────
# pivot table reshapes data into a grid:
#   rows = hours (0-23)
#   columns = days (1-7)
#   values = delay rate %
# This grid is exactly what sns.heatmap() needs

print("Making Chart 4: Heatmap (most impressive chart)...")

pivot = (df.groupby(['hour', 'day_of_week'])['is_delayed']
           .mean()
           .unstack()  # converts grouped data into a 2D grid
           * 100)

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    pivot,
    annot=True,      # show % number inside every cell
    fmt='.0f',        # format as whole number (no decimals)
    cmap='YlOrRd',   # color: yellow=low, orange=medium, red=high
    ax=ax,
    linewidths=0.3,  # thin lines between cells
    xticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
    yticklabels=range(24)
)
ax.set_title('Delay Rate (%) by Hour and Day of Week',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Hour of Day', fontsize=12)
plt.tight_layout()
plt.savefig('Outputs/charts/04_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: Outputs/charts/04_heatmap.png")


# ── CHART 5: Delay cause stacked bar ────────────────────
print("Making Chart 5: Delay causes...")

cause_cols   = ['weather_delay','carrier_delay','nas_delay','late_aircraft_delay']
cause_labels = ['Weather','Carrier','NAS','Late Aircraft']
cause_colors = ["#ECC98D","#D6AA70","#B9854E","#996432"]

delayed_only     = df[df['is_delayed']==1]
cause_by_airline = delayed_only.groupby('airline')[cause_cols].mean()
cause_by_airline.columns = cause_labels

fig, ax = plt.subplots(figsize=(14, 7))
cause_by_airline.plot(kind='bar', stacked=True, ax=ax,
                      color=cause_colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Airline', fontsize=12, fontweight='semibold')
ax.set_ylabel('Avg Delay Minutes', fontsize=12, fontweight='semibold')
ax.set_title('Delay Cause Breakdown by Airline (Delayed Flights Only)',
             fontsize=15, fontweight='heavy', pad=15)
ax.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Outputs/charts/05_delay_causes.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Outputs/charts/05_delay_causes.png")

# ── CHART 6: Delay distribution histogram + boxplot ─────
print("Making Chart 6: Delay distribution...")

#Professional style
sns.set_theme(style='white')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Left Histogram

delayed = df[df['arr_delay']>15]['arr_delay']
sns.histplot(delayed.clip(0,300), bins=35, color="#E96540",
            edgecolor='white', alpha=0.8, ax=axes[0])
# Mean line
axes[0].axvline(delayed.mean(), color='#3D405B', ls='--',
               label=f'Mean: {delayed.mean():.0f} min')
axes[0].set_title('How Long Are Delays? (delayed flights only)',
                   fontsize=15, fontweight='bold', pad=15)
axes[0].set_xlabel('Arrival Delay (minutes)', fontsize=12, fontweight='semibold')
axes[0].set_ylabel('Number of Flights',fontsize=12 , fontweight='semibold')
axes[0].legend(frameon=False)

# Right Boxplot
sns.boxplot(data=df[df['arr_delay']>0], x='time_of_day', y='arr_delay',
            ax=axes[1], palette='Set2',
            order=['Morning','Afternoon','Evening','Night'], showfliers=False ,linewidth=1.5)
axes[1].set_ylim(0, 200)
axes[1].set_title('Delay Duration by Time of Day', fontsize=15, fontweight='bold', pad=15)
axes[1].set_xlabel('Time of Day', fontsize=12, fontweight='semibold')
axes[1].set_ylabel('Arrival Delay (minutes)', fontsize=12, fontweight='semibold')

# Improve ticks
for ax in axes:
    ax.tick_params(axis='both', labelsize=10)

# Remove unnecessary boarders
sns.despine()

# Better spacing
plt.tight_layout(pad=3)
plt.tight_layout()
plt.savefig('Outputs/charts/06_delay_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Outputs/charts/06_delay_distribution.png")

# ── CHART 7: Correlation matrix ──────────────────────────
print("Making Chart 7: Correlation matrix...")

df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 17 <= x <=20 else 0 )

num_cols = ['dep_delay','arr_delay','distance','hour','day_of_week',
            'month','is_weekend','is_peak_hour','is_long_haul','is_delayed']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide top half (duplicate)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, mask=mask, linewidths=0.5, vmin=-1, vmax=1, center=0)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Outputs/charts/07_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Outputs/charts/07_correlation_matrix.png")

# ── CHART 8: Season comparison ────────────────────────────
print("Making Chart 8: Season comparison...")

season_order = ['Winter','Spring','Summer','Fall']
season_colors= ["#78BBF1","#B2F0B3","#EABB73","#DE736C"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

season_delay = df.groupby('season')['is_delayed'].mean()*100
season_delay = season_delay.reindex(season_order)
axes[0].bar(season_delay.index, season_delay.values, color=season_colors)
axes[0].set_title('Delay Rate by Season', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Delay Rate (%)', fontsize=12, fontweight='semibold')
for i, v in enumerate(season_delay.values):
    axes[0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')

season_avg = df.groupby('season')['arr_delay'].mean()
season_avg = season_avg.reindex(season_order)
bars2= axes[1].bar(season_avg.index, season_avg.values, color=season_colors)
for bar in bars2:
    height = bar.get_height()

    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        f'{height:.1f} min',
        ha= 'center',
        fontsize=10,
        fontweight='bold'
    )
axes[1].set_title('Avg Arrival Delay by Season', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Avg Delay (minutes)',fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('Outputs/charts/08_season_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Outputs/charts/08_season_analysis.png")
print("\nAll 8 charts saved to Outputs/charts/")





