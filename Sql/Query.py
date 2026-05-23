# ── QUERY 1: Overall summary ─────────────────────────
# What it does: counts flights, calculates avg delay,
# delay rate %, and cancel rate % — all in one query

print("\\n[Q1] Overall Summary")

import pandas as pd
import sqlite3

conn= sqlite3.connect('Data/flights.db')  
q1= pd.read_sql("""
    select
        count(*)    as total_flights,
        Round(Avg(arr_delay),2)     as avg_arrival_delay_mins,
        Round(
            sum(case when is_delayed =1 then 1.0 else 0 end)/count(*)*100
        ,2)          AS delay_rate_percent,
        Round(
            sum(case when cancelled=1 then 1.0 else 0 end)/count(*)*100
        ,2)         AS cancel_rate_percent
        from flights
    """,conn)

print(q1.to_string(index=False))
# Save as CSV
q1.to_csv('Outputs/q1_summary.csv', index=False)
print("Saved: Outputs/q1_summary.csv") 



# ── QUERY 2: Airline ranking with RANK() window function ──
# RANK() is special — it assigns a rank number based on
# a column value. OVER (ORDER BY ...) tells it what to rank by
# This is called a Window Function — advanced SQL skill

print("[Q2] Airline Ranking")

q2=pd.read_sql("""
    Select
        airline,
        count(*)    as Total_flights,
        Round(Avg(arr_delay),2)     as avg_delay_mins,
        ROUND(
            SUM(CASE WHEN is_delayed=1 THEN 1.0 ELSE 0 END)
            / COUNT(*) * 100
        , 1)                            AS delay_rate_pct,
        ROUND(
            SUM(CASE WHEN cancelled=1 THEN 1.0 ELSE 0 END)
            / COUNT(*) * 100
        , 2)                            AS cancel_rate_pct
    FROM flights
    GROUP BY airline
    ORDER BY avg_delay_mins DESC
""", conn) 

# Add rank column in pandas (SQLite doesn't support RANK() easily)
q2['delay_rank'] = q2['avg_delay_mins'].rank(ascending=False).astype(int)
q2 = q2.sort_values('delay_rank')

print(q2.to_string(index= False))
q2.to_csv("Outputs/q2_airline_ranking.csv", index= False)
print("Saved: output/q2_airline_ranking.csv")




# ── QUERY 3: Top 10 worst airports ───────────────────────
# HAVING is like WHERE but for grouped data.
# We use HAVING COUNT(*) > 1000 to only include airports
# with enough flights — otherwise a tiny airport with
# 2 delayed flights would unfairly top the list

print("\\n[Q3] Top 10 Worst Airports")

q3 = pd.read_sql("""
    SELECT
        destination                     AS airport,
        COUNT(*)                        AS total_arrivals,
        ROUND(AVG(arr_delay), 1)        AS avg_arrival_delay,
        ROUND(
            SUM(CASE WHEN is_delayed=1 THEN 1.0 ELSE 0 END)
            / COUNT(*) * 100
        , 1)                            AS pct_delayed
    FROM flights
    WHERE cancelled = 0
    GROUP BY destination
    HAVING COUNT(*) > 1000
    ORDER BY avg_arrival_delay DESC
    LIMIT 10
""", conn)

print(q3.to_string(index=False))
q3.to_csv('Outputs/q3_worst_airports.csv', index=False)
print("Saved: Outputs/q3_worst_airports.csv")




# ── QUERY 4: Monthly trend using CTE ─────────────────────
# CTE = Common Table Expression. It's like creating a
# temporary table just for this query.
# WITH monthly AS (...) creates a temp table called "monthly"
# Then the main SELECT reads from that temp table.
# Why? Because we need the monthly averages FIRST,
# then calculate a rolling average ON TOP of those averages.
# You can't do that in one step — CTE lets you do it in two.

print("\\n[Q4] Monthly Delay Trend")

q4 = pd.read_sql("""
    WITH monthly AS (
        SELECT
            month,
            COUNT(*)                    AS total_flights,
            ROUND(AVG(arr_delay), 1)    AS avg_delay,
            ROUND(
                SUM(CASE WHEN is_delayed=1 THEN 1.0 ELSE 0 END)
                / COUNT(*) * 100
            , 1)                        AS delay_rate_pct
        FROM flights
        WHERE cancelled = 0
        GROUP BY month
    )
    SELECT *
    FROM monthly
    ORDER BY month
""", conn)

# Add rolling 3-month average using pandas (simpler than SQL here)
q4['rolling_3m_avg'] = q4['avg_delay'].rolling(window=3, center=True).mean().round(1)

print(q4.to_string(index=False))
q4.to_csv('Outputs/q4_monthly_trend.csv', index=False)
print("Saved: Outputs/q4_monthly_trend.csv")


# ── QUERY 5: Delay cause breakdown ───────────────────────
print("\\n[Q5] Delay Cause Breakdown by Airline")

q5 = pd.read_sql("""
    SELECT
        airline,
        ROUND(AVG(weather_delay), 1)        AS avg_weather_delay,
        ROUND(AVG(carrier_delay), 1)        AS avg_carrier_delay,
        ROUND(AVG(nas_delay), 1)            AS avg_nas_delay,
        ROUND(AVG(late_aircraft_delay), 1)  AS avg_late_aircraft
    FROM flights
    WHERE cancelled = 0 AND is_delayed = 1
    GROUP BY airline
    ORDER BY avg_carrier_delay DESC
""", conn)
print(q5.to_string(index=False))
q5.to_csv('Outputs/q5_delay_causes.csv', index=False)
print("Saved: output/q5_delay_causes.csv")



# ── QUERY 6: Hour x Day heatmap data ─────────────────────
print("\\n[Q6] Heatmap Data (Hour x Day of Week)")

q6 = pd.read_sql("""
    SELECT
        hour,
        day_of_week,
        COUNT(*)                                    AS total_flights,
        ROUND(
            SUM(CASE WHEN is_delayed=1 THEN 1.0 ELSE 0 END)
            / COUNT(*) * 100
        , 1)                                        AS delay_rate_pct
    FROM flights
    WHERE cancelled = 0
    GROUP BY hour, day_of_week
    ORDER BY hour, day_of_week
""", conn)
print(q6.to_string(index=False))
print(f"\nTotal combinations: {len(q6)} (should be 168)")
q6.to_csv('Outputs/q6_heatmap_data.csv', index=False)
print("Saved: output/q6_heatmap_data.csv")


# ── QUERY 7: Top delayed routes ───────────────────────────
print("\\n[Q7] Top 15 Most Delayed Routes")

q7 = pd.read_sql("""
    SELECT
        origin || ' to ' || destination    AS route,
        COUNT(*)                            AS total_flights,
        ROUND(AVG(arr_delay), 1)            AS avg_delay_mins,
        ROUND(
            SUM(CASE WHEN is_delayed=1 THEN 1.0 ELSE 0 END)
            / COUNT(*) * 100
        , 1)                                AS delay_rate_pct
    FROM flights
    WHERE cancelled = 0
    GROUP BY origin, destination
    HAVING COUNT(*) > 50
    ORDER BY avg_delay_mins DESC
    LIMIT 15
""", conn)

print(q7.to_string(index=False))
q7.to_csv('Outputs/q7_worst_routes.csv', index=False)
print("Saved: output/q7_worst_routes.csv")



# ── Export data for Power BI and Tableau ─────────────────
# Your database has 50,000 rows — export all of them
# No need for random sampling on small datasets

print("\\nExporting dashboard CSV...")
dashboard_df = pd.read_sql("""
    SELECT
        flight_id, date, airline, origin, destination,
        dep_delay, arr_delay, distance,
        day_of_week, month, hour,
        cancelled, diverted,
        weather_delay, carrier_delay,
        nas_delay, late_aircraft_delay,
        is_delayed
    FROM flights
    ORDER BY date
""", conn)

dashboard_df.to_csv('Dashboard/flights_for_dashboard.csv', index=False)
print(f"Exported {len(dashboard_df):,} rows to dashboard/flights_for_dashboard.csv")

# Always close the connection at the end
conn.close()
print("\nDatabase connection closed.")
print("Phase 2 complete! Check your output/ folder for all CSV files.")
