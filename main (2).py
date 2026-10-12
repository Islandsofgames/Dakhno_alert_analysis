import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
CSV_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/master/datasets/official_data_en.csv"
print("Downloading data...")
df = pd.read_csv(CSV_URL)

# Note: The exact column names might vary slightly in the CSV.
# We assume 'region_title', 'start', and 'end' exist. Update these if the dataset headers differ.
REGION_COL = 'region_title' if 'region_title' in df.columns else df.columns[0]
START_COL = 'start' if 'start' in df.columns else 'started_at'
END_COL = 'end' if 'end' in df.columns else 'finished_at'
print(df[REGION_COL].unique())
# 2. Data Cleaning & Feature Engineering
print("Processing dates and calculating durations...")
df[START_COL] = pd.to_datetime(df[START_COL], errors='coerce')
df[END_COL] = pd.to_datetime(df[END_COL], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=[START_COL])

# Calculate duration in minutes
df['duration_minutes'] = (df[END_COL] - df[START_COL]).dt.total_seconds() / 60

# Extract time features for the heatmap and timeline
df['date'] = df[START_COL].dt.date
df['hour'] = df[START_COL].dt.hour
df['day_of_week'] = df[START_COL].dt.day_name()
df['year_month'] = df[START_COL].dt.to_period('M')

# Set up the visual style
sns.set_theme(style="whitegrid")

# ==========================================
# GRAPH 1: Top 15 Regions by Alert Count
# ==========================================
plt.figure(figsize=(12, 8))
top_regions = df[REGION_COL].value_counts().head(15)
sns.barplot(x=top_regions.values, y=top_regions.index, palette="Reds_r")
plt.title("Top 15 Regions by Total Number of Alerts", fontsize=16)
plt.xlabel("Number of Alerts")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("1_alerts_by_region.png")
plt.show()

# ==========================================
# GRAPH 2: Alerts Over Time (Monthly Trend)
# ==========================================
plt.figure(figsize=(14, 6))
monthly_alerts = df.groupby('year_month').size()
monthly_alerts.plot(kind='line', marker='o', color='darkred', linewidth=2)
plt.title("Air Raid Alerts Over Time (Monthly)", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Total Alerts Nationwide")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("2_alerts_over_time.png")
plt.show()

# ==========================================
# GRAPH 3: Time of Day vs. Day of Week Heatmap
# ==========================================
plt.figure(figsize=(12, 6))
# Order days logically
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(days_order)

sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=.5, fmt="d", annot=False)
plt.title("Heatmap: When do alerts most commonly start?", fontsize=16)
plt.xlabel("Hour of Day (24h)")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig("3_alerts_heatmap.png")
plt.show()

# ==========================================
# GRAPH 4: Distribution of Alert Durations
# ==========================================
plt.figure(figsize=(10, 6))
# Filter out extreme outliers (e.g., > 10 hours) for a cleaner graph
duration_data = df[df['duration_minutes'] < 600]['duration_minutes']
sns.histplot(duration_data, bins=60, color='coral', kde=True)
plt.title("Distribution of Alert Durations (Under 10 Hours)", fontsize=16)
plt.xlabel("Duration (Minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("4_duration_distribution.png")
plt.show()

# ==========================================
# GRAPH 5: Total Time Under Alert (Hours)
# ==========================================
plt.figure(figsize=(12, 8))
# Calculate total hours per region
total_hours = (df.groupby(REGION_COL)['duration_minutes'].sum() / 60).sort_values(ascending=False).head(15)

sns.barplot(x=total_hours.values, y=total_hours.index, palette="mako")
plt.title("Top 15 Regions by Total Time Under Alert (Hours)", fontsize=16)
plt.xlabel("Total Hours in Shelter")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("5_total_hours_by_region.png")
plt.show()

# ==========================================
# GRAPH 6: Tactics Scatter Plot (Time vs. Duration)
# ==========================================
plt.figure(figsize=(12, 8))
# Filter out massive outliers (>10 hours) so the clusters are readable
scatter_data = df[df['duration_minutes'] < 600]

# Using alpha=0.1 makes overlapping points darker, showing density
sns.scatterplot(
    data=scatter_data,
    x='hour',
    y='duration_minutes',
    alpha=0.1,
    color='purple',
    s=20 # dot size
)
plt.title("Tactical Patterns: Start Time vs. Alert Duration", fontsize=16)
plt.xlabel("Hour of Day (24h)")
plt.ylabel("Alert Duration (Minutes)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("6_time_vs_duration_scatter.png")
plt.show()

# ==========================================
# GRAPH 7: Regional Comparison - 14-Day Rolling Average
# ==========================================
plt.figure(figsize=(14, 6))

# Select a few distinct regions to compare (e.g., Capital, Frontline, West)
target_regions = [       'Zhytomyrska oblast',
        'Cherkaska oblast',      'Mykolaivska oblast',
    'Kirovohradska oblast',
          'Kyivska oblast',
          'Lvivska oblast',              'Kyiv City']

# Filter data for only these regions
df_compare = df[df[REGION_COL].isin(target_regions)]

# Sum the duration (in hours) per region per day
daily_regional_hours = df_compare.groupby([REGION_COL, 'date'])['duration_minutes'].sum() / 60
daily_regional_hours = daily_regional_hours.reset_index()

# Pivot the data so dates are rows and regions are columns
pivot_df = daily_regional_hours.pivot(index='date', columns=REGION_COL, values='duration_minutes').fillna(0)

# Calculate a 14-day rolling average to smooth out the noise
rolling_df = pivot_df.rolling(window=14).mean()

# Plot the smoothed data
sns.lineplot(data=rolling_df, linewidth=2.5)
plt.title("Daily Time Under Alert (14-Day Rolling Average)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Average Hours per Day")
plt.legend(title="Region")
plt.tight_layout()
plt.savefig("7_regional_rolling_average.png")
plt.show()

print("✅ Analytics complete. Graphs have been displayed and saved as PNG files.")


