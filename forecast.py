import datetime
import os
import time
import urllib.parse
import warnings

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import TFT
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

# ==============================================================================
# 1. LOAD AND PREPARE ALERT DATA
# ==============================================================================
LOCAL_CSV = "official_data_en.csv"
CSV_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/master/datasets/official_data_en.csv"

if os.path.exists(LOCAL_CSV):
    df = pd.read_csv(LOCAL_CSV)
else:
    df = pd.read_csv(CSV_URL)
    df.to_csv(LOCAL_CSV, index=False)

target_regions = [
    'Vinnytska oblast', 'Zhytomyrska oblast', 'Cherkaska oblast', 'Mykolaivska oblast',
    'Kirovohradska oblast', 'Khmelnytska oblast', 'Kyivska oblast', 'Ivano-Frankivska oblast',
    'Lvivska oblast', 'Rivnenska oblast', 'Volynska oblast', 'Kyiv City',
    'Ternopilska oblast', 'Chernihivska oblast', 'Poltavska oblast', 'Donetska oblast',
    'Sumska oblast', 'Dnipropetrovska oblast', 'Kharkivska oblast', 'Odeska oblast',
    'Zaporizka oblast', 'Chernivetska oblast', 'Zakarpatska oblast', 'Luhanska oblast',
    'Khersonska oblast'
]

OBLAST_COL = 'oblast' if 'oblast' in df.columns else 'region_title'
START_COL = 'started_at' if 'started_at' in df.columns else 'start'
END_COL = 'finished_at' if 'finished_at' in df.columns else 'end'

if 'level' in df.columns:
    df = df[df['level'] == 'oblast']

df[START_COL] = pd.to_datetime(df[START_COL], errors='coerce')
df[END_COL] = pd.to_datetime(df[END_COL], errors='coerce')
df = df.dropna(subset=[START_COL, END_COL])
df = df[df[OBLAST_COL].isin(target_regions)]

df['duration_hours'] = (df[END_COL] - df[START_COL]).dt.total_seconds() / 3600
df = df[(df['duration_hours'] > 0) & (df['duration_hours'] < 24)]
df['date'] = pd.to_datetime(df[START_COL].dt.date)

daily_df = df.groupby('date')['duration_hours'].sum().reset_index()
full_date_range = pd.date_range(start=daily_df['date'].min(), end=daily_df['date'].max(), freq='D')
daily_df = daily_df.set_index('date').reindex(full_date_range).fillna(0.0).reset_index()
daily_df = daily_df.rename(columns={'index': 'date'})

LAST_DOCUMENTED_DATE = daily_df['date'].max()
print(f"Latest documented date in dataset: {LAST_DOCUMENTED_DATE.strftime('%Y-%m-%d')}")

# ==============================================================================
# 2. FUTURE EXOGENOUS (Calendar & Time Decay)
# ==============================================================================
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek.astype(float)
daily_df['month'] = daily_df['date'].dt.month.astype(float)
daily_df['day_of_month'] = daily_df['date'].dt.day.astype(float)
daily_df['is_weekend'] = daily_df['date'].dt.dayofweek.isin([5, 6]).astype(float)

symbolic_dates = [(5, 8), (5, 9), (6, 28), (8, 23), (8, 24), (8, 25), (8, 26), (1, 1), (12, 31)]
daily_df['is_symbolic_date'] = daily_df['date'].apply(
    lambda d: 1.0 if (d.month, d.day) in symbolic_dates else 0.0
).astype(float)

HALF_LIFE_DAYS = 90.0
daily_df['days_old'] = (LAST_DOCUMENTED_DATE - daily_df['date']).dt.days
daily_df['time_decay_weight'] = (0.5 ** (daily_df['days_old'] / HALF_LIFE_DAYS)).astype(float)

futr_exog_features = [
    'day_of_week', 'month', 'day_of_month', 'is_weekend',
    'is_symbolic_date', 'time_decay_weight'
]

# ==============================================================================
# 3. HISTORICAL EXOGENOUS (Cached News + Tactical Rolling Counters)
# ==============================================================================
NEWS_TEXT_CACHE = "news_text_cache.csv"
NEWS_SIGNALS_CACHE = "news_signals_cache.csv"

N_COMPONENTS = 8
pca_cols = [f'news_signal_{i+1}' for i in range(N_COMPONENTS)]

if os.path.exists(NEWS_SIGNALS_CACHE):
    signals_df = pd.read_csv(NEWS_SIGNALS_CACHE)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    if all(col in signals_df.columns for col in pca_cols):
        print(f"📦 Loading precomputed news embeddings from '{NEWS_SIGNALS_CACHE}'...")
        daily_df = daily_df.merge(signals_df[['date'] + pca_cols], on='date', how='left')
        daily_df[pca_cols] = daily_df[pca_cols].fillna(0.0)
    else:
        os.remove(NEWS_SIGNALS_CACHE)

if not all(col in daily_df.columns for col in pca_cols):
    if os.path.exists(NEWS_TEXT_CACHE):
        print(f"📂 Loading raw text from '{NEWS_TEXT_CACHE}'...")
        text_cache_df = pd.read_csv(NEWS_TEXT_CACHE)
        text_cache_df['date'] = pd.to_datetime(text_cache_df['date'])
    else:
        text_cache_df = pd.DataFrame(columns=['date', 'daily_summary'])

    daily_df = daily_df.merge(text_cache_df[['date', 'daily_summary']], on='date', how='left')
    missing_dates = daily_df[daily_df['daily_summary'].isna()]['date'].tolist()

    if missing_dates:
        print(f"📡 Fetching {len(missing_dates)} missing days via Google News RSS...")
        current_cache = text_cache_df.copy()
        new_records = []
        for i, d in enumerate(missing_dates):
            next_day = d + datetime.timedelta(days=1)
            q = urllib.parse.quote(f"Ukraine (war OR strike OR missile) after:{d.strftime('%Y-%m-%d')} before:{next_day.strftime('%Y-%m-%d')}")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

            feed = feedparser.parse(rss_url)
            headlines = [entry.title for entry in feed.entries[:4]]
            summary = " | ".join(headlines) if headlines else "No major news reported."

            daily_df.loc[daily_df['date'] == d, 'daily_summary'] = summary
            new_records.append({'date': d, 'daily_summary': summary})
            time.sleep(0.2)

            if (i + 1) % 20 == 0 or (i + 1) == len(missing_dates):
                batch_df = pd.DataFrame(new_records)
                current_cache = pd.concat([current_cache, batch_df], ignore_index=True).drop_duplicates(subset=['date'])
                current_cache.to_csv(NEWS_TEXT_CACHE, index=False)
                new_records = []

    print("\nEncoding news text with BAAI/bge-m3...")
    encoder = SentenceTransformer('BAAI/bge-m3')
    raw_embeddings = encoder.encode(daily_df['daily_summary'].tolist(), show_progress_bar=True)

    pca = PCA(n_components=N_COMPONENTS)
    compressed_signals = pca.fit_transform(raw_embeddings)
    for i, col in enumerate(pca_cols):
        daily_df[col] = compressed_signals[:, i].astype(float)

    daily_df[['date'] + pca_cols].to_csv(NEWS_SIGNALS_CACHE, index=False)
    print(f"✅ Saved news signals to '{NEWS_SIGNALS_CACHE}'.")

daily_df['rolling_7d_hours'] = daily_df['duration_hours'].rolling(7, min_periods=1).mean().astype(float)
hist_exog_features = pca_cols + ['rolling_7d_hours']

# ==============================================================================
# 4. PREPARE FULL TRAINING SET (100% of Documented History)
# ==============================================================================
daily_df['y_log'] = np.log1p(daily_df['duration_hours'])

train_df = pd.DataFrame({
    'unique_id': 'Target_Regions',
    'ds': daily_df['date'],
    'y': daily_df['y_log'],
})

for col in futr_exog_features + hist_exog_features:
    train_df[col] = daily_df[col]

FORECAST_HORIZON = 30
print(f"Training on all available records ({len(train_df)} days)...")

# ==============================================================================
# 5. CONSTRUCT FUTURE EXOGENOUS DATAFRAME (Next 30 Days)
# ==============================================================================
future_dates = pd.date_range(
    start=LAST_DOCUMENTED_DATE + pd.Timedelta(days=1),
    periods=FORECAST_HORIZON,
    freq='D'
)

future_df = pd.DataFrame({
    'unique_id': 'Target_Regions',
    'ds': future_dates
})

future_df['day_of_week'] = future_df['ds'].dt.dayofweek.astype(float)
future_df['month'] = future_df['ds'].dt.month.astype(float)
future_df['day_of_month'] = future_df['ds'].dt.day.astype(float)
future_df['is_weekend'] = future_df['ds'].dt.dayofweek.isin([5, 6]).astype(float)
future_df['is_symbolic_date'] = future_df['ds'].apply(
    lambda d: 1.0 if (d.month, d.day) in symbolic_dates else 0.0
).astype(float)
future_df['time_decay_weight'] = 1.0  # Most recent operational regime

# ==============================================================================
# 6. TRAIN TFT AND GENERATE FUTURE FORECAST
# ==============================================================================
print("\nInitializing Temporal Fusion Transformer (TFT) on GPU...")
models = [
    TFT(
        h=FORECAST_HORIZON,
        input_size=120,
        hidden_size=128,
        loss=MSE(),
        learning_rate=1e-3,
        max_steps=1200,
        random_seed=42,
        futr_exog_list=futr_exog_features,
        hist_exog_list=hist_exog_features,
        accelerator='gpu',
        devices=1,
        lr_scheduler=CosineAnnealingLR,
        lr_scheduler_kwargs={'T_max': 1200, 'eta_min': 1e-6}
    )
]

nf = NeuralForecast(models=models, freq='D')

print("Training model on complete dataset...")
nf.fit(df=train_df)

print(f"Generating live {FORECAST_HORIZON}-day operational forecast...")
forecast_raw = nf.predict(futr_df=future_df[['unique_id', 'ds'] + futr_exog_features]).reset_index()

forecast_df = future_df[['ds']].merge(forecast_raw[['ds', 'TFT']], on='ds')
forecast_df['predicted_hours'] = np.expm1(forecast_df['TFT']).clip(lower=0.0)

# Export predictions
forecast_df[['ds', 'predicted_hours']].to_csv('forecast_next_30_days.csv', index=False)
print("✅ Forecast exported to 'forecast_next_30_days.csv'.")
print("\nProjected Alert Hours for Upcoming Period:")
print(forecast_df[['ds', 'predicted_hours']].to_string(index=False))

# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

context_days = 45
context_df = daily_df.iloc[-context_days:]

# Historical context line
plt.plot(
    context_df['date'],
    context_df['duration_hours'],
    label='Historical Actuals (Past 45 Days)',
    color='#2c3e50',
    linewidth=2.0
)

# Connect last historical observation to forecast line
connect_dates = [context_df['date'].iloc[-1]] + forecast_df['ds'].tolist()
connect_values = [context_df['duration_hours'].iloc[-1]] + forecast_df['predicted_hours'].tolist()

plt.plot(
    connect_dates,
    connect_values,
    label='TFT 30-Day Forward Forecast',
    color='#e67e22',
    linewidth=2.5,
    linestyle='--'
)

plt.axvline(
    x=LAST_DOCUMENTED_DATE,
    color='#2980b9',
    linestyle=':',
    linewidth=2.0,
    label=f"Forecast Cutoff ({LAST_DOCUMENTED_DATE.strftime('%Y-%m-%d')})"
)

plt.title('Real Forward Air Raid Forecast (Next 30 Days)', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Projected Total Alert Hours Across Target Regions')
plt.legend()
plt.tight_layout()
plt.savefig('live_forward_forecast.png')
plt.show()

print("✅ Chart saved to 'live_forward_forecast.png'.")