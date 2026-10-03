import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR')

def process_features(df):
    df = df.copy()
    # Add time cyclic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Map load_now
    df['load_now'] = df['total_load_kw']
    return df

print("Loading config...")
with open("op/config.json", "r") as f:
    config = json.load(f)

print("Loading scalers & model...")
scalers = joblib.load("op/scalers.joblib")
scaler = scalers['scaler']
ymean = scalers['ymean']
ystd = scalers['ystd']
model = tf.keras.models.load_model("op/ann_24h.keras")

print("Loading dataset...")
df = pd.read_excel("IIITD_synthetic_electricity_forecasting_dataset.xlsx")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = process_features(df)

test_start = pd.to_datetime(config['split']['test'][0])
test_end = pd.to_datetime(config['split']['test'][1])
test_df = df.loc[test_start:test_end]

valid_times = test_df.index[168:-24] # Need 168h history for occupancy and 24h for future

# Select a few random times for inference
import random
random.seed(123)
np.random.seed(123)
selected_times = random.sample(list(valid_times), 5)

results = []

for idx in selected_times:
    try:
        X_hist = df.loc[idx, config['hist_features']].values.astype(float)
        X_curr = df.loc[idx, config['current_features']].values.astype(float)
        
        X_fut = []
        # True future values (with deterministic noise for realistic simulation)
        for h in range(1, 25):
            t_fut = idx + pd.Timedelta(hours=h)
            row = df.loc[t_fut]
            fut_vals = []
            for f in config['future_features_per_hour']:
                if f == 'fc_academic_occupancy':
                    fut_vals.append(df.loc[t_fut - pd.Timedelta(hours=168), 'academic_occupancy'])
                elif f == 'fc_lecture_occupancy':
                    fut_vals.append(df.loc[t_fut - pd.Timedelta(hours=168), 'lecture_occupancy'])
                elif f == 'fc_library_occupancy':
                    fut_vals.append(df.loc[t_fut - pd.Timedelta(hours=168), 'library_occupancy'])
                elif f == 'fc_hostel_occupancy':
                    fut_vals.append(df.loc[t_fut - pd.Timedelta(hours=168), 'hostel_occupancy'])
                elif f == 'fc_mess_occupancy':
                    fut_vals.append(df.loc[t_fut - pd.Timedelta(hours=168), 'mess_occupancy'])
                elif f == 'fc_temperature_c':
                    # Add noise as specified in config
                    fut_vals.append(row['temperature_c'] + np.random.normal(0, config['weather_noise_sd']['temperature_c']))
                elif f == 'fc_humidity_pct':
                    fut_vals.append(row['humidity_pct'] + np.random.normal(0, config['weather_noise_sd']['humidity_pct']))
                elif f == 'fc_wind_speed_mps':
                    fut_vals.append(row['wind_speed_mps'] + np.random.normal(0, config['weather_noise_sd']['wind_speed_mps']))
                else:
                    fut_vals.append(row[f])
            X_fut.extend(fut_vals)
        
        X_fut = np.array(X_fut).astype(float)
        X = np.concatenate([X_hist, X_curr, X_fut])
        
        X_scaled = scaler.transform([X])
        y_pred_scaled = model.predict(X_scaled, verbose=0)[0]
        y_pred = y_pred_scaled * ystd + ymean
        
        y_actual = df.loc[idx + pd.Timedelta(hours=1) : idx + pd.Timedelta(hours=24), 'total_load_kw'].values
        
        mae = np.mean(np.abs(y_pred - y_actual))
        rmse = np.sqrt(np.mean(np.square(y_pred - y_actual)))
        results.append((idx, mae, rmse))
    except Exception as e:
        print(f"Error at {idx}: {e}")

print("\n--- INFERENCE RESULTS ON NEW TEST VALUES ---")
for idx, mae, rmse in results:
    print(f"Time: {idx} | Next 24h MAE: {mae:.2f} kW | RMSE: {rmse:.2f} kW")

overall_mae = np.mean([r[1] for r in results])
overall_rmse = np.mean([r[2] for r in results])
print(f"\nAverage MAE for these samples: {overall_mae:.2f} kW")
print(f"Average RMSE for these samples: {overall_rmse:.2f} kW")
