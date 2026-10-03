import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta

app = Flask(__name__)

# Load Model & Config
base_dir = os.path.dirname(os.path.abspath(__file__))
op_dir = os.path.join(base_dir, 'op')
config_path = os.path.join(op_dir, 'config.json')
scaler_path = os.path.join(op_dir, 'scalers.joblib')
model_path = os.path.join(op_dir, 'ann_24h.keras')

with open(config_path, "r") as f:
    config = json.load(f)

scalers = joblib.load(scaler_path)
scaler = scalers['scaler']
ymean = scalers['ymean']
ystd = scalers['ystd']

# Setting TF log level
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(model_path)

# Load dataset for base context (lags, occupancy)
df = pd.read_excel(os.path.join(base_dir, "IIITD_synthetic_electricity_forecasting_dataset.xlsx"))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Process cyclic features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['load_now'] = df['total_load_kw']

# Get the last valid timestamp in the dataset to use as our "current time" for historical lags
test_end = pd.to_datetime(config['split']['test'][1]) - pd.Timedelta(hours=24)
base_idx = df.index[df.index <= test_end][-1]

import holidays
def get_calendar_context(date_str):
    d = pd.to_datetime(date_str)
    # IIIT Delhi is in India
    in_holidays = d in holidays.IN() 
    is_weekend = 1 if d.dayofweek >= 5 else 0
    is_holiday = 1 if in_holidays else 0
    is_working_day = 1 if (not is_weekend and not is_holiday) else 0
    return {
        'date': date_str,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_working_day': is_working_day
    }

@app.route('/')
def index():
    return render_template('v5.html')

import holidays

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json or {}
        forecast_date_str = req_data.get('forecast_date')
        
        # Determine the base index (T=0) based on forecast_date
        if forecast_date_str:
            target_date = pd.to_datetime(forecast_date_str).replace(hour=0, minute=0, second=0)
            
            # LIVE MODE: If date is beyond dataset, seamlessly map to a matching historical week
            if target_date > df.index[-1] - pd.Timedelta(days=2):
                # Find a safe proxy date with the same month and day of week
                safe_mask = (df.index.month == target_date.month) & (df.index.dayofweek == target_date.dayofweek) & (df.index < df.index[-1] - pd.Timedelta(days=7))
                potential_bases = df.index[safe_mask]
                if len(potential_bases) > 0:
                    local_base_idx = potential_bases[-1].replace(hour=0, minute=0, second=0)
                else:
                    local_base_idx = df.index[-24 * 7].replace(hour=0, minute=0, second=0)
            else:
                valid_indices = df.index[df.index < target_date]
                if len(valid_indices) == 0:
                    return jsonify({'success': False, 'error': 'Date too early for historical data.'})
                local_base_idx = valid_indices[-1]
        else:
            # Default to the end of the test set if no date provided
            test_end = pd.to_datetime(config['split']['test'][1]) - pd.Timedelta(hours=24)
            local_base_idx = df.index[df.index <= test_end][-1]
            target_date = local_base_idx + pd.Timedelta(hours=1)

        # 1. Fetch Real Weather Data from Open-Meteo
        lat, lon = 28.5458, 77.2732
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&forecast_days=2"
        resp = requests.get(weather_url).json()
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_weather = []
        times = [datetime.fromisoformat(t) for t in resp['hourly']['time']]
        try:
            start_idx = times.index(current_hour)
        except ValueError:
            start_idx = 0
            
        for i in range(start_idx + 1, start_idx + 25):
            if i < len(times):
                future_weather.append({
                    'temp': resp['hourly']['temperature_2m'][i],
                    'humidity': resp['hourly']['relative_humidity_2m'][i],
                    'wind': resp['hourly']['wind_speed_10m'][i]
                })
            else:
                future_weather.append(future_weather[-1])

        # 2. Automatically derive calendar context for the target date
        in_holidays = holidays.IN(years=[target_date.year])
        is_holiday = 1 if target_date.date() in in_holidays else 0
        is_weekend = 1 if target_date.dayofweek >= 5 else 0
        is_working_day = 0 if (is_holiday or is_weekend) else 1
        
        # User explicitly sets exam state
        is_exam = int(req_data.get('is_exam', 0))

        overrides = {
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'is_working_day': is_working_day,
            'is_exam': is_exam,
            'is_mid_sem': is_exam,
            'is_end_sem': 0,
            'is_pre_exam': 0,
            'is_post_exam': 0
        }

        # Base state features
        X_hist = df.loc[local_base_idx, config['hist_features']].values.astype(float)
        
        curr_row = df.loc[local_base_idx].copy()
        for k, v in overrides.items():
            if k in curr_row: curr_row[k] = v
            
        X_curr = []
        for f in config['current_features']:
            X_curr.append(curr_row[f])
        X_curr = np.array(X_curr).astype(float)
        
        X_fut = []
        # Check for What-If Overrides
        ov_acad = req_data.get('academic_occupancy')
        ov_lec = req_data.get('lecture_occupancy')
        ov_lib = req_data.get('library_occupancy')
        ov_hos = req_data.get('hostel_occupancy')
        ov_mess = req_data.get('mess_occupancy')

        forecast_labels = []
        for h in range(1, 25):
            t_fut = local_base_idx + pd.Timedelta(hours=h)
            actual_forecast_time = target_date + timedelta(hours=h-1)
            forecast_labels.append(actual_forecast_time.strftime('%Y-%m-%d %H:00'))
            
            # Dynamic calendar context
            ctx = get_calendar_context(actual_forecast_time.strftime('%Y-%m-%d'))
                
            fut_vals = []
            for f in config['future_features_per_hour']:
                if f == 'fc_academic_occupancy': fut_vals.append(float(ov_acad) if ov_acad is not None else (df.loc[t_fut - pd.Timedelta(hours=168), 'academic_occupancy'] if (t_fut - pd.Timedelta(hours=168)) in df.index else 50))
                elif f == 'fc_lecture_occupancy': fut_vals.append(float(ov_lec) if ov_lec is not None else (df.loc[t_fut - pd.Timedelta(hours=168), 'lecture_occupancy'] if (t_fut - pd.Timedelta(hours=168)) in df.index else 20))
                elif f == 'fc_library_occupancy': fut_vals.append(float(ov_lib) if ov_lib is not None else (df.loc[t_fut - pd.Timedelta(hours=168), 'library_occupancy'] if (t_fut - pd.Timedelta(hours=168)) in df.index else 20))
                elif f == 'fc_hostel_occupancy': fut_vals.append(float(ov_hos) if ov_hos is not None else (df.loc[t_fut - pd.Timedelta(hours=168), 'hostel_occupancy'] if (t_fut - pd.Timedelta(hours=168)) in df.index else 40))
                elif f == 'fc_mess_occupancy': fut_vals.append(float(ov_mess) if ov_mess is not None else (df.loc[t_fut - pd.Timedelta(hours=168), 'mess_occupancy'] if (t_fut - pd.Timedelta(hours=168)) in df.index else 15))
                elif f == 'fc_temperature_c': fut_vals.append(future_weather[h-1]['temp'])
                elif f == 'fc_humidity_pct': fut_vals.append(future_weather[h-1]['humidity'])
                elif f == 'fc_wind_speed_mps': fut_vals.append(future_weather[h-1]['wind'])
                elif f == 'hour_sin': fut_vals.append(np.sin(2 * np.pi * actual_forecast_time.hour / 24))
                elif f == 'hour_cos': fut_vals.append(np.cos(2 * np.pi * actual_forecast_time.hour / 24))
                elif f == 'dow_sin': fut_vals.append(np.sin(2 * np.pi * actual_forecast_time.weekday() / 7))
                elif f == 'dow_cos': fut_vals.append(np.cos(2 * np.pi * actual_forecast_time.weekday() / 7))
                elif f == 'month_sin': fut_vals.append(np.sin(2 * np.pi * actual_forecast_time.month / 12))
                elif f == 'month_cos': fut_vals.append(np.cos(2 * np.pi * actual_forecast_time.month / 12))
                elif f == 'is_weekend': fut_vals.append(ctx['is_weekend'])
                elif f == 'is_holiday': fut_vals.append(ctx['is_holiday'])
                elif f == 'is_working_day': fut_vals.append(ctx['is_working_day'])
                elif f == 'activity_level': fut_vals.append(float(req_data.get('activity_level', 1)))
                elif f == 'is_exam': fut_vals.append(req_data.get('is_exam', 0))
                elif f == 'is_mid_sem': fut_vals.append(req_data.get('is_mid_sem', 0))
                elif f == 'is_end_sem': fut_vals.append(req_data.get('is_end_sem', 0))
                elif f == 'is_pre_exam': fut_vals.append(req_data.get('is_pre_exam', 0))
                elif f == 'is_post_exam': fut_vals.append(req_data.get('is_post_exam', 0))
                else:
                    fut_vals.append(0)
            X_fut.extend(fut_vals)
            
        X_fut = np.array(X_fut).astype(float)
        X = np.concatenate([X_hist, X_curr, X_fut])
        
        # 3. Predict
        X_scaled = scaler.transform([X])
        y_pred_scaled = model.predict(X_scaled, verbose=0)[0]
        y_pred = y_pred_scaled * ystd + ymean
        
        # History
        hist_start = local_base_idx - pd.Timedelta(hours=23)
        historical_load = df.loc[hist_start:local_base_idx, 'total_load_kw'].values.tolist()
        
        return jsonify({
            'success': True,
            'labels': forecast_labels,
            'predictions': y_pred.tolist(),
            'weather': future_weather,
            'history': historical_load,
            'calendar_context': {
                'is_holiday': bool(is_holiday),
                'is_weekend': bool(is_weekend),
                'is_working_day': bool(is_working_day)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
