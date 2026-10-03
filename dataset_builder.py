import pandas as pd
import numpy as np
import holidays

def create_time_features(df):
    """Adds cyclical time features and calendar flags."""
    df = df.copy()
    
    # Cyclical hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Cyclical day of week (0-6)
    df['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Weekend flag
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # US Holidays (assuming US-based BDG2 site, adjust country if needed)
    us_holidays = holidays.US(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # Mock Exam Period (e.g., mid-May and mid-December) - YOU MUST CUSTOMIZE THIS
    df['is_exam_period'] = 0
    df.loc[(df.index.month == 5) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    df.loc[(df.index.month == 12) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    
    return df

def create_lag_and_rolling_features(df, target_col='campus_load'):
    """Creates historical lag and rolling statistics features."""
    df = df.copy()
    
    # Lag features
    df['load_lag_24'] = df[target_col].shift(24)
    df['load_lag_48'] = df[target_col].shift(48)
    df['load_lag_168'] = df[target_col].shift(168) # 1 week ago
    
    # Rolling features (shifted by 24h to avoid leaking the 24h forecast horizon)
    # This means the rolling mean is calculated on the day prior to the forecast
    df['rolling_mean_24h'] = df[target_col].shift(24).rolling(window=24).mean()
    df['rolling_std_24h'] = df[target_col].shift(24).rolling(window=24).std()
    
    return df

def build_dataset(meter_df, meta_df, weather_df, target_site_id='Panther'):
    """
    Main pipeline to convert raw BDG2 data into the hackathon training dataset.
    
    Parameters:
    - meter_df: DataFrame of electricity_cleaned.csv
    - meta_df: DataFrame of metadata.csv
    - weather_df: DataFrame of weather.csv
    - target_site_id: The specific university site to isolate
    """
    
    # 1. Filter metadata for the specific university site
    site_buildings = meta_df[meta_df['site_id'] == target_site_id]['building_id'].tolist()
    print(f"Found {len(site_buildings)} buildings for site '{target_site_id}'")
    
    # 2. Extract meter readings for these buildings and sum to get Campus Load
    # Assuming meter_df is wide format (timestamp as index, building_ids as columns)
    available_buildings = [b for b in site_buildings if b in meter_df.columns]
    campus_load = meter_df[available_buildings].sum(axis=1, skipna=False)
    
    df = pd.DataFrame({'campus_load': campus_load})
    df.index = pd.to_datetime(df.index)
    
    # 3. Handle Missing Values (Interpolate short gaps, flag them)
    df['is_missing'] = df['campus_load'].isna().astype(int)
    
    # Interpolate gaps of 3 hours or less using linear interpolation
    df['campus_load'] = df['campus_load'].interpolate(method='linear', limit=3, limit_direction='forward')
    
    # 4. Merge Weather data
    weather_df.index = pd.to_datetime(weather_df.index)
    site_weather = weather_df[weather_df['site_id'] == target_site_id][['airTemperature', 'dewTemperature']]
    df = df.join(site_weather, how='left')
    
    # 5. Build Calendar and Time Features
    df = create_time_features(df)
    
    # 6. Build Lag and Rolling Features
    df = create_lag_and_rolling_features(df, target_col='campus_load')
    
    # Drop rows at the beginning that have NaN due to lags (e.g., first 168 hours)
    # But DO NOT drop internal NaNs (longer than 3h gaps), handle those in the PyTorch Dataset!
    
    return df

if __name__ == "__main__":
    print("This is a starter script. To run it, you will need to download the BDG2 data CSVs.")
    print("Download metadata.csv, weather.csv, and electricity_cleaned.csv")
    print("Then load them via pd.read_csv() and pass them to build_dataset().")
