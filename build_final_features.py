import pandas as pd
import numpy as np
import holidays

def create_time_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Use Indian Holidays instead of US
    in_holidays = holidays.IN(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda x: 1 if x.date() in in_holidays else 0)
    
    # Exam Period
    df['is_exam_period'] = 0
    df.loc[(df.index.month == 5) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    df.loc[(df.index.month == 12) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    
    return df

def create_lag_and_rolling_features(df, target_col='campus_load'):
    df = df.copy()
    df['load_lag_24'] = df[target_col].shift(24)
    df['load_lag_48'] = df[target_col].shift(48)
    df['load_lag_168'] = df[target_col].shift(168)
    
    # Added min_periods=12 so a single missing hour doesn't destroy the whole 24h rolling window
    df['rolling_mean_24h'] = df[target_col].shift(24).rolling(window=24, min_periods=12).mean()
    df['rolling_std_24h'] = df[target_col].shift(24).rolling(window=24, min_periods=12).std()
    
    return df

def main():
    print("Loading synthetic campus dataset...")
    df = pd.read_csv("campus_load_dataset.csv", index_col='timestamp', parse_dates=True)
    
    # FIX: Interpolate FIRST before calculating lags and rolling features!
    print("Handling missing values (Interpolation)...")
    df['is_missing'] = df['campus_load'].isna().astype(int)
    df['campus_load'] = df['campus_load'].interpolate(method='linear', limit=3)
    
    print("Applying time and calendar features...")
    df = create_time_features(df)
    
    print("Applying lag and rolling statistics...")
    df = create_lag_and_rolling_features(df, target_col='campus_load')
    
    # Drop rows that naturally have NaN because of the 168-hour lag (the first week of data)
    df = df.dropna(subset=['load_lag_168'])
    
    print("Saving final training dataset...")
    df.to_csv("final_training_data_v2.csv")
    
    print(f"Done! Final dataset shape: {df.shape}")
    print(f"Remaining NaNs in load: {df['campus_load'].isna().sum()}")

if __name__ == "__main__":
    main()
