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
    
    # Indian Holidays
    in_holidays = holidays.IN(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda x: 1 if x.date() in in_holidays else 0)
    
    # Indian University Exam Periods (May and December roughly)
    df['is_exam_period'] = 0
    df.loc[(df.index.month == 5) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    df.loc[(df.index.month == 12) & (df.index.day >= 10) & (df.index.day <= 24), 'is_exam_period'] = 1
    
    return df

def create_lag_and_rolling_features(df, target_col='campus_load'):
    df = df.copy()
    # Shift lags by exact hours
    df['load_lag_24'] = df[target_col].shift(24)
    df['load_lag_48'] = df[target_col].shift(48)
    df['load_lag_168'] = df[target_col].shift(168)
    
    # Rolling features with min_periods to prevent cascade NaNs
    df['rolling_mean_24h'] = df[target_col].shift(24).rolling(window=24, min_periods=12).mean()
    df['rolling_std_24h'] = df[target_col].shift(24).rolling(window=24, min_periods=12).std()
    
    return df

def main():
    print("Loading raw IIITD high-resolution data...")
    # Load the real dataset
    raw_df = pd.read_csv(r"d:\hackathon campus electrical load assessment\iiitd_data\energy_dataset\all_buildings_power.csv")
    
    # Convert Unix timestamp to Datetime
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], unit='s')
    raw_df.set_index('timestamp', inplace=True)
    
    print("Aggregating building-level meters to Campus Total Load...")
    # Sum across all building columns to get total campus load.
    # Note: sum(skipna=True) handles missing individual building meters gracefully.
    raw_df['campus_load'] = raw_df.sum(axis=1, skipna=True)
    
    print("Resampling from 1-minute to 1-hour resolution...")
    # Resample to hourly averages
    hourly_df = pd.DataFrame(raw_df['campus_load'].resample('1h').mean())
    
    print("Handling missing gaps safely...")
    hourly_df['is_missing'] = hourly_df['campus_load'].isna().astype(int)
    # Interpolate only small gaps (up to 3 hours). Larger gaps stay NaN so they don't corrupt ML.
    hourly_df['campus_load'] = hourly_df['campus_load'].interpolate(method='linear', limit=3)
    
    print("Generating context features (Indian Holidays, Time)...")
    hourly_df = create_time_features(hourly_df)
    
    print("Generating lag and rolling features...")
    hourly_df = create_lag_and_rolling_features(hourly_df, target_col='campus_load')
    
    # Drop rows that are inherently NaN because of the 168-hour lag (first week)
    hourly_df = hourly_df.dropna(subset=['load_lag_168'])
    
    output_file = r"d:\hackathon campus electrical load assessment\professional_campus_dataset.csv"
    hourly_df.to_csv(output_file)
    print(f"\nSuccess! Real IIITD Professional Dataset generated: {output_file}")
    print(f"Dataset shape: {hourly_df.shape}")
    print(hourly_df.head())

if __name__ == "__main__":
    main()
