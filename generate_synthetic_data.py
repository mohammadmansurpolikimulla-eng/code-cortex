import pandas as pd
import numpy as np
import holidays

def generate_synthetic_campus_data(start_date="2022-01-01", end_date="2023-12-31 23:00:00", output_file="campus_load_dataset.csv"):
    """
    Generates a highly realistic synthetic electricity dataset mimicking an educational campus or lab.
    """
    # 1. Fix Truncated End: Added 23:00:00 to the end_date to ensure the last day is full
    date_rng = pd.date_range(start=start_date, end=end_date, freq='h')
    df = pd.DataFrame(index=date_rng)
    
    # 2. Weather Simulation
    day_of_year = df.index.dayofyear
    hour_of_day = df.index.hour
    
    temp_annual = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
    temp_daily = 5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    df['air_temperature'] = temp_annual + temp_daily + np.random.normal(0, 2, len(df))
    df['humidity'] = np.clip(60 + 20 * np.cos(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5, len(df)), 10, 100)
    
    # 3. Campus Load Simulation
    base_load = 500
    hourly_profile = -200 * np.cos(2 * np.pi * (hour_of_day - 2) / 24) + 100
    
    # Break the identical M-F pattern (Monday highest, Friday lowest)
    dow_multipliers = np.array([1.05, 1.02, 1.00, 0.98, 0.92, 0.60, 0.55])
    dow_mult = dow_multipliers[df.index.dayofweek]
    
    # 4. Inject Holiday Signal (Indian Holidays)
    in_holidays = holidays.IN(years=[2022, 2023])
    is_holiday = df.index.map(lambda x: x.date() in in_holidays)
    holiday_mult = np.where(is_holiday, 0.65, 1.0) # 35% drop on Indian holidays
    
    # 5. Inject Exam Signal (Higher load due to late-night study / 24h library)
    is_exam = ((df.index.month == 5) & (df.index.day >= 10) & (df.index.day <= 24)) | \
              ((df.index.month == 12) & (df.index.day >= 10) & (df.index.day <= 24))
    exam_mult = np.where(is_exam, 1.15, 1.0) # 15% increase
    
    # Weather dependency (AC load)
    ac_load = np.maximum(df['air_temperature'] - 28, 0) * 12
    
    # Combine signals + extra noise for realism
    load = (base_load + hourly_profile) * dow_mult * holiday_mult * exam_mult + ac_load
    load += np.random.normal(0, 30, len(df)) # Increased noise
    df['campus_load'] = np.maximum(load, 100)
    
    # 6. Inject Missing Values (to practice interpolation)
    missing_indices = np.random.choice(len(df), size=int(len(df)*0.02), replace=False)
    df.loc[df.index[missing_indices], 'campus_load'] = np.nan
    
    df.to_csv(output_file, index_label='timestamp')
    print(f"Success! Generated synthetic dataset: {output_file} ({len(df)} rows)")

if __name__ == "__main__":
    generate_synthetic_campus_data()
