"""Build a REAL hourly campus load dataset from the IIITD i-BLEND energy dataset.

Source: iiitd_data/energy_dataset/all_buildings_power.csv (1-minute resolution, watts)
Output: real_iiitd_hourly_dataset.csv

Design choices aligned with problem statement (JIG26_04):
- Load target     : campus_load_kw = hourly mean of sum of all building meters
- Missing meters  : is_missing flag per hour (all meters down) + gap handling note
- Holidays        : major Indian holidays 2013-2017
- Exam periods    : IIITD semester-exam heuristic (mid-Nov to early-Dec, mid-Apr to early-May)
- Calendar/lag    : cyclical encodings, weekend/holiday/exam flags, lags 24/48/168, rolling stats
"""
import numpy as np
import pandas as pd

SRC = "iiitd_data/energy_dataset/all_buildings_power.csv"
OUT = "real_iiitd_hourly_dataset.csv"

BUILDINGS = ["Academic", "Boys_main", "Boys_backup", "Facilities",
             "Girls_main", "Girls_backup", "Lecture", "Library", "Mess"]

# ---- Major Indian public holidays 2013-2017 (approximate fixed dates) ----
HOLIDAYS = set()
def add(y, m, d): HOLIDAYS.add((y, m, d))
for y in range(2013, 2018):
    add(y, 1, 26); add(y, 1, 1)                      # Republic Day, New Year
    add(y, 8, 15); add(y, 10, 2); add(y, 12, 25)     # Independence, Gandhi Jayanti, Christmas
    add(y, 5, 1)                                     # Labour Day (Delhi)
# Variable festivals (actual dates)
for d in ["2013-03-27", "2014-03-17", "2015-03-06", "2016-03-24", "2017-03-13"]: add(*map(int, d.split("-")))  # Holi
for d in ["2013-11-03", "2014-10-23", "2015-11-11", "2016-10-30", "2017-10-19"]: add(*map(int, d.split("-")))  # Diwali
for d in ["2013-10-16", "2014-10-06", "2015-09-25", "2016-09-13", "2017-09-02"]: add(*map(int, d.split("-")))  # Eid al-Adha (Bakrid)
for d in ["2013-08-09", "2014-07-29", "2015-07-18", "2016-07-07", "2017-06-27"]: add(*map(int, d.split("-")))  # Eid al-Fitr

def is_exam(ts):
    # IIITD end-semester exams: ~mid-Nov to early-Dec, mid-Apr to early-May
    return int((ts.month == 11 and ts.day >= 15) or ts.month == 12 and ts.day <= 8
               or ts.month == 4 and ts.day >= 20 or ts.month == 5 and ts.day <= 10)

def main():
    print("Reading 1-minute building power data ...")
    df = pd.read_csv(SRC)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("ts").drop(columns=["timestamp"])

    df["all_missing"] = df[BUILDINGS].isna().all(axis=1)
    df["campus_w"] = df[BUILDINGS].sum(axis=1, min_count=1)

    print("Resampling to hourly ...")
    hourly = pd.DataFrame({
        "campus_load_kw": df["campus_w"].resample("1h").mean() / 1000.0,
        "missing_minutes": df["all_missing"].resample("1h").sum().astype(int),
    })
    hourly["is_missing"] = (hourly["missing_minutes"] >= 30).astype(int)

    # forward/back-fill gaps up to 3 hours so lags are usable; flag larger gaps
    hourly["campus_load_kw"] = hourly["campus_load_kw"].interpolate(limit=3, limit_area="inside")
    hourly = hourly.dropna(subset=["campus_load_kw"]).reset_index().rename(columns={"ts": "timestamp"})

    ts = hourly["timestamp"]
    dow = ts.dt.dayofweek
    hourly["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    hourly["weekday_sin"] = np.sin(2 * np.pi * dow / 7)
    hourly["weekday_cos"] = np.cos(2 * np.pi * dow / 7)
    hourly["is_weekend"] = (dow >= 5).astype(int)
    hourly["is_holiday"] = [int((t.year, t.month, t.day) in HOLIDAYS) for t in ts]
    hourly["is_exam_period"] = ts.map(is_exam)

    s = hourly.set_index("timestamp")["campus_load_kw"]
    for lag in (24, 48, 168):
        hourly[f"load_lag_{lag}"] = s.shift(lag).values
    hourly["rolling_mean_24h"] = s.rolling(24).mean().shift(1).values
    hourly["rolling_std_24h"] = s.rolling(24).std().shift(1).values
    hourly["ratio_to_lag_168"] = s.values / s.shift(168).values

    hourly = hourly.dropna(subset=["load_lag_168", "rolling_mean_24h"]).reset_index(drop=True)
    hourly.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(hourly)} rows, {hourly['timestamp'].min()} -> {hourly['timestamp'].max()}")
    print(hourly.head(3).to_string())
    print("is_missing hours:", hourly["is_missing"].sum())

if __name__ == "__main__":
    main()
