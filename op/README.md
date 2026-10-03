# Electricity Load Planning Assistant (JIG26_04, Track A)
Direct multi-output ANN forecasting the next 24 hourly campus loads as ONE sequence (no recursion).
- Data: IIIT-D synthetic hourly dataset (2013-2017), not resampled. Exam flags are experimental/contextual, not official records.
- Features: lags (1/24/168), rolling stats (past only via shift(1)), cyclical calendar, occupancy, academic context,
  plus per-future-hour calendar/academic context, occupancy proxy (same hour previous week) and SIMULATED FORECAST WEATHER
  (lagged weather + Gaussian noise, SD 1.5 C / 5 pct / 0.8 m/s). Actual future weather and occupancy are never model inputs.
- Split: chronological 70/15/15 with a 24-sample gap; scaler fit on train only; peak threshold = train 90th percentile (994.4 kW).
- Baselines: persistence, previous day, previous week. Metrics: overall / peak / off-peak MAE and RMSE (kW).
- Outage test: 24 h simulated meter loss, reconstruction (two-sided interpolation = stress test; prev-week = causal), post-outage forecast.
- Decision-support only: the system does not control or reduce consumption.
- Weather API layer is swappable (SimulatedForecastWeather / OpenMeteoForecastWeather); the ANN is not retrained when weather changes.
