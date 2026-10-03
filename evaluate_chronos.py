import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv("ultimate_campus_dataset.csv")
    
    # AutoGluon expects an item_id and timestamp column
    df['item_id'] = 'campus'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # We want to forecast the last 5 days (120 hours)
    prediction_length = 120
    
    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    print(f"Total dataset length: {len(ts_df)}")
    
    # Split into train (everything except last 5 days) and test (last 5 days)
    # AutoGluon's slice_by_timestep can be used, or just simple df slicing before conversion
    train_df = df.iloc[:-prediction_length]
    test_df = df.iloc[-prediction_length:]
    
    train_data = TimeSeriesDataFrame.from_data_frame(
        train_df, id_column="item_id", timestamp_column="timestamp"
    ).convert_frequency(freq="H")
    test_data = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp"
    ).convert_frequency(freq="H")
    
    print(f"Train data length: {len(train_data)}")
    print(f"Prediction length: {prediction_length} hours (5 days)")
    
    # Initialize Predictor
    print("\nInitializing Chronos-Bolt predictor...")
    # We use Chronos-Bolt-Base which is very fast and uses minimal VRAM
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="campus_load",
        eval_metric="sMAPE",
        freq="H"
    )
    
    # Fit the model
    # For Chronos, 'fit' largely just loads the pretrained weights and prepares the pipeline
    predictor.fit(
        train_data,
        hyperparameters={
            "Chronos": {"model_path": "amazon/chronos-bolt-base"}
        },
        time_limit=600 # 10 mins max
    )
    
    print("\nGenerating forecasts for the next 120 hours...")
    predictions = predictor.predict(train_data)
    
    print("\nEvaluating accuracy on the last 5 days...")
    # Evaluate compares predictions vs the actuals in test_data
    # Note: evaluate returns a dictionary of metrics if multiple are specified, or a single float
    # We can use leaderboard to get a nice dataframe of metrics
    leaderboard = predictor.leaderboard(test_data)
    print("\nModel Leaderboard / Accuracy Metrics:")
    print(leaderboard[['model', 'score_test', 'score_val']])
    
    # Calculate additional metrics manually for clarity
    y_pred = predictions['mean'].values
    y_true = test_df['campus_load'].values
    
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\nCustom Evaluation:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Plotting the results
    plt.figure(figsize=(15, 6))
    
    # Plot the last 10 days of training data for context
    context_length = 240
    history = train_df.iloc[-context_length:]
    
    plt.plot(history['timestamp'], history['campus_load'], label='Historical Data', color='black')
    plt.plot(test_df['timestamp'], test_df['campus_load'], label='Actual Data (Last 5 Days)', color='green')
    
    # The predictions index is a MultiIndex (item_id, timestamp)
    pred_timestamps = predictions.index.get_level_values('timestamp')
    plt.plot(pred_timestamps, predictions['mean'], label='Chronos-Bolt Forecast', color='blue', linestyle='--')
    
    # Plot prediction intervals (0.1 to 0.9 quantiles if available)
    if '0.1' in predictions.columns and '0.9' in predictions.columns:
        plt.fill_between(pred_timestamps, predictions['0.1'], predictions['0.9'], color='blue', alpha=0.2, label='80% Confidence Interval')
        
    plt.title("Campus Electrical Load Forecast using Chronos-Bolt-Base")
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chronos_forecast_5days.png")
    print("\nSaved plot to chronos_forecast_5days.png")

if __name__ == "__main__":
    main()
