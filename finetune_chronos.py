import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os

def main():
    print("Loading dataset for fine-tuning...")
    df = pd.read_csv("ultimate_campus_dataset.csv")
    
    df['item_id'] = 'campus'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # We want to forecast the last 5 days (120 hours)
    prediction_length = 120
    
    # Split into train (everything except last 5 days) and test (last 5 days)
    train_df = df.iloc[:-prediction_length]
    test_df = df.iloc[-prediction_length:]
    
    train_data = TimeSeriesDataFrame.from_data_frame(
        train_df, id_column="item_id", timestamp_column="timestamp"
    ).convert_frequency(freq="H")
    
    test_data = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp" 
    ).convert_frequency(freq="H")
    
    print("\nInitializing Chronos-Bolt predictor for FINE-TUNING...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="campus_load",
        eval_metric="sMAPE",
        freq="H"
    )
    
    # Enable fine-tuning!
    # We use LoRA to efficiently fine-tune Chronos on the GPU
    predictor.fit(
        train_data,
        hyperparameters={
            "Chronos": {
                "model_path": "amazon/chronos-bolt-base",
                "fine_tune": True, 
                "optimization.learning_rate": 1e-4, 
                "optimization.max_epochs": 3,
                "batch_size": 8
            }
        },
        time_limit=1800 # Allow up to 30 minutes for fine-tuning
    )
    
    print("\nEvaluating Fine-Tuned Model...")
    leaderboard = predictor.leaderboard(test_data)
    print("\nLeaderboard / Accuracy Metrics after Fine-Tuning:")
    print(leaderboard[['model', 'score_test', 'score_val']])

if __name__ == "__main__":
    main()
