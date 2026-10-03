import os
import urllib.request
import pandas as pd
from dataset_builder import build_dataset

def download_with_progress(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}... (this might take a moment)")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists. Skipping download.")

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    meta_url = "https://raw.githubusercontent.com/BuildingDataGenomeProject2/BDG2_open/master/data/metadata/metadata.csv"
    weather_url = "https://raw.githubusercontent.com/BuildingDataGenomeProject2/BDG2_open/master/data/weather/weather.csv"
    elec_url = "https://media.githubusercontent.com/media/BuildingDataGenomeProject2/BDG2_open/master/data/meters/cleaned/electricity_cleaned.csv"
    
    download_with_progress(meta_url, f"{data_dir}/metadata.csv")
    download_with_progress(weather_url, f"{data_dir}/weather.csv")
    
    print("\nFetching BDG2 electricity data...")
    print("NOTE: This file is ~400MB. It will take a few minutes depending on network speed.")
    download_with_progress(elec_url, f"{data_dir}/electricity_cleaned.csv")
    
    print("\nLoading data into Pandas (this requires ~500MB of RAM)...")
    meta_df = pd.read_csv(f"{data_dir}/metadata.csv")
    weather_df = pd.read_csv(f"{data_dir}/weather.csv", index_col='timestamp')
    meter_df = pd.read_csv(f"{data_dir}/electricity_cleaned.csv", index_col='timestamp')
    
    print("\nBuilding final dataset for the 'Panther' university site (Florida)...")
    final_df = build_dataset(meter_df, meta_df, weather_df, target_site_id='Panther')
    
    output_file = "campus_load_dataset.csv"
    final_df.to_csv(output_file)
    print(f"\n✅ Success! Fully featured dataset saved to: {output_file}")
    print(f"Dataset shape: {final_df.shape}")
    print(final_df.head())

if __name__ == "__main__":
    main()
