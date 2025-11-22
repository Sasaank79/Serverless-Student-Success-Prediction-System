import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

def download_data():
    print("Downloading dataset from UCI Repository...")
    
    # fetch dataset 
    # ID 697 is for "Predict Students' Dropout and Academic Success"
    dataset = fetch_ucirepo(id=697) 
    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets 
    
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/raw/data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    download_data()
