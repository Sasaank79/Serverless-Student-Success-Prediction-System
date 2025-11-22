import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading data for EDA...")
    df = pd.read_csv('data/raw/data.csv')
    
    # Basic Info
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum().sum())
    
    # Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Target', data=df)
    plt.title('Target Distribution')
    plt.savefig('reports/figures/target_distribution.png')
    print("Saved target distribution plot to reports/figures/target_distribution.png")
    
    # Correlation Matrix (Numerical only)
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('reports/figures/correlation_matrix.png')
    print("Saved correlation matrix to reports/figures/correlation_matrix.png")

if __name__ == "__main__":
    perform_eda()
