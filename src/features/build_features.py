import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def build_features():
    print("Building features...")
    df = pd.read_csv('data/raw/data.csv')
    
    # Target Encoding
    # Target is 'Dropout', 'Graduate', 'Enrolled'
    # We can map them to 0, 1, 2 or keep as strings for some models. 
    # For general compatibility, let's encode them.
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    print(f"Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Separate features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Identify categorical and numerical columns
    # In this dataset, most 'categorical' features are already integer encoded (e.g., Marital Status).
    # However, for models like Linear Regression, OneHotEncoding might be better.
    # For Tree-based models, integer encoding is often fine.
    # Given we will use XGBoost/CatBoost, we can keep them as is or use OneHot.
    # Let's use OneHot for true categorical variables to be safe and robust.
    # But wait, the dataset description says many are already encoded.
    # Let's check the column names.
    
    # For this baseline, we will scale numerical features.
    # We'll treat all inputs as numerical for now since they are encoded.
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    # Save as numpy arrays or pandas dfs
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=X.columns).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/X_test.csv', index=False)
    
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    build_features()
