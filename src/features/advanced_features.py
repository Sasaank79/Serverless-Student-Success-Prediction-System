import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

def build_advanced_features():
    print("Building advanced features...")
    df = pd.read_csv('data/raw/data.csv')
    
    # --- 1. Advanced Feature Engineering ---
    # Ratios
    # Avoid division by zero by adding a small epsilon or handling zeros
    df['approved_enrolled_ratio_1st'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1e-5)
    df['approved_enrolled_ratio_2nd'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1e-5)
    
    df['eval_enrolled_ratio_1st'] = df['Curricular units 1st sem (evaluations)'] / (df['Curricular units 1st sem (enrolled)'] + 1e-5)
    df['eval_enrolled_ratio_2nd'] = df['Curricular units 2nd sem (evaluations)'] / (df['Curricular units 2nd sem (enrolled)'] + 1e-5)
    
    # Aggregations across semesters
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['total_enrolled'] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
    df['total_grade'] = df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
    
    # Grade change (Trend)
    df['grade_change'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    
    print("Added interaction and aggregation features.")
    
    # --- 2. Encoding ---
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    print(f"Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # --- 3. Splitting ---
    # Stratified split to maintain class distribution in test/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Original Train shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
    
    # --- 4. SMOTE (Train only) ---
    # Only oversample the minority classes (Dropout usually)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Train shape: {X_train_resampled.shape}, Class distribution: {np.bincount(y_train_resampled)}")
    
    # --- 5. Scaling ---
    scaler = StandardScaler()
    # Fit on resampled train data
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 6. Save ---
    os.makedirs('data/processed_v2', exist_ok=True)
    
    feature_names = X.columns.tolist()
    
    pd.DataFrame(X_train_scaled, columns=feature_names).to_csv('data/processed_v2/X_train.csv', index=False)
    pd.DataFrame(X_val_scaled, columns=feature_names).to_csv('data/processed_v2/X_val.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names).to_csv('data/processed_v2/X_test.csv', index=False)
    
    pd.DataFrame(y_train_resampled, columns=['Target']).to_csv('data/processed_v2/y_train.csv', index=False)
    pd.DataFrame(y_val, columns=['Target']).to_csv('data/processed_v2/y_val.csv', index=False)
    pd.DataFrame(y_test, columns=['Target']).to_csv('data/processed_v2/y_test.csv', index=False)
    
    print("Processed V2 data saved to data/processed_v2/")

if __name__ == "__main__":
    build_advanced_features()
