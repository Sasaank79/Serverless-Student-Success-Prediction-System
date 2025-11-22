import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os

def train_best_model():
    print("Training best model...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    
    # Best params from previous tuning (approximate or default good ones)
    # Since we lost the exact output, we'll use a robust set.
    best_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'booster': 'gbtree',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    
    # Save
    model.save_model("src/models/best_model.json")
    print("Model saved to src/models/best_model.json")

if __name__ == "__main__":
    train_best_model()
