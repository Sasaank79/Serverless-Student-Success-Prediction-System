import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_advanced_models():
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    
    mlflow.set_experiment("student_dropout_prediction_advanced")
    
    models = {
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(random_state=42, verbose=0)
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)
            
            # Metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            try:
                roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
            except ValueError:
                roc_auc = 0.0
            
            print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log params
            mlflow.log_params(model.get_params())
            
            # Log model
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, name)
            elif name == "LightGBM":
                mlflow.lightgbm.log_model(model, name)
            elif name == "CatBoost":
                mlflow.catboost.log_model(model, name)
            
            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'reports/figures/cm_{name}.png')
            mlflow.log_artifact(f'reports/figures/cm_{name}.png')
            plt.close()

if __name__ == "__main__":
    train_advanced_models()
