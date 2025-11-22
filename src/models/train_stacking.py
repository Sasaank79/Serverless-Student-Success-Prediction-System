import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_stacking_ensemble():
    print("Loading V2 processed data...")
    X_train = pd.read_csv('data/processed_v2/X_train.csv')
    X_val = pd.read_csv('data/processed_v2/X_val.csv')
    y_train = pd.read_csv('data/processed_v2/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed_v2/y_val.csv').values.ravel()
    
    mlflow.set_experiment("student_dropout_prediction_stacking")
    
    # Base models (using default or slightly tuned params)
    # Ideally, we should use the best params found via Optuna for each.
    # For this demo, we'll use robust defaults.
    estimators = [
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1)),
        ('cat', cb.CatBoostClassifier(random_state=42, verbose=0))
    ]
    
    # Meta-learner
    final_estimator = LogisticRegression(max_iter=1000)
    
    # Stacking Classifier
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5, # Internal CV for stacking
        n_jobs=-1,
        passthrough=False # Only use meta-features
    )
    
    with mlflow.start_run(run_name="Stacking_Ensemble"):
        print("Training Stacking Ensemble...")
        clf.fit(X_train, y_train)
        
        # Evaluate on Validation Set
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0
            
        print(f"Stacking Ensemble - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(clf, "stacking_model")
        
        # Save locally
        import joblib
        joblib.dump(clf, "src/models/stacking_model.pkl")
        print("Stacking model saved to src/models/stacking_model.pkl")
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
        plt.title('Confusion Matrix - Stacking Ensemble')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/figures/cm_stacking.png')
        mlflow.log_artifact('reports/figures/cm_stacking.png')
        plt.close()

if __name__ == "__main__":
    train_stacking_ensemble()
