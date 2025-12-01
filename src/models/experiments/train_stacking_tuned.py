import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def load_data():
    print("Loading V2 processed data...")
    X_train = pd.read_csv('data/processed_v2/X_train.csv')
    X_val = pd.read_csv('data/processed_v2/X_val.csv')
    y_train = pd.read_csv('data/processed_v2/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed_v2/y_val.csv').values.ravel()
    return X_train, X_val, y_train, y_val

def optimize_xgboost(trial, X, y):
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': 3,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }
    
    # Use CV for robust optimization
    # For speed in this demo, we'll use a simple train/val split inside optuna if passed, 
    # but here we are passed X_train which is already a split. 
    # To be proper, we should use cross_val_score here, but it's slow.
    # We'll use a hold-out set from X (which is X_train from main).
    # Actually, let's just use a simple fit on X and predict on a subset if possible, 
    # or rely on the fact that X is the training set. 
    # We need a validation set for early stopping or metric calculation.
    # Let's assume X, y are the training set. We'll split it internally.
    
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return accuracy_score(y_v, preds)

def optimize_lightgbm(trial, X, y):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': 3,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(**param)
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return accuracy_score(y_v, preds)

def optimize_catboost(trial, X, y):
    param = {
        'loss_function': 'MultiClass',
        'verbose': 0,
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = cb.CatBoostClassifier(**param)
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return accuracy_score(y_v, preds)

def train_tuned_stacking():
    X_train, X_val, y_train, y_val = load_data()
    
    mlflow.set_experiment("student_dropout_prediction_tuned_stacking")
    
    # 1. Optimize Base Learners
    print("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_train, y_train), n_trials=10) # 10 trials for speed
    best_xgb_params = study_xgb.best_params
    best_xgb_params['use_label_encoder'] = False
    best_xgb_params['eval_metric'] = 'mlogloss'
    print(f"Best XGB params: {best_xgb_params}")
    
    print("Optimizing LightGBM...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda trial: optimize_lightgbm(trial, X_train, y_train), n_trials=10)
    best_lgb_params = study_lgb.best_params
    best_lgb_params['verbose'] = -1
    print(f"Best LGB params: {best_lgb_params}")
    
    print("Optimizing CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: optimize_catboost(trial, X_train, y_train), n_trials=10)
    best_cat_params = study_cat.best_params
    best_cat_params['verbose'] = 0
    print(f"Best Cat params: {best_cat_params}")
    
    # 2. Build Stacking Ensemble with Optimized Learners
    estimators = [
        ('xgb', xgb.XGBClassifier(**best_xgb_params, random_state=42)),
        ('lgb', lgb.LGBMClassifier(**best_lgb_params, random_state=42)),
        ('cat', cb.CatBoostClassifier(**best_cat_params, random_state=42))
    ]
    
    final_estimator = LogisticRegression(max_iter=1000)
    
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    with mlflow.start_run(run_name="Tuned_Stacking_Ensemble"):
        print("Training Tuned Stacking Ensemble...")
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0
            
        print(f"Tuned Stacking Ensemble - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log params (nested)
        mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})
        mlflow.log_params({f"lgb_{k}": v for k, v in best_lgb_params.items()})
        mlflow.log_params({f"cat_{k}": v for k, v in best_cat_params.items()})
        
        # Log model
        mlflow.sklearn.log_model(clf, "tuned_stacking_model")
        
        # Save locally
        joblib.dump(clf, "src/models/stacking_model_tuned.pkl")
        print("Tuned Stacking model saved to src/models/stacking_model_tuned.pkl")
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
        plt.title('Confusion Matrix - Tuned Stacking')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/figures/cm_tuned_stacking.png')
        mlflow.log_artifact('reports/figures/cm_tuned_stacking.png')
        plt.close()

if __name__ == "__main__":
    train_tuned_stacking()
