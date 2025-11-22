import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, f1_score
import os

def objective(trial):
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    
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
    
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    
    return accuracy

def tune_hyperparameters():
    print("Starting hyperparameter tuning with Optuna...")
    mlflow.set_experiment("student_dropout_prediction_tuning")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    # Train best model
    print("Training best model...")
    best_params = trial.params
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    
    with mlflow.start_run(run_name="Best_XGBoost"):
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"Best Model - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.xgboost.log_model(best_model, "best_xgboost_model")
        
        # Save locally for API
        best_model.save_model("src/models/best_model.json")
        print("Best model saved to src/models/best_model.json")

if __name__ == "__main__":
    tune_hyperparameters()
