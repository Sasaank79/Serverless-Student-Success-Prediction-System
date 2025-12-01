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
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPipeline:
    def __init__(self, data_path='data/processed_v2/'):
        self.data_path = data_path
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        self.best_params = {}

    def load_data(self):
        logger.info("Loading processed data...")
        try:
            self.X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'))
            self.X_val = pd.read_csv(os.path.join(self.data_path, 'X_val.csv'))
            self.y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv')).values.ravel()
            self.y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv')).values.ravel()
            logger.info(f"Data loaded. Train shape: {self.X_train.shape}, Val shape: {self.X_val.shape}")
        except FileNotFoundError:
            logger.error("Data files not found. Please run feature engineering first.")
            raise

    def _optimize_xgboost(self, trial):
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
        
        # Simple hold-out validation for speed in optimization loop
        from sklearn.model_selection import train_test_split
        X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
        
        model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return accuracy_score(y_v, preds)

    def _optimize_lightgbm(self, trial):
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
        X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
        
        model = lgb.LGBMClassifier(**param)
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return accuracy_score(y_v, preds)

    def _optimize_catboost(self, trial):
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
        X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
        
        model = cb.CatBoostClassifier(**param)
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return accuracy_score(y_v, preds)

    def optimize_hyperparameters(self, n_trials=10):
        logger.info(f"Starting Hyperparameter Optimization ({n_trials} trials each)...")
        
        # XGBoost
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(self._optimize_xgboost, n_trials=n_trials)
        self.best_params['xgb'] = study_xgb.best_params
        self.best_params['xgb']['use_label_encoder'] = False
        self.best_params['xgb']['eval_metric'] = 'mlogloss'
        
        # LightGBM
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(self._optimize_lightgbm, n_trials=n_trials)
        self.best_params['lgb'] = study_lgb.best_params
        self.best_params['lgb']['verbose'] = -1
        
        # CatBoost
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(self._optimize_catboost, n_trials=n_trials)
        self.best_params['cat'] = study_cat.best_params
        self.best_params['cat']['verbose'] = 0
        
        logger.info("Optimization Complete.")

    def train_stacking_ensemble(self):
        logger.info("Training Stacking Ensemble with optimized parameters...")
        
        estimators = [
            ('xgb', xgb.XGBClassifier(**self.best_params['xgb'], random_state=42)),
            ('lgb', lgb.LGBMClassifier(**self.best_params['lgb'], random_state=42)),
            ('cat', cb.CatBoostClassifier(**self.best_params['cat'], random_state=42))
        ]
        
        final_estimator = LogisticRegression(max_iter=1000)
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training Complete.")

    def evaluate(self):
        logger.info("Evaluating Model...")
        y_pred = self.model.predict(self.X_val)
        y_prob = self.model.predict_proba(self.X_val)
        
        acc = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        try:
            roc_auc = roc_auc_score(self.y_val, y_prob, multi_class='ovr')
        except:
            roc_auc = 0.0
            
        logger.info(f"Results - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Save Confusion Matrix
        cm = confusion_matrix(self.y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
        plt.title('Confusion Matrix - Stacking Ensemble')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/figures/cm_pipeline.png')
        plt.close()
        logger.info("Confusion matrix saved to reports/figures/cm_pipeline.png")

    def save_model(self, path='src/models/stacking_model_tuned.pkl'):
        logger.info(f"Saving model to {path}...")
        joblib.dump(self.model, path)
        logger.info("Model saved.")

    def package_and_deploy_instructions(self):
        print("\n" + "="*50)
        print("🚀 DEPLOYMENT INSTRUCTIONS")
        print("="*50)
        print("To package and deploy this model to AWS Lambda:")
        print("\n1. Build the Docker Image:")
        print("   docker build -t student-dropout-api -f lambda/Dockerfile .")
        print("\n2. Tag for ECR (Replace with your URI):")
        print("   docker tag student-dropout-api YOUR_ECR_URI:latest")
        print("\n3. Push to ECR:")
        print("   docker push YOUR_ECR_URI:latest")
        print("\n4. Update Lambda Function:")
        print("   (Use AWS Console or CLI to point Lambda to the new image)")
        print("="*50 + "\n")

def run_pipeline():
    pipeline = ModelPipeline()
    pipeline.load_data()
    pipeline.optimize_hyperparameters(n_trials=5) # Reduced for demo speed
    pipeline.train_stacking_ensemble()
    pipeline.evaluate()
    pipeline.save_model()
    pipeline.package_and_deploy_instructions()

if __name__ == "__main__":
    run_pipeline()
