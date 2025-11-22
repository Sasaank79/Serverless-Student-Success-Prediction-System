import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

def explain_model():
    print("Loading data and model for explainability...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model("src/models/best_model.json")
    
    # Feature Importance (Built-in)
    print("Generating Feature Importance plot...")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, height=0.5)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    print("Saved feature importance to reports/figures/feature_importance.png")
    
    # SHAP Analysis
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values len: {len(shap_values)}")
        print(f"SHAP values[0] shape: {shap_values[0].shape}")
    else:
        print(f"SHAP values shape: {shap_values.shape}")

    # Summary Plot
    plt.figure(figsize=(10, 8))
    # For multiclass, shap_values is often a list of arrays [n_samples, n_features] for each class
    # Or an array [n_samples, n_features, n_classes] depending on version.
    # XGBoost usually returns [n_samples, n_features, n_classes] or list.
    
    if isinstance(shap_values, list):
        # Plot for class 0
        shap.summary_plot(shap_values[0], X_test, show=False)
    elif len(shap_values.shape) == 3:
         shap.summary_plot(shap_values[:, :, 0], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
        
    plt.title('SHAP Summary Plot (Dropout Class)')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_dropout.png')
    print("Saved SHAP summary to reports/figures/shap_summary_dropout.png")
    
    # Summary Plot (Bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Bar)')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_bar.png')
    print("Saved SHAP bar plot to reports/figures/shap_bar.png")

if __name__ == "__main__":
    explain_model()
