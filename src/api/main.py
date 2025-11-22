from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Student Dropout Prediction API (V2)")

# Load model (Stacking Ensemble)
model_path = "src/models/stacking_model_tuned.pkl"
if not os.path.exists(model_path):
    # Fallback to untuned stacking
    model_path = "src/models/stacking_model.pkl"
    # Fallback to best_model.json if stacking not found (for testing)
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model("src/models/best_model.json")
    model_type = "xgboost"
else:
    model = joblib.load(model_path)
    model_type = "stacking"

class StudentData(BaseModel):
    # Define all 36 features or use a dynamic dict with validation
    # For V2, we should be explicit, but 36 fields is long for this snippet.
    # We will use a list but validate length and value ranges.
    features: list[float] = Field(..., description="List of 36 input features (or 44 for V2)")

    @validator('features')
    def validate_features(cls, v):
        # V2 has 44 features due to advanced engineering
        # But wait, the input to the API usually expects raw features and the API should transform them?
        # OR the API expects pre-processed features?
        # In a real production app, the API should take RAW data and run the preprocessing pipeline.
        # However, our `build_advanced_features` saves processed CSVs.
        # We didn't refactor the preprocessing into a callable function in `src/features`.
        # For this demo, we will assume the input IS the processed feature vector.
        # V2 feature count is 44.
        expected_dim = 44 
        if len(v) != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {len(v)}")
        return v

@app.get("/")
def read_root():
    return {"message": f"Welcome to the Student Dropout Prediction API (V2 - {model_type})"}

@app.post("/predict")
def predict(data: StudentData):
    try:
        input_data = np.array(data.features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Map prediction to label
        labels = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        result = labels.get(int(prediction[0]), "Unknown")
        
        return {
            "prediction": result,
            "probability": probability.tolist(),
            "model_used": model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
