from fastapi.testclient import TestClient
from src.api.main import app
import pandas as pd
import numpy as np

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Student Dropout Prediction API"}

def test_predict():
    # Load a sample from test set (V2)
    X_test = pd.read_csv('data/processed_v2/X_test.csv')
    sample = X_test.iloc[0].tolist()
    
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Dropout", "Enrolled", "Graduate"]
    print(f"Prediction: {data['prediction']}")

if __name__ == "__main__":
    test_read_root()
    test_predict()
    print("All tests passed!")
