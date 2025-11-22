import joblib
from pathlib import Path

class ModelLoader:
    """
    Manages model artifact loading and inference.
    """
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at 
{self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, data):
        return self.model.predict(data)
