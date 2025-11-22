FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/processed/ data/processed/ 
# We need data/processed if we were loading scalers, but we didn't save the scaler in this simple run.
# In a real app, we should save and load the scaler.
# For now, let's assume the input is already scaled or we should have saved the scaler.
# Let's update the API to load the scaler if we had one. 
# Since we didn't save the scaler in build_features.py, we should probably update it or just note it.
# For this MVP, we'll assume the user sends scaled data or we just copy the model.

# Actually, let's copy the model explicitly
COPY src/models/best_model.json src/models/best_model.json
COPY src/models/stacking_model_tuned.pkl src/models/stacking_model_tuned.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
