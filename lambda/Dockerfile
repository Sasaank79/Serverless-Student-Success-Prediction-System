FROM python:3.9-slim

WORKDIR /var/task

# Install system dependencies (cmake & libgomp for LightGBM)
# Clean up apt cache immediately to save space
RUN apt-get update && \
    apt-get install -y cmake libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create a minimal requirements file for Lambda (exclude heavy dev tools)
RUN echo "fastapi" > requirements_lambda.txt && \
    echo "uvicorn" >> requirements_lambda.txt && \
    echo "pydantic" >> requirements_lambda.txt && \
    echo "joblib" >> requirements_lambda.txt && \
    echo "pandas" >> requirements_lambda.txt && \
    echo "numpy" >> requirements_lambda.txt && \
    echo "scikit-learn" >> requirements_lambda.txt && \
    echo "xgboost" >> requirements_lambda.txt && \
    echo "lightgbm" >> requirements_lambda.txt && \
    echo "catboost" >> requirements_lambda.txt && \
    echo "mangum" >> requirements_lambda.txt && \
    echo "awslambdaric" >> requirements_lambda.txt

# Install dependencies with --no-cache-dir to save space
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements_lambda.txt --no-cache-dir

COPY src/ src/
COPY src/models/stacking_model_tuned.pkl src/models/stacking_model_tuned.pkl

# Create handler
RUN echo "from mangum import Mangum\nfrom src.api.main import app\nhandler = Mangum(app)" > handler.py

# Set entrypoint for Lambda RIC
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "handler.handler" ]
