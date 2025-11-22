# Student Success AI - Serverless Pipeline

Deploys an XGBoost ensemble model to AWS Lambda using Docker containers. 
Handles cold starts via provisioned concurrency and ensures high 
availability for inference requests.

## Architecture

- **Inference**: AWS Lambda (Python 3.9 container)
- **Model**: XGBoost Classifier
- **Infrastructure**: Docker, AWS ECR

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

## Deployment

Build and push the Docker image:

```bash
docker build -t student-success-ai -f lambda/Dockerfile .
```
