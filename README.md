# 🚀 Serverless Student Dropout Prediction System  
*A full end-to-end ML pipeline: from raw data → model → API → Docker → AWS Lambda.*

This project started as a simple “train a model on the UCI student dataset” task.  
But I wanted to go beyond the usual notebook experiment. So I turned it into a **production-style ML system** with a real deployment flow, real engineering decisions, and a fully serverless inference API.

It’s not a real university product — it’s a **technical prototype** designed to show how a modern ML workflow comes together in practice. Everything from preprocessing to Dockerization to Lambda is implemented cleanly and reproducibly.

---

## 🔍 What This Project Does

- Takes student demographic + academic records  
- Learns patterns related to dropout / continuation / graduation  
- Trains a **stacking ensemble** (XGBoost + LightGBM + CatBoost → Logistic Regression)  
- Handles messy categorical/numerical data  
- Deals with class imbalance the right way (SMOTE *inside* CV only)  
- Serves predictions through a **FastAPI** endpoint  
- Packs the whole thing into a Docker image  
- Deploys it to **AWS Lambda** as a serverless API  

---

# 🧠 Why I Built It  
Most ML projects stop at “here’s my accuracy.”  
I built this to demonstrate the **entire ML lifecycle**, not just the model.

I wanted to show that I can:

- design a pipeline  
- train + tune models  
- package them correctly  
- deploy them serverlessly  
- debug real-world infra problems  

I ran into practical issues (CloudShell storage → moved to EC2 → ECR → Lambda), which made this a realistic engineering experience.

---

# 🏗️ System Architecture

```text
             Offline (Training)                         Online (Inference)
 ┌──────────────────────────────┐           ┌────────────────────────────────┐
 │ UCI Student Dataset          │           │ Client sends JSON to API      │
 └───────────────┬──────────────┘           └────────────────────────────────┘
                 │                                      │
                 ▼                                      ▼
      ┌──────────────────────┐              ┌────────────────────────────┐
      │ Data & Feature Build │              │ FastAPI app on AWS Lambda  │
      │ - Encoding           │              │ - Input validation          │
      │ - Scaling            │              │ - Preprocessing             │
      │ - Feature engineering│              └───────────────┬────────────┘
      └───────────────┬──────┘                              │
                      ▼                                      ▼
      ┌────────────────────────────────┐        ┌──────────────────────────┐
      │ Stacking Ensemble Training     │        │ Preloaded Stacking Model │
      │ - XGBoost / LGBM / CatBoost    │        └──────────────────────────┘
      │ - SMOTE in CV only            │                       │
      └────────────────────────────────┘                       ▼
                      │                             JSON prediction response
                      ▼
           Model artifact (pkl)
                      │
       Docker Image (Lambda Runtime)
                      │
                AWS ECR → Lambda
```

---

# 📦 Repository Structure

```text
Student_Success_AI/
├── README.md
├── requirements.txt
├── aws_deployment.md
├── Dockerfile
│
├── lambda/
│   ├── Dockerfile
│   └── handler.py
│
├── src/
│   ├── api/main.py
│   ├── data/
│   ├── features/
│   └── models/
│
└── tests/
    └── test_api.py
```

---

# 🧪 Dataset  
- UCI “Predict Students’ Dropout and Academic Success”  
- ~4,424 samples, 36 features  
- Target: `Dropout`, `Enrolled`, `Graduate`  
- Challenges: small dataset, imbalanced classes, mixed variable types  

---

# ⚙️ Training Pipeline

- One-hot encoding (nominal)  
- Ordinal encoding (where applicable)  
- Standard scaling (after train-test split)  
- Feature engineering:
  - semester aggregates  
  - performance ratios  
- SMOTE applied **inside** cross-validation  
- Stacking ensemble:
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - Logistic Regression (meta-learner)  

Primary metric: **Macro-F1**  
Typical result: **≈ 0.77 Macro-F1**

---

# 🌐 Running the API Locally

Start FastAPI:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "age": 21,
        "gender": "M",
        "tuition_fees_up_to_date": 1,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_approved": 4
      }'
```

Example response:

```json
{
  "prediction": "Dropout",
  "probabilities": {
    "Dropout": 0.78,
    "Enrolled": 0.15,
    "Graduate": 0.07
  }
}
```

---

# 🐳 Docker

Local:

```bash
docker build -t student-success-api .
docker run -p 8000:8000 student-success-api
```

Lambda image:

```bash
docker build -t student-success-lambda -f lambda/Dockerfile .
```

---

# ☁️ AWS Lambda Deployment

Summary of the deployment flow:

```bash
# 1. Build image
docker build -t student-success-lambda -f lambda/Dockerfile .

# 2. Push to ECR
docker push <account>.dkr.ecr.<region>.amazonaws.com/student-success-lambda

# 3. Point Lambda to the ECR image
```

CloudShell ran out of space, so I used a small EC2 instance to build the Docker image.  
This is exactly the kind of practical issue real ML engineers deal with.

---

# 🔍 Tests

```bash
pytest tests/
```

---

# ⚠️ Limitations

- Dataset is small and from one institution  
- Prototype system (not connected to a real university SIS)  
- Manual deployment (no CI/CD yet)  
- Lambda image can be optimized further  

---

# 💡 Future Improvements

- CI/CD with GitHub Actions  
- Terraform/SAM packaging for infrastructure  
- Better API validation  
- Simple UI dashboard for advisors  
- Model registry + versioning  
- Drift monitoring  

---

# 📄 License  
MIT License
