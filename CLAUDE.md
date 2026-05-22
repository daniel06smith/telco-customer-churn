# Telco Customer Churn — Project Guide

## What this project is
Binary classification model that predicts whether a telecom customer will churn.
Packaged as a Streamlit web app, containerized with Docker, and eventually deployed on AWS.

**Primary metric: recall on class 1 (churn).** Missing a real churner (false negative) is
more costly than a false alarm. The model uses threshold=0.331 tuned by Optuna.

---

## Project structure

```
telco-customer-churn/
├── app/app.py                  # Streamlit prediction UI
├── data/
│   └── raw/                    # Source CSV (not committed to git)
├── models/
│   ├── best_model.pkl          # Trained LightGBM (not committed to git)
│   └── feature_columns.json    # Column list expected by the model
├── notebooks/                  # EDA + modelling exploration
├── scripts/
│   ├── run_pipeline.py         # End-to-end train + MLflow tracking
│   └── tune.py                 # Optuna hyperparameter search
├── src/telco_churn/            # Python package
│   ├── config.py               # All paths, column names, best params
│   ├── data/load.py
│   ├── data/preprocess.py
│   ├── features/build.py
│   ├── models/train.py
│   └── models/evaluate.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── requirements.txt
```

---

## Development workflow

### Set up environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .                # installs src/telco_churn as a package
pip install -r requirements.txt
```

### Train the model
```bash
python scripts/run_pipeline.py
```
Outputs: `models/best_model.pkl`, `models/feature_columns.json`
MLflow UI: `mlflow ui` then open http://localhost:5000

### Run Optuna tuning
```bash
python scripts/tune.py
```

### Run Streamlit app locally
```bash
streamlit run app/app.py
```

---

## Docker

### Build
```bash
docker build -t telco-churn-app .
```

### Run
```bash
docker run -p 8501:8501 telco-churn-app
```
Then open http://localhost:8501

### Notes
- The trained model is baked into the image at build time (COPY models/ models/)
- `jupyterlab` is excluded from requirements.txt — the container only runs the app
- `.dockerignore` excludes: `.venv`, `mlruns`, `notebooks`, `data`

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | Done | Modularize into `src/telco_churn/` package |
| 2 | Done | MLflow experiment tracking |
| 3 | Done | Optuna hyperparameter tuning |
| 4 | Done | Streamlit prediction app |
| 5 | Done | Docker containerization |
| 6 | Next | AWS deployment (ECR + App Runner) |
| 7 | Later | GitHub Actions CI/CD |

---

## AWS deployment (Phase 6 — next steps)

1. Push image to Amazon ECR (Elastic Container Registry)
2. Deploy to AWS App Runner (simplest path: connects to ECR, auto-scales, HTTPS included)

Commands (when ready):
```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repo (once)
aws ecr create-repository --repository-name telco-churn-app

# Tag and push
docker tag telco-churn-app:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/telco-churn-app:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/telco-churn-app:latest
```

---

## Key decisions and why

- **LightGBM over XGBoost/RF**: best recall in experiments
- **threshold=0.331**: tuned by Optuna to maximize recall, not default 0.5
- **`src/` layout**: keeps package importable in editable mode (`pip install -e .`)
- **Model baked into Docker image**: simplest approach; no external model registry needed at this scale
- **`python:3.11-slim` base image**: smaller than full Python image, has everything we need
