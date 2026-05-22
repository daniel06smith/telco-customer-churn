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
| 6 | Next | Streamlit Community Cloud (public demo link) |
| 7 | Next | AWS deployment (ECR + App Runner) — for learning + resume |
| 8 | Later | GitHub Actions CI/CD |

---

## Deployment strategy (decided 2026-05-22)

Two-track approach:
- **Streamlit Community Cloud** — free, always-on, public URL for recruiters to click
- **AWS ECR + App Runner** — for learning AWS and resume credibility; kept paused when not demoing

Why two tracks: App Runner paused = no public access without AWS credentials. Streamlit Community Cloud solves the "recruiter can click it anytime" problem for free.

### Phase 6: Streamlit Community Cloud
1. Push project to GitHub (public repo)
2. Go to share.streamlit.io, connect GitHub, select `app/app.py`
3. Set any secrets/env vars if needed
4. Get a public URL like `https://yourapp.streamlit.app`

Note: Streamlit Community Cloud runs from the GitHub repo directly — it does NOT use Docker. The `requirements.txt` is used to install dependencies.

### Phase 7: AWS deployment

**Explain each AWS tool when we get to it** — user is new to AWS and wants to understand
the purpose and use case of each service, not just run commands.

Tools involved and what they do (expand when teaching):
- **AWS CLI** — command-line tool to control all AWS services from your terminal
- **IAM** — Identity and Access Management; controls who/what can do what in AWS
- **Amazon ECR** — private Docker registry (like Docker Hub but inside your AWS account)
- **AWS App Runner** — runs your container as a web service; handles HTTPS, scaling, routing

Cost-minimization approach:
- Keep App Runner paused when not actively demoing
- Use smallest instance (0.25 vCPU / 0.5 GB RAM)
- ECR storage only: ~$0.08/month when paused

---

## Key decisions and why

- **LightGBM over XGBoost/RF**: best recall in experiments
- **threshold=0.331**: tuned by Optuna to maximize recall, not default 0.5
- **`src/` layout**: keeps package importable in editable mode (`pip install -e .`)
- **Model baked into Docker image**: simplest approach; no external model registry needed at this scale
- **`python:3.11-slim` base image**: smaller than full Python image, has everything we need
