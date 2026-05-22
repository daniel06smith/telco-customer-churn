# Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to churn, built to support sales and retention teams in prioritising outreach.

**Live app → [daniel06smith-telco-customer-churn.streamlit.app](https://daniel06smith-telco-customer-churn.streamlit.app/)**

---

## What it does

You enter a customer's account and service details into a form. The model returns:

- **Churn probability** — how likely this customer is to leave (0–100%)
- **Risk classification** — Low / Medium / High based on the tuned decision threshold
- **Where this customer sits** — mini charts showing how their key metrics compare to typical churn patterns across the customer base
- **Actionable scenarios** — re-runs the model with one change at a time (e.g. "switch to annual contract") and shows only the changes that would meaningfully reduce their risk

The goal is to give a frontline sales agent something concrete to act on in 60 seconds, not just a number.

---

## Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM (gradient boosted trees) |
| Primary metric | Recall on churners (class 1) |
| Decision threshold | 0.331 (tuned by Optuna, not the default 0.5) |
| Features | 25 (after one-hot encoding 19 raw inputs) |
| Training data | IBM Telco Customer Churn dataset (~7,000 customers) |

### Why recall, not accuracy?

Missing a real churner (false negative) is more costly than flagging someone who stays (false alarm). A false negative means lost revenue with no chance of intervention. A false alarm means a retention call that wasn't strictly necessary — much cheaper. The threshold of 0.331 was tuned specifically to catch more churners, at the cost of some precision.

### Features used

After preprocessing and one-hot encoding:

- **Demographics** — gender, senior citizen status, partner, dependents  
- **Account** — tenure, contract type, monthly charges, total charges, paperless billing, payment method  
- **Services** — phone, multiple lines, internet service type, online security, online backup, device protection, tech support, streaming TV, streaming movies

---

## Using the live app

No installation needed. Open the link above and:

1. **Dashboard page** — start here to understand what the model has learned overall: which features matter most and how each one shifts churn probability across its full range
2. **Churn Predictor page** — fill in the customer's details and click *Predict Churn Risk*

The app is read-only and doesn't store any data you enter.

---

## Running locally

### 1. Clone the repo

```bash
git clone https://github.com/daniel06smith/telco-customer-churn.git
cd telco-customer-churn
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install -e .              # installs the local telco_churn package
```

### 3. Run the app

The trained model is already committed to the repo, so the app works immediately:

```bash
streamlit run app/Predictor.py
```

Then open [http://localhost:8501](http://localhost:8501).

### 4. Retrain the model (optional)

If you want to retrain from scratch you'll need the source data. Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV at:

```
data/raw/telco_customer_churn_data.csv
```

Then run:

```bash
python scripts/run_pipeline.py
```

This trains the model, saves `models/best_model.pkl` and `models/feature_columns.json`, and logs the run to MLflow. To view the MLflow experiment UI:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000).

To re-run Optuna hyperparameter search:

```bash
python scripts/tune.py
```

---

## Running with Docker

```bash
docker build -t telco-churn-app .
docker run -p 8501:8501 telco-churn-app
```

Open [http://localhost:8501](http://localhost:8501). The trained model is baked into the image at build time — no external model registry needed.

---

## Project structure

```
telco-customer-churn/
├── app/
│   ├── Predictor.py            # Main Streamlit page (churn prediction form)
│   ├── pages/
│   │   └── Dashboard.py        # Dashboard page (feature importance + PDP charts)
│   └── utils.py                # Shared model loading, PDP computation, inference
├── data/
│   └── raw/                    # Source CSV (not committed — see above)
├── models/
│   ├── best_model.pkl          # Trained LightGBM model
│   └── feature_columns.json    # Ordered feature list expected by the model
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_modelling.ipynb      # Model development and evaluation
├── scripts/
│   ├── run_pipeline.py         # End-to-end train + MLflow tracking
│   └── tune.py                 # Optuna hyperparameter search
├── src/telco_churn/            # Installable Python package
│   ├── config.py               # Paths, column definitions, best params
│   ├── data/load.py
│   ├── data/preprocess.py
│   ├── features/build.py
│   ├── models/train.py
│   └── models/evaluate.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Limitations

This project is a learning exercise built on a public dataset. Before treating predictions as ground truth, it's worth understanding what the model cannot see:

### Data it was never given
- **Customer service history** — complaint frequency, call centre interactions, and ticket resolution times are strong churn signals in real telcos, but aren't in this dataset
- **Usage patterns** — how much data a customer actually uses, whether they're near their plan limits, call duration trends
- **Competitive context** — whether a competitor has recently launched a cheaper plan in this customer's area
- **Promotional history** — whether this customer has already received retention offers and ignored them

### Structural limitations
- **Revenue-agnostic** — a customer paying $20/month and one paying $200/month receive the same churn probability. In practice you'd weight interventions by customer lifetime value
- **Static snapshot** — the model sees one row per customer at a single point in time. It can't detect trends (e.g. a customer who was happy last quarter but whose monthly charges just jumped)
- **Small dataset** — ~7,000 customers is adequate for a demo but small for a real model. Rare segment patterns (e.g. senior citizens on fibre with 2-year contracts) may be underrepresented
- **Synthetic/public data** — the IBM dataset was designed for educational use. A model trained on it may not transfer well to a real operator's customer base without retraining
- **No uncertainty estimate** — the app shows a single probability, not a confidence interval. A prediction of 42% from a model uncertain between 20% and 65% should be treated differently from one the model is confident about

### App-level limitations
- **Counterfactuals are rule-based** — the "what would reduce this customer's risk?" section tests a fixed shortlist of scenarios (contract type, tech support, security, payment method). It won't surface unusual combinations or features not in that list
- **PDP charts use a single baseline** — the curves show churn probability for a typical median customer as one feature varies. They are not true population-averaged partial dependence plots, so they may not reflect how a very atypical customer would respond to changes

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Modularise into `src/telco_churn/` package |
| 2 | ✅ Done | MLflow experiment tracking |
| 3 | ✅ Done | Optuna hyperparameter tuning |
| 4 | ✅ Done | Streamlit prediction app |
| 5 | ✅ Done | Docker containerisation |
| 6 | ✅ Done | Streamlit Community Cloud (public URL) |
| 7 | 🔜 Next | AWS deployment — ECR + App Runner |
| 8 | 🔜 Next | GitHub Actions CI/CD |
| 9 | 💡 Later | SHAP explanations per prediction |
| 10 | 💡 Later | Batch prediction for whole customer base |
| 11 | 💡 Later | Scheduled retraining + data drift detection |
| 12 | 💡 Later | Customer lifetime value weighting |

### Phase 7 — AWS (in progress)

Two-track deployment strategy:
- **Streamlit Community Cloud** — free, always-on, public URL for anyone to access
- **AWS ECR + App Runner** — for learning AWS services and resume credibility; kept paused when not actively demoing to minimise cost (~$0.08/month for ECR storage only when paused)

---

## Tech stack

| Layer | Tool |
|---|---|
| Model | LightGBM |
| Preprocessing | pandas, scikit-learn |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| App | Streamlit |
| Containerisation | Docker |
| Deployment | Streamlit Community Cloud / AWS App Runner |
| Language | Python 3.11+ |

---

## Acknowledgements

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) via Kaggle.
