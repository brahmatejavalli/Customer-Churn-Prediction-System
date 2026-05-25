# Customer Churn Prediction System

An end-to-end ML pipeline that predicts telecom customer churn — from raw data to a deployable REST API.

---

## Problem Statement

Customer churn is one of the most costly problems in subscription businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This system predicts which customers are at high risk of cancelling, enabling retention teams to intervene before it happens.

**Business Question:** *Which customers will churn, how likely is it, and how confident are we?*

---

## Project Structure

```
customer_churn_project/
│
├── data/
│   └── churn.csv                   # Telco customer dataset (7,043 records)
│
├── model/
│   ├── model.pkl                   # Saved best model
│   ├── scaler.pkl                  # StandardScaler for numeric features
│   ├── feature_names.pkl           # Column alignment for inference
│   ├── roc_curve.png               # ROC comparison — all 3 models
│   ├── feature_importance.png      # Top predictive features
│   └── eda_*.png                   # 5 EDA visualizations
│
├── src/
│   ├── preprocess.py               # Cleaning, feature engineering, encoding
│   ├── eda.py                      # Exploratory data analysis charts
│   ├── train.py                    # Train, evaluate, compare, and save models
│   ├── predict.py                  # CLI inference on new customers
│   └── api.py                      # FastAPI REST service
│
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Telco Customer Churn (IBM Sample / Kaggle) |
| Rows | 7,043 customers |
| Features | 20 (after dropping customerID) |
| Target | Churn (Yes = 21.8%, No = 78.2%) |
| Class imbalance | Mild — handled via ROC-AUC optimization |

---

## EDA — Key Findings

| Segment | Churn Rate | Insight |
|---|---|---|
| Month-to-month contract | ~42% | Highest risk segment |
| One-year contract | ~11% | 4× lower than monthly |
| Two-year contract | ~3% | Highly retained |
| Fiber optic internet | ~30% | Premium service, higher expectations |
| New customers (0–12 months) | ~35% | Critical early retention window |
| Long-term customers (3+ years) | ~8% | Loyalty strongly reduces churn |
| No tech support | ~25% | Support access is a retention lever |

**Top predictive features:** Contract type, tenure, monthly charges, internet service, tech support

---

## Model Comparison

Three classification models were trained with 5-fold cross-validation:

| Model | Accuracy | ROC-AUC | CV ROC-AUC | Std Dev |
|---|---|---|---|---|
| **Logistic Regression** | **80.1%** | **0.760** | **0.768** | ±0.009 |
| Gradient Boosting | 78.9% | 0.757 | 0.759 | ±0.010 |
| Random Forest | 77.8% | 0.716 | 0.717 | ±0.006 |

### Final Model: Logistic Regression

**Why it was selected:**

1. **Highest ROC-AUC (0.760)** — ROC-AUC was prioritized over accuracy because the dataset is imbalanced (21.8% churn). Accuracy alone is misleading; a model predicting "No churn" for everyone would hit ~78% accuracy.
2. **Best CV stability (std=0.009)** — Low variance indicates the model generalizes well and is not overfitting.
3. **Interpretability** — Coefficients make it easy to explain *why* a customer is flagged as high risk — critical for business stakeholders.
4. **Production readiness** — Faster inference and smaller model size vs. ensemble methods with near-identical performance.

> Gradient Boosting was very close in AUC (0.757) but showed higher variance and lower interpretability for marginal gain.

---

## Evaluation Metrics

```
              precision    recall  f1-score   support

   No Churn       0.82      0.96      0.88      1102
      Churn       0.61      0.23      0.34       307

    accuracy                           0.80      1409
   macro avg       0.71      0.60      0.61      1409
weighted avg       0.77      0.80      0.76      1409

ROC-AUC:    0.760
CV ROC-AUC: 0.768 +/- 0.009
```

**Note on recall:** Recall for the Churn class (0.23) reflects a trade-off with imbalanced data. Future improvement: apply `class_weight='balanced'` or tune the classification threshold.

---

## Feature Engineering

| Feature | Description | Rationale |
|---|---|---|
| `tenure_group` | Bucketed into New / Mid / Long-term | Captures non-linear loyalty effects |
| `charge_ratio` | MonthlyCharges / TotalCharges | Signals whether a customer is newly high-spend |

---

## REST API (FastAPI)

### Start the server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model/info` | Model metadata, performance, selection rationale |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for up to 500 customers |

### Example Request / Response

```bash
POST /predict
```
```json
{
  "customerID": "TC-00123",
  "tenure": 5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 70.35,
  "TotalCharges": "351.75"
}
```
```json
{
  "customerID": "TC-00123",
  "churn_prediction": "Yes",
  "churn_probability": 0.6721,
  "risk_level": "High",
  "confidence": "Moderate confidence (67%)"
}
```

### Risk Scoring

| Risk Level | Probability | Recommended Action |
|---|---|---|
| Low | < 30% | No intervention |
| Medium | 30–60% | Proactive outreach / offer |
| High | > 60% | Immediate retention campaign |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib

# Train the model
python src/train.py

# Run EDA charts
python src/eda.py

# CLI predictions
python src/predict.py

# Start API
uvicorn src.api:app --reload
```

---

## Tech Stack

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | FastAPI | Uvicorn | Pydantic | Joblib

---

## Resume Bullet Points

```
Customer Churn Prediction System
Python, Pandas, NumPy, Scikit-learn, FastAPI

- Built end-to-end ML pipeline: EDA -> preprocessing -> feature engineering
  -> model training -> evaluation -> REST API deployment
- Compared 3 classification models; selected Logistic Regression with
  ROC-AUC 0.760 and 80.1% accuracy based on performance and interpretability
- Identified key churn drivers via EDA: contract type (42% monthly vs 3%
  two-year churn rate) and tenure cohorts
- Deployed model as FastAPI REST service with single and batch prediction
  endpoints, input validation, and Low/Medium/High risk scoring per customer
```

---

*Built by Vallu Brahma Teja | MCA Graduate | Data Science*
