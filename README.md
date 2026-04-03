# Credit Risk Scoring System
### End-to-End Default Prediction | LightGBM · FastAPI · Business Optimization

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![AUC](https://img.shields.io/badge/AUC-0.783-brightgreen) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal) ![Dataset](https://img.shields.io/badge/Dataset-Home%20Credit-orange) ![CI](https://github.com/kaushalkumarma2025/home-credit-default-risk/actions/workflows/test.yml/badge.svg)

---

## What This Is

A production-oriented credit scoring system built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The system ingests raw applicant data, engineers predictive features, scores applicants via LightGBM, and exposes a REST API that returns a default probability with a business-optimized approve/reject decision.

The threshold is not set at 0.5. It is calibrated against actual loan economics — a missed default costs 33× more than a wrongly rejected applicant — yielding **₹93.9 crore net business value** on the held-out test set.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Dataset size | 307,511 applicants, 122 raw features |
| Default rate | 8.07% (severe class imbalance) |
| Baseline (Logistic Regression) | AUC 0.766 |
| Final model (LightGBM) | AUC **0.783**, KS **42.7%** |
| Optimized threshold | Calibrated to loan cost structure |
| Net business value (test set) | **₹93.9 crore** |

---

## Pipeline

```
Raw Data (8 tables)
    │
    ▼
[1] Data Processing
    · Anomaly correction (DAYS_EMPLOYED outliers)
    · Missing value imputation (median/mode + missingness flags)
    · One-hot encoding for categoricals
    │
    ▼
[2] Feature Engineering  (~50+ features)
    · Credit-to-income, annuity-to-income ratios
    · Income per household member
    · EXT_SOURCE aggregations (mean, min, max)
    · Employment stability ratio
    · Bureau & previous application aggregates
    │
    ▼
[3] Modeling
    · Logistic Regression (baseline)
    · LightGBM (final) with class-weight balancing
    · Evaluation: AUC-ROC, KS Statistic, Gini
    │
    ▼
[4] Business Threshold Optimization
    · FN cost: ₹5,00,000 | FP cost: ₹15,000
    · Threshold selected at max expected business value
    │
    ▼
[5] FastAPI Scoring Endpoint
    · Input → feature engineering → model → decision + explanation
```

---

## Repository Structure

```
home-credit-default-risk/
│
├── notebooks/
│   ├── Home_credit.ipynb       # Full ML pipeline: EDA → features → modeling → business layer
│   └── images/                 # EDA and model evaluation plots (20 figures)
│
├── src/
│   └── feature_engineering.py  # Feature engineering extracted as reusable module
│
├── api/
│   └── app.py                  # FastAPI scoring API (/predict, /health endpoints)
│
├── models/
│   ├── lgbm_credit_risk.pkl    # Trained LightGBM classifier
│   ├── feature_columns.pkl     # Column order used at training time
│   └── pipeline_config.csv     # Imputation values and encoding config
│
├── tests/
│   └── test_features.py        # Unit tests for feature engineering logic
│
├── data/                       # Empty — see Dataset section below
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Raw dataset CSVs are not tracked. Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) and place in `data/raw/` before running the notebook.

---

## API

**`POST /predict`** — Score a single applicant

```json
// Request
{
  "AMT_INCOME_TOTAL": 202500,
  "AMT_CREDIT": 406597,
  "AMT_ANNUITY": 24700,
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -1200,
  "CNT_FAM_MEMBERS": 2,
  "EXT_SOURCE_1": 0.45,
  "EXT_SOURCE_2": 0.26,
  "EXT_SOURCE_3": 0.50
}

// Response
{
  "default_probability": 0.8146,
  "decision": "REJECT",
  "risk_level": "HIGH",
  "key_drivers": [
    "Low external credit score",
    "High credit-to-income ratio",
    "Short employment history"
  ]
}
```

The API reconstructs all engineered features internally — callers send raw applicant fields only.

**`GET /health`** — Service liveness check  
**`GET /docs`** — Auto-generated Swagger UI (FastAPI)

---

## Running Locally

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Start the API**
```bash
cd api
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test via Swagger UI.

**Run the tests**
```bash
pytest tests/ -v
```

**Run the full notebook**

Open `notebooks/Home_credit.ipynb` in Jupyter. Requires the Home Credit CSVs in `data/raw/`.

---

## Why This Matters

In credit risk, a high AUC does not automatically translate to good business outcomes. A model that scores well on the leaderboard can still approve too many defaulters or reject too many creditworthy applicants — depending on where you draw the line.

This project treats the threshold as a business decision, not a modeling afterthought. The cost structure (33:1 FN/FP asymmetry) is encoded directly into threshold selection, which is why the system recovers ₹93.9 crore in net value rather than just reporting a metric.

Feature engineering — not the model — drove most of the performance lift. The jump from logistic regression (AUC 0.766) to LightGBM (AUC 0.783) is modest. The real work was constructing signals the raw data does not contain: behavioral ratios, cross-table aggregates, and missingness indicators.

---

## Key Design Decisions

**Why LightGBM?** Gradient boosting handles the heterogeneous feature space (continuous, categorical, binary flags) and class imbalance better than linear models. The `scale_pos_weight` parameter was tuned to the 8.07% default rate.

**Why not use 0.5 as threshold?** At 0.5, the model optimizes accuracy — useless on imbalanced data. The threshold is selected by maximizing expected business value across the full cost/benefit curve.

**Why missingness flags?** Whether a value is missing is itself predictive in credit data. A missing `EXT_SOURCE` score signals no external credit history — a meaningful signal, not just noise.

---

## Business Impact

| Outcome | Cost |
|---|---|
| False Negative (default approved) | ₹5,00,000 |
| False Positive (good applicant rejected) | ₹15,000 |

The 33:1 cost asymmetry justifies a conservative threshold. The model's optimized operating point recovers significant value versus a naive 50th-percentile cutoff.

---

## What's Next

- [ ] SHAP explanations integrated into API response (replace heuristic `key_drivers`)
- [ ] Model monitoring with data drift detection (Evidently)
- [ ] Challenger model (XGBoost / CatBoost) comparison
- [ ] Score-band segmentation (Excellent / Good / Fair / Poor / Decline)

---

## Dataset

[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

Data is not included in this repository. Download from Kaggle and place CSVs in `data/raw/`.

---

## Author

Built to bridge the gap between standard ML practice and real-world lending decisions. The parts that required the most thought — and are most underrepresented in typical portfolio projects — were threshold optimization under asymmetric costs and ensuring the feature pipeline runs identically in training and at inference time.

If you have questions about the modeling choices or want to discuss the business framing, feel free to open an issue.
