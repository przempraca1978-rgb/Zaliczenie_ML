# 🏠 Denmark Housing — Price per SQM Prediction

A machine learning pipeline for predicting residential property price per square meter in Denmark, built with XGBoost and optimized using Optuna.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Model](#model)
- [Results](#results)
- [Changelog](#changelog)

---

## Overview

The goal of this project is to predict the **price per square meter** of residential properties in Denmark based on historical transaction data. The model uses a log-transformed target variable (`log_price_per_sqm`) to handle skewness, and is evaluated on real-price scale (DKK/m²).

Key design decisions:
- **Time-based train/test split** — training on pre-2024 data, testing on 2024
- **No data leakage** — outlier removal and validation splits are computed solely on training data
- **Cyclical encoding** of month to preserve seasonality

---

## Project Structure

```
dk-housing-sqm/
│
├── dk_housing_model_v2.py   # Main pipeline script
├── README.md                # This file
└── DKHousingPrices.parquet  # Source data (not included in repo)
```

---

## Dataset

- **Source:** [Kaggle — Denmark Housing Prices](https://www.kaggle.com/)
- **Format:** `.parquet`
- **Coverage:** Transactions from 2019-10-25 onwards
- **Target variable:** `price_per_sqm` (DKK per m²), log-transformed during training

Key columns used:

| Column | Description |
|---|---|
| `date` | Transaction date |
| `purchase_price` | Final sale price (DKK) |
| `sqm` | Property size in square meters |
| `year_build` | Year the property was built |
| `zip_code` | Postal code |
| `house_type` | Type of property |

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- matplotlib
- pyarrow (for `.parquet` support)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dk-housing-sqm.git
cd dk-housing-sqm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna matplotlib pyarrow
```

---

## Usage

1. Place your `DKHousingPrices.parquet` file on your machine.
2. Update the file path inside `main()`:

```python
df = load_data("/your/path/to/DKHousingPrices.parquet")
```

3. Run the script:

```bash
python dk_housing_model_v2.py
```

The script will:
- engineer features and split data
- run Optuna hyperparameter search (20 trials)
- train the final model with early stopping
- print evaluation metrics and top feature importances

---

## Pipeline

```
load_data()
    └── feature_engineering()
            └── split_data()                  ← outlier removal on train only
                    ├── optimize_hyperparameters()   ← time-based Optuna search
                    │       └── objective()          ← XGBoost + early stopping
                    ├── train_model()                ← final model + early stopping
                    └── evaluate()
                            ├── feature_importance()
                            └── analyze_errors()
```

---

## Model

**Algorithm:** XGBoost Regressor (`reg:squarederror`)

**Hyperparameters tuned via Optuna:**

| Parameter | Search Range |
|---|---|
| `n_estimators` | 200 – 500 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.01 – 0.1 |
| `subsample` | 0.7 – 1.0 |
| `colsample_bytree` | 0.7 – 1.0 |

**Early stopping:** 50 rounds (prevents overfitting on `n_estimators`)

**Validation strategy in Optuna:** time-based — train on `year < 2022`, validate on `year >= 2022`

---

## Results

Metrics are reported on the **2024 holdout set**, on real price scale (DKK/m²):

| Metric | Value |
|---|---|
| MAE | — |
| RMSE | — |
| MAE % | — |

> Fill in after running the model on your data.

---

## Changelog

### v2 (current)
- **FIX #1:** Outlier removal (quantile 0.98) moved after train/test split — computed only on training labels to eliminate data leakage
- **FIX #2:** Replaced random validation split in Optuna with a time-based split (`year < 2022` / `year >= 2022`)
- **FIX #3:** Added `early_stopping_rounds=50` to XGBoost in both Optuna objective and final model training

### v1
- Initial pipeline with full feature engineering, log-transform target, and Optuna optimization
