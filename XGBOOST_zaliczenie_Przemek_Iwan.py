"""
PROJECT: PREDICTION PRICE PER SQM (DENMARK)
"""

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# =========================
# 1. LOAD DATA
# =========================
def load_data(path):
    df = pd.read_parquet(path)
    df = df[df["date"] >= "2019-10-25"]
    return df


# =========================
# 2. FEATURE ENGINEERING
# =========================
def feature_engineering(df):

    df = df.drop(columns=[
        "quarter","address","city","sales_type",
        "dk_ann_infl_rate%", "yield_on_mortgage_credit_bonds%",
        "no_rooms", "region", "sqm_price", "house_id"
    ])

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.drop(columns=["date"])

    df["age"] = df["year"] - df["year_build"]
    df = df.drop(columns=["year_build", "month", "%_change_between_offer_and_purchase"])

    # ZIP -> area
    df["area"] = df["zip_code"].astype(str).str[:2]

    df = pd.get_dummies(df, columns=["area", "house_type"], drop_first=True)

    # TARGET
    df = df[df["sqm"] > 0]

    df["price_per_sqm"] = df["purchase_price"] / df["sqm"]
    df = df[df["price_per_sqm"] > 0]

    df["log_price_per_sqm"] = np.log(df["price_per_sqm"])

    df = df.drop(columns=["purchase_price"])

    # OUTLIERS
    df = df[df["log_price_per_sqm"] < df["log_price_per_sqm"].quantile(0.98)]

    # CLEAN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()


    df = df.astype(float)

    return df


# =========================
# 3. SPLIT
# =========================
def split_data(df):

    X = df.drop(columns=["log_price_per_sqm", "price_per_sqm", "zip_code"])
    y = df["log_price_per_sqm"]

    X_train = X[df["year"] < 2024]
    X_test = X[df["year"] == 2024]

    y_train = y[df["year"] < 2024]
    y_test = y[df["year"] == 2024]

    return X_train, X_test, y_train, y_test


# =========================
# 4. MODEL
# =========================
def baseline_model(X_train, y_train):

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# =========================
# 5. OPTUNA
# =========================
def objective(trial, X_train, y_train, X_val, y_val):

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=trial.suggest_int("n_estimators", 200, 500),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    y_pred_real = np.exp(y_pred)
    y_val_real = np.exp(y_val)

    return mean_absolute_error(y_val_real, y_pred_real)


def optimize_hyperparameters(X_train, y_train):

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_val, y_val), n_trials=20)

    print("\nBest params:", study.best_params)
    return study.best_params


# =========================
# 6. FINAL MODEL
# =========================
def train_model(X_train, y_train, best_params):

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )

    model.fit(X_train, y_train)
    return model


# =========================
# 7. EVALUATION
# =========================
def evaluate(model, X_train, y_train, X_test, y_test):

    y_pred = model.predict(X_test)

    y_pred_real = np.exp(y_pred)
    y_test_real = np.exp(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print("\n=== RESULTS ===")
    print("Średnia cena sqm:", round(y_test_real.mean(), 2))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAE %:", round(mae / y_test_real.mean(), 3))

    return y_test_real, y_pred_real


# =========================
# 8. SHAP
# =========================



def shap_analysis(model, X_test, feature_name="sqm"):

    print("\n=== SHAP ANALYSIS ===")

    # ---------------------------------
    # SHAP EXPLAINER
    # ---------------------------------
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    # ---------------------------------
    # 1. PLOT nr 1-SUMMARY PLOT
    # ---------------------------------
    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(
        "SHAP Summary Plot - Feature Impact on Model Prediction",
        fontsize=16,
        pad=20
    )

    plt.xlabel(
        "SHAP value (impact on prediction)",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------
    # FEATURE IMPORTANCE BAR PLOT
    # ---------------------------------
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title(
        "Feature Importance Based on Mean Absolute SHAP Values",
        fontsize=16,
        pad=20
    )

    plt.xlabel(
        "Mean Absolute SHAP Value",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------
    # DEPENDENCE PLOT
    # ---------------------------------
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_test,
        interaction_index=None,
        show=False
    )

    plt.title(
        f"Dependence Plot: Impact of {feature_name} on Prediction of prices",
        fontsize=16,
        pad=20
    )


    plt.xlabel(
        f"{feature_name} value",
        fontsize=12
    )

    plt.ylabel(
        f"SHAP value for {feature_name}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

# =========================
# MAIN
# =========================
def main():

    df = load_data("/Users/macbook/Desktop/DKHousingPrices.parquet")
    df = feature_engineering(df)

    print(df.dtypes)  # debug OK

    X_train, X_test, y_train, y_test = split_data(df)

    model = baseline_model(X_train, y_train)

    evaluate(model, X_train, y_train, X_test, y_test)

    shap_analysis(model, X_test)


if __name__ == "__main__":
    main()



