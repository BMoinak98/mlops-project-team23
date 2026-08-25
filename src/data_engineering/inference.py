import os
from typing import Dict, Any

import mlflow
import mlflow.spark

from pyspark.sql import SparkSession


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:6091"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "Telco Customer Churn - Spark ML"
)

# Optional explicit fallback.
# Example:
# RF_MODEL_URI=runs:/<run-id>/random-forest-model
RF_MODEL_URI = os.getenv("RF_MODEL_URI")


# ------------------------------------------------------------
# Spark
# ------------------------------------------------------------

_spark = None
_model = None
_model_info = None


def get_spark() -> SparkSession:
    global _spark

    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("TelcoCustomerChurn_Inference")
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.memoryOverhead", "512m")
            .config("spark.network.timeout", "800s")
            .config("spark.executor.heartbeatInterval", "60s")
            .config("spark.python.worker.reuse", "true")
            .getOrCreate()
        )

        _spark.sparkContext.setLogLevel("WARN")

    return _spark


# ------------------------------------------------------------
# MLflow
# ------------------------------------------------------------

def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    print(
        f"MLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )


def find_best_model():
    """
    Find the model with the highest ROC-AUC in the configured
    MLflow experiment.

    Expected training runs have:
        param.model = LogisticRegression / RandomForest
        metric.roc_auc
    """

    configure_mlflow()

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.roc_auc DESC"],
        max_results=100,
    )

    if runs.empty:
        raise RuntimeError(
            "No MLflow runs found in the experiment."
        )

    # Only consider runs which actually contain the model
    # information expected from the training script.
    valid_runs = runs[
        runs["params.model"].isin(
            ["LogisticRegression", "RandomForest"]
        )
    ]

    if valid_runs.empty:
        raise RuntimeError(
            "No LogisticRegression or RandomForest "
            "runs found in MLflow."
        )

    # Drop runs without ROC-AUC.
    valid_runs = valid_runs.dropna(
        subset=["metrics.roc_auc"]
    )

    if valid_runs.empty:
        raise RuntimeError(
            "No candidate runs contain roc_auc."
        )

    best_run = valid_runs.iloc[0]

    model_type = best_run["params.model"]
    run_id = best_run["run_id"]
    roc_auc = best_run["metrics.roc_auc"]

    if model_type == "RandomForest":
        artifact_path = "random-forest-model"
    else:
        artifact_path = "logistic-regression-model"

    model_uri = f"runs:/{run_id}/{artifact_path}"

    return {
        "model_uri": model_uri,
        "model_type": model_type,
        "run_id": run_id,
        "roc_auc": float(roc_auc),
    }


def find_random_forest_model():
    """
    Fallback model selection.

    First use RF_MODEL_URI if explicitly supplied.
    Otherwise find the best RandomForest run in MLflow.
    """

    configure_mlflow()

    if RF_MODEL_URI:
        return {
            "model_uri": RF_MODEL_URI,
            "model_type": "RandomForest",
            "run_id": None,
            "roc_auc": None,
        }

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=100
    )

    if runs.empty:
        raise RuntimeError(
            "No MLflow runs found."
        )

    rf_runs = runs[
        runs["params.model"] == "RandomForest"
    ]

    if rf_runs.empty:
        raise RuntimeError(
            "No RandomForest model found in MLflow."
        )

    # Prefer the RandomForest with the highest ROC-AUC.
    rf_runs = rf_runs.dropna(
        subset=["metrics.roc_auc"]
    )

    if rf_runs.empty:
        raise RuntimeError(
            "RandomForest runs do not contain roc_auc."
        )

    best_rf = rf_runs.iloc[0]

    run_id = best_rf["run_id"]
    roc_auc = best_rf["metrics.roc_auc"]

    model_uri = (
        f"runs:/{run_id}/random-forest-model"
    )

    return {
        "model_uri": model_uri,
        "model_type": "RandomForest",
        "run_id": run_id,
        "roc_auc": float(roc_auc),
    }


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model():
    """
    Load the best MLflow model.

    Automatic selection:
        highest ROC-AUC

    Fallback:
        RandomForest
    """

    global _model
    global _model_info

    if _model is not None:
        return _model, _model_info

    try:
        print("Trying to automatically find best model...")

        model_info = find_best_model()

        print(
            f"Best model: {model_info['model_type']} "
            f"(ROC-AUC={model_info['roc_auc']:.4f})"
        )

    except Exception as exc:
        print(
            "Automatic model selection failed: "
            f"{exc}"
        )

        print("Falling back to RandomForest...")

        model_info = find_random_forest_model()

        print(
            f"Fallback model: RandomForest "
            f"(ROC-AUC={model_info['roc_auc']})"
        )

    spark = get_spark()

    print(
        f"Loading Spark model from: "
        f"{model_info['model_uri']}"
    )

    _model = mlflow.spark.load_model(
        model_info["model_uri"]
    )

    _model_info = model_info

    print("Model loaded successfully.")

    return _model, _model_info


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count",
    "avg_monthly_spend",
]

ALL_FEATURES = (
    CATEGORICAL_COLUMNS +
    NUMERIC_COLUMNS
)


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference for a single customer.
    """

    model, model_info = load_model()
    spark = get_spark()

    # Make sure we only pass expected fields.
    row = {
        feature: features.get(feature)
        for feature in ALL_FEATURES
    }

    # Convert pandas/python values into a Spark DataFrame.
    df = spark.createDataFrame([row])

    predictions = model.transform(df)

    result = predictions.select(
        "prediction",
        "probability",
        "rawPrediction"
    ).collect()[0]

    prediction = int(result["prediction"])

    probability = result["probability"]

    # Probability of class 1 = churn.
    churn_probability = float(
        probability[1]
    )

    return {
        "prediction": prediction,
        "churn": prediction == 1,
        "churn_probability": churn_probability,
        "model": model_info["model_type"],
        "model_run_id": model_info["run_id"],
        "model_roc_auc": model_info["roc_auc"],
    }


def get_model_info() -> Dict[str, Any]:
    """
    Return information about the currently loaded model.
    """

    _, model_info = load_model()

    return model_info
