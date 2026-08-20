# ============================================================
# TELCO CUSTOMER CHURN - APACHE SPARK + MLFLOW
# ============================================================
import os
import platform
from pathlib import Path

if platform.system() == "Windows":
    HADOOP_HOME = r"C:\hadoop"

    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["hadoop.home.dir"] = HADOOP_HOME
    os.environ["PATH"] += os.pathsep + os.path.join(
        HADOOP_HOME,
        "bin"
    )

    print("HADOOP_HOME:", os.environ.get("HADOOP_HOME"))
    print(
        "winutils exists:",
        os.path.exists(
            os.path.join(
                HADOOP_HOME,
                "bin",
                "winutils.exe"
            )
        )
    )

LINUX_PATH = "/storage/data/datasets/team23_dataset/"
FILE_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if platform.system() == "Windows":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / FILE_NAME

elif platform.system() == "Linux":
    DATA_PATH = Path(
        LINUX_PATH
    ) / FILE_NAME

else:
    raise RuntimeError(
        f"Unsupported operating system: {platform.system()}"
    )
OUTPUT_DIR = DATA_PATH.parent / "processed"

TRAIN_PATH = OUTPUT_DIR / "train"
TEST_PATH = OUTPUT_DIR / "test"
print(f"Data Path is {DATA_PATH}")
print(f"Train Data Path is {TRAIN_PATH}")
print(f"Test Data Path is {TEST_PATH}")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, trim, count, sum as spark_sum
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

import mlflow
import mlflow.spark


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("TelcoCustomerChurn_train")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")



train_df = spark.read.parquet(str(TRAIN_PATH))
test_df = spark.read.parquet(str(TEST_PATH))

# ============================================================
# 2. HANDLE CLASS IMBALANCE
# ============================================================

# Calculate class frequencies from TRAINING DATA ONLY

class_counts = (
    train_df.groupBy("label")
    .count()
    .collect()
)

counts = {
    row["label"]: row["count"]
    for row in class_counts
}

negative_count = counts.get(0.0)
positive_count = counts.get(1.0)

total_count = negative_count + positive_count


negative_weight = total_count / (
    2.0 * negative_count
)

positive_weight = total_count / (
    2.0 * positive_count
)


train_df = train_df.withColumn(
    "classWeight",
    when(
        col("label") == 1.0,
        positive_weight
    ).otherwise(
        negative_weight
    )
)

categorical_columns = [
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
    "PaymentMethod"
]

# ============================================================
# 3. DEFINE FEATURES
# ============================================================

categorical_features = categorical_columns + [
    "tenure_group"
]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count",
    "avg_monthly_spend"
]


# Convert categorical values to indexes
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_features
]


indexed_columns = [
    c + "_index"
    for c in categorical_features
]


# One-hot encode categorical variables
encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=[
        c + "_encoded"
        for c in categorical_features
    ]
)


encoded_columns = [
    c + "_encoded"
    for c in categorical_features
]


# Assemble all features
assembler = VectorAssembler(
    inputCols=numeric_features + encoded_columns,
    outputCol="unscaled_features",
    handleInvalid="keep"
)


# Standardization
scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withMean=False,
    withStd=True
)


# ============================================================
# 4. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)


lr_pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        scaler,
        lr
    ]
)


# ============================================================
# 5. MODEL 2 - RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    featuresCol="unscaled_features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=200,
    maxDepth=8,
    seed=42
)


rf_pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        rf
    ]
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, test_data):

    predictions = model.transform(test_data)

    # ROC-AUC
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(predictions)


    # PR-AUC
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    pr_auc = pr_evaluator.evaluate(predictions)


    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)


    # F1
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = f1_evaluator.evaluate(predictions)


    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    precision = precision_evaluator.evaluate(predictions)


    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    recall = recall_evaluator.evaluate(predictions)


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc
    }
# ============================================================
# MLFLOW TRACKING CONFIGURATION
# ============================================================

if platform.system() == "Windows":
    MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

print("MLflow tracking URI:",  mlflow.get_tracking_uri())

mlflow.set_experiment(
    "Telco Customer Churn - Spark ML"
)

# ============================================================
# 7. MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Telco Customer Churn - Spark ML"
)


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

with mlflow.start_run(
    run_name="LogisticRegression"
):

    model_lr = lr_pipeline.fit(train_df)

    metrics_lr = evaluate_model(
        model_lr,
        test_df
    )

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "maxIter",
        100
    )

    mlflow.log_param(
        "regParam",
        0.01
    )

    mlflow.log_metrics(metrics_lr)

    mlflow.spark.log_model(
        model_lr,
        "logistic-regression-model"
    )

    print(
        "Logistic Regression:",
        metrics_lr
    )


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

with mlflow.start_run(
    run_name="RandomForest"
):

    model_rf = rf_pipeline.fit(train_df)

    metrics_rf = evaluate_model(
        model_rf,
        test_df
    )

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_param(
        "numTrees",
        200
    )

    mlflow.log_param(
        "maxDepth",
        8
    )

    mlflow.log_metrics(metrics_rf)

    mlflow.spark.log_model(
        model_rf,
        "random-forest-model"
    )

    print(
        "Random Forest:",
        metrics_rf
    )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("\nLogistic Regression")
for metric, value in metrics_lr.items():
    print(f"{metric}: {value:.4f}")


print("\nRandom Forest")
for metric, value in metrics_rf.items():
    print(f"{metric}: {value:.4f}")


spark.stop()