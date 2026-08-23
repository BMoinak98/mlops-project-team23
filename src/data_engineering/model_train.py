import os
import shutil
from pathlib import Path

os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
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

from common_util import load_config

# ============================================================
# 1. LOAD CONFIG & PATHS
# ============================================================
config = load_config()

TRAIN_PATH = config["train_data_path"]
TEST_PATH = config["test_data_path"]

print(f"Train Data Path: {TRAIN_PATH}")
print(f"Test Data Path: {TEST_PATH}")

# ============================================================
# 2. CREATE SPARK SESSION
# ============================================================
spark = (
    SparkSession.builder
    .appName("TelcoCustomerChurn_train")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.memoryOverhead", "1024m")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.network.timeout", "800s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.python.worker.reuse", "true")
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Directory for truncating iterative ML execution lineages
CHECKPOINT_DIR = "/tmp/spark_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

train_df = spark.read.parquet(str(TRAIN_PATH))
test_df = spark.read.parquet(str(TEST_PATH))

# ============================================================
# 3. HANDLE CLASS IMBALANCE
# ============================================================
class_counts = train_df.groupBy("label").count().collect()
counts = {row["label"]: row["count"] for row in class_counts}

negative_count = counts.get(0.0, 1)
positive_count = counts.get(1.0, 1)
total_count = negative_count + positive_count

negative_weight = total_count / (2.0 * negative_count)
positive_weight = total_count / (2.0 * positive_count)

train_df = train_df.withColumn(
    "classWeight",
    when(col("label") == 1.0, positive_weight).otherwise(negative_weight)
)

train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
train_df.count()
test_df.count()

# ============================================================
# 4. DEFINE FEATURES & PIPELINE STAGES
# ============================================================
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

categorical_features = categorical_columns + ["tenure_group"]
numeric_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges",
    "TotalCharges", "service_count", "avg_monthly_spend"
]

indexed_columns = [c + "_index" for c in categorical_features]
encoded_columns = [c + "_encoded" for c in categorical_features]

indexer = StringIndexer(
    inputCols=categorical_features,
    outputCols=indexed_columns,
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns
)

assembler = VectorAssembler(
    inputCols=numeric_features + encoded_columns,
    outputCol="unscaled_features",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

# ============================================================
# 5. MODEL PIPELINES
# ============================================================
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

lr_pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, lr])

rf = RandomForestClassifier(
    featuresCol="unscaled_features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=100,
    maxDepth=8,
    seed=42,
    checkpointInterval=10
)

rf_pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, test_data):
    predictions = model.transform(test_data).persist(StorageLevel.MEMORY_AND_DISK)
    predictions.count()

    auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    metrics = {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "roc_auc": auc_evaluator.evaluate(predictions),
        "pr_auc": pr_evaluator.evaluate(predictions)
    }

    predictions.unpersist()
    return metrics

# ============================================================
# 7. MLFLOW TRACKING CONFIGURATION
# ============================================================
mlflow_cfg = config.get("mlflow", {})
mlflow_uri = mlflow_cfg.get("tracking_uri", "http://mlflow:6091")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

exp_name = mlflow_cfg.get("experiment_name", "Telco Customer Churn - Spark ML v2")
mlflow.set_experiment(exp_name)

os.makedirs("/tmp/mlflow_tmp", exist_ok=True)

# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================
with mlflow.start_run(run_name="LogisticRegression"):
    model_lr = lr_pipeline.fit(train_df)
    metrics_lr = evaluate_model(model_lr, test_df)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param(
        "artifact_path",
        "logistic-regression-model"
    )
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("regParam", 0.01)
    mlflow.log_metrics(metrics_lr)

    mlflow.spark.log_model(
        spark_model=model_lr,
        artifact_path="logistic-regression-model",
        dfs_tmpdir="/tmp/mlflow_tmp"
    )

    print("Logistic Regression:", metrics_lr)

# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================
with mlflow.start_run(run_name="RandomForest"):
    model_rf = rf_pipeline.fit(train_df)
    metrics_rf = evaluate_model(model_rf, test_df)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param(
        "artifact_path",
        "random-forest-model"
    )
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 8)
    mlflow.log_metrics(metrics_rf)

    mlflow.spark.log_model(
        spark_model=model_rf,
        artifact_path="random-forest-model",
        dfs_tmpdir="/tmp/mlflow_tmp"
    )

    print("Random Forest:", metrics_rf)

# ============================================================
# 10. MODEL COMPARISON & CLEANUP
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

train_df.unpersist()
test_df.unpersist()
spark.stop()
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)