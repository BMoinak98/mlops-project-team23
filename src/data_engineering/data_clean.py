# ============================================================
# TELCO CUSTOMER CHURN - APACHE SPARK DATA CLEANING
# ============================================================
import os
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, trim, count, sum as spark_sum
)
import mlflow
import mlflow.spark

from common_util import load_config

# ============================================================
# 1. LOAD CONFIG & PATHS
# ============================================================
config = load_config()

DATA_PATH = config["raw_data_path"]
TRAIN_PATH = config["train_data_path"]
TEST_PATH = config["test_data_path"]

print(f"Raw Data Path: {DATA_PATH}")
print(f"Train Data Output: {TRAIN_PATH}")
print(f"Test Data Output: {TEST_PATH}")

# ============================================================
# 2. CREATE SPARK SESSION
# ============================================================
spark = (
    SparkSession.builder
    .appName("TelcoCustomerChurn_data_clean")
    .master("local[2]")                      # Limit to 2 CPU threads (leaves remaining cores for Airflow)
    .config("spark.driver.memory", "2g")      # Capped to 2GB to prevent JVM freeze
    .config("spark.executor.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4") # Low partition count for small tabular data
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ============================================================
# 3. LOAD DATA
# ============================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH))
).cache()

print("Initial records:", df.count())
df.printSchema()
df.show(5, truncate=False)


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove unnecessary whitespace
df = df.select([trim(col(c)).alias(c) for c in df.columns])


# TotalCharges is known to contain blank values in this dataset.
# Convert it explicitly to numeric.
df = df.withColumn(
    "TotalCharges",
    when(
        col("TotalCharges") == "",
        None
    ).otherwise(col("TotalCharges").cast("double"))
)


# Cast numeric columns
df = (
    df.withColumn("tenure", col("tenure").cast("double"))
      .withColumn("MonthlyCharges", col("MonthlyCharges").cast("double"))
      .withColumn("SeniorCitizen", col("SeniorCitizen").cast("double"))
)


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

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


# Numeric missing values -> median
for c in numeric_columns:

    median_value = df.approxQuantile(
        c,
        [0.5],
        0.01
    )[0]

    df = df.fillna(
        {c: median_value}
    )


# Categorical missing values -> "Unknown"
cat_defaults = {c: "Unknown" for c in categorical_columns}
df = df.fillna(cat_defaults)


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

df = df.dropDuplicates(["customerID"])


# ============================================================
# 6. TARGET ENCODING
# ============================================================

df = df.withColumn(
    "label",
    when(col("Churn") == "Yes", 1.0)
    .otherwise(0.0)
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

# Customer tenure groups
df = df.withColumn(
    "tenure_group",
    when(col("tenure") <= 12, "0-12")
    .when(col("tenure") <= 24, "13-24")
    .when(col("tenure") <= 48, "25-48")
    .otherwise("49+")
)


# Number of services subscribed to
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

service_count_expression = None

for c in service_columns:

    current = when(
        (col(c) == "Yes") |
        (col(c) == "Fiber optic") |
        (col(c) == "DSL"),
        1
    ).otherwise(0)

    if service_count_expression is None:
        service_count_expression = current
    else:
        service_count_expression = (
            service_count_expression + current
        )

df = df.withColumn(
    "service_count",
    service_count_expression
)


# Average monthly spending over customer lifetime
df = df.withColumn(
    "avg_monthly_spend",
    when(
        col("tenure") > 0,
        col("TotalCharges") / col("tenure")
    ).otherwise(col("MonthlyCharges"))
)


# ============================================================
# 8. REMOVE CUSTOMER ID
# ============================================================

df = df.drop("customerID", "Churn")

df = df.localCheckpoint()

# ============================================================
# 9. CHECK CLASS DISTRIBUTION
# ============================================================

total_records = df.count()
print(f"Total processed records: {total_records}")
print("Class distribution:")

(
    df.groupBy("label")
      .count()
      .withColumn("percentage", (col("count") / total_records) * 100)
      .show()
)

# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================
for path_str in [str(TRAIN_PATH), str(TEST_PATH)]:
    if os.path.exists(path_str):
        shutil.rmtree(path_str, ignore_errors=True)

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_df.write.mode("overwrite").parquet(str(TRAIN_PATH))
test_df.write.mode("overwrite").parquet(str(TEST_PATH))

print("Training records:", train_df.count())
print("Testing records:", test_df.count())


spark.stop()