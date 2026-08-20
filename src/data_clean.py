# ============================================================
# TELCO CUSTOMER CHURN - APACHE SPARK + MLFLOW
# ============================================================
import os
import platform

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

from pathlib import Path
import platform
import yaml
# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("TelcoCustomerChurn_data_clean")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. LOAD DATA
# ============================================================
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

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH))
)

print("Initial records:", df.count())
df.printSchema()
df.show(5, truncate=False)


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove unnecessary whitespace
for column in df.columns:
    df = df.withColumn(column, trim(col(column)))


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
for c in categorical_columns:
    df = df.fillna(
        {c: "Unknown"}
    )


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


# ============================================================
# 9. CHECK CLASS DISTRIBUTION
# ============================================================

print("Class distribution:")

(
    df.groupBy("label")
      .count()
      .withColumn(
          "percentage",
          col("count") / df.count() * 100
      )
      .show()
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_df.write.mode("overwrite").parquet(str(TRAIN_PATH))
test_df.write.mode("overwrite").parquet(str(TEST_PATH))

print("Training records:", train_df.count())
print("Testing records:", test_df.count())


spark.stop()