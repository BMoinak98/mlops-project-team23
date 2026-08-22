# ============================================================
# TELCO CUSTOMER CHURN
# APACHE SPARK + RAY TUNE + MLFLOW
#
# Workflow:
# 1. Load processed train/test data
# 2. Split train data into tuning-train and validation
# 3. Tune Logistic Regression with Ray Tune
# 4. Tune Random Forest with Ray Tune
# 5. Log tuning results and parameters to MLflow
# 6. Retrain best models on complete train_df
# 7. Evaluate final models on untouched test_df
# 8. Select best final model
# 9. Log final model artifacts to MLflow
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import platform
from pathlib import Path

import mlflow
import mlflow.spark

import ray
from ray import tune

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


from common_util import load_config

# ============================================================
# 1. LOAD CONFIG & PATHS
# ============================================================

config = load_config()

# Hadoop setup (if specified in config)
hadoop_home = config.get("hadoop_home")
if hadoop_home:
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home
    os.environ["PATH"] += os.pathsep + os.path.join(hadoop_home, "bin")
    print("HADOOP_HOME configured:", hadoop_home)

TRAIN_PATH = config["train_data_path"]
TEST_PATH = config["test_data_path"]

print(f"Train Data Path: {TRAIN_PATH}")
print(f"Test Data Path: {TEST_PATH}")


# ============================================================
# 3. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName(
        "TelcoCustomerChurn_RayTune_MLflow"
    )
    .master("local[*]")
    .config(
        "spark.sql.shuffle.partitions",
        "8"
    )
    .getOrCreate()
)


spark.sparkContext.setLogLevel("WARN")


print("\n================================")
print("SPARK CONFIGURATION")
print("================================")

print("Spark Version:", spark.version)


# ============================================================
# 4. LOAD PROCESSED DATA
# ============================================================

train_df = (
    spark.read
    .parquet(str(TRAIN_PATH))
)

test_df = (
    spark.read
    .parquet(str(TEST_PATH))
)


print("\n================================")
print("DATASET SIZE")
print("================================")

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
#
# The original test set remains untouched.
# Validation data is created ONLY from train_df.
# ============================================================

tuning_train_df, validation_df = (
    train_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )
)


# Cache because the datasets will be used repeatedly
tuning_train_df = tuning_train_df.cache()

validation_df = validation_df.cache()

train_df = train_df.cache()

test_df = test_df.cache()


print("\n================================")
print("TRAIN / VALIDATION SPLIT")
print("================================")

print(
    "Tuning train rows:",
    tuning_train_df.count()
)

print(
    "Validation rows:",
    validation_df.count()
)


# ============================================================
# 6. DEFINE COLUMNS
# ============================================================

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


categorical_features = (
    categorical_columns
    + [
        "tenure_group"
    ]
)


numeric_features = [

    "SeniorCitizen",

    "tenure",

    "MonthlyCharges",

    "TotalCharges",

    "service_count",

    "avg_monthly_spend"
]


# ============================================================
# 7. FUNCTION:
# ADD CLASS WEIGHTS
#
# IMPORTANT:
# Class weights are calculated from the training dataset only.
# ============================================================

def add_class_weights(dataframe):

    class_counts = (
        dataframe
        .groupBy("label")
        .count()
        .collect()
    )


    counts = {

        row["label"]: row["count"]

        for row in class_counts
    }


    negative_count = counts.get(0.0)

    positive_count = counts.get(1.0)


    if (
        negative_count is None
        or positive_count is None
    ):

        raise ValueError(
            "Both classes must exist "
            "in the training dataset."
        )


    total_count = (
        negative_count
        + positive_count
    )


    negative_weight = (
        total_count
        / (
            2.0
            * negative_count
        )
    )


    positive_weight = (
        total_count
        / (
            2.0
            * positive_count
        )
    )


    weighted_df = (
        dataframe
        .withColumn(
            "classWeight",

            when(
                col("label") == 1.0,
                positive_weight
            )
            .otherwise(
                negative_weight
            )
        )
    )


    return weighted_df


# ============================================================
# 8. ADD CLASS WEIGHTS
#
# Separate weights are calculated for:
# - tuning training data
# - final complete training data
# ============================================================

tuning_train_weighted = (
    add_class_weights(
        tuning_train_df
    )
    .cache()
)


full_train_weighted = (
    add_class_weights(
        train_df
    )
    .cache()
)


print("\n================================")
print("CLASS DISTRIBUTION")
print("================================")


train_df.groupBy(
    "label"
).count().show()


# ============================================================
# 9. PIPELINE CREATION FUNCTIONS
#
# New pipeline objects are created for every trial.
# ============================================================

def create_lr_pipeline(
    max_iter,
    reg_param,
    elastic_net_param
):

    # ----------------------------------------
    # STRING INDEXERS
    # ----------------------------------------

    indexers = [

        StringIndexer(
            inputCol=column,
            outputCol=(
                column
                + "_index"
            ),
            handleInvalid="keep"
        )

        for column
        in categorical_features
    ]


    indexed_columns = [

        column
        + "_index"

        for column
        in categorical_features
    ]


    # ----------------------------------------
    # ONE HOT ENCODER
    # ----------------------------------------

    encoder = OneHotEncoder(

        inputCols=indexed_columns,

        outputCols=[

            column
            + "_encoded"

            for column
            in categorical_features
        ]
    )


    encoded_columns = [

        column
        + "_encoded"

        for column
        in categorical_features
    ]


    # ----------------------------------------
    # VECTOR ASSEMBLER
    # ----------------------------------------

    assembler = VectorAssembler(

        inputCols=(
            numeric_features
            + encoded_columns
        ),

        outputCol=(
            "unscaled_features"
        ),

        handleInvalid="keep"
    )


    # ----------------------------------------
    # STANDARD SCALER
    # ----------------------------------------

    scaler = StandardScaler(

        inputCol=(
            "unscaled_features"
        ),

        outputCol=(
            "features"
        ),

        withMean=False,

        withStd=True
    )


    # ----------------------------------------
    # LOGISTIC REGRESSION
    # ----------------------------------------

    lr = LogisticRegression(

        featuresCol="features",

        labelCol="label",

        weightCol="classWeight",

        maxIter=int(max_iter),

        regParam=float(reg_param),

        elasticNetParam=float(
            elastic_net_param
        )
    )


    # ----------------------------------------
    # COMPLETE PIPELINE
    # ----------------------------------------

    pipeline = Pipeline(

        stages=(

            indexers

            + [

                encoder,

                assembler,

                scaler,

                lr
            ]
        )
    )


    return pipeline


# ============================================================
# 10. RANDOM FOREST PIPELINE
# ============================================================

def create_rf_pipeline(

    num_trees,

    max_depth,

    max_bins,

    min_instances_per_node

):

    # ----------------------------------------
    # STRING INDEXERS
    # ----------------------------------------

    indexers = [

        StringIndexer(
            inputCol=column,
            outputCol=(
                column
                + "_index"
            ),
            handleInvalid="keep"
        )

        for column
        in categorical_features
    ]


    indexed_columns = [

        column
        + "_index"

        for column
        in categorical_features
    ]


    # ----------------------------------------
    # ONE HOT ENCODER
    # ----------------------------------------

    encoder = OneHotEncoder(

        inputCols=indexed_columns,

        outputCols=[

            column
            + "_encoded"

            for column
            in categorical_features
        ]
    )


    encoded_columns = [

        column
        + "_encoded"

        for column
        in categorical_features
    ]


    # ----------------------------------------
    # VECTOR ASSEMBLER
    # ----------------------------------------

    assembler = VectorAssembler(

        inputCols=(
            numeric_features
            + encoded_columns
        ),

        outputCol=(
            "unscaled_features"
        ),

        handleInvalid="keep"
    )


    # ----------------------------------------
    # RANDOM FOREST
    #
    # Random Forest does NOT require scaling.
    # ----------------------------------------

    rf = RandomForestClassifier(

        featuresCol=(
            "unscaled_features"
        ),

        labelCol="label",

        weightCol="classWeight",

        numTrees=int(num_trees),

        maxDepth=int(max_depth),

        maxBins=int(max_bins),

        minInstancesPerNode=int(
            min_instances_per_node
        ),

        seed=42
    )


    # ----------------------------------------
    # COMPLETE PIPELINE
    # ----------------------------------------

    pipeline = Pipeline(

        stages=(

            indexers

            + [

                encoder,

                assembler,

                rf
            ]
        )
    )


    return pipeline


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    evaluation_data
):

    predictions = (
        model.transform(
            evaluation_data
        )
    )


    # ----------------------------------------
    # ROC-AUC
    # ----------------------------------------

    roc_evaluator = (
        BinaryClassificationEvaluator(

            labelCol="label",

            rawPredictionCol=(
                "rawPrediction"
            ),

            metricName=(
                "areaUnderROC"
            )
        )
    )


    roc_auc = (
        roc_evaluator.evaluate(
            predictions
        )
    )


    # ----------------------------------------
    # PR-AUC
    # ----------------------------------------

    pr_evaluator = (
        BinaryClassificationEvaluator(

            labelCol="label",

            rawPredictionCol=(
                "rawPrediction"
            ),

            metricName=(
                "areaUnderPR"
            )
        )
    )


    pr_auc = (
        pr_evaluator.evaluate(
            predictions
        )
    )


    # ----------------------------------------
    # ACCURACY
    # ----------------------------------------

    accuracy_evaluator = (
        MulticlassClassificationEvaluator(

            labelCol="label",

            predictionCol="prediction",

            metricName="accuracy"
        )
    )


    accuracy = (
        accuracy_evaluator.evaluate(
            predictions
        )
    )


    # ----------------------------------------
    # F1
    # ----------------------------------------

    f1_evaluator = (
        MulticlassClassificationEvaluator(

            labelCol="label",

            predictionCol="prediction",

            metricName="f1"
        )
    )


    f1 = (
        f1_evaluator.evaluate(
            predictions
        )
    )


    # ----------------------------------------
    # PRECISION
    # ----------------------------------------

    precision_evaluator = (
        MulticlassClassificationEvaluator(

            labelCol="label",

            predictionCol="prediction",

            metricName=(
                "weightedPrecision"
            )
        )
    )


    precision = (
        precision_evaluator.evaluate(
            predictions
        )
    )


    # ----------------------------------------
    # RECALL
    # ----------------------------------------

    recall_evaluator = (
        MulticlassClassificationEvaluator(

            labelCol="label",

            predictionCol="prediction",

            metricName=(
                "weightedRecall"
            )
        )
    )


    recall = (
        recall_evaluator.evaluate(
            predictions
        )
    )


    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc)
    }


# ============================================================
# 12. MLFLOW CONFIGURATION
# ============================================================

mlflow_cfg = config.get("mlflow", {})
mlflow_uri = mlflow_cfg.get("tracking_uri", "http://mlflow:6091")
mlflow.set_tracking_uri(mlflow_uri)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

exp_name = mlflow_cfg.get("finetune_experiment_name", "Telco Customer Churn - Spark Ray Tune")
mlflow.set_experiment(exp_name)


# ============================================================
# 13. START RAY
#
# local_mode=True is intentional for this project.
#
# This keeps Ray trials in the local Python process and avoids
# creating multiple independent Spark sessions / Java gateways.
#
# max_concurrent_trials=1 also ensures sequential tuning.
# ============================================================

ray.init(

    local_mode=True,

    ignore_reinit_error=True,

    include_dashboard=False
)


# ============================================================
# 14. LOGISTIC REGRESSION RAY TUNE FUNCTION
# ============================================================

def tune_logistic_regression(config):

    pipeline = create_lr_pipeline(

        max_iter=config["maxIter"],

        reg_param=config["regParam"],

        elastic_net_param=(
            config["elasticNetParam"]
        )
    )


    model = (
        pipeline.fit(
            tuning_train_weighted
        )
    )


    metrics = evaluate_model(

        model,

        validation_df
    )


    tune.report(

        roc_auc=metrics["roc_auc"],

        pr_auc=metrics["pr_auc"],

        f1=metrics["f1"],

        accuracy=metrics["accuracy"],

        precision=metrics["precision"],

        recall=metrics["recall"]
    )


# ============================================================
# 15. LOGISTIC REGRESSION SEARCH SPACE
#
# 6 TRIALS ONLY
# ============================================================

lr_search_space = {

    "maxIter":

        tune.choice([
            50,
            100,
            150
        ]),


    "regParam":

        tune.loguniform(
            0.0001,
            0.1
        ),


    "elasticNetParam":

        tune.choice([
            0.0,
            0.25,
            0.5,
            0.75
        ])
}


# ============================================================
# 16. RANDOM FOREST RAY TUNE FUNCTION
# ============================================================

def tune_random_forest(config):

    pipeline = create_rf_pipeline(

        num_trees=(
            config["numTrees"]
        ),

        max_depth=(
            config["maxDepth"]
        ),

        max_bins=(
            config["maxBins"]
        ),

        min_instances_per_node=(
            config[
                "minInstancesPerNode"
            ]
        )
    )


    model = (
        pipeline.fit(
            tuning_train_weighted
        )
    )


    metrics = evaluate_model(

        model,

        validation_df
    )


    tune.report(

        roc_auc=metrics["roc_auc"],

        pr_auc=metrics["pr_auc"],

        f1=metrics["f1"],

        accuracy=metrics["accuracy"],

        precision=metrics["precision"],

        recall=metrics["recall"]
    )


# ============================================================
# 17. RANDOM FOREST SEARCH SPACE
#
# 6 TRIALS ONLY
# ============================================================

rf_search_space = {

    "numTrees":

        tune.choice([
            50,
            100,
            150
        ]),


    "maxDepth":

        tune.choice([
            4,
            6,
            8,
            10
        ]),


    "maxBins":

        tune.choice([
            32,
            64
        ]),


    "minInstancesPerNode":

        tune.choice([
            1,
            2,
            4
        ])
}


# ============================================================
# 18. TUNE LOGISTIC REGRESSION
# ============================================================

print("\n================================")
print("RAY TUNE")
print("LOGISTIC REGRESSION")
print("================================")


with mlflow.start_run(
    run_name=(
        "RayTune_LogisticRegression"
    )
):

    mlflow.set_tag(
        "model_type",
        "LogisticRegression"
    )


    mlflow.set_tag(
        "run_type",
        "hyperparameter_tuning"
    )


    lr_tuner = tune.Tuner(

        tune_logistic_regression,


        param_space=(
            lr_search_space
        ),


        tune_config=(
            tune.TuneConfig(

                metric="roc_auc",

                mode="max",

                num_samples=6,

                max_concurrent_trials=1
            )
        )
    )


    lr_results = (
        lr_tuner.fit()
    )


    best_lr_result = (
        lr_results.get_best_result(

            metric="roc_auc",

            mode="max"
        )
    )


    best_lr_config = (
        best_lr_result.config
    )


    best_lr_validation_metrics = (
        best_lr_result.metrics
    )


    print(
        "\nBest Logistic Regression Parameters:"
    )

    print(
        best_lr_config
    )


    print(
        "\nBest Logistic Regression "
        "Validation Metrics:"
    )

    print(
        best_lr_validation_metrics
    )


    # ----------------------------------------
    # LOG BEST PARAMETERS
    # ----------------------------------------

    mlflow.log_params({

        "best_maxIter":
            best_lr_config["maxIter"],

        "best_regParam":
            best_lr_config["regParam"],

        "best_elasticNetParam":
            best_lr_config[
                "elasticNetParam"
            ],

        "num_tuning_trials":
            6
    })


    # ----------------------------------------
    # LOG BEST VALIDATION METRICS
    # ----------------------------------------

    mlflow.log_metrics({

        "best_validation_roc_auc":
            best_lr_validation_metrics[
                "roc_auc"
            ],

        "best_validation_pr_auc":
            best_lr_validation_metrics[
                "pr_auc"
            ],

        "best_validation_f1":
            best_lr_validation_metrics[
                "f1"
            ],

        "best_validation_accuracy":
            best_lr_validation_metrics[
                "accuracy"
            ],

        "best_validation_precision":
            best_lr_validation_metrics[
                "precision"
            ],

        "best_validation_recall":
            best_lr_validation_metrics[
                "recall"
            ]
    })


    # ----------------------------------------
    # LOG ALL RAY TUNE RESULTS AS CSV ARTIFACT
    # ----------------------------------------

    lr_results_df = (
        lr_results.get_dataframe()
    )


    lr_results_csv = (
        PROJECT_ROOT
        / "ray_tune_lr_results.csv"
    )


    lr_results_df.to_csv(

        lr_results_csv,

        index=False
    )


    mlflow.log_artifact(
        str(lr_results_csv),
        artifact_path=(
            "ray_tune_results"
        )
    )


# ============================================================
# 19. TUNE RANDOM FOREST
# ============================================================

print("\n================================")
print("RAY TUNE")
print("RANDOM FOREST")
print("================================")


with mlflow.start_run(
    run_name=(
        "RayTune_RandomForest"
    )
):

    mlflow.set_tag(
        "model_type",
        "RandomForest"
    )


    mlflow.set_tag(
        "run_type",
        "hyperparameter_tuning"
    )


    rf_tuner = tune.Tuner(

        tune_random_forest,


        param_space=(
            rf_search_space
        ),


        tune_config=(

            tune.TuneConfig(

                metric="roc_auc",

                mode="max",

                num_samples=6,

                max_concurrent_trials=1
            )
        )
    )


    rf_results = (
        rf_tuner.fit()
    )


    best_rf_result = (
        rf_results.get_best_result(

            metric="roc_auc",

            mode="max"
        )
    )


    best_rf_config = (
        best_rf_result.config
    )


    best_rf_validation_metrics = (
        best_rf_result.metrics
    )


    print(
        "\nBest Random Forest Parameters:"
    )

    print(
        best_rf_config
    )


    print(
        "\nBest Random Forest "
        "Validation Metrics:"
    )

    print(
        best_rf_validation_metrics
    )


    # ----------------------------------------
    # LOG BEST PARAMETERS
    # ----------------------------------------

    mlflow.log_params({

        "best_numTrees":
            best_rf_config[
                "numTrees"
            ],

        "best_maxDepth":
            best_rf_config[
                "maxDepth"
            ],

        "best_maxBins":
            best_rf_config[
                "maxBins"
            ],

        "best_minInstancesPerNode":
            best_rf_config[
                "minInstancesPerNode"
            ],

        "num_tuning_trials":
            6
    })


    # ----------------------------------------
    # LOG BEST VALIDATION METRICS
    # ----------------------------------------

    mlflow.log_metrics({

        "best_validation_roc_auc":
            best_rf_validation_metrics[
                "roc_auc"
            ],

        "best_validation_pr_auc":
            best_rf_validation_metrics[
                "pr_auc"
            ],

        "best_validation_f1":
            best_rf_validation_metrics[
                "f1"
            ],

        "best_validation_accuracy":
            best_rf_validation_metrics[
                "accuracy"
            ],

        "best_validation_precision":
            best_rf_validation_metrics[
                "precision"
            ],

        "best_validation_recall":
            best_rf_validation_metrics[
                "recall"
            ]
    })


    # ----------------------------------------
    # LOG ALL RAY TUNE RESULTS AS CSV ARTIFACT
    # ----------------------------------------

    rf_results_df = (
        rf_results.get_dataframe()
    )


    rf_results_csv = (
        PROJECT_ROOT
        / "ray_tune_rf_results.csv"
    )


    rf_results_df.to_csv(

        rf_results_csv,

        index=False
    )


    mlflow.log_artifact(

        str(rf_results_csv),

        artifact_path=(
            "ray_tune_results"
        )
    )


# ============================================================
# 20. RETRAIN BEST LOGISTIC REGRESSION
#
# IMPORTANT:
# Training now uses the COMPLETE ORIGINAL train_df.
#
# This includes:
# - tuning training data
# - validation data
#
# test_df is NOT used for training.
# ============================================================

print("\n================================")
print("FINAL LOGISTIC REGRESSION")
print("TRAINING ON FULL TRAIN DATA")
print("================================")


with mlflow.start_run(
    run_name=(
        "Final_LogisticRegression"
    )
):

    mlflow.set_tag(
        "model_type",
        "LogisticRegression"
    )


    mlflow.set_tag(
        "run_type",
        "final_model"
    )


    final_lr_pipeline = (
        create_lr_pipeline(

            max_iter=(
                best_lr_config[
                    "maxIter"
                ]
            ),

            reg_param=(
                best_lr_config[
                    "regParam"
                ]
            ),

            elastic_net_param=(
                best_lr_config[
                    "elasticNetParam"
                ]
            )
        )
    )


    final_lr_model = (
        final_lr_pipeline.fit(
            full_train_weighted
        )
    )


    final_lr_metrics = (
        evaluate_model(

            final_lr_model,

            test_df
        )
    )


    # ----------------------------------------
    # LOG FINAL PARAMETERS
    # ----------------------------------------

    mlflow.log_params({

        "maxIter":
            best_lr_config[
                "maxIter"
            ],

        "regParam":
            best_lr_config[
                "regParam"
            ],

        "elasticNetParam":
            best_lr_config[
                "elasticNetParam"
            ],

        "training_data":
            "complete_train_df",

        "test_data":
            "held_out_test_df"
    })


    # ----------------------------------------
    # LOG FINAL TEST METRICS
    # ----------------------------------------

    mlflow.log_metrics({

        "test_accuracy":
            final_lr_metrics[
                "accuracy"
            ],

        "test_precision":
            final_lr_metrics[
                "precision"
            ],

        "test_recall":
            final_lr_metrics[
                "recall"
            ],

        "test_f1":
            final_lr_metrics[
                "f1"
            ],

        "test_roc_auc":
            final_lr_metrics[
                "roc_auc"
            ],

        "test_pr_auc":
            final_lr_metrics[
                "pr_auc"
            ]
    })


    # ----------------------------------------
    # LOG MODEL ARTIFACT
    # ----------------------------------------

    mlflow.spark.log_model(

        spark_model=(
            final_lr_model
        ),

        artifact_path=(
            "model"
        )
    )


    print(
        "\nFinal Logistic Regression "
        "Test Metrics:"
    )

    print(
        final_lr_metrics
    )


# ============================================================
# 21. RETRAIN BEST RANDOM FOREST
#
# Uses complete train_df.
# ============================================================

print("\n================================")
print("FINAL RANDOM FOREST")
print("TRAINING ON FULL TRAIN DATA")
print("================================")


with mlflow.start_run(
    run_name=(
        "Final_RandomForest"
    )
):

    mlflow.set_tag(
        "model_type",
        "RandomForest"
    )


    mlflow.set_tag(
        "run_type",
        "final_model"
    )


    final_rf_pipeline = (
        create_rf_pipeline(

            num_trees=(
                best_rf_config[
                    "numTrees"
                ]
            ),

            max_depth=(
                best_rf_config[
                    "maxDepth"
                ]
            ),

            max_bins=(
                best_rf_config[
                    "maxBins"
                ]
            ),

            min_instances_per_node=(

                best_rf_config[
                    "minInstancesPerNode"
                ]
            )
        )
    )


    final_rf_model = (
        final_rf_pipeline.fit(
            full_train_weighted
        )
    )


    final_rf_metrics = (
        evaluate_model(

            final_rf_model,

            test_df
        )
    )


    # ----------------------------------------
    # LOG FINAL PARAMETERS
    # ----------------------------------------

    mlflow.log_params({

        "numTrees":
            best_rf_config[
                "numTrees"
            ],

        "maxDepth":
            best_rf_config[
                "maxDepth"
            ],

        "maxBins":
            best_rf_config[
                "maxBins"
            ],

        "minInstancesPerNode":

            best_rf_config[
                "minInstancesPerNode"
            ],

        "training_data":
            "complete_train_df",

        "test_data":
            "held_out_test_df"
    })


    # ----------------------------------------
    # LOG FINAL TEST METRICS
    # ----------------------------------------

    mlflow.log_metrics({

        "test_accuracy":
            final_rf_metrics[
                "accuracy"
            ],

        "test_precision":
            final_rf_metrics[
                "precision"
            ],

        "test_recall":
            final_rf_metrics[
                "recall"
            ],

        "test_f1":
            final_rf_metrics[
                "f1"
            ],

        "test_roc_auc":
            final_rf_metrics[
                "roc_auc"
            ],

        "test_pr_auc":
            final_rf_metrics[
                "pr_auc"
            ]
    })


    # ----------------------------------------
    # LOG MODEL ARTIFACT
    # ----------------------------------------

    mlflow.spark.log_model(

        spark_model=(
            final_rf_model
        ),

        artifact_path=(
            "model"
        )
    )


    print(
        "\nFinal Random Forest "
        "Test Metrics:"
    )

    print(
        final_rf_metrics
    )


# ============================================================
# 22. FINAL MODEL COMPARISON
# ============================================================

print("\n")
print("================================")
print("FINAL MODEL COMPARISON")
print("================================")


print(
    "\nFINAL LOGISTIC REGRESSION"
)


for metric, value in (
    final_lr_metrics.items()
):

    print(
        f"{metric}: {value:.4f}"
    )


print(
    "\nFINAL RANDOM FOREST"
)


for metric, value in (
    final_rf_metrics.items()
):

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 23. SELECT FINAL BEST MODEL
#
# Selection metric: ROC-AUC
# ============================================================

if (
    final_lr_metrics["roc_auc"]
    >=
    final_rf_metrics["roc_auc"]
):

    best_model_name = (
        "LogisticRegression"
    )

    best_model = (
        final_lr_model
    )

    best_model_metrics = (
        final_lr_metrics
    )


else:

    best_model_name = (
        "RandomForest"
    )

    best_model = (
        final_rf_model
    )

    best_model_metrics = (
        final_rf_metrics
    )


print("\n")
print("================================")
print("BEST FINAL MODEL")
print("================================")

print(
    "Model:",
    best_model_name
)

print(
    "ROC-AUC:",
    best_model_metrics[
        "roc_auc"
    ]
)

print(
    "PR-AUC:",
    best_model_metrics[
        "pr_auc"
    ]
)

print(
    "F1:",
    best_model_metrics[
        "f1"
    ]
)


# ============================================================
# 24. LOG BEST MODEL SUMMARY
#
# This run stores the overall comparison.
# ============================================================

with mlflow.start_run(
    run_name=(
        "Best_Model_Selection"
    )
):

    mlflow.set_tag(
        "best_model",
        best_model_name
    )


    mlflow.log_param(

        "selection_metric",

        "roc_auc"
    )


    mlflow.log_metrics({

        "best_model_roc_auc":
            best_model_metrics[
                "roc_auc"
            ],

        "best_model_pr_auc":
            best_model_metrics[
                "pr_auc"
            ],

        "best_model_f1":
            best_model_metrics[
                "f1"
            ],

        "best_model_accuracy":
            best_model_metrics[
                "accuracy"
            ],

        "best_model_precision":
            best_model_metrics[
                "precision"
            ],

        "best_model_recall":
            best_model_metrics[
                "recall"
            ]
    })


# ============================================================
# 25. CLEANUP
# ============================================================

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")


# Stop Ray
ray.shutdown()


# Unpersist Spark DataFrames
tuning_train_df.unpersist()

validation_df.unpersist()

train_df.unpersist()

test_df.unpersist()

tuning_train_weighted.unpersist()

full_train_weighted.unpersist()


# Stop Spark
spark.stop()