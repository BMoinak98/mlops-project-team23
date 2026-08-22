from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'team23',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='telco_churn_mlops_pipeline',
    default_args=default_args,
    description='Telco Customer Churn Data Cleaning and MLflow Model Training Pipeline',
    start_date=datetime(2026, 1, 1),
    schedule='@daily',
    catchup=False,
    tags=['mlops', 'spark', 'mlflow']
) as dag:

    data_cleaning_task = BashOperator(
        task_id='data_cleaning',
        bash_command='python3 /app/src/data_engineering/data_clean.py --config /app/config/config-docker.yml',
        env={
            'PYTHONPATH': '/app/src/data_engineering',
            'MLFLOW_TRACKING_URI': 'http://mlflow:6091',
        }
    )

    model_training_task = BashOperator(
        task_id='model_training',
        bash_command='python3 /app/src/data_engineering/model_train.py --config /app/config/config-docker.yml',
        env={
            'PYTHONPATH': '/app/src/data_engineering',
            'MLFLOW_TRACKING_URI': 'http://mlflow:6091',
        }
    )

    data_cleaning_task >> model_training_task