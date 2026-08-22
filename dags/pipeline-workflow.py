from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

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
    tags=['mlops', 'spark', 'mlflow', 'docker']
) as dag:

    # Common DockerOperator configurations
    docker_operator_kwargs = {
        'image': 'ghcr.io/bmoinak98/mlops-project-team23/ml-runner:latest',
        'auto_remove': 'success',
        'docker_url': 'unix:///var/run/docker.sock',
        'network_mode': 'mlops-net',
        'working_dir': '/app/src/data_engineering',
        'environment': {
            'PYTHONPATH': '/app/src/data_engineering',
        },
        'mounts': [
            Mount(source='/home/da25m591/data', target='/data', type='bind'),
            Mount(source='/home/da25m591/project/src/data_engineering', target='/app/src/data_engineering', type='bind'),
            Mount(source='/home/da25m591/project/config', target='/app/config', type='bind'),
        ],
    }

    data_cleaning_task = DockerOperator(
        task_id='data_cleaning',
        command='python3 /app/src/data_engineering/data_clean.py --config /app/config/config-docker.yml',
        **docker_operator_kwargs
    )

    model_training_task = DockerOperator(
        task_id='model_training',
        command='python3 /app/src/data_engineering/model_train.py --config /app/config/config-docker.yml',
        **docker_operator_kwargs
    )

    data_cleaning_task >> model_training_task