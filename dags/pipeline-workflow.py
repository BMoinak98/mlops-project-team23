from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id='docker_python_app_pipeline',
    start_date=datetime(2026, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['production', 'docker']
) as dag:

    run_app_container = DockerOperator(
        task_id='execute_python_job',
        image='ghcr.io/YOUR_GITHUB_USERNAME/my-python-app:latest',
        command='python main.py',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='host'
    )