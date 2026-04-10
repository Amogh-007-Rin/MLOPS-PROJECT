from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from etl import run_etl

default_args = {
    'owner': 'snehdeep',
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'nasa_etl_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

run_etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl,
    dag=dag
)

run_etl_task
