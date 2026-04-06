from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_file():
    with open("./Dataset/Raw-Dataset/neo.csv") as f:
        print(f.read())

with DAG(
    "read_neo_csv",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@once",
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id="print_csv",
        python_callable=print_file)
