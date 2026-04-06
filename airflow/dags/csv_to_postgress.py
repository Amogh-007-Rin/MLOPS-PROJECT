from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

CSV_PATH = "/opt/airflow/data/neo.csv"
TABLE_NAME = "neo"
POSTGRES_CONNECTION = "postgresql+psycopg2://mlops:password123@postgres:5432/mlops"

def load_csv_to_postgres():
    df = pd.read_csv(CSV_PATH)
    engine = create_engine(POSTGRES_CONNECTION)
    df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)

with DAG(
    dag_id="csv_to_postgres",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    load_task = PythonOperator(
        task_id="load_csv_to_postgres",
        python_callable=load_csv_to_postgres
    )
