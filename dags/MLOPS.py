from datetime import datetime
import pandas as pd
import os 
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.etl import preprocess_data
#from airflow.utils.dates import days_ago
# ---------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------

with DAG(
    dag_id="voyage_folder",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "voyage_analytics"],) as dag:


    etl_task = PythonOperator(
        task_id="etl_task",          # You must explicitly provide a task_id
        python_callable=preprocess_data  # Point to the function you want to run
    )


    
    etl_task 
