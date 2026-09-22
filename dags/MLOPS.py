from datetime import datetime
import pandas as pd
import os 
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator,PythonVirtualenvOperator 
from src.etl import preprocess_data
from src.TravelRecommendation.travelrecom import log_model_to_mlflow
#from airflow.hooks.base import BaseHook
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
        python_callable=preprocess_data,  # Point to the function you want to run
        
    )
    
    def train_and_log_model():
        # Your existing log_model_to_mlflow function       
        from src.TravelRecommendation.travelrecom import log_model_to_mlflow
        log_model_to_mlflow()
    
    Travel_recom_task = PythonVirtualenvOperator (
        task_id="Travel_recom_task",          # You must explicitly provide a task_id
        python_callable=train_and_log_model)
        
    

    
    etl_task >> Travel_recom_task 
