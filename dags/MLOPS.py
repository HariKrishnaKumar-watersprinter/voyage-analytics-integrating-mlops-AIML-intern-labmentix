from datetime import datetime
import pandas as pd
import os 
from airflow import DAG
from astro import sql as aql
from astro.files import File
from astro.sql.table import Table, Metadata



# ---------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------

with DAG(
    dag_id="voyage_folder",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "voyage_analytics"],
) as dag:

    
    pass