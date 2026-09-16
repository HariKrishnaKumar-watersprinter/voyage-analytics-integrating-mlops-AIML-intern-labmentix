from datetime import datetime
import pandas as pd
import os 
#from astro import sql as aql
import io
import gdown
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from database import engine
# Define the Google Drive FOLDER ID (Replace with your actual Folder ID)


def download_to_memory(file_id_or_url):
    """
    Downloads a file from Google Drive directly into a memory buffer (BytesIO)
    instead of writing it to the local disk.
    """
    print("Downloading dataset from Google Drive into memory...")
    buffer = io.BytesIO()
    
    # gdown.download writes to a file-like object seamlessly
    gdown.download(file_id_or_url, buffer, quiet=False)
    
    # Reset buffer pointer to the beginning so pandas can read it
    buffer.seek(0)
    return buffer
def preprocess_data():
    """Preprocessing logic."""
    print("Preprocessing data...")
    # --- 1. DATA CLEANING ---
    # flight dataset
    path="https://drive.google.com/file/d/15YcZLQpGbuGV7VcNwbOaDtmRnzt8i6mN/view?usp=sharing"
    memory_buffer = download_to_memory(path)
    df_raw = pd.read_csv(memory_buffer)
    memory_buffer.close()
    df = df_raw.dropna()
    df = df_raw.drop_duplicates()
    try:
        df.to_sql(
            name='flight',
            con=engine,
            if_exists='replace',
            index=False,
          
        )
        print(f"Successfully loaded flight dataset.")
    except Exception as e:
        print(f"Error loading flight dataset: {e}")
    
    #hotel.csv dataset 
    path="https://drive.google.com/file/d/1rAnqFGB0C8gqYZW_bhNyhitXXz7Ro73q/view?usp=sharing"
    memory_buffer = download_to_memory(path)
    df_raw = pd.read_csv(memory_buffer)
    memory_buffer.close()
    df = df_raw.dropna()
    df = df_raw.drop_duplicates()
    try:
        df.to_sql(
            name='hotel',
            con=engine,
            if_exists='replace',
            index=False,
          
        )
        print(f"Successfully loaded hotel dataset.")
    except Exception as e:
        print(f"Error loading hotel dataset: {e}")
    
    
    
    #user.csv
    path="https://drive.google.com/file/d/1DLsL8y9Uct2UlNaBmjfO051cs5Tx8Dwh/view?usp=sharing"
    memory_buffer = download_to_memory(path)
    df_raw = pd.read_csv(memory_buffer)
    memory_buffer.close()
    df = df_raw.dropna()
    df = df_raw.drop_duplicates()
    try:
        df.to_sql(
            name='user',
            con=engine,
            if_exists='replace',
            index=False,
          
        )
        print(f"Successfully loaded user dataset.")
    except Exception as e:
        print(f"Error loading user dataset: {e}")
    
if __name__=="__main__":
        preprocess_data()

