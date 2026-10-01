from pydantic_settings import BaseSettings
import psycopg2
import os 
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
   
    database_url: str 
    

    class Config:
        env_file = ".env"


settings = Settings()
