import pandas as pd
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from database import engine

flight= pd.read_sql("SELECT * FROM flight", engine)
hotel= pd.read_sql("SELECT * FROM hotel", engine)
user= pd.read_sql("SELECT * FROM user", engine)


