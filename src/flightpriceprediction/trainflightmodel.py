import pandas as pd
import numpy as np
import sys
import pathlib
import os
import mlflow
from mlflow.pyfunc import PythonModel # Import PythonModel base class
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
#import re
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from database import engine



load_dotenv()

def evaluate_model(model, x_train1, X_test1, y_train1, y_test1):
    # Fit model and predict
    model_pred = model.fit(x_train1, y_train1)
    y_pred = model_pred.predict(X_test1)
    
    metrics = { 
        'mae': mean_absolute_error(y_test1, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test1, y_pred)),
        'r2': r2_score(y_test1, y_pred)
    }
    
    return model_pred, y_pred, pd.DataFrame(metrics, index=[0])
   
def flight_model():
    flight = pd.read_sql("SELECT * FROM flight", engine)
    
    # Feature engineering
    flight['route'] = flight['from'] + ' -> ' + flight['to']
    flight['date_parsed'] = pd.to_datetime(flight['date'], format='%m/%d/%Y')
    
    # FIX 1: Convert Period to string or drop it completely. 
    # MLflow cannot serialize Pandas Period objects, causing INVALID_PARAMETER_VALUE.
    flight['month'] = flight['date_parsed'].dt.to_period('M').astype(str) 
    
    flight['day_of_week'] = flight['date_parsed'].dt.dayofweek   # 0=Monday
    flight['is_weekend'] = flight['day_of_week'].isin([5, 6]).astype(int)
    flight['month_num'] = flight['date_parsed'].dt.month
    flight['quarter'] = flight['date_parsed'].dt.quarter
    flight.drop_duplicates(inplace=True)
    
    # Outlier handling (Note: This currently just filters the dataframe but doesn't reassign it. 
    # If you meant to remove outliers, you need: flight = flight[~((flight['price'] < lower_bound) | (flight['price'] > upper_bound))])
    Q1 = flight['price'].quantile(0.25)
    Q3 = flight['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    flight = flight[~((flight['price'] < lower_bound) | (flight['price'] > upper_bound))]
     
    # Data transform
    categorical_cols = ['from', 'to', 'flightType', 'agency']
    flights_encoded = pd.get_dummies(flight, columns=categorical_cols, drop_first=True,dtype=int)
    drop_cols = ['travelCode', 'userCode', 'date', 'date_parsed', 'month', 'route']
    feature_cols = [c for c in flights_encoded.columns if c not in drop_cols + ['price']]
    
    # Splitting
    X = flights_encoded[feature_cols]
    y = flights_encoded['price']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    pipeline_steps = [('scaler', RobustScaler())]
    preprocess_pipeline = Pipeline(pipeline_steps)
    x_train_scaled = preprocess_pipeline.fit_transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
    x_test_scaled = preprocess_pipeline.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)
    
    # Model training
    print('🚀 Start Model building')
    xgb_pipeline = Pipeline(steps=[('model', XGBRegressor(random_state=42, subsample=1.0, n_estimators=200, max_depth=10, learning_rate=0.1, colsample_bytree=1.0))])
    
    # FIX 2: Removed the redundant xgb_model.fit(x_train, y_train) at the end 
    # which was overwriting the properly trained scaled model with an unscaled one.
    xgb_model, xgb_pred, xgb_metrics = evaluate_model(xgb_pipeline, x_train_scaled, x_test_scaled, y_train, y_test)
    print(xgb_metrics)
    pure_xgb = xgb_model.named_steps["model"]
    # MLflow Tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Flight_Price_Prediction")
    #mlflow.autolog() # Autolog will now work without serialization errors
    
    with mlflow.start_run(run_name="XGBoost_Flight_Price_Prediction"):
        mae = mean_absolute_error(y_test, xgb_pred)
        rmse = np.sqrt(mean_squared_error(y_test, xgb_pred)) 
        r2 = r2_score(y_test, xgb_pred)
        mlflow.log_params(xgb_model.get_params())
        mlflow.log_metric("mean absolute error", mae)
        mlflow.log_metric("root mean squared error", rmse)
        mlflow.log_metric("r2 score", r2)   
        mlflow.set_tag('Training Info', 'XGBoost model for Flight price prediction')
        signature1 = mlflow.models.infer_signature(x_test_scaled, xgb_model.predict(x_test_scaled))
        mlflow.xgboost.log_model(
            pure_xgb,  
            name="xgboost_flight_prediciton", 
            signature=signature1,
            input_example=x_test_scaled.head(5),
            registered_model_name="xgboost_flight_prediciton" 
        )
        mlflow.sklearn.log_model(
        preprocess_pipeline,
        name="preprocess_pipeline",
        serialization_format="cloudpickle",
        registered_model_name="flight_preprocess_pipeline"
    )
        print("Model and pipeline successfully logged to MLflow!")

if __name__ == "__main__":
    flight_model()
    