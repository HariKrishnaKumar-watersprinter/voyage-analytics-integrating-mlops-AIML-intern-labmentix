import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
import sys
import pathlib
import os
import datetime
from datetime import datetime
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
from src.gender_Classification_model.gen_train import user_df
from src.flightpriceprediction.app import df

PREPROCESSOR = None
PREDICTOR = None
TRAINED_DATE = None
CURRENT_VERSIONS = {
    "Linear_SVM_gender_classification": None,
    "gender_classification_preprocess_pipeline": None
}




def get_latest_model_uri(model_name: str) -> tuple | None:
    client = MlflowClient()
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No registered model found with name '{model_name}'")
        
        latest_version = max(model_versions, key=lambda mv: int(mv.version))
        timestamp_ms = latest_version.creation_timestamp
        timestamp_sec = timestamp_ms / 1000.0
        trained_date = datetime.utcfromtimestamp(timestamp_sec)
        model_uri = f"models:/{model_name}/{latest_version.version}"
        print(f"Latest version found for {model_name}: v{latest_version.version}")
        return model_uri, trained_date, latest_version.version
    except Exception as e:
        print(f"Error getting latest model: {e}")
        return None


def load_models_into_state_gen(force_reload: bool = False):
    """Loads both models into global memory. 
    If force_reload=False, it only loads if a new version is detected."""
    global PREPROCESSOR, PREDICTOR, TRAINED_DATE, CURRENT_VERSIONS

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("Environment variable MLFLOW_TRACKING_URI is not set")

    mlflow.set_tracking_uri(tracking_uri)

    # 1. Load Preprocessor
    prep_result = get_latest_model_uri('gender_classification_preprocess_pipeline')
    if prep_result:
        prep_uri, _, prep_version = prep_result
        # Check if update is needed
        if force_reload or CURRENT_VERSIONS["gender_classification_preprocess_pipeline"] != prep_version:
            print(f"[Auto-Reload] New preprocessor version detected (v{prep_version}). Reloading...")
            try:
                PREPROCESSOR = mlflow.sklearn.load_model(prep_uri)
                CURRENT_VERSIONS["gender_classification_preprocess_pipeline"] = prep_version
                print("[Auto-Reload] Preprocessor loaded successfully!")
            except Exception as e:
                print(f"[Auto-Reload] Error loading preprocessor: {e}")
        elif not force_reload:
            pass # Silently skip if up to date (avoids spamming logs every 60s)

    # 2. Load Predictor
    pred_result = get_latest_model_uri('Linear_SVM_gender_classification')
    if pred_result:
        pred_uri, trained_date, pred_version = pred_result
        # Check if update is needed
        if force_reload or CURRENT_VERSIONS["Linear_SVM_gender_classification"] != pred_version:
            print(f"[Auto-Reload] New predictor version detected (v{pred_version}). Reloading...")
            try:
                PREDICTOR = mlflow.pyfunc.load_model(pred_uri)
                TRAINED_DATE = trained_date
                CURRENT_VERSIONS["Linear_SVM_gender_classification"] = pred_version
                print("[Auto-Reload] Predictor model loaded successfully!")
            except Exception as e:
                print(f"[Auto-Reload] Error loading predictor: {e}")
        elif not force_reload:
            pass
    return PREPROCESSOR,PREDICTOR,TRAINED_DATE

#def background_model_poller():
   # """Background thread function to periodically check for new models."""
  #  print(f"[Background Poller] Starting. Will check MLflow every {POLL_INTERVAL_SECONDS} seconds.")
   # while True:
       # time.sleep(POLL_INTERVAL_SECONDS)
      #  try:
            # force_reload=False means it only downloads if the version number changed
          #  load_models_into_state(force_reload=False)
      #  except Exception as e:
           # print(f"[Background Poller] Error during poll: {e}")


def get_underlying_model_gen(pyfunc_model):
    try:
        return pyfunc_model.unwrap_python_model()
    except Exception:
        pass
    try:
        return pyfunc_model._model_impl.python_model
    except Exception:
        pass
    try:
        return pyfunc_model._model_impl
    except Exception:
        pass
    raise AttributeError("Could not extract the underlying Python model")

def gen_pred(user_df,df,PREPROCESSOR,PREDICTOR): 
    
    st.title("🚻 Gender Classification")
    options = {
        "companies": list(user_df["company"].unique()),
        "agencies": list(df["agency"].unique()),
        "flight_types": list(df["flightType"].unique()),
    }
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name(optional — boosts accuracy)",None)
    
        
    age = c2.slider("Age", 18, 80, 32)
    company = c3.selectbox("Company", options["companies"])
    c4, c5, c6 = st.columns(3)
    ftype = c4.selectbox("Preferred flight type", options["flight_types"])
    agency = c5.selectbox("Preferred agency", options["agencies"])
    trips = c6.slider("Total trips", 1, 60, 10)
    c7, c8, c9 = st.columns(3)
    avg_price = c7.number_input("Avg ticket (R$)", 0.0, 5000.0, 700.0, 50.0)
    spend = c8.number_input("Total flight spend (R$)", 0.0, 200000.0, 8000.0, 500.0)
    hotel_spend = c9.number_input("Total hotel spend (R$)", 0.0, 200000.0, 6000.0, 500.0)
    if st.button("Classify gender", type="primary"):
        if name is None or name.strip() == "":
            st.error("Name is required")
            st.stop()
        data=pd.DataFrame([{"company":company,"name":name,"age":age,}])
        
        data["name"] = data["name"].str.lower().str.strip()

        data["first_name"] = data["name"].str.split().str[0]

        data["first_name_length"] = (data["first_name"].str.len())

        data["last_letter"] = (data["first_name"].str[-1])

        data["vowel_count"] = (data["first_name"].str.count(r"[aeiou]"))

        data["vowel_ratio"] = (data["vowel_count"]/ data["first_name_length"].replace(0, np.nan)).fillna(0)

        data["age"] = pd.to_numeric(data["age"],errors="coerce")
        data=data.drop(columns=['name'])
        
        #PREPROCESSOR1=get_underlying_model(PREPROCESSOR)
        data = PREPROCESSOR.transform(data)
        
        new_column_names = PREPROCESSOR.get_feature_names_out()
        
        data=pd.DataFrame(data.toarray(),columns=new_column_names)
        
        pred = PREDICTOR.predict(data)[0]
        #score = PREDICTOR.decision_function(data)[0]
        
        
        #print(score)
        if pred == 1 :
            label = "Female"
        else:
            label = "Male"
       
        st.metric("Predicted gender", label, f"predicted value({label}) = {pred:.3f}")
       
        
        X = pd.DataFrame([{"name":name,"company":company,"age":age,'Gender':label,'Preferred flight type':ftype,
        'Preferred agency':agency,'Total trips':trips,'Avg ticket (R$)':avg_price,'Total flight spend (R$)':spend,'Total hotel spend (R$)':hotel_spend}])
        st.dataframe(X)
        
       

if __name__ == "__main__":
    load_models_into_state()
    gen_pred()