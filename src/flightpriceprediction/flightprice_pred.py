import os
import sys
import pathlib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import streamlit as st
import requests
load_dotenv()

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
#from src.flightpriceprediction.app import df

def flight_price_pred(df):   
    st.title("💰 Flight Price Prediction")
    options = {
        "cities": list(df["from"].unique()),
        "agencies": list(df["agency"].unique()),
        "flight_types": list(df["flightType"].unique()),
    }
    c1, c2 = st.columns(2)
    frm = c1.selectbox("From", options["cities"])
    to = c2.selectbox("To", options["cities"], index=min(1, len(options["cities"]) - 1))
    ftype = st.radio("Flight type", options["flight_types"], horizontal=True)
    agency = st.selectbox("Agency", options["agencies"])
    d = st.date_input("Departure date", format="MM/DD/YYYY")
    hit = (df["from"] == frm) & (df["to"] == to)
    a1, a2 = st.columns(2)
    
    use_api = st.toggle("Call REST API (Docker/K8s) instead of local model")
    api_url = st.text_input("API base URL", "http://flask_backend:5000", disabled=not use_api)
    
    if st.button("Predict price", type="primary"):
        payload = {
            "from": str(frm), 
            "to": str(to), 
            "flightType": str(ftype), 
            "agency": str(agency),
            "date": d.strftime("%m/%d/%Y")
        }
        
        # Initialize distance so it doesn't cause UnboundLocalError later
        distance = None 
        
        try:
            if use_api:
                r = requests.post(f"{api_url}/predict", json=payload)
                r.raise_for_status() # This will jump to except block if not 200
                price = r.json()["predicted_price"]
                distance = r.json()["distance"]
            else:
                row = {k: payload[k] for k in fl_p["categorical"]}
                row.update()
                price = float(price_m.predict(pd.DataFrame([row])[fl_p["numeric"] + fl_p["categorical"]])[0])
                
            st.success(f"Estimated ticket price: **R$ {price:,.2f}**")
            
            # Display results (handle missing distance gracefully)
            result_data = {"from": frm, "to": to, "flight_type": ftype, "agency": agency, "price": price}
            if distance is not None:
                result_data["distance"] = distance
            st.dataframe(pd.DataFrame([result_data]))
            
            if not hit.empty:
                hist = df[(df["from"] == frm) & (df["to"] == to)]["price"].mean()
                st.caption(f"Route historical average: R$ {hist:,.2f}")
                
        except Exception as e:
            # Check if 'r' exists in local scope AND if it's an API error
            if use_api and 'r' in locals() and hasattr(r, 'text'):
                st.error(f"API Error Response: {r.text}") 
            else:
                st.error(f"Prediction failed: {e}")
