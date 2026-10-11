import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.database import engine
from src.flightpriceprediction.flightprice_pred import flight_price_pred
from src.gender_Classification_model.gender_pred import gen_pred,load_models_into_state_gen 
from src.TravelRecommendation.recomm_pred import stramlit_trav_recomm,load_model_for_inference
from dashboard.user_dash import user_dashboard
from dashboard.flight_dash import flight_dashboard
from dashboard.hotel_dash import hotel_dashboard

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")
@st.cache_resource(show_spinner="Loading data")
def Load_data():
    # Load data
    flight= pd.read_sql( 'SELECT * FROM flight',engine)
    user_df= pd.read_sql('SELECT * FROM user_data',engine)
    hotel = pd.read_sql('SELECT * FROM hotel',engine)
    return flight,user_df,hotel
flight,user_df,hotel=Load_data()
@st.cache_resource(show_spinner="Loading model")
def load_model():
    preprocessor_gen,predictor_gen,_= load_models_into_state_gen()
    model=load_model_for_inference()
    return preprocessor_gen,predictor_gen,model
preprocessor_gen,predictor_gen,model = load_model()
MENU = ["🏠 Overview", "📊 EDA Dashboard", 
        "💰 Flight Price Prediction", "🚻 Gender Classification",
        "🏨 Travel Recommendations"]

page = st.sidebar.radio("Navigate", MENU)

if page == MENU[0]:
    
    
    st.title("✈️ Voyage Analytics")
    c = st.columns(5)
    c[0].metric("Flight records", f"{len(flight):,}")
    c[1].metric("Travellers", f"{len(user_df):,}")
    c[2].metric("Hotel stays", f"{len(hotel):,}")
    c[3].metric("Avg ticket (R$)", f"{flight['price'].mean():,.0f}")
    c[4].metric("Top agency", flight["agency"].mode().iat[0])
    st.dataframe(flight.sample(5, random_state=1), width='stretch')
    st.dataframe(user_df.sample(5, random_state=1), width='stretch')
    st.dataframe(hotel.sample(5, random_state=1), width='stretch')
if page == MENU[1]:
    st.title("📊 EDA Dashboard")
    t1, t2, t3 = st.tabs(["User Analytics", "Flight Analytics", "Hotel Analytics"])
    with t1:
        user_dashboard(user_df)
    with t2:
        flight_dashboard(flight)
    with t3:
        hotel_dashboard(hotel)
if page == MENU[2]:
    flight_price_pred(flight)

if page == MENU[3]:
    gen_pred(user_df,flight,preprocessor_gen,predictor_gen)


if page == MENU[4]:    
    stramlit_trav_recomm(model,hotel)