import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def hotel_dashboard(df):
    st.title("Hotel Analytics")

    # ---------------------------------------------------------
    # 1. Distributions: Price, Days, Total Cost (1x3 layout)
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_price = px.histogram(
            df, x='price', nbins=40,
            title='Price per Night Distribution',
            marginal='violin', # Replaces kde=True with a smooth distribution curve
            template='plotly_white'
        )
        fig_price.update_layout(barmode='overlay', xaxis_title='Price', yaxis_title='Count')
        st.plotly_chart(fig_price, width='stretch')

    with col2:
        # For discrete days, we extract the edges to mimic seaborn's bins=range(...)
        # This ensures bars are centered on the integer day values
        day_max = df['days'].max()
        day_bins = dict(start=0.5, end=day_max + 1.5, size=1)
        
        fig_days = px.histogram(
            df, x='days',
            title='Stay Length (days) Distribution',
            template='plotly_white'
        )
        fig_days.update_traces(xbins=day_bins)
        fig_days.update_layout(xaxis_title='Days', yaxis_title='Count', bargap=0.2)
        fig_days.update_traces(texttemplate='%{y:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_days, width='stretch')

    with col3:
        fig_total = px.histogram(
            df, x='total', nbins=40,
            title='Total Cost Distribution',
            marginal='violin', # Replaces kde=True
            template='plotly_white'
        )
        fig_total.update_layout(barmode='overlay', xaxis_title='Total Cost', yaxis_title='Count')
        st.plotly_chart(fig_total, width='stretch')


    # ---------------------------------------------------------
    # 2. Top 15 Places by Avg Price/Night
    # ---------------------------------------------------------
    col4, col5 = st.columns(2)
    with col4:
        place_avg_price = df.groupby('place')['price'].mean().sort_values(ascending=False).head(15).reset_index()
        place_avg_price.columns = ['place', 'avg_price'] # Rename for Plotly

        fig_places = px.bar(
        place_avg_price, x='avg_price', y='place',
        title='Top 15 Places by Avg Price/Night',
        orientation='h', # Horizontal bar chart
        template='plotly_white'
    )
    # Reverse Y-axis to match Seaborn's top-to-bottom descending order
        fig_places.update_layout(
        xaxis_title='Avg Price',
        yaxis_title='',
        yaxis=dict(autorange="reversed")
    )
        fig_places.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_places, width='stretch')


    # ---------------------------------------------------------
    # 3. Top 15 Most-Booked Hotels
    # ---------------------------------------------------------
    with col5:
        top_hotels = df['name'].value_counts().head(15).reset_index()
        top_hotels.columns = ['name', 'count'] # Rename for Plotly

        fig_hotels = px.bar(
        top_hotels, x='count', y='name',
        title='Top 15 Most-Booked Hotels',
        orientation='h', # Horizontal bar chart
        template='plotly_white'
    )
    # Reverse Y-axis to match Seaborn's top-to-bottom descending order
        fig_hotels.update_layout(
        xaxis_title='Number of Bookings',
        yaxis_title='',
        yaxis=dict(autorange="reversed")
    )
        fig_hotels.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_hotels, width='stretch')

