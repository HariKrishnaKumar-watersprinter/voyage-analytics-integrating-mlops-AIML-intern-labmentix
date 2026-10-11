import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def flight_dashboard(df):
    st.title("Flight Analytics")
    
    # ---------------------------------------------------------
    # 1. Price Distribution & Outlier Check
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, x='price', nbins=50, 
            title='Price Distribution',
            marginal='violin', # Replaces kde=True with a smooth distribution curve
            template='plotly_white'
        )
        fig_hist.update_layout(barmode='overlay', xaxis_title='Price', yaxis_title='Count')
        st.plotly_chart(fig_hist, width='stretch')

    with col2:
        fig_box = px.box(
            df, y='price', 
            title='Price — Outlier Check',
            template='plotly_white'
        )
        fig_box.update_layout(yaxis_title='Price')
        st.plotly_chart(fig_box, width='stretch')


    # ---------------------------------------------------------
    # 2. Price by Flight Type
    # ---------------------------------------------------------
    col3, col4 = st.columns(2)
    with col3:
        fig_flight_type = px.box(
        df, x='flightType', y='price', 
        title='Price by Flight Type',
        color='flightType',
        template='plotly_white')
        st.plotly_chart(fig_flight_type, width='stretch')


    # ---------------------------------------------------------
    # 3. Price by Agency
    # ---------------------------------------------------------
    with col4:
        fig_agency = px.box(
        df, x='agency', y='price', 
        title='Price by Agency',
        color='agency',
        template='plotly_white'
    )
        st.plotly_chart(fig_agency, width='stretch')


    # ---------------------------------------------------------
    # 4. Price vs Distance & Price vs Time (Scatterplots)
    # ---------------------------------------------------------
    col5, col6 = st.columns(2)
    df_sample = df.sample(5000, random_state=42) # Sample once to keep data consistent

    with col5:
        fig_dist = px.scatter(
            df_sample, x='distance', y='price', 
            title='Price vs Distance (5k sample)',
            opacity=0.3, # Replaces seaborn alpha
            template='plotly_white'
        )
        st.plotly_chart(fig_dist, width='stretch')

    with col6:
        fig_time = px.scatter(
            df_sample, x='time', y='price', 
            title='Price vs Time (5k sample)',
            opacity=0.3,
            template='plotly_white'
        )
        st.plotly_chart(fig_time, width='stretch')


    # ---------------------------------------------------------
    # 5. Top 15 Routes by Flight Count
    # ---------------------------------------------------------
    col7, col8 = st.columns(2)
    with col7:
        df['route'] = df['from'] + ' -> ' + df['to']
        top_routes = df['route'].value_counts().head(15).reset_index()
        top_routes.columns = ['route', 'count']

        fig_routes = px.bar(
        top_routes, x='count', y='route', 
        title='Top 15 Routes by Flight Count',
        orientation='h', # Horizontal bar chart
        template='plotly_white')
    # Reverse Y-axis to match Seaborn's top-to-bottom ordering
        fig_routes.update_layout(
        xaxis_title='Number of Flights', 
        yaxis_title='', 
        yaxis=dict(autorange="reversed"))
        fig_routes.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_routes, width='stretch')


    # ---------------------------------------------------------
    # 6. Average Flight Price by Month (Line Chart)
    # ---------------------------------------------------------
    with col8:
        df['date_parsed'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df['month'] = df['date_parsed'].dt.to_period('M')
    
        monthly_avg_price = df.groupby('month')['price'].mean().reset_index()
    # Convert Period to timestamp for Plotly compatibility
        monthly_avg_price['month'] = monthly_avg_price['month'].dt.to_timestamp()

        fig_monthly = px.line(
        monthly_avg_price, x='month', y='price', 
        title='Average Flight Price by Month',
        markers=True, # Replaces marker='o'
        template='plotly_white'
    )
        fig_monthly.update_layout(
        xaxis_title='Month', 
        yaxis_title='Avg Price',
        xaxis_tickangle=-45 # Replaces ax4.tick_params(axis='x', rotation=45)
    )
        st.plotly_chart(fig_monthly, width='stretch')


    # ---------------------------------------------------------
    # 7. Correlation Heatmap
    # ---------------------------------------------------------
    numeric_cols = ['price', 'time', 'distance']
    corr_matrix = df[numeric_cols].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r', # Red-Blue reversed scale mimics seaborn 'coolwarm'
        zmin=-1, zmax=1, # Sets the scale range identical to seaborn
        text=corr_matrix.round(2).values, # Rounding for cleaner annotations
        texttemplate="%{text}", # Displays the text inside the boxes
        hoverinfo='z'
    ))
    fig_heatmap.update_layout(
        title='Correlation Heatmap — Flights',
        template='plotly_white'
    )
    st.plotly_chart(fig_heatmap, width='stretch')
