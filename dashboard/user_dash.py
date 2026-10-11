import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def user_dashboard(df):
    st.title("User Analytics")
    
    # ---------------------------------------------------------
    # 1. Gender Distribution (Already Plotly)
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        gender_counts = df['gender'].value_counts()
    
        fig_gender = go.Figure(data=go.Bar(
        x=gender_counts.index,
        y=gender_counts.values,
        marker_color=['blue', 'pink', 'grey'] 
    ))
    
        fig_gender.update_layout(
        title='Gender Distribution',
        xaxis_title='Gender',
        yaxis_title='Count',
        template='plotly_white'
    )
        fig_gender.update_traces(texttemplate='%{y:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_gender, width='stretch')

    # ---------------------------------------------------------
    # 2. Age Distribution (Histogram + KDE) & Outlier Check (Boxplot)
    # ---------------------------------------------------------
    # Use st.columns to mimic the 1x2 subplot layout
    

    with col2:
        # Histogram with KDE (Marginal violin plot acts as a great KDE alternative in Plotly)
        fig_hist = px.histogram(
            df, 
            x='age', 
            nbins=20, 
            title='Age Distribution',
            marginal='violin', # Adds a KDE-like smooth curve on top
            template='plotly_white'
        )
        # Update barmode to overlay the marginal plot nicely
        fig_hist.update_layout(barmode='overlay', xaxis_title='Age', yaxis_title='Count')
        #fig_hist.update_traces(hovertemplate='%{y:.2f}',textfont_size=20,textposition='auto')
        st.plotly_chart(fig_hist, width='stretch')
    
    col3, col4 = st.columns(2)
    with col3:
        # Boxplot for Outlier Check
        fig_box = px.box(
            df, 
            y='age', 
            title='Age — Outlier Check',
            template='plotly_white'
        )
        fig_box.update_layout(yaxis_title='Age')
        st.plotly_chart(fig_box, width='stretch')

    # ---------------------------------------------------------
    # 3. Age by Gender (Boxplot)
    # ---------------------------------------------------------
    with col4:
        fig_gender_age = px.box(
        df, 
        x='gender', 
        y='age', 
        title='Age by Gender',
        color='gender', # Adds distinct colors for different genders
        template='plotly_white')
        fig_gender_age.update_layout(xaxis_title='Gender', yaxis_title='Age')
        st.plotly_chart(fig_gender_age, width='stretch')

    # ---------------------------------------------------------
    # 4. Top 15 Companies by User Count (Horizontal Bar Chart)
    # ---------------------------------------------------------
    top_companies = df['company'].value_counts().head(15).reset_index()
    top_companies.columns = ['company', 'count'] # Rename columns for Plotly

    # For a horizontal bar chart, we assign 'company' to x and 'count' to y, 
    # then swap the axes in update_layout.
    fig_companies = px.bar(
        top_companies, 
        x='count', 
        y='company', 
        title='Top 15 Companies by User Count',
        orientation='h', # Specifies horizontal bar chart
        template='plotly_white'
    )
    
    # In Seaborn, horizontal bars are ordered bottom-to-top. 
    # Plotly orders top-to-bottom by default, so we reverse the y-axis.
    fig_companies.update_layout(
        xaxis_title='Number of Users', 
        yaxis_title='', 
        yaxis=dict(autorange="reversed")
    )
    fig_companies.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
    st.plotly_chart(fig_companies, width='stretch')
