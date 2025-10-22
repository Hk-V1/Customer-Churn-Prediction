"""
app.py
======
Customer Churn Prediction - Streamlit Application

This interactive app allows users to:
1. Input customer details via sidebar
2. Get real-time churn predictions
3. View prediction probability and confidence
4. Explore model performance metrics
5. Visualize feature importances

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load('best_model.pkl')
        preprocessing_objects = joblib.load('preprocessing_objects.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, preprocessing_objects, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `train_model.py` first.")
        st.stop()

model, preprocessing_objects, metadata = load_model_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_input_features(user_input):
    """Create engineered features from user input"""
    df = pd.DataFrame([user_input])
    
    # Create engineered features (same as in training)
    df['EngagementRatio'] = df['ViewingHoursPerWeek'] / (df['WatchlistSize'] + 1)
    df['ViewingIntensity'] = df['ViewingHoursPerWeek'] * df['AverageViewingDuration']
    df['CostPerHour'] = df['MonthlyCharges'] / (df['ViewingHoursPerWeek'] + 1)
    df['LifetimeValue'] = df['TotalCharges'] / (df['AccountAge'] + 1)
    df['DownloadRatio'] = df['ContentDownloadsPerMonth'] / (df['ViewingHoursPerWeek'] + 1)
    df['SupportIntensity'] = df['SupportTicketsPerMonth'] * df['AccountAge']
    df['IsHighValue'] = (df['TotalCharges'] > 1000).astype(int)  # Threshold based on typical median
    df['IsHeavyUser'] = (df['ViewingHoursPerWeek'] > 20).astype(int)  # Threshold based on typical 75th percentile
    
    return df

def preprocess_input(df):
    """Preprocess user input using saved preprocessing objects"""
    cat_cols = preprocessing_objects['cat_cols']
    num_cols = preprocessing_objects['num_cols']
    label_encoders = preprocessing_objects['label_encoders']
    scaler = preprocessing_objects['scaler']
    
    # Encode categorical variables
    for col in cat_cols:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Ensure correct feature order
    df = df[preprocessing_objects['feature_names']]
    
    return df

def predict_churn(user_input):
    """Make churn prediction"""
    # Create features
    df = create_input_features(user_input)
    
    # Preprocess
    df_processed = preprocess_input(df)
    
    # Predict
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0]
    
    # Decode prediction
    target_encoder = preprocessing_objects['target_encoder']
    prediction_label = target_encoder.inverse_transform([prediction])[0]
    
    return prediction_label, probability

def create_gauge_chart(probability):
    """Create gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100],
