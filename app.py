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
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'size': 16}
    )
    
    return fig

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time ML-powered churn prediction system</div>', unsafe_allow_html=True)
    
    # Sidebar - User Input
    st.sidebar.header("📝 Customer Information")
    st.sidebar.markdown("Fill in the customer details below to predict churn probability.")
    
    # Create input fields
    user_input = {}
    
    st.sidebar.subheader("Account Details")
    user_input['AccountAge'] = st.sidebar.number_input(
        "Account Age (months)", 
        min_value=0, 
        max_value=100, 
        value=12,
        help="How long the customer has been with the service"
    )
    
    user_input['MonthlyCharges'] = st.sidebar.number_input(
        "Monthly Charges ($)", 
        min_value=0.0, 
        max_value=200.0, 
        value=50.0,
        step=5.0
    )
    
    user_input['TotalCharges'] = st.sidebar.number_input(
        "Total Charges ($)", 
        min_value=0.0, 
        max_value=10000.0, 
        value=600.0,
        step=50.0
    )
    
    user_input['SubscriptionType'] = st.sidebar.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )
    
    user_input['PaymentMethod'] = st.sidebar.selectbox(
        "Payment Method",
        ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
    )
    
    user_input['PaperlessBilling'] = st.sidebar.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
    
    st.sidebar.subheader("Usage Patterns")
    
    user_input['ViewingHoursPerWeek'] = st.sidebar.slider(
        "Viewing Hours per Week",
        min_value=0,
        max_value=100,
        value=15,
        help="Average hours spent viewing content per week"
    )
    
    user_input['AverageViewingDuration'] = st.sidebar.slider(
        "Average Viewing Duration (minutes)",
        min_value=0,
        max_value=200,
        value=45
    )
    
    user_input['ContentDownloadsPerMonth'] = st.sidebar.number_input(
        "Content Downloads per Month",
        min_value=0,
        max_value=100,
        value=5
    )
    
    user_input['WatchlistSize'] = st.sidebar.number_input(
        "Watchlist Size",
        min_value=0,
        max_value=200,
        value=20
    )
    
    st.sidebar.subheader("Content Preferences")
    
    user_input['ContentType'] = st.sidebar.selectbox(
        "Content Type",
        ["Movies", "TV Shows", "Both", "Documentaries"]
    )
    
    user_input['GenrePreference'] = st.sidebar.selectbox(
        "Genre Preference",
        ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Documentary"]
    )
    
    user_input['UserRating'] = st.sidebar.slider(
        "User Rating (1-5)",
        min_value=1,
        max_value=5,
        value=4
    )
    
    st.sidebar.subheader("Device & Features")
    
    user_input['MultiDeviceAccess'] = st.sidebar.selectbox(
        "Multi-Device Access",
        ["Yes", "No"]
    )
    
    user_input['DeviceRegistered'] = st.sidebar.selectbox(
        "Device Registered",
        ["Mobile", "TV", "Computer", "Tablet"]
    )
    
    user_input['ParentalControl'] = st.sidebar.selectbox(
        "Parental Control",
        ["Yes", "No"]
    )
    
    user_input['SubtitlesEnabled'] = st.sidebar.selectbox(
        "Subtitles Enabled",
        ["Yes", "No"]
    )
    
    st.sidebar.subheader("Support & Demographics")
    
    user_input['SupportTicketsPerMonth'] = st.sidebar.number_input(
        "Support Tickets per Month",
        min_value=0,
        max_value=20,
        value=1
    )
    
    user_input['Gender'] = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "📈 Insights"])
    
    with tab1:
        if predict_button:
            with st.spinner("Analyzing customer data..."):
                # Make prediction
                prediction, probability = predict_churn(user_input)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Prediction Result")
                    
                    if prediction == "Yes":
                        st.markdown(
                            f'<div class="prediction-box churn-yes">⚠️ HIGH RISK - Customer Likely to Churn</div>',
                            unsafe_allow_html=True
                        )
                        st.error("🚨 This customer shows signs of potential churn. Consider retention strategies!")
                    else:
                        st.markdown(
                            f'<div class="prediction-box churn-no">✅ LOW RISK - Customer Likely to Stay</div>',
                            unsafe_allow_html=True
                        )
                        st.success("😊 This customer appears satisfied and likely to continue the service!")
                    
                    # Confidence metrics
                    st.markdown("### Confidence Metrics")
                    churn_prob = probability[1]
                    no_churn_prob = probability[0]
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
                    with col_b:
                        st.metric("Retention Probability", f"{no_churn_prob*100:.1f}%")
                
                with col2:
                    st.markdown("### Probability Gauge")
                    gauge = create_gauge_chart(churn_prob)
                    st.plotly_chart(gauge, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Recommendations")
                
                if prediction == "Yes":
                    st.warning("**Recommended Actions:**")
                    recommendations = [
                        "🎁 Offer a personalized discount or promotional deal",
                        "📞 Reach out proactively to understand concerns",
                        "⭐ Provide premium content or exclusive features trial",
                        "🤝 Assign a dedicated customer success manager",
                        "📊 Analyze viewing patterns for personalized content suggestions"
                    ]
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.info("**Engagement Opportunities:**")
                    opportunities = [
                        "🎯 Encourage referrals with a reward program",
                        "📢 Share new features and upcoming content",
                        "⭐ Request feedback and testimonials",
                        "🎊 Celebrate milestones (anniversaries, viewing hours)",
                        "💬 Maintain regular communication through newsletters"
                    ]
                    for opp in opportunities:
                        st.markdown(f"- {opp}")
                
                # Feature impact
                st.markdown("---")
                st.markdown("### 📊 Key Factors Influencing Prediction")
                
                # Calculate some key metrics
                engagement_score = (user_input['ViewingHoursPerWeek'] / 40) * 100
                value_score = (user_input['TotalCharges'] / 5000) * 100
                support_score = max(0, 100 - (user_input['SupportTicketsPerMonth'] * 20))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Engagement Score",
                        f"{min(engagement_score, 100):.0f}/100",
                        delta="High" if engagement_score > 50 else "Low"
                    )
                
                with col2:
                    st.metric(
                        "Customer Value",
                        f"{min(value_score, 100):.0f}/100",
                        delta="High" if value_score > 50 else "Low"
                    )
                
                with col3:
                    st.metric(
                        "Satisfaction Score",
                        f"{support_score:.0f}/100",
                        delta="High" if support_score > 70 else "Low"
                    )
        
        else:
            st.info("👈 Fill in the customer details in the sidebar and click 'Predict Churn' to get started!")
            
            # Show example
            st.markdown("### 📋 Example Use Case")
            st.markdown("""
            **Scenario:** A streaming service wants to identify customers at risk of canceling their subscription.
            
            **How to use:**
            1. Enter customer details in the sidebar (account age, charges, viewing habits, etc.)
            2. Click the 'Predict Churn' button
            3. Review the prediction and probability
            4. Take action based on recommendations
            
            **Benefits:**
            - 🎯 Proactive customer retention
            - 💰 Reduced churn rate
            - 📈 Improved customer lifetime value
            - 🤝 Better customer relationships
            """)
    
    with tab2:
        st.markdown("### 🏆 Model Performance Metrics")
        
        # Display model metadata
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = metadata['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        
        st.markdown("---")
        
        # Load and display visualizations if available
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('model_comparison.png'):
                st.markdown("### Model Comparison")
                st.image('model_comparison.png', use_container_width=True)
        
        with col2:
            if os.path.exists('correlation_heatmap.png'):
                st.markdown("### Feature Correlations")
                st.image('correlation_heatmap.png', use_container_width=True)
        
        st.markdown("---")
        
        # Model information
        st.markdown("### 🤖 Model Information")
        st.info(f"**Selected Model:** {metadata['model_name']}")
        
        st.markdown("""
        **Model Training Details:**
        - Algorithm: Ensemble-based machine learning
        - Training Data: Historical customer data with churn labels
        - Features: 19 original features + 8 engineered features
        - Validation: 5-fold cross-validation
        - Optimization: Grid search for hyperparameter tuning
        """)
    
    with tab3:
        st.markdown("### 📈 Feature Importance Analysis")
        
        if os.path.exists('feature_importance.png'):
            st.image('feature_importance.png', use_container_width=True)
            
            st.markdown("---")
            
            # Load feature importance data
            if os.path.exists('feature_importance.csv'):
                feat_imp = pd.read_csv('feature_importance.csv')
                
                st.markdown("### 📊 Top Features Impact on Churn")
                
                # Create interactive bar chart
                fig = px.bar(
                    feat_imp.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature description
                st.markdown("### 📖 Feature Descriptions")
                st.markdown("""
                - **EngagementRatio**: Viewing hours relative to watchlist size
                - **ViewingIntensity**: Total content consumption (hours × duration)
                - **CostPerHour**: Monthly charges divided by viewing hours
                - **LifetimeValue**: Total charges normalized by account age
                - **SupportIntensity**: Support tickets accumulated over account lifetime
                - **MonthlyCharges**: Current monthly subscription cost
                - **SupportTicketsPerMonth**: Number of support requests per month
                - **AccountAge**: Duration of customer relationship
                """)
        else:
            st.warning("Feature importance visualization not available. Please run the training script first.")
        
        st.markdown("---")
        
        # Display other visualizations
        st.markdown("### 📊 Dataset Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('churn_distribution.png'):
                st.markdown("#### Churn Distribution")
                st.image('churn_distribution.png', use_container_width=True)
        
        with col2:
            if os.path.exists('churn_correlations.png'):
                st.markdown("#### Feature-Churn Correlations")
                st.image('churn_correlations.png', use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Customer Churn Prediction System | Built with Streamlit & Machine Learning</p>
        <p>Model trained on historical customer data | Predictions are probabilistic estimates</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    main()
