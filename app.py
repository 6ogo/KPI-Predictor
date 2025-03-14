import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import re
import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import base64

# Custom modules (from previous artifacts)
from feature_engineering import enhanced_feature_engineering
from model_training import train_and_evaluate_models
from subject_recommendation import build_subject_recommendation_model, recommend_subject
from visualizations import create_visualizations

# Set page config
st.set_page_config(
    page_title="Email Campaign KPI Predictor",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Load and preprocess the campaign data"""
    try:
        customer_df = pd.read_csv('customer_data.csv')
        delivery_df = pd.read_csv('delivery_data.csv')
        
        # Basic preprocessing
        customer_df = customer_df.drop_duplicates(subset=['InternalName', 'Primary k isWoman OptOut'])
        delivery_df['open_rate'] = (delivery_df['Opens'] / delivery_df['Utskick']) * 100
        
        return customer_df, delivery_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- Model Training and Caching ---
@st.cache_resource
def build_models(customer_df, delivery_df):
    """Build and cache the prediction and recommendation models"""
    try:
        # Enhanced feature engineering
        processed_data = enhanced_feature_engineering(delivery_df, customer_df)
        
        # Define features and target for the open rate model
        features = processed_data.drop(['open_rate', 'InternalName', 'subject', 'contact_date'], axis=1, errors='ignore')
        target = processed_data['open_rate']
        
        # Train model and evaluate
        best_model, model_results = train_and_evaluate_models(features, target)
        
        # Build subject recommendation model
        subject_recommendations, subject_patterns = build_subject_recommendation_model(processed_data)
        
        # Get list of unique values for categorical features (for UI dropdowns)
        categorical_values = {}
        for col in ['county', 'dialog', 'syfte', 'product', 'bolag']:
            if col in processed_data.columns:
                categorical_values[col] = processed_data[col].unique().tolist()
        
        return {
            'best_model': best_model,
            'model_results': model_results,
            'subject_recommendations': subject_recommendations,
            'subject_patterns': subject_patterns,
            'categorical_values': categorical_values,
            'feature_names': features.columns.tolist()
        }
    except Exception as e:
        st.error(f"Error building models: {e}")
        return None

# Function to extract subject features (simplified version for the UI)
def extract_subject_features(subject):
    """Extract features from a subject line for prediction"""
    features = {}
    features['length'] = len(subject)
    features['has_personalization'] = 1 if re.search(r'\b(your|you|du|din|ditt|dina)\b', subject.lower()) else 0
    features['has_question'] = 1 if '?' in subject else 0
    features['has_numbers'] = 1 if re.search(r'\d', subject) else 0
    features['has_uppercase_words'] = 1 if re.search(r'\b[A-Z]{2,}\b', subject) else 0
    features['word_count'] = len(subject.split())
    return features

# --- Main App ---
def main():
    # Header & Intro
    st.title("📧 Email Campaign KPI Predictor")
    st.write("""
    This tool uses machine learning to predict open rates for your email campaigns and provides 
    recommendations for targeting and subject lines to improve performance.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        customer_df, delivery_df = load_data()
    
    if customer_df is None or delivery_df is None:
        st.error("Failed to load data. Please check file paths and formats.")
        return
    
    # Load or build models
    with st.spinner("Preparing models..."):
        models = build_models(customer_df, delivery_df)
    
    if models is None:
        st.error("Failed to build models. Please check the data and try again.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Campaign Predictor", "Performance Insights", "Data Export"])
    
    # Tab 1: Campaign Predictor
    with tab1:
        st.header("Campaign Parameter Input")
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic campaign parameters
            st.subheader("Campaign Settings")
            
            # Get values from models for dropdowns
            cat_values = models['categorical_values']
            
            # Input fields - use available values from the data
            selected_county = st.selectbox(
                "Target County", 
                options=cat_values.get('county', ["Stockholm", "Göteborg och Bohuslän", "Skåne"])
            )
            
            selected_dialog = st.selectbox(
                "Dialog", 
                options=cat_values.get('dialog', ["Welcome", "Monthly", "Promo"])
            )
            
            selected_syfte = st.selectbox(
                "Campaign Purpose", 
                options=cat_values.get('syfte', ["Information", "Sales", "Service"])
            )
            
            selected_product = st.selectbox(
                "Product", 
                options=cat_values.get('product', ["Product A", "Product B", "Service X"])
            )
            
            selected_bolag = st.selectbox(
                "Company", 
                options=cat_values.get('bolag', ["Main Company", "Subsidiary A", "Subsidiary B"])
            )
                
        with col2:
            # More campaign parameters
            st.subheader("Audience & Content")
            
            # Demographics
            avg_age = st.slider("Average Recipient Age", 18, 80, 35)
            pct_women = st.slider("Percentage Women (%)", 0, 100, 50)
            
            # Send time
            send_date = st.date_input("Send Date", datetime.date.today())
            send_time = st.time_input("Send Time", datetime.time(9, 0))
            
            # Convert to day of week and hour
            day_of_week = send_date.weekday()
            hour_of_day = send_time.hour
            is_weekend = 1 if day_of_week >= 5 else 0  # 5=Sat, 6=Sun
            
            # Subject line
            subject = st.text_input("Subject Line", "Check out our latest offers!")
            
            # Extract subject features
            subject_features = extract_subject_features(subject)
        
        # Create input data for prediction
        input_data = pd.DataFrame({
            'county': [selected_county],
            'dialog': [selected_dialog],
            'syfte': [selected_syfte],
            'product': [selected_product],
            'bolag': [selected_bolag],
            'avg_age': [avg_age],
            'pct_women': [pct_women],
            'day_of_week': [day_of_week],
            'hour_of_day': [hour_of_day],
            'is_weekend': [is_weekend],
            'length': [subject_features['length']],
            'has_personalization': [subject_features['has_personalization']],
            'has_question': [subject_features['has_question']],
            'has_numbers': [subject_features['has_numbers']],
            'has_uppercase_words': [subject_features['has_uppercase_words']],
            'word_count': [subject_features['word_count']]
        })
        
        # Add any missing columns expected by the model with default values
        for feature in models['feature_names']:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Only keep columns that the model expects
        input_data = input_data[models['feature_names']]
        
        # Make prediction with current input
        best_model = models['best_model']
        current_prediction = best_model.predict(input_data)[0]
        
        # Generate recommendations
        
        # 1. Targeting recommendation
        # Find the best performing county
        best_county = delivery_df.groupby('county')['open_rate'].mean().idxmax() if 'county' in delivery_df.columns else "Stockholm"
        
        # Create input with recommended targeting
        targeting_data = input_data.copy()
        targeting_data['county'] = best_county
        targeting_prediction = best_model.predict(targeting_data)[0]
        
        # 2. Subject line recommendation
        # Get recommendation
        recommended_subject = recommend_subject(subject, delivery_df, 
                                                models['subject_recommendations'], 
                                                models['subject_patterns'])
        
        # Extract features from recommended subject
        recommended_subject_features = extract_subject_features(recommended_subject)
        
        # Create input with recommended subject
        subject_data = input_data.copy()
        for feature, value in recommended_subject_features.items():
            if feature in subject_data.columns:
                subject_data[feature] = value
        
        subject_prediction = best_model.predict(subject_data)[0]
        
        # 3. Combined recommendation
        combined_data = targeting_data.copy()
        for feature, value in recommended_subject_features.items():
            if feature in combined_data.columns:
                combined_data[feature] = value
        
        combined_prediction = best_model.predict(combined_data)[0]
        
        # Show predictions
        st.header("Predictions & Recommendations")
        
        # Create visualizations
        figures = create_visualizations(
            current_prediction, 
            targeting_prediction, 
            subject_prediction, 
            combined_prediction, 
            best_model, 
            input_data
        )
        
        # Display visualizations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(figures['comparison'], use_container_width=True)
            
            st.subheader("Current Campaign")
            st.info(f"**Predicted Open Rate:** {current_prediction:.2f}%")
            
            st.subheader("Subject Line Recommendation")
            st.success(f"**Recommended Subject:** '{recommended_subject}'")
            st.info(f"**Predicted Open Rate:** {subject_prediction:.2f}%")
        
        with col2:
            st.plotly_chart(figures['gauge'], use_container_width=True)
            
            st.subheader("Targeting Recommendation")
            st.success(f"**Recommended County:** {best_county}")
            st.info(f"**Predicted Open Rate:** {targeting_prediction:.2f}%")
            
            st.subheader("Combined Recommendation")
            st.success(f"**Targeting:** {best_county} with Subject: '{recommended_subject}'")
            st.info(f"**Predicted Open Rate:** {combined_prediction:.2f}%")
        
        # Additional charts
        st.header("Additional Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(figures['radar'], use_container_width=True)
        
        with col2:
            if figures['features'] is not None:
                st.plotly_chart(figures['features'], use_container_width=True)
    
    # Tab 2: Performance Insights
    with tab2:
        st.header("Campaign Performance Insights")
        
        # Create historical performance charts
        if 'contact_date' in delivery_df.columns:
            delivery_df['contact_date'] = pd.to_datetime(delivery_df['contact_date'])
            delivery_df['month'] = delivery_df['contact_date'].dt.strftime('%Y-%m')
            
            # Monthly performance
            monthly_performance = delivery_df.groupby('month').agg(
                avg_open_rate=('open_rate', 'mean'),
                total_sends=('Utskick', 'sum'),
                total_opens=('Opens', 'sum')
            ).reset_index()
            
            # Plot monthly open rates
            fig_monthly = px.line(
                monthly_performance, 
                x='month', 
                y='avg_open_rate',
                title='Monthly Average Open Rate Trend',
                labels={'avg_open_rate': 'Open Rate (%)', 'month': 'Month'},
                markers=True
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Performance by day of week
            if 'day_of_week' in delivery_df.columns:
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                delivery_df['day_name'] = delivery_df['day_of_week'].apply(lambda x: day_names[x])
                
                day_performance = delivery_df.groupby('day_name').agg(
                    avg_open_rate=('open_rate', 'mean'),
                    count=('InternalName', 'count')
                ).reset_index()
                
                # Sort by days of week
                day_performance['day_order'] = day_performance['day_name'].apply(lambda x: day_names.index(x))
                day_performance = day_performance.sort_values('day_order')
                
                # Plot by day of week
                fig_days = px.bar(
                    day_performance, 
                    x='day_name', 
                    y='avg_open_rate',
                    text=day_performance['avg_open_rate'].round(1).astype(str) + '%',
                    title='Open Rate by Day of Week',
                    labels={'avg_open_rate': 'Open Rate (%)', 'day_name': 'Day'},
                    color='avg_open_rate'
                )
                fig_days.update_traces(textposition='auto')
                
                st.plotly_chart(fig_days, use_container_width=True)
                
                # Performance by county
                if 'county' in delivery_df.columns:
                    county_performance = delivery_df.groupby('county').agg(
                        avg_open_rate=('open_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_open_rate', ascending=False)
                    
                    # Plot by county
                    fig_county = px.bar(
                        county_performance, 
                        x='county', 
                        y='avg_open_rate',
                        text=county_performance['avg_open_rate'].round(1).astype(str) + '%',
                        title='Open Rate by County',
                        labels={'avg_open_rate': 'Open Rate (%)', 'county': 'County'},
                        color='avg_open_rate'
                    )
                    fig_county.update_traces(textposition='auto')
                    fig_county.update_layout(xaxis={'categoryorder': 'total descending'})
                    
                    st.plotly_chart(fig_county, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Model Performance")
        st.write(f"The prediction model has been trained and evaluated using cross-validation.")
        
        if models['model_results']:
            for model_name, results in models['model_results'].items():
                st.write(f"**{model_name}**: MAE = {results['mae']:.2f}%")
    
    # Tab 3: Data Export
    with tab3:
        st.header("Export Predictions & Recommendations")
        
        # Create a dataframe with all predictions
        export_data = pd.DataFrame({
            'Scenario': ['Current Campaign', 'Recommended Targeting', 'Recommended Subject', 'Combined Recommendation'],
            'Open Rate (%)': [current_prediction, targeting_prediction, subject_prediction, combined_prediction],
            'Improvement (%)': [0, targeting_prediction - current_prediction, subject_prediction - current_prediction, combined_prediction - current_prediction],
            'County': [selected_county, best_county, selected_county, best_county],
            'Subject': [subject, subject, recommended_subject, recommended_subject]
        })
        
        st.dataframe(export_data)
        
        # CSV download button
        csv = export_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="campaign_predictions.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Create a report
        st.subheader("Campaign Report")
        report = f"""
        # Email Campaign Prediction Report
        **Date:** {datetime.date.today().strftime('%Y-%m-%d')}
        
        ## Campaign Parameters
        - **County:** {selected_county}
        - **Dialog:** {selected_dialog}
        - **Purpose:** {selected_syfte}
        - **Product:** {selected_product} 
        - **Company:** {selected_bolag}
        - **Average Age:** {avg_age}
        - **Percentage Women:** {pct_women}%
        - **Send Date/Time:** {send_date.strftime('%Y-%m-%d')} at {send_time.strftime('%H:%M')}
        - **Subject Line:** "{subject}"
        
        ## Predictions & Recommendations
        - **Current Campaign:** {current_prediction:.2f}% open rate
        - **Recommended County:** {best_county} ({targeting_prediction:.2f}% open rate, +{targeting_prediction - current_prediction:.2f}%)
        - **Recommended Subject:** "{recommended_subject}" ({subject_prediction:.2f}% open rate, +{subject_prediction - current_prediction:.2f}%)
        - **Combined Recommendation:** {best_county} with "{recommended_subject}" ({combined_prediction:.2f}% open rate, +{combined_prediction - current_prediction:.2f}%)
        
        ## Potential Impact
        Implementing these recommendations could improve your open rate by {combined_prediction - current_prediction:.2f} percentage points, 
        which represents a {((combined_prediction - current_prediction) / current_prediction) * 100:.1f}% increase.
        """
        
        st.markdown(report)
        
        # Download report button
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="campaign_report.md">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Model export option
        st.subheader("Export Model")
        if st.button("Export Prediction Model"):
            try:
                # Save model to pickle file
                with open('email_campaign_model.pkl', 'wb') as f:
                    pickle.dump(best_model, f)
                st.success("Model exported successfully! You can download it from your app directory.")
            except Exception as e:
                st.error(f"Error exporting model: {e}")

if __name__ == "__main__":
    main()