import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64
import os

# Custom modules
from feature_engineering import enhanced_feature_engineering, extract_subject_features
from multi_metric_model import train_multi_metric_models, predict_metrics
from recommendations import generate_recommendations, format_predictions
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
        
        # Calculate rates if they don't exist yet
        if 'open_rate' not in delivery_df.columns:
            delivery_df['open_rate'] = (delivery_df['Opens'] / delivery_df['Utskick']) * 100
        
        if 'click_rate' not in delivery_df.columns and 'Opens' in delivery_df.columns:
            delivery_df['click_rate'] = (delivery_df['Clicks'] / delivery_df['Opens']) * 100
            
        if 'optout_rate' not in delivery_df.columns and 'Opens' in delivery_df.columns:
            delivery_df['optout_rate'] = (delivery_df['Optout'] / delivery_df['Opens']) * 100
        
        # Handle infinities and NaNs
        delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        delivery_df.fillna({
            'click_rate': 0,
            'optout_rate': 0
        }, inplace=True)
        
        return customer_df, delivery_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- Model Training and Caching ---
@st.cache_resource
def build_models(customer_df, delivery_df):
    """Build and cache the prediction and recommendation models"""
    try:
        # Train multi-metric models
        model_results = train_multi_metric_models(delivery_df, customer_df)
        
        # Build subject recommendation model
        subject_recommendations, subject_patterns = build_subject_recommendation_model(delivery_df)
        
        # Add subject recommendations to model results
        model_results['subject_recommendations'] = subject_recommendations
        model_results['subject_patterns'] = subject_patterns
        
        return model_results
    except Exception as e:
        st.error(f"Error building models: {e}")
        return None

# --- Main App ---
def main():
    # Header & Intro
    st.title("📧 Email Campaign KPI Predictor")
    st.write("""
    This tool uses machine learning to predict email campaign performance and provides 
    recommendations for targeting and subject lines to improve your KPIs.
    
    - **Subject Line Recommendations**: Optimize for open rates only
    - **Targeting Recommendations**: Optimize for open, click, and optout rates
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        customer_df, delivery_df = load_data()
    
    if customer_df is None or delivery_df is None:
        st.error("Failed to load data. Please check file paths and formats.")
        return
    
    # Load or build models
    with st.spinner("Preparing models..."):
        model_results = build_models(customer_df, delivery_df)
    
    if model_results is None:
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
            cat_values = model_results['categorical_values']
            
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
            'subject': [subject]  # Add the actual subject for reference
        })
        
        # Add subject features
        for feature, value in subject_features.items():
            input_data[feature] = value
        
        # Add any missing columns expected by the model with default values
        for feature in model_results['feature_names']:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Only keep columns that the model expects
        model_features = input_data[model_results['feature_names']]
        
        # Generate recommendations
        recommendations = generate_recommendations(
            model_features,
            model_results['models'],
            delivery_df,
            subject_patterns=model_results['subject_patterns']
        )
        
        # Format predictions for display
        formatted_predictions = format_predictions(recommendations)
        
        # Show predictions
        st.header("Predictions & Recommendations")
        
        # Create visualizations
        figures = create_visualizations(formatted_predictions)
        
        # Display visualizations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(figures['open_rate'], use_container_width=True)
            
            st.subheader("Current Campaign")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Open Rate", f"{formatted_predictions['current']['open_rate']:.2f}%")
            with metric_col2:
                st.metric("Click Rate", f"{formatted_predictions['current']['click_rate']:.2f}%")
            with metric_col3:
                st.metric("Optout Rate", f"{formatted_predictions['current']['optout_rate']:.2f}%")
            
            st.subheader("Subject Line Recommendation")
            st.success(f"**Recommended Subject:** '{formatted_predictions['subject']['text']}'")
            st.info(f"**Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% (Change: {formatted_predictions['subject']['open_rate_diff']:.2f}%)")
            st.caption("Note: Subject line optimization only affects open rate")
        
        with col2:
            st.plotly_chart(figures['subject_impact'], use_container_width=True)
            
            st.subheader("Targeting Recommendation")
            st.success(f"**Recommended County:** {formatted_predictions['targeting']['county']}")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Open Rate", 
                          f"{formatted_predictions['targeting']['open_rate']:.2f}%", 
                          f"{formatted_predictions['targeting']['open_rate_diff']:.2f}%")
            with metric_col2:
                st.metric("Click Rate", 
                          f"{formatted_predictions['targeting']['click_rate']:.2f}%", 
                          f"{formatted_predictions['targeting']['click_rate_diff']:.2f}%")
            with metric_col3:
                st.metric("Optout Rate", 
                          f"{formatted_predictions['targeting']['optout_rate']:.2f}%", 
                          f"{formatted_predictions['targeting']['optout_rate_diff']:.2f}%")
            
            st.subheader("Combined Recommendation")
            st.success(f"**Targeting:** {formatted_predictions['combined']['county']} with Subject: '{formatted_predictions['combined']['subject']}'")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Open Rate", 
                          f"{formatted_predictions['combined']['open_rate']:.2f}%", 
                          f"{formatted_predictions['combined']['open_rate_diff']:.2f}%")
            with metric_col2:
                st.metric("Click Rate", 
                          f"{formatted_predictions['combined']['click_rate']:.2f}%", 
                          f"{formatted_predictions['combined']['click_rate_diff']:.2f}%")
            with metric_col3:
                st.metric("Optout Rate", 
                          f"{formatted_predictions['combined']['optout_rate']:.2f}%", 
                          f"{formatted_predictions['combined']['optout_rate_diff']:.2f}%")
        
        # Additional charts
        st.header("Additional Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(figures['targeting_metrics'], use_container_width=True)
        
        with col2:
            st.plotly_chart(figures['radar'], use_container_width=True)
        
        # Detailed comparison table
        st.plotly_chart(figures['table'], use_container_width=True)
    
    # Tab 2: Performance Insights
    with tab2:
        st.header("Campaign Performance Insights")
        
        # Create historical performance charts
        if 'contact_date' in delivery_df.columns:
            delivery_df['contact_date'] = pd.to_datetime(delivery_df['contact_date'])
            delivery_df['month'] = delivery_df['contact_date'].dt.strftime('%Y-%m')
            
            # Create tabs for different metrics and analyses
            metric_tab1, metric_tab2, metric_tab3, bolag_tab = st.tabs(["Open Rate", "Click Rate", "Optout Rate", "Company Analysis"])
            
            with metric_tab1:
                # Monthly open rate performance
                monthly_opens = delivery_df.groupby('month').agg(
                    avg_open_rate=('open_rate', 'mean'),
                    total_sends=('Utskick', 'sum'),
                    total_opens=('Opens', 'sum')
                ).reset_index()
                
                # Plot monthly open rates
                fig_monthly_opens = px.line(
                    monthly_opens, 
                    x='month', 
                    y='avg_open_rate',
                    title='Monthly Average Open Rate Trend',
                    labels={'avg_open_rate': 'Open Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_opens, use_container_width=True)
                
                # Open rate by county
                if 'county' in delivery_df.columns:
                    county_opens = delivery_df.groupby('county').agg(
                        avg_open_rate=('open_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_open_rate', ascending=False)
                    
                    # Plot by county
                    fig_county_opens = px.bar(
                        county_opens, 
                        x='county', 
                        y='avg_open_rate',
                        text=county_opens['avg_open_rate'].round(1).astype(str) + '%',
                        title='Open Rate by County',
                        labels={'avg_open_rate': 'Open Rate (%)', 'county': 'County'},
                        color='avg_open_rate'
                    )
                    fig_county_opens.update_traces(textposition='auto')
                    fig_county_opens.update_layout(xaxis={'categoryorder': 'total descending'})
                    
                    st.plotly_chart(fig_county_opens, use_container_width=True)
            
            with metric_tab2:
                # Monthly click rate performance
                monthly_clicks = delivery_df.groupby('month').agg(
                    avg_click_rate=('click_rate', 'mean'),
                    total_opens=('Opens', 'sum'),
                    total_clicks=('Clicks', 'sum')
                ).reset_index()
                
                # Plot monthly click rates
                fig_monthly_clicks = px.line(
                    monthly_clicks, 
                    x='month', 
                    y='avg_click_rate',
                    title='Monthly Average Click Rate Trend',
                    labels={'avg_click_rate': 'Click Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_clicks, use_container_width=True)
                
                # Click rate by county
                if 'county' in delivery_df.columns:
                    county_clicks = delivery_df.groupby('county').agg(
                        avg_click_rate=('click_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_click_rate', ascending=False)
                    
                    # Plot by county
                    fig_county_clicks = px.bar(
                        county_clicks, 
                        x='county', 
                        y='avg_click_rate',
                        text=county_clicks['avg_click_rate'].round(1).astype(str) + '%',
                        title='Click Rate by County',
                        labels={'avg_click_rate': 'Click Rate (%)', 'county': 'County'},
                        color='avg_click_rate'
                    )
                    fig_county_clicks.update_traces(textposition='auto')
                    fig_county_clicks.update_layout(xaxis={'categoryorder': 'total descending'})
                    
                    st.plotly_chart(fig_county_clicks, use_container_width=True)
            
            with metric_tab3:
                # Monthly optout rate performance
                monthly_optouts = delivery_df.groupby('month').agg(
                    avg_optout_rate=('optout_rate', 'mean'),
                    total_opens=('Opens', 'sum'),
                    total_optouts=('Optout', 'sum')
                ).reset_index()
                
                # Plot monthly optout rates
                fig_monthly_optouts = px.line(
                    monthly_optouts, 
                    x='month', 
                    y='avg_optout_rate',
                    title='Monthly Average Optout Rate Trend',
                    labels={'avg_optout_rate': 'Optout Rate (%)', 'month': 'Month'},
                    markers=True
                )
                st.plotly_chart(fig_monthly_optouts, use_container_width=True)
                
                # Optout rate by county
                if 'county' in delivery_df.columns:
                    county_optouts = delivery_df.groupby('county').agg(
                        avg_optout_rate=('optout_rate', 'mean'),
                        count=('InternalName', 'count')
                    ).reset_index().sort_values('avg_optout_rate', ascending=True)  # Lower is better
                    
                    # Plot by county
                    fig_county_optouts = px.bar(
                        county_optouts, 
                        x='county', 
                        y='avg_optout_rate',
                        text=county_optouts['avg_optout_rate'].round(1).astype(str) + '%',
                        title='Optout Rate by County',
                        labels={'avg_optout_rate': 'Optout Rate (%)', 'county': 'County'},
                        color='avg_optout_rate',
                        color_continuous_scale='Reds_r'  # Reversed scale (red is bad)
                    )
                    fig_county_optouts.update_traces(textposition='auto')
                    fig_county_optouts.update_layout(xaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(fig_county_optouts, use_container_width=True)
            
            # Company (Bolag) analysis tab
            with bolag_tab:
                st.subheader("Performance Analysis by Company (Bolag)")
                st.write("""
                This analysis shows how customers from different companies (Bolag) engage with emails.
                While campaigns are sent globally, this can help identify if certain company segments 
                have different engagement patterns.
                """)
                
                # Import and run Bolag analysis
                from bolag_analysis import create_bolag_analysis
                
                # Check if customer data has Bolag information
                if 'Bolag' in customer_df.columns:
                    with st.spinner("Analyzing company performance..."):
                        bolag_figures, bolag_performance = create_bolag_analysis(delivery_df, customer_df)
                    
                    # Create sub-tabs for different metrics
                    bolag_subtab1, bolag_subtab2, bolag_subtab3, bolag_subtab4 = st.tabs([
                        "Open Rate by Company", "Click Rate by Company", 
                        "Optout Rate by Company", "Metrics Comparison"
                    ])
                    
                    with bolag_subtab1:
                        st.plotly_chart(bolag_figures['open_rate'], use_container_width=True)
                        
                    with bolag_subtab2:
                        st.plotly_chart(bolag_figures['click_rate'], use_container_width=True)
                        
                    with bolag_subtab3:
                        st.plotly_chart(bolag_figures['optout_rate'], use_container_width=True)
                        
                    with bolag_subtab4:
                        st.plotly_chart(bolag_figures['comparison'], use_container_width=True)
                        
                    # Display data table
                    st.subheader("Company Performance Data")
                    st.dataframe(bolag_performance.sort_values('total_customers', ascending=False))
                    
                    # Download link for Bolag analysis
                    csv = bolag_performance.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="company_performance.csv">Download Company Analysis CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Bolag (company) information not found in customer data. Unable to perform company analysis.")
        
        # Model performance metrics
        st.subheader("Model Performance")
        st.write(f"The prediction models have been trained and evaluated using cross-validation.")
        
        if 'performance' in model_results:
            for metric, results in model_results['performance'].items():
                st.write(f"**{metric}**: MAE = {results['mae']:.2f}%")
    
    # Tab 3: Data Export
    with tab3:
        st.header("Export Predictions & Recommendations")
        
        # Create a dataframe with all predictions
        export_data = pd.DataFrame({
            'Metric': ['Open Rate (%)', 'Click Rate (%)', 'Optout Rate (%)'],
            'Current': [
                formatted_predictions['current']['open_rate'],
                formatted_predictions['current']['click_rate'],
                formatted_predictions['current']['optout_rate']
            ],
            'Recommended Targeting': [
                formatted_predictions['targeting']['open_rate'],
                formatted_predictions['targeting']['click_rate'],
                formatted_predictions['targeting']['optout_rate']
            ],
            'Targeting Improvement': [
                formatted_predictions['targeting']['open_rate_diff'],
                formatted_predictions['targeting']['click_rate_diff'],
                formatted_predictions['targeting']['optout_rate_diff']
            ],
            'Recommended Subject (Open Rate Only)': [
                formatted_predictions['subject']['open_rate'],
                "N/A",
                "N/A"
            ],
            'Subject Improvement': [
                formatted_predictions['subject']['open_rate_diff'],
                "N/A",
                "N/A"
            ],
            'Combined Recommendation': [
                formatted_predictions['combined']['open_rate'],
                formatted_predictions['combined']['click_rate'],
                formatted_predictions['combined']['optout_rate']
            ],
            'Combined Improvement': [
                formatted_predictions['combined']['open_rate_diff'],
                formatted_predictions['combined']['click_rate_diff'],
                formatted_predictions['combined']['optout_rate_diff']
            ]
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
        
        ## Current Campaign Predictions
        - **Open Rate:** {formatted_predictions['current']['open_rate']:.2f}%
        - **Click Rate:** {formatted_predictions['current']['click_rate']:.2f}%
        - **Optout Rate:** {formatted_predictions['current']['optout_rate']:.2f}%
        
        ## Subject Line Recommendation (Affects Open Rate Only)
        - **Recommended Subject:** "{formatted_predictions['subject']['text']}"
        - **Predicted Open Rate:** {formatted_predictions['subject']['open_rate']:.2f}% (Change: {formatted_predictions['subject']['open_rate_diff']:.2f}%)
        
        ## Targeting Recommendation (Affects All Metrics)
        - **Recommended County:** {formatted_predictions['targeting']['county']}
        - **Predicted Open Rate:** {formatted_predictions['targeting']['open_rate']:.2f}% (Change: {formatted_predictions['targeting']['open_rate_diff']:.2f}%)
        - **Predicted Click Rate:** {formatted_predictions['targeting']['click_rate']:.2f}% (Change: {formatted_predictions['targeting']['click_rate_diff']:.2f}%)
        - **Predicted Optout Rate:** {formatted_predictions['targeting']['optout_rate']:.2f}% (Change: {formatted_predictions['targeting']['optout_rate_diff']:.2f}%)
        
        ## Combined Recommendation
        - **Targeting:** {formatted_predictions['combined']['county']}
        - **Subject:** "{formatted_predictions['combined']['subject']}"
        - **Predicted Open Rate:** {formatted_predictions['combined']['open_rate']:.2f}% (Change: {formatted_predictions['combined']['open_rate_diff']:.2f}%)
        - **Predicted Click Rate:** {formatted_predictions['combined']['click_rate']:.2f}% (Change: {formatted_predictions['combined']['click_rate_diff']:.2f}%)
        - **Predicted Optout Rate:** {formatted_predictions['combined']['optout_rate']:.2f}% (Change: {formatted_predictions['combined']['optout_rate_diff']:.2f}%)
        
        ## Potential Impact
        Implementing the combined recommendations could improve your open rate by {formatted_predictions['combined']['open_rate_diff']:.2f} percentage points, 
        which represents a {((formatted_predictions['combined']['open_rate'] - formatted_predictions['current']['open_rate']) / formatted_predictions['current']['open_rate'] * 100) if formatted_predictions['current']['open_rate'] > 0 else 0:.1f}% increase.
        """
        
        st.markdown(report)
        
        # Download report button
        b64 = base64.b64encode(report.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="campaign_report.md">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()