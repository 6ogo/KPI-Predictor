import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import base64
import os
import joblib
import datetime


# Custom modules
from feature_engineering import enhanced_feature_engineering, extract_subject_features
from multi_metric_model import train_multi_metric_models, predict_metrics
from recommendations import generate_recommendations, format_predictions
from subject_recommendation import build_subject_recommendation_model, recommend_subject
from visualizations import create_visualizations
from model_metadata import track_model_performance, model_needs_retraining, get_model_version

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

# --- Model Training, Validation and Caching ---
@st.cache_resource
def build_models(customer_df, delivery_df):
    """Build, validate, save and reuse prediction and recommendation models"""
    import os
    import pickle
    import joblib
    
    models_dir = "saved_models"
    models_path = os.path.join(models_dir, "email_campaign_models.joblib")
    subject_model_path = os.path.join(models_dir, "subject_recommendation_model.joblib")
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if models already exist
    if os.path.exists(models_path) and os.path.exists(subject_model_path):
        try:
            # Load existing models
            st.info("Found existing models. Attempting to load...")
            model_results = joblib.load(models_path)
            subject_data = joblib.load(subject_model_path)
            
            # Add default version if not present (for backwards compatibility)
            if 'version' not in model_results:
                model_results['version'] = f"legacy-{datetime.datetime.now().strftime('%Y%m%d')}"
            
            # Validate loaded models
            validation_result = validate_models(model_results, delivery_df, customer_df)
            
            if validation_result['valid']:
                # Models are valid, add subject recommendations
                model_results['subject_recommendations'] = subject_data['subject_recommendations']
                model_results['subject_patterns'] = subject_data['subject_patterns']
                st.success("✅ Successfully loaded pre-trained models from disk.")
                
                # Track model performance using the imported function
                track_model_performance(model_results)
                
                return model_results
            else:
                # Models failed validation
                st.warning(f"⚠️ Loaded models didn't pass validation: {validation_result['message']}")
                st.info("Training new models...")
                
                # Display recommendations
                if 'recommendations' in validation_result:
                    st.subheader("Recommendations for Improving Model Performance")
                    for i, rec in enumerate(validation_result['recommendations']):
                        st.markdown(f"{i+1}. {rec}")
        except Exception as e:
            st.warning(f"⚠️ Error loading models: {e}")
            st.info("Training new models...")
    
    try:
        # Train multi-metric models
        # Use enhanced_train_multi_metric_models to add versioning and metadata
        from multi_metric_model import enhanced_train_multi_metric_models
        model_results = enhanced_train_multi_metric_models(delivery_df, customer_df)
        
        # Build subject recommendation model
        subject_recommendations, subject_patterns = build_subject_recommendation_model(delivery_df)
        
        # Add subject recommendations to model results
        model_results['subject_recommendations'] = subject_recommendations
        model_results['subject_patterns'] = subject_patterns
        
        # Track the initial performance of the newly trained model
        track_model_performance(model_results)
        
        # Save the models
        try:
            # Save multi-metric models
            joblib.dump(model_results, models_path)
            
            # Save subject model data separately
            subject_data = {
                'subject_recommendations': subject_recommendations,
                'subject_patterns': subject_patterns
            }
            joblib.dump(subject_data, subject_model_path)
            
            st.success("✅ Trained and saved models to disk for future use.")
        except Exception as e:
            st.warning(f"⚠️ Error saving models: {e}")
        
        return model_results
    except Exception as e:
        st.error(f"🚨 Error building models: {e}")
        return None

def validate_models(model_results, delivery_df, customer_df):
    """
    Validate loaded models to ensure they're still producing reasonable results
    
    Parameters:
    - model_results: The loaded model results
    - delivery_df: DataFrame with delivery data
    - customer_df: DataFrame with customer data
    
    Returns:
    - Dictionary with validation results
    """
    from multi_metric_model import predict_metrics
    import pandas as pd
    import numpy as np
    
    # Get expected metrics ranges from historical data
    expected_ranges = {
        'open_rate': {
            'min': max(0, delivery_df['open_rate'].mean() - 3 * delivery_df['open_rate'].std()),
            'max': min(100, delivery_df['open_rate'].mean() + 3 * delivery_df['open_rate'].std())
        },
        'click_rate': {
            'min': max(0, delivery_df['click_rate'].mean() - 3 * delivery_df['click_rate'].std()),
            'max': min(100, delivery_df['click_rate'].mean() + 3 * delivery_df['click_rate'].std())
        },
        'optout_rate': {
            'min': max(0, delivery_df['optout_rate'].mean() - 3 * delivery_df['optout_rate'].std()),
            'max': min(100, delivery_df['optout_rate'].mean() + 3 * delivery_df['optout_rate'].std())
        }
    }
    
    # Sample a small subset of delivery data for testing
    sample_size = min(5, len(delivery_df))
    sample_data = delivery_df.sample(sample_size)
    
    # Feature names expected by the model
    feature_names = model_results['feature_names']
    
    # Prepare sample data for prediction
    from feature_engineering import extract_subject_features
    
    try:
        # Create test inputs
        test_inputs = []
        for _, row in sample_data.iterrows():
            # Create basic input data
            input_data = pd.DataFrame({
                'county': [row.get('county', 'Stockholm')],
                'dialog': [row.get('dialog', 'Monthly')],
                'syfte': [row.get('syfte', 'Information')],
                'product': [row.get('product', 'Product A')],
                'bolag': [row.get('bolag', 'Main Company')],
                'avg_age': [row.get('avg_age', 35)],
                'pct_women': [row.get('pct_women', 50)],
                'day_of_week': [row.get('day_of_week', 0)],
                'hour_of_day': [row.get('hour_of_day', 9)],
                'is_weekend': [row.get('is_weekend', 0)],
                'subject': [row.get('subject', 'Newsletter')]
            })
            
            # Add subject features
            if 'subject' in input_data.columns:
                subject_features = extract_subject_features(input_data['subject'][0])
                for feature, value in subject_features.items():
                    input_data[feature] = value
            
            # Add any missing columns expected by the model with default values
            for feature in feature_names:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            
            # Only keep columns that the model expects
            model_input = input_data[feature_names]
            test_inputs.append(model_input)
        
        # Make predictions
        valid_predictions = True
        errors = []
        
        for test_input in test_inputs:
            predictions = predict_metrics(test_input, model_results['models'])
            
            # Check if predictions are within expected ranges
            for metric, value in predictions.items():
                if metric in expected_ranges:
                    if value < expected_ranges[metric]['min'] or value > expected_ranges[metric]['max']:
                        valid_predictions = False
                        errors.append(f"{metric} prediction {value:.2f} is outside expected range "
                                     f"({expected_ranges[metric]['min']:.2f}-{expected_ranges[metric]['max']:.2f})")
        
        if valid_predictions:
            return {'valid': True}
        else:
            recommendations = generate_model_improvement_recommendations(delivery_df, errors)
            return {
                'valid': False,
                'message': "Model predictions outside expected ranges",
                'errors': errors,
                'recommendations': recommendations
            }
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error validating models: {str(e)}"
        }


def generate_model_improvement_recommendations(delivery_df, errors):
    """
    Generate recommendations for improving model training and fit
    
    Parameters:
    - delivery_df: DataFrame with delivery data
    - errors: List of validation errors
    
    Returns:
    - List of recommendations
    """
    recommendations = []
    
    # Sample size check
    if len(delivery_df) < 50:
        recommendations.append("Your dataset is relatively small. Consider collecting more campaign data "
                              "to improve model performance.")
    
    # Check for imbalanced data
    for metric in ['open_rate', 'click_rate', 'optout_rate']:
        if metric in delivery_df.columns:
            metric_mean = delivery_df[metric].mean()
            metric_std = delivery_df[metric].std()
            
            if metric_std < 1:
                recommendations.append(f"Your {metric} data has low variability (std={metric_std:.2f}). "
                                      f"Consider including more diverse campaigns to improve predictions.")
            
            # Check for extreme skew
            skew = delivery_df[metric].skew()
            if abs(skew) > 2:
                recommendations.append(f"Your {metric} data is highly skewed (skew={skew:.2f}). "
                                      f"Consider transforming this metric or collecting more balanced data.")
    
    # Check for missing or inaccurate features
    if 'subject' in delivery_df.columns:
        missing_subjects = delivery_df['subject'].isna().sum()
        if missing_subjects > 0.1 * len(delivery_df):
            recommendations.append(f"Found {missing_subjects} missing subject lines. "
                                  f"This may affect subject line feature extraction and recommendations.")
    
    # Look at specific errors
    for error in errors:
        if 'open_rate' in error.lower():
            recommendations.append("Open rate prediction issues may be improved by better subject line features "
                                  "or adding sending time features.")
        if 'click_rate' in error.lower():
            recommendations.append("Click rate prediction issues may be improved by adding content relevance "
                                  "features or customer engagement history.")
        if 'optout_rate' in error.lower():
            recommendations.append("Optout rate prediction issues may be improved by adding frequency "
                                  "features or better segmentation features.")
    
    # Hyperparameter recommendations
    recommendations.append("Consider tuning model hyperparameters using cross-validation "
                          "to find the best configuration for your specific data.")
    
    # Feature engineering recommendations
    recommendations.append("Enhance feature engineering by extracting more semantic features "
                         "from subject lines or adding customer engagement history.")
    
    return recommendations

def track_prediction_performance(formatted_predictions, actual_metrics=None):
    """
    Track prediction performance for a campaign
    
    Parameters:
    - formatted_predictions: The predictions made for the campaign
    - actual_metrics: Optional actual metrics if available
    """
    import os
    import pandas as pd
    import datetime
    
    # Create a predictions log file path
    predictions_log_path = "saved_models/prediction_log.csv"
    
    # Prepare prediction data
    prediction_data = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pred_open_rate': formatted_predictions['current']['open_rate'],
        'pred_click_rate': formatted_predictions['current']['click_rate'],
        'pred_optout_rate': formatted_predictions['current']['optout_rate']
    }
    
    # Add actual metrics if available
    if actual_metrics:
        prediction_data.update({
            'actual_open_rate': actual_metrics.get('open_rate'),
            'actual_click_rate': actual_metrics.get('click_rate'),
            'actual_optout_rate': actual_metrics.get('optout_rate'),
            'open_rate_error': actual_metrics.get('open_rate', 0) - formatted_predictions['current']['open_rate'],
            'click_rate_error': actual_metrics.get('click_rate', 0) - formatted_predictions['current']['click_rate'],
            'optout_rate_error': actual_metrics.get('optout_rate', 0) - formatted_predictions['current']['optout_rate']
        })
    
    # Convert to DataFrame
    pred_df = pd.DataFrame([prediction_data])
    
    # Save to CSV (append if exists)
    if os.path.exists(predictions_log_path):
        existing_log = pd.read_csv(predictions_log_path)
        updated_log = pd.concat([existing_log, pred_df], ignore_index=True)
        updated_log.to_csv(predictions_log_path, index=False)
    else:
        os.makedirs(os.path.dirname(predictions_log_path), exist_ok=True)
        pred_df.to_csv(predictions_log_path, index=False)
    
    return prediction_data

def track_model_performance(model_results, actual_metrics=None):
    """
    Track model performance over time and log it
    
    Parameters:
    - model_results: Dictionary containing model information
    - actual_metrics: Optional dictionary with actual metrics if available
    
    Returns:
    - Dictionary with performance tracking info
    """
    import pandas as pd
    import os
    import datetime
    
    performance_log_path = "saved_models/performance_log.csv"
    
    # Create log data
    log_data = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': model_results.get('version', 'unknown')
    }
    
    # Add model performance metrics
    if 'performance' in model_results:
        for metric, results in model_results['performance'].items():
            log_data[f'{metric}_mae'] = results.get('mae', None)
            log_data[f'{metric}_std'] = results.get('std', None)
    
    # Add actual metrics if available
    if actual_metrics:
        for metric, value in actual_metrics.items():
            log_data[f'actual_{metric}'] = value
    
    # Create or append to log file
    log_df = pd.DataFrame([log_data])
    
    if os.path.exists(performance_log_path):
        existing_log = pd.read_csv(performance_log_path)
        updated_log = pd.concat([existing_log, log_df], ignore_index=True)
        updated_log.to_csv(performance_log_path, index=False)
    else:
        os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)
        log_df.to_csv(performance_log_path, index=False)
    
    return log_data


def model_needs_retraining(performance_log_path="saved_models/performance_log.csv", threshold=0.05):
    """
    Check if model should be retrained based on performance degradation
    
    Parameters:
    - performance_log_path: Path to the performance log CSV file
    - threshold: Threshold for performance degradation (0.05 = 5%)
    
    Returns:
    - Boolean indicating if retraining is recommended
    - Reason for recommendation
    """
    import pandas as pd
    import os
    
    if not os.path.exists(performance_log_path):
        return False, "No performance history available"
    
    try:
        log_df = pd.read_csv(performance_log_path)
        
        if len(log_df) < 5:
            return False, "Not enough performance history to make a decision"
        
        # Check performance trends for each metric
        metrics = [col.replace('_mae', '') for col in log_df.columns if col.endswith('_mae')]
        
        for metric in metrics:
            # Get MAE values
            mae_col = f'{metric}_mae'
            if mae_col not in log_df.columns:
                continue
                
            # Get first and most recent 5 entries
            first_entries = log_df.head(5)[mae_col].mean()
            recent_entries = log_df.tail(5)[mae_col].mean()
            
            # Calculate degradation
            if first_entries > 0:
                degradation = (recent_entries - first_entries) / first_entries
                
                if degradation > threshold:
                    return True, f"{metric} performance has degraded by {degradation*100:.1f}%"
        
        return False, "Model performance is stable"
    except Exception as e:
        return False, f"Error analyzing performance history: {str(e)}"


def get_model_version():
    """Generate a version identifier for models"""
    import datetime
    import hashlib
    import random
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    
    return f"{timestamp}-{random_suffix}"

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
    tab1, tab2, tab3, tab4 = st.tabs(["Campaign Predictor", "Performance Insights", "Data Export", "Model Management"])
    
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
        
        # Track prediction performance
        track_prediction_performance(formatted_predictions)
        
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

    # Tab 4: Model Management
    with tab4:
        st.header("Model Management")
        
        # Model status overview
        st.subheader("Model Status")
        
        # Check for existing models
        models_dir = "saved_models"
        models_path = os.path.join(models_dir, "email_campaign_models.joblib")
        subject_model_path = os.path.join(models_dir, "subject_recommendation_model.joblib")
        performance_log_path = os.path.join(models_dir, "performance_log.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Models")
            
            if os.path.exists(models_path):
                model_stats = os.stat(models_path)
                model_date = datetime.datetime.fromtimestamp(model_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                model_size = model_stats.st_size / (1024 * 1024)  # Convert to MB
                
                st.success("✅ Models found")
                st.info(f"Last updated: {model_date}")
                st.info(f"Size: {model_size:.2f} MB")
                
                # Option to force retrain
                if st.button("Force Retrain Models"):
                    if os.path.exists(models_path):
                        os.remove(models_path)
                    if os.path.exists(subject_model_path):
                        os.remove(subject_model_path)
                    st.warning("Models deleted. Please refresh the page to retrain.")
            else:
                st.error("❌ No saved models found")
                st.info("Models will be trained automatically when you run the application")
        
        with col2:
            st.markdown("### Performance Tracking")
            
            if os.path.exists(performance_log_path):
                try:
                    log_df = pd.read_csv(performance_log_path)
                    st.success(f"✅ Performance log found ({len(log_df)} entries)")
                    
                    # Check if retraining is recommended
                    needs_retraining, reason = model_needs_retraining(performance_log_path)
                    
                    if needs_retraining:
                        st.warning(f"⚠️ Retraining recommended: {reason}")
                    else:
                        st.info(f"✓ {reason}")
                    
                    # Plot performance trends
                    if len(log_df) > 1:
                        metrics = [col.replace('_mae', '') for col in log_df.columns if col.endswith('_mae')]
                        
                        for metric in metrics:
                            mae_col = f'{metric}_mae'
                            if mae_col in log_df.columns:
                                fig = px.line(
                                    log_df, x='timestamp', y=mae_col,
                                    title=f'{metric} Model Performance Over Time',
                                    labels={'timestamp': 'Time', mae_col: 'Mean Absolute Error (%)'},
                                    markers=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing performance log: {e}")
            else:
                st.error("❌ No performance log found")
                st.info("Performance will be tracked once models are trained")
        
        # Model export and import
        st.subheader("Export/Import Models")
        
        export_col, import_col = st.columns(2)
        
        with export_col:
            st.markdown("### Export Models")
            if os.path.exists(models_path) and os.path.exists(subject_model_path):
                if st.button("Export Models"):
                    with st.spinner("Preparing export..."):
                        import zipfile
                        import base64
                        
                        # Create a zip file with models
                        zip_path = "saved_models/models_export.zip"
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            zipf.write(models_path, os.path.basename(models_path))
                            zipf.write(subject_model_path, os.path.basename(subject_model_path))
                            if os.path.exists(performance_log_path):
                                zipf.write(performance_log_path, os.path.basename(performance_log_path))
                        
                        # Create download link
                        with open(zip_path, "rb") as f:
                            bytes = f.read()
                            b64 = base64.b64encode(bytes).decode()
                            href = f'<a href="data:application/zip;base64,{b64}" download="email_campaign_models.zip">Download Models</a>'
                            st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No models available for export")
        
        with import_col:
            st.markdown("### Import Models")
            uploaded_file = st.file_uploader("Upload model zip file", type="zip")
            
            if uploaded_file is not None:
                if st.button("Import Models"):
                    with st.spinner("Importing models..."):
                        try:
                            import zipfile
                            import io
                            
                            # Create models directory if it doesn't exist
                            os.makedirs(models_dir, exist_ok=True)
                            
                            # Extract zip contents
                            with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zip_ref:
                                zip_ref.extractall(models_dir)
                            
                            st.success("Models imported successfully. Please refresh the page to use them.")
                        except Exception as e:
                            st.error(f"Error importing models: {e}")
        
        # Model performance details
        st.subheader("Model Performance Details")
        
        if 'performance' in model_results:
            for metric, results in model_results['performance'].items():
                st.write(f"**{metric}**: MAE = {results['mae']:.2f}%")
                
                # Add feature importance if available
                if 'models' in model_results and metric in model_results['models']:
                    try:
                        model = model_results['models'][metric]
                        
                        # For XGBoost models
                        if hasattr(model, 'feature_importances_') or (hasattr(model, 'named_steps') and hasattr(model.named_steps.get('regressor', None), 'feature_importances_')):
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                feature_names = model_results['feature_names']
                            else:
                                importances = model.named_steps['regressor'].feature_importances_
                                # Extract feature names from preprocessor if available
                                if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                                else:
                                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                            
                            # Create importance DataFrame
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            # Plot importance
                            fig = px.bar(
                                importance_df, 
                                x='Importance', 
                                y='Feature', 
                                orientation='h',
                                title=f'Top 10 Features for {metric}',
                                labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                                color='Importance'
                            )
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error showing feature importance: {e}")
        else:
            st.info("No model performance data available")

if __name__ == "__main__":
    main()