import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMRegressor
import datetime
import os

# Import functions from model_metadata
from model_metadata import get_model_version, save_model_metadata

def train_multi_metric_models(delivery_df, customer_df):
    """Train separate models for predicting open rate, click rate, and optout rate"""
    
    # Process data - handle the missing 'Contact date' issue
    # Skip the enhanced_feature_engineering which might be causing the error
    # And process the data directly
    processed_data = process_data_directly(delivery_df, customer_df)
    
    # Calculate metrics if not already present
    if 'open_rate' not in processed_data.columns:
        processed_data['open_rate'] = (processed_data['Opens'] / processed_data['Sendouts']) * 100
    
    if 'click_rate' not in processed_data.columns:
        processed_data['click_rate'] = (processed_data['Clicks'] / processed_data['Opens']) * 100
    
    if 'optout_rate' not in processed_data.columns:
        processed_data['optout_rate'] = (processed_data['Optout'] / processed_data['Opens']) * 100
    
    # Replace infinities and NaNs with 0 for click_rate and optout_rate (when Opens = 0)
    processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_data.fillna({
        'click_rate': 0,
        'optout_rate': 0
    }, inplace=True)
    
    # Define features for all models
    # Use only columns that we know exist in the data
    columns_to_drop = ['open_rate', 'click_rate', 'optout_rate', 
                      'InternalName', 'Subject', 'Date', 
                      'Opens', 'Clicks', 'Optout', 'Sendouts']
    
    features = processed_data.drop([col for col in columns_to_drop if col in processed_data.columns], 
                                  axis=1, errors='ignore')
    
    # Check for duplicate columns and remove them
    if len(features.columns) != len(set(features.columns)):
        # Find duplicate columns
        duplicate_cols = features.columns[features.columns.duplicated()].tolist()
        print(f"Found duplicate columns: {duplicate_cols}")
        
        # Keep only one instance of each duplicate column
        features = features.loc[:, ~features.columns.duplicated()]
    
    # Define targets
    targets = {
        'open_rate': processed_data['open_rate'],
        'click_rate': processed_data['click_rate'],
        'optout_rate': processed_data['optout_rate']
    }
    
    # Identify feature types
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    # Define model parameters
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Train models for each target metric
    models = {}
    performance = {}
    
    for metric, target in targets.items():
        print(f"Training model for {metric}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(**xgb_params))
        ])
        
        # Fit model
        pipeline.fit(features, target)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            pipeline, features, target,
            cv=5, scoring='neg_mean_absolute_error'
        )
        
        # Store model and performance
        models[metric] = pipeline
        performance[metric] = {
            'mae': -np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
        
        print(f"  {metric} model MAE: {-np.mean(cv_scores):.4f}")
    
    # For subject line optimization, we only need the open_rate model
    # Create a specialized model for subject line features only
    subject_features = [
        col for col in features.columns 
        if col.startswith('length') or col.startswith('has_') or col.startswith('word_count')
    ]
    
    if subject_features:
        # If we have subject-specific features, train a specialized model
        subject_pipeline = Pipeline([
            ('preprocessor', StandardScaler()),
            ('regressor', xgb.XGBRegressor(**xgb_params))
        ])
        
        subject_pipeline.fit(features[subject_features], targets['open_rate'])
        models['subject_open_rate'] = subject_pipeline
    
    return {
        'models': models,
        'performance': performance,
        'feature_names': features.columns.tolist(),
        'categorical_values': {
            col: features[col].unique().tolist() 
            for col in categorical_features
        }
    }

def process_data_directly(delivery_df, customer_df):
    """
    Process the data directly without relying on enhanced_feature_engineering
    that might be causing the 'Contact date' error
    """
    # Make copies to avoid modifying originals
    delivery_df = delivery_df.copy()
    customer_df = customer_df.copy()
    
    # Ensure we have a 'subject' column (lowercase) for processing
    if 'subject' not in delivery_df.columns and 'Subject' in delivery_df.columns:
        delivery_df['subject'] = delivery_df['Subject']
    
    # Basic time features from Date if available
    if 'Date' in delivery_df.columns:
        try:
            delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
            delivery_df['day_of_week'] = delivery_df['Date'].dt.dayofweek
            delivery_df['hour_of_day'] = delivery_df['Date'].dt.hour
            delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6]).astype(int)
        except Exception as e:
            print(f"Warning: Could not process Date column: {e}")
            # Add default values
            delivery_df['day_of_week'] = 0
            delivery_df['hour_of_day'] = 12
            delivery_df['is_weekend'] = 0
    else:
        # Add default values if Date column is missing
        delivery_df['day_of_week'] = 0
        delivery_df['hour_of_day'] = 12
        delivery_df['is_weekend'] = 0
    
    # Add subject features - critical for the model
    subject_feature_names = ['length', 'has_personalization', 'has_question', 
                           'has_numbers', 'has_uppercase_words', 'has_emoji', 'word_count']
    
    # We need to ensure we have all subject features
    from feature_engineering import extract_subject_features
    
    # Get the subject column (either 'subject' or 'Subject')
    if 'subject' in delivery_df.columns:
        subject_col = 'subject'
    elif 'Subject' in delivery_df.columns:
        subject_col = 'Subject'
        # Also create lowercase version for consistency
        delivery_df['subject'] = delivery_df['Subject']
    else:
        # Create empty subject if none exists
        delivery_df['subject'] = ""
        subject_col = 'subject'
    
    # Generate subject features
    subject_features = delivery_df[subject_col].apply(extract_subject_features).apply(pd.Series)
    
    # Add each feature to the dataframe
    for feature in subject_feature_names:
        if feature in subject_features.columns:
            delivery_df[feature] = subject_features[feature]
        else:
            # Add default values for missing features
            if feature == 'length':
                delivery_df[feature] = delivery_df[subject_col].str.len()
            elif feature == 'word_count':
                delivery_df[feature] = delivery_df[subject_col].str.split().str.len()
            else:
                delivery_df[feature] = 0  # Default for boolean features
    
    # Calculate aggregate customer demographics if available
    if 'Bolag' in customer_df.columns:
        # Check Age column
        age_col = None
        for col_name in ['Age', 'age', 'AGE']:
            if col_name in customer_df.columns:
                age_col = col_name
                break
                
        if age_col:
            # We can calculate age stats
            customer_aggs = customer_df.groupby('InternalName').agg({
                age_col: ['mean', 'std', 'min', 'max']
            })
            
            # Flatten multi-level column index
            customer_aggs.columns = ['_'.join(col).strip() for col in customer_aggs.columns.values]
            customer_aggs.reset_index(inplace=True)
            
            # Rename columns for clarity
            age_mean_col = f"{age_col}_mean"
            age_std_col = f"{age_col}_std"
            age_min_col = f"{age_col}_min"
            age_max_col = f"{age_col}_max"
            
            column_renames = {
                age_mean_col: 'avg_age',
                age_std_col: 'std_age',
                age_min_col: 'min_age',
                age_max_col: 'max_age'
            }
            
            customer_aggs.rename(columns=column_renames, inplace=True)
            
            # Merge with delivery data
            delivery_df = delivery_df.merge(customer_aggs, on='InternalName', how='left')
        
        # Check Gender column
        gender_col = None
        for col_name in ['Gender', 'gender', 'GENDER']:
            if col_name in customer_df.columns:
                gender_col = col_name
                break
                
        if gender_col:
            # Calculate percentage of women
            gender_aggs = customer_df.groupby('InternalName').apply(
                lambda x: (x[gender_col].str.lower() == 'f').mean() * 100 if len(x) > 0 else 50
            ).reset_index(name='pct_women')
            
            # Merge with delivery data
            delivery_df = delivery_df.merge(gender_aggs, on='InternalName', how='left')
        
        # Fill missing values with reasonable defaults
        delivery_df.fillna({
            'avg_age': 40,
            'std_age': 15,
            'min_age': 18,
            'max_age': 80,
            'pct_women': 50
        }, inplace=True)
    else:
        # Add default demographics if not available
        if 'avg_age' not in delivery_df.columns:
            delivery_df['avg_age'] = 40
        if 'std_age' not in delivery_df.columns:
            delivery_df['std_age'] = 15
        if 'min_age' not in delivery_df.columns:
            delivery_df['min_age'] = 18
        if 'max_age' not in delivery_df.columns:
            delivery_df['max_age'] = 80
        if 'pct_women' not in delivery_df.columns:
            delivery_df['pct_women'] = 50
    
    # Map Dialog, Syfte, and Produkt to expected lowercase names if needed
    column_mappings = {
        'Dialog': 'dialog',
        'Syfte': 'syfte',
        'Produkt': 'product'
    }
    
    for orig_col, new_col in column_mappings.items():
        if orig_col in delivery_df.columns and new_col not in delivery_df.columns:
            delivery_df[new_col] = delivery_df[orig_col]
    
    # Handle any missing columns that the model might expect
    expected_columns = ['dialog', 'syfte', 'product', 'county', 'day_of_week', 'hour_of_day', 'is_weekend']
    
    for col in expected_columns:
        if col not in delivery_df.columns:
            delivery_df[col] = 'Unknown'
    
    return delivery_df

def predict_metrics(input_data, models, metrics=None):
    """
    Predict multiple metrics based on input data
    
    Parameters:
    - input_data: DataFrame with input features
    - models: Dict of trained models for different metrics
    - metrics: List of metrics to predict (default: all available)
    
    Returns:
    - Dictionary with predicted values for each metric
    """
    if metrics is None:
        metrics = list(models.keys())
    
    predictions = {}
    
    for metric in metrics:
        if metric in models:
            try:
                predictions[metric] = models[metric].predict(input_data)[0]
            except Exception as e:
                print(f"Error predicting {metric}: {e}")
                # Fall back to a default value
                predictions[metric] = 0.0
    
    return predictions

def enhanced_train_multi_metric_models(delivery_df, customer_df):
    """Enhanced version of train_multi_metric_models that adds versioning and metadata"""
    # Call the function directly
    model_results = train_multi_metric_models(delivery_df, customer_df)
    
    # Add version using imported function from model_metadata
    model_results['version'] = get_model_version()
    
    # Add data stats
    model_results['num_campaigns'] = len(delivery_df)
    
    # Save metadata using imported function from model_metadata
    save_model_metadata(model_results)
    
    return model_results